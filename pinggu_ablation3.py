import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from PIL import Image
import warnings
import time
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import math
from skimage.metrics import structural_similarity as ssim

from models.ablation_unet import AblationUNet
from diffusion.DDPM import DiffDDPMSampler_cond
from data.dataset import CTMRI
from utils.metrics import calculate_metrics

warnings.filterwarnings("ignore")


def compute_noise_prediction_error_gpu(model, x_t, t, target_noise, device):
    """
    A. 噪声预测误差 vs. 时间步
    计算模型在不同时间步的噪声预测误差
    """
    try:
        model.eval()
        with torch.no_grad():
            predicted_noise = model(x_t, t)

            # 计算RMSE
            mse = F.mse_loss(predicted_noise, target_noise, reduction='mean')
            rmse = torch.sqrt(mse)

            # 计算SNR (Signal-to-Noise Ratio)
            signal_power = torch.mean(target_noise ** 2)
            noise_power = torch.mean((predicted_noise - target_noise) ** 2)

            if noise_power > 0:
                snr = 10 * torch.log10(signal_power / noise_power)
            else:
                snr = torch.tensor(100.0, device=device)

            # 计算ε-MSE (late/low-noise权重)
            epsilon_mse = mse  # 基础MSE，可以根据时间步加权

            return {
                'rmse': rmse.item(),
                'snr': snr.item() if not torch.isinf(snr) else 100.0,
                'mse': mse.item(),
                'epsilon_mse': epsilon_mse.item()
            }
    except Exception as e:
        print(f"Error in noise prediction analysis: {e}")
        return {'rmse': 0.0, 'snr': 0.0, 'mse': 0.0, 'epsilon_mse': 0.0}


def compute_one_step_kl_gpu(model, x_t_input, t, true_noise, alphas, betas, alphas_bar, device, use_beta_tilde=True):
    """
    KL( p_theta(x_{t-1}|x_t) || q(x_{t-1}|x_t,x0) ) in the DIFFERENCE domain.
    假设相同对角协方差：beta_tilde_t * I（或 beta_t * I）
    """
    with torch.no_grad():
        eps_pred = model(x_t_input, t)                       # [B,1,H,W]
        d_t      = x_t_input[:, :1]                          # 差异通道
        alpha_t  = alphas[t].view(-1,1,1,1)
        beta_t   = betas[t].view(-1,1,1,1)
        abar_t   = alphas_bar[t].view(-1,1,1,1)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_abar_t = torch.sqrt(1 - abar_t)

        # 均值
        mu_theta = (1.0 / sqrt_alpha_t) * (d_t - (beta_t / sqrt_one_minus_abar_t) * eps_pred)
        mu_q     = (1.0 / sqrt_alpha_t) * (d_t - (beta_t / sqrt_one_minus_abar_t) * true_noise)

        # 协方差：beta_tilde_t 或直接用 beta_t
        if use_beta_tilde:
            # beta_tilde_t = beta_t * (1 - abar_{t-1}) / (1 - abar_t)
            # 注意 t 为张量索引；需处理 t-1 的边界
            t_minus = torch.clamp(t-1, min=0)
            abar_tm1 = alphas_bar[t_minus].view(-1,1,1,1)
            beta_tilde = beta_t * (1 - abar_tm1) / (1 - abar_t + 1e-12)
            sigma2 = beta_tilde
        else:
            sigma2 = beta_t

        diff = (mu_q - mu_theta).flatten(1)
        # KL = 0.5 * diff^T Sigma^{-1} diff
        kl = 0.5 * (diff.pow(2).sum(dim=1) / (sigma2.view(sigma2.size(0), -1).mean(dim=1) + 1e-12))  # 简洁实现
        # 额外：ε 角相似度
        cos = F.cosine_similarity(eps_pred.flatten(1), true_noise.flatten(1), dim=1)
        return {'kl': kl.mean().item(), 'eps_cos': cos.mean().item()}


def compute_d0_estimation_quality_gpu(model, x_t_input, t, true_d0, alphas_bar, device):
    """
    B. d̂₀ 估计精度 vs. 时间步 (差异域)
    计算模型对差异d₀的估计质量
    """
    with torch.no_grad():
        eps_pred = model(x_t_input, t)
        d_t = x_t_input[:, :1]
        sqrt_abar = torch.sqrt(alphas_bar[t]).view(-1,1,1,1)
        sqrt_one_minus_abar = torch.sqrt(1 - alphas_bar[t]).view(-1,1,1,1)

        d0_hat = (d_t - sqrt_one_minus_abar * eps_pred) / (sqrt_abar + 1e-12)

        # 归一化到 [0,1] 再算指标
        est = torch.clamp((d0_hat + 1)/2, 0, 1)
        tru = torch.clamp((true_d0 + 1)/2, 0, 1)

        mse = F.mse_loss(est, tru, reduction='mean')
        psnr_val = ( -10.0 * torch.log10(mse + 1e-12) ).item()
        # SSIM：按 batch/通道平均
        est_np = est.detach().cpu().numpy()
        tru_np = tru.detach().cpu().numpy()
        ssim_list = []
        for b in range(est_np.shape[0]):
            for c in range(est_np.shape[1]):
                ssim_list.append(ssim(est_np[b, c], tru_np[b, c], data_range=1.0))
        ssim_val = float(np.mean(ssim_list))
        mae = F.l1_loss(est, tru, reduction='mean').item()
        return {'psnr': psnr_val, 'ssim': ssim_val, 'mae': mae, 'mse': mse.item()}


def compute_x_estimation_quality_gpu(model, x_t_input, t, condition, true_x, alphas_bar, device):
    """
    从差异域还原：d̂0 = (d_t - sqrt(1-abar_t)*eps_pred)/sqrt(abar_t)
    x̂ = condition + d̂0
    """
    with torch.no_grad():
        eps_pred = model(x_t_input, t)
        d_t = x_t_input[:, :1]
        sqrt_abar = torch.sqrt(alphas_bar[t]).view(-1,1,1,1)
        sqrt_one_minus_abar = torch.sqrt(1 - alphas_bar[t]).view(-1,1,1,1)

        d0_hat = (d_t - sqrt_one_minus_abar * eps_pred) / (sqrt_abar + 1e-12)
        x_hat = condition[:, :1] + d0_hat

        # 归一化到 [0,1] 再算指标（根据你的预处理来，若已在[0,1]可省略）
        est = torch.clamp((x_hat + 1)/2, 0, 1)
        tru = torch.clamp((true_x + 1)/2, 0, 1)

        mse = F.mse_loss(est, tru, reduction='mean')
        psnr_val = ( -10.0 * torch.log10(mse + 1e-12) ).item()
        # SSIM：按 batch/通道平均
        est_np = est.detach().cpu().numpy()
        tru_np = tru.detach().cpu().numpy()
        ssim_list = []
        for b in range(est_np.shape[0]):
            for c in range(est_np.shape[1]):
                ssim_list.append(ssim(est_np[b, c], tru_np[b, c], data_range=1.0))
        ssim_val = float(np.mean(ssim_list))
        mae = F.l1_loss(est, tru, reduction='mean').item()
        return {'psnr': psnr_val, 'ssim': ssim_val, 'mae': mae, 'mse': mse.item()}


def get_diffusion_schedule(beta_1, beta_T, T, device):
    """获取扩散过程的调度参数"""
    betas = torch.linspace(beta_1, beta_T, T, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument("--attn", nargs='+', type=int, default=[2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--sample_num", type=int, default=12)
    # 修正：改为更合理的时间步划分
    parser.add_argument("--timestep_analysis", nargs='+', type=int,
                        default=[50, 150, 250, 350, 500, 650, 800, 950],
                        help="Timesteps to analyze")
    parser.add_argument("--dataset_name", type=str, default="/home/midi/datasets/HavardMedicalImage/CT-MRI/test/")
    parser.add_argument("--save_weight_dir", type=str, default="./results/HavardMRI-CT")
    parser.add_argument("--encoder_middle_weight", type=str,
                        default="ablation/encoder_middle_DIFFDDPM/model_epoch_3973.pt")
    parser.add_argument("--full_mamba_weight", type=str, default="mamba_unet_DIFFDDPM/model_epoch_4050.pt")
    parser.add_argument("--output_dir", type=str, default="advanced_decoder_analysis")
    parser.add_argument("--dataset_type", type=str, default="CTMRI")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for evaluation")
    parser.add_argument("--random_seed", type=int, default=2025, help="Random seed for reproducibility")
    return parser.parse_args()


def load_model(args, ablation_config, weight_path, device, model_name=""):
    """加载模型"""
    print(f"Loading {model_name} model...")
    # 设置通道数
    if args.dataset_type in ["mripet", "mrispect"]:
        in_channels = 6
        out_channels = 3
    else:
        in_channels = 2
        out_channels = 1

    # 初始化模型
    model = AblationUNet(
        args.T, args.ch, args.ch_mult, args.attn,
        args.num_res_blocks, args.dropout,
        in_channels=in_channels,
        out_channels=out_channels,
        use_mamba_encoder=ablation_config['use_mamba_encoder'],
        use_mamba_middle=ablation_config['use_mamba_middle'],
        use_mamba_decoder=ablation_config['use_mamba_decoder']
    ).to(device)

    # 加载权重
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        if args.use_ema and 'ema' in checkpoint:
            model.load_state_dict(checkpoint['ema'])
        else:
            model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"{model_name} model loaded successfully")
    return model


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    output_dir = os.path.join(args.save_weight_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 获取扩散调度参数
    betas, alphas, alphas_bar = get_diffusion_schedule(args.beta_1, args.beta_T, args.T, device)

    # 定义两个模型的配置
    encoder_middle_config = {
        "use_mamba_encoder": True,
        "use_mamba_middle": True,
        "use_mamba_decoder": False,
        "description": "Mamba blocks in encoder and middle (ResNet decoder)"
    }

    full_mamba_config = {
        "use_mamba_encoder": True,
        "use_mamba_middle": True,
        "use_mamba_decoder": True,
        "description": "Full MambaUNet (Mamba decoder)"
    }

    # 加载两个模型
    encoder_middle_path = os.path.join(args.save_weight_dir, args.encoder_middle_weight)
    model_encoder_middle = load_model(args, encoder_middle_config, encoder_middle_path, device, "encoder_middle")

    full_mamba_path = os.path.join(args.save_weight_dir, args.full_mamba_weight)
    model_full_mamba = load_model(args, full_mamba_config, full_mamba_path, device, "full_mamba")

    # 设置数据集
    dataset = CTMRI(args.dataset_name, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 存储分析结果
    results = {
        'encoder_middle': {
            'noise_prediction': defaultdict(list),  # A. 噪声预测误差
            'd0_estimation': defaultdict(list),  # B. d₀估计精度 (差异域)
            'x_estimation': defaultdict(list),   # B2. x̂估计精度 (像素域)
            'one_step_kl': defaultdict(list)     # C. 一步KL散度
        },
        'full_mamba': {
            'noise_prediction': defaultdict(list),
            'd0_estimation': defaultdict(list),
            'x_estimation': defaultdict(list),
            'one_step_kl': defaultdict(list)
        }
    }

    sample_count = 0
    print("Starting advanced decoder ablation analysis...")
    start_total_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= args.sample_num:
                break

            sample_start_time = time.time()
            target = batch['target'].to(device)
            condition = batch['condition'].to(device)

            # 获取输出通道数
            out_channels = model_encoder_middle.out_channels

            # 准备真实的d₀ (差异域)
            true_d0 = target - condition  # 差异作为真实的d₀

            print(f"\nProcessing sample {sample_count + 1}/{args.sample_num}")

            # 对每个时间步进行分析
            for t_val in args.timestep_analysis:
                print(f"  Analyzing timestep {t_val}...")

                # 创建时间步张量
                t = torch.full((args.batch_size,), t_val, device=device, dtype=torch.long)

                # 生成带噪声的x_t (使用固定种子确保一致性)
                torch.manual_seed(args.random_seed + sample_count * 1000 + t_val)
                true_noise = torch.randn_like(true_d0)

                sqrt_alphas_bar_t = torch.sqrt(alphas_bar[t]).view(-1, 1, 1, 1)
                sqrt_one_minus_alphas_bar_t = torch.sqrt(1 - alphas_bar[t]).view(-1, 1, 1, 1)

                # 在差异域上添加噪声
                d_t = sqrt_alphas_bar_t * true_d0 + sqrt_one_minus_alphas_bar_t * true_noise

                # 准备模型输入 (concatenate with condition)
                x_t_input_em = torch.cat((d_t, condition), dim=1)
                x_t_input_fm = torch.cat((d_t, condition), dim=1)

                # === A. 噪声预测误差分析 ===
                noise_analysis_em = compute_noise_prediction_error_gpu(
                    model_encoder_middle, x_t_input_em, t, true_noise, device
                )
                noise_analysis_fm = compute_noise_prediction_error_gpu(
                    model_full_mamba, x_t_input_fm, t, true_noise, device
                )

                results['encoder_middle']['noise_prediction'][t_val].append(noise_analysis_em)
                results['full_mamba']['noise_prediction'][t_val].append(noise_analysis_fm)

                # === B. d₀估计精度分析 (差异域) ===
                d0_analysis_em = compute_d0_estimation_quality_gpu(
                    model_encoder_middle, x_t_input_em, t, true_d0, alphas_bar, device
                )
                d0_analysis_fm = compute_d0_estimation_quality_gpu(
                    model_full_mamba, x_t_input_fm, t, true_d0, alphas_bar, device
                )

                results['encoder_middle']['d0_estimation'][t_val].append(d0_analysis_em)
                results['full_mamba']['d0_estimation'][t_val].append(d0_analysis_fm)

                # === B2. x̂估计精度分析 (像素域) ===
                x_analysis_em = compute_x_estimation_quality_gpu(
                    model_encoder_middle, x_t_input_em, t, condition, target, alphas_bar, device
                )
                x_analysis_fm = compute_x_estimation_quality_gpu(
                    model_full_mamba, x_t_input_fm, t, condition, target, alphas_bar, device
                )

                results['encoder_middle']['x_estimation'][t_val].append(x_analysis_em)
                results['full_mamba']['x_estimation'][t_val].append(x_analysis_fm)

                # === C. 一步KL散度分析 ===
                kl_analysis_em = compute_one_step_kl_gpu(
                    model_encoder_middle, x_t_input_em, t, true_noise, alphas, betas, alphas_bar, device
                )
                kl_analysis_fm = compute_one_step_kl_gpu(
                    model_full_mamba, x_t_input_fm, t, true_noise, alphas, betas, alphas_bar, device
                )

                results['encoder_middle']['one_step_kl'][t_val].append(kl_analysis_em)
                results['full_mamba']['one_step_kl'][t_val].append(kl_analysis_fm)

            sample_count += 1
            sample_time = time.time() - sample_start_time
            print(f"  Sample {sample_count} completed in {sample_time:.2f}s")

    total_time = time.time() - start_total_time
    print(f"\nTotal processing time: {total_time:.2f}s")

    # === 计算和输出结果 ===
    print("\n" + "=" * 100)
    print("ADVANCED DECODER ABLATION ANALYSIS RESULTS")
    print("=" * 100)

    print(f"\nModels compared:")
    print(f"  - Encoder+Middle (ResNet Decoder): {encoder_middle_config['description']}")
    print(f"  - Full Mamba (Mamba Decoder): {full_mamba_config['description']}")
    print(f"  - Samples analyzed: {sample_count}")
    print(f"  - Timesteps analyzed: {args.timestep_analysis}")
    print(f"  - Total time: {total_time:.2f}s")

    # 保存详细结果到文件
    results_file = os.path.join(output_dir, 'advanced_decoder_analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("ADVANCED DECODER ABLATION ANALYSIS RESULTS\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Models compared:\n")
        f.write(f"  - Encoder+Middle (ResNet Decoder): {encoder_middle_config['description']}\n")
        f.write(f"  - Full Mamba (Mamba Decoder): {full_mamba_config['description']}\n")
        f.write(f"  - Samples analyzed: {sample_count}\n")
        f.write(f"  - Timesteps analyzed: {args.timestep_analysis}\n")
        f.write(f"  - Total time: {total_time:.2f}s\n\n")

        # A. 噪声预测误差分析
        f.write("A. NOISE PREDICTION ERROR ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("Timestep | EncMiddle_RMSE | FullMamba_RMSE | EncMiddle_SNR | FullMamba_SNR | ΔRMSE | ΔSNR\n")
        f.write("-" * 80 + "\n")

        for t_val in sorted(args.timestep_analysis):
            if t_val in results['encoder_middle']['noise_prediction']:
                em_metrics = results['encoder_middle']['noise_prediction'][t_val]
                fm_metrics = results['full_mamba']['noise_prediction'][t_val]

                em_rmse = np.mean([m['rmse'] for m in em_metrics])
                fm_rmse = np.mean([m['rmse'] for m in fm_metrics])
                em_snr = np.mean([m['snr'] for m in em_metrics])
                fm_snr = np.mean([m['snr'] for m in fm_metrics])

                delta_rmse = fm_rmse - em_rmse
                delta_snr = fm_snr - em_snr

                f.write(
                    f"{t_val:8d} | {em_rmse:13.6f} | {fm_rmse:13.6f} | {em_snr:12.2f} | {fm_snr:12.2f} | {delta_rmse:6.4f} | {delta_snr:5.2f}\n")

        # B. d₀估计精度分析 (差异域)
        f.write(f"\nB. D₀ ESTIMATION QUALITY ANALYSIS (Difference Domain)\n")
        f.write("-" * 80 + "\n")
        f.write("Timestep | EncMiddle_PSNR | FullMamba_PSNR | EncMiddle_SSIM | FullMamba_SSIM | ΔPSNR | ΔSSIM\n")
        f.write("-" * 80 + "\n")

        for t_val in sorted(args.timestep_analysis):
            if t_val in results['encoder_middle']['d0_estimation']:
                em_metrics = results['encoder_middle']['d0_estimation'][t_val]
                fm_metrics = results['full_mamba']['d0_estimation'][t_val]

                em_psnr = np.mean([m['psnr'] for m in em_metrics])
                fm_psnr = np.mean([m['psnr'] for m in fm_metrics])
                em_ssim = np.mean([m['ssim'] for m in em_metrics])
                fm_ssim = np.mean([m['ssim'] for m in fm_metrics])

                delta_psnr = fm_psnr - em_psnr
                delta_ssim = fm_ssim - em_ssim

                f.write(
                    f"{t_val:8d} | {em_psnr:13.2f} | {fm_psnr:13.2f} | {em_ssim:13.4f} | {fm_ssim:13.4f} | {delta_psnr:6.2f} | {delta_ssim:6.4f}\n")

        # B2. x̂估计精度分析 (像素域)
        f.write(f"\nB2. X̂ ESTIMATION QUALITY ANALYSIS (Pixel Domain)\n")
        f.write("-" * 80 + "\n")
        f.write("Timestep | EncMiddle_PSNR | FullMamba_PSNR | EncMiddle_SSIM | FullMamba_SSIM | ΔPSNR | ΔSSIM\n")
        f.write("-" * 80 + "\n")

        for t_val in sorted(args.timestep_analysis):
            if t_val in results['encoder_middle']['x_estimation']:
                em_metrics = results['encoder_middle']['x_estimation'][t_val]
                fm_metrics = results['full_mamba']['x_estimation'][t_val]

                em_psnr = np.mean([m['psnr'] for m in em_metrics])
                fm_psnr = np.mean([m['psnr'] for m in fm_metrics])
                em_ssim = np.mean([m['ssim'] for m in em_metrics])
                fm_ssim = np.mean([m['ssim'] for m in fm_metrics])

                delta_psnr = fm_psnr - em_psnr
                delta_ssim = fm_ssim - em_ssim

                f.write(
                    f"{t_val:8d} | {em_psnr:13.2f} | {fm_psnr:13.2f} | {em_ssim:13.4f} | {fm_ssim:13.4f} | {delta_psnr:6.2f} | {delta_ssim:6.4f}\n")

        # C. 一步KL散度分析
        f.write(f"\nC. ONE-STEP KL DIVERGENCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("Timestep | EncMiddle_KL | FullMamba_KL | EncMiddle_EpsCos | FullMamba_EpsCos | ΔKL | ΔEpsCos\n")
        f.write("-" * 80 + "\n")

        for t_val in sorted(args.timestep_analysis):
            if t_val in results['encoder_middle']['one_step_kl']:
                em_metrics = results['encoder_middle']['one_step_kl'][t_val]
                fm_metrics = results['full_mamba']['one_step_kl'][t_val]

                em_kl = np.mean([m['kl'] for m in em_metrics])
                fm_kl = np.mean([m['kl'] for m in fm_metrics])
                em_cos = np.mean([m['eps_cos'] for m in em_metrics])
                fm_cos = np.mean([m['eps_cos'] for m in fm_metrics])

                delta_kl = fm_kl - em_kl
                delta_cos = fm_cos - em_cos

                f.write(
                    f"{t_val:8d} | {em_kl:11.6f} | {fm_kl:11.6f} | {em_cos:14.4f} | {fm_cos:14.4f} | {delta_kl:5.4f} | {delta_cos:8.4f}\n")

        # 修正：按照正确的噪声级别划分总结分析
        f.write(f"\nSUMMARY ANALYSIS\n")
        f.write("=" * 80 + "\n")

        # 修正的时间段划分
        high_noise_steps = [t for t in args.timestep_analysis if t >= 700]  # High-noise (大t)
        mid_noise_steps = [t for t in args.timestep_analysis if 300 < t < 700]  # Mid-noise
        low_noise_steps = [t for t in args.timestep_analysis if t <= 300]  # Low-noise (小t)

        for period_name, period_steps in [("High-noise (t≥700)", high_noise_steps),
                                          ("Mid-noise (300<t<700)", mid_noise_steps),
                                          ("Low-noise (t≤300)", low_noise_steps)]:
            if period_steps:
                f.write(f"\n{period_name} timesteps:\n")

                # 噪声预测改善
                period_rmse_improvements = []
                period_snr_improvements = []
                for t_val in period_steps:
                    if t_val in results['encoder_middle']['noise_prediction']:
                        em_metrics = results['encoder_middle']['noise_prediction'][t_val]
                        fm_metrics = results['full_mamba']['noise_prediction'][t_val]

                        em_rmse = np.mean([m['rmse'] for m in em_metrics])
                        fm_rmse = np.mean([m['rmse'] for m in fm_metrics])
                        em_snr = np.mean([m['snr'] for m in em_metrics])
                        fm_snr = np.mean([m['snr'] for m in fm_metrics])

                        period_rmse_improvements.append(fm_rmse - em_rmse)
                        period_snr_improvements.append(fm_snr - em_snr)

                if period_rmse_improvements:
                    f.write(
                        f"  Avg ΔRMSE (noise): {np.mean(period_rmse_improvements):.6f} ± {np.std(period_rmse_improvements):.6f}\n")
                    f.write(
                        f"  Avg ΔSNR (noise): {np.mean(period_snr_improvements):.2f} ± {np.std(period_snr_improvements):.2f}\n")

                # d₀估计改善 (差异域)
                period_psnr_improvements = []
                period_ssim_improvements = []
                for t_val in period_steps:
                    if t_val in results['encoder_middle']['d0_estimation']:
                        em_metrics = results['encoder_middle']['d0_estimation'][t_val]
                        fm_metrics = results['full_mamba']['d0_estimation'][t_val]

                        em_psnr = np.mean([m['psnr'] for m in em_metrics])
                        fm_psnr = np.mean([m['psnr'] for m in fm_metrics])
                        em_ssim = np.mean([m['ssim'] for m in em_metrics])
                        fm_ssim = np.mean([m['ssim'] for m in fm_metrics])

                        period_psnr_improvements.append(fm_psnr - em_psnr)
                        period_ssim_improvements.append(fm_ssim - em_ssim)

                if period_psnr_improvements:
                    f.write(
                        f"  Avg ΔPSNR (d₀): {np.mean(period_psnr_improvements):.2f} ± {np.std(period_psnr_improvements):.2f}\n")
                    f.write(
                        f"  Avg ΔSSIM (d₀): {np.mean(period_ssim_improvements):.4f} ± {np.std(period_ssim_improvements):.4f}\n")

                # x̂估计改善 (像素域)
                period_x_psnr_improvements = []
                period_x_ssim_improvements = []
                for t_val in period_steps:
                    if t_val in results['encoder_middle']['x_estimation']:
                        em_metrics = results['encoder_middle']['x_estimation'][t_val]
                        fm_metrics = results['full_mamba']['x_estimation'][t_val]

                        em_psnr = np.mean([m['psnr'] for m in em_metrics])
                        fm_psnr = np.mean([m['psnr'] for m in fm_metrics])
                        em_ssim = np.mean([m['ssim'] for m in em_metrics])
                        fm_ssim = np.mean([m['ssim'] for m in fm_metrics])

                        period_x_psnr_improvements.append(fm_psnr - em_psnr)
                        period_x_ssim_improvements.append(fm_ssim - em_ssim)

                if period_x_psnr_improvements:
                    f.write(
                        f"  Avg ΔPSNR (x̂): {np.mean(period_x_psnr_improvements):.2f} ± {np.std(period_x_psnr_improvements):.2f}\n")
                    f.write(
                        f"  Avg ΔSSIM (x̂): {np.mean(period_x_ssim_improvements):.4f} ± {np.std(period_x_ssim_improvements):.4f}\n")

                # KL散度改善
                period_kl_improvements = []
                period_cos_improvements = []
                for t_val in period_steps:
                    if t_val in results['encoder_middle']['one_step_kl']:
                        em_metrics = results['encoder_middle']['one_step_kl'][t_val]
                        fm_metrics = results['full_mamba']['one_step_kl'][t_val]

                        em_kl = np.mean([m['kl'] for m in em_metrics])
                        fm_kl = np.mean([m['kl'] for m in fm_metrics])
                        em_cos = np.mean([m['eps_cos'] for m in em_metrics])
                        fm_cos = np.mean([m['eps_cos'] for m in fm_metrics])

                        period_kl_improvements.append(fm_kl - em_kl)
                        period_cos_improvements.append(fm_cos - em_cos)

                if period_kl_improvements:
                    f.write(
                        f"  Avg ΔKL: {np.mean(period_kl_improvements):.6f} ± {np.std(period_kl_improvements):.6f}\n")
                    f.write(
                        f"  Avg ΔEpsCos: {np.mean(period_cos_improvements):.4f} ± {np.std(period_cos_improvements):.4f}\n")

    print(f"\nDetailed results saved to: {results_file}")

    # 同时在控制台输出关键结果
    print(f"\nKEY FINDINGS:")
    print("-" * 50)

    # 计算整体改善
    all_rmse_improvements = []
    all_d0_psnr_improvements = []
    all_d0_ssim_improvements = []
    all_x_psnr_improvements = []
    all_x_ssim_improvements = []
    all_kl_improvements = []

    for t_val in args.timestep_analysis:
        if (t_val in results['encoder_middle']['noise_prediction'] and
                t_val in results['encoder_middle']['d0_estimation'] and
                t_val in results['encoder_middle']['x_estimation'] and
                t_val in results['encoder_middle']['one_step_kl']):
            # 噪声预测
            em_noise = results['encoder_middle']['noise_prediction'][t_val]
            fm_noise = results['full_mamba']['noise_prediction'][t_val]
            em_rmse = np.mean([m['rmse'] for m in em_noise])
            fm_rmse = np.mean([m['rmse'] for m in fm_noise])
            all_rmse_improvements.append(fm_rmse - em_rmse)

            # d₀估计 (差异域)
            em_d0 = results['encoder_middle']['d0_estimation'][t_val]
            fm_d0 = results['full_mamba']['d0_estimation'][t_val]
            em_d0_psnr = np.mean([m['psnr'] for m in em_d0])
            fm_d0_psnr = np.mean([m['psnr'] for m in fm_d0])
            em_d0_ssim = np.mean([m['ssim'] for m in em_d0])
            fm_d0_ssim = np.mean([m['ssim'] for m in fm_d0])

            all_d0_psnr_improvements.append(fm_d0_psnr - em_d0_psnr)
            all_d0_ssim_improvements.append(fm_d0_ssim - em_d0_ssim)

            # x̂估计 (像素域)
            em_x = results['encoder_middle']['x_estimation'][t_val]
            fm_x = results['full_mamba']['x_estimation'][t_val]
            em_x_psnr = np.mean([m['psnr'] for m in em_x])
            fm_x_psnr = np.mean([m['psnr'] for m in fm_x])
            em_x_ssim = np.mean([m['ssim'] for m in em_x])
            fm_x_ssim = np.mean([m['ssim'] for m in fm_x])

            all_x_psnr_improvements.append(fm_x_psnr - em_x_psnr)
            all_x_ssim_improvements.append(fm_x_ssim - em_x_ssim)

            # KL散度
            em_kl = results['encoder_middle']['one_step_kl'][t_val]
            fm_kl = results['full_mamba']['one_step_kl'][t_val]
            em_kl_val = np.mean([m['kl'] for m in em_kl])
            fm_kl_val = np.mean([m['kl'] for m in fm_kl])

            all_kl_improvements.append(fm_kl_val - em_kl_val)

    if all_rmse_improvements and all_d0_psnr_improvements:
        print(
            f"Overall Avg ΔRMSE (noise prediction): {np.mean(all_rmse_improvements):.6f} ± {np.std(all_rmse_improvements):.6f}")
        print(
            f"Overall Avg ΔPSNR (d₀ estimation): {np.mean(all_d0_psnr_improvements):.2f} ± {np.std(all_d0_psnr_improvements):.2f}")
        print(
            f"Overall Avg ΔSSIM (d₀ estimation): {np.mean(all_d0_ssim_improvements):.4f} ± {np.std(all_d0_ssim_improvements):.4f}")
        print(
            f"Overall Avg ΔPSNR (x̂ estimation): {np.mean(all_x_psnr_improvements):.2f} ± {np.std(all_x_psnr_improvements):.2f}")
        print(
            f"Overall Avg ΔSSIM (x̂ estimation): {np.mean(all_x_ssim_improvements):.4f} ± {np.std(all_x_ssim_improvements):.4f}")
        print(
            f"Overall Avg ΔKL: {np.mean(all_kl_improvements):.6f} ± {np.std(all_kl_improvements):.6f}")

        # 判断改善方向
        rmse_better = np.mean(all_rmse_improvements) < 0  # RMSE越小越好
        d0_psnr_better = np.mean(all_d0_psnr_improvements) > 0  # PSNR越大越好
        d0_ssim_better = np.mean(all_d0_ssim_improvements) > 0  # SSIM越大越好
        x_psnr_better = np.mean(all_x_psnr_improvements) > 0  # PSNR越大越好
        x_ssim_better = np.mean(all_x_ssim_improvements) > 0  # SSIM越大越好
        kl_better = np.mean(all_kl_improvements) < 0  # KL散度越小越好

        print(f"\nMamba Decoder Performance:")
        print(f"  Noise Prediction: {'BETTER' if rmse_better else 'WORSE'} (ΔRMSE < 0 is better)")
        print(f"  d₀ Estimation (PSNR): {'BETTER' if d0_psnr_better else 'WORSE'} (ΔPSNR > 0 is better)")
        print(f"  d₀ Estimation (SSIM): {'BETTER' if d0_ssim_better else 'WORSE'} (ΔSSIM > 0 is better)")
        print(f"  x̂ Estimation (PSNR): {'BETTER' if x_psnr_better else 'WORSE'} (ΔPSNR > 0 is better)")
        print(f"  x̂ Estimation (SSIM): {'BETTER' if x_ssim_better else 'WORSE'} (ΔSSIM > 0 is better)")
        print(f"  KL Divergence: {'BETTER' if kl_better else 'WORSE'} (ΔKL < 0 is better)")

    print(f"\nAnalysis completed! Check {results_file} for detailed results.")


if __name__ == '__main__':
    main()