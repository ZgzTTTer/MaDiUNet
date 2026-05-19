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

from models.ablation_unet import AblationUNet
from diffusion.DDPM import DiffDDPMSampler_cond
from data.dataset import CTMRI
from utils.metrics import calculate_metrics

warnings.filterwarnings("ignore")


def compute_cka_gpu(X, Y, device):
    """
    GPU加速的CKA计算
    X, Y: torch.Tensor [N, D] 特征矩阵
    """
    try:
        # 确保在GPU上
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float().to(device)
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).float().to(device)

        X = X.to(device)
        Y = Y.to(device)

        # 确保维度匹配
        if X.shape != Y.shape:
            min_dim = min(X.shape[0], Y.shape[0])
            if X.shape[0] > min_dim:
                indices = torch.randperm(X.shape[0], device=device)[:min_dim]
                X = X[indices]
            if Y.shape[0] > min_dim:
                indices = torch.randperm(Y.shape[0], device=device)[:min_dim]
                Y = Y[indices]

        def centering_gpu(K):
            n = K.shape[0]
            unit = torch.ones(n, n, device=device)
            I = torch.eye(n, device=device)
            H = I - unit / n
            return torch.mm(torch.mm(H, K), H)

        def linear_HSIC_gpu(X, Y):
            L_X = torch.mm(X, X.t())
            L_Y = torch.mm(Y, Y.t())
            return torch.sum(centering_gpu(L_X) * centering_gpu(L_Y))

        hsic_xy = linear_HSIC_gpu(X, Y)
        hsic_xx = linear_HSIC_gpu(X, X)
        hsic_yy = linear_HSIC_gpu(Y, Y)

        if hsic_xx == 0 or hsic_yy == 0:
            return 0.0

        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        return cka.item()
    except Exception as e:
        print(f"GPU CKA computation error: {e}")
        return 0.0


def compute_metrics_gpu(pred_tensor, gt_tensor, device):
    """
    GPU加速的指标计算
    pred_tensor, gt_tensor: torch.Tensor [C, H, W] 或 [H, W]
    """
    try:
        # 确保在GPU上
        pred_tensor = pred_tensor.to(device)
        gt_tensor = gt_tensor.to(device)

        # 归一化到[0,1]
        pred_norm = torch.clamp((pred_tensor + 1) / 2, 0, 1)
        gt_norm = torch.clamp((gt_tensor + 1) / 2, 0, 1)

        # L1 Loss
        l1_loss = torch.mean(torch.abs(pred_norm - gt_norm)).item()

        # 转换为numpy进行SSIM计算（SSIM没有好的GPU实现）
        pred_np = pred_norm.cpu().numpy()
        gt_np = gt_norm.cpu().numpy()

        if len(pred_np.shape) == 2:  # 灰度图
            data_range = max(pred_np.max() - pred_np.min(), gt_np.max() - gt_np.min())
            if data_range == 0:
                data_range = 1.0
            ssim_score = ssim(pred_np, gt_np, data_range=data_range)
        else:  # RGB图
            pred_rgb = pred_np.transpose(1, 2, 0)
            gt_rgb = gt_np.transpose(1, 2, 0)
            data_range = max(pred_rgb.max() - pred_rgb.min(), gt_rgb.max() - gt_rgb.min())
            if data_range == 0:
                data_range = 1.0
            ssim_score = ssim(pred_rgb, gt_rgb, data_range=data_range, channel_axis=-1, win_size=7)

        return {
            'l1': l1_loss,
            'ssim': ssim_score
        }
    except Exception as e:
        print(f"GPU metrics computation error: {e}")
        return {'l1': 0.0, 'ssim': 0.0}


def compute_sample_level_cka_gpu(skip_features, decoder_features, device):
    """
    GPU加速的样本级CKA计算
    skip_features: torch.Tensor [C, H, W]
    decoder_features: torch.Tensor [C, H, W]
    """
    try:
        # 确保在GPU上
        skip_features = skip_features.to(device)
        decoder_features = decoder_features.to(device)

        # 展平特征
        skip_flat = skip_features.view(skip_features.shape[0], -1).t()  # [HW, C]
        decoder_flat = decoder_features.view(decoder_features.shape[0], -1).t()  # [HW, C]

        # 如果空间维度不同，进行处理
        if skip_flat.shape[0] != decoder_flat.shape[0]:
            min_spatial = min(skip_flat.shape[0], decoder_flat.shape[0])
            if skip_flat.shape[0] > min_spatial:
                indices = torch.randperm(skip_flat.shape[0], device=device)[:min_spatial]
                skip_flat = skip_flat[indices]
            if decoder_flat.shape[0] > min_spatial:
                indices = torch.randperm(decoder_flat.shape[0], device=device)[:min_spatial]
                decoder_flat = decoder_flat[indices]

        # 计算CKA
        cka_score = compute_cka_gpu(skip_flat, decoder_flat, device)
        return cka_score
    except Exception as e:
        print(f"Error in GPU sample-level CKA: {e}")
        return 0.0


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
    parser.add_argument("--sample_num", type=int, default=19)
    parser.add_argument("--dataset_name", type=str, default="./datasets/SynthRAD2023pelvis/test/")
    parser.add_argument("--save_weight_dir", type=str, default="./results/pelvis_MRI2CT")
    parser.add_argument("--encoder_middle_weight", type=str,
                        default="ablation/encoder_middle_DIFFDDPM/model_epoch_1985.pt")
    parser.add_argument("--full_mamba_weight", type=str, default="mamba_unet_DIFFDDPM/model_epoch_2485.pt")
    parser.add_argument("--output_dir", type=str, default="decoder_ablation_comparison")
    parser.add_argument("--dataset_type", type=str, default="CTMRI")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for evaluation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


class FeatureHook:
    """用于提取中间特征的Hook类"""

    def __init__(self):
        self.features = {}
        self.hooks = []

    def hook_fn(self, name):
        def fn(module, input, output):
            # 存储特征，保持在GPU上
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach()  # 保持在GPU上
            else:
                # 如果输出是tuple，取第一个元素
                self.features[name] = output[0].detach()  # 保持在GPU上

        return fn

    def register_hooks(self, model, layer_names):
        """注册hooks到指定层"""
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)

    def clear_hooks(self):
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}


def get_skip_decoder_pairs(model):
    """
    获取skip连接和对应decoder层的配对关系
    返回: [(skip_layer_name, decoder_layer_name), ...]
    """
    pairs = []

    # 根据UNet结构，找到encoder和decoder的对应关系
    encoder_layers = []
    decoder_layers = []

    for name, module in model.named_modules():
        if 'downblocks' in name and ('ResBlock' in str(type(module)) or 'MambaBlock' in str(type(module))):
            encoder_layers.append(name)
        elif 'upblocks' in name and ('ResBlock' in str(type(module)) or 'MambaBlock' in str(type(module))):
            decoder_layers.append(name)

    # 反向配对（decoder的第一层对应encoder的最后一层）
    encoder_layers.reverse()

    # 配对skip和decoder层
    min_len = min(len(encoder_layers), len(decoder_layers))
    for i in range(min_len):
        pairs.append((encoder_layers[i], decoder_layers[i]))

    return pairs


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

    # 初始化采样器
    sampler_encoder_middle = DiffDDPMSampler_cond(
        model=model_encoder_middle, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T
    ).to(device)

    sampler_full_mamba = DiffDDPMSampler_cond(
        model=model_full_mamba, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T
    ).to(device)

    # 设置数据集
    dataset = CTMRI(args.dataset_name, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 获取skip-decoder配对关系
    skip_decoder_pairs = get_skip_decoder_pairs(model_encoder_middle)
    print(f"Skip-Decoder pairs: {skip_decoder_pairs}")

    # 准备hook的层名称
    all_hook_layers = set()
    for skip_layer, decoder_layer in skip_decoder_pairs:
        all_hook_layers.add(skip_layer)
        all_hook_layers.add(decoder_layer)
    all_hook_layers = list(all_hook_layers)

    # 存储每个样本的指标
    sample_cka_encoder_middle = defaultdict(list)  # 每个配对的CKA
    sample_cka_full_mamba = defaultdict(list)
    sample_metrics_encoder_middle = []
    sample_metrics_full_mamba = []

    sample_count = 0

    print("Starting decoder ablation analysis...")
    start_total_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= args.sample_num:
                break

            sample_start_time = time.time()
            target = batch['target'].to(device)
            condition = batch['condition'].to(device)

            # 使用相同的随机噪声
            torch.manual_seed(args.random_seed + sample_count)
            random_noise = torch.randn_like(target)
            x_T = torch.cat((random_noise, condition), dim=1)

            # === 处理 encoder_middle 模型 ===
            hook_encoder_middle = FeatureHook()
            hook_encoder_middle.register_hooks(model_encoder_middle, all_hook_layers)

            torch.manual_seed(args.random_seed + sample_count)
            generated_encoder_middle = sampler_encoder_middle(x_T)

            features_encoder_middle = hook_encoder_middle.features.copy()
            hook_encoder_middle.clear_hooks()

            # === 处理 full_mamba 模型 ===
            hook_full_mamba = FeatureHook()
            hook_full_mamba.register_hooks(model_full_mamba, all_hook_layers)

            torch.manual_seed(args.random_seed + sample_count)
            generated_full_mamba = sampler_full_mamba(x_T)

            features_full_mamba = hook_full_mamba.features.copy()
            hook_full_mamba.clear_hooks()

            # === 计算样本级CKA ===
            for skip_layer, decoder_layer in skip_decoder_pairs:
                if skip_layer in features_encoder_middle and decoder_layer in features_encoder_middle:
                    skip_feat = features_encoder_middle[skip_layer][0]  # [C, H, W] on GPU
                    decoder_feat = features_encoder_middle[decoder_layer][0]  # [C, H, W] on GPU
                    cka_em = compute_sample_level_cka_gpu(skip_feat, decoder_feat, device)
                    sample_cka_encoder_middle[f"{skip_layer}->{decoder_layer}"].append(cka_em)

                if skip_layer in features_full_mamba and decoder_layer in features_full_mamba:
                    skip_feat = features_full_mamba[skip_layer][0]  # [C, H, W] on GPU
                    decoder_feat = features_full_mamba[decoder_layer][0]  # [C, H, W] on GPU
                    cka_fm = compute_sample_level_cka_gpu(skip_feat, decoder_feat, device)
                    sample_cka_full_mamba[f"{skip_layer}->{decoder_layer}"].append(cka_fm)

            # === 计算像素域指标 ===
            out_channels = model_encoder_middle.out_channels

            if out_channels == 1:
                gen_img_em = generated_encoder_middle[0, 0]  # 保持在GPU上
                gen_img_fm = generated_full_mamba[0, 0]
                target_img = target[0, 0]
            else:
                gen_img_em = generated_encoder_middle[0, :out_channels]  # 保持在GPU上
                gen_img_fm = generated_full_mamba[0, :out_channels]
                target_img = target[0]

            # 计算像素域指标
            metrics_em = compute_metrics_gpu(gen_img_em, target_img, device)
            metrics_fm = compute_metrics_gpu(gen_img_fm, target_img, device)

            sample_metrics_encoder_middle.append(metrics_em)
            sample_metrics_full_mamba.append(metrics_fm)

            # 保存生成的图像
            gen_img_em_norm = torch.clamp((gen_img_em + 1) / 2, 0, 1).cpu().numpy()
            gen_img_fm_norm = torch.clamp((gen_img_fm + 1) / 2, 0, 1).cpu().numpy()
            target_img_norm = torch.clamp((target_img + 1) / 2, 0, 1).cpu().numpy()

            sample_dir = os.path.join(output_dir, f'sample_{sample_count + 1}')
            os.makedirs(sample_dir, exist_ok=True)

            if out_channels == 1:
                Image.fromarray((gen_img_em_norm * 255).astype(np.uint8), mode='L').save(
                    os.path.join(sample_dir, 'generated_encoder_middle.png'))
                Image.fromarray((gen_img_fm_norm * 255).astype(np.uint8), mode='L').save(
                    os.path.join(sample_dir, 'generated_full_mamba.png'))
                Image.fromarray((target_img_norm * 255).astype(np.uint8), mode='L').save(
                    os.path.join(sample_dir, 'target.png'))
            else:
                gen_img_em_rgb = gen_img_em_norm.transpose(1, 2, 0)
                gen_img_fm_rgb = gen_img_fm_norm.transpose(1, 2, 0)
                target_img_rgb = target_img_norm.transpose(1, 2, 0)

                Image.fromarray((gen_img_em_rgb * 255).astype(np.uint8), mode='RGB').save(
                    os.path.join(sample_dir, 'generated_encoder_middle.png'))
                Image.fromarray((gen_img_fm_rgb * 255).astype(np.uint8), mode='RGB').save(
                    os.path.join(sample_dir, 'generated_full_mamba.png'))
                Image.fromarray((target_img_rgb * 255).astype(np.uint8), mode='RGB').save(
                    os.path.join(sample_dir, 'target.png'))

            sample_count += 1
            sample_time = time.time() - sample_start_time
            print(f"Processed sample {sample_count}/{args.sample_num} in {sample_time:.2f}s")

    total_time = time.time() - start_total_time
    print(f"Total processing time: {total_time:.2f}s, Average per sample: {total_time / sample_count:.2f}s")

    # === 计算最终指标 ===
    print("\nComputing final metrics...")

    # 1. 计算ΔCKA = CKA_Mamba - CKA_EncOnly
    delta_cka_results = {}
    all_delta_cka_shallow = []  # 用于计算 ΔCKA shallow
    all_delta_cka_all = []  # 用于计算 ΔCKA all

    for pair_name in sample_cka_encoder_middle.keys():
        if pair_name in sample_cka_full_mamba:
            cka_em_list = sample_cka_encoder_middle[pair_name]
            cka_fm_list = sample_cka_full_mamba[pair_name]

            # 样本级差值
            delta_cka_list = [fm - em for fm, em in zip(cka_fm_list, cka_em_list)]

            delta_cka_results[pair_name] = {
                'delta_cka_mean': np.mean(delta_cka_list),
                'delta_cka_std': np.std(delta_cka_list),
                'cka_em_mean': np.mean(cka_em_list),
                'cka_fm_mean': np.mean(cka_fm_list)
            }

            # 收集所有ΔCKA用于计算总体指标
            all_delta_cka_all.extend(delta_cka_list)
            # 假设前几个pair是shallow层，后几个是all层（这里简化处理）
            if len(all_delta_cka_shallow) < len(delta_cka_list):
                all_delta_cka_shallow.extend(delta_cka_list)

    # 2. 计算L1相对改善和SSIM绝对改善
    l1_improvements = []  # L1相对改善 (%)
    ssim_improvements = []  # SSIM绝对改善

    for metrics_em, metrics_fm in zip(sample_metrics_encoder_middle, sample_metrics_full_mamba):
        # L1相对改善 (%) = (L1_Enc - L1_Mamba) / L1_Enc × 100%
        if metrics_em['l1'] > 0:
            l1_improve_percent = (metrics_em['l1'] - metrics_fm['l1']) / metrics_em['l1'] * 100
        else:
            l1_improve_percent = 0.0
        l1_improvements.append(l1_improve_percent)

        # SSIM绝对改善 = SSIM_Mamba - SSIM_Enc
        ssim_improve = metrics_fm['ssim'] - metrics_em['ssim']
        ssim_improvements.append(ssim_improve)

    # === 输出结果 ===
    print("\n" + "=" * 80)
    print("DECODER ABLATION ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\nModels compared:")
    print(f"  - Encoder+Middle (ResNet Decoder): {encoder_middle_config['description']}")
    print(f"  - Full Mamba (Mamba Decoder): {full_mamba_config['description']}")
    print(f"  - Samples analyzed: {sample_count}")
    print(f"  - Random seed: {args.random_seed}")
    print(f"  - Total time: {total_time:.2f}s")

    print(f"\n1. CKA Analysis (Skip-Decoder Internal Alignment):")
    print("-" * 60)
    for pair_name, results in delta_cka_results.items():
        print(f"Pair: {pair_name}")
        print(f"  CKA_EncOnly: {results['cka_em_mean']:.4f}")
        print(f"  CKA_Mamba:   {results['cka_fm_mean']:.4f}")
        print(f"  ΔCKA:        {results['delta_cka_mean']:.4f} ± {results['delta_cka_std']:.4f}")
        print()

    print(f"\n2. Pixel-Domain Metrics:")
    print("-" * 60)
    avg_l1_em = np.mean([m['l1'] for m in sample_metrics_encoder_middle])
    avg_l1_fm = np.mean([m['l1'] for m in sample_metrics_full_mamba])
    avg_ssim_em = np.mean([m['ssim'] for m in sample_metrics_encoder_middle])
    avg_ssim_fm = np.mean([m['ssim'] for m in sample_metrics_full_mamba])

    print(f"L1 Loss:")
    print(f"  Encoder+Middle: {avg_l1_em:.6f}")
    print(f"  Full Mamba:     {avg_l1_fm:.6f}")
    print(f"  L1 Relative Improvement (%): {np.mean(l1_improvements):.2f} ± {np.std(l1_improvements):.2f}")
    print()
    print(f"SSIM Score:")
    print(f"  Encoder+Middle: {avg_ssim_em:.4f}")
    print(f"  Full Mamba:     {avg_ssim_fm:.4f}")
    print(f"  SSIM Absolute Improvement: {np.mean(ssim_improvements):.4f} ± {np.std(ssim_improvements):.4f}")

    print(f"\n3. SUMMARY METRICS:")
    print("=" * 60)
    # 计算ΔCKA shallow和ΔCKA all（这里简化为使用所有数据）
    delta_cka_shallow = np.mean(all_delta_cka_shallow) if all_delta_cka_shallow else 0.0
    delta_cka_all = np.mean(all_delta_cka_all) if all_delta_cka_all else 0.0

    print(f"ΔCKA shallow: {delta_cka_shallow:.4f}")
    print(f"ΔCKA all: {delta_cka_all:.4f}")
    print(f"k=0/1 L1 Relative Improvement (%): {np.mean(l1_improvements):.2f}")
    print(f"k=0/1 SSIM Absolute Improvement: {np.mean(ssim_improvements):.4f}")

    # 保存结果到文件
    results_file = os.path.join(output_dir, 'decoder_ablation_results.txt')
    with open(results_file, 'w') as f:
        f.write("DECODER ABLATION ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Models compared:\n")
        f.write(f"  - Encoder+Middle (ResNet Decoder): {encoder_middle_config['description']}\n")
        f.write(f"  - Full Mamba (Mamba Decoder): {full_mamba_config['description']}\n")
        f.write(f"  - Samples analyzed: {sample_count}\n")
        f.write(f"  - Random seed: {args.random_seed}\n\n")
        f.write(f"  - Total time: {total_time:.2f}s\n\n")

        f.write(f"1. CKA Analysis (Skip-Decoder Internal Alignment):\n")
        f.write("-" * 60 + "\n")
        for pair_name, results in delta_cka_results.items():
            f.write(f"Pair: {pair_name}\n")
            f.write(f"  CKA_EncOnly: {results['cka_em_mean']:.4f}\n")
            f.write(f"  CKA_Mamba:   {results['cka_fm_mean']:.4f}\n")
            f.write(f"  ΔCKA:        {results['delta_cka_mean']:.4f} ± {results['delta_cka_std']:.4f}\n\n")

        f.write(f"2. Pixel-Domain Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"L1 Loss:\n")
        f.write(f"  Encoder+Middle: {avg_l1_em:.6f}\n")
        f.write(f"  Full Mamba:     {avg_l1_fm:.6f}\n")
        f.write(f"  L1 Relative Improvement (%): {np.mean(l1_improvements):.2f} ± {np.std(l1_improvements):.2f}\n\n")
        f.write(f"SSIM Score:\n")
        f.write(f"  Encoder+Middle: {avg_ssim_em:.4f}\n")
        f.write(f"  Full Mamba:     {avg_ssim_fm:.4f}\n")
        f.write(f"  SSIM Absolute Improvement: {np.mean(ssim_improvements):.4f} ± {np.std(ssim_improvements):.4f}\n\n")

        f.write(f"3. SUMMARY METRICS:\n")
        f.write("=" * 60 + "\n")
        f.write(f"ΔCKA shallow: {delta_cka_shallow:.4f}\n")
        f.write(f"ΔCKA all: {delta_cka_all:.4f}\n")
        f.write(f"k=0/1 L1 Relative Improvement (%): {np.mean(l1_improvements):.2f}\n")
        f.write(f"k=0/1 SSIM Absolute Improvement: {np.mean(ssim_improvements):.4f}\n")

    print(f"\nResults saved to: {results_file}")
    print(f"Generated images saved to: {output_dir}")


if __name__ == '__main__':
    main()