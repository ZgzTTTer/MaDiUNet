import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from models.unet import VanillaUNet, ResUNet, MambaUNet, DenseUNet, SwinUNet
from models.ablation_unet import AblationUNet
from diffusion.DDPM import DDPMTrainer_cond, DDPMSampler_cond, DiffDDPMTrainer_cond, DiffDDPMSampler_cond
from diffusion.DDIM import DDIMTrainer_cond, DDIMSampler_cond, DiffDDIMTrainer_cond, DiffDDIMSampler_cond
from utils.metrics import calculate_metrics, calculate_metrics_rgb
from data.dataset import FootDataset2, MRIPET, MRISPECT, CTMRI, T1T2
from torch.utils.data import DataLoader
from PIL import Image
import lpips
from pytorch_fid import fid_score
import tempfile
import warnings
import time
import argparse

warnings.filterwarnings("ignore")


def save_gray(arr, path):
    """Save a single‐channel array in [0,1] as an 8‐bit grayscale PNG."""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)


def save_color(arr3, path):
    """Save a 3×HxW array in [0,1] as an 8‐bit RGB PNG."""
    img = Image.fromarray((arr3.transpose(1, 2, 0) * 255).astype(np.uint8))
    img.save(path)


def get_ablation_config(ablation_type):
    """Get the configuration for different ablation experiments"""
    configs = {
        "baseline": {
            "use_mamba_encoder": False,
            "use_mamba_middle": False,
            "use_mamba_decoder": False,
            "description": "Baseline ResUNet"
        },
        "encoder_only": {
            "use_mamba_encoder": True,
            "use_mamba_middle": False,
            "use_mamba_decoder": False,
            "description": "Mamba blocks in encoder only"
        },
        "encoder_middle": {
            "use_mamba_encoder": True,
            "use_mamba_middle": True,
            "use_mamba_decoder": False,
            "description": "Mamba blocks in encoder and middle"
        },
        "full_mamba": {
            "use_mamba_encoder": True,
            "use_mamba_middle": True,
            "use_mamba_decoder": True,
            "description": "Full MambaUNet"
        }
    }
    return configs[ablation_type]
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_type", type=str, default="encoder_middle",
                        choices=["baseline", "encoder_only", "encoder_middle", "full_mamba"],
                        help="Type of ablation experiment")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for evaluation")
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
    parser.add_argument("--save_weight_dir", type=str, default=".")
    parser.add_argument("--model_weight_path", type=str, default="model_epoch.pt")
    parser.add_argument("--output_dir", type=str, default="encoder_middle")
    parser.add_argument("--dataset_type", type=str, default="CTMRI")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    ablation_config = get_ablation_config(args.ablation_type)

    # Create output directory
    output_dir = os.path.join(args.save_weight_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)

    # Set channels based on dataset type
    if args.dataset_type in ["mripet", "mrispect"]:
        in_channels = 6  # RGB (target) + RGB (condition)
        out_channels = 3
    else:
        in_channels = 2
        out_channels = 1

    # Initialize model
    net_model = AblationUNet(
        args.T, args.ch, args.ch_mult, args.attn,
        args.num_res_blocks, args.dropout,
        in_channels=in_channels,
        out_channels=out_channels,
        use_mamba_encoder=ablation_config['use_mamba_encoder'],
        use_mamba_middle=ablation_config['use_mamba_middle'],
        use_mamba_decoder=ablation_config['use_mamba_decoder']
    ).to(device)

    # Load model weights
    model_path = os.path.join(args.save_weight_dir, args.model_weight_path)
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        if args.use_ema and 'ema' in checkpoint:
            print("Using EMA model for evaluation")
            net_model.load_state_dict(checkpoint['ema'])
        else:
            print("Using regular model for evaluation")
            net_model.load_state_dict(checkpoint['model'])
    else:
        print("Using regular model for evaluation")
        net_model.load_state_dict(checkpoint)

    net_model.eval()

    # Initialize sampler
    sampler = DiffDDPMSampler_cond(model=net_model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).to(device)

    # Setup dataset
    dataset = CTMRI(args.dataset_name, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize metrics
    ssim_values = []
    mse_values = []
    psnr_values = []
    lpips_values = []
    mae_values = []

    lpips_metric = lpips.LPIPS(net='vgg').to(device)

    sample_count = 0

    with tempfile.TemporaryDirectory() as fid_temp_dir:
        real_images_dir = os.path.join(fid_temp_dir, 'real')
        generated_images_dir = os.path.join(fid_temp_dir, 'generated')
        os.makedirs(real_images_dir)
        os.makedirs(generated_images_dir)

        with torch.no_grad():
            start_time = time.time()
            for batch in dataloader:
                target = batch['target'].to(device)
                condition = batch['condition'].to(device)

                random_noise = torch.randn_like(target)
                x_T = torch.cat((random_noise, condition), dim=1)

                generated_images = sampler(x_T)

                if out_channels == 1:
                    generated_image = generated_images[:, 0:1, :, :]
                    target_image = target
                    condition_image = condition
                else:
                    generated_image = generated_images[:, :out_channels, :, :]
                    target_image = target
                    condition_image = condition

                gen_img_np = generated_image[0].detach().cpu().numpy()
                tgt_img_np = target_image[0].detach().cpu().numpy()
                cond_img_np = condition_image[0].detach().cpu().numpy()

                if out_channels > 1:
                    gen_img_np = np.transpose(gen_img_np, (1, 2, 0))
                    tgt_img_np = np.transpose(tgt_img_np, (1, 2, 0))
                    cond_img_np = np.transpose(cond_img_np, (1, 2, 0))
                else:
                    gen_img_np = gen_img_np[0]
                    tgt_img_np = tgt_img_np[0]
                    cond_img_np = cond_img_np[0]

                # Normalize to [0, 1]
                gen_img_norm = np.clip((gen_img_np + 1) / 2.0, 0, 1)
                tgt_img_norm = np.clip((tgt_img_np + 1) / 2.0, 0, 1)
                cond_img_norm = np.clip((cond_img_np + 1) / 2.0, 0, 1)

                print(f"Sample {sample_count + 1}")
                print(f"Generated image min: {gen_img_norm.min()}, max: {gen_img_norm.max()}")
                print(f"Target image min: {tgt_img_norm.min()}, max: {tgt_img_norm.max()}")
                print(f"Condition image min: {cond_img_norm.min()}, max: {cond_img_norm.max()}")

                if out_channels == 1:
                    ssim_val = ssim(tgt_img_norm, gen_img_norm, data_range=1)
                else:
                    ssim_val = ssim(
                        tgt_img_norm, gen_img_norm, data_range=1,
                        multichannel=True, win_size=7
                    )

                ssim_values.append(ssim_val)
                mse_val = mean_squared_error(tgt_img_norm.flatten(), gen_img_norm.flatten())
                mse_values.append(mse_val)
                psnr_val = psnr(tgt_img_norm, gen_img_norm, data_range=1)
                psnr_values.append(psnr_val)
                mae_val = np.mean(np.abs(tgt_img_norm - gen_img_norm))
                mae_values.append(mae_val)

                if out_channels == 1:
                    gen_lpips = generated_image.repeat(1, 3, 1, 1)
                    tgt_lpips = target_image.repeat(1, 3, 1, 1)
                else:
                    gen_lpips = generated_image
                    tgt_lpips = target_image
                lpips_val = lpips_metric(gen_lpips, tgt_lpips).item()
                lpips_values.append(lpips_val)

                # Save images
                if out_channels == 1:
                    mode = 'L'
                    gen_img_to_save = (gen_img_norm * 255).astype(np.uint8)
                    tgt_img_to_save = (tgt_img_norm * 255).astype(np.uint8)
                else:
                    mode = 'RGB'
                    gen_img_to_save = (gen_img_norm * 255).astype(np.uint8)
                    tgt_img_to_save = (tgt_img_norm * 255).astype(np.uint8)

                Image.fromarray(gen_img_to_save, mode=mode).save(
                    os.path.join(generated_images_dir, f'generated_{sample_count + 1}.png'))
                Image.fromarray(tgt_img_to_save, mode=mode).save(
                    os.path.join(real_images_dir, f'real_{sample_count + 1}.png'))

                sample_output_dir = os.path.join(output_dir, f'sample_{sample_count + 1}')
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)
                if out_channels == 1:
                    cond_mode = 'L'
                    cond_img_to_save = (cond_img_norm * 255).astype(np.uint8)
                else:
                    cond_mode = 'RGB'
                    cond_img_to_save = (cond_img_norm * 255).astype(np.uint8)
                Image.fromarray(cond_img_to_save, mode=cond_mode).save(os.path.join(sample_output_dir, 'condition.png'))
                Image.fromarray(tgt_img_to_save, mode=mode).save(os.path.join(sample_output_dir, 'target.png'))
                Image.fromarray(gen_img_to_save, mode=mode).save(os.path.join(sample_output_dir, 'generated.png'))

                sample_count += 1
                if sample_count >= args.sample_num:
                    break

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time taken for image generation: {total_time:.2f} seconds")

            fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir],
                                                            args.batch_size, device, dims=2048)

    # Print final metrics
    average_ssim = np.mean(ssim_values)
    average_mse = np.mean(mse_values)
    average_psnr = np.mean(psnr_values)
    average_lpips = np.mean(lpips_values)
    average_mae = np.mean(mae_values)

    print(f"Average SSIM: {average_ssim:.4f}")
    print(f"Average MSE: {average_mse:.4f}")
    print(f"Average PSNR: {average_psnr:.4f}")
    print(f"Average LPIPS: {average_lpips:.4f}")
    print(f"Average MAE: {average_mae:.4f}")
    print(f"FID: {fid_value:.4f}")
