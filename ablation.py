import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
import logging
import numpy as np
import argparse

from models.ablation_unet import AblationUNet
from diffusion.DDPM import DDPMTrainer_cond, DDPMSampler_cond, DiffDDPMTrainer_cond, DiffDDPMSampler_cond
from diffusion.DDIM import DDIMTrainer_cond, DDIMSampler_cond, DiffDDIMTrainer_cond, DiffDDIMSampler_cond
from data.dataset import FootDataset2, MRIPET, MRISPECT, CTMRI, T1T2
from utils.metrics import calculate_metrics, calculate_metrics_rgb
from utils.ema import EMA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_type", type=str, default="full_mamba",
                        choices=["baseline", "encoder_only", "encoder_middle", "full_mamba"],
                        help="Type of ablation experiment")
    parser.add_argument("--diffusion_name", type=str, default="DIFFDDPM",
                        choices=["DDPM", "DIFFDDPM", "DDIM", "DIFFDDIM"])
    parser.add_argument("--dataset_type", type=str, default="ctmri",
                        choices=["foot", "mripet", "mrispect", "ctmri", "T1T2"])
    parser.add_argument("--dataset_train_dir", type=str, default="../datasets/SynthRAD2023brain/train/")
    parser.add_argument("--dataset_val_dir", type=str, default="../datasets/SynthRAD2023brain/val/")
    parser.add_argument("--out_name", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument("--attn", nargs='+', type=int, default=[2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=3050)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--save_weight_dir", type=str, default=".")
    parser.add_argument("--resume_ckpt", type=str, default=".")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--val_start_epoch", type=int, default=3001)
    parser.add_argument("--val_num", type=int, default=20)
    # Add EMA-related arguments
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for validation and checkpoints")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    return parser.parse_args()


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


def should_save_model(current_metrics, best_metrics):
    """
    Determine if the model should be saved based on metric thresholds
    """
    ssim_threshold = 0.005  # Save if within 0.005 of best SSIM
    psnr_threshold = 0.01  # Save if within 0.01 of best PSNR

    ssim_condition = current_metrics['ssim'] >= best_metrics['ssim'] - ssim_threshold
    psnr_condition = current_metrics['psnr'] >= best_metrics['psnr'] - psnr_threshold
    mae_condition = current_metrics['mae'] < best_metrics['mae']  # Keep original condition for MAE

    # Update best metrics if current ones are better
    if current_metrics['ssim'] > best_metrics['ssim']:
        best_metrics['ssim'] = current_metrics['ssim']
    if current_metrics['psnr'] > best_metrics['psnr']:
        best_metrics['psnr'] = current_metrics['psnr']
    if current_metrics['mae'] < best_metrics['mae']:
        best_metrics['mae'] = current_metrics['mae']

    return ssim_condition or psnr_condition or mae_condition


def main():
    args = parse_args()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Get ablation configuration
    ablation_config = get_ablation_config(args.ablation_type)
    
    # Setup logging
    save_weight_dir = os.path.join(args.save_weight_dir, args.out_name)
    os.makedirs(save_weight_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_weight_dir, "training_log.log")),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Args: {args}")
    logging.info(f"Ablation Config: {ablation_config['description']}")
    logging.info(f"Mamba Encoder: {ablation_config['use_mamba_encoder']}")
    logging.info(f"Mamba Middle: {ablation_config['use_mamba_middle']}")
    logging.info(f"Mamba Decoder: {ablation_config['use_mamba_decoder']}")

    # Setup data
    dataset_classes = {
        "foot": FootDataset2,
        "ctmri": CTMRI,
        "mripet": MRIPET,
        "mrispect": MRISPECT,
        "T1T2": T1T2
    }

    dataset_class = dataset_classes[args.dataset_type]
    train_dataset = dataset_class(args.dataset_train_dir, image_size=args.image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_dataset = dataset_class(args.dataset_val_dir, image_size=args.image_size)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Determine input and output channels based on dataset type
    if args.dataset_type in ["mripet", "mrispect"]:
        in_channels = 6  # RGB + RGB
        out_channels = 3  # RGB output
    else:
        in_channels = 2  # Two grayscale images
        out_channels = 1  # One grayscale output

    # Initialize ablation model
    net_model = AblationUNet(
        args.T, args.ch, args.ch_mult, args.attn,
        args.num_res_blocks, args.dropout,
        in_channels=in_channels,
        out_channels=out_channels,
        use_mamba_encoder=ablation_config['use_mamba_encoder'],
        use_mamba_middle=ablation_config['use_mamba_middle'],
        use_mamba_decoder=ablation_config['use_mamba_decoder']
    ).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in net_model.parameters())
    trainable_params = sum(p.numel() for p in net_model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Initialize EMA if enabled
    if args.use_ema:
        ema = EMA(args.ema_decay)
        ema_model = ema.copy_to(net_model)
        ema_model.to(device)
        logging.info(f"EMA enabled with decay rate: {args.ema_decay}")
    else:
        ema = None
        ema_model = None

    optimizer = torch.optim.AdamW(
        net_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    # Initialize trainer and sampler
    trainer_classes = {
        "DDPM": DDPMTrainer_cond,
        "DIFFDDPM": DiffDDPMTrainer_cond,
        "DDIM": DDIMTrainer_cond,
        "DIFFDDIM": DiffDDIMTrainer_cond,
    }

    sampler_classes = {
        "DDPM": DDPMSampler_cond,
        "DIFFDDPM": DiffDDPMSampler_cond,
        "DDIM": DDIMSampler_cond,
        "DIFFDDIM": DiffDDIMSampler_cond,
    }

    trainer = trainer_classes[args.diffusion_name](net_model, args.beta_1, args.beta_T, args.T).to(device)

    # Create sampler with appropriate model (EMA or regular)
    eval_model = ema_model if args.use_ema else net_model
    sampler = sampler_classes[args.diffusion_name](eval_model, args.beta_1, args.beta_T, args.T).to(device)

    # Resume from checkpoint if specified
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            net_model.load_state_dict(checkpoint['model'])
            if args.use_ema and 'ema' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            net_model.load_state_dict(checkpoint)
        logging.info(f"Loaded checkpoint from {args.resume_ckpt}")

    # Training loop
    prev_time = time.time()
    best_metrics = {'psnr': -float('inf'), 'ssim': -float('inf'), 'mae': float('inf')}

    for epoch in range(args.start_epoch, args.n_epochs + 1):
        net_model.train()

        losses = []

        # Training step
        for batch in train_dataloader:
            optimizer.zero_grad()
            condition = batch['condition'].to(device)
            target = batch['target'].to(device)
            x_0 = torch.cat((target, condition), 1)

            loss = trainer(x_0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
            optimizer.step()

            # Update EMA model if enabled
            if args.use_ema:
                ema.step_ema(ema_model, net_model)

            losses.append(loss.item())

        # Logging
        avg_loss = np.mean(losses)
        time_elapsed = datetime.timedelta(seconds=(time.time() - prev_time))
        time_left = datetime.timedelta(seconds=(args.n_epochs - epoch) * (time.time() - prev_time))
        prev_time = time.time()

        logging.info(
            f"[Epoch {epoch}/{args.n_epochs}] "
            f"[ETA: {time_left}] "
            f"[Duration: {time_elapsed}] "
            f"[Loss: {avg_loss:.4f}]"
        )

        # Validation step
        if epoch >= args.val_start_epoch:
            eval_model = ema_model if args.use_ema else net_model
            eval_model.eval()
            metrics_list = []

            with torch.no_grad():
                for i, eval_batch in enumerate(val_dataloader):
                    if i >= args.val_num:  # Validate on specified number of samples
                        break

                    condition = eval_batch['condition'].to(device)
                    target = eval_batch['target'].to(device)

                    x_T = torch.cat((torch.randn_like(target), condition), 1)

                    generated_images = sampler(x_T)

                    # Extract only the target channels from the generated images
                    if args.dataset_type in ["mripet", "mrispect"]:
                        generated_image = generated_images[0, :3].cpu().numpy()  # Take first 3 channels for RGB
                        target_image = target[0].cpu().numpy()

                        # Convert from CHW to HWC format
                        generated_image = generated_image.transpose(1, 2, 0)
                        target_image = target_image.transpose(1, 2, 0)

                        metrics = calculate_metrics_rgb(generated_image, target_image)
                    else:
                        generated_image = generated_images[0, 0].cpu().numpy()  # Take first channel for grayscale
                        target_image = target[0, 0].cpu().numpy()
                        metrics = calculate_metrics(generated_image, target_image)

                    metrics_list.append(metrics)

            # Calculate average metrics
            avg_metrics = {
                key: np.mean([m[key] for m in metrics_list])
                for key in metrics_list[0].keys()
            }

            logging.info(
                f"[Epoch {epoch}] "
                f"[SSIM: {avg_metrics['ssim']:.4f}] "
                f"[PSNR: {avg_metrics['psnr']:.4f}] "
                f"[MAE: {avg_metrics['mae']:.4f}]"
            )

            # Save model if metrics meet the threshold criteria
            if should_save_model(avg_metrics, best_metrics):
                save_path = os.path.join(save_weight_dir, f'model_epoch_{epoch}.pt')
                if args.use_ema:
                    torch.save({
                        'model': net_model.state_dict(),
                        'ema': ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        # 'ablation_config': ablation_config,
                    }, save_path)
                else:
                    torch.save({
                        'model': net_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        # 'ablation_config': ablation_config,
                    }, save_path)
                logging.info(
                    f"Model saved at epoch {epoch} with "
                    f"SSIM: {avg_metrics['ssim']:.4f} (best: {best_metrics['ssim']:.4f}), "
                    f"PSNR: {avg_metrics['psnr']:.4f} (best: {best_metrics['psnr']:.4f}), "
                    f"MAE: {avg_metrics['mae']:.4f} (best: {best_metrics['mae']:.4f})"
                )

            # Save checkpoint periodically
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(save_weight_dir, f'ckpt_{epoch}.pt')
                if args.use_ema:
                    torch.save({
                        'model': net_model.state_dict(),
                        'ema': ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        # 'ablation_config': ablation_config,
                    }, checkpoint_path)
                else:
                    torch.save({
                        'model': net_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        # 'ablation_config': ablation_config,
                    }, checkpoint_path)

        # Save checkpoint every 500 epochs
        if epoch % 1000 == 0:
            checkpoint_path = os.path.join(save_weight_dir, f'ckpt_{epoch}.pt')
            if args.use_ema:
                torch.save({
                    'model': net_model.state_dict(),
                    'ema': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    # 'ablation_config': ablation_config,
                }, checkpoint_path)
            else:
                torch.save({
                    'model': net_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    # 'ablation_config': ablation_config,
                }, checkpoint_path)

    # Save final model
    final_path = os.path.join(save_weight_dir, 'final_model.pt')
    if args.use_ema:
        torch.save({
            'model': net_model.state_dict(),
            'ema': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': args.n_epochs,
            # 'ablation_config': ablation_config,
        }, final_path)
    else:
        torch.save({
            'model': net_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': args.n_epochs,
            # 'ablation_config': ablation_config,
        }, final_path)
    
    logging.info(f"Training completed! Final model saved to {final_path}")
    logging.info(f"Ablation experiment: {ablation_config['description']}")


if __name__ == "__main__":
    main()