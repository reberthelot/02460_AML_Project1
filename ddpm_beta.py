import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import MNIST
from vae import vae_load_or_create
from ddpm import DDPM, train, ddpm_load
from ddpm_models import LatentResNet
from fid import compute_fid
from utils import save_loss_plot

"""
Script for training Latent DDPMs on Beta-VAE models with different beta values and evaluating using FID scores.
"""

def main(args):
    """
    Main function to train DDPM models on latent spaces of Beta-VAEs for different beta values and compute FID scores.
    
    Args:
        args: Parsed command-line arguments containing training and evaluation parameters.
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all VAE model paths
    vae_model_paths = glob.glob(os.path.join(args.vae_dir, '*.pt'))
    
    if args.specific_model:
        vae_model_paths = [os.path.join(args.vae_dir, args.specific_model)]
        if not os.path.exists(vae_model_paths[0]):
            raise FileNotFoundError(f"The specified model was not found: {vae_model_paths[0]}")


    fid_results = []
    
    # Prepare real data for FID computation (only needs to be done once)
    print("Loading real MNIST data for FID calculation...")
    mnist_data = MNIST.MNIST(batch_size=args.fid_batch_size, diffusion=True, binarized=False)
    real_images_for_fid = next(iter(mnist_data.test_loader))
    if isinstance(real_images_for_fid, (list, tuple)):
        real_images_for_fid = real_images_for_fid[0]
    real_images_for_fid = real_images_for_fid.to(args.device)


    for vae_path in tqdm(vae_model_paths, desc="Training DDPMs for VAEs"):
        try:
            # --- 1. Setup and Model Checking ---
            base_name = os.path.basename(vae_path)
            try:
                beta_str = base_name.split('beta_')[1].replace('.pt', '')
                beta = float(beta_str)
            except (IndexError, ValueError) as e:
                print(f"\\nCould not parse beta from filename: {base_name}. Error: {e}. Skipping.")
                continue

            # Define model directory and name
            model_dir = os.path.join(args.output_dir, f"ddpm_beta_{beta_str}")
            model_filename = "model.pt"
            model_save_path = os.path.join(model_dir, model_filename)

            # Check if this model has already been trained
            if os.path.exists(model_save_path) and not args.force_retrain:
                print(f"\\nDDPM for beta={beta} already trained. Found at: {model_save_path}. Skipping training.")
            else:
                # --- 2. Load VAE and Data ---
                print(f"\\n--- Training DDPM for beta = {beta} ---")
                os.makedirs(model_dir, exist_ok=True)
                
                print(f"Loading VAE from: {vae_path}")
                vae_model = vae_load_or_create(
                    checkpoint_path=vae_path,
                    device=args.device
                )
                encoder = vae_model.encoder
                
                print("Creating latent space dataset...")
                latent_data = MNIST.LatentMNIST(encoder=encoder, batch_size=args.batch_size, diffusion=False, binarized=True, device=args.device)
                train_loader = latent_data.train_loader

                x_sample = next(iter(train_loader))
                if isinstance(x_sample, (list, tuple)):
                    x_sample = x_sample[0]
                D = x_sample.shape[1] # Latent dimension

                # --- 3. Define and Train DDPM ---
                print("Defining LatentResNet DDPM...")
                network = LatentResNet(
                    D,
                    hidden_dim=args.resnet_hidden_dim,
                    num_blocks=args.resnet_num_blocks,
                    time_dim=args.resnet_time_dim,
                )
                model = DDPM(network, T=args.T).to(args.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

                print(f"Starting training for {args.epochs} epochs...")
                loss_history = train(model, optimizer, train_loader, args.epochs, args.device, scheduler)
                
                # --- 4. Save Model ---
                print(f"Saving trained model to: {model_save_path}")
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'network': 'resnet',
                    'D': D,
                    'T': args.T,
                    'beta_vae': vae_path,
                    'hidden_dim': args.resnet_hidden_dim,
                    'num_blocks': args.resnet_num_blocks,
                    'time_dim': args.resnet_time_dim,
                }
                torch.save(save_dict, model_save_path)

                # Save loss plot
                save_loss_plot(loss_history, os.path.join(model_dir, "loss_plot.png"), 
                             title=f"DDPM Training Loss (beta={beta})", xlabel="Step", ylabel="Loss")

            # --- 5. Compute FID Score ---
            print(f"\\nComputing FID for beta = {beta}...")
            print('LOAD')
            # Load the trained DDPM
            ddpm_model, D, _ = ddpm_load(model_save_path, args.device)
            # We also need the corresponding VAE's decoder
            vae_model_for_fid = vae_load_or_create(
                checkpoint_path=vae_path,
                device=args.device
            )
            decoder_for_fid = vae_model_for_fid.decoder
            # Generate samples
            ddpm_model.eval()
            with torch.no_grad():
                latent_samples = ddpm_model.sample((args.fid_batch_size, D))
                # Decode latent samples to image space and adjust range to [-1, 1] for FID
                generated_images = decoder_for_fid(latent_samples).mean 
                generated_images = (generated_images.view(args.fid_batch_size, 1, 28, 28) * 2) - 1 # from [0, 1] to [-1, 1]
            # Save a batch of generated images for inspection
            save_image(generated_images, os.path.join(model_dir, "generated_samples.png"))

            # Compute FID
            print(real_images_for_fid.shape,generated_images.shape)
            fid_score = compute_fid(real_images_for_fid.view(args.fid_batch_size, 1, 28, 28), generated_images, device=args.device, classifier_ckpt=args.classifier_ckpt)
            print(f"FID Score for beta={beta}: {fid_score:.4f}")
            fid_results.append({'beta': beta, 'fid': fid_score})

        except Exception as e:
            print(f"\\nAn error occurred while processing {vae_path}: {e}")
            print("Skipping to the next model.")
            continue

    # --- 6. Save and Plot FID Results ---
    if not fid_results:
        print("\\nNo FID scores were computed. Exiting without saving results.")
        return

    print("\\n--- Aggregating and plotting results ---")
    fid_df = pd.DataFrame(fid_results)
    fid_df = fid_df.sort_values(by='beta').reset_index(drop=True)
    
    csv_path = os.path.join(args.output_dir, 'fid_results.csv')
    print(f"Saving FID results to {csv_path}")
    fid_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(fid_df['beta'], fid_df['fid'], 'o-', label='FID Score')
    plt.xscale('log')
    plt.xlabel('Beta VAE')
    plt.ylabel('FID Score')
    plt.title('FID Score vs. Beta VAE for Latent DDPMs (ResNet)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plot_path = os.path.join(args.output_dir, 'fid_vs_beta_plot.png')
    plt.savefig(plot_path)
    print(f"Final plot saved to {plot_path}")
    print("\\nScript finished successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Latent DDPMs on Beta-VAE models and compute FID.")
    
    # --- Paths and Directories ---
    parser.add_argument('--vae-dir', type=str, default='models/latent_ddpm/ddpm_beta/vae', help='Directory containing pre-trained Beta-VAE models.')
    parser.add_argument('--output-dir', type=str, default='models/latent_ddpm/ddpm_beta', help='Directory to save trained models, results, and plots.')
    parser.add_argument('--classifier-ckpt', type=str, default='mnist_classifier.pth', help='Path to the pre-trained MNIST classifier for FID.')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train each DDPM.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--T', type=int, default=1000, help='Number of steps in the diffusion process.')
    
    # --- Model Architecture ---
    parser.add_argument('--resnet-hidden-dim', type=int, default=512, help='Hidden dimension of LatentResNet.')
    parser.add_argument('--resnet-num-blocks', type=int, default=4, help='Number of residual blocks for LatentResNet.')
    parser.add_argument('--resnet-time-dim', type=int, default=16, help='Time-embedding dimension for LatentResNet.')

    # --- FID Calculation ---
    parser.add_argument('--fid-batch-size', type=int, default=1000, help='Number of images to generate for FID calculation.')

    # --- Control Flow ---
    parser.add_argument('--force-retrain', action='store_true', help='If set, retrain models even if they already exist.')
    parser.add_argument('--specific-model', type=str, default=None, help='If set, only train for this specific model filename in vae-dir.')


    args = parser.parse_args()
    
    print("--- Configuration ---")
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}")
    print("---------------------")

    main(args)
