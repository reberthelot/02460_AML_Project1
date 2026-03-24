"""
Visualization and evaluation script for comparing three generative models: DDPM, Latent DDPM, and VAE.
Generates plots for model samples, latent space distributions, and computes FID scores and sampling performance metrics.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.lines import Line2D


from ddpm import ddpm_load
from vae import vae_load_or_create
from fid import compute_fid
from MNIST import MNIST
from utils import plot_samples, plot_posterior_prior

def main(args):
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    ddpm_model, _, _ = ddpm_load(args.regular_ddpm, device)
    ddpm_model.eval()

    
    # Load VAE for latent DDPM
    vae_for_latent_model = vae_load_or_create(
        checkpoint_path=args.vae_for_latent_ddpm,
        device=device
    ).eval()
    latent_ddpm_model, _, _ = ddpm_load(args.latent_ddpm, device)
    latent_ddpm_model.eval()
    # Load binarized VAE with beta=5.0
    vae_binarized_model = vae_load_or_create(
        checkpoint_path=args.regular_vae,
        device=device
    ).eval()


    latent_dim = vae_for_latent_model.encoder.encoder_net.network[-1].out_features // 2
    
    print("Models loaded successfully.")

    # --- 1. Generate and Plot Representative Samples ---
    ddpm_sample_shape = (4, 784)
    reshape_for_plot = lambda x: x.view(-1, 1, 28, 28)

    with torch.no_grad():
        ddpm_samples_raw = ddpm_model.sample(shape=ddpm_sample_shape)
        ddpm_samples = reshape_for_plot(ddpm_samples_raw)
        
        latent_z = latent_ddpm_model.sample(shape=(4, latent_dim))
        latent_ddpm_samples = vae_for_latent_model.decoder(latent_z).mean.reshape(4, 1, 28, 28)
        
        vae_samples = vae_binarized_model.sample(n_samples=4).reshape(4, 1, 28, 28)

    plot_samples(ddpm_samples, latent_ddpm_samples, vae_samples, os.path.join(args.output_dir, "model_samples.png"))

    # --- 2. & 3. Compute FID Scores and Measure Sampling Time ---
    print("\n--- FID and Sampling Performance ---")
    
    num_fid_samples = 1000
    print(f"Generating {num_fid_samples} samples from each model for FID and Timing...")

    # Load real data for comparison, scaled to [-1, 1] as required by FID function
    mnist_data = MNIST(batch_size=num_fid_samples, diffusion=True)
    real_images, _ = next(iter(mnist_data.test_loader))
    # MNIST loader flattens samples; reshape for the FID classifier (conv2d expects 4D)
    real_images = real_images.view(-1, 1, 28, 28).to(device)

    # --- DDPM ---
    print('DDPM')
    ddpm_fid_shape = (num_fid_samples, 784)
    start_time = time.time()
    with torch.no_grad():
        ddpm_gen_fid_raw = ddpm_model.sample(shape=ddpm_fid_shape)
    ddpm_time = time.time() - start_time
    ddpm_sps = num_fid_samples / ddpm_time
    ddpm_gen_fid = reshape_for_plot(ddpm_gen_fid_raw)

    # --- Latent DDPM ---
    print('Latent DDPM')
    start_time = time.time()
    with torch.no_grad():
        latent_z_fid = latent_ddpm_model.sample(shape=(num_fid_samples, latent_dim))
        latent_gen_fid_raw = vae_for_latent_model.decoder(latent_z_fid).mean
    latent_ddpm_time = time.time() - start_time
    latent_sps = num_fid_samples / latent_ddpm_time
    # Rescale to [-1, 1]
    latent_gen_fid = (latent_gen_fid_raw.view(num_fid_samples, 1, 28, 28) * 2) - 1

    # --- VAE (Binarized) ---
    print('VAE')
    start_time = time.time()
    with torch.no_grad():
        vae_bin_gen_fid_raw = vae_binarized_model.sample(n_samples=num_fid_samples)
    vae_time = time.time() - start_time
    vae_sps = num_fid_samples / vae_time
    # Rescale to [-1, 1]
    vae_bin_gen_fid = (vae_bin_gen_fid_raw.view(num_fid_samples, 1, 28, 28) * 2) - 1

    print("Calculating FID scores...")
    fid_ddpm = compute_fid(real_images, ddpm_gen_fid, device)
    fid_latent_ddpm = compute_fid(real_images, latent_gen_fid, device)
    fid_vae = compute_fid(real_images, vae_bin_gen_fid, device)
    
    print("\nResults:")
    header = f"| {'Model':<13} | {'FID Score':<15} | {'Samples/Second':<16} |"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    print(f"| {'DDPM':<13} | {fid_ddpm:15.4f} | {ddpm_sps:<16.2f} |")
    print(f"| {'Latent DDPM':<13} | {fid_latent_ddpm:15.4f} | {latent_sps:<16.2f} |")
    print(f"| {'VAE':<13} | {fid_vae:15.4f} | {vae_sps:<16.2f} |")
    print(separator)
    print("Note: For the latent DDPM, the prompt mentioned reporting FID for different Beta values.")
    print("This is ambiguous for a DDPM. The reported score is for the provided model.")

    # --- 4. Plot Latent Space Distributions ---
    print("\nGenerating latent space comparison plot...")
    
    # Get a large batch of test data (binarized for the beta-VAE)
    mnist_full_test = MNIST(batch_size=10000, binarized=True)
    test_data, test_labels = next(iter(mnist_full_test.test_loader))
    test_data = test_data.to(device)

    # Define the latent dimension for sampling, ensuring it matches the model's configuration.
    # We use the binarized VAE's latent dim as the reference.
    vae_latent_dim = vae_binarized_model.encoder.encoder_net.network[-1].out_features // 2

    # Generate the three distributions
    with torch.no_grad():
        # 1. Aggregate Posterior from the Beta-VAE
        q_posterior = vae_binarized_model.encoder(test_data)
        z_posterior = q_posterior.mean.cpu().numpy()

        # 2. Prior from the Beta-VAE
        z_prior_vae = vae_binarized_model.prior.sample(torch.Size([10000])).cpu().numpy()

        # 3. Learned distribution from the Latent DDPM
        z_ddpm = latent_ddpm_model.sample(shape=(10000, vae_latent_dim)).cpu().numpy()

    # Combine all data to fit a single PCA instance
    # This ensures all distributions are projected into the same coordinate system.
    combined_data = np.vstack([z_posterior, z_prior_vae, z_ddpm])
    
    print("Fitting PCA on combined latent data...")
    pca = PCA(n_components=2)
    pca.fit(combined_data)

    # Transform each distribution into the shared 2D space
    z_posterior_2d = pca.transform(z_posterior)
    z_prior_vae_2d = pca.transform(z_prior_vae)
    z_ddpm_2d = pca.transform(z_ddpm)

    # Generate the plot
    plot_posterior_prior(
        z_posterior_2d,
        test_labels.numpy(),
        z_prior_vae_2d,
        z_ddpm_2d,
        vae_binarized_model.beta,
        os.path.join(args.output_dir, "latent_space_comparison.png")
    )

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Part 2 evaluation script for generative models.")
    parser.add_argument('regular_ddpm', type=str, help='Path to the regular DDPM model checkpoint (.pt)')
    parser.add_argument('latent_ddpm', type=str, help='Path to the latent DDPM model checkpoint (.pt)')
    parser.add_argument('vae_for_latent_ddpm', type=str, help='Path to the VAE model used for the latent DDPM (.pt)')
    parser.add_argument('regular_vae', type=str, help='Path to the regular VAE model trained on binarized MNIST (.pt)')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save plots and results.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu, cuda)')
    
    
    # e.g. python plotting.py models/ddpm.pt models/latent_ddpm.pt models/vae_for_lddpm.pt models/vae_binarized.pt
    # The script currently cannot run without the files. This is just for demonstration.

    # Example of how you might call this script:
    # python plotting.py model.pt model.pt model.pt model.pt
    
    parsed_args = parser.parse_args()
    main(parsed_args)
