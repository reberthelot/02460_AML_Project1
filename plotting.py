
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
from vae import vae_load
from fid import compute_fid
from MNIST import MNIST

def plot_samples(ddpm_samples, latent_ddpm_samples, vae_samples, save_path):
    """Generates and saves a 3x4 grid of samples from the three models."""
    fig, axs = plt.subplots(3, 4, figsize=(8, 6), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

    # Set row labels as y-axis for the first column
    axs[0, 0].set_ylabel('DDPM', rotation=0, labelpad=20, fontweight='bold', loc='center')
    axs[1, 0].set_ylabel('Latent DDPM', rotation=0, labelpad=20, fontweight='bold', loc='center')
    axs[2, 0].set_ylabel('VAE (Binarized)', rotation=0, labelpad=20, fontweight='bold', loc='center')

    for i in range(4):
        # DDPM samples
        axs[0, i].imshow(ddpm_samples[i].squeeze().cpu().numpy(), cmap='gray')
        axs[0, i].axis('off')

        # Latent DDPM samples
        axs[1, i].imshow(latent_ddpm_samples[i].squeeze().cpu().numpy(), cmap='gray')
        axs[1, i].axis('off')

        # VAE samples
        axs[2, i].imshow(vae_samples[i].squeeze().cpu().numpy(), cmap='gray')
        axs[2, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Sample plot saved to {save_path}")

def plot_posterior_prior(vae_binarized_model,z_prior_2d,z_posterior_2d,test_labels,z_prior_lddpm_2d,z_learned_lddpm_2d,z_post_lddpm_2d,latent_plot_path):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.set_theme(style="whitegrid")
    # Plotting
    axs[0].set_title(f"β-VAE Latent Space (β={vae_binarized_model.beta:.1f})")
    sns.kdeplot(x=z_prior_2d[:, 0], y=z_prior_2d[:, 1], ax=axs[0], fill=True, cmap="Blues", label="Prior (p(z))")
    scatter = axs[0].scatter(z_posterior_2d[:, 0], z_posterior_2d[:, 1], c=test_labels, cmap='tab10', alpha=0.5, s=5)
    legend1 = axs[0].legend()
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Agg. Posterior (q(z|x))', markerfacecolor='gray', markersize=10)]
    axs[0].legend(handles=legend_elements, loc='upper right')
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")

    # Plotting
    axs[1].set_title("Latent DDPM Distribution")
    sns.kdeplot(x=z_prior_lddpm_2d[:, 0], y=z_prior_lddpm_2d[:, 1], ax=axs[1], color='black', linestyles='--', label="DDPM Prior (N(0,1))")
    sns.kdeplot(x=z_learned_lddpm_2d[:, 0], y=z_learned_lddpm_2d[:, 1], ax=axs[1], color='blue', label="Learned DDPM Dist.")
    axs[1].scatter(z_post_lddpm_2d[:, 0], z_post_lddpm_2d[:, 1], alpha=0.3, s=5, label="Agg. VAE Posterior")
    axs[1].legend()
    axs[1].set_xlabel("PC 1")
    axs[1].set_ylabel("PC 2")

    fig.tight_layout()
    plt.savefig(latent_plot_path)
    plt.close()
    print(f"Latent space plot saved to {latent_plot_path}")

def main(args):
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Models ---
    print("Loading models...")
    ddpm_model, _, _ = ddpm_load(args.regular_ddpm, device)
    ddpm_model.eval()

    # For VAEs, we might need to provide hardcoded arguments if they weren't saved in the checkpoint.
    # We'll assume a standard configuration, but this might need adjustment.
    vae_hardcoded_args = {'latent_dim': 20, 'prior': 'gaussian', 'beta': 1.0} 
    
    vae_for_latent_model = vae_load(args.vae_for_latent_ddpm, vae_hardcoded_args, device).eval()
    latent_ddpm_model, _, _ = ddpm_load(args.latent_ddpm, device)
    latent_ddpm_model.eval()
    # Try loading with specific beta-VAE args if available, otherwise fallback
    try:
        binarized_vae_args = {'latent_dim': 20, 'prior': 'gaussian', 'beta': 5.0} # A guess for a beta-VAE
        vae_binarized_model = vae_load(args.regular_vae, binarized_vae_args, device).eval()
    except Exception:
        print("Could not load binarized VAE with beta=5.0, falling back to beta=1.0")
        vae_binarized_model = vae_load(args.regular_vae, vae_hardcoded_args, device).eval()

    latent_dim = vae_for_latent_model.encoder.encoder_net.network[-1].out_features // 2
    
    print("Models loaded successfully.")

    # --- 1. Generate and Plot Representative Samples ---
    # The Unet in this project expects a flattened input, same as the FcNetwork.
    # Therefore, we always sample with a flattened shape and reshape the output.
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
    
    num_fid_samples = 100
    print(f"Generating {num_fid_samples} samples from each model for FID and Timing...")

    # Load real data for comparison, scaled to [-1, 1] as required by FID function
    mnist_data = MNIST(batch_size=num_fid_samples, diffusion=True)
    real_images, _ = next(iter(mnist_data.test_loader))
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
    
    # Get a large batch of test data
    mnist_full_test = MNIST(batch_size=10000, binarized=True)
    test_data, test_labels = next(iter(mnist_full_test.test_loader))
    test_data = test_data.to(device)


    # ----- Left Plot: Beta-VAE -----
    with torch.no_grad():
        q_posterior = vae_binarized_model.encoder(test_data)
        z_posterior = q_posterior.mean.cpu().numpy()
        z_prior = vae_binarized_model.prior.sample(torch.Size([10000])).cpu().numpy()

    # PCA to 2D
    pca_vae = PCA(n_components=2)
    combined_z_vae = np.vstack([z_posterior, z_prior])
    pca_vae.fit(combined_z_vae)
    z_posterior_2d = pca_vae.transform(z_posterior)
    z_prior_2d = pca_vae.transform(z_prior)

    # ----- Right Plot: Latent DDPM -----
    vae_lddpm_latent_dim = vae_for_latent_model.encoder.encoder_net.network[-1].out_features // 2
    mnist_full_test_lddpm = MNIST(batch_size=10000, diffusion=False) # Data for this VAE might not be binarized
    test_data_lddpm, _ = next(iter(mnist_full_test_lddpm.test_loader))
    test_data_lddpm = test_data_lddpm.to(device)
    with torch.no_grad():
        # Aggregate posterior from the VAE used by the Latent DDPM
        q_posterior_lddpm = vae_for_latent_model.encoder(test_data_lddpm)
        z_posterior_lddpm = q_posterior_lddpm.mean.cpu().numpy()
        # Samples from the learned latent DDPM distribution
        z_learned_lddpm = latent_ddpm_model.sample(shape=(10000, vae_lddpm_latent_dim)).cpu().numpy()
        # True prior for the DDPM is N(0,1)
        z_prior_lddpm = torch.randn(10000, vae_lddpm_latent_dim).cpu().numpy()

    # PCA to 2D
    pca_lddpm = PCA(n_components=2)
    combined_z_lddpm = np.vstack([z_posterior_lddpm, z_learned_lddpm, z_prior_lddpm])
    pca_lddpm.fit(combined_z_lddpm)

    z_post_lddpm_2d = pca_lddpm.transform(z_posterior_lddpm)
    z_learned_lddpm_2d = pca_lddpm.transform(z_learned_lddpm)
    z_prior_lddpm_2d = pca_lddpm.transform(z_prior_lddpm)

    plot_posterior_prior(vae_binarized_model,
                         z_prior_2d,z_posterior_2d,
                         test_labels,
                         z_prior_lddpm_2d,z_learned_lddpm_2d,
                         z_post_lddpm_2d,
                         os.path.join(args.output_dir, "latent_space_comparison.png")
                        )
    
if __name__ == '__main__':

    # python part2_plotting.py output_PartB/model_ddpm_100.pt output_PartB/model_ddpm_bvae.pt results_beta_flow/model_flow_beta_1e-06.pt results_beta_flow/model_flow_beta_1.pt --output_dir output_PartB
    # python part2_plotting.py output_PartB/model_ddpm_100.pt output_PartB/model_ddpm_bvae_unet.pt results_beta_flow/model_flow_beta_1e-06.pt results_beta_flow/model_flow_beta_1.pt --output_dir output_PartB

    parser = argparse.ArgumentParser(description="Part 2 evaluation script for generative models.")
    parser.add_argument('regular_ddpm', type=str, help='Path to the regular DDPM model checkpoint (.pt)')
    parser.add_argument('latent_ddpm', type=str, help='Path to the latent DDPM model checkpoint (.pt)')
    parser.add_argument('vae_for_latent_ddpm', type=str, help='Path to the VAE model used for the latent DDPM (.pt)')
    parser.add_argument('regular_vae', type=str, help='Path to the regular VAE model trained on binarized MNIST (.pt)')
    parser.add_argument('--output_dir', type=str, default='output_PartB', help='Directory to save plots and results.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu, cuda)')
    
    # This is a dummy argument to make the script runnable without providing actual model files for testing.
    # In a real run, these would be paths to actual .pt files.
    # e.g. python part2_plotting.py models/ddpm.pt models/latent_ddpm.pt models/vae_for_lddpm.pt models/vae_binarized.pt
    # The script currently cannot run without the files. This is just for demonstration.
    
    # Create dummy files for demonstration if they don't exist
    # if not os.path.exists('model.pt'):
    #     torch.save({}, 'model.pt')

    # Example of how you might call this script:
    # python part2_plotting.py model.pt model.pt model.pt model.pt
    
    parsed_args = parser.parse_args()
    main(parsed_args)
