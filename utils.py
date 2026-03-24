import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

"""
Utility functions for model checkpoint manipulation, parameter counting, and visualization.
Provides tools for modifying saved checkpoints, inspecting model parameters, saving plots, and displaying distributions.
"""

def count_parameters(model):
    """Count the total number of parameters in a model or state dictionary."""
    return sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))


def modify_checkpoint(checkpoint_path: str, params: dict = None, count_params: bool = False):
    """
    Opens an existing .pt checkpoint, appends or modifies parameters,
    optionally counts and prints parameters, and saves it back.
    """
    # Hardcoded parameters to add/update
    

    try:
        # Load the existing checkpoint
        # Using map_location='cpu' to ensure it opens regardless of original device
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if not isinstance(checkpoint, dict) and (params or count_params):
            print("Error: Checkpoint is not a dictionary. Creating a new structure.")
            checkpoint = {'model_state_dict': checkpoint}
        
        if isinstance(checkpoint, dict):
            for k, v in checkpoint.items():
                    print(f"{k} = {v}")
        # Append/Update the parameters
        for key, value in params.items():
            if not(checkpoint.get(key)):
                checkpoint[key] = value
                print(f"Appended: {key} = {value}")

        if count_params:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            print(f"Total Parameters in State Dict: {count_parameters(state_dict):,}")

        torch.save(checkpoint, checkpoint_path)
        print(f"Successfully updated checkpoint at: {checkpoint_path}")

    except FileNotFoundError:
        print(f"Error: The file {checkpoint_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def save_loss_plot(loss_history, save_path, title="Training Loss Evolution", xlabel="Training Step", ylabel="Loss"):
    """
    Create and save a loss plot from training history.
    
    Args:
        loss_history (list): List of loss values at each step/epoch.
        save_path (str): Full path where the plot should be saved (including filename).
        title (str): Title for the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def plot_samples(ddpm_samples, latent_ddpm_samples, vae_samples, save_path):
    """Generates and saves a 3x4 grid of samples from the three models."""
    fig, axs = plt.subplots(3, 4, figsize=(8, 6), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

    # Set row labels as y-axis for the first column
    axs[0, 0].set_ylabel('DDPM', rotation=90, labelpad=20, fontweight='bold', loc='center')
    axs[1, 0].set_ylabel('Latent DDPM', rotation=90, labelpad=20, fontweight='bold', loc='center')
    axs[2, 0].set_ylabel('VAE (Binarized)', rotation=90, labelpad=20, fontweight='bold', loc='center')

    for i in range(4):
        # DDPM samples
        axs[0, i].imshow(ddpm_samples[i].squeeze().cpu().numpy(), cmap='gray')
        if i > 0:
            axs[0, i].axis('off')
        else:
            # Remove ticks but keep y-label on first column
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])

        # Latent DDPM samples
        axs[1, i].imshow(latent_ddpm_samples[i].squeeze().cpu().numpy(), cmap='gray')
        if i > 0:
            axs[1, i].axis('off')
        else:
            axs[1, i].set_xticks([])
            axs[1, i].set_yticks([])

        # VAE samples
        axs[2, i].imshow(vae_samples[i].squeeze().cpu().numpy(), cmap='gray')
        if i > 0:
            axs[2, i].axis('off')
        else:
            axs[2, i].set_xticks([])
            axs[2, i].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Sample plot saved to {save_path}")


def plot_posterior_prior(z_posterior_2d, test_labels, z_prior_vae_2d, z_ddpm_2d, beta_val, latent_plot_path):
    """
    Generates a single plot comparing the aggregate posterior against the VAE prior and the learned DDPM distribution.
    """
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.set_theme(style="whitegrid", context="talk")

    # 1. Plot the Aggregate Posterior (as points)
    scatter = ax.scatter(
        z_posterior_2d[:, 0], z_posterior_2d[:, 1],
        c=test_labels,
        cmap='tab10',
        s=45,
        alpha=0.7,
        edgecolors='none',
        vmin=-0.5, vmax=9.5,
        zorder=1
    )

    # 2. Plot the VAE Prior (as contours)
    sns.kdeplot(
        x=z_prior_vae_2d[:, 0], y=z_prior_vae_2d[:, 1],
        fill=False,
        thresh=0.01,
        levels=8,
        color="black",
        linewidths=2.5,
        alpha=0.85,
        ax=ax,
        zorder=2
    )

    # 3. Plot the Learned DDPM Distribution (as contours)
    sns.kdeplot(
        x=z_ddpm_2d[:, 0], y=z_ddpm_2d[:, 1],
        fill=False,
        thresh=0.01,
        levels=8,
        color="blue",
        linewidths=2.5,
        alpha=0.85,
        ax=ax,
        zorder=3
    )

    # 4. Create a comprehensive legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Aggregate Posterior ($q(z|x)$)',
               markerfacecolor='gray', markersize=12),
        Line2D([0], [0], color='black', lw=2.5, label='$\\beta$-VAE Prior ($p(z)$)'),
        Line2D([0], [0], color='blue', lw=2.5, label='Learned Latent DDPM')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, frameon=True, shadow=True)

    # 5. Add a colorbar for the digit classes
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_alpha(1.0)
    cbar._draw_all()
    cbar.set_label('MNIST Digit Class', fontweight='bold', labelpad=15)
    cbar.ax.set_yticklabels([f'Digit {i}' for i in range(10)])
    cbar.ax.tick_params(size=0)

    # 6. Set titles and labels
    ax.set_title(f"Latent Space Comparison ($\\beta={beta_val:.1f}$ VAE)", fontsize=20, pad=20, fontweight='bold')
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    # 7. Final styling
    sns.despine(offset=10, trim=True)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(latent_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Latent space plot saved to {latent_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify and/or count parameters in a saved model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pt model file")
    parser.add_argument("--append", nargs='+', default=[], help="Key-value pairs to append to the checkpoint (e.g., key1=value1 key2=value2)")
    parser.add_argument("--count", action='store_true', help="Count and print the parameters in the model")
    args = parser.parse_args()
    
    # Parse append arguments
    append_params = {}
    for item in args.append:
        try:
            key, value = item.split("=", 1)
            append_params[key] = value  # Store as string for simplicity
        except ValueError:
            print(f"Warning: Invalid append argument format: {item}. Expected key=value.")

    modify_checkpoint(checkpoint_path=args.model, params=append_params, count_params=args.count)