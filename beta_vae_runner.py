import os
import torch
import torch.nn as nn
import numpy as np
import random
import argparse

import MNIST
from vae import train, vae_load_or_create
import flow

"""
Script for running grid search over beta values for Beta-VAE training with different priors.
Directly imports and uses VAE training utilities instead of subprocess calls.
"""

# --- CLI Configuration ---
parser = argparse.ArgumentParser(description="Beta-VAE Hyperparameter Runner")
parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'])
parser.add_argument('--betas', type=float, nargs='+', default=[1e-6, 1e-4, 1e-2, 1e-1])
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--mask-type', type=str, default='randominit', choices=['checkerboard', 'channelwise', 'randominit'], help='Mask type for flow prior')
parser.add_argument('--K', type=int, default=32, help='Number of components in MoG prior')
args_runner = parser.parse_args()

# --- Setup ---
prior = args_runner.prior
betas = args_runner.betas
output_dir = f'results_beta_{prior}'
os.makedirs(output_dir, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(args_runner.seed)
np.random.seed(args_runner.seed)
random.seed(args_runner.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args_runner.seed)

print(f"\n=== STARTING BETA-VAE GRID SEARCH: PRIOR={prior.upper()} ===")
print(f"Results will be saved in: {output_dir}")
print(f"Device: {args_runner.device}")

results_summary = []

# --- Main Grid Search Loop ---
for beta in betas:
    model_filename = f"model_{prior}_beta_{beta}.pt"
    model_path = os.path.join(output_dir, model_filename)
    
    print("\n" + "="*70)
    print(f"  RUNNING EXPERIMENT: beta = {beta}")
    print("="*70 + "\n")
    
    try:
        # --- 1. Training Phase ---
        print(f"Training VAE with beta={beta}, prior={prior}...")
        
        # Load data
        data = MNIST.MNIST(batch_size=args_runner.batch_size, diffusion=False, binarized=True)
        mnist_train_loader = data.train_loader
        mnist_test_loader = data.test_loader
        
        # Get sample to determine latent dimension
        x_sample = next(iter(mnist_train_loader))
        if isinstance(x_sample, (list, tuple)):
            x_sample = x_sample[0]
        D = x_sample.shape[1]
        M = 32  # Default latent dimension
        
        # Create new model
        model = vae_load_or_create(
            checkpoint_path=None,
            latent_dim=M,
            prior_type=prior,
            beta=beta,
            device=args_runner.device,
            K=args_runner.K,
            mask_type=args_runner.mask_type
        )
        
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args_runner.lr)
        
        # Train model
        elbo_history = train(model, optimizer, mnist_train_loader, args_runner.epochs, args_runner.device)
        
        # Save model
        save_dict = {
            'model_state_dict': model.state_dict(),
            'args': {
                'latent_dim': M,
                'prior': prior,
                'beta': beta,
                'K': args_runner.K,
                'mask_type': args_runner.mask_type
            }
        }
        torch.save(save_dict, model_path)
        print(f"Model saved to: {model_path}")
        
        # --- 2. Evaluation Phase ---
        print(f"\nEvaluating model for beta={beta}...")
        
        # Load trained model
        model = vae_load_or_create(
            checkpoint_path=model_path,
            latent_dim=M,
            prior_type=prior,
            beta=beta,
            device=args_runner.device,
            K=args_runner.K,
            mask_type=args_runner.mask_type
        )
        model.eval()
        
        # Compute average ELBO on test set
        total_elbo = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(args_runner.device)
                batch_elbo = model.elbo(x)
                total_elbo += batch_elbo.item()
                num_batches += 1
        
        avg_elbo = total_elbo / num_batches if num_batches > 0 else 0.0
        
        print(f"\n[SUCCESS] Beta: {beta} | Average ELBO: {avg_elbo:.4f}")
        results_summary.append((beta, avg_elbo))
        
    except Exception as e:
        print(f"\n[!] Training failed for beta={beta}. Error: {e}")
        print("Skipping to next...")
        continue

# --- Final Summary ---
print("\n" + "#"*30)
print("   FINAL RESULTS SUMMARY")
print("#"*30)
print(f"{'Beta':<15} | {'Average ELBO':<15}")
print("-" * 33)
for b, e in results_summary:
    print(f"{b:<15} | {e:<15.4f}")
print("#"*30)