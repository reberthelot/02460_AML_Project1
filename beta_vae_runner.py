import os, subprocess, re, argparse

# --- CLI Setup ---
parser = argparse.ArgumentParser(description="Beta-VAE Hyperparameter Runner")
parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'])
# List of beta values to test
parser.add_argument('--betas', type=list, default=[1e-6, 1e-4, 1e-2, 1e-1])
args_runner = parser.parse_args()

# --- Configuration ---
prior = args_runner.prior
betas = args_runner.betas
output_dir = f'results_beta_{prior}'
os.makedirs(output_dir, exist_ok=True)

print(f"\n=== TRAINING BETA-VAE SERIES: PRIOR={prior.upper()} ===")

for beta in betas:
    # Model name includes the beta value to avoid overwriting files
    model_name = f"model_{prior}_beta_{beta}.pt"
    model_path = os.path.join(output_dir, model_name)
    
    print(f"\n> Training for beta = {beta}...")
    
    # 1. Training Phase
    # Ensure your vae.py script is configured to accept the --beta argument
    train_proc = subprocess.run([
        "python", "vae.py", "train", 
        "--prior", prior, 
        "--model", model_path, 
        "--epochs", "10", 
        "--beta", str(beta)  # Pass the beta value
    ], capture_output=True, text=True)

    if train_proc.returncode != 0:
        print(f"   !!! TRAINING FAILED for beta={beta} !!!")
        print(f"   ERROR: {train_proc.stderr}")
        continue
    else:
        print(f"   Success! Model saved at: {model_path}")

    # 2. Evaluation Phase
    # Running a quick test to capture the ELBO for logging
    test_res = subprocess.run([
        "python", "vae.py", "test", 
        "--prior", prior, 
        "--model", model_path,
        "--beta", str(beta)
    ], capture_output=True, text=True)

    # Extracting the Average ELBO using Regex
    match = re.search(r"Average ELBO:\s*([-+]?\d*\.\d+|\d+)", test_res.stdout)
    if match:
        print(f"   Average ELBO: {match.group(1)}")
    else:
        print("   Warning: Could not extract ELBO from test output.")