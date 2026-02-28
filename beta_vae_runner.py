import os
import subprocess
import sys
import re
import argparse

# --- CLI Configuration ---
parser = argparse.ArgumentParser(description="Beta-VAE Hyperparameter Runner")
parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'])
# nargs='+' allows passing multiple values: --betas 1e-6 1e-4 0.1
parser.add_argument('--betas', type=float, nargs='+', default=[1e-6, 1e-4, 1e-2, 1e-1])
args_runner = parser.parse_args()

# --- Setup ---
prior = args_runner.prior
betas = args_runner.betas
output_dir = f'results_beta_{prior}'
os.makedirs(output_dir, exist_ok=True)

def run_and_display(cmd):
    """
    Runs a command and streams the output (stdout & stderr) 
    to the terminal in real-time.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge stderr into stdout to catch tqdm bars
        text=True,
        bufsize=1
    )
    
    full_output = []
    
    # Read output line by line as it happens
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        full_output.append(line)
        
    process.stdout.close()
    return_code = process.wait()
    return return_code, "".join(full_output)

print(f"\n=== STARTING BETA-VAE GRID SEARCH: PRIOR={prior.upper()} ===")
print(f"Results will be saved in: {output_dir}")

results_summary = []

for beta in betas:
    model_filename = f"model_{prior}_beta_{beta}.pt"
    
    print("\n" + "="*70)
    print(f"  RUNNING EXPERIMENT: beta = {beta}")
    print("="*70 + "\n")
    
    # 1. Training Phase
    # Note: We separate folder and filename to avoid path conflicts in vae.py
    train_cmd = [
        "python", "vae.py", "train", 
        "--prior", prior, 
        "--saved-folder", output_dir, 
        "--model", model_filename, 
        "--epochs", "10", 
        "--beta", str(beta)
    ]
    
    ret_code, train_output = run_and_display(train_cmd)

    if ret_code != 0:
        print(f"\n[!] Training failed for beta={beta}. Skipping to next...")
        continue

    # 2. Evaluation Phase (Test)
    print(f"\n> Evaluating model for beta={beta}...")
    test_cmd = [
        "python", "vae.py", "test", 
        "--prior", prior, 
        "--saved-folder", output_dir,
        "--model", model_filename,
        "--beta", str(beta)
    ]
    
    ret_code, test_output = run_and_display(test_cmd)

    # 3. Extract final ELBO using Regex
    match = re.search(r"Average ELBO:\s*([-+]?\d*\.\d+|\d+)", test_output)
    if match:
        elbo_val = match.group(1)
        print(f"\n[SUCCESS] Beta: {beta} | Average ELBO: {elbo_val}")
        results_summary.append((beta, elbo_val))
    else:
        print(f"\n[WARNING] Training finished but ELBO value not found in output.")

# --- Final Summary ---
print("\n" + "#"*30)
print("   FINAL RESULTS SUMMARY")
print("#"*30)
print(f"{'Beta':<10} | {'Average ELBO':<15}")
print("-" * 28)
for b, e in results_summary:
    print(f"{b:<10} | {e:<15}")
print("#"*30)