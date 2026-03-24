import os, subprocess, re, time, random, argparse
import numpy as np

"""
Script for benchmarking VAE models with different priors across multiple runs.
Trains and evaluates VAEs, extracting ELBO scores and computing statistics.
"""

# --- CLI Setup ---
parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'])
parser.add_argument('--runs', type=int, default=5)
args_runner = parser.parse_args()

# --- Configuration ---
prior = args_runner.prior
# IMPORTANT: vae.py expects this folder to exist
os.makedirs('output_PartA', exist_ok=True) 

output_dir = f'results_{prior}'
os.makedirs(output_dir, exist_ok=True)

seeds = [random.randint(1, 10000) for _ in range(args_runner.runs)]
elbos = []
output_file = os.path.join(output_dir, f"benchmark_{prior}.txt")

print(f"\n=== BENCHMARKING PRIOR: {prior.upper()} ({args_runner.runs} runs) ===")

with open(output_file, "w") as f:
    f.write(f"Prior: {prior.upper()} | Seeds: {seeds}\n\n")
    
    for i, seed in enumerate(seeds):
        model_path = f"model_{prior}_run_{i}.pt" 
        print(f"Run {i+1}/{args_runner.runs} (Seed {seed}) - Training...")
        
        # 1. Training Phase - Catching errors
        train_proc = subprocess.run([
            "python", "vae.py", "train", 
            "--prior", prior, "--model", model_path, 
            "--epochs", "10", "--seed", str(seed)
        ], capture_output=True, text=True)

        if train_proc.returncode != 0:
            print(f"   !!! TRAINING CRASHED at Run {i+1} !!!")
            print(f"   ERROR: {train_proc.stderr}") # This tells you WHY it fails
            continue

        # 2. Testing Phase
        print(f"Run {i+1}/{args_runner.runs} - Testing...")
        test_res = subprocess.run([
            "python", "vae.py", "test", 
            "--prior", prior, "--model", model_path, "--seed", str(seed)
        ], capture_output=True, text=True)

        # 3. Regex Extraction
        match = re.search(r"Average ELBO:\s*([-+]?\d*\.\d+|\d+)", test_res.stdout)
        
        if match:
            elbo = float(match.group(1))
            elbos.append(elbo)
            print(f"   Done! ELBO: {elbo:.4f}")
            f.write(f"Run {i+1}: {elbo:.4f}\n")
        else:
            print("   Error: Could not find ELBO in output.")
            print(f"   DEBUG - Output received: {test_res.stdout[:100]}...") 

    if elbos:
        final_stat = f"\nFINAL: {np.mean(elbos):.2f} ± {np.std(elbos):.2f}"
        print(final_stat)
        f.write(final_stat)