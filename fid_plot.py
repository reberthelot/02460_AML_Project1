from typing import Dict, Optional, Callable, Sequence

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from fid import compute_fid
from ddpm import ddpm_load
from vae import vae_load


@torch.no_grad()
def fid_table(
    x_real: torch.Tensor,
    gens: Dict[str, torch.Tensor],
    device: str = "cpu",
    ckpt: str = "mnist_classifier.pth",
) -> pd.DataFrame:
    x_real = x_real.to(device)
    rows = []
    for model, x_gen in gens.items():
        x_gen = x_gen.to(device)
        fid = float(compute_fid(x_real=x_real, x_gen=x_gen, device=device, classifier_ckpt=ckpt))
        rows.append({"model": model, "fid": fid})
        print(f"{model:>12} | FID={fid:.6f}")
    return pd.DataFrame(rows)


@torch.no_grad()
def fid_beta_table(
    x_real: torch.Tensor,
    sample_beta: Callable[[float, int, str], torch.Tensor],
    betas: Sequence[float],
    n: int,
    device: str = "cpu",
    ckpt: str = "mnist_classifier.pth",
) -> pd.DataFrame:
    x_real = x_real.to(device)
    rows = []
    for b in betas:
        x_gen = sample_beta(float(b), n, device).to(device)
        fid = float(compute_fid(x_real=x_real, x_gen=x_gen, device=device, classifier_ckpt=ckpt))
        rows.append({"beta": float(b), "fid": fid})
        print(f"beta={b:.6g} | FID={fid:.6f}")
    return pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)


def plot_fid_beta(df: pd.DataFrame, save: Optional[str] = None, title: str = "FID vs beta"):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="beta", y="fid", marker="o")
    plt.xscale("log")
    plt.xlabel("beta (log scale)")
    plt.ylabel("FID")
    plt.title(title)
    if save:
        import os
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        plt.savefig(save, dpi=200, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()


def plot_fid(df: pd.DataFrame, save: Optional[str] = None):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="model", y="fid")
    plt.ylabel("FID")
    plt.xlabel("Model")
    if save:
        import os
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        plt.savefig(save, dpi=200, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()


def mnist(n: int, root: str = "../data", device: str = "cpu") -> torch.Tensor:
    ds = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    dl = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=True, drop_last=True)
    x, _ = next(iter(dl))
    return x.to(device) * 2 - 1

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    n = 100

    betas = [1, 0.1, 0.01, 0.0001, 1e-6]

    x_real = mnist(n=n, device=device)

    print("Loading DDPM")
    ddpm, D_ddpm, _ = ddpm_load("models/model_ddpm_100.pt", dev)
    ddpm = ddpm.eval()

    with torch.no_grad():
        print("Sampling DDPM")
        x_ddpm = ddpm.sample(shape=(n, D_ddpm)).to(device).view(n, 1, 28, 28)

    print("Loading Latent DDPM (beta=1e-6)")
    latent_ddpm, D_lat, _ = ddpm_load("models/model_ddpm_bvae_unet.pt", dev)
    latent_ddpm = latent_ddpm.eval()

    hardcoded = {"K": 32, "latent_dim": 32, "prior": "mog", "beta": 1}

    print("Loading beta-VAEs and sampling")
    bvae_samples = {}
    beta_fid_rows = []

    for b in betas:
        path = f"results_beta_flow/model_flow_beta_{b}.pt"

        bvae = vae_load(
            checkpoint_path=path,
            hardcoded_arguments=hardcoded,
            device=dev,
        ).eval()

        decoder = bvae.decoder

        with torch.no_grad():
            z = torch.randn(n, hardcoded["latent_dim"], device=device)
            x_bvae = decoder(z).mean

            if x_bvae.dim() == 2:
                x_bvae = x_bvae.view(n, 1, 28, 28)
            elif x_bvae.dim() == 3:
                x_bvae = x_bvae.unsqueeze(1)

            if x_bvae.max() <= 1.0 and x_bvae.min() >= 0.0:
                x_bvae = x_bvae * 2 - 1

        bvae_samples[f"bVAE_beta={b}"] = x_bvae

        fid_val = float(
            compute_fid(
                x_real=x_real,
                x_gen=x_bvae,
                device=device,
                classifier_ckpt="mnist_classifier.pth",
            )
        )

        beta_fid_rows.append({"beta": b, "fid": fid_val})
        print(f"bVAE beta={b} | FID={fid_val:.6f}")

        if b == 1e-6:
            with torch.no_grad():
                z_lat = latent_ddpm.sample(shape=(n, D_lat)).to(device)
                x_latent = decoder(z_lat).mean

                if x_latent.dim() == 2:
                    x_latent = x_latent.view(n, 1, 28, 28)
                elif x_latent.dim() == 3:
                    x_latent = x_latent.unsqueeze(1)

                if x_latent.max() <= 1.0 and x_latent.min() >= 0.0:
                    x_latent = x_latent * 2 - 1

    gens = {
        "DDPM": x_ddpm,
        "LatentDDPM": x_latent,
        f"bVAE (β=0.0001)": bvae_samples["bVAE_beta=0.0001"],
    }

    df_models = fid_table(x_real, gens, device=device)
    print(df_models)
    plot_fid(df_models)

    df_beta = pd.DataFrame(beta_fid_rows).sort_values("beta")
    plot_fid_beta(df_beta)


if __name__ == "__main__":
    main()