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
    n = 5000

    x_real = mnist(n=n, device=device)

    ddpm, _, _ = ddpm_load("models/model_ddpm_100.pt", torch.device(device))
    ddpm = ddpm.eval()

    latent_ddpm, _, _ = ddpm_load("models/model_ddpm_bvae_unet.pt", torch.device(device))
    latent_ddpm = latent_ddpm.eval()

    with torch.no_grad():
        x_ddpm = ddpm.sample(shape=(n, 784)).to(device).view(-1, 1, 28, 28)
        x_latent_ddpm = latent_ddpm.sample(shape=(n, 784)).to(device).view(-1, 1, 28, 28)

    gens = {
        "DDPM": x_ddpm,
        "LatentDDPM": x_latent_ddpm,
    }

    df = fid_table(x_real, gens, device=device)
    print(df)
    plot_fid(df)


if __name__ == "__main__":
    main()