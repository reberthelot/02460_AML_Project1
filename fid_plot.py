from typing import Dict, Optional

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from typing import Callable, Sequence

from fid import compute_fid


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


def noise(x: torch.Tensor, s: float) -> torch.Tensor:
    return torch.clamp(x + s * torch.randn_like(x), -1, 1)


def main():
    """
    Currently this file runs some test data using gaussian noise. No implementation of actual models.
    """
    device = "cpu"
    n = 5000
    x_real = mnist(n=n, device=device)

    gens = {
        "Perfect": noise(x_real, 0.0),
        "LowNoise": noise(x_real, 0.1),
        "MediumNoise": noise(x_real, 0.3),
        "HighNoise": noise(x_real, 0.8),
    }
    df = fid_table(x_real, gens, device=device)
    plot_fid(df)

    def sample_beta_dummy(beta: float, n: int, device: str) -> torch.Tensor:
        x = mnist(n=n, device=device)
        sigma = min(1.0, max(0.0, 5.0 * beta**0.5))  # arbitrary mapping just for testing
        return noise(x, sigma)

    betas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    dfb = fid_beta_table(x_real, sample_beta_dummy, betas, n=n, device=device)
    print("\nBeta table:")
    print(dfb)
    plot_fid_beta(dfb, title="Test: FID vs beta (dummy)")


if __name__ == "__main__":
    main()