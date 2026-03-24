"""
This script provides a set of classes for loading and pre-processing the MNIST dataset.
The main classes are `MNIST` and `LatentMNIST`. `MNIST` loads the standard MNIST dataset,
while `LatentMNIST` can be used to project the data into a latent space using a
pre-trained encoder. The classes offer options for binarization and diffusion-style
pre-processing.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

class MNIST:
    """
    A simple class to define the MNIST distribution.
    """
    def __init__(self, batch_size=64, diffusion=False, binarized=False):
        """
        Initializes the MNIST dataset with optional transformations.

        Args:
            batch_size (int): The batch size for the data loaders.
            diffusion (bool): If True, rescale pixel values to [-1, 1].
            binarized (bool): If True, binarize the images to 0/1.
        """
        self.batch_size = batch_size

        # build transform step by step depending on the flags
        steps = [transforms.ToTensor(),
                 transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255)]

        # diffusion scaling (do this before binarization if both are set)
        if diffusion:
            steps.append(transforms.Lambda(lambda x: (x - 0.5) * 2.0))

        # binarize last if requested
        if binarized:
            # note: if diffusion=True the values are in [-1,1], but threshold at
            # 0.5 still works because the previous step moved the original 0.5 to 0.
            steps.append(transforms.Lambda(lambda x: (x > (0.0 if diffusion else 0.5)).float()))

        # finally flatten regardless of mode
        steps.append(transforms.Lambda(lambda x: x.flatten()))

        transform = transforms.Compose(steps)
        self.train_loader = DataLoader(datasets.MNIST('../data/', train=True, download=True, transform=transform), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(datasets.MNIST('../data/', train=False, download=True, transform=transform), batch_size=self.batch_size, shuffle=True)
        self.xlim = (0,1)
        self.ylim = (0,1)



class LatentMNIST:
    """
    Modified MNIST class that can project data into a latent space.
    If an 'encoder' is provided, the loaders will yield latent vectors (z).
    """
    def __init__(self, encoder=None, batch_size=64, diffusion=False, binarized=False, device='cpu'):
        """
        Initializes the LatentMNIST dataset.

        Args:
            encoder: A pre-trained encoder to transform the data.
            batch_size (int): The batch size for the data loaders.
            diffusion (bool): If True, rescale pixel values to [-1, 1].
            binarized (bool): If True, binarize the images to 0/1.
            device (str): The device to use for computations.
        """
        self.batch_size = batch_size
        self.device = device
        
        # 1. Build the standard pixel-level transforms
        steps = [transforms.ToTensor(),
                 transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255)]

        if diffusion:
            steps.append(transforms.Lambda(lambda x: (x - 0.5) * 2.0))

        if binarized:
            steps.append(transforms.Lambda(lambda x: (x > (0.0 if diffusion else 0.5)).float()))

        steps.append(transforms.Lambda(lambda x: x.flatten()))
        transform = transforms.Compose(steps)

        # 2. Initialize standard datasets
        train_ds = datasets.MNIST('../data/', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('../data/', train=False, download=True, transform=transform)

        # 3. If an encoder is provided, transform the entire dataset into Latent Space
        if encoder is not None:
            self.train_loader = self._convert_to_latent(train_ds, encoder)
            self.test_loader = self._convert_to_latent(test_ds, encoder)
        else:
            self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    def _convert_to_latent(self, dataset, encoder):
        """
        Helper to pass the dataset through the VAE encoder once.

        Args:
            dataset: The dataset to convert.
            encoder: The encoder to use for the conversion.

        Returns:
            A DataLoader with the latent representations of the data.
        """
        encoder.to(self.device)
        encoder.eval()
        
        all_z, all_y = [], []
        
        # Use a temporary loader to process the raw images
        temp_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for x, y in temp_loader:
                x = x.to(self.device)
                res = encoder(x)
                z = res.mean
                
                all_z.append(z.cpu())
                all_y.append(y)
        latent_ds = TensorDataset(torch.cat(all_z), torch.cat(all_y))
        return DataLoader(latent_ds, batch_size=self.batch_size, shuffle=True)
