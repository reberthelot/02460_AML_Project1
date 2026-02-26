# Code for DTU course 02460 (Advanced Machine Learning Spring) by Paul Jeha and Jes Frellsen, 2024
# Modified to contain the MNIST class

import torch

class MNIST:
    def __init__(self, batch_size=64, diffusion=False, binarized=False):
        """
        A simple class to define the MNIST distribution.

        The behaviour is controlled by two boolean flags:
          * ``diffusion``: if ``True`` the pixel values are rescaled from [0,1]
            to [-1,1], which is common for diffusion models.
          * ``binarized``: if ``True`` the images are thresholded to 0/1 after
            adding a small bit of noise. This can be combined with ``diffusion``
            to produce a centered binary version.

        The four resulting modes are therefore:
          1. neither flag: standard noisy MNIST in [0,1]
          2. only ``diffusion``: centered in [-1,1]
          3. only ``binarized``: binary {0,1} (with noise)
          4. both flags: binary {0,1} but centred around zero before thresholding.
        """
        self.batch_size = batch_size
        from torchvision import datasets, transforms

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
        self.train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/', train=True, download=True, transform=transform), batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/', train=False, download=True, transform=transform), batch_size=self.batch_size, shuffle=True)
        self.xlim = (0,1)
        self.ylim = (0,1)

    # def __call__(self):
    #     """
    #     Return the distribution.
    #     Returns:
    #     distribution: [torch.distributions.Distribution]
    #     """
    #     return self.distribution


