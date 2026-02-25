# Code for DTU course 02460 (Advanced Machine Learning Spring) by Paul Jeha and Jes Frellsen, 2024
# Version 1.1 (2024-02-05)
import torch
import torch.distributions as td
from torch.distributions.mixture_same_family import MixtureSameFamily


class TwoGaussians:
    def __init__(self):
        """
        A simple class to define an uneven mixture of two Gaussians.
        """

        mixture = td.Categorical(torch.tensor([1/5, 4/5]))
        components = td.Independent(
            td.Normal(
                torch.tensor(
                    [
                        [0.75, 0.75],
                        [0.25, 0.25],
                    ]
                ),
                torch.tensor(
                    [
                        [0.1, 0.1],
                        [0.1, 0.1],
                    ]
                ),
            ),
            1,
        )

        self.distribution = MixtureSameFamily(mixture, components)

        self.xlim = (0, 1)
        self.ylim = (0, 1)

    def __call__(self):
        """
        Return the distribution.
        Returns:
        distribution: [torch.distributions.Distribution]
        """
        return self.distribution


class ExtendedUniform(td.Uniform):
    def __init__(self, low, high, validate_args=None, outside_value=-float('inf')):
        """
        A uniform distribution that returns a constant value for values outside the support
        """
        super(ExtendedUniform, self).__init__(low, high, validate_args)
        self.outside_value = outside_value

    def log_prob(self, value):
        # Check if the value is within the support
        in_support = (value >= self.low) & (value <= self.high)
        safe_value = torch.where(in_support, value, self.low)
        log_prob = super().log_prob(safe_value)
        
        # Set log_prob to self.outside_value for values outside the support
        log_prob = torch.where(in_support, log_prob, torch.full_like(log_prob, self.outside_value))
        return log_prob
    
    @td.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return td.constraints.real



class Chequerboard:
    def __init__(self, grid_size=3, bounds=[0.0, 1.0]):
        """
        A simple class to define the Chequerboard distribution.
        """

        square_size = (bounds[1] - bounds[0]) / grid_size

        weights = []
        low_list = []
        high_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    low_x = bounds[0] + i * square_size
                    high_x = low_x + square_size
                    low_y = bounds[0] + j * square_size
                    high_y = low_y + square_size
                    
                    low_list.append([low_x, low_y])
                    high_list.append([high_x, high_y])
                    weights.append(1.0)

        mixture = td.Categorical(torch.tensor(weights))
        #components = td.Independent(td.uniform.Uniform(torch.tensor(low_list), torch.tensor(high_list)), 1)
        components = td.Independent(ExtendedUniform(torch.tensor(low_list), torch.tensor(high_list)), 1)
        self.distribution = MixtureSameFamily(mixture, components)

        self.xlim = bounds
        self.ylim = bounds

    def __call__(self):
        """
        Return the distribution.
        Returns:
        distribution: [torch.distributions.Distribution]
        """
        return self.distribution


class MNIST:
    def __init__(self, batch_size=1000, diffusion=False, binarized=False):
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

if __name__ == "__main__":
    # Illustrate how to make a DataLoader using the Checkerboard class
    toy = Chequerboard()
    loader = torch.utils.data.DataLoader(toy().sample((1000000,)), batch_size=1000, shuffle=True)
    print(next(iter(loader)).shape)
