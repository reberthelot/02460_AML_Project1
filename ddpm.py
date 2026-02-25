# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """
        batch_size = x.shape[0]
        # Sampling t uniformly from {0,....,T-1})
        t = torch.randint(0, self.T, size=(batch_size, 1), device=x.device)
        # Sampling the uniform
        epsilon = torch.randn_like(x,device=x.device)
        output = self.network(torch.sqrt(self.alpha_cumprod[t]) * x + torch.sqrt(1 - self.alpha_cumprod[t]) * epsilon, t.float() / self.T)
        # Division of time by self.T in order to cap the time between 0 and 1, else time is predominating compared to images.
        neg_elbo = torch.sum((epsilon - output) ** 2, dim=1)
        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)
        batch_size = x_t.shape[0]
        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1) :
            if t > 0 : 
                z = torch.randn(shape).to(self.alpha.device)
            else :
                z = 0
            # Pass loop
            x_t = 1 / torch.sqrt(self.alpha[t]) * (x_t - (1-self.alpha[t]) /
                                torch.sqrt(1-self.alpha_cumprod[t]) * self.network(
                                    x_t,torch.full((batch_size,1),fill_value=t/self.T, dtype=torch.float).to(self.alpha.device)
                                    )
                                ) + torch.sqrt(self.beta[t]) * z
            # Division of time by self.T in order to cap the time between 0 and 1 - else it will predominate compared to the x.

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device, scheduler=None):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        raise TypeError("scheduler must be a torch.optim.lr_scheduler.LRScheduler instance or None")

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
        if scheduler is not None:
            scheduler.step() # Update learning rate after each epoch


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import sys, os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    sys.path.append(project_root_dir)
    import ToyData as ToyData

    # python week3/ddpm.py train --data tg --model model_ddpm.pt --device cuda --epochs 10
    # python week3/ddpm.py sample --data tg --model model_ddpm.pt --device cuda --samples sample_ddpm_tg.png


    # python week3/ddpm.py train --data cb --model model_ddpm.pt --device cuda
    # python week3/ddpm.py sample --data cb --model model_ddpm.pt --device cuda --samples sample_ddpm_cb.png
    
    # python week3/ddpm.py train --data mnist --model model_ddpm_mnist.pt --device cuda --epochs 30 --batch-size 64 --network unet --lr 1e-3
    # python week3/ddpm.py sample --data mnist --model model_ddpm.pt --device cuda --samples sample_ddpm_mnist.png
    
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--network', type=str, default='fully', choices=['unet', 'fully'], help='Choose the network type (default: %(default)s)')
    parser.add_argument('--T', type=float, default=1000, metavar='V', help='Number of steps in the diffusion process (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    n_data = 10000000
    if args.data == 'mnist':
        toy = ToyData.MNIST(batch_size=args.batch_size,diffusion=True)
        train_loader = toy.train_loader
        test_loader = toy.test_loader
        
    else:
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)

    # Print the shape of a batch of data
    x_sample = next(iter(train_loader))
    if isinstance(x_sample, (list, tuple)):
        x_sample = x_sample[0]

    print("\n--- DataLoader Information ---")
    print("Train Loader:")
    print(f"  Number of batches: {len(train_loader)}")
    print(f"  Batch size: {train_loader.batch_size}")
    if hasattr(train_loader.dataset, '__len__'):
        print(f"  Total elements: {len(train_loader.dataset)}")
    else:
        print("  Total elements: N/A (dataset does not have a length)")

    print("Test Loader:")
    print(f"  Number of batches: {len(test_loader)}")
    print(f"  Batch size: {test_loader.batch_size}")
    if hasattr(test_loader.dataset, '__len__'):
        print(f"  Total elements: {len(test_loader.dataset)}")
    else:
        print("  Total elements: N/A (dataset does not have a length)")
    print("----------------------------\n")

    # Define prior distribution
    D = x_sample.shape[1]

    # Initialize num_hidden to None, in case it's not a fully connected network
    num_hidden = None

    # Define the network
    if args.network == 'fully':
        num_hidden = 64
        network = FcNetwork(D, num_hidden)
    else :
        import unet
        network = unet.Unet()

    # Set the number of steps in the diffusion process
    T = args.T

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device, scheduler)

        # Save model
        save_dict = {
            'model_state_dict': model.state_dict(),
            'network': args.network,
            'D': D,
        }
        if args.network == 'fully':
            save_dict['num_hidden'] = num_hidden
        torch.save(save_dict, args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        checkpoint = torch.load(args.model, map_location=torch.device(args.device))
        
        loaded_network_type = checkpoint['network']        
        if loaded_network_type == 'fully':
            loaded_num_hidden = checkpoint['num_hidden']
            network_to_use = FcNetwork(checkpoint['D'], loaded_num_hidden)
        elif loaded_network_type == 'unet':
            import unet
            network_to_use = unet.Unet()
        else:
            raise ValueError(f"Unknown network type: {loaded_network_type}")

        # Re-initialize the DDPM model with the correct network
        model = DDPM(network_to_use, T=T).to(args.device) 
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.data == 'mnist':
            number_to_plot = 100
            model.eval()
            with torch.no_grad():
                samples = (model.sample((number_to_plot,D))).cpu()
            samples = samples / 2 + 0.5 # Reverse transformation
            samples = samples.view(-1, 1, 28, 28)
            save_image(samples, args.samples, nrow=10)

        else :
            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = (model.sample((10000,D))).cpu() 

            # Transform the samples back to the original space
            samples = samples /2 + 0.5

            # Plot the density of the toy data and the model samples
            coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()
