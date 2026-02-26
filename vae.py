# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import MNIST
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import flow as flow
import random
import numpy as np
import os
import matplotlib.pyplot as plt

# Different prior types - Gaussian, MoG and Flow-based
class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def log_prob(self, z):
        return td.Independent(td.Normal(self.mean, self.std), 1).log_prob(z)

    def sample(self, shape):
        return td.Independent(td.Normal(self.mean, self.std), 1).sample(shape)

class MoGPrior(nn.Module):
    def __init__(self, M, K=10):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K
        self.mean = nn.Parameter(torch.randn(K, M))
        self.std_log = nn.Parameter(torch.zeros(K, M))
        self.mixture_logits = nn.Parameter(torch.zeros(K))

    def log_prob(self, z):
        mix = td.Categorical(logits=self.mixture_logits)
        comp = td.Independent(td.Normal(self.mean, torch.exp(self.std_log)), 1)
        return td.MixtureSameFamily(mix, comp).log_prob(z)

    def sample(self, shape):
        mix = td.Categorical(logits=self.mixture_logits)
        comp = td.Independent(td.Normal(self.mean, torch.exp(self.std_log)), 1)
        return td.MixtureSameFamily(mix, comp).sample(shape)

class FlowPrior(flow.Flow):
    def __init__(self,base,transformation):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(FlowPrior, self).__init__(base, transformation)
    

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """

        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """

        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder,beta):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """

        q = self.encoder(x)
        z = q.rsample()

        log_q = q.log_prob(z)

        log_p_prior = self.prior.log_prob(z)
        x_reshaped = x.view(-1, 28, 28) 

        log_p_x_z = self.decoder(z).log_prob(x_reshaped)

        elbo = torch.mean(log_p_x_z - self.beta * (log_q - log_p_prior) , dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model by returning the mean of p(x|z).
        """

        z = self.prior.sample(torch.Size([n_samples]))
        dist = self.decoder(z)
        return dist.mean
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
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

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    elbo_history = []

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            elbo_history.append(-loss.item())

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
    return elbo_history



if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    
    parser.add_argument('--plotname', type=str, default=None, help='filename for the loss plot (default: derived from model name)')
    parser.add_argument('--saved-folder',type=str, default='output_PartA',help='folder for outputs (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model_Flow_cont.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='type of prior p(z) (default: %(default)s)')
    parser.add_argument('--mask-type', type=str, default='checkerboard', choices=['checkerboard', 'channelwise','randominit'], help='type of mask to use in the coupling layers (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--K', type=int, default=32, metavar='N', help='The number of components in the mixture model. (default: %(default)s)')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1, metavar='V', help='beta parameter for the ELBO, must be between 0 and 1. (default: %(default)s)')
    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device


    # Load MNIST as binarized at 'thresshold' and create data loaders
    data = MNIST.MNIST(batch_size=args.batch_size,diffusion=not(True),binarized=True)
    mnist_train_loader = data.train_loader
    mnist_test_loader = data.test_loader
    # Define prior distribution
    M = args.latent_dim
    
    #prior type selection
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        prior = MoGPrior(M, args.K)
    else: # flowbase prior
        base = flow.GaussianBase(M)
        # Define transformations
        transformations = []    
        num_transformations = 12
        num_hidden = 256

        # Create the transformation mask
        if args.mask_type == 'checkerboard':
            mask = torch.Tensor([1 if (i) % 2 == 0 else 0 for i in range(M)])
        elif args.mask_type == 'channelwise':
            mask = torch.zeros((M,))
            mask[M//2:] = 1

        # Create the transformation layers
        for i in range(num_transformations):
            if args.mask_type == 'randominit':
                mask = torch.randint(0, 2, (M,))
            else :
                mask = (1-mask) # Flip the mask
            scale_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M),nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))
            transformations.append(flow.MaskedCouplingLayer(scale_net, translation_net, mask))
            
        prior = FlowPrior(base,transformations)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    beta = args.beta
    model = VAE(prior, decoder, encoder,beta).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        history = train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        # Ensure output_PartA directory exists for plots
        os.makedirs(args.saved_folder, exist_ok=True)

        # Determine plot filename
        if args.plotname:
            plot_filename = args.plotname
        else: # Derive from model name, e.g., model_ddpm_mnist.pt -> loss_model_ddpm_mnist.png
            model_base_name = os.path.splitext(args.model)[0]
            plot_filename = f"elbo_{model_base_name}.png"

        #Plotting training loss
        plt.figure(figsize=(10, 6))
        plt.plot(history, label='ELBO', color='royalblue', alpha=0.8)
        plt.xlabel('Iterations')
        plt.ylabel('ELBO')
        plt.title(f'ELBO Evolution - Prior: {args.prior.upper()}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plot_path = os.path.join(args.saved_folder, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"ELBO plot saved to: {plot_path}")
        torch.save(model.state_dict(), os.path.join(args.saved_folder, args.model))
    
    elif args.mode == 'test':
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        import numpy as np
        from matplotlib.lines import Line2D

        # Set visual style for publication-quality plots
        sns.set_theme(style="whitegrid", context="talk")
        
        # 1. Load the trained model and set to evaluation mode
        model.load_state_dict(torch.load(os.path.join(args.saved_folder, args.model), map_location=torch.device(args.device)))
        model.eval()
        
        all_z = []
        all_labels = []
        total_elbo = 0
        num_batches = 0
        
        # 2. Encode test data and calculate ELBO
        print(f"Evaluating model and encoding test data on {device}...")
        with torch.no_grad():
            for x, y in tqdm(mnist_test_loader):
                x = x.to(device)
                
                # Compute batch ELBO (returns the mean ELBO for the current batch)
                batch_elbo = model.elbo(x)
                total_elbo += batch_elbo.item()
                num_batches += 1
                
                # Use the mean of the encoder distribution for cleaner visualization
                q = model.encoder(x)
                all_z.append(q.mean.cpu()) 
                all_labels.append(y)
            
            # 3. Sample from the Prior to visualize its density
            print(f"Sampling from {args.prior} prior...")
            z_prior = model.prior.sample(torch.Size([10000])).cpu()

        # Compute final average ELBO across all test batches
        avg_elbo = total_elbo / num_batches
        print(f"\n[Test Result] Average ELBO: {avg_elbo:.4f}")

        all_z_cat = torch.cat(all_z, dim=0)
        all_labels_cat = torch.cat(all_labels, dim=0)

        # 4. Dimensionality Reduction using PCA
        # We fit PCA on the union of posterior and prior to ensure a shared coordinate system
        pca = PCA(n_components=2)
        combined_data = torch.cat([all_z_cat, z_prior], dim=0)
        pca.fit(combined_data)
        
        z_viz = pca.transform(all_z_cat)
        z_prior_viz = pca.transform(z_prior)

        # --- Plotting Construction ---
        fig, ax = plt.subplots(figsize=(14, 11))
        
        # Define a discrete 10-color palette for MNIST digits
        discrete_cmap = plt.get_cmap('tab10', 10)

        # 5. Plot the Posterior (POINTS) - zorder=1 (background)
        # alpha is set to 0.8 for visibility, but colorbar will be forced to 1.0 later
        scatter = ax.scatter(
            z_viz[:, 0], z_viz[:, 1], 
            c=all_labels_cat, 
            cmap=discrete_cmap, 
            s=45,               # Larger points for clarity
            alpha=0.8,          
            edgecolors='none', 
            vmin=-0.5, vmax=9.5,
            zorder=1
        )

        # 6. Plot the Prior (CONTOURS) - zorder=2 (on top of points)
        # We use a solid black color and thick lines to ensure they "cut" through the points
        sns.kdeplot(
            x=z_prior_viz[:, 0], y=z_prior_viz[:, 1], 
            fill=False,         # Line contours only
            thresh=0.01,        
            levels=12,          # Number of density levels
            color="black",      # High contrast against colors
            linewidths=2.5,     # Thick lines for visibility
            alpha=0.85,         
            ax=ax,
            zorder=2
        )

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Aggregate Posterior (q(z|x))',
                   markerfacecolor='gray', markersize=10),
            Line2D([0], [0], color='black', lw=2.5, label='Prior (p(z))')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, shadow=True)

        # 7. Intelligent Axis Scaling
        # Use percentiles to avoid outliers (common in Flow priors) squashing the main plot
        all_data_viz = np.vstack([z_viz, z_prior_viz])
        xlims = np.percentile(all_data_viz[:, 0], [0.5, 99.5])
        ylims = np.percentile(all_data_viz[:, 1], [0.5, 99.5])
        margin = 1.15
        ax.set_xlim(xlims[0] * margin, xlims[1] * margin)
        ax.set_ylim(ylims[0] * margin, ylims[1] * margin)

        # 8. Enhanced Categorical Colorbar
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
        # Force colorbar opacity to 1.0 to make colors more vibrant than the points
        cbar.set_alpha(1.0)
        cbar._draw_all() 
        
        cbar.set_label('MNIST Digit Class', fontweight='bold', labelpad=15)
        cbar.ax.set_yticklabels([f'Digit {i}' for i in range(10)])
        cbar.ax.tick_params(size=0) # Remove ticks for a cleaner modern look

        # 9. Title and Labels
        ax.set_title(f"VAE Latent Space Analysis ({args.prior.upper()})\nTest ELBO: {avg_elbo:.2f}", 
                     fontsize=18, pad=20, fontweight='bold')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        sns.despine(offset=10, trim=True)
        ax.grid(True, linestyle='--', alpha=0.2)

        # Save and output
        plt.savefig(os.path.join(args.saved_folder, args.samples), dpi=300, bbox_inches='tight')
        print(f"Figure saved in {os.path.join(args.saved_folder, args.samples)}")
        
    elif args.mode == 'sample':
        model.load_state_dict(torch.load(os.path.join(args.saved_folder, args.model), map_location=torch.device(args.device)))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu() 
            save_image(samples.view(64, 1, 28, 28), os.path.join(args.saved_folder, args.samples))
