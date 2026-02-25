# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import flow as flow

class FlowPrior(flow.Flow):
    def __init__(self,base,transformation):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(FlowPrior, self).__init__(base, transformation)
    # def forward(self, z):
    #     return self.

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
        """
        1. Sortie du réseau (encoder_net) :
           Le réseau produit un vecteur de taille 2*M (ex: 64 si M=32).
           M est la dimension de l'espace latent (le nombre de "curseurs" qui 
           décrivent l'image).

        2. torch.chunk(..., 2, dim=-1) :
           On coupe ce vecteur en deux parties égales de taille M :
           - mean (mu) : Les M premières valeurs (position dans l'espace latent).
           - std (log_sigma) : Les M dernières valeurs (incertitude sur chaque dimension).

        3. Paramétrisation de la distribution :
           - loc=mean : La moyenne de la gaussienne pour chaque dimension latente.
           - scale=torch.exp(std) : On passe à l'exponentielle pour garantir que
             l'écart-type soit toujours POSITIF (le réseau prédit en fait un log-sigma).
           
        4. td.Independent(..., 1) :
           On traite ces M dimensions comme un seul vecteur aléatoire multidimensionnel.
        """

        # C'est ici qu'on a le reparametrization trick !

        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net, sigma=0.1): # sigma=0.1 est une bonne valeur de départ
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.sigma = sigma

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        # On récupère les paramètres du réseau
        # out aura une forme (batch_size, 2, 28, 28)
        # Le réseau ne sort plus que la moyenne (batch_size, 28, 28)
        mean = self.decoder_net(z)
        
        # On définit un écart-type fixe (scalaire ou tenseur de même taille)
        # La valeur 0.1 est appliquée à chaque pixel
        return td.Independent(td.Normal(loc=mean, scale=self.sigma), 2)

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
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

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """

        """
        Calcul de l'ELBO (Evidence Lower Bound) :
        
        1. log_prob(x) [Reconstruction] :
           - Le décodeur définit une distribution de Bernoulli pour CHAQUE pixel (28x28).
           - td.Independent(..., 2) regroupe ces pixels comme un seul événement.
           - .log_prob(x) calcule la log-vraisemblance : c'est l'équivalent probabiliste 
             de l'erreur de reconstruction (proche de la Binary Cross-Entropy).
           - Dimension de sortie : (batch_size,) -> un score global par image.

        2. kl_divergence [Régularisation] :
           - Mesure la distance entre la distribution de l'encodeur q(z|x) et le prior p(z).
           - td.Independent(..., 1) regroupe les dimensions de l'espace latent M.
           - Force l'espace latent à être organisé (proche d'une Normale Standard).
           - Dimension de sortie : (batch_size,) -> un score de complexité par image.

        3. Réduction :
           - L'ELBO est calculé par élément : (batch_size,) - (batch_size,)
           - torch.mean(..., dim=0) réduit le tout à un scalaire (moyenne du batch)
             pour que l'optimiseur puisse mettre à jour les poids du réseau.
        """

        q = self.encoder(x)
        z = q.rsample()

        log_q = q.log_prob(z)
        log_p_prior = self.prior.log_prob(z)

        elbo = torch.mean(self.decoder(z).log_prob(x) - (log_q - log_p_prior) , dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model by returning the mean of p(x|z).
        """
        # 1. Échantillonner dans l'espace latent à partir du Prior Flow
        z = self.prior.sample(torch.Size([n_samples]))
        
        # 2. Obtenir la distribution de sortie du décodeur
        dist = self.decoder(z)
        
        # 3. Retourner la moyenne (au lieu de dist.sample())
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

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()



if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model_Flow_cont.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--mask-type', type=str, default='checkerboard', choices=['checkerboard', 'channelwise','randominit'], help='type of mask to use in the coupling layers (default: %(default)s)')

    #python week2/02460_week2_vae.py train --model model_flow.pt --device cuda --batch-size 64 --epochs 10 --latent-dim 32
    #python week2/02460_week2_vae.py test --model model_flow.pt --device cuda --samples latent_space_flow.png --latent-dim 32
    #python week2/02460_week2_vae.py sample --model model_flow.pt --device cuda --samples sample_vae_flow_mnist.png --latent-dim 32

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
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
        nn.Linear(512, 784),        # On repasse à 784 au lieu de 784*2
        nn.Unflatten(-1, (28, 28))  # On unflatten directement en (28, 28)
    )

    # Define VAE model
    decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)
    
    elif args.mode =='test':
        all_z = []
        all_labels = []
        total_elbo = 0
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        model.eval()
        with torch.no_grad():
            for x, y in tqdm(mnist_test_loader):
                x = x.to(device)
                batch_elbo = model.elbo(x)
                total_elbo += batch_elbo.item() * x.size(0) # le size(0) permet de pondérer par la taille du batch
                q = model.encoder(x)
                z = q.rsample()
                all_z.append(z.cpu())
                all_labels.append(y)

        mean_elbo = total_elbo / len(mnist_test_loader.dataset)
        print(f"\nTest Set ELBO: {mean_elbo:.4f}")

        # Plots
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        all_z_cat = torch.cat(all_z, dim=0)
        all_labels_cat = torch.cat(all_labels, dim=0)

        if M>2:
            print("PCA processing...")
            pca = PCA(n_components=2)
            z_viz = pca.fit_transform(all_z_cat)
        else:
            z_viz = all_z_cat

        plt.figure(figsize=(10,10))
        scatter = plt.scatter(z_viz[:,0], z_viz[:,1], c=all_labels_cat)
        plt.colorbar(scatter, label='Digit Class')
        plt.title("2D projection of the latent space")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig(args.samples)
        print("Figure saved")
        





    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()
        with torch.no_grad():
            # Génère 64 images (les moyennes de p(x|z))
            samples = model.sample(64).cpu() 
            # On sauvegarde les moyennes directement
            save_image(samples.view(64, 1, 28, 28), args.samples)
