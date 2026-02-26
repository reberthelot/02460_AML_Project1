import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from ddpm import DDPM, FcNetwork, train
import sys, os

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    sys.path.append(project_root_dir)
    import MNIST as MNIST

    # python ddpm.py train --data mnist --model model_ddpm_mnist.pt --device cuda --epochs 30 --batch-size 64 --network unet --lr 1e-3
    # python ddpm.py sample --data mnist --model model_ddpm.pt --device cuda --samples sample_ddpm_mnist.png
    
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--binarized', type=bool, default=False, choices=[True, False], help='Whether or not to binarize the images (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--network', type=str, default='fully', choices=['unet', 'fully'], help='Choose the network type (default: %(default)s)')
    parser.add_argument('--T', type=float, default=1000, metavar='V', help='Number of steps in the diffusion process (default: %(default)s)')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', choices=['ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR'], help='Scheduler to Use (default: %(default)s)')



    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    data = MNIST.MNIST(batch_size=args.batch_size,diffusion=not(args.binarized),binarized=args.binarized)
    train_loader = data.train_loader
    test_loader = data.test_loader
        
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
        if args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif args.scheduler == 'ReduceLROnPlateau':
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

            # Plot the density of the data data and the model samples
            coordinates = [[[x,y] for x in np.linspace(*data.xlim, 1000)] for y in np.linspace(*data.ylim, 1000)]
            prob = torch.exp(data().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob, extent=[data.xlim[0], data.xlim[1], data.ylim[0], data.ylim[1]], origin='lower', cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(data.xlim)
            ax.set_ylim(data.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()
