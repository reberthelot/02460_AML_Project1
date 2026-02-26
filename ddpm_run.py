import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from ddpm import DDPM, FcNetwork, train
import MNIST as MNIST

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    

    # python ddpm_run.py train --model model_ddpm_mnist.pt --device cuda --epochs 10 --batch-size 64 --network unet --lr 1e-3
    # python ddpm_run.py sample --model model_ddpm_mnist.pt --device cuda --samples sample_ddpm_mnist.png
    

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    
    parser.add_argument('--plotname', type=str, default=None, help='filename for the loss plot (default: derived from model name)')
    parser.add_argument('--saved-folder',type=str, default='output_PartB',help='folder for outputs (default: %(default)s)')

    parser.add_argument('--binarized', type=bool, default=False, choices=[True, False], help='Whether or not to binarize the images (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    
    parser.add_argument('--network', type=str, default='fully', choices=['unet', 'fully'], help='Choose the network type (default: %(default)s)')
    parser.add_argument('--T', type=float, default=1000, metavar='V', help='Number of steps in the diffusion process (default: %(default)s)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', choices=['ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR','None'], help='Scheduler to Use (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')


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

    # Define prior distribution shape
    D = x_sample.shape[1]

    # Set the number of steps in the diffusion process
    T = args.T


    # Choose mode to run
    if args.mode == 'train':
        # Define the network
        if args.network == 'fully':
            num_hidden = 64
            network = FcNetwork(D, num_hidden)
        else :
            import unet
            network = unet.Unet()
        # Define model
        model = DDPM(network, T=T).to(args.device)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.scheduler == 'CosineAnnealingLR' :
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'ExponentialLR' :
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif args.scheduler == 'ReduceLROnPlateau' :
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        else : #None
            scheduler = None

        # Train model
        loss_history = train(model, optimizer, train_loader, args.epochs, args.device, scheduler)

        # Save model
        save_dict = {
            'model_state_dict': model.state_dict(),
            'network': args.network,
            'D': D,
        }
        if args.network == 'fully':
            save_dict['num_hidden'] = num_hidden

        # Ensure output_PartB directory exists for plots
        os.makedirs(args.saved_folder, exist_ok=True)

        # Determine plot filename
        if args.plotname:
            plot_filename = args.plotname
        else: # Derive from model name, e.g., model_ddpm_mnist.pt -> loss_model_ddpm_mnist.png
            model_base_name = os.path.splitext(args.model)[0]
            plot_filename = f"loss_{model_base_name}.png"
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title("Training Loss Evolution")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(args.saved_folder, plot_filename))
        print(f"Loss plot saved to {os.path.join(args.saved_folder,args.model)}")
        torch.save(save_dict, os.path.join(args.saved_folder,args.model))

    elif args.mode == 'sample':
        import numpy as np

        # Load the model
        checkpoint = torch.load(args.model, map_location=torch.device(args.device))
        print(f'Selected model type: {checkpoint["network"]}')
        if checkpoint['network'] == 'fully':
            loaded_num_hidden = checkpoint['num_hidden']
            network_to_use = FcNetwork(checkpoint['D'], loaded_num_hidden)
        elif checkpoint['network'] == 'unet':
            import unet
            network_to_use = unet.Unet()
        else:
            raise ValueError(f"Unknown network type: {checkpoint['network']}")
        # Initialize the DDPM model with the correct network
        model = DDPM(network_to_use, T=T).to(args.device) 
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        with torch.no_grad():
            samples = model.sample((64,D)).cpu()
            if not(args.binarized): # Diffusion
                samples = samples / 2 + 0.5 # Reverse transformation
            save_image(samples.view(64, 1, 28, 28), os.path.join(args.saved_folder,args.samples))