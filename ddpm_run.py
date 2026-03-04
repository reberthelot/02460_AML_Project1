import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from ddpm import DDPM, train, ddpm_load
import MNIST as MNIST

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    

    # python ddpm_run.py train --model model_ddpm.pt --device cuda --epochs 100 --batch-size 64 --network unet --lr 1e-3
    # python ddpm_run.py sample --model model_ddpm.pt --device cuda --samples sample_ddpm.png
    

    # python ddpm_run.py train --model model_ddpm_bvae_unet.pt --beta-vae results_beta_flow/model_flow_beta_1e-06.pt --device cuda --epochs 100 --batch-size 64 --lr 1e-3 --network unet
    # python ddpm_run.py sample --model model_ddpm_bvae_unet.pt --beta-vae results_beta_flow/model_flow_beta_1e-06.pt --device cuda --samples sample_ddpm_bvae_unet.png


    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample','test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--beta-vae', type=str, default=None, help='Full path to the beta vae. If not none, it will train based on a beta-vae(default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

    # The beta VAE different to none will over-ride the network argument to fully.

    parser.add_argument('--network', type=str, default='fully',choices=['unet', 'fully', 'resnet'],help='Choose the network type (default: %(default)s)')
    parser.add_argument('--T', type=int, default=1000, metavar='V',help='Number of steps in the diffusion process (default: %(default)s)')

    parser.add_argument('--plotname', type=str, default=None,help='filename for the loss plot (default: derived from model name)')
    parser.add_argument('--saved-folder', type=str, default='output_PartB',help='folder for outputs (default: %(default)s)')

    # latent‑unet dims and resnet hyperparameters
    parser.add_argument('--latent-dims', type=int, nargs='+',default=[256, 128, 64],help='dimensions for the latent U‑Net (space separated list)')
    parser.add_argument('--resnet-hidden-dim', type=int, default=512, help='hidden dimension of LatentResNet (default: %(default)s)')
    parser.add_argument('--resnet-num-blocks', type=int, default=4,help='number of residual blocks for LatentResNet')
    parser.add_argument('--resnet-time-dim', type=int, default=128,help='time‑embedding dimension for LatentResNet')
    

    parser.add_argument('--binarized', type=bool, default=False, choices=[True, False], help='Whether or not to binarize the images (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    

    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', choices=['ReduceLROnPlateau', 'ExponentialLR','CosineAnnealingLR','None'], help='Scheduler to Use (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    
    
    # We must update train_loader and test_loader
    if args.beta_vae :
        from vae import vae_load

        temporary_hardcoded = {
            'K': 32,
            'latent_dim': 32,
            'prior': 'mog',
            'beta': 1
            }
        # Temporary hardcode
        vae_model = vae_load(checkpoint_path=args.beta_vae, hardcoded_arguments=temporary_hardcoded,device=args.device)

        # Retrieve encoder and decoder
        encoder = vae_model.encoder
        decoder = vae_model.decoder
        args.binarized = True
        
        # Generate the data
        data = MNIST.LatentMNIST(encoder=encoder,batch_size=args.batch_size,diffusion=False,binarized=True,device=args.device)
        train_loader = data.train_loader
        test_loader = data.test_loader

    else :
        # Generate the data
        data = MNIST.MNIST(batch_size=args.batch_size,diffusion=not(args.binarized),binarized=args.binarized)
        train_loader = data.train_loader
        test_loader = data.test_loader



    # Print the shape of a batch of data
    x_sample = next(iter(train_loader))
    if isinstance(x_sample, (list, tuple)):
        x_sample = x_sample[0]

    # Define prior distribution shape
    D = x_sample.shape[1]
    
    # Set the number of steps in the diffusion process
    T = args.T

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

    


    # Choose mode to run
    if args.mode == 'train':
        import ddpm_models
        # Define the network
        if args.network == 'fully':
            num_hidden = 64
            network = ddpm_models.FcNetwork(D, num_hidden)
        elif args.network == 'unet':
            if args.beta_vae :
                network = ddpm_models.LatentUnet(D, dims=args.latent_dims)
            else :
                network = ddpm_models.Unet()
        elif args.network == 'resnet':
            network = ddpm_models.LatentResNet(D,
                hidden_dim=args.resnet_hidden_dim,
                num_blocks=args.resnet_num_blocks,
                time_dim=args.resnet_time_dim,
            )
        else:
            raise ValueError(f"Unknown network type: {args.network}")


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
            'T': T,
            'beta_vae': args.beta_vae,
            'latent_dim': args.latent_dim
        }
        if args.network == 'fully':
            save_dict['num_hidden'] = num_hidden

        # add extra hyper‑parameters when appropriate
        if isinstance(network, ddpm_models.LatentUnet):
            save_dict['dims'] = network.dims
        if isinstance(network, ddpm_models.LatentResNet):
            save_dict['hidden_dim'] = network.hidden_dim
            save_dict['num_blocks'] = network.num_blocks
            save_dict['time_dim'] = network.time_dim

        # Ensure output_PartB directory exists for plots
        os.makedirs(args.saved_folder, exist_ok=True)

        # Determine plot filename
        if args.plotname:
            plot_filename = args.plotname
        else: # Derive from model name, e.g., model_ddpm.pt -> loss_model_ddpm.png
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

        model, _, _ = ddpm_load(os.path.join(args.saved_folder, args.model), args.device)

        model.eval()
        with torch.no_grad():
            if args.beta_vae :
                samples = decoder(model.sample((64,D))).mean.cpu()
            else :
                samples = model.sample((64,D)).cpu()
            if not(args.binarized): # Diffusion
                samples = samples / 2 + 0.5 # Reverse transformation
            save_image(samples.view(64, 1, 28, 28), os.path.join(args.saved_folder,args.samples))