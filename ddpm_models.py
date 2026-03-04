# MIT License

# Copyright (c) 2022 Muhammad Firmansyah Kasim

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following code is a modified version of the original code from the author. The original code can be found at
# https://github.com/mfkasim1/score-based-tutorial/blob/main/03-SGM-with-SDE-MNIST.ipynb

import torch
import torch.nn as nn

class Unet(nn.Module):
    """
    A simple U-Net architecture for MNIST that takes an input image and time
    """
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = nn.ModuleList([
            nn.Sequential(
                # input is the output of convs[4]
                nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                nn.LogSigmoid(),
            ),
            nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                nn.LogSigmoid(),
                nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)  # (..., 1, 28, 28)
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal


class ResidualBlock(nn.Module):
    """
    A residual block for vector processing.
    It uses GroupNorm and SiLU activation for better stability in DDPMs.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        # The core of the residual connection: x + f(x)
        return self.activation(x + self.block(x))

class LatentResNet(nn.Module):
    """
    A Residual MLP architecture for DDPMs operating on 1D latent vectors.
    """
    def __init__(self, D, hidden_dim=512, num_blocks=4, time_dim=128):
        """
        Parameters:
        D: [int] Dimension of the input latent vector.
        hidden_dim: [int] Width of the hidden layers.
        num_blocks: [int] Number of residual blocks.
        time_dim: [int] Dimension of the time embedding.
        """
        super().__init__()
        
        # store the hyper‑parameters for checkpointing
        self.D = D
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.time_dim = time_dim

        # 1. Time Embedding MLP
        # ...
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 2. Input Projection
        self.input_proj = nn.Linear(D + time_dim, hidden_dim)
        
        # 3. Residual Backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # 4. Output Header
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, D)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch_size, D) - Noisy latent vector.
        t: (batch_size, 1) - Time step (usually normalized between 0 and 1).
        """
        # Embed the time step
        t_emb = self.time_mlp(t) # (batch_size, time_dim)
        
        # Concatenate latent vector and time embedding
        h = torch.cat([x, t_emb], dim=1)
        
        # Initial projection
        h = self.input_proj(h)
        
        # Pass through residual blocks
        for block in self.res_blocks:
            h = block(h)
            
        # Final reconstruction of the noise (epsilon)
        output = self.final_layer(h)
        return output
    
    def __repr__(self):
            return (f"{self.__class__.__name__}(D={self.D}, hidden_dim={self.hidden_dim}, "
                    f"num_blocks={self.num_blocks}, time_dim={self.time_dim})")


class LatentUnet(nn.Module):
    """
    A U-Net-like architecture for latent vectors that takes an input vector and time.
    This is intended for use in a DDPM operating on the latent space of a VAE.
    """
    def __init__(self, D, dims=[256, 128, 64]):
        """
        Initialize the LatentUnet.

        Parameters:
        D: [int]
            The dimension of the input latent vector.
        dims: [list of int]
            A list of dimensions for the downscaling/upscaling path.
        """
        super().__init__()
        self.D = D
        self.dims = list(dims)

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        # Downscaling path
        # The first layer takes the latent vector D and time (1) as input
        current_dims = [D + 1] + dims
        
        for i in range(len(current_dims) - 1):
            self.down_layers.append(nn.Sequential(
                nn.Linear(current_dims[i], current_dims[i+1]),
                nn.ReLU()
            ))
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(dims[-1], dims[-1]),
            nn.ReLU()
        )
        
        # Upscaling path
        reversed_dims = dims[::-1]
        for i in range(len(reversed_dims) - 1):
            # Input is from previous up_layer + skip connection from down_layer
            in_dim = reversed_dims[i] + reversed_dims[i+1]
            out_dim = reversed_dims[i+1]
            self.up_layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU()
            ))
            
        # Final layer to map back to the original latent dimension D
        self.final_layer = nn.Linear(dims[0], D)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LatentUnet.

        Parameters:
        x: [torch.Tensor]
            The input latent vector of dimension `(batch_size, D)`.
        t: [torch.Tensor]
            The time steps of dimension `(batch_size, 1)`.
        
        Returns:
        [torch.Tensor]
            The output vector of dimension `(batch_size, D)`.
        """
        xt = torch.cat([x, t], dim=1)
        
        skip_connections = []
        signal = xt
        for i, layer in enumerate(self.down_layers):
            signal = layer(signal)
            if i < len(self.down_layers) - 1:
                skip_connections.append(signal)
        
        signal = self.bottleneck(signal)
        
        for layer in self.up_layers:
            skip_signal = skip_connections.pop()
            signal = torch.cat([signal, skip_signal], dim=1)
            signal = layer(signal)
            
        output = self.final_layer(signal)
        return output
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}(D={self.D}, dims={self.dims})"



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

