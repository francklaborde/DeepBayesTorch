import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deconv import DeconvLayer
from mlp import MLPLayer


class Generator(nn.Module):
    def __init__(
        self,
        input_shape,
        dimH,
        dimZ,
        dimY,
        n_channel,
        last_activation,
        name="generator",
        decoder_input_shape=None,
    ):
        super(Generator, self).__init__()

        # Define p(z|y) as an MLP
        fc_layers_pzy = [dimY, dimH, dimZ * 2]
        self.pzy_layers = nn.ModuleList()
        for i in range(len(fc_layers_pzy) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pzy) - 1 else "relu"
            layer_name = f"{name}_pzy_mlp_l{i}"
            self.pzy_layers.append(
                MLPLayer(fc_layers_pzy[i], fc_layers_pzy[i + 1], activation, layer_name)
            )

        # Define p(x|z)
        filter_width = 5
        if decoder_input_shape is None:
            decoder_input_shape = [
                (4, 4, n_channel * 4),
                (8, 8, n_channel * 2),
                (16, 16, n_channel),
                (input_shape[1], input_shape[2], input_shape[0]),
            ]

        fc_layers_pxz = [dimZ, dimH, int(np.prod(decoder_input_shape[0]))]
        self.mlp_layers_pxz = nn.ModuleList()
        for i in range(len(fc_layers_pxz) - 1):
            layer_name = f"{name}_pxz_mlp_l{i}"
            self.mlp_layers_pxz.append(
                MLPLayer(fc_layers_pxz[i], fc_layers_pxz[i + 1], "relu", layer_name)
            )

        self.conv_layers = nn.ModuleList()
        for i in range(len(decoder_input_shape) - 1):
            activation = (
                last_activation if i + 1 == len(decoder_input_shape) - 1 else "relu"
            )
            layer_name = f"{name}_conv_l{i}"
            output_shape = decoder_input_shape[i + 1]
            input_shape = decoder_input_shape[i]
            filter_shape = (
                input_shape[-1],  # in_channels
                output_shape[-1],  # out_channels
                filter_width,  # filter height
                filter_width,  # filter width
            )
            self.conv_layers.append(
                DeconvLayer(
                    input_shape, output_shape, filter_shape, activation, layer_name
                )
            )
        print("decoder shared Conv Net of size", decoder_input_shape)

    def pzy_params(self, y):
        out = y
        for layer in self.pzy_layers:
            out = layer(out)
        mu, log_sig = torch.chunk(out, 2, dim=1)  # Split into mu and log_sigma
        return mu, log_sig

    def pxz_params(self, z):
        out = z
        for layer in self.mlp_layers_pxz:
            out = layer(out)

        # Reshape MLP output to match the first decoder input shape
        decoder_initial_shape = self.conv_layers[0].input_shape
        out = out.view(
            out.shape[0],
            decoder_initial_shape[2],  # Channels
            decoder_initial_shape[0],  # Height
            decoder_initial_shape[1],  # Width
        )

        for layer in self.conv_layers:
            out = layer(out)
        return out


def sample_gaussian(mu, log_sig):
    return mu + torch.exp(log_sig) * torch.randn_like(mu)


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from torch.utils.data import DataLoader

    from utils.cifar10 import CustomCIFAR10Dataset

    # CIFAR-10 dimensions
    input_shape = (3, 32, 32)  # CIFAR-10 images
    dimH = 256
    dimZ = 128
    dimY = 10
    n_channel = 64

    # Instantiate generator
    generator = Generator(
        input_shape, dimH, dimZ, dimY, n_channel, last_activation="sigmoid"
    )

    # Load Custom CIFAR-10 Dataset
    cifar_dataset = CustomCIFAR10Dataset(
        path="./data", train=True, labels=[0, 1], conv=True
    )
    cifar_loader = DataLoader(cifar_dataset, batch_size=64, shuffle=True)

    # Test generator with CIFAR-10 data
    for images, labels in cifar_loader:
        labels_one_hot = F.one_hot(labels, num_classes=dimY).float()
        mu, log_sig = generator.pzy_params(labels_one_hot)
        z = sample_gaussian(mu, log_sig)
        pxz_output = generator.pxz_params(z)

        print(f"Latent vector z shape: {z.shape}")
        print(f"p(z|y) mu shape: {mu.shape}, log_sigma shape: {log_sig.shape}")
        print(f"p(x|z) output shape: {pxz_output.shape}")
        break
