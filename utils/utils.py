import os

import torch


def load_data(data_name, path, labels=None, conv=False, seed=0):
    """
    Loads dataset based on its name.
    Args:
        data_name (str): Name of the dataset ('mnist', 'omni', 'cifar10').
        path (str): Path to the dataset directory.
        labels (list, optional): List of labels to filter.
        conv (bool): Whether to retain 4D shape for convolutional models.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: data_train, data_test, labels_train, labels_test
    """
    if data_name == "mnist":
        from .mnist import CustomMNISTDataset

        train_dataset = CustomMNISTDataset(
            path=path, train=True, digits=labels, conv=conv
        )
        test_dataset = CustomMNISTDataset(
            path=path, train=False, digits=labels, conv=conv
        )

    elif data_name == "cifar10":
        from .cifar10 import CustomCIFAR10Dataset

        train_dataset = CustomCIFAR10Dataset(
            path=path, train=True, labels=labels, conv=conv, seed=seed
        )
        test_dataset = CustomCIFAR10Dataset(
            path=path, train=False, labels=labels, conv=conv, seed=seed
        )
    elif data_name == "gtsrb":
        from .gtsrb import GermanTrafficSignDataset

        train_dataset = GermanTrafficSignDataset(
            root_dir=path, train=True, labels=labels
        )
        test_dataset = GermanTrafficSignDataset(
            root_dir=path, train=False, labels=labels
        )
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")

    return train_dataset, test_dataset


def save_params(model, filename, checkpoint):
    """
    Save model parameters to a file.
    Args:
        model (torch.nn.Module): PyTorch model.
        filename (str): Path to save the parameters.
        checkpoint (int): Checkpoint index for versioning.
    """
    encoder, generator = model
    filename = f"{filename}_{checkpoint}.pth"
    state_dict = {"encoder": encoder.state_dict(), "generator": generator.state_dict()}
    torch.save(state_dict, filename)
    print(f"Parameters saved at {filename}")


def load_params(model, filename, checkpoint):
    """
    Load model parameters from a file.
    Args:
        model (torch.nn.Module): PyTorch model.
        filename (str): Path to load the parameters from.
        checkpoint (int): Checkpoint index for versioning.
    """
    filename = f"{filename}_{checkpoint}.pth"
    encoder, generator = model
    if os.path.exists(filename):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(filename, map_location=device)
        encoder.load_state_dict(state_dict["encoder"])
        generator.load_state_dict(state_dict["generator"])
        print(f"Loaded parameters from {filename}")
    else:
        print(f"Checkpoint {filename} not found. Skipping parameter loading.")


def init_variables(model, optimizer=None):
    """
    Initialize model parameters and optionally an optimizer.
    Args:
        model (torch.nn.Module): PyTorch model.
        optimizer (torch.optim.Optimizer, optional): Optimizer to reset.
    """
    model.apply(reset_weights)
    if optimizer:
        optimizer.state = {}


def reset_weights(layer):
    """
    Resets weights of a layer.
    """
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()

def load_model(data_name, vae_type, checkpoint_index, device=None):
    """
    Load a trained model with given parameters.

    Args:
        data_name (str): Dataset name (e.g., 'mnist').
        vae_type (str): Model type ('A', 'B', ..., 'G').
        checkpoint_index (int): Index of the checkpoint to load.
        dimZ (int, optional): Latent dimension. Default is 64.
        dimH (int, optional): Hidden layer size. Default is 500.
        device (torch.device, optional): Device to load the model onto.

    Returns:
        encoder, generator: The loaded encoder and generator models.
    """
    if data_name == "mnist":
        if vae_type == "A":
            from models.conv_generator_mnist_A import Generator
        elif vae_type == "B":
            from models.conv_generator_mnist_B import Generator
        elif vae_type == "C":
            from models.conv_generator_mnist_C import Generator
        elif vae_type == "D":
            from models.conv_generator_mnist_D import Generator
        elif vae_type == "E":
            from models.conv_generator_mnist_E import Generator
        elif vae_type == "F":
            from models.conv_generator_mnist_F import Generator
        elif vae_type == "G":
            from models.conv_generator_mnist_G import Generator
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")
        from models.conv_encoder_mnist import GaussianConvEncoder as Encoder

        input_shape = (1, 28, 28)
        n_channel = 64
        dimZ = 64
        dimH = 500
        dimY = 10
    elif data_name == "cifar10" or data_name == "gtsrb":
        if vae_type == "A":
            from models.conv_generator_cifar10_A import Generator
        elif vae_type == "B":
            from models.conv_generator_cifar10_B import Generator
        elif vae_type == "C":
            from models.conv_generator_cifar10_C import Generator
        elif vae_type == "D":
            from models.conv_generator_cifar10_D import Generator
        elif vae_type == "E":
            from models.conv_generator_cifar10_E import Generator
        elif vae_type == "F":
            from models.conv_generator_cifar10_F import Generator
        elif vae_type == "G":
            from models.conv_generator_cifar10_G import Generator
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")
        from models.conv_encoder_cifar10 import GaussianConvEncoder as Encoder

        input_shape = (3, 32, 32)
        n_channel = 128
        dimZ = 128
        dimH = 1000
        dimY = 10
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    if data_name == "gtsrb":
        dimY = 43

    generator = Generator(input_shape, dimH, dimZ, dimY, n_channel, "sigmoid", "gen")
    encoder = Encoder(input_shape, dimH, dimZ, dimY, n_channel, "enc")

    path_name = f"{data_name}_conv_vae_{vae_type}_{dimZ}/"
    filename = f"save/{path_name}checkpoint"

    load_params((encoder, generator), filename, checkpoint_index)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    generator = generator.to(device)

    return encoder, generator, dimY
