import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from matplotlib import cm
from tqdm import tqdm

from alg.vae_new import bayes_classifier, construct_optimizer
from attacks.momentum_iterative_method import momentum_iterative_method
from utils.utils import load_data, load_params


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

    return encoder, generator


def perform_attacks(
    data_name, epsilons, batch_size, save_result="./results/", device=None
):
    """
    Perform FGSM attack on a given model, evaluate accuracy vs. epsilon, and save results.

    Args:
        data_name (str): Dataset name ('mnist').
        vae_type (str): Model type ('A', 'B', ..., 'G').
        checkpoint_index (int): Index of the checkpoint to load.
        epsilons (list): List of epsilon values for FGSM attack.
        save_dir (str): Directory to save results.
    """
    vae_types = ["A", "B", "C", "D", "E", "F", "G"]

    attack_methods = ["FGSM", "PGD", "MIM"]
    accuracies = {vae_type: {} for vae_type in vae_types}
    for vae_type in vae_types:
        encoder, generator = load_model(data_name, vae_type, 0)
        encoder.eval()
        input_shape = (1, 28, 28) if data_name == "mnist" else (3, 32, 32)
        dimY = 10 if data_name != "gtsrb" else 43
        ll = "l2"
        K = 10

        if vae_type == "A":
            dec = (generator.pyz_params, generator.pxzy_params)
            from alg.lowerbound_functions import lowerbound_A as lowerbound
        elif vae_type == "B":
            dec = (generator.pzy_params, generator.pxzy_params)
            from alg.lowerbound_functions import lowerbound_B as lowerbound
        elif vae_type == "C":
            dec = (generator.pyzx_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_C as lowerbound
        elif vae_type == "D":
            dec = (generator.pyzx_params, generator.pzx_params)
            from alg.lowerbound_functions import lowerbound_D as lowerbound
        elif vae_type == "E":
            dec = (generator.pyz_params, generator.pzx_params)
            from alg.lowerbound_functions import lowerbound_E as lowerbound
        elif vae_type == "F":
            dec = (generator.pyz_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_F as lowerbound
        elif vae_type == "G":
            dec = (generator.pzy_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_G as lowerbound
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")

        enc_conv = encoder.encoder_conv
        enc_mlp = encoder.enc_mlp
        enc = (enc_conv, enc_mlp)
        X_ph = torch.zeros(1, *input_shape).to(next(encoder.parameters()).device)
        Y_ph = torch.zeros(1, dimY).to(next(encoder.parameters()).device)
        _, eval_fn = construct_optimizer(X_ph, Y_ph, enc, dec, ll, K, vae_type)

        _, test_dataset = load_data(data_name, path="./data", labels=None, conv=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        os.makedirs(save_result, exist_ok=True)
        accuracies[vae_type] = {attack: [] for attack in attack_methods}

        for epsilon in epsilons:
            for attack in attack_methods:
                correct = 0
                total = 0

                for images, labels in tqdm(
                    test_loader, desc=f"{attack}, epsilon={epsilon}"
                ):
                    images, labels = images.to(
                        next(encoder.parameters()).device
                    ), labels.to(next(encoder.parameters()).device)

                    if attack == "FGSM":
                        adv_images = fast_gradient_method(
                            enc_conv,
                            images,
                            eps=epsilon,
                            norm=np.inf,
                            clip_min=0.0,
                            clip_max=1.0,
                            sanity_checks=False,
                        )
                    elif attack == "PGD":
                        adv_images = projected_gradient_descent(
                            enc_conv,
                            images,
                            eps=epsilon,
                            eps_iter=0.01,
                            nb_iter=100,
                            clip_min=0.0,
                            clip_max=1.0,
                            rand_init=True,
                            norm=np.inf,
                            sanity_checks=False,
                        )
                    elif attack == "MIM":
                        adv_images = momentum_iterative_method(
                            enc_conv,
                            images,
                            eps=epsilon,
                            eps_iter=0.01,
                            nb_iter=100,
                            decay_factor=1.0,
                            clip_min=0.0,
                            clip_max=1.0,
                            norm=np.inf,
                            sanity_checks=False,
                        )
                    else:
                        raise ValueError(f"Unsupported attack: {attack}")

                    y_pred = bayes_classifier(
                        adv_images,
                        (enc_conv, enc_mlp),
                        dec,
                        ll,
                        dimY,
                        lowerbound=lowerbound,
                        K=10,
                        beta=1.0,
                    )
                    correct += (torch.argmax(y_pred, dim=1) == labels).sum().item()
                    total += labels.size(0)

                accuracies[vae_type][attack].append(correct / total)

    with open(
        os.path.join(save_result, f"{data_name}_accuracy_vs_epsilon.json"), "w"
    ) as f:
        json.dump(accuracies, f)

    return accuracies

def perform_attacks_v2(
    data_name, epsilons, batch_size, save_result="./results/", save_raw="./raw_result_attack", device=None
):
    """
    Perform adversarial attacks on a given model, save generated adversarial examples, and evaluate accuracy.
    """
    vae_types = ["A", "B", "C", "D", "E", "F", "G"]
    attack_methods = ["FGSM", "PGD", "MIM"]
    accuracies = {vae_type: {} for vae_type in vae_types}

    os.makedirs(save_raw, exist_ok=True)  # Ensure the raw result directory exists

    for vae_type in vae_types:
        encoder, generator = load_model(data_name, vae_type, 0)
        encoder.eval()
        input_shape = (1, 28, 28) if data_name == "mnist" else (3, 32, 32)
        dimY = 10 if data_name != "gtsrb" else 43
        ll = "l2"
        K = 10

        if vae_type == "A":
            dec = (generator.pyz_params, generator.pxzy_params)
            from alg.lowerbound_functions import lowerbound_A as lowerbound
        elif vae_type == "B":
            dec = (generator.pzy_params, generator.pxzy_params)
            from alg.lowerbound_functions import lowerbound_B as lowerbound
        elif vae_type == "C":
            dec = (generator.pyzx_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_C as lowerbound
        elif vae_type == "D":
            dec = (generator.pyzx_params, generator.pzx_params)
            from alg.lowerbound_functions import lowerbound_D as lowerbound
        elif vae_type == "E":
            dec = (generator.pyz_params, generator.pzx_params)
            from alg.lowerbound_functions import lowerbound_E as lowerbound
        elif vae_type == "F":
            dec = (generator.pyz_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_F as lowerbound
        elif vae_type == "G":
            dec = (generator.pzy_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_G as lowerbound
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")

        enc_conv = encoder.encoder_conv
        enc_mlp = encoder.enc_mlp
        enc = (enc_conv, enc_mlp)
        # _, eval_fn = construct_optimizer(X_ph, Y_ph, enc, dec, ll, K, vae_type)

        _, test_dataset = load_data(data_name, path="./data", labels=None, conv=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        accuracies[vae_type] = {attack: [] for attack in attack_methods}

        for epsilon in epsilons:
            for attack in attack_methods:
                x_adv = []
                y_adv = []
                x_clean = []
                y_clean = []
                correct = 0
                total = 0

                for images, labels in tqdm(
                    test_loader, desc=f"{attack}, epsilon={epsilon}"
                ):
                    images, labels = images.to(device), labels.to(device)

                    if attack == "FGSM":
                        adv_images = fast_gradient_method(
                            enc_conv,
                            images,
                            eps=epsilon,
                            norm=np.inf,
                            clip_min=0.0,
                            clip_max=1.0,
                            sanity_checks=False,
                        )
                    elif attack == "PGD":
                        adv_images = projected_gradient_descent(
                            enc_conv,
                            images,
                            eps=epsilon,
                            eps_iter=0.01,
                            nb_iter=100,
                            clip_min=0.0,
                            clip_max=1.0,
                            rand_init=True,
                            norm=np.inf,
                            sanity_checks=False,
                        )
                    elif attack == "MIM":
                        adv_images = momentum_iterative_method(
                            enc_conv,
                            images,
                            eps=epsilon,
                            eps_iter=0.01,
                            nb_iter=100,
                            decay_factor=1.0,
                            clip_min=0.0,
                            clip_max=1.0,
                            norm=np.inf,
                            sanity_checks=False,
                        )
                    else:
                        raise ValueError(f"Unsupported attack: {attack}")

                    y_pred = bayes_classifier(
                        adv_images,
                        (enc_conv, enc_mlp),
                        dec,
                        ll,
                        dimY,
                        lowerbound=lowerbound,
                        K=10,
                        beta=1.0,
                    )
                    correct += (torch.argmax(y_pred, dim=1) == labels).sum().item()
                    total += labels.size(0)

                    # Save batch data
                    x_adv.append(adv_images.cpu().numpy())
                    y_adv.append(y_pred.cpu().numpy())
                    x_clean.append(images.cpu().numpy())
                    y_clean.append(labels.cpu().numpy())

                # Concatenate and save results
                x_adv = np.concatenate(x_adv, axis=0)
                y_adv = np.concatenate(y_adv, axis=0)
                x_clean = np.concatenate(x_clean, axis=0)
                y_clean = np.concatenate(y_clean, axis=0)

                save_path = os.path.join(
                    save_raw, f"{vae_type}_{attack}_epsilon_{epsilon}.npz"
                )
                np.savez(
                    save_path,
                    x_adv=x_adv,
                    y_adv=y_adv,
                    x_clean=x_clean,
                    y_clean=y_clean,
                )
                print(f"Saved adversarial examples to {save_path}")

                accuracies[vae_type][attack].append(correct / total)

    with open(
        os.path.join(save_result, f"{data_name}_accuracy_vs_epsilon.json"), "w"
    ) as f:
        json.dump(accuracies, f)

    return accuracies



def plot_results(json_file, save_dir, data_name, epsilons):
    with open(json_file, "r") as f:
        accuracies = json.load(f)
    vae_types = list(accuracies.keys())
    attack_methods = list(accuracies[vae_types[0]].keys())
    letter_to_title = {
        "A": "GFZ",
        "B": "GFY",
        "C": "GBZ",
        "D": "GBY",
        "E": "DFX",
        "F": "DFZ",
        "G": "DBX",
    }
    fig, axes = plt.subplots(len(attack_methods), 1, figsize=(4, 12))
    num_vae_types = len(vae_types)
    cmap = cm.get_cmap("rainbow", num_vae_types)
    for i, attack in enumerate(attack_methods):
        for j, vae_type in enumerate(vae_types):
            color = cmap(j)
            axes[i].plot(
                epsilons,
                accuracies[vae_type][attack],
                marker="o",
                label=letter_to_title[vae_type],
                linewidth=2,
                color=color,
            )
        axes[i].set_title(f"{attack} victim acc")
        axes[i].set_xlabel("Epsilon")
        axes[i].grid(linestyle="--")
        axes[i].legend()

    # Save the plot
    plt.tight_layout()
    filename = f"{data_name}__accuracy_vs_epsilon_combined.png"
    plt.savefig(
        os.path.join(
            save_dir,
            filename,
        )
    )
    plt.close()
    print(
        "Saved combined accuracy plot for all attacks to",
        os.path.join(save_dir, filename),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform FGSM attack on a trained VAE."
    )

    parser.add_argument(
        "--data_name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "gtsrb"],
        help="Dataset name (e.g., 'mnist').",
    )

    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="List of epsilon values for FGSM attack.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for evaluating the model.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./results/", help="Directory to save results."
    )
    parser.add_argument("--compute", action="store_true", help="Compute the results.")
    parser.add_argument(
        "--plot", action="store_true", help="Plot the results from the JSON file."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="JSON file containing the results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to load the model onto (e.g., 'cuda:0').",
    )

    args = parser.parse_args()
    if args.compute:
        results = perform_attacks_v2(
            data_name=args.data_name,
            epsilons=args.epsilons,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            device=args.device,
        )
    if args.plot:
        if args.json_file is None:
            print("No JSON file specified.")
            args.json_file = os.path.join(
                args.save_dir, f"{args.data_name}_accuracy_vs_epsilon.json"
            )
        plot_results(args.json_file, args.save_dir, args.data_name, args.epsilons)
