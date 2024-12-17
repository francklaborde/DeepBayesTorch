import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm

import argparse


def comp_detect_rate(dict_result, detection_method):
    """
    Compute the detection rate of a specific detection method.

    Parameters
    ----------
    dict_result : dict
        Dictionary containing the detection results.
    detection_method : str
        Name of the detection method.

    Returns
    -------
    list
        Detection rates of the detection method for different epsilons.
    """

    if detection_method == "Marginal":
        return dict_result["tp_logpx"]
    elif detection_method == "Logit":
        return dict_result["TP_logpxy"]
    elif detection_method == "KL":
        return dict_result["TP_kl"]
    else:
        raise ValueError("Invalid detection method.")
    
def plot_detection_rate(data_name, attack, epsilons, data_dir = "./detection_results", save_dir = "./detection_results"):
    """
    Plot the detection rate of one attack on all the VAE for different values of epsilon.
    
    Parameters
    ----------
    data_name : str
        Name of the dataset.
    attack : str
        Name of the attack.
    save_dir : str
        Directory to save the plot.
    """
    
    vae_types = ["A", "B", "C", "D", "E", "F", "G"]
    detection_methods = ["Marginal", "Logit", "KL"]
    detection_rates = {detect_method: {} for detect_method in detection_methods}

    for detect_method in detection_methods:
        for vae_type in vae_types:
            with open(f"{data_dir}/{data_name}_{vae_type}_{attack}_detection_results.json", "r") as f:
                detection_rate = json.load(f)
            
            detection_rates[detect_method][vae_type] = comp_detect_rate(detection_rate, detect_method)

    # Plotting
    # Create subplots for each detection method
    letter_to_title = {
        "A": "GFZ",
        "B": "GFY",
        "C": "GBZ",
        "D": "GBY",
        "E": "DFX",
        "F": "DFZ",
        "G": "DBX",
    }
    num_vae_types = len(vae_types)
    cmap = cm.get_cmap("rainbow", num_vae_types)

    fig, axes = plt.subplots(1, len(detection_methods), figsize=(12, 4))
    for idx, detect_method in enumerate(detection_methods):
        ax = axes[idx]
        for j, vae_type in enumerate(vae_types):
            if vae_type in detection_rates[detect_method]:
                if len(epsilons) != len(detection_rates[detect_method][vae_type]):
                    raise ValueError("Length of epsilons and detection rates do not match : epsilons is length %i and dectection_rate is len %i."%(len(epsilons), len(detection_rates[detect_method][vae_type])))
                detection_rates[detect_method][vae_type] = np.array(detection_rates[detect_method][vae_type])
                ax.plot(
                    epsilons,
                    detection_rates[detect_method][vae_type]/100,
                    marker="o",
                    label=letter_to_title[vae_type],
                    linewidth=2,
                    color=cmap(j))
        
        ax.set_title(f"{attack}, TP {detect_method} detection")
        ax.set_xlabel("Epsilon")
        ax.grid(linestyle="--")
        #ax.legend()
        

    plt.tight_layout()
    save_path = f"{save_dir}/{data_name}_{attack}_detection_rates_subplots.png"
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Subplots saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the detection rate of one attack on all the VAE for different values of epsilon.")
    parser.add_argument("--data_name",
                        type=str,
                        default="mnist",
                        help="Name of the dataset.")
    parser.add_argument("--attack",
                        type=str,
                        default="FGSM",
                        help="Name of the attack.")
    parser.add_argument("--data_dir",
                        type=str,
                        default="./detection_results",
                        help="Directory containing the detection results.")
    parser.add_argument("--save_dir",
                        type=str,
                        default="./detection_results",
                        help="Directory to save the plot.")
    parser.add_argument("--epsilons",
                        type=float,
                        nargs="+",
                        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help="List of epsilon values for the attacks.")

    args = parser.parse_args()
    plot_detection_rate(data_name=args.data_name,
                        attack=args.attack,
                        epsilons=args.epsilons,
                        data_dir=args.data_dir,
                        save_dir=args.save_dir)