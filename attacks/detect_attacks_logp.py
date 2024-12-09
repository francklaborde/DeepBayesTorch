import argparse
import os 
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alg.vae_new import bayes_classifier, construct_optimizer
from utils.utils import load_model, load_data

# Utility: Log probabilities
def compute_log_probabilities(logits, y, text, comp_logit_dist = False):
    """
    Computes various log-probability statistics for given itorchuts.
    
    Args:
        logits: output of the classifier.
        y: True labels for itorchuts.

    Returns:
        logp_x: Log probabilities for all classes.
        logp_x_given_y: Log probability of the true class for each itorchut.
        probs: Raw probabilities for all classes.
        max_logp: Maximum log-probability for each itorchut (confidence measure).
        pred_class: Predicted class for each itorchut.
    """
    logp_x = F.log_softmax(logits, dim=-1)
    logpx_mean = torch.mean(logp_x)
    logpx_std = torch.sqrt(torch.var(logp_x))
    logpxy = torch.sum(y * logits, axis=1)
    logpxy_mean = []; logpxy_std = []
    for i in range(y.shape[1]):
        ind = torch.where(y[:, i] == 1)[0]
        logpxy_mean.append(torch.mean(logpxy[ind]))
        logpxy_std.append(torch.sqrt(torch.var(logpxy[ind])))

    print('%s: logp(x) = %.3f +- %.3f, logp(x|y) = %.3f +- %.3f' \
          % (text, logpx_mean, logpx_std, torch.mean(logpxy_mean), torch.mean(logpxy_std)))
    
    results = [logp_x, logpx_mean, logpx_std, logpxy, logpxy_mean, logpxy_std]

    # compute distribution of the logits
    if comp_logit_dist:
        logit_mean = []
        logit_std = []
        logit_kl_mean = []
        logit_kl_std = []
        softmax_mean = []
        for i in xrange(y.shape[1]):
            ind = torch.where(y[:, i] == 1)[0]
            logit_mean.append(torch.mean(logits[ind], 0))
            logit_std.append(torch.sqrt(torch.var(logits[ind], 0)))

            logit_tmp = logits[ind] - logsumexp(logits[ind], axis=1)[:, torch.newaxis]
            softmax_mean.append(torch.mean(torch.exp(logit_tmp), 0))
            logit_kl = torch.sum(softmax_mean[i] * (torch.log(softmax_mean[i]) - logit_tmp), 1)
            
            logit_kl_mean.append(torch.mean(logit_kl))
            logit_kl_std.append(torch.sqrt(torch.var(logit_kl)))
        
        results.extend([logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean]) 

    return results

def comp_detect(x, x_mean, x_std, alpha, plus):
    if plus:
        detect_rate = torch.mean(x > x_mean + alpha * x_std)
    else:
        detect_rate = torch.mean(x < x_mean - alpha * x_std)
    return detect_rate * 100

def search_alpha(x, x_mean, x_std, target_rate = 5.0, plus = False):
    alpha_min = 0.0
    alpha_max = 3.0
    alpha_now = 1.5
    detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
    T = 0
    while torch.abs(detect_rate - target_rate) > 0.01 and T < 20:
        if detect_rate > target_rate:
            alpha_min = alpha_now
        else:
            alpha_max = alpha_now
        alpha_now = 0.5 * (alpha_min + alpha_max)
        detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
        T += 1
    return alpha_now, detect_rate

### Function du chat
# # Utility: KL divergence
# def compute_kl_divergence(p, q):
#     return F.kl_div(p.log(), q, reduction='batchmean')

# # Utility: Entropy
# def compute_entropy(log_probs):
#     probs = log_probs.exp()
#     entropy = -torch.sum(probs * log_probs, dim=-1)
#     return entropy

# # Utility: Perturbation metrics
# def compute_perturbation_metrics(x, x_adv):
#     l2_norm = torch.norm(x - x_adv, p=2, dim=(1, 2, 3)).mean().item()
#     linf_norm = torch.norm(x - x_adv, p=float('inf'), dim=(1, 2, 3)).mean().item()
#     return l2_norm, linf_norm

def test_attacks(data_name, batch_size, epsilons, targeted, attack_method, victim_name, save, save_dir="./results/"):
 
    vae_types = ["A", "B", "C", "D", "E", "F", "G"]
    attack_methods = ["FGSM", "PGD", "MIM"]

    detection_rate = {vae_type: {} for vae_type in vae_types}
    
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

        os.makedirs(save_dir, exist_ok=True)
        detection_rate[vae_type] = {attack: [] for attack in attack_methods}

        for epsilon in epsilons:
            detected = 0
            total = 0

            for images, labels in tqdm(
                test_loader, desc=f"{attack_method}, epsilon={epsilon}"
            ):
                images, labels = images.to(
                    next(encoder.parameters()).device
                ), labels.to(next(encoder.parameters()).device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RVAE experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--data', '-D', type=str, default='mnist')
    parser.add_argument('--conv', '-C', action='store_true', default=False)
    parser.add_argument('--guard', '-G', type=str, default='bayes_K10')
    parser.add_argument('--targeted', '-T', action='store_true', default=False)
    parser.add_argument('--attack', '-A', type=str, default='fgsm_eps0.10')
    parser.add_argument('--victim', '-V', type=str, default='mlp')
    parser.add_argument('--save', '-S', action='store_true', default=False)

    args = parser.parse_args()
    test_attacks(args.batch_size, args.conv, args.guard, args.targeted, 
                 args.attack, args.victim, args.data, args.save)

