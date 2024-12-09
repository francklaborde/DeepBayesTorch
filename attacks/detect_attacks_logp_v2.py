import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def comp_logp(logit, y, text, comp_logit_dist=False):
    """
    Compute log probabilities and related statistics.

    Args:
        logit (ndarray): logits of shape (N, nb_classes).
        y (ndarray): one-hot labels of shape (N, nb_classes).
        text (str): Tag for printing results.
        comp_logit_dist (bool): Whether to compute distribution of logits.

    Returns:
        List of computed statistics.
    """
    # logsumexp over classes
    logpx = np.logsumexp(logit, axis=1)
    logpx_mean = np.mean(logpx)
    logpx_std = np.sqrt(np.var(logpx))
    
    logpxy = np.sum(y * logit, axis=1)
    logpxy_mean = []
    logpxy_std = []

    for i in range(y.shape[1]):
        ind = np.where(y[:, i] == 1)[0]
        if len(ind) > 0:
            logpxy_mean.append(np.mean(logpxy[ind]))
            logpxy_std.append(np.sqrt(np.var(logpxy[ind])))
        else:
            # If no samples for this class, just append NaNs
            logpxy_mean.append(float('nan'))
            logpxy_std.append(float('nan'))

    print('%s: logp(x) = %.3f +- %.3f, logp(x|y) = %.3f +- %.3f' %
          (text, logpx_mean, logpx_std, np.nanmean(logpxy_mean), np.nanmean(logpxy_std)))

    results = [logpx, logpx_mean, logpx_std, logpxy, logpxy_mean, logpxy_std]

    if comp_logit_dist:
        # Compute distribution of logits
        nb_classes = y.shape[1]
        logit_mean = []
        logit_std = []
        logit_kl_mean = []
        logit_kl_std = []
        softmax_mean_list = []
        # softmax of mean distribution
        for i in range(nb_classes):
            ind = np.where(y[:, i] == 1)[0]
            if len(ind) > 0:
                logit_class = logit[ind]
                logit_mean.append(np.mean(logit_class, axis=0))
                logit_std.append(np.sqrt(np.var(logit_class, axis=0)))

                # Compute softmax and KL divergence
                logit_tmp = logit_class - np.logsumexp(logit_class, axis=1)[:, np.newaxis]
                softmax_mean = np.mean(np.exp(logit_tmp), axis=0)
                softmax_mean_list.append(softmax_mean)

                # KL divergence from softmax_mean distribution to each sample's distribution
                # KL(Pmean || Pi) = sum(Pmean * (log(Pmean) - log(Pi)))
                # where Pi is from each sample logit_tmp
                # Actually, we want the mean KL over samples:
                # logit_tmp are log probabilities of each sample.
                # We can approximate KL by using the mean softmax distribution and comparing to each sample.
                # However, in original code, it seems a different approach was taken.
                # We'll follow the original logic closely.
                # The original code snippet seems incorrect in computing KL directly for each sample distribution.
                # Instead, it computed KL based on softmax_mean. We'll replicate that logic:
                # logit_kl = sum(Pmean * (log(Pmean)-log(pi))) over i, averaged over samples.
                # Actually, in original code, it seems to incorrectly apply logit_kl on each sample. 
                # We'll just skip this detail and keep the original approach.
                
                # We'll compute the KL divergence per sample and then average:
                # Pi = exp(logit_tmp)
                # KL(Pmean || Pi) = sum(Pmean * (log(Pmean) - log(Pi)))
                # But Pi differs per sample. We'll compute it for each sample:
                pi = np.exp(logit_tmp)
                # For each sample:
                # kl_i = sum(Pmean * (log(Pmean)-log(pi_i)))
                kl_vals = []
                for sample_idx in range(pi.shape[0]):
                    kl_val = np.sum(softmax_mean * (np.log(softmax_mean) - logit_tmp[sample_idx]))
                    kl_vals.append(kl_val)
                kl_vals = np.array(kl_vals)
                logit_kl_mean.append(np.mean(kl_vals))
                logit_kl_std.append(np.sqrt(np.var(kl_vals)))
            else:
                # If no samples for that class
                logit_mean.append(np.full((nb_classes,), np.nan))
                logit_std.append(np.full((nb_classes,), np.nan))
                logit_kl_mean.append(np.nan)
                logit_kl_std.append(np.nan)
                softmax_mean_list.append(np.full((nb_classes,), np.nan))

        results.extend([logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean_list])

    return results

def comp_detect(x, x_mean, x_std, alpha, plus):
    """
    Compute detection rate given a criterion:
    If plus=True: detect if x > x_mean + alpha * x_std
    else: detect if x < x_mean - alpha * x_std
    """
    if plus:
        detect_rate = np.mean(x > x_mean + alpha * x_std)
    else:
        detect_rate = np.mean(x < x_mean - alpha * x_std)
    return detect_rate * 100

def search_alpha(x, x_mean, x_std, target_rate=5.0, plus=False):
    """
    Binary search for alpha such that detection rate is close to target_rate.
    """
    alpha_min = 0.0
    alpha_max = 3.0
    alpha_now = 1.5
    detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
    T = 0
    while np.abs(detect_rate - target_rate) > 0.01 and T < 20:
        if detect_rate > target_rate:
            alpha_min = alpha_now
        else:
            alpha_max = alpha_now
        alpha_now = 0.5 * (alpha_min + alpha_max)
        detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
        T += 1
    return alpha_now, detect_rate

def test_attacks(
    model: nn.Module, 
    x_train, y_train, 
    x_clean, y_clean, 
    x_adv, y_adv,
    epsilons: list,
    nb_classes: int,
    save=False, 
    guard_name='bayes_model',
    victim_name='mlp',
    data_name='mnist',
    all_attack=True,
    targeted=False
):
    """
    Evaluate detection metrics on clean and adversarial examples.

    Args:
        model (nn.Module): PyTorch model for evaluation.
        x_train, y_train: Training samples (for baseline stats).
        x_clean, y_clean: Clean test samples and labels.
        x_adv, y_adv: Adversarial samples and corresponding predicted labels from victim model.
        nb_classes (int): Number of classes.
        save (bool): Whether to save results to disk.
        guard_name (str): Identifier for the "guard" model.
        victim_name (str): Identifier for the victim model.
        data_name (str): Dataset name.
        targeted (bool): Whether the attack is targeted.

    Returns:
        results (dict): Dictionary of detection statistics.
    """
    # Convert everything to numpy for computations
    # y_* should be numpy arrays of shape (N, nb_classes)
    # x_* are (N, C, H, W) and can be converted to torch for predictions.
    # We'll assume they are already numpy arrays.
    attack_methods = ["FGSM", "PGD", "MIM"]

    if all_attack:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Function to get logits from model in batches
    def get_logits(model, x, batch_size=100):
        logits_list = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                X_batch = torch.from_numpy(x[i:i+batch_size]).float().to(device)
                # model should output logits directly or we can handle that logic here.
                # Assuming model(x) gives logits.
                logit = model(X_batch)
                # move to cpu, convert to numpy
                logits_list.append(logit.cpu().numpy())
        return np.concatenate(logits_list, axis=0)

    # Compute logits for clean and adv data under the guard model
    y_logit_clean = get_logits(model, x_clean)
    y_logit_adv = get_logits(model, x_adv)

    # Identify successful attacks (where victim's adv prediction != clean label)
    correct_prediction = (np.argmax(y_adv, 1) == np.argmax(y_clean, 1))
    success_rate = 100.0 * (1 - np.mean(correct_prediction))
    ind_success = np.where(correct_prediction == 0)[0]

    if len(ind_success) == 0:
        print('No successful attacks found.')
        return {}

    # Compute L2, L0, L_inf perturbations for successful attacks
    diff = x_adv[ind_success] - x_clean[ind_success]
    l2_diff = np.sqrt(np.sum(diff**2, axis=(1, 2, 3)))
    li_diff = np.max(np.abs(diff), axis=(1, 2, 3))
    l0_diff = np.sum((diff != 0), axis=(1, 2, 3))
    print('perturb for successful attack: L_2 = %.3f +- %.3f' % (np.mean(l2_diff), np.sqrt(np.var(l2_diff))))
    print('perturb for successful attack: L_inf = %.3f +- %.3f' % (np.mean(li_diff), np.sqrt(np.var(li_diff))))
    print('perturb for successful attack: L_0 = %.3f +- %.3f' % (np.mean(l0_diff), np.sqrt(np.var(l0_diff))))

    # Compute log probabilities
    # Also get training logits for baseline stats
    y_logit_train = get_logits(model, x_train)
    results_train = comp_logp(y_logit_train, y_train, 'train', comp_logit_dist=True)
    results_clean = comp_logp(y_logit_clean, y_clean, 'clean')
    results_adv = comp_logp(y_logit_adv[ind_success], y_adv[ind_success], 'adv (wrong)')

    # Detection based on logp(x)
    # If guard_name in ['mlp', 'cnn'], plus=True else False (following original logic)
    plus = True if guard_name in ['mlp', 'cnn'] else False
    alpha, detect_rate = search_alpha(results_train[0], results_train[1], results_train[2], plus=plus)
    fp_logpx = comp_detect(results_train[0], results_train[1], results_train[2], alpha, plus=plus)
    tp_logpx = comp_detect(results_adv[0], results_train[1], results_train[2], alpha, plus=plus)
    print('false alarm rate (logp(x)):', fp_logpx)
    print('detection rate (logp(x)):', tp_logpx)

    # Detection based on logp(x|y)
    fp_rate = []
    tp_rate_vals = []
    for i in range(nb_classes):
        ind = np.where(y_train[:, i] == 1)[0]
        if len(ind) == 0:
            continue
        alpha_c, _ = search_alpha(results_train[3][ind], results_train[4][i], results_train[5][i], plus=plus)
        fp_c = comp_detect(results_train[3][ind], results_train[4][i], results_train[5][i], alpha_c, plus=plus)
        fp_rate.append(fp_c)

        adv_ind = np.where(y_adv[ind_success][:, i] == 1)[0]
        if len(adv_ind) == 0:
            continue
        tp_c = comp_detect(results_adv[3][adv_ind], results_train[4][i], results_train[5][i], alpha_c, plus=plus)
        tp_rate_vals.append(tp_c)
    if len(tp_rate_vals) > 0:
        FP_logpxy = np.mean(fp_rate)
        TP_logpxy = np.mean(tp_rate_vals)
    else:
        FP_logpxy = np.nan
        TP_logpxy = np.nan
    print('false alarm rate (logp(x|y)):', FP_logpxy)
    print('detection rate (logp(x|y)):', TP_logpxy)

    # KL-based detection
    # Extract the logit distribution stats from training
    # last 5 results from results_train are [logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean]
    logit_mean, logit_std, kl_mean, kl_std, softmax_mean_list = results_train[-5:]

    # We need to compute KL on train and adv again per class
    fp_rate_kl = []
    tp_rate_kl = []
    for i in range(nb_classes):
        # compute KL for the training samples of class i
        ind = np.where(y_train[:, i] == 1)[0]
        if len(ind) == 0 or np.isnan(kl_mean[i]):
            continue
        # compute KL wrt. softmax_mean_list[i]
        logit_train_i = y_logit_train[ind]
        logit_tmp = logit_train_i - np.logsumexp(logit_train_i, axis=1)[:, np.newaxis]
        pi = np.exp(logit_tmp)
        pmean = softmax_mean_list[i]
        kl_values_train = []
        for j in range(pi.shape[0]):
            kl_val = np.sum(pmean * (np.log(pmean) - logit_tmp[j]))
            kl_values_train.append(kl_val)
        kl_values_train = np.array(kl_values_train)
        alpha_c, _ = search_alpha(kl_values_train, kl_mean[i], kl_std[i], plus=True)
        fp_c = comp_detect(kl_values_train, kl_mean[i], kl_std[i], alpha_c, plus=True)
        fp_rate_kl.append(fp_c)

        # adv
        adv_ind = np.where(y_adv[ind_success][:, i] == 1)[0]
        if len(adv_ind) == 0:
            continue
        logit_adv_i = y_logit_adv[ind_success][adv_ind]
        logit_tmp_adv = logit_adv_i - np.logsumexp(logit_adv_i, axis=1)[:, np.newaxis]
        pi_adv = np.exp(logit_tmp_adv)
        kl_values_adv = []
        for j in range(pi_adv.shape[0]):
            kl_val = np.sum(pmean * (np.log(pmean) - logit_tmp_adv[j]))
            kl_values_adv.append(kl_val)
        kl_values_adv = np.array(kl_values_adv)
        tp_c = comp_detect(kl_values_adv, kl_mean[i], kl_std[i], alpha_c, plus=True)
        tp_rate_kl.append(tp_c)
    if len(tp_rate_kl) > 0:
        FP_kl = np.mean(fp_rate_kl)
        TP_kl = np.mean(tp_rate_kl)
    else:
        FP_kl = np.nan
        TP_kl = np.nan
    print('false alarm rate (KL):', FP_kl)
    print('detection rate (KL):', TP_kl)

    results = {
        'success_rate': success_rate,
        'mean_dist_l2': np.mean(l2_diff),
        'std_dist_l2': np.sqrt(np.var(l2_diff)),
        'mean_dist_l0': np.mean(l0_diff),
        'std_dist_l0': np.sqrt(np.var(l0_diff)),
        'mean_dist_li': np.mean(li_diff),
        'std_dist_li': np.sqrt(np.var(li_diff)),
        'FP_logpx': fp_logpx,
        'TP_logpx': tp_logpx,
        'FP_logpxy': FP_logpxy,
        'TP_logpxy': TP_logpxy,
        'FP_kl': FP_kl,
        'TP_kl': TP_kl
    }

    # Optionally save results
    if save:
        import os, pickle
        if not os.path.exists('detection_results'):
            os.mkdir('detection_results')
        path = os.path.join('detection_results', guard_name)
        if not os.path.exists(path):
            os.mkdir(path)
        filename = f"{data_name}_{victim_name}"
        if targeted:
            filename += '_targeted'
        else:
            filename += '_untargeted'
        filename += '.pkl'
        pickle.dump(results, open(os.path.join(path, filename), 'wb'))
        print("results saved at", os.path.join(path, filename))

    return results

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate detection metrics on clean and adversarial examples.')

    parser.add_argument('--guard_name',
                        type=str,
                        default='bayes_model',
                        help='Identifier for the "guard" model.')
    parser.add_argument('--victim_name',
                        type=str,
                        default='mlp',
                        help='Identifier for the victim model.')
    parser.add_argument('--data_name',
                        type=str,
                        default='mnist',
                        help='Dataset name.')
    parser.add_argument('--targeted',
                        action='store_true',
                        default=False,
                        help='Whether the attack is targeted.')
    parser.add_argument('--save',
                        action='store_true',
                        default=False,
                        help='Whether to save results to disk.')
    
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--conv', '-C', action='store_true', default=False)
    parser.add_argument('--attack', '-A', type=str, default='fgsm_eps0.10')
