import argparse
from tqdm import tqdm
import os, pickle, sys, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append('../')
sys.path.append('../utils/')
from utils import utils
from alg.vae_new import bayes_classifier

def comp_lowerbound_func(vae_type):
    if vae_type == "A":
        from alg.lowerbound_functions import lowerbound_A as lowerbound_func
    elif vae_type == "B":
        from alg.lowerbound_functions import lowerbound_B as lowerbound_func
    elif vae_type == "C":
        from alg.lowerbound_functions import lowerbound_C as lowerbound_func
    elif vae_type == "D":
        from alg.lowerbound_functions import lowerbound_D as lowerbound_func
    elif vae_type == "E":
        from alg.lowerbound_functions import lowerbound_E as lowerbound_func
    elif vae_type == "F":
        from alg.lowerbound_functions import lowerbound_F as lowerbound_func
    elif vae_type == "G":
        from alg.lowerbound_functions import lowerbound_G as lowerbound_func
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return lowerbound_func

def create_decoder(vae_type, generator):
    if vae_type == "A":
        dec = (generator.pyz_params, generator.pxzy_params)
    elif vae_type == "B":
        dec = (generator.pzy_params, generator.pxzy_params)
    elif vae_type == "C":
        dec = (generator.pyzx_params, generator.pxz_params)
    elif vae_type == "D":
        dec = (generator.pyzx_params, generator.pzx_params)
    elif vae_type == "E":
        dec = (generator.pyz_params, generator.pzx_params)
    elif vae_type == "F":
        dec = (generator.pyz_params, generator.pxz_params)
    elif vae_type == "G":
        dec = (generator.pzy_params, generator.pxz_params)
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return dec

def comp_logp(logit, y, text, comp_logit_dist=False):
    """
    Compute log probabilities and related statistics.

    Args:
        logit (Tensor): logits of shape (N, nb_classes).
        y (Tensor): one-hot labels of shape (N, nb_classes).
        text (str): Tag for printing results.
        comp_logit_dist (bool): Whether to compute distribution of logits.

    Returns:
        List of computed statistics.
    """
    # logsumexp over classes
    logpx = torch.logsumexp(logit, axis=0)
    logpx_mean = torch.mean(logpx)
    logpx_std = torch.sqrt(torch.var(logpx))
    
    logpxy = torch.sum(y * logit, axis=1)
    logpxy_mean = []
    logpxy_std = []

    nb_classes = y.shape[1]
    for i in range(nb_classes):
        ind = torch.where(y[:, i] == 1)[0]
        if len(ind) > 0:
            logpxy_mean.append(torch.mean(logpxy[ind]))
            logpxy_std.append(torch.sqrt(torch.var(logpxy[ind], unbiased=False)))
        else:
            # If no samples for this class, just append NaNs
            logpxy_mean.append(float('nan'))
            logpxy_std.append(float('nan'))

    print('%s: logp(x) = %.3f +- %.3f, logp(x|y) = %.3f +- %.3f' %
          (text, logpx_mean, logpx_std, torch.nanmean(torch.Tensor(logpxy_mean)), torch.nanmean(torch.Tensor(logpxy_std))))

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
            ind = torch.where(y[:, i] == 1)[0]
            if len(ind) > 0:
                logit_class = logit[ind]
                logit_mean.append(torch.mean(logit_class, axis=0))
                logit_std.append(torch.sqrt(torch.var(logit_class, axis=0)))

                # Compute softmax and KL divergence
                logit_tmp = logit_class - torch.logsumexp(logit_class, axis=1)[:, torch.newaxis]
                softmax_mean = torch.mean(torch.exp(logit_tmp), axis=0)
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
                pi = torch.exp(logit_tmp)
                # For each sample:
                # kl_i = sum(Pmean * (log(Pmean)-log(pi_i)))
                kl_vals = []
                for sample_idx in range(pi.shape[0]):
                    kl_val = torch.sum(softmax_mean * (torch.log(softmax_mean) - logit_tmp[sample_idx]))
                    kl_vals.append(kl_val)
                kl_vals = torch.Tensor(kl_vals)
                logit_kl_mean.append(torch.mean(kl_vals))
                logit_kl_std.append(torch.sqrt(torch.var(kl_vals)))
            else:
                # If no samples for that class
                logit_mean.append(torch.full((nb_classes,), torch.nan))
                logit_std.append(torch.full((nb_classes,), torch.nan))
                logit_kl_mean.append(torch.nan)
                logit_kl_std.append(torch.nan)
                softmax_mean_list.append(torch.full((nb_classes,), torch.nan))

        results.extend([logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean_list])

    return results

def comp_detect(x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor, alpha: float, plus: bool):
    """
    Compute detection rate given a criterion:
    If plus=True: 
        detect if x > x_mean + alpha * x_std
    else: 
        detect if x < x_mean - alpha * x_std
    """
    if plus:
        detect_rate = torch.mean((x > x_mean + alpha * x_std).float())
    else:
        detect_rate = torch.mean((x < x_mean - alpha * x_std).float())
    return detect_rate * 100

def search_alpha(x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor, target_rate=5.0, plus=False):
    """
    Binary search for alpha such that detection rate is close to target_rate.
    """
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

def test_attacks(
    attack_method: str,
    epsilons: list,
    modele_attacked=str,
    path_data="./mnist_results",
    save=False, 
    data_name='mnist',
):
    """
    Evaluate detection metrics on clean and adversarial examples.

    Args:
        attack_method (str): Name of the attack method performed.
        epsilons (list): List of epsilon values for the attack.
        modele_attacked (str): Identifier for the model attacked.
        path_data (str): Path to pkl files containing the data from the attacks.
        save (bool): Whether to save results to disk.
        data_name (str): Name of the dataset.

    Returns:
        results (dict): Dictionary of detection statistics.
    """
    success_rate_list = []
    l2_diff_mean_list = []
    l2_diff_std_list = []
    l0_diff_mean_list = []
    l0_diff_std_list = []
    li_diff_mean_list = []
    li_diff_std_list = []
    fp_logpx_list = []
    tp_logpx_list = []
    FP_logpxy_list = []
    TP_logpxy_list = []
    FP_kl_list = []
    TP_kl_list = []

    dimY = 10 if data_name != "gtsrb" else 43
    for epsilon in epsilons:
        print(f"-----------------  Running detection for epsilon={epsilon}  ---------------------")
        if epsilon == 0:
            print('perturb for successful attack: L_2 = %.3f +- %.3f' % (0.0, 0.0))
            print('perturb for successful attack: L_inf = %.3f +- %.3f' % (0.0, 0.0))
            print('perturb for successful attack: L_0 = %.3f +- %.3f' % (0.0, 0.0))
            print('false alarm rate (logp(x)):', 0.0)
            print('detection rate (logp(x)):', 0.0)
            print('false alarm rate (logp(x|y)):', 0.0)
            print('detection rate (logp(x|y)):', 0.0)
            print('false alarm rate (KL):', 0.0)
            print('detection rate (KL):', 0.0)
            success_rate_list.append(0.0)
            l2_diff_mean_list.append(0.0)
            l2_diff_std_list.append(0.0)
            l0_diff_mean_list.append(0.0)
            l0_diff_std_list.append(0.0)
            li_diff_mean_list.append(0.0)
            li_diff_std_list.append(0.0)
            fp_logpx_list.append(0.0)
            tp_logpx_list.append(0.0)
            FP_logpxy_list.append(0.0)
            TP_logpxy_list.append(0.0)
            FP_kl_list.append(0.0)
            TP_kl_list.append(0.0)
            continue
        filename_data = f"{modele_attacked}_{attack_method}_{data_name}_epsilon_{epsilon}.pkl"
        file_path = os.path.join(path_data, filename_data)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data= pickle.load(f)
                x_clean = data['x_clean']
                y_clean = data['y_clean']
                x_adv = data['x_adv']
                y_adv = data['y_adv']
                y_logit_clean = data['y_clean_logits']
                y_logit_adv = data['y_adv_logits']
        else:
            raise ValueError(f"File {file_path} not found, probably the attack has not be performed for epsilon={epsilon}.")

        # Identify successful attacks (where victim's adv prediction != clean label)
        correct_prediction = (torch.argmax(y_adv, dim=1) == torch.argmax(y_clean, dim=1))
        success_rate = 100.0 * (torch.mean(correct_prediction.to(torch.float32)))
        ind_success = torch.where(correct_prediction)[0]

        if len(ind_success) == 0:
            print("The attack has 0% success rate in the classifier.")
            diff = torch.zeros_like(x_clean)
        else:
            diff = x_adv[ind_success] - x_clean[ind_success]

        # Compute L2, L0, L_inf perturbations for successful attacks
        
        l2_diff = torch.sqrt(torch.sum(diff**2, dim=(1, 2, 3)))
        li_diff = torch.amax(torch.abs(diff), dim=(1, 2, 3))
        l0_diff = torch.sum(diff != 0, dim=(1, 2, 3))
        print('perturb for successful attack: L_2 = %.3f +- %.3f' % (torch.mean(l2_diff), torch.sqrt(torch.var(l2_diff, unbiased=False))))
        print('perturb for successful attack: L_inf = %.3f +- %.3f' % (torch.mean(li_diff), torch.sqrt(torch.var(li_diff,unbiased=False))))
        print('perturb for successful attack: L_0 = %.3f +- %.3f' % (torch.mean(l0_diff.to(torch.float32)), torch.sqrt(torch.var(l0_diff.to(torch.float32), unbiased=False))))

        # y_adv and y_clean are not one-hot labels, we need to convert them
        y_clean_one_hot = F.one_hot(torch.argmax(y_clean, dim=1), num_classes=dimY).to(torch.float32)
        y_adv_one_hot = F.one_hot(torch.argmax(y_adv, dim=1), num_classes=dimY).to(torch.float32)
        
        # Compute log probabilities
        results_clean = comp_logp(y_logit_clean, y_clean_one_hot,'clean')
        results_adv = comp_logp(y_logit_adv[ind_success], y_adv_one_hot[ind_success], 'adv (wrong)')

        # Detection based on logp(x)
        # If guard_name in ['mlp', 'cnn'], plus=True else False (following original logic)
        plus = True if modele_attacked in ['mlp', 'cnn'] else False
        alpha, detect_rate = search_alpha(results_clean[0], results_clean[1], results_clean[2], plus=plus)
        fp_logpx = comp_detect(results_clean[0], results_clean[1], results_clean[2], alpha, plus=plus)
        tp_logpx = comp_detect(results_adv[0], results_clean[1], results_clean[2], alpha, plus=plus)
        print('false alarm rate (logp(x)):', fp_logpx.item())
        print('detection rate (logp(x)):', tp_logpx.item())

        # Detection based on logp(x|y)
        fp_rate = []
        tp_rate_vals = []
        for i in range(dimY):
            ind = torch.where(y_clean_one_hot[:, i] == 1)[0]
            if len(ind) == 0:
                continue
            alpha_c, _ = search_alpha(results_clean[3][ind], results_clean[4][i], results_clean[5][i], plus=plus)
            fp_c = comp_detect(results_clean[3][ind], results_clean[4][i], results_clean[5][i], alpha_c, plus=plus)
            fp_rate.append(fp_c)

            adv_ind = torch.where(y_adv_one_hot[ind_success][:, i] == 1)[0]
            if len(adv_ind) == 0:
                continue
            tp_c = comp_detect(results_adv[3][adv_ind], results_clean[4][i], results_clean[5][i], alpha_c, plus=plus)
            tp_rate_vals.append(tp_c)
        if len(tp_rate_vals) > 0:
            FP_logpxy = torch.mean(torch.Tensor(fp_rate))
            TP_logpxy = torch.mean(torch.Tensor(tp_rate_vals))
            print('false alarm rate (logp(x|y)):', FP_logpxy.item())
            print('detection rate (logp(x|y)):', TP_logpxy.item())
        else:
            FP_logpxy = torch.nan
            TP_logpxy = torch.nan
            print('false alarm rate (logp(x|y)):', FP_logpxy)
            print('detection rate (logp(x|y)):', TP_logpxy)

        # KL-based detection
        # Extract the logit distribution stats from training
        # last 5 results from results_train are [logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean]
        _, _, kl_mean, kl_std, softmax_mean_list = results_clean[-5:]

        # We need to compute KL on train and adv again per class
        fp_rate_kl = []
        tp_rate_kl = []
        
        for i in range(dimY):
            # compute KL for the training samples of class i
            ind = torch.where(y_clean_one_hot[:, i] == 1)[0]
            if len(ind) == 0 or torch.isnan(kl_mean[i]):
                continue
            # compute KL wrt. softmax_mean_list[i]
            logit_clean_i = y_clean[ind]
            logit_tmp = logit_clean_i - torch.logsumexp(logit_clean_i, axis=1, keepdim=True)
            pi = torch.exp(logit_tmp)
            pmean = softmax_mean_list[i]
            kl_values_train = []
            for j in range(pi.shape[0]):
                kl_val = torch.sum(pmean * (torch.log(pmean) - logit_tmp[j]))
                kl_values_train.append(kl_val)
            kl_values_train = torch.Tensor(kl_values_train)
            alpha_c, _ = search_alpha(kl_values_train, kl_mean[i], kl_std[i], plus=True)
            fp_c = comp_detect(kl_values_train, kl_mean[i], kl_std[i], alpha_c, plus=True)
            fp_rate_kl.append(fp_c)

            # adv
            adv_ind = torch.where(y_adv_one_hot[ind_success][:, i] == 1)[0]
            if len(adv_ind) == 0:
                continue
            logit_adv_i = y_logit_adv[ind_success][adv_ind]
            logit_tmp_adv = logit_adv_i - torch.logsumexp(logit_adv_i, dim=0, keepdim=True)
            pi_adv = torch.exp(logit_tmp_adv)
            kl_values_adv = []
            for j in range(pi_adv.shape[0]):
                kl_val = torch.sum(pmean * (torch.log(pmean) - logit_tmp_adv[j]))
                kl_values_adv.append(kl_val)
            kl_values_adv = torch.Tensor(kl_values_adv)
            tp_c = comp_detect(kl_values_adv, kl_mean[i], kl_std[i], alpha_c, plus=True)
            tp_rate_kl.append(tp_c)
        if len(tp_rate_kl) > 0:
            FP_kl = torch.mean(torch.Tensor(fp_rate_kl))
            TP_kl = torch.mean(torch.Tensor(tp_rate_kl))
            print('false alarm rate (KL):', FP_kl.item())
            print('detection rate (KL):', TP_kl.item())
        else:
            FP_kl = torch.nan
            TP_kl = torch.nan
            print('false alarm rate (KL):', FP_kl)
            print('detection rate (KL):', TP_kl)

        success_rate_list.append(success_rate)
        l2_diff_mean_list.append(torch.mean(l2_diff))
        l2_diff_std_list.append(torch.sqrt(torch.var(l2_diff, unbiased=False)))
        l0_diff_mean_list.append(torch.mean(l0_diff.to(torch.float32))) 
        l0_diff_std_list.append(torch.sqrt(torch.var(l0_diff.to(torch.float32), unbiased=False)))
        li_diff_mean_list.append(torch.mean(li_diff))
        li_diff_std_list.append(torch.sqrt(torch.var(li_diff, unbiased=False)))
        fp_logpx_list.append(fp_logpx)
        tp_logpx_list.append(tp_logpx)
        FP_logpxy_list.append(FP_logpxy)
        TP_logpxy_list.append(TP_logpxy)
        FP_kl_list.append(FP_kl)
        TP_kl_list.append(TP_kl)

    results = {
        'success_rate': list(np.array(success_rate_list)),
        'l2_diff_mean': list(np.array(l2_diff_mean_list)),
        'l2_diff_std': list(np.array(l2_diff_std_list)),
        'l0_diff_mean': list(np.array(l0_diff_mean_list)),
        'l0_diff_std': list(np.array(l0_diff_std_list)),
        'li_diff_mean': list(np.array(li_diff_mean_list)),
        'li_diff_std': list(np.array(li_diff_std_list)),
        'fp_logpx': list(np.array(fp_logpx_list)),
        'tp_logpx': list(np.array(tp_logpx_list)),
        'FP_logpxy': list(np.array(FP_logpxy_list)),
        'TP_logpxy': list(np.array(TP_logpxy_list)),
        'FP_kl': list(np.array(FP_kl_list)),
        'TP_kl': list(np.array(TP_kl_list))
    }

    # Optionally save results
    if save:
        if not os.path.exists('detection_results'):
            os.mkdir('detection_results')
        path = os.path.join('detection_results')
        if not os.path.exists(path):
            os.mkdir(path)
        filename = f"{data_name}_{modele_attacked}_{attack_method}_detection_results"
        filename += '.json'
        with open(os.path.join(path, filename), 'w') as f:
            json.dump(results, f)
        print("results saved at", os.path.join(path, filename))

    return results

def plot_detection_rate(results, epsilons, vae_type, attack):
    """
    Plot detection rate for different epsilon values.
    """
    import matplotlib.pyplot as plt

    plt.plot(epsilons, results['tp_logpx'], label='logp(x)')
    plt.plot(epsilons, results['TP_logpxy'], label='logp(x|y)')
    plt.plot(epsilons, results['TP_kl'], label='KL')
    plt.legend()
    plt.xlabel('epsilon')
    plt.ylabel('detection rate')
    plt.title('Detection rate for %s %s'%(vae_type, attack))
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate detection metrics on clean and adversarial examples.')

    parser.add_argument('--model_attacked', '-M',
                        type=str,
                        default='A',
                        help='Identifier for the model attacked.')
    parser.add_argument('--attack',
                        '-A',
                        type=str,
                        default='FGSM')
    parser.add_argument('--path_data',
                        type=str,
                        default='./mnist_results',
                        help='Path to pkl files containing the data from the attacks.')
    parser.add_argument('--data_name',
                        type=str,
                        default='mnist',
                        help='Dataset name.')
    parser.add_argument('--save',
                        action='store_true',
                        default=False,
                        help='Whether to save results to disk.')
    parser.add_argument('--plot',
                        action='store_true',
                        default=False,
                        help='Whether to plot detection rate.')
    parser.add_argument("--epsilons",
                        type=float,
                        nargs="+",
                        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help="List of epsilon values for FGSM attack.",
    )
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--conv', '-C', action='store_true', default=False)

    args = parser.parse_args()
    result = test_attacks(
        attack_method=args.attack,
        epsilons=args.epsilons,
        modele_attacked=args.model_attacked,
        path_data=args.path_data,
        data_name=args.data_name,
        save=args.save
    )
    plot_detection_rate(result, args.epsilons, args.model_attacked, args.attack)
    # for vae in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    #     for attack in ['FGSM', 'PGD', 'MIM']:
    #         args = parser.parse_args(['-M', vae, '-A', attack])
    #         result = test_attacks(
    #             attack_method=args.attack,
    #             epsilons=args.epsilons,
    #             modele_attacked=args.model_attacked,
    #             path_data=args.path_data,
    #             data_name=args.data_name,
    #             save=args.save
    #         )

    #         plot_detection_rate(result, args.epsilons, args.model_attacked, args.attack)
    