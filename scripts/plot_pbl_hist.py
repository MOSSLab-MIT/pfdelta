#!/usr/bin/env python3
"""
Compute per-sample power-balance-loss (PBL) metrics on validation datasets
for a saved run and plot a histogram of a chosen metric (mean, l2, max).

Base plotting logic is AI-generated.
Usage:
  python scripts/plot_pbl_hist.py --run /path/to/run_folder --metric max --out out.png

"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root is importable
sys.path.append(os.getcwd())

from core.utils.main_utils import load_registry
from core.utils.registry import registry
from core.utils.pf_losses_utils import PowerBalanceLoss
from torch_geometric.nn import global_add_pool, global_max_pool


def extract_state_dict(ckpt):
    # Robustly extract a state_dict from a loaded checkpoint object
    if isinstance(ckpt, dict):
        for candidate in ("state_dict", "model_state_dict", "model", "net"): 
            if candidate in ckpt and isinstance(ckpt[candidate], dict):
                return ckpt[candidate]
        # If values are tensors it's probably already a state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        # search for nested dict that looks like a state_dict
        for v in ckpt.values():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                return v
    raise RuntimeError("Could not find a state_dict in the checkpoint")


# def maybe_strip_module(sd):
#     # If the keys have 'module.' prefix, strip it
#     if any(k.startswith('module.') for k in sd.keys()):
#         return {k.replace('module.', '', 1): v for k, v in sd.items()}
#     return sd


def compute_per_sample_pbl(trainer, val_loader, metric='max', device=torch.device('cpu')):
    # Move model to device and set eval mode
    trainer.model.to(device)
    trainer.model.eval()

    model_name = trainer.config['model']['name']
    if "canos_pf" == model_name:
        model_name = "CANOS"
    if "powerflownet" == model_name:
        model_name = "PFNet"
    pbl_class = PowerBalanceLoss(model_name)

    per_sample_vals = []

    for data in val_loader:
        data = data.to(device)
        with torch.no_grad():
            outputs = trainer.model(data)
       
        # Collect predictions and edge_attr
        preds = pbl_class.collect_model_predictions(model_name, data, outputs)
        V_pred, theta_pred, Pnet, Qnet = preds['predictions']
        r, x, bs, tau, theta_shift = preds['edge_attr']
        edge_index = data['bus','branch','bus'].edge_index
        src, dst = edge_index

        Pbus_pred = torch.zeros_like(V_pred)
        Qbus_pred = torch.zeros_like(V_pred)

        # compute flows (match implementation in PowerBalanceLoss)
        Y_real = torch.real(1/(r + 1j*x))
        Y_imag = torch.imag(1/(r + 1j*x))
        suscept = bs
        delta_theta1 = theta_pred[src] - theta_pred[dst]
        delta_theta2 = theta_pred[dst] - theta_pred[src]

        # Active power flows
        P_flow_src = V_pred[src] * V_pred[dst] / tau * (
            -Y_real * torch.cos(delta_theta1 - theta_shift) - Y_imag * torch.sin(delta_theta1 - theta_shift)
        ) + Y_real * (V_pred[src] / tau)**2

        P_flow_dst = V_pred[dst] * V_pred[src] / tau * (
            -Y_real * torch.cos(delta_theta2 - theta_shift) - Y_imag * torch.sin(delta_theta2 - theta_shift)
        ) + Y_real * V_pred[dst]**2

        # Reactive power flows
        Q_flow_src = V_pred[src] * V_pred[dst] / tau * (
            -Y_real * torch.sin(delta_theta1 - theta_shift) + Y_imag * torch.cos(delta_theta1 - theta_shift)
        ) - (Y_imag + suscept / 2) * (V_pred[src] / tau)**2

        Q_flow_dst = V_pred[dst] * V_pred[src] / tau * (
            -Y_real * torch.sin(delta_theta2 - theta_shift) + Y_imag * torch.cos(delta_theta2 - theta_shift)
        ) - (Y_imag + suscept / 2) * V_pred[dst]**2

        Pbus_pred = torch.zeros_like(V_pred).scatter_add_(0, src, P_flow_src)
        Pbus_pred = Pbus_pred.scatter_add_(0, dst, P_flow_dst)

        Qbus_pred = torch.zeros_like(V_pred).scatter_add_(0, src, Q_flow_src)
        Qbus_pred = Qbus_pred.scatter_add_(0, dst, Q_flow_dst)

        shunt_g = data['bus'].shunt[:,0]
        shunt_b = data['bus'].shunt[:,1]

        delta_P = Pnet - Pbus_pred - V_pred**2 * shunt_g
        delta_Q = Qnet - Qbus_pred + V_pred**2 * shunt_b

        # set slack mismatches to zero (same as loss)
        try:
            slack_idx = data['slack','slack_link','bus'].edge_index[1]
            # change to small epsilon in collect_model_preds
            delta_P[slack_idx] = 0
            delta_Q[slack_idx] = 0
        except Exception:
            # if slack not present, ignore
            pass

        delta_PQ_2 = delta_P**2 + delta_Q**2
        delta_PQ_magnitude = torch.sqrt(delta_PQ_2)

        batch_idx = data['bus'].batch
        # per-sample mean
        sum_mag = global_add_pool(delta_PQ_2, batch_idx)
        counts = torch.bincount(batch_idx)
        counts = counts.to(sum_mag.dtype)
        mean_per_sample = sum_mag / counts

        # per-sample l2 (sqrt of sum of squares per sample)
        sum_delta2 = global_add_pool(delta_PQ_2, batch_idx)
        l2_per_sample = torch.sqrt(sum_delta2)

        # per-sample max.
        max_per_sample = global_max_pool(delta_PQ_magnitude, batch_idx)
    
        if metric == 'max':
            per_sample_vals.extend(max_per_sample.cpu().numpy().tolist())
        elif metric == 'mean':
            per_sample_vals.extend(mean_per_sample.cpu().numpy().tolist())
        elif metric == 'l2':
            per_sample_vals.extend(l2_per_sample.cpu().numpy().tolist())
        else:
            raise ValueError('Unknown metric: ' + metric)

    return np.array(per_sample_vals)


def main():
    # set up environment for running
    parser = argparse.ArgumentParser()
    # parser.add_argument('--run', required=True, help='run folder (the folder that contains config.yaml, model.pt)')
    parser.add_argument('--metric', default='max', choices=['max','mean','l2'])
    # parser.add_argument('--out', default='pbl_hist.png')
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--device', default=None, help="device to run on: 'cpu' or 'cuda'. Default: cuda if available else cpu")
    args = parser.parse_args()
    from files_to_plot import best_runs_folders

    models_eval_output = dict()
    for trained_model_name, folders in best_runs_folders.items():
        eval_output = []
        for run_folder in folders:
            if not os.path.isdir(run_folder):
                raise RuntimeError('run folder does not exist: ' + run_folder)
            if args.device is None:
                device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device_str = args.device
            device = torch.device(device_str)

            # load config
            config_path = os.path.join(run_folder, 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['functional']['is_debug'] = True

            # load registry and trainer
            load_registry()
            trainer_name = config['functional']['trainer_name']
            trainer_class = registry.get_trainer_class(trainer_name)
            trainer = trainer_class(config)

            # load checkpoint
            ckpt_path = os.path.join(run_folder, 'model.pt')
            # load checkpoint onto CPU first or directly onto device
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
            except Exception:
                ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = extract_state_dict(ckpt)
            # state_dict = maybe_strip_module(state_dict)
            # attempt to load; allow partial loads
            # move model to chosen device before loading weights
            trainer.device = device
            trainer.model.to(device)
            res = trainer.model.load_state_dict(state_dict, strict=False)
            print('load_state_dict result:', res)

            all_vals = []
            # compute per-sample pbl across all validation dataloaders (trainer.dataloaders[1:])
            for i, val_loader in enumerate(trainer.dataloaders[1:]):
                print('Processing validation loader', i)
                vals = compute_per_sample_pbl(trainer, val_loader, metric=args.metric, device=device)
                print(f' - collected {len(vals)} samples')
                all_vals.append(vals)
            if len(all_vals) == 0:
                raise RuntimeError('No validation dataloaders found on trainer')
            all_vals = np.array(all_vals).flatten()
            
            eval_output.append(all_vals)
        eval_output = np.average(eval_output, axis=0)
        models_eval_output[trained_model_name] = eval_output
        # plot histogram
    models = ["CANOS", "PFNet"]
    loss_functions = ["mse", "constraint", "constraint_with_mse", "pbl"]
    for loss_function in loss_functions:
        plt.figure(figsize=(6,4))
        plt.xlabel(f'PBL ({args.metric})')
        plt.ylabel('Percentage of samples (out of 5400)')
        plt.title(f'Validation PBL histogram ({args.metric})')
        plt.tight_layout()
        
        # Collect all values across models
        all_values = np.concatenate([
            models_eval_output[model + "_" + loss_function] for model in models
        ])

        # Define common bins across both models
        bins = np.linspace(all_values.min(), all_values.max(), args.bins)

        # Plot histograms with the same bin edges
        for model in models:
            name = model + "_" + loss_function
            print(f"stdev {name}", np.std(models_eval_output[name]))
            plt.hist(models_eval_output[name], bins=bins, alpha=0.5, label=model, cumulative=True, density=True)
        # for model in models:
        #     name = model + "_" + loss_function
        #     plt.hist(models_eval_output[model+"_" +loss_function], bins=args.bins, alpha=0.5, label=model)

        # plt.legend(loc='upper right')
        # plt.savefig(f"{loss_function}_pbl_{args.metric}_cumulative_hist.png", dpi=200)
        # print(f'Saved histogram to {loss_function}_pbl_hist.png' )

    # plt.figure(figsize=(6,4))
    # plt.hist(model_eval_output, bins=args.bins)
    # plt.xlabel(f'PBL ({args.metric})')
    # plt.ylabel('Number of samples')
    # plt.title(f'Validation PBL histogram ({args.metric})')
    # plt.tight_layout()
    # plt.savefig(f"{trained_model_name}_pbl_hist.png", dpi=200)
    # print(f'Saved histogram to {trained_model_name}_pbl_hist.png' )

    # run_folder = args.run
    # if not os.path.isdir(run_folder):
    #     raise RuntimeError('run folder does not exist: ' + run_folder)

    # if args.device is None:
    #     device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # else:
    #     device_str = args.device
    # device = torch.device(device_str)

    # # load config
    # config_path = os.path.join(run_folder, 'config.yaml')
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # config['functional']['is_debug'] = True

    # # load registry and trainer
    # load_registry()
    # trainer_name = config['functional']['trainer_name']
    # trainer_class = registry.get_trainer_class(trainer_name)
    # trainer = trainer_class(config)

    # # load checkpoint
    # ckpt_path = os.path.join(run_folder, 'model.pt')
    # # load checkpoint onto CPU first or directly onto device
    # try:
    #     ckpt = torch.load(ckpt_path, map_location=device)
    # except Exception:
    #     ckpt = torch.load(ckpt_path, map_location='cpu')
    # state_dict = extract_state_dict(ckpt)
    # state_dict = maybe_strip_module(state_dict)
    # # attempt to load; allow partial loads
    # # move model to chosen device before loading weights
    # trainer.device = device
    # trainer.model.to(device)
    # res = trainer.model.load_state_dict(state_dict, strict=False)
    # print('load_state_dict result:', res)

    # # compute per-sample pbl across all validation dataloaders (trainer.dataloaders[1:])
    # all_vals = []
    # for i, val_loader in enumerate(trainer.dataloaders[1:]):
    #     print('Processing validation loader', i)
    #     vals = compute_per_sample_pbl(trainer, val_loader, metric=args.metric, device=device)
    #     print(f' - collected {len(vals)} samples')
    #     all_vals.append(vals)
    # if len(all_vals) == 0:
    #     raise RuntimeError('No validation dataloaders found on trainer')
    # all_vals = np.concatenate(all_vals, axis=0)

    # # plot histogram
    # plt.figure(figsize=(6,4))
    # plt.hist(all_vals, bins=args.bins)
    # plt.xlabel(f'PBL ({args.metric})')
    # plt.ylabel('Number of samples')
    # plt.title(f'Validation PBL histogram ({args.metric})')
    # plt.tight_layout()
    # plt.savefig(args.out, dpi=200)
    # print('Saved histogram to', args.out)


if __name__ == '__main__':
    main()
