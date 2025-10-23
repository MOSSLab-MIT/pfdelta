import sys
import os
import json
import statistics
import argparse
import copy

import IPython

# Change working directory to one above
sys.path.append(os.getcwd())

import torch

from scripts.utils import find_run, load_config, load_trainer


def parser():
    parser = argparse.ArgumentParser(
        description="Loads the trainer and the trainable weights."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        required=True,
        help="Folder from which to calculate test errors.",
    )
    parser.add_argument(
        "--case_name",
        type=str,
        default="",
        required=True,
        help="Folder from which to calculate test errors.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Load run paths and configs
    args = parser()
    root = os.path.join("runs", args.root)
    case_name = args.case_name
    seeds = os.listdir(root)
    seeds_paths = [find_run(seed) for seed in seeds]
    seeds_configs = [load_config(run) for run in seeds_paths]

    # Process config files
    for config in seeds_configs:
        # Modify loss calculations
        losses = config["optim"]["train_params"]["train_loss"]
        for loss in losses:
            if loss["name"] == "universal_power_balance":
                pbl_model_name = loss["model"]
                break
        losses = [
            {
                "name": "universal_power_balance",
                "model": pbl_model_name
            }
        ]
        config["optim"]["val_params"]["val_loss"] = losses

        # Modify datasets
        base_dataset = config["dataset"]["datasets"][0]
        base_dataset["task"] = 1.3
        base_dataset["case_name"] = case_name
        datasets = [
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_feasible_n"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_feasible_n-1"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_feasible_n-2"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_near infeasible_n"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_near infeasible_n-1"
            },
            {
                **copy.deepcopy(base_dataset),
                "split": f"separate_{case_name}_test_near infeasible_n-2"
            },
        ]
        config["dataset"] = {
            "datasets": datasets
        }
        config["optim"]["train_params"]["batch_size"] = 2000
        config["optim"]["val_params"]["batch_size"] = 2000

    seeds_trainers = [load_trainer(config) for config in seeds_configs]
    seeds_losses = []
    for i, trainer in enumerate(seeds_trainers):
        print("\nWorking on seed", i)
        print("-"*20)
        losses = {}
        for dataset_type, dataloader in zip(
            ["n", "n-1", "n-2",  "c2i-n", "c2i-n1", "c2i-n2"],
            trainer.dataloaders
        ):
            print("Calculating", dataset_type)
            pbl_mean = trainer.calc_one_val_error(dataloader, i)
            losses[dataset_type] = {
                "PBL Mean": pbl_mean[0],
                "PBL Max": trainer.val_loss[0].power_balance_max.item()
            }

        losses["close2inf"] = {
            "PBL Mean": torch.tensor([
                losses["c2i-n"]["PBL Mean"],
                losses["c2i-n1"]["PBL Mean"],
                losses["c2i-n2"]["PBL Mean"],
            ]).mean().item(),
            "PBL Max": torch.tensor([
                losses["c2i-n"]["PBL Max"],
                losses["c2i-n1"]["PBL Max"],
                losses["c2i-n2"]["PBL Max"],
            ]).max().item(),            
        }
        seeds_losses.append(losses)

    keys = ["n", "n-1", "n-2", "close2inf"]
    pbl_means = {
        "n": [],
        "n-1": [],
        "n-2": [],
        "close2inf": [],
    }
    pbl_maxs = {
        "n": [],
        "n-1": [],
        "n-2": [],
        "close2inf": [],
    }
    for losses in seeds_losses:
        for key in keys:
            pbl_means[key].append(losses[key]["PBL Mean"])
            pbl_maxs[key].append(losses[key]["PBL Max"])

    print(f"\nPRINTING AVGS FOR {root} +- 1SD ON {case_name}")
    print("-"*60)
    for test_type, losses in pbl_means.items():
        mean = torch.tensor(losses).mean()
        std = torch.tensor(losses).std()
        print(f"PBL Mean {test_type}: {mean.item()} +- {std.item()}")

    for test_type, losses in pbl_maxs.items():
        mean = torch.tensor(losses).mean()
        std = torch.tensor(losses).std()
        print(f"PBL Max {test_type}: {mean.item()} +- {std.item()}")

    # Save specific values
    if os.path.exists("test_3.1_errors_p_seeds.json"):
        with open("test_3.1_errors_p_seeds.json", "r") as f:
            results = json.load(f)
    else:
        results = {}

    if root not in results:
        results[root] = {}

    results[root][case_name] = {
        "PBL Mean": pbl_means,
        "PBL Max": pbl_maxs
    }
    with open("test_3.1_errors_p_seeds.json", "w") as f:
        json.dump(results, f, indent=2)
