import sys
import os
import json
import statistics
import argparse
import copy
import time
from tqdm import tqdm

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
        config["optim"]["train_params"]["batch_size"] = 100
        config["optim"]["val_params"]["batch_size"] = 100

    seeds_trainers = [load_trainer(config) for config in seeds_configs]
    seeds_times = []
    for i, trainer in enumerate(seeds_trainers):
        print("\nWorking on seed", i)
        print("-"*20)
        inference_times = {}
        device = trainer.device
        for dataset_type, dataloader in zip(
            ["n", "n-1", "n-2",  "c2i-n", "c2i-n1", "c2i-n2"],
            trainer.dataloaders
        ):
            print("Calculating", dataset_type)

            # Warming up
            message = "Warming GPU..."
            for data in tqdm(dataloader, desc=message):
                data = data.to(device)
                _ = trainer.model(data)

            # Calculating inference time
            message = "Tracking inference time..."
            times = []
            for data in tqdm(dataloader, desc=message):
                data = data.to(device)

                # Calculate output
                with torch.no_grad():
                    tic = time.time()
                    _ = trainer.model(data)
                    toc = time.time()
                # Calculate inference time
                inference_time = (toc - tic) * 100
                times.append(inference_time)

            dataset_size = len(dataloader.dataset)
            inference_times[dataset_type] = sum(times) / dataset_size

        inference_times["close2inf"] = (
            inference_times["c2i-n"] +
            inference_times["c2i-n1"] +
            inference_times["c2i-n2"]
        ) / 3
        seeds_times.append(inference_times)

    keys = ["n", "n-1", "n-2", "close2inf"]
    total_times = {
        "n": [],
        "n-1": [],
        "n-2": [],
        "close2inf": [],
    }
    for times in seeds_times:
        for key in keys:
            total_times[key].append(times[key])

    print(f"\nPRINTING TIMES FOR {root} +- 1SD ON {case_name}")
    print("-"*50)
    for test_type, times in total_times.items():
        mean = torch.tensor(times).mean()
        std = torch.tensor(times).std()
        print(f"Inference time on {test_type}: {mean.item()} +- {std.item()}")

    # Save specific values
    if os.path.exists("times_p_seeds.json"):
        with open("times_p_seeds.json", "r") as f:
            results = json.load(f)
    else:
        results = {}

    if root not in results:
        results[root] = {}

    results[root][case_name] = total_times
    with open("times_p_seeds.json", "w") as f:
        json.dump(results, f, indent=2)
