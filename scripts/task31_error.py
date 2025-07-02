import sys
import os
import statistics
import time
import argparse
import copy

import IPython

# Change working directory to one above
sys.path.append(os.getcwd())

from scripts.utils import (
    find_run,
    load_config,
    load_trainer
)

def parser():
    parser = argparse.ArgumentParser(
        description="Loads the trainer and the trainable weights.")
    parser.add_argument('--root', type=str, default="", required=True,
        help="Folder from which to calculate test errors.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Load run paths and configs
    args = parser()
    root = os.path.join("runs", args.root)
    seeds = os.listdir(root)
    seeds_paths = [find_run(seed) for seed in seeds]
    seeds_configs = [load_config(run) for run in seeds_paths]

    # Set up additional losses
    losses_to_analyze_inputs = [
        {
            "name": "universal_power_balance",
            "model": "GNS"
        },
        {
            "name": "recycle_loss",
            "loss_name": "PBL Max",
            "keyword": "pbl_pf",
            "recycled_parameter": "power_balance_max"
        }
    ]
    # Load trainers modified so that the val dataset is on the test split desired
    batch_size = 100
    for config in seeds_configs:
        val_dataset = config["dataset"]["datasets"][1]
        val_dataset["task"] = 1.3
        val_dataset["split"] = "test"
        val_dataset["case_name"] = "case500_seeds"
        case_name = val_dataset["case_name"]
        val_params = config["optim"]["val_params"]
        val_params["batch_size"] = batch_size
        val_params["val_loss"].extend(losses_to_analyze_inputs)

    seeds_trainers = [load_trainer(config) for config in seeds_configs]

    # Calculate running losses
    seeds_dataloaders = [trainer.dataloaders[1] for trainer in seeds_trainers]
    seeds_running_losses = [
        trainer.calc_one_val_error(dataloader, 0)
        for dataloader, trainer in zip(seeds_dataloaders, seeds_trainers)
    ]

    # Average over the trainers and save in a dictionary
    seeds_losses = []
    for running_losses, dataloader, trainer in zip(seeds_running_losses, seeds_dataloaders, seeds_trainers):
        losses = {
            loss_name: running_loss / len(dataloader) 
            for loss_name, running_loss in zip(trainer.val_loss_names, running_losses)
        }
        seeds_losses.append(losses)

    # Calculate mean and error bars
    losses_to_analyze = ["PBL Mean", "PBL Max"]
    for loss_name in losses_to_analyze:
        losses = [seed_loss[loss_name] for seed_loss in seeds_losses]
        print(f"{case_name}, {loss_name}, mean: {statistics.mean(losses)}")
        print(f"{case_name}, {loss_name}, std: {statistics.stdev(losses)}")
        print(f"{case_name}, {loss_name}, points: {losses}")
    print("\n\n")

    device = "cuda"
    model_times = []
    for trainer in seeds_trainers:
        print("Calculating a model")
        model = trainer.model.to(device)
        loader = trainer.dataloaders[1]
        times = []
        for data in loader:
            data = data.to(device)
            start = time.time()
            out = model(data)
            end = time.time()
            times.append((end-start)*batch_size)
        model_times.append(sum(times) / 6000)
        print("Seed 1: ", model_times[-1])

    print(f"{case_name}, runtime, mean: {statistics.mean(model_times)}")
    print(f"{case_name}, runtime, std: {statistics.stdev(model_times)}")

    IPython.embed()