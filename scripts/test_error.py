import sys
import os
import statistics
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
            "model": "CANOS"
        },
        {
            "name": "recycle_loss",
            "loss_name": "PBL Max",
            "keyword": "pbl_pf",
            "recycled_parameter": "power_balance_max"
        }
    ]

    for topo, task, split in zip(
            ["none", "n-1", "n-2", "nose"],
            [1.3, 1.3, 1.3, 4.3],
            ["separate_test_feasible_none", "separate_test_feasible_n-1", "separate_test_feasible_n-2", "test"]
    ):
        copy_config = copy.deepcopy(seeds_configs)
        # Load trainers modified so that the val dataset is on the test split desired
        for config in copy_config:
            val_dataset = config["dataset"]["datasets"][1]
            val_dataset["task"] = task
            val_dataset["split"] = split
            val_params = config["optim"]["val_params"]
            val_params["batch_size"] = 2000
            val_params["val_loss"].extend(losses_to_analyze_inputs)

        seeds_trainers = [load_trainer(config) for config in copy_config]

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
            print(f"{topo}, {loss_name}, mean: {statistics.mean(losses)}")
            print(f"{topo}, {loss_name}, std: {statistics.stdev(losses)}")
            print(f"{topo}, {loss_name}, points: {losses}")
        print("\n\n")

    IPython.embed()