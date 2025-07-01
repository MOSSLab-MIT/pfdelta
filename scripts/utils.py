import os
import yaml
import json
import glob

import torch

from core.utils.main_utils import load_registry
from core.utils.registry import registry


def find_run(run_name):
    r"""Finds the path to the run given in run_name."""
    pattern = os.path.join("runs", "**", run_name)
    matching_runs = glob.glob(pattern, recursive=True)

    num_matches = len(matching_runs)
    if len(matching_runs) > 1:
        print(f"WARNING! Found {len(matching_runs)} runs with that name." + \
            "Only the first one will be used.")
        print("Here's all of them:")
        for i, folder in enumerate(matching_runs):
            print(f"{i}.", folder)

    return matching_runs[0] if len(matching_runs) > 0 else -1

def load_config_and_trainer(run_location):
    config = load_config(run_location)
    trainer = load_trainer(config)

    return trainer, config

def load_config(run_location):
    # Load config file
    config_location = os.path.join(run_location, "config.yaml")
    with open(config_location, 'r') as f:
        config = yaml.safe_load(f)
    config["functional"]["is_debug"] = True

    return config

def load_trainer(config):
    run_location = config["functional"]["run_location"]

    # Load registry
    load_registry()

    # Load trainer, empty
    trainer_name = config["functional"]["trainer_name"]
    trainer_class = registry.get_trainer_class(trainer_name)
    trainer = trainer_class(config)

    # Load train errors
    train_location = os.path.join(run_location, "train.json")
    with open(train_location, 'r') as f:
        train_errors = json.load(f)
    trainer.train_errors = train_errors

    # Load val errors
    val_location = os.path.join(run_location, "val.json")
    with open(val_location, 'r') as f:
        val_errors = json.load(f)
    trainer.val_errors = val_errors

    # Load model parameters
    model_location = os.path.join(run_location, "model.pt")
    model_params = torch.load(model_location, map_location="cpu")
    trainer.model.load_state_dict(model_params)
    trainer.model.eval()

    return trainer
