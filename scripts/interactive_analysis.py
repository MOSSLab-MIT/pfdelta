import os
import sys
import yaml
import argparse

import torch

# Change working directory to one above
sys.path.append(os.getcwd())

from scripts.utils import find_run, load_config_and_trainer


def parser():
    parser = argparse.ArgumentParser(
        description="Loads the trainer and the trainable weights.")
    parser.add_argument('--run_name', type=str, default="", required=True,
        help="Folder from which the best run is found.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parser()
    run_location = find_run(args.run_name)

    # Load trainer
    trainer, config = load_config_and_trainer(run_location)
