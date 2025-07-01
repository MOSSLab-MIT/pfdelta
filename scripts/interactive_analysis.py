import os
import sys
import yaml
import copy
import argparse
import IPython

import torch

# Change working directory to one above
sys.path.append(os.getcwd())

from scripts.utils import (
    find_run,
    load_config_and_trainer,
    load_config,
    load_trainer,
)


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
    config = load_config(run_location)
    trainer = load_trainer(copy.deepcopy(config))

    IPython.embed()
