import torch
import numpy as np
import os
import json


FEASIBILITY_CONFIG = {
    "just feasible": {
        "none": 56000,
        "n-1": 29000,
        "n-2": 20000,
        "test": {"none": 2000, "n-1": 2000, "n-2": 2000}
    },
    "around the nose": {
        "none": 7200,
        "n-1": 7200,
        "n-2": 7200,
        "test": None  # no test set for this regime
    },
    "just the nose": {
        "none": 2000,
        "n-1": 2000,
        "n-2": 2000,
        "test": {"none": 200, "n-1": 200, "n-2": 200}
    },
}

TASK_CONFIG = {
    1.1: {"none": 54000, "n-1": 0, "n-2": 0},
    1.2: {"none": 27000, "n-1": 27000, "n-2": 0},
    1.3: {"none": 18000, "n-1": 18000, "n-2": 18000},
    2.1: {"none": 18000, "n-1": 18000, "n-2": 18000},
    2.2: {"none": 12000, "n-1": 12000, "n-2": 12000},
    2.3: {"none": 6000,  "n-1": 6000,  "n-2": 6000},
    3.1: {"none": 18000, "n-1": 18000, "n-2": 18000},
    3.2: {"none": 18000, "n-1": 18000, "n-2": 18000},
    3.3: {"none": 18000, "n-1": 18000, "n-2": 18000}, # can add 4.1 and 4.2 later? 
}


def mean0_var1(x, mean, std):
    x = (x - mean) / std
    return x


def create_train_test_mapping_json(case_name, seed=11, feasibility_setting="just feasible", root_dir="data/pfdelta_data/"):
    """ """
    feasibility_config = FEASIBILITY_CONFIG[feasibility_setting]
    root =  os.path.join(root_dir, case_name)

    for grid_type in ["none", "n-1", "n-2"]:
        num_samples = feasibility_config[grid_type]
        indices = np.arange(num_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        shuffle_path = os.path.join(root, grid_type, "raw_shuffle.json")
        mappings = {int(i): int(j) for i, j in enumerate(indices)}
        with open(shuffle_path, "w") as f:
            json.dump(mappings, f, indent=3)
        

if __name__ == "__main__":
    create_train_test_mapping_json("case14_seeds", seed=11, feasibility_setting="just feasible", root_dir="data/pfdelta_data/")
    create_train_test_mapping_json("case118_seeds", seed=11, feasibility_setting="just feasible", root_dir="data/pfdelta_data/")
