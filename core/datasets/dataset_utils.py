import torch
import numpy as np
import os
import json


FEASIBILITY_CONFIG = {
    "feasible": {
        "none": 56000,
        "n-1": 29000,
        "n-2": 20000,
        "test": {"none": 2000, "n-1": 2000, "n-2": 2000}
    },
    "approaching infeasible": {
        "none": 7200,
        "n-1": 7200,
        "n-2": 7200,
        "test": None  # no test set for this regime
    },
    "near infeasible": {
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


def canos_pf_data_mean0_var1(stats, data):
    means = stats["mean"]
    stds = stats["std"]

    def exception_transform(x, mean, std):
        r"""Transforms every value to mean 0, var 1 unless std is 0, in which
        case the value is just transformed to 0."""
        ones_std = std == 0.
        std[ones_std] = 1.
        x = mean0_var1(x, mean, std)
        return x

    values_to_change = [
        ("bus", "x"),
        ("PV", "x"),
        ("PQ", "x"),
        ("slack", "x")
    ]

    for dtype, entry in values_to_change:
        mean = means[dtype][entry]
        std = stds[dtype][entry]
        x = data[dtype][entry]
        data[dtype][entry] = exception_transform(x, mean, std)

    return data


def pfnet_data_mean0_var1(stats, data):
    means = stats["mean"]
    stds = stats["std"]
    eps = 1e-7

    x_mean = means["bus"]["x"] # shape [6]
    x_std = stds["bus"]["x"] + eps # shape [6]
    x_cont = data["bus"]["x"][:, 4:10]
    data["bus"]["x"][:, 4:10] = (x_cont - x_mean) / x_std

    y_mean = means["bus"]["y"] # shape [6]
    y_std = stds["bus"]["y"] + eps # shape [6]
    y_cont = data["bus"]["y"]
    data["bus"]["y"] = (y_cont - y_mean) / y_std

    edge_mean = means[("bus", "branch", "bus")]["edge_attr"]
    edge_std = stds[("bus", "branch", "bus")]["edge_attr"] + eps
    edge_attr = data[("bus", "branch", "bus")]["edge_attr"]
    data[("bus", "branch", "bus")].edge_attr = (edge_attr - edge_mean) / edge_std

    data["case_name"] = stats["casename"]
    return data


def canos_pf_slack_mean0_var1(stats, data):
    means = stats["mean"]
    stds = stats["std"]

    def exception_transform(x, mean, std):
        r"""Transforms every value to mean 0, var 1 unless std is 0, in which
        case the value is just transformed to 0."""
        ones_std = std == 0.
        std[ones_std] = 1.
        x = mean0_var1(x, mean, std)
        return x

    values_to_change = [
        ("slack", "y")
    ]

    for dtype, entry in values_to_change:
        mean = means[dtype][entry]
        std = stds[dtype][entry]
        x = data[dtype][entry]
        data[dtype][entry] = exception_transform(x, mean, std)

    return data

if __name__ == "__main__":
    # create_train_test_mapping_json("case14_seeds", seed=11, feasibility_setting="just feasible", root_dir="data/pfdelta_data/")
    create_train_test_mapping_json("case118_seeds", seed=11, feasibility_setting="feasible", root_dir="data/pfdelta_data/")
