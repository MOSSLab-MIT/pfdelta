import json
import argparse
import copy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root_to_task = {
    "CANOS": "runs/canos_task_1_3",
    "GNS": "runs/gns_task_1_3",
    "PFNet": "runs/pfnet_task_1_3",
}

def parser():
    parser = argparse.ArgumentParser(
        description="Plots the results for cases other than the train one"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Picks which model you want to plot"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # args = parser()
    # model = args.model

    # Errors on 57, 500
    with open("test_3.1_errors_p_seeds.json", "r") as f:
        errors = json.load(f)

    # Errors on 118
    with open("test_errors_p_seeds.json", "r") as f:
        errors118 = json.load(f)

    # Inference times on 57, 500
    with open("times_p_seeds.json", "r") as f:
        times = json.load(f)

    # Add 118 to dictionary
    for root, errors_model in errors.items():
        errors_model["case118"] = copy.deepcopy(errors118[root])
        # Temporary fix
        errors_model["case2000"] = copy.deepcopy(errors_model["case500"])

    # Fix for now
    for root, times_model in times.items():
        times_model["case118"] = copy.deepcopy(times_model["case57"])
        times_model["case2000"] = copy.deepcopy(times_model["case500"])

    # Summarize results
    for 

    import ipdb
    ipdb.set_trace()
