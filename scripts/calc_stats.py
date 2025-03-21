import os
import sys
import argparse
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.analysis_tools import (
    find_run,
    load_model,
    load_test_datasets,
    load_losses,
    model_inference
)
from core.utils.main_utils import load_registry

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_name", "-r",
    type=str,
    help="Name of the run that will be analyzed."
)


def calculate_stats(data, outputs, all_data):
    bus_type = data["bus"].bus_type
    edge_index = data["bus", "forward", "bus"].edge_index
    vm, va, p, q = outputs.T
    # VM at PQ
    PQ_mask = bus_type == 1
    vm_mean = vm[PQ_mask].mean()
    vm_min = vm[PQ_mask].min().item()
    vm_max = vm[PQ_mask].max().item()
    # Save
    vm_data = all_data["vm"]
    vm_data["mean"].append(vm_mean)
    vm_data["min"].append(vm_min)
    vm_data["max"].append(vm_max)
    # Q at PV
    PV_mask = bus_type == 2
    q_mean = q[PV_mask].mean()
    q_min = q[PV_mask].min().item()
    q_max = q[PV_mask].max().item()
    # Save
    q_data = all_data["q"]
    q_data["mean"].append(q_mean)
    q_data["min"].append(q_min)
    q_data["max"].append(q_max)
    # VA at all branches
    fr, to = edge_index
    va_diff = torch.abs(va[fr] - va[to])
    va_mean = va_diff.mean()
    va_min = va_diff.min().item()
    va_max = va_diff.max().item()
    # Save
    va_data = all_data["va"]
    va_data["mean"].append(va_mean)
    va_data["min"].append(va_min)
    va_data["max"].append(va_max)
    # P for slack bus
    Slack_mask = bus_type == 3
    p_mean = p[Slack_mask].mean()
    p_min = p[Slack_mask].min().item()
    p_max = p[Slack_mask].max().item()
    # Save
    p_data = all_data["p"]
    p_data["mean"].append(p_mean)
    p_data["min"].append(p_min)
    p_data["max"].append(p_max)


def print_stats(all_data, key):
    means = torch.stack(all_data[key]["mean"])
    mean = means.mean().item()
    std = means.std().item()
    minn = min(all_data[key]["min"])
    maxx = max(all_data[key]["max"])
    print(f"Mean: {mean}")
    print(f"STD: {std}")
    print(f"Min: {minn}")
    print(f"Max: {maxx}\n")


if __name__ == "__main__":
    parsed, extra_args = parser.parse_known_args()

    run_name = parsed.run_name
    all_runs = find_run(run_name)
    assert len(all_runs) > 0, "No run found with this name!"
    assert len(all_runs) == 1, "More than one run bare this name!"
    run_location = all_runs[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE BEING USED: {device}")
    print(f"Loading run in: {run_location}\n")

    print("Loading registry...")
    load_registry()
    print("Registry loaded!\n")
    print("Loading model...")
    config, model = load_model(run_location, device)
    print(f"Model {model.__class__.__name__} loaded!\n")
    print("Loading dataset...")
    # Let's make it only the first validation set
    datasets = load_test_datasets(config)
    print(f"Dataset loaded!\n")

    # Evaluate model
    model = model.eval().to(device)
    print("Calculating test errors...")
    seconds_per_inference = []
    for i, dataset in enumerate(datasets):
        model_results = model_inference(model, dataset)
        if i == 0:
            all_outputs = model_results[0]
        speed = model_results[1]
        seconds_per_inference.append(speed)
    print("All outputs and inference speeds calculated!\n")

    print("INFERENCE SPEEDS\n" + "-"*12 + "\n")
    for dataset, speed in zip(datasets, seconds_per_inference):
        dataset_name = dataset.mat_file_name
        print(f"{dataset_name}\t: {speed} per network.")
    print()

    vm_data = {
        "mean": [],
        "min": [],
        "max": [],
    }
    va_data = {
        "mean": [],
        "min": [],
        "max": [],
    }
    p_data = {
        "mean": [],
        "min": [],
        "max": [],
    }
    q_data = {
        "mean": [],
        "min": [],
        "max": [],
    }
    all_data = {
        "vm": vm_data,
        "va": va_data,
        "p": p_data,
        "q": q_data
    }

    for i, data in enumerate(datasets[0]):
        calculate_stats(data, all_outputs[i], all_data)

    print("STATISTICS RESULTS\n" + "-"*18 + "\n")

    print("** Voltage magnitude at PQ buses **")
    print_stats(all_data, "vm")

    print("** Reactive power  at PV buses **")
    print_stats(all_data, "q")

    print("** Voltage angle difference  at branches **")
    print_stats(all_data, "va")

    print("** P at the slack bus **")
    print_stats(all_data, "p")
