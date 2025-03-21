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
    evaluate_model
)
from core.utils.main_utils import load_registry

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_name", "-r",
    type=str,
    help="Name of the run that will be analyzed."
)


if __name__ == "__main__":
    parsed, extra_args = parser.parse_known_args()

    run_name = parsed.run_name
    all_runs = find_run(run_name)
    assert len(all_runs) > 0, "No run found with this name!"
    assert len(all_runs) == 1, "More than one run bare this name!"
    run_location = all_runs[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading run in: {run_location}\n")

    print("Loading registry...")
    load_registry()
    print("Registry loaded!\n")
    print("Loading model...")
    config, model = load_model(run_location, device)
    print(f"Model {model.__class__.__name__} loaded!\n")
    print("Loading datasets...")
    datasets = load_test_datasets(config)
    print(f"A total of {len(datasets)} datasets loaded!\n")
    print("Loading losses...")
    loss_funcs, loss_names = load_losses(config)
    print(f"A total of {len(loss_funcs)} loss functions loaded!\n")

    # Evaluate model
    model = model.eval().to(device)
    losses_per_dataset = []
    print("Calculating test errors...")
    for dataset in datasets:
        loss, _ = evaluate_model(model, dataset, loss_funcs)
        losses_per_dataset.append(loss)
    print("All test errors calculated!\n")

    print("Model performance on test sets!\n" + "-"*30 + "\n")
    for i, losses in enumerate(losses_per_dataset):
        dataset_name = datasets[i].mat_file_name
        print("\t" + dataset_name + "\n\t" + "-"*len(dataset_name))
        for name, loss in zip(loss_names, losses):
            print(f"\t{name}: {loss}")
        print()
