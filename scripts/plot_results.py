import os
import glob
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


parser = argparse.ArgumentParser(
    description="Plot error values for a specific run and error key.")
parser.add_argument('--run_name', type=str, required=True,
    help="Name of the run to plot errors for.")
parser.add_argument('--error', type=str, default="_first_one",
    help="Error being plotted.")
parser.add_argument('--log', action="store_true", default=False,
    help="Change scale to log.")
args = parser.parse_args()


def find_run_folder(run_name):
    """Searches for the folder in 'runs' using glob and ensures the name is
    unique."""
    # Assuming the root folder is "runs"
    run_pattern = os.path.join("runs", "**", run_name)
    matching_folders = glob.glob(run_pattern, recursive=True)

    if len(matching_folders) == 0:
        print(f"Error: No folder named '{run_name}' found in 'runs'.")
        return None
    elif len(matching_folders) > 1:
        print(f"Error: Multiple folders with the name '{run_name}' found." \
            + "Please ensure the name is unique.")
        return None
    else:
        return matching_folders[0]  # Return the unique matching folder

def plot_errors(run_folder, error_key):
    """Loads the train.json file and plots the errors for the given run and
    error key."""
    max_ticks = 15

    # Build the path to the train.json file
    train_path = os.path.join(run_folder, 'train.json')
    with open(train_path, 'r') as f:
        train_data = json.load(f)

    val_path = os.path.join(run_folder, 'val.json')
    with open(val_path, 'r') as f:
        val_data = json.load(f)

    summary_path = os.path.join(run_folder, 'summary.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    if error_key == "_first_one":
        first_epoch = list(train_data.keys())[0]
        error_key = list(train_data[first_epoch].keys())[0]

    # Set up figure
    plt.figure(figsize=(10, 6))
    plt.xlabel('Training point')
    plt.ylabel(f'{error_key}')
    plt.title(f'{run_name} - {error_key}')

    ## Train values
    # Extract errors for each epoch
    epochs = sorted(train_data.keys(), key=int)  # Sort epochs numerically
    highest_epoch = int(epochs[-1]) + 1
    right_gaps = len(epochs) // max_ticks
    epochs = epochs[::right_gaps] + [epochs[-1]]
    # Plot train errors
    errors = [train_data[epoch].get(error_key, None) for epoch in epochs]
    plt.plot(epochs, errors, marker='o', linestyle='-', color='b', label="Train")

    ## Val values
    epochs = sorted(val_data.keys(), key=int)  # Sort epochs numerically
    highest_epoch = int(epochs[-1]) + 1
    right_gaps = len(epochs) // max_ticks
    epochs = epochs[::right_gaps] + [epochs[-1]]
    # Plot val errors
    num_vals = len(val_data[epochs[0]])
    for i in range(num_vals):
        errors = [val_data[epoch][i].get(error_key, None) for epoch in epochs]
        plt.plot(epochs, errors, marker='o', linestyle='-', label=f"Val {i}")

    # Change to logscale if needed
    if args.log:
        plt.yscale('log')

    # Print summary
    print(json.dumps(summary, indent=3))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Gather arguments
    run_name = args.run_name
    error_name = args.error

    # Find path to run
    run_path = find_run_folder(run_name)
    assert run_path is not None, "Run not found!"
    print(f"Run name found in path: {run_path}")

    # Plot the errors for the given run and error key
    plot_errors(run_path, error_name)

