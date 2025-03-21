import os
import json
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Find the run with the lowest average error.")
    parser.add_argument('--root', type=str, default="",
        help="Folder from which the best run is found.")
    parser.add_argument('--error', type=str, required=True,
        help="Error to gather from runs.")
    args = parser.parse_args()

    return args

def calculate_average(error_key, val_list):
    """Calculates the average of the specified error_key in the val list."""
    val_errors = [errors[error_key] for errors in val_list]
    val_error = sum(val_errors) / len(val_list)

    return val_error

def find_best_run(root_folder, error_key):
    """Traverses the root folder, calculates the average error, and finds the
    run with the lowest average error."""
    lowest_avg = float('inf')
    best_run_path = None

    # Traverse the root directory
    for root, dirs, files in os.walk(root_folder):
        # Check for summary.json file in each run folder
        if 'summary.json' in files:
            summary_path = os.path.join(root, 'summary.json')
            try:
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)

                val_list = summary_data['val']
                avg_error = calculate_average(error_key, val_list)

                # If the current run has a lower average error, update the best run
                if avg_error < lowest_avg:
                    lowest_avg = avg_error
                    best_run_path = root
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {summary_path}: {e}")
    
    return best_run_path, lowest_avg


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Gather arguments
    root = os.path.join("runs", args.root)
    error_name = args.error

    # Find the run with the lowest average error
    best_run, lowest_avg = find_best_run(root, error_name)
    
    if best_run:
        print(f"The run with the lowest average error is: {best_run}")
        print(f"Lowest average error: {lowest_avg}")
    else:
        print("No valid runs found.")

