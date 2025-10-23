import os
import sys
import json
import argparse
import re
import numpy as np


# Change working directory to one above
sys.path.append(os.getcwd())

def find_summary_error(root_folder, error_key):
    """Traverses the root folder, outputs first validation error found in summary.json."""
    # Traverse the root directory
    # summary_og_path = os.path.join(root_folder, 'summary.json')
    filename = "summary.json"
    pattern = re.compile(rf"^{root_folder}.*$")
    outer_root = "./runs/runs/graph_conv_task_1_3"
    for root, dirs, files in os.walk(outer_root):
        # Check for summary.json file in each run folder
        if (pattern.match(root) != None) and ('summary.json' in files):
            summary_path = os.path.join(root, 'summary.json')  
            try:
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)

              
                summary_val = summary_data["val"][0]
                return summary_val[error_key]
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading {summary_path}: {e}")
                return


if __name__ == "__main__":
    # Parse command line arguments
    # args = parse_arguments()
    root = "./runs/runs/graph_conv_task_1_3"
    model_hd_layers = [(512, 16), (1024,5)]
    best_by_error = "bestByMSE"
    model_name = "graphconv"

    lr = ["1e-3", "1e-4", "5e-4"]
    seeds = [11, 22, 33]

    results = dict()
    for i in range(len(model_hd_layers)):
        lr_dict = dict()
        results[model_hd_layers[i]] = lr_dict
        for j in range(len(lr)):
            lr_dict[lr[j]] = []
            for k in range(len(seeds)):
                run_folder = root + f"/graphconv_hd_{model_hd_layers[i][0]}_layers_{model_hd_layers[i][1]}_seed{seeds[k]}_lr{lr[j]}_{best_by_error}"
                summary_error = find_summary_error(run_folder, error_key="MSELoss")
                # print("SUMMARY_ERROR: ", summary_error)
                lr_dict[lr[j]].append(summary_error)

    averages = {}
    for key, lr_dict in results.items():
        averages[key] = {}
        for lr, values in lr_dict.items():
            arr = np.array(values)
            averages[key][lr] = {
                'mean': np.mean(arr),
                'std': np.std(arr, ddof=1)  # sample std (like statistics.stdev)
            }

    # Print results neatly
    for key, lr_stats in averages.items():
        print(f"\nConfig {key}:")
        for lr, stats_dict in lr_stats.items():
            print(f"  {lr}: mean = {stats_dict['mean']:.6f}, std = {stats_dict['std']:.6f}")
