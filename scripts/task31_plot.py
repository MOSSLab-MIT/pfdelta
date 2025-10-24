import json
import argparse
import copy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

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
    models = ["CANOS", "GNS", "PFNet"]
    final_errors = {}
    final_times = {}
    for model in models:
        root = root_to_task[model]
        final_errors[model] = {}
        final_times[model] = {}

        cases = errors[root].keys()
        for case_name in cases:
            per_type_errors = errors[root][case_name]["PBL Mean"]
            per_type_times = times[root][case_name]
            all_values_errors = []
            all_values_times = []
            for (value_type, weight) in zip(
                ["n", "n-1", "n-2", "close2inf"],
                [2000, 2000, 2000, 600]
            ):
                # Errors
                values = torch.tensor(per_type_errors[value_type]) * weight
                all_values_errors.append(values)

                values = torch.tensor(per_type_times[value_type]) * weight
                all_values_times.append(values)

            all_values = torch.stack(all_values_errors)
            final_errors[model][case_name] = (all_values.sum(dim=0) / 6600).tolist()

            all_values = torch.stack(all_values_times)
            final_times[model][case_name] = (all_values.sum(dim=0) / 6600).tolist()

    # Transform all the data into a dataframe
    rows = []
    for model, datasets in final_errors.items():
        for dataset, error_list in datasets.items():
            time_list = final_times[model][dataset]
            for err, t in zip(error_list, time_list):
                rows.append({
                    "Model": model,
                    "Dataset": dataset,
                    "Error": err,
                    "Time": t
                })

    ## ADD NEWTON RAPHSON RESULTS
    # Run times
    case57_runtimes = pd.read_csv(
        "/mnt/home/donti-group-shared/pfdelta_neurips/runtimes_results/runtimes_case57_both.csv")
    case118_runtimes = pd.read_csv(
        "/mnt/home/donti-group-shared/pfdelta_neurips/runtimes_results/runtimes_case118_both.csv")
    case500_runtimes = pd.read_csv(
        "/mnt/home/donti-group-shared/pfdelta_neurips/runtimes_results/runtimes_case500_both.csv")
    # PBL
    case57_pbl = pd.read_csv(
        "/mnt/home/donti-group-shared/pfdelta_neurips/runtimes_results/results_PBL_case57.csv")
    case118_pbl = pd.read_csv(
        "/mnt/home/donti-group-shared/pfdelta_neurips/runtimes_results/results_PBL_case500.csv")
    case500_pbl = pd.read_csv(
        "/mnt/home/donti-group-shared/pfdelta_neurips/runtimes_results/results_PBL_case500.csv")

    # Add them to rows
    for case_times, case_err, dataset in zip([
        case57_runtimes,
        case118_runtimes,
        case500_runtimes,
    ], [
       case57_pbl,
       case118_pbl,
       case500_pbl,
    ], [
        "case57",
        "case118",
        "case500"
    ]
    ):
        only_converged = case_times[case_times["converged"] == True]
        case_err.rename(columns={"topo": "topology_perturb"}, inplace=True)
        combination = pd.merge(
            case_err,
            only_converged,
            on=["sample_idx", "run", "sample_type", "topology_perturb"],
            how="inner"
        )
        for row in combination.iterrows():
            row = row[1]
            rows.append({
                "Model": "NR",
                "Dataset": dataset,
                "Error": row["pbl_mean"],
                "Time": row["solve_time"]
            })

    data = pd.DataFrame(rows)

    # MAKE FIGURE
    datasets = ["case57", "case118", "case500"]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = axes.flatten()
    for i, dataset in enumerate(datasets):
        ax1 = axes[i]
        subset = data[data["Dataset"] == dataset]

        # Left y-axis: Time
        sns.barplot(
            x="Model",
            y="Time",
            data=subset,
            ax=ax1,
            color="skyblue",
            width=0.3,
            label="Time",
            alpha=0.7
        )

        # Right y-axis: Error
        ax2 = ax1.twinx()
        sns.barplot(
            x="Model",
            y="Error",
            data=subset,
            ax=ax2,
            color="red",
            width=0.3,
            label="Error",
            alpha=0.7
        )

        shift = 0.15
        # Shift bars
        for bars in ax1.containers:
            for bar in bars:
                bar.set_x(bar.get_x() - shift)
        for bars in ax2.containers:
            for bar in bars:
                bar.set_x(bar.get_x() + shift)
        # Shift error lines
        for line in ax1.lines:
            line.set_xdata(line.get_xdata() - shift)
        for line in ax2.lines:
            line.set_xdata(line.get_xdata() + shift)


        # Fix ax1 labels
        ax1.set_title(dataset)
        ax1.set_ylim(0, 0.08)
        if i > 0:
            ax1.set_ylabel("")
            # ax1.set_yticks([])
            ax1.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
            ax1.tick_params(left=False, labelleft=False)
        else:
            ax1.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
        ax1.yaxis.grid(True)

        # Fix ax2 labels
        ax2.set_yscale("log")
        ax2.set_ylim(1e-2, 150)
        if i == len(datasets) - 1:
            ax2.set_yticks([1e-2, 1e-1, 1e-0, 1e+1, 1e+2]) # 1e+2])
        else:
            ax2.set_ylabel("")
            ax2.set_yticks([])
            # ax2.set_yticks([1e-2, 1e-1, 1e-0, 1e+1, 1e+2]) # 1e+2])
            # ax2.tick_params(left=False, labelleft=False)
        # ax2.yaxis.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig("figures/task31_test.svg")
    # plt.show()
