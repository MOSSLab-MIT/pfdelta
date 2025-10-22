import json
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


root_to_task = {
    ("CANOS", 1.1): "runs/canos_task_1_1",
    ("CANOS", 1.2): "runs/canos_task_1_2",
    ("CANOS", 1.3): "runs/canos_task_1_3",
    ("CANOS", 2.3): "runs/canos_task_2_3",
    ("CANOS", 4.1): "runs/canos_task_4_1",
    ("CANOS", 4.2): "runs/canos_task_4_2",
    ("CANOS", 4.3): "runs/canos_task_4_3",
    ("GNS", 1.1): "runs/gns_task_1_1",
    ("GNS", 1.2): "runs/gns_task_1_2",
    ("GNS", 1.3): "runs/gns_task_1_3",
    ("GNS", 2.3): "runs/gns_task_2_3",
    ("GNS", 4.1): "runs/gns_task_4_1",
    ("GNS", 4.2): "runs/gns_task_4_2",
    ("GNS", 4.3): "runs/gns_task_4_3",
    ("PFNet", 1.1): "runs/pfnet_task_1_1",
    ("PFNet", 1.2): "runs/pfnet_task_1_2",
    ("PFNet", 1.3): "runs/pfnet_task_1_3",
    ("PFNet", 2.3): "runs/pfnet_task_2_3",
    ("PFNet", 4.1): "runs/pfnet_task_4_1",
    ("PFNet", 4.2): "runs/pfnet_task_4_2",
    ("PFNet", 4.3): "runs/pfnet_task_4_3",
}

def dict_to_df(d, model_name):
    rows = []
    for key, values in d.items():
        for v in values:
            rows.append({
                'Grid Attribute': key,
                'Power Balance Loss (Mean)': v,
                'Model': model_name
            })
    return pd.DataFrame(rows)

def parser():
    parser = argparse.ArgumentParser(
        description="Plots the results formed in scripts/test_error.py"
    )
    parser.add_argument(
        "--row",
        type=int,
        required=True,
        help="Picks which task you want to plot"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parser()
    row = args.row

    # Loads results
    with open("test_errors_p_seeds.json", "r") as f:
        all_results = json.load(f)

    palette = {
        "CANOS": "#D9A6D9",
        "GNS": "#F27AA4", # "#F2A6B5",
        "PFNet": "#69D8FF"
    }
    if row == 1:
        tasks = [1.1, 1.2, 1.3, 2.3]
    else:
        tasks = [4.1, 4.2, 4.3]

    n = len(tasks)
    fontsize=20
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    for i, task in enumerate(tasks):
        ax = axes[i]
        root_name_canos = root_to_task[("CANOS", task)]
        root_name_gns = root_to_task[("GNS", task)]
        root_name_pfnet = root_to_task[("PFNet", task)]
        results = {
            "CANOS": all_results[root_name_canos]["PBL Mean"],
            "GNS": all_results[root_name_gns]["PBL Mean"],
            "PFNet": all_results[root_name_pfnet]["PBL Mean"]
        }
        results = pd.concat([
            dict_to_df(results["CANOS"], "CANOS"),
            dict_to_df(results["GNS"], "GNS"),
            dict_to_df(results["PFNet"], "PFNet"),
        ])
        sns.boxplot(
            data=results,
            x="Grid Attribute",
            y="Power Balance Loss (Mean)",
            hue="Model",
            ax=ax,
            palette=palette,
            boxprops=dict(edgecolor='white')
        )

        ax.set_xlabel("")
        ax.set_xticklabels(
            ["N", "N-1", "N-2", "C2I"],
            fontsize=fontsize
        )

        ax.tick_params(axis="both", labelsize=fontsize,)
        ax.set_ylabel(
            "Power Balance Loss (Mean)",
            fontsize=fontsize,
            fontweight="bold"
        )

        ax.yaxis.grid(True)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_ylim(0, 2.0)

        ax.get_legend().remove()
        ax.set_title(
            f"Task {task}",
            fontsize=fontsize,
            fontweight="bold"
        )

    if row == 1:
        ax.legend(
            title="Model",
            loc="upper left",
            fontsize=fontsize-3,
            title_fontsize=fontsize-3
        )
    else:
        ax.legend(
            title="Model",
            loc="lower right",
            fontsize=fontsize-3,
            title_fontsize=fontsize-3
        )
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(
        #     handles, labels,
        #     title="Model",
        #     fontsize=fontsize-3,
        #     title_fontsize=fontsize-3,
        #     loc='center right'  # or 'lower center', 'right', etc.
        # )

    plt.subplots_adjust(wspace=0.05)
    plt.savefig("figures/test.svg")
