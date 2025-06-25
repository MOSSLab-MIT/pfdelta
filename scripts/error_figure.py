import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

sys.path.append(os.getcwd())

from scripts.error_data import (
    task11_mean, task11_max,
    task12_mean, task12_max,
    task13_mean, task13_max,
    task23_mean, task23_max,
    task41_mean, task41_max,
    task42_mean, task42_max,
)


def flatten_task_dict(task_dict):
    records = []
    for model, scenarios in task_dict.items():
        for scenario, values in scenarios.items():
            for val in values:
                if model == "CANOS":
                    model_name = model + "-PF"
                elif model == "GNS":
                    model_name = model + "-S"
                else:
                    model_name = model
                records.append({
                    "Model": model_name,
                    "Test set": scenario,
                    "Power Balance Loss": val
                })
    return pd.DataFrame(records)

def makeplot(df, loss_type, taskname):
    # Change font and size
    plt.rcParams['font.family'] = 'Arial'

    # Change color palette
    if loss_type == "mean":
        custom_colors = ["#D8BFD8", "#FFB6C1", "#ADD8E6"]  # adjust to taste
    else:
        custom_colors = ["#F0A8F0", "#FF99AD", "#83D3FB"]
    model_names = df["Model"].unique()
    palette = dict(zip(model_names, custom_colors))

    have_legend = (taskname == 1.3 or taskname == 4.2)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        x="Test set",
        y="Power Balance Loss",
        hue="Model",
        data=df,
        palette=palette,
        linecolor="black", 
        legend=have_legend,
    )

    # Format
    plt.xlabel("")
    plt.ylabel("Power Balance Loss", fontsize=20)
    if have_legend:
        plt.legend(fontsize=20, title_fontsize=20, loc="upper right")
    ax.set_xticklabels(["N", "N-1", "N-2", "Close-to-Inf."], fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(axis="y", linestyle="-.")
    if loss_type == "mean":
        plt.yticks([0., 0.8, 1.6, 2.4, 3.2])
        plt.ylim([0., 3.2])
    else:
        plt.ylim([1, 1500])
        plt.yscale("log")

    # plt.show()
    plt.savefig(f"{taskname}__{loss_type}.png", dpi=600, format="png")

# df = flatten_task_dict(task41_mean)
# makeplot(df, "mean", 4.2)

for data, taskname in zip(
    [task11_max, task12_max, task13_max, task23_max, task41_max, task42_max],
    [1.1, 1.2, 1.3, 2.3, 4.1, 4.2]
):
    df = flatten_task_dict(data)
    makeplot(df, "max", taskname)

for data, taskname in zip(
    [task11_mean, task12_mean, task13_mean, task23_mean, task41_mean, task42_mean],
    [1.1, 1.2, 1.3, 2.3, 4.1, 4.2]
):
    df = flatten_task_dict(data)
    makeplot(df, "mean", taskname)
