import os
import json
import torch

cases = ["case500"]
topologies = ["none", "n-1", "n-2"]
runs = [1, 2, 3]
types = ["normal"]
root_dir = "inference_results_NR_pt2"

results = {}

for case in cases:
    print(f"\n=== {case.upper()} ===")

    run_means = []

    for run in runs:
        run_runtimes = []

        for topo in topologies:
            for t in types:
                path = os.path.join(root_dir, case, topo, t, f"run_{run}", "runtime_NR_test.json")
                if not os.path.isfile(path):
                    continue

                with open(path) as f:
                    data = json.load(f)

                for entry in data:
                    if entry["converged"]:
                        run_runtimes.append(entry["solve_time"])

        if run_runtimes:
            run_tensor = torch.tensor(run_runtimes, dtype=torch.float32)
            run_mean = run_tensor.mean()
            run_means.append(run_mean)
            print(f"  Run {run}: {len(run_runtimes)} converged samples → mean = {run_mean.item():.4f} s")
        else:
            print(f"  Run {run}: No converged samples.")

    if run_means:
        means_tensor = torch.stack(run_means)
        overall_mean = means_tensor.mean().item()
        overall_std = means_tensor.std(unbiased=False).item()

        results[case] = {
            "run_means": [x.item() for x in means_tensor],
            "overall_mean": overall_mean,
            "overall_std": overall_std,
        }

        print(f"\n  Summary for {case}:")
        print(f"    Mean of run means: {overall_mean:.4f} s")
        print(f"    Std deviation:     {overall_std:.4f} s")
    else:
        print(f"\n  No converged samples found for {case}")
