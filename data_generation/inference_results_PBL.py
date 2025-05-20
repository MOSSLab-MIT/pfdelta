import os
import json
import torch
from tqdm import tqdm
from classes_for_analysis import PFDeltaDataset, PowerBalanceLoss

root_dir = "inference_results_NR_pt2"
case_name = "case500"
topologies = ["none", "n-1", "n-2"]
runs = [1, 2, 3]
data_types = ["normal"]

results = {}

for data_type in data_types:
    print(f"\nEvaluating PBL for data_type = '{data_type}'")

    if data_type == "normal":
        run_means_across_topos = []

        for run in tqdm(runs, desc="    Runs (normal aggregated)", position=0):
            all_sample_pbls = []

            for topo in topologies:
                run_path = os.path.join(root_dir, case_name, topo, data_type, f"run_{run}")
                print(f"\n    {topo}/run_{run}: {run_path}")

                file_list = sorted(fname for fname in os.listdir(run_path)
                                if fname.endswith(".json") and fname.startswith("solved_sample"))
                for fname in tqdm(file_list, desc=f"        Samples", leave=False, position=1):
                    with open(os.path.join(run_path, fname)) as f:
                        pm_case = json.load(f)
                    data = PFDeltaDataset.build_heterodata(None, pm_case, feasibility=True)
                    pbl_tensor = PowerBalanceLoss()(data)
                    pbl_scalar = pbl_tensor.mean().item()
                    all_sample_pbls.append(pbl_scalar)

            if all_sample_pbls:
                run_tensor = torch.tensor(all_sample_pbls, dtype=torch.float32)
                run_mean = run_tensor.mean()
                run_means_across_topos.append(run_mean)
                print(f"      Run {run} mean PBL across all topologies: {run_mean.item():.4e}")
            else:
                print(f"      No data found in run {run}")

        if run_means_across_topos:
            means_tensor = torch.stack(run_means_across_topos)
            overall_mean = means_tensor.mean()
            overall_std = means_tensor.std(unbiased=False)

            results[(data_type, "all")] = {
                "run_means": means_tensor,
                "overall_mean": overall_mean.item(),
                "overall_std": overall_std.item()
            }

            print(f"\n  Summary for 'normal' (all topologies):")
            print(f"    Mean of run means:  {overall_mean.item():.4e}")
            print(f"    Std deviation:      {overall_std.item():.4e}")
        else:
            print("  No PBL data found for 'normal'")

    elif data_type == "nose":
        run_pbl_means = []
        run_pbl_maxes = []
        run_delta_P_maxes = []
        run_delta_Q_maxes = []
        for run in tqdm(runs, desc="    Runs (hard)", position=0):
            run_sample_pbls = []

            for topo in tqdm(topologies, desc=f"      Topologies", leave=False, position=1):
                run_path = os.path.join(root_dir, case_name, topo, data_type, f"run_{run}")
                print(f"\n    {topo}/run_{run}: {run_path}")

                file_list = sorted(fname for fname in os.listdir(run_path) if fname.endswith(".json") and fname.startswith("solved_sample"))
                for fname in tqdm(file_list, desc=f"        Samples", leave=False, position=2):
                    with open(os.path.join(run_path, fname)) as f:
                        pm_case = json.load(f)
                    data = PFDeltaDataset.build_heterodata(None, pm_case, feasibility=True)
                    pbl_tensor, delta_P_abs, delta_Q_abs = PowerBalanceLoss()(data)
                    pbl_scalar = pbl_tensor.mean().item()
                    run_sample_pbls.append(pbl_scalar)
                    run_delta_P_maxes.append((delta_P_abs.item(), fname))
                    run_delta_Q_maxes.append((delta_Q_abs.item(), fname))

            if run_sample_pbls:
                run_tensor = torch.tensor(run_sample_pbls, dtype=torch.float32)
                run_mean = run_tensor.mean()
                run_max = run_tensor.max()
                run_pbl_means.append(run_mean)
                run_pbl_maxes.append(run_max)
                print(f"      Run {run} mean PBL: {run_mean.item():.4e}")
                print(f"      Run {run} max PBL:  {run_max.item():.4e}")
                max_P_val, max_P_file = max(run_delta_P_maxes, key=lambda x: x[0])
                max_Q_val, max_Q_file = max(run_delta_Q_maxes, key=lambda x: x[0])

                print(f"      Max ΔP in run {run}: {max_P_val:.4e} from {max_P_file}")
                print(f"      Max ΔQ in run {run}: {max_Q_val:.4e} from {max_Q_file}")
            else:
                print(f"      No data found in run {run}")

        if run_pbl_means:
            run_means_tensor = torch.stack(run_pbl_means)
            run_maxes_tensor = torch.stack(run_pbl_maxes)

            overall_mean = run_means_tensor.mean()
            overall_std = run_means_tensor.std(unbiased=False)
            max_of_run_means = run_means_tensor.max()

            mean_of_run_maxes = run_maxes_tensor.mean()
            max_of_run_maxes = run_maxes_tensor.max()

            results[(data_type, "all")] = {
                "run_means": run_means_tensor,
                "run_maxes": run_maxes_tensor,
                "overall_mean": overall_mean.item(),
                "overall_std": overall_std.item(),
                "max_of_run_means": max_of_run_means.item(),
                "mean_of_run_maxes": mean_of_run_maxes.item(),
                "max_of_run_maxes": max_of_run_maxes.item()
            }

            print(f"\n  Summary for 'hard' (all topologies):")
            print(f"    Mean of run means:    {overall_mean.item():.4e}")
            print(f"    Std deviation:        {overall_std.item():.4e}")
            print(f"    Max of run means:     {max_of_run_means.item():.4e}")
            print(f"    Mean of run maxes:    {mean_of_run_maxes.item():.4e}")
            print(f"    Max of run maxes:     {max_of_run_maxes.item():.4e}")
        else:
            print("  No PBL data found for 'hard'")
