#!/usr/bin/env python
import os
import re
import json
import argparse
from typing import Any, Dict, Optional
from collections import defaultdict
import pandas as pd

import torch
from tqdm import tqdm

# ---- Repo imports ----
from core.datasets.pfdelta_dataset import PFDeltaDataset

try:
    from core.utils.pf_losses_utils import PowerBalanceLoss
except Exception as e:
    raise ImportError(
        "Could not import PowerBalanceLoss from core.utils.pf_losses_utils. "
        "Please verify the module path or adjust the import in this script."
    ) from e


TOPOLOGIES = ["none", "n-1", "n-2"]
RUNS = [1, 2, 3]
FNAME_RE = re.compile(r"solved_sample[_-](\d+)\.json$")


def _parse_sample_idx(filename: str) -> Optional[int]:
    m = FNAME_RE.search(filename)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _to_1d_tensor(x: Any) -> torch.Tensor:
    """Coerce PBL output to a 1D float32 tensor (no grad)."""
    if isinstance(x, torch.Tensor):
        return x.detach().flatten().to(dtype=torch.float32)
    if isinstance(x, (float, int)):
        return torch.tensor([float(x)], dtype=torch.float32)
    if isinstance(x, (list, tuple)):
        parts = []
        for el in x:
            if isinstance(el, torch.Tensor):
                parts.append(el.detach().flatten().to(dtype=torch.float32))
            elif isinstance(el, (float, int)):
                parts.append(torch.tensor([float(el)], dtype=torch.float32))
            elif isinstance(el, (list, tuple)):
                parts.append(_to_1d_tensor(el))
            else:
                try:
                    parts.append(torch.as_tensor(el, dtype=torch.float32).flatten())
                except Exception:
                    pass
        if parts:
            return torch.cat(parts)
        return torch.as_tensor(x, dtype=torch.float32).flatten()
    return torch.as_tensor(x, dtype=torch.float32).flatten()


def get_pbl_results(
    case_name: str,
    root_dir: str,
) -> Dict[str, Dict[int, Dict[int, torch.Tensor]]]:
    """
    Returns:
        results[topo][run][sample_idx] = tensor (1D)
    """
    results: Dict[str, Dict[int, Dict[int, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    pbl = PowerBalanceLoss()

    for topo in TOPOLOGIES:
        for run in RUNS:
            run_path = os.path.join(
                root_dir, case_name, topo, "raw", f"run_{run}"
            )  # TODO: Check if raw actually has to be here?
            if not os.path.exists(run_path):
                print(f"Warning: path does not exist: {run_path}")
                continue

            all_files = sorted(
                [
                    f
                    for f in os.listdir(run_path)
                    if f.endswith(".json") and f.startswith("solved_sample")
                ]
            )
            if not all_files:
                print(f"Warning: no solved_sample JSON files found in {run_path}")
                continue

            for fname in tqdm(
                all_files, desc=f"{case_name}:{topo}:run_{run}", leave=False
            ):
                sample_idx = _parse_sample_idx(fname)
                if sample_idx is None:
                    continue

                file_path = os.path.join(run_path, fname)
                with open(file_path, "r") as f:
                    pm_case = json.load(f)

                data = PFDeltaDataset.build_heterodata(pm_case, is_cpf_sample=False)
                with torch.no_grad():
                    pbl_out = pbl(data)

                # TODO: this will potentially not be needed, pbl_out could be enough
                vec = _to_1d_tensor(pbl_out)
                if vec.numel() == 0:
                    continue

                results[topo][run][sample_idx] = vec

    return results


def compute_statistics(
    results: Dict[str, Dict[int, Dict[int, torch.Tensor]]], out_dir: str, case_name: str
) -> None:
    """
    Writes two CSVs:
      1) results_<case>.csv with columns:
         topo, run, sample_idx, pbl_mean, pbl_max
      2) summary_<case>.csv with columns:
         topo, run, mean_pbl_mean, mean_pbl_max, n_samples
    """
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, f"results_{case_name}.csv")
    summary_csv = os.path.join(out_dir, f"summary_{case_name}.csv")

    # ---- Build per-sample dataframe ----
    rows = []
    for topo, runs in results.items():
        for run, samples in runs.items():
            for sample_idx, vec in samples.items():
                pbl_mean = float(vec.mean().item())
                pbl_max = float(vec.max().item())
                rows.append(
                    {
                        "topo": topo,
                        "run": run,
                        "sample_idx": sample_idx,
                        "pbl_mean": pbl_mean,
                        "pbl_max": pbl_max,
                    }
                )

    df_results = pd.DataFrame(rows)
    df_results.sort_values(by=["topo", "run", "sample_idx"], inplace=True)
    df_results.to_csv(results_csv, index=False)

    # ---- Build summary dataframe ----
    df_summary = (
        df_results.groupby(["run"])
        .agg(
            mean_pbl_mean=("pbl_mean", "mean"),
            mean_pbl_max=("pbl_max", "mean"),
            n_samples=("sample_idx", "count"),
        )
        .reset_index()
    )

    df_summary.to_csv(summary_csv, index=False)

    print(f"[✔] Saved results to {results_csv}")
    print(f"[✔] Saved summary to {summary_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate Power Balance Loss (PBL) results."
    )
    p.add_argument("case_name", type=str, help="Case name (e.g., case500)")
    p.add_argument(
        "--root_dir",
        type=str,
        default="runtime_results",
        help="Root directory where results are stored",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="runtime_results",
        help="Output directory for CSVs. Default: <root_dir>/<case_name>/stats",
    )
    return p.parse_args()


def main():
    args = parse_args()
    results = get_pbl_results(case_name=args.case_name, root_dir=args.root_dir)
    compute_statistics(results, out_dir=args.out_dir, case_name=args.case_name)


if __name__ == "__main__":
    main()
