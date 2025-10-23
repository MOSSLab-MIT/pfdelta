#!/usr/bin/env python
import pdb
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import argparse
from typing import Any, Dict, Optional
from collections import defaultdict
from torch_geometric.data import HeteroData
import pandas as pd

import torch
from tqdm import tqdm


import sys
print(f"Python path: {sys.path}")

# ---- Repo imports ----
from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.utils.pf_losses_utils import PowerBalanceLoss


TOPOLOGIES = ["n", "n-1", "n-2"]
FNAME_RE = re.compile(r"solved_sample[_-](\d+)\.json$")

def build_heterodata(
    pm_case: Dict[str, Any], is_cpf_sample: bool = False
    ) -> HeteroData:
        """Convert a parsed PowerModels JSON dict into a HeteroData object.

        Args:
            pm_case (dict): PowerModels network data dictionary stored in a JSON case. For samples produced by continuation power flow (CPF) samples the
                dictionary structure is expected to include "solved_net"
                with the solved network; for raw PowerModels cases the
                structure is expected to contain "network" and
                "solution" keys.
            is_cpf_sample (bool): if True, treat pm_case as CPF sample
                (solved_net present) and adapt assertions/checks.

        Returns:
            torch_geometric.data.HeteroData: heterogenous graph with node
            sets ("bus", "gen", "load", optionally "PV","PQ","slack") and
            edge types ("bus","branch","bus"), ("gen","gen_link","bus"),
            ("load","load_link","bus") plus attributes and labels.
            The structure is further detailed in the README.
        """
        data = HeteroData()

        if not is_cpf_sample:
            # If sample was generated via CPF, solved_net contains both the network and solution information in one dict
            network_data = pm_case
            solution_data = pm_case

        # Bus nodes
        pf_x, pf_y = [], []
        bus_voltages = []
        bus_type = []
        bus_shunts = []
        bus_gen, bus_demand = [], []
        voltage_limits = []

        PQ_bus_x, PQ_bus_y = [], []
        PV_bus_x, PV_bus_y = [], []
        PV_demand, PV_generation = [], []
        slack_x, slack_y = [], []
        slack_demand, slack_generation = [], []
        PV_to_bus, PQ_to_bus, slack_to_bus = [], [], []
        pq_idx, pv_idx, slack_idx = 0, 0, 0

        for bus_id_str, bus in sorted(
            network_data["bus"].items(), key=lambda x: int(x[0])
        ):
            bus_id = int(bus_id_str)
            bus_idx = bus_id - 1  # PowerModels uses 1-based indexing
            bus_sol = solution_data["bus"][bus_id_str]

            va, vm = bus_sol["va"], bus_sol["vm"]
            bus_voltages.append(torch.tensor([va, vm]))
            vmin, vmax = bus["vmin"], bus["vmax"]
            voltage_limits.append(torch.tensor([vmin, vmax]))

            # Shunts
            gs, bs = 0.0, 0.0
            for shunt in network_data["shunt"].values():
                if int(shunt["shunt_bus"]) == bus_id:
                    gs += shunt["gs"]
                    bs += shunt["bs"]
            bus_shunts.append(torch.tensor([gs, bs]))

            # Load
            pd, qd = 0.0, 0.0
            for load in network_data["load"].values():
                if int(load["load_bus"]) == bus_id:
                    pd += load["pd"]
                    qd += load["qd"]

            bus_demand.append(torch.tensor([pd, qd]))

            # Gen
            pg, qg = 0.0, 0.0
            for gen_id, gen in sorted(
                network_data["gen"].items(), key=lambda x: int(x[0])
            ):
                if int(gen["gen_bus"]) == bus_id:
                    if gen["gen_status"] == 1:
                        gen_sol = solution_data["gen"][gen_id]
                        pg += gen_sol["pg"]
                        qg += gen_sol["qg"]
                    else:
                        if is_cpf_sample:
                            pass
                            # assert gen["pg"] == 0 and gen["qg"] == 0, (
                            #     f"Expected gen {gen_id} to be off"
                            # )
                        else:
                            pass
                            # assert solution_data["gen"].get(gen_id) is None, (
                            #     f"Expected gen {gen_id} to be off."
                            # )

            bus_gen.append(torch.tensor([pg, qg]))

            # Decide final bus type
            bus_type_now = bus["bus_type"]

            if bus_type_now == 2 and pg == 0.0 and qg == 0.0:
                bus_type_now = 1  # PV bus with no gen --> becomes PQ
            bus_type.append(torch.tensor(bus_type_now))

            if bus_type_now == 1:
                pf_x.append(torch.tensor([pd, qd]))
                pf_y.append(torch.tensor([va, vm]))

                PQ_bus_x.append(torch.tensor([pd, qd]))
                PQ_bus_y.append(torch.tensor([va, vm]))
                PQ_to_bus.append(torch.tensor([pq_idx, bus_idx]))
                pq_idx += 1
            elif bus_type_now == 2:
                pf_x.append(torch.tensor([pg - pd, vm]))
                pf_y.append(torch.tensor([qg - qd, va]))

                PV_bus_x.append(torch.tensor([pg - pd, vm]))
                PV_bus_y.append(torch.tensor([qg - qd, va]))
                PV_demand.append(torch.tensor([pd, qd]))
                PV_generation.append(torch.tensor([pg, qg]))
                PV_to_bus.append(torch.tensor([pv_idx, bus_idx]))
                pv_idx += 1
            elif bus_type_now == 3:
                pf_x.append(torch.tensor([va, vm]))
                pf_y.append(torch.tensor([pg - pd, qg - qd]))

                slack_x.append(torch.tensor([va, vm]))
                slack_y.append(torch.tensor([pg - pd, qg - qd]))
                slack_demand.append(torch.tensor([pd, qd]))
                slack_generation.append(torch.tensor([pg, qg]))
                slack_to_bus.append(torch.tensor([slack_idx, bus_idx]))
                slack_idx += 1

        # Generator nodes
        generation, limits, slack_gen = [], [], []

        for gen_id, gen in sorted(network_data["gen"].items(), key=lambda x: int(x[0])):
            if gen["gen_status"] == 1:
                gen_sol = solution_data["gen"][gen_id]
                pmin, pmax, qmin, qmax = (
                    gen["pmin"],
                    gen["pmax"],
                    gen["qmin"],
                    gen["qmax"],
                )
                pgen, qgen = gen_sol["pg"], gen_sol["qg"]
                limits.append(torch.tensor([pmin, pmax, qmin, qmax]))
                generation.append(torch.tensor([pgen, qgen]))
                is_slack = torch.tensor(
                    1
                    if network_data["bus"][str(gen["gen_bus"])]["bus_type"] == 3
                    else 0,
                    dtype=torch.bool,
                )
                slack_gen.append(is_slack)
            else:
                if is_cpf_sample:
                    pass
                    # assert (
                    #     solution_data["gen"][gen_id]["pg"] == 0
                    #     and solution_data["gen"][gen_id]["qg"] == 0
                    # ), f"Expected gen {gen_id} to be off"
                else:
                    pass
                    # assert solution_data["gen"].get(gen_id) is None, (
                    #     f"Expected gen {gen_id} to be off."
                    # )

        # Load nodes
        demand = []
        for load_id, load in sorted(
            network_data["load"].items(), key=lambda x: int(x[0])
        ):
            pd, qd = load["pd"], load["qd"]
            demand.append(torch.tensor([pd, qd]))

        # Edges
        # bus to bus edges
        edge_index, edge_attr, edge_label, edge_limits = [], [], [], []
        for branch_id_str, branch in sorted(
            network_data["branch"].items(), key=lambda x: int(x[0])
        ):
            if branch["br_status"] == 0:
                continue  # Skip inactive branches

            from_bus = int(branch["f_bus"]) - 1
            to_bus = int(branch["t_bus"]) - 1
            edge_index.append(torch.tensor([from_bus, to_bus]))
            edge_attr.append(
                torch.tensor(
                    [
                        branch["br_r"],
                        branch["br_x"],
                        branch["g_fr"],
                        branch["b_fr"],
                        branch["g_to"],
                        branch["b_to"],
                        branch["tap"],
                        branch["shift"],
                    ]
                )
            )

            edge_limits.append(torch.tensor([branch["rate_a"]]))

            branch_sol = solution_data["branch"].get(branch_id_str)
            if branch_sol:
                try:
                    edge_label.append(torch.tensor([
                        branch_sol['pf'], branch_sol['qf'],
                        branch_sol['pt'], branch_sol['qt']
                    ]))
                except Exception:
                    edge_label.append(torch.tensor([0.0, 0.0, 0.0, 0.0]))

        # bus to gen edges
        gen_to_bus_index = []
        for gen_id, gen in sorted(network_data["gen"].items(), key=lambda x: int(x[0])):
            if gen["gen_status"] == 1:
                gen_bus = torch.tensor(gen["gen_bus"]) - 1
                gen_to_bus_index.append(torch.tensor([int(gen_id) - 1, gen_bus]))

        # bus to load edges
        load_to_bus_index = []
        for load_id, load in sorted(
            network_data["load"].items(), key=lambda x: int(x[0])
        ):
            load_bus = torch.tensor(load["load_bus"]) - 1
            load_to_bus_index.append(torch.tensor([int(load_id) - 1, load_bus]))

        # Create graph nodes and edges
        data["bus"].x = torch.stack(pf_x)
        data["bus"].y = torch.stack(pf_y)
        data["bus"].bus_gen = torch.stack(bus_gen)  # aggregated
        data["bus"].bus_demand = torch.stack(bus_demand)  # aggregated
        data["bus"].bus_voltages = torch.stack(bus_voltages)
        data["bus"].bus_type = torch.stack(bus_type)
        data["bus"].shunt = torch.stack(bus_shunts)
        data["bus"].limits = torch.stack(
            voltage_limits
        )  # These correspond to limits in the original pglib case file, not all limits are enforced in our dataset

        data["gen"].limits = torch.stack(
            limits
        )  # These correspond to limits in the original pglib case file, not all limits are enforced in our dataset
        data["gen"].generation = torch.stack(generation)
        data["gen"].slack_gen = torch.stack(slack_gen)

        data["load"].demand = torch.stack(demand)

        for link_name, edges in {
            ("bus", "branch", "bus"): edge_index,
            ("gen", "gen_link", "bus"): gen_to_bus_index,
            ("load", "load_link", "bus"): load_to_bus_index,
        }.items():
            edge_tensor = torch.stack(edges, dim=1)
            data[link_name].edge_index = edge_tensor
            if link_name != ("bus", "branch", "bus"):
                data[
                    (link_name[2], link_name[1], link_name[0])
                ].edge_index = edge_tensor.flip(0)
            if link_name == ("bus", "branch", "bus"):
                data[link_name].edge_attr = torch.stack(edge_attr)
                data[link_name].edge_label = torch.stack(edge_label)
                data[link_name].edge_limits = torch.stack(
                    edge_limits
                )  # These correspond to limits in the original pglib case file, these limits are not enforced in our dataset

        return data


class PowerBalanceAnalysis(PowerBalanceLoss):
    def __init__(self):
        super().__init__("")

    def __call__(self, data: HeteroData) -> float:
        # Physical quantities from ground truth
        V = data["bus"].bus_voltages[:, 1]  # vm
        theta = data["bus"].bus_voltages[:, 0]  # va
        Pnet = data["bus"].bus_gen[:, 0] - data["bus"].bus_demand[:, 0]
        Qnet = data["bus"].bus_gen[:, 1] - data["bus"].bus_demand[:, 1]

        shunt_g = data["bus"].shunt[:, 0]
        shunt_b = data["bus"].shunt[:, 1]

        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = data["bus", "branch", "bus"].edge_attr

        r = edge_attr[:, 0]
        x = edge_attr[:, 1]
        g_fr = edge_attr[:, 2]
        b_fr = edge_attr[:, 3]
        g_to = edge_attr[:, 4]
        b_to = edge_attr[:, 5]
        tau = edge_attr[:, 6]
        theta_shift = edge_attr[:, 7]

        src, dst = edge_index

        Y = 1 / (r + 1j * x)
        Y_real = torch.real(Y)
        Y_imag = torch.imag(Y)

        delta_theta1 = theta[src] - theta[dst]
        delta_theta2 = theta[dst] - theta[src]

        # Active power flow
        P_flow_src = (
            V[src]
            * V[dst]
            / tau
            * (
                -Y_real * torch.cos(delta_theta1 - theta_shift)
                - Y_imag * torch.sin(delta_theta1 - theta_shift)
            )
            + Y_real * (V[src] / tau) ** 2
        )

        P_flow_dst = (
            V[dst]
            * V[src]
            / tau
            * (
                -Y_real * torch.cos(delta_theta2 - theta_shift)
                - Y_imag * torch.sin(delta_theta2 - theta_shift)
            )
            + Y_real * V[dst] ** 2
        )

        # Reactive power flow
        Q_flow_src = (
            V[src]
            * V[dst]
            / tau
            * (
                -Y_real * torch.sin(delta_theta1 - theta_shift)
                + Y_imag * torch.cos(delta_theta1 - theta_shift)
            )
            - (Y_imag + b_fr) * (V[src] / tau) ** 2
        )

        Q_flow_dst = (
            V[dst]
            * V[src]
            / tau
            * (
                -Y_real * torch.sin(delta_theta2 - theta_shift)
                + Y_imag * torch.cos(delta_theta2 - theta_shift)
            )
            - (Y_imag + b_to) * V[dst] ** 2
        )

        # Aggregate to buses
        Pbus_pred = torch.zeros_like(V).scatter_add_(0, src, P_flow_src)
        Pbus_pred = Pbus_pred.scatter_add_(0, dst, P_flow_dst)

        Qbus_pred = torch.zeros_like(V).scatter_add_(0, src, Q_flow_src)
        Qbus_pred = Qbus_pred.scatter_add_(0, dst, Q_flow_dst)

        # Power mismatch
        delta_P = Pnet - Pbus_pred - V**2 * shunt_g
        delta_Q = Qnet - Qbus_pred + V**2 * shunt_b

        # Compute the loss as the sum of squared mismatches
        delta_PQ_2 = delta_P**2 + delta_Q**2

        # Calculate PBL Mean
        delta_PQ_magnitude = torch.sqrt(delta_PQ_2)
        self.power_balance_mean = delta_PQ_magnitude.mean()

        return self.power_balance_mean

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
        results[topo][sample_type][run][sample_idx] = tensor (1D)
    """
    results: Dict[str, Dict[str, Dict[int, Dict[int, torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    pbl = PowerBalanceAnalysis()

    for topo in TOPOLOGIES:
        for sample_type in ["raw", "nose"]:
            for run_num in [1, 2, 3]:
                run_path = os.path.join(
                    root_dir, case_name, topo, sample_type, f"run_{run_num}"
                )
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
                    all_files, desc=f"{case_name}:{topo}:{sample_type}:run_{run_num}", leave=True
                ):
                    sample_idx = _parse_sample_idx(fname)
                    if sample_idx is None:
                        continue

                    file_path = os.path.join(run_path, fname)
                    with open(file_path, "r") as f:
                        pm_case = json.load(f)

                    data = build_heterodata(pm_case, is_cpf_sample=False) # note that we're just saving the updated net here.
                    with torch.no_grad():
                        pbl_out = pbl(data)

                    # TODO: this will potentially not be needed, pbl_out could be enough
                    vec = _to_1d_tensor(pbl_out)
                    if vec.numel() == 0:
                        continue

                    results[topo][sample_type][run_num][sample_idx] = vec

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
    results_csv = os.path.join(out_dir, f"results_PBL_{case_name}.csv")
    summary_csv = os.path.join(out_dir, f"summary_PBL_{case_name}.csv")

    # ---- Build per-sample dataframe ----
    rows = []
    for topo, sample_types in results.items():
        for sample_type, runs in sample_types.items():
            for run_num, samples in runs.items():
                for sample_idx, vec in samples.items():
                    pbl_mean = float(vec.mean().item())
                    pbl_max = float(vec.max().item())
                    rows.append(
                        {
                        "topo": topo,
                        "run": run_num,
                        "sample_idx": sample_idx,
                        "sample_type": sample_type,
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
        default="runtimes_results",
        help="Root directory where results are stored",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="runtimes_results",
        help="Output directory for CSVs. Default: <root_dir>/<case_name>/stats",
    )
    return p.parse_args()

def main():
    args = parse_args()
    results = get_pbl_results(case_name=args.case_name, root_dir=args.root_dir)
    compute_statistics(results, out_dir=args.out_dir, case_name=args.case_name)


if __name__ == "__main__":
    main()
