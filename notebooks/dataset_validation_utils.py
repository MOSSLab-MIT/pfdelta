from typing import List, Tuple
from torch_geometric.data import HeteroData
import torch
import matplotlib.pyplot as plt
import numpy as np
from core.utils.pf_losses_utils import PowerBalanceLoss
import altair as alt
import pandas as pd


# Helper functions to validate contingencies
def _counts_from_heterodata(hd) -> Tuple[int, int]:
    """
    Return (n_branches, n_gens) from a PyG HeteroData sample.

    - Branches: from ('bus','branch','bus').edge_index (num edges)
    - Generators: inferred from any tensor in hd['gen'] store
    """
    rel = ("bus", "branch", "bus")
    if rel not in getattr(hd, "edge_types", []):
        raise ValueError("HeteroData missing ('bus','branch','bus') relation.")
    n_branches = hd[rel]["edge_index"].shape[1]

    n_gens = 0
    if "gen" in getattr(hd, "node_types", []):
        gstore = hd["gen"]
        n_gens = gstore["generation"].shape[0]

    return n_branches, n_gens


def all_samples_have_k_contingencies(
    base_case: "HeteroData",
    samples: List["HeteroData"],
    k: int,
    verbose: bool = True,
) -> bool:
    """
    Assume N–k samples physically remove outaged branches/generators.
    Return True iff every sample has exactly k contingencies, i.e.:

        k == (N_branches - sample_branches) + (N_gens - sample_gens)
    """
    base_br, base_g = _counts_from_heterodata(base_case)

    ok = True
    bad = []

    for i, s in enumerate(samples):
        samp_br, samp_g = _counts_from_heterodata(s)

        delta_br = base_br - samp_br
        delta_g = base_g - samp_g
        cnt = delta_br + delta_g

        if cnt != k:
            ok = False
            bad.append((i, f"found {cnt} (Δbranches={delta_br}, Δgens={delta_g})"))

    if verbose:
        if ok:
            print(
                f"✅ All samples have exactly k={k} contingencies "
                f"(base: branches={base_br}, gens={base_g})."
            )
        else:
            print(
                f"❌ Some samples do not have exactly k={k} contingencies "
                f"(base: branches={base_br}, gens={base_g})."
            )
            for i, msg in bad[:10]:
                print(f"  - sample[{i}]: {msg}")
            if len(bad) > 10:
                print(f"  ... and {len(bad) - 10} more")

    return ok


def add_ids_to_base(base_hd):
    """Attach edge_id (per branch) and gen_id (per generator) to the BASE."""
    rel = ("bus", "branch", "bus")
    E = base_hd[rel]["edge_index"].shape[1]
    base_hd[rel]["edge_id"] = torch.arange(E, dtype=torch.long)

    gstore = base_hd["gen"]
    G = gstore["generation"].shape[0]
    base_hd["gen"]["gen_id"] = torch.arange(G, dtype=torch.long)
    return base_hd


# --- branches: fast order-preserving alignment (delete-only) ---
def present_branch_ids_by_alignment(base_hd, sample_hd, rel=("bus", "branch", "bus")):
    base_ei = base_hd[rel].edge_index.cpu().numpy()  # (2, E_base)
    samp_ei = sample_hd[rel].edge_index.cpu().numpy()  # (2, E_samp)
    E_base, E_samp = base_ei.shape[1], samp_ei.shape[1]
    present = np.zeros(E_base, dtype=bool)
    i = j = 0
    while i < E_base and j < E_samp:
        if base_ei[0, i] == samp_ei[0, j] and base_ei[1, i] == samp_ei[1, j]:
            present[i] = True
            i += 1
            j += 1
        else:
            i += 1
    return present  # length E_base (True=kept, False=outaged)


def present_gen_ids_by_link(base_hd, sample_hd):
    """
    Return a boolean mask (length = #base generators) indicating which base generators
    are present in `sample_hd`, using ONLY the ('gen','gen_link','bus') edges.

    Assumes:
      - The gen->bus link stores the original generator index: (gen_id - 1)
      - Links are included ONLY for active (present) generators in the sample
      - Those indices are within [0, G_base-1]
    """
    # infer G_base from any per-gen tensor in the BASE
    if "gen" not in base_hd.node_types:
        return np.array([], dtype=bool)
    gstore = base_hd["gen"]
    for key in ("generation", "limits", "slack_gen"):
        if key in gstore:
            G_base = gstore[key].shape[0]
            break
    else:
        G_base = 0

    present = np.zeros(G_base, dtype=bool)

    rel = ("gen", "gen_link", "bus")
    if rel not in sample_hd.edge_types:
        # no links in the sample ⇒ everything looks outaged
        return present

    # row 0 of edge_index is generator indices
    gen_idx = sample_hd[rel]["edge_index"][0].cpu().numpy()
    if gen_idx.size == 0:
        return present

    # mark those generators as present (defensive bounds check)
    gen_idx = np.unique(gen_idx)
    gen_idx = gen_idx[(gen_idx >= 0) & (gen_idx < G_base)]
    present[gen_idx] = True
    return present


# --- main: outage counts across a set of samples ---
def compute_outage_counts_with_base_ids(
    base_hd,
    samples: List["HeteroData"],
) -> Tuple[np.ndarray, np.ndarray]:
    rel = ("bus", "branch", "bus")
    E_base = base_hd[rel].edge_index.shape[1]
    G_base = 0
    if "gen" in base_hd.node_types:
        gstore = base_hd["gen"]
        G_base = gstore["generation"].shape[0]

    branch_counts = np.zeros(E_base, dtype=int)
    gen_counts = np.zeros(G_base, dtype=int)

    for s in samples:
        present_br = present_branch_ids_by_alignment(base_hd, s)
        branch_counts[~present_br] += 1

        present_g = present_gen_ids_by_link(base_hd, s)
        gen_counts[~present_g] += 1

    return branch_counts, gen_counts


# ---------- 4) Plot ----------
def plot_outage_histograms(branch_counts, gen_counts, title_prefix=""):
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(np.arange(len(branch_counts)), branch_counts)
    ax1.set_xlabel("Branch ID (base index)")
    ax1.set_ylabel("Outage count")
    ax1.set_title(f"{title_prefix}Branch outage counts")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(np.arange(len(gen_counts)), gen_counts)
    ax2.set_xlabel("Generator ID (base index)")
    ax2.set_ylabel("Outage count")
    ax2.set_title(f"{title_prefix}Generator outage counts")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    plt.show()


def plot_grouped_outage_histograms(
    base_hd,
    samples_n1: List["HeteroData"],
    samples_n2: List["HeteroData"],
    title_prefix: str = "",
    normalize: bool = False,  # set True to plot rates instead of counts
    savepath_branches: str | None = None,
    savepath_gens: str | None = None,
):
    # Count outages per base index for each dataset
    br_n1, gen_n1 = compute_outage_counts_with_base_ids(base_hd, samples_n1)
    br_n2, gen_n2 = compute_outage_counts_with_base_ids(base_hd, samples_n2)

    # Optionally normalize by number of samples (gives outage frequency)
    if normalize:
        br_n1 = br_n1 / max(1, len(samples_n1))
        br_n2 = br_n2 / max(1, len(samples_n2))
        gen_n1 = gen_n1 / max(1, len(samples_n1))
        gen_n2 = gen_n2 / max(1, len(samples_n2))

    # ---- Branches ----
    x_b = np.arange(len(br_n1))
    width = 0.45  # bar width for grouped bars
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(x_b - width / 2, br_n1, width, label="N-1")
    ax1.bar(x_b + width / 2, br_n2, width, label="N-2")
    ax1.set_xlabel("Branch ID (base index)")
    ax1.set_ylabel("Outage " + ("rate" if normalize else "count"))
    ax1.set_title(f"{title_prefix}Branch outages (N-1 vs N-2)")
    ax1.set_xticks(x_b)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax1.legend()
    fig1.tight_layout()
    if savepath_branches:
        fig1.savefig(savepath_branches, dpi=200, bbox_inches="tight")
    width = 0.45  # bar width for grouped bars
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(x_b - width / 2, br_n1, width, label="N-1")
    ax1.bar(x_b + width / 2, br_n2, width, label="N-2")
    ax1.set_xlabel("Branch ID (base index)")
    ax1.set_ylabel("Outage " + ("rate" if normalize else "count"))
    ax1.set_title(f"{title_prefix}Branch outages (N-1 vs N-2)")
    ax1.set_xticks(x_b)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax1.legend()
    fig1.tight_layout()
    if savepath_branches:
        fig1.savefig(savepath_branches, dpi=200, bbox_inches="tight")

    # ---- Generators ----
    x_g = np.arange(len(gen_n1))
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(x_g - width / 2, gen_n1, width, label="N-1")
    ax2.bar(x_g + width / 2, gen_n2, width, label="N-2")
    ax2.set_xlabel("Generator ID (base index)")
    ax2.set_ylabel("Outage " + ("rate" if normalize else "count"))
    ax2.set_title(f"{title_prefix}Generator outages (N-1 vs N-2)")
    ax2.set_xticks(x_g)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    if savepath_gens:
        fig2.savefig(savepath_gens, dpi=200, bbox_inches="tight")

    plt.show()


def samples_with_slack_gen_outage(
    base_hd,
    samples: List["HeteroData"],
    nd: int = 8,
) -> List[int]:
    """
    Return the indices of samples where any base slack generator is missing (outaged).

    Assumes:
      - base_hd['gen']['slack_gen'] is a 0/1 (or bool) vector of length G_base
      - Generators in samples are physically removed when outaged
      - present_gen_ids_by_alignment(base_hd, sample_hd, nd) returns a bool mask
        of length G_base indicating which base generators are present in the sample,
        matching ONLY on 'generation' rows (order-agnostic).
    """
    if "gen" not in base_hd.node_types or "slack_gen" not in base_hd["gen"]:
        raise ValueError("Base case must include gen node type and 'slack_gen' vector.")

    slack_mask = base_hd["gen"]["slack_gen"].cpu().numpy().astype(bool)
    slack_idxs = np.where(slack_mask)[0]
    if slack_idxs.size == 0:
        # no slack gen flagged—nothing to check
        return []

    bad_samples = []
    for i, s in enumerate(samples):
        present = present_gen_ids_by_link(base_hd, s)  # your generation-only matcher
        # if any base slack gen is absent in this sample, mark it
        if np.any(~present[slack_idxs]):
            bad_samples.append(i)

    return bad_samples


def _to_long_df(values_n1, values_n2, kind: str):
    ids = np.arange(len(values_n1))
    df = pd.DataFrame(
        {
            f"{kind} ID": ids,
            "N-1": values_n1,
            "N-2": values_n2,
        }
    )
    long = df.melt(
        id_vars=[f"{kind} ID"],
        value_vars=["N-1", "N-2"],
        var_name="Topology",
        value_name="Value",
    )
    # ⚠️ Make the ID column an ordered categorical with categories = full id list
    long[f"{kind} ID"] = pd.Categorical(
        long[f"{kind} ID"], categories=ids, ordered=True
    )
    return long, ids


def build_grouped_bar_chart(
    values_n1,
    values_n2,
    *,
    kind: str,
    normalize=False,
    width_per_bar=14,
    min_width=1100,
    height=280,
    title_prefix="",
):
    long, ids = _to_long_df(values_n1, values_n2, kind)
    ylabel = f"Outage {'rate' if normalize else 'count'}"
    total_width = max(min_width, int(width_per_bar * len(values_n1)))

    chart = (
        alt.Chart(long, title=f"{title_prefix}{kind} outages (N-1 vs N-2)")
        .mark_bar()
        .encode(
            # ⚠️ Fix order & tick positions explicitly (works with numeric IDs)
            x=alt.X(
                f"{kind} ID:O",
                sort=ids.tolist(),
                axis=alt.Axis(values=ids.tolist(), labelOverlap=False),
            ),
            xOffset=alt.XOffset("Topology:N"),  # side-by-side bars
            y=alt.Y("Value:Q", title=ylabel),
            color=alt.Color("Topology:N", scale=alt.Scale(domain=["N-1", "N-2"])),
            tooltip=[
                f"{kind} ID:O",
                "Topology:N",
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        )
        .properties(width=total_width, height=height)
    )
    return chart


def build_outage_panels(
    base_hd,
    samples_n1,
    samples_n2,
    *,
    normalize: bool = False,
    title_prefix: str = "",
    width_per_bar: int = 14,
    min_width: int = 1100,
    height: int = 280,
):
    """
    Convenience wrapper: compute counts and build both charts.
    Returns:
      (branch_chart, branch_html, gen_chart, gen_html)
    """
    br_n1, gen_n1 = compute_outage_counts_with_base_ids(base_hd, samples_n1)
    br_n2, gen_n2 = compute_outage_counts_with_base_ids(base_hd, samples_n2)

    if normalize:
        br_n1 = br_n1 / max(1, len(samples_n1))
        br_n2 = br_n2 / max(1, len(samples_n2))
        gen_n1 = gen_n1 / max(1, len(samples_n1))
        gen_n2 = gen_n2 / max(1, len(samples_n2))

    br_chart = build_grouped_bar_chart(
        br_n1,
        br_n2,
        kind="Branch",
        normalize=normalize,
        width_per_bar=width_per_bar,
        min_width=min_width,
        height=height,
        title_prefix=title_prefix,
    )
    gen_chart = build_grouped_bar_chart(
        gen_n1,
        gen_n2,
        kind="Generator",
        normalize=normalize,
        width_per_bar=width_per_bar,
        min_width=min_width,
        height=height,
        title_prefix=title_prefix,
    )
    return br_chart, gen_chart


# Helper functions to validate power balance satisfaction


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

        return delta_P, delta_Q


def check_power_balance(dataset: List[HeteroData]):
    pbl = PowerBalanceAnalysis()
    results = []
    for data in dataset:
        # for each sample in the dataset:
        # run PBL
        delta_P, delta_Q = pbl(
            data
        )  # need to fix this in the function call because this will return the mean
        # append results to list
        results.append((delta_P, delta_Q))

    stats = compute_power_balance_stats(results)
    return stats


def compute_power_balance_stats(
    power_balance_result: List[Tuple[torch.Tensor, torch.Tensor]],
):
    delta_P, delta_Q = zip(*power_balance_result)

    # detach to avoid tracking gradients and flatten
    dp = torch.cat([d.detach().reshape(-1) for d in delta_P])
    dq = torch.cat([d.detach().reshape(-1) for d in delta_Q])

    # handle empty tensors defensively
    def torch_stats(t: torch.Tensor):
        if t.numel() == 0:
            return {"max": None, "min": None, "mean": None, "std": None}
        return {
            "max": torch.max(t.abs()).item(),
            "min": torch.min(t.abs()).item(),
            "mean": torch.mean(t.abs()).item(),
            "median": torch.median(t.abs()).item(),
            # match numpy's population std (ddof=0)
            "std": torch.std(t.abs(), unbiased=False).item(),
        }

    stats = {"delta_P": torch_stats(dp), "delta_Q": torch_stats(dq)}
    return stats
