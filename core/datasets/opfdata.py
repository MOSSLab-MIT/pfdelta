# This code contains the data used to train CANOS-OPF, originally proposed here: 
# 
#   Piloto, L., Liguori, S., Madjiheurem, S., Zgubic, M., Lovett, S., 
#   Tomlinson, H., … Witherspoon, S. (2024). CANOS: A Fast and Scalable 
#   Neural AC-OPF Solver Robust To N-1 Perturbations. arXiv [Cs.LG]. 
#   Retrieved from http://arxiv.org/abs/2403.17660
#
# It uses the OPFDataset class (available on from torch_geometric.datasets). 
# 
# Code was implemented by Anvita Bhagavathula, Alvaro Carbonero, and Ana K. Rivera 
# in April 2025.

import os
from functools import partial

from torch_geometric.datasets import OPFDataset

from core.utils.registry import registry
from core.datasets.data_stats import opfdata_stats
from core.datasets.dataset_utils import mean0_var1


def opfdata_mean0_var1(stats, data):
    means = stats["mean"]
    stds = stats["std"]

    def exception_transform(x, mean, std):
        r"""Transforms every value to mean 0, var 1 unless std is 0, in which
        case the value is just transformed to 0."""
        ones_std = std == 0.0
        std[ones_std] = 1.0
        x = mean0_var1(x, mean, std)
        return x

    # Bus and gen limits
    data["bus"]["v_lims"] = data["bus"]["x"][:, 2:]
    data["generator"]["p_lims"] = data["generator"]["x"][:, 2:4]
    data["generator"]["q_lims"] = data["generator"]["x"][:, 5:7]

    # Branch values
    data["bus", "ac_line", "bus"]["branch_vals"] = data["bus", "ac_line", "bus"][
        "edge_attr"
    ]
    data["bus", "transformer", "bus"]["branch_vals"] = data[
        "bus", "transformer", "bus"
    ]["edge_attr"]

    # Loads and shunts
    data["load"]["unnormalized"] = data["load"]["x"]
    data["shunt"]["unnormalized"] = data["shunt"]["x"]

    values_to_change = [
        ("bus", "x"),
        # ("bus", "y"),
        ("generator", "x"),
        # ("generator", "y"),
        ("load", "x"),
        ("shunt", "x"),
        (("bus", "ac_line", "bus"), "edge_attr"),
        # (("bus", "ac_line", "bus"), "edge_label"),
        (("bus", "transformer", "bus"), "edge_attr"),
        # (("bus", "transformer", "bus"), "edge_label"),
    ]

    for dtype, entry in values_to_change:
        mean = means[dtype][entry]
        std = stds[dtype][entry]
        x = data[dtype][entry]
        data[dtype][entry] = exception_transform(x, mean, std)

    return data


@registry.register_dataset("opfdata")
class OPFData(OPFDataset):
    def __init__(
        self,
        split="train",
        case_name="pglib_opf_case14_ieee",
        num_groups=1,
        topological_perturbations=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        self.root = os.path.join("data", "opfdata")

        if pre_transform is not None:
            if pre_transform == "mean_zero_variance_one":
                category = "n_minus_one" if topological_perturbations else "none"
                stats = opfdata_stats[case_name][category]
                pre_transform = partial(opfdata_mean0_var1, stats)
            else:
                raise ValueError(f"Transform {pre_transform} not recognized!")

        super().__init__(
            root=self.root,
            split=split,
            case_name=case_name,
            num_groups=num_groups,
            topological_perturbations=topological_perturbations,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
