import warnings
from functools import partial
from typing import Any, Dict

import torch

from core.utils.registry import registry
from core.datasets.data_stats import pfnet_pfdata_stats
from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.datasets.dataset_utils import (
    canos_pf_data_mean0_var1,
    canos_pf_slack_mean0_var1,
    pfnet_data_mean0_var1,
)


#############################################################################
#      CUSTOM PFDELTADATASET CLASSES TAILORED TO PER-MODEL PREPROCESSING
#############################################################################

@registry.register_dataset("pfdeltaGNS")
class PFDeltaGNS(PFDeltaDataset):
    """
    PFDelta dataset inherited class specialized for GNS (Graph Network Solver) model.

    This class extends the base PFDeltaDataset to include GNS-specific
    bus-level node features, such as initial voltage magnitude and angle (v, θ),
    and exposes additional attributes required by the GNS training pipeline.

    Parameters
    ----------
    root_dir : str, optional
        Path to the dataset root directory. Defaults to "data".
    case_name : str, optional
        Name of the power grid case (e.g., "case118"). Defaults to an empty string.
    split : str, optional
        Dataset split to load ('train', 'val', or 'test'). Defaults to "train".
    model : str, optional
        Model name identifier. Defaults to "GNS".
    task : float or str, optional
        Task identifier (e.g., 1.1, 3.1). Defaults to 1.3.
    add_bus_type : bool, optional
        If True, includes bus type encodings as additional features.
    transform : callable, optional
        Optional transform applied on data objects before returning.
    pre_transform : callable, optional
        Transform applied before saving the processed dataset.
    pre_filter : callable, optional
        Filter function applied to data objects before processing.
    force_reload : bool, optional
        If True, forces dataset reprocessing. Defaults to False.

    Notes
    -----
    This variant initializes per-bus features for GNS training, including:
        - theta : initial voltage angle
        - v : initial voltage magnitude
        - pd, qd : active/reactive demand
        - pg, qg : active/reactive generation
        - x_gns : stacked tensor of [theta, v] per bus
    """
    def __init__(
        self,
        root_dir="data",
        case_name="",
        split="train",
        model="GNS",
        task=1.1,
        add_bus_type=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        super().__init__(
            root_dir=root_dir,
            case_name=case_name,
            split=split,
            model=model,
            task=task,
            add_bus_type=add_bus_type,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

    def build_heterodata(self, pm_case: dict, is_cpf_sample: bool = False):
        """
        Build a `HeteroData` graph with additional GNS-specific fields.

        Parameters
        ----------
        pm_case : dict
            PowerModels.jl-style case dictionary with bus, branch, gen, and load data.
        is_cpf_sample : bool, optional
            If True, marks data as a continuation power flow (CPF) sample.

        Returns
        -------
        data : torch_geometric.data.HeteroData
            The constructed heterogeneous graph with added GNS-specific features.
        """
        # call base version
        data = super().build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)
        num_buses = data["bus"].x.size(0)
        num_gens = data["gen"].generation.size(0)
        num_loads = data["load"].demand.size(0)

        # Init bus-level fields
        v_buses = torch.zeros(num_buses)
        theta_buses = torch.zeros(num_buses)
        pd_buses = torch.zeros(num_buses)
        qd_buses = torch.zeros(num_buses)
        pg_buses = torch.zeros(num_buses)
        qg_buses = torch.zeros(num_buses)

        # Read bus types
        bus_types = data["bus"].bus_type
        x_gns = torch.zeros((num_buses, 2))

        for bus_idx in range(num_buses):
            bus_type = bus_types[bus_idx].item()
            pf_x = data["bus"].x[bus_idx]
            pf_y = data["bus"].x[bus_idx]
            bus_demand = data["bus"].bus_demand[bus_idx]
            bus_gen = data["bus"].bus_gen[bus_idx]

            if bus_type == 1:  # PQ bus
                # Flat start for PQ bus
                x_gns[bus_idx] = torch.tensor([0.0, 1.0])
                pd = pf_x[0]
                qd = pf_x[1]
                pg = bus_gen[0]
                qg = bus_gen[1]
            elif bus_type == 2:  # PV bus
                v = pf_x[1]
                theta = torch.tensor(0.0)
                x_gns[bus_idx] = torch.stack([theta, v])
                pd = bus_demand[0]
                qd = bus_demand[1]
                pg = bus_gen[0]
                qg = bus_gen[1]
            elif bus_type == 3:  # Slack bus
                x_gns[bus_idx] = pf_x
                pd = bus_demand[0]
                qd = bus_demand[1]
                pg = bus_gen[0]
                qg = bus_gen[1]

            v_buses[bus_idx] = x_gns[bus_idx][1]
            theta_buses[bus_idx] = x_gns[bus_idx][0]
            pd_buses[bus_idx] = pd
            qd_buses[bus_idx] = qd
            pg_buses[bus_idx] = pg
            qg_buses[bus_idx] = qg

        # Store in bus
        data["bus"].x_gns = x_gns
        data["bus"].v = v_buses
        data["bus"].theta = theta_buses
        data["bus"].pd = pd_buses
        data["bus"].qd = qd_buses
        data["bus"].pg = pg_buses
        data["bus"].qg = qg_buses
        data["bus"].delta_p = torch.zeros_like(v_buses)
        data["bus"].delta_q = torch.zeros_like(v_buses)
        data["gen"].num_nodes = num_gens
        data["load"].num_nodes = num_loads

        if self.pre_transform:
            data = self.pre_transform(data)

        return data


@registry.register_dataset("pfdeltaCANOS")
class PFDeltaCANOS(PFDeltaDataset):
    """
    PFDelta dataset variant specialized for the CANOS  model.

    This subclass of PFDeltaDataset applies optional CANOS-specific
    preprocessing and normalization transforms, and prunes the heterogeneous
    graph to include only the node and edge types required by CANOS
    (bus, PV, PQ, and slack).

    Parameters
    ----------
    root_dir : str, optional
        Path to the dataset root directory. Defaults to "data".
    case_name : str, optional
        Name of the power grid case (e.g., "case118"). Defaults to an empty string.
    split : str, optional
        Dataset split to load ('train', 'val', or 'test'). Defaults to "train".
    model : str, optional
        Model name identifier. Defaults to "CANOS".
    task : float or str, optional
        Task identifier (e.g., 1.1, 3.1). Defaults to 1.3.
    add_bus_type : bool, optional
        Whether to include bus type encodings as part of node features.
    transform : callable or str, optional
        Transform applied to data objects after loading.
        If set to "canos_pf_slack_mean0_var1", applies CANOS normalization
        using case-specific statistics.
    pre_transform : callable or str, optional
        Transform applied before saving the processed dataset.
        If set to "canos_pf_data_mean0_var1", applies CANOS normalization
        using case-specific statistics.
    pre_filter : callable, optional
        Optional filter applied to data objects before processing.
    force_reload : bool, optional
        If True, forces dataset reprocessing. Defaults to False.

    Notes
    -----
    - Normalization statistics are retrieved from :data:`canos_pfdelta_stats`
      based on the provided case name.
    - The resulting `HeteroData` graph only includes the node types
      relevant to CANOS: bus, PV, PQ, and slack.
    """

    def __init__(
        self,
        root_dir="data",
        case_name="",
        split="train",
        model="CANOS",
        task=1.1,
        add_bus_type=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        if pre_transform is not None:
            if pre_transform == "canos_pf_data_mean0_var1":
                stats = canos_pfdelta_stats[case_name]
                pre_transform = partial(canos_pf_data_mean0_var1, stats)

        if transform is not None:
            if transform == "canos_pf_slack_mean0_var1":
                stats = canos_pfdelta_stats[case_name]
                transform = partial(canos_pf_slack_mean0_var1, stats)

        super().__init__(
            root_dir=root_dir,
            case_name=case_name,
            split=split,
            model=model,
            task=task,
            add_bus_type=add_bus_type,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

    def build_heterodata(self, pm_case: dict, is_cpf_sample: bool = False):
        """
        Build a CANOS-compatible `HeteroData` object.

        This method constructs the base heterogeneous graph via
        \PFDeltaDataset.build_heterodata`, then prunes it to
        retain only the node and edge types used by CANOS 
        (bus, PV, PQ, slack).

        Parameters
        ----------
        pm_case : dict
            PowerModels.jl-style case dictionary containing bus, branch,
            generator, and load data.
        is_cpf_sample : bool, optional
            If True, marks the data as a continuation power flow (CPF) sample.

        Returns
        -------
        data : torch_geometric.data.HeteroData
            The processed heterogeneous graph with CANOS-specific node
            and edge types only.
        """
        # call base version
        data = super().build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)

        # Now prune the data to only keep bus, PV, PQ, slack
        keep_nodes = {"bus", "PV", "PQ", "slack"}

        for node_type in list(data.node_types):
            if node_type not in keep_nodes:
                del data[node_type]

        for edge_type in list(data.edge_types):
            src, _, dst = edge_type
            if src not in keep_nodes or dst not in keep_nodes:
                del data[edge_type]

        if self.pre_transform:
            data = self.pre_transform(data)

        return data


@registry.register_dataset("pfdeltaPFNet")
class PFDeltaPFNet(PFDeltaDataset):
    """
    PFDelta dataset variant specialized for the PFNet model.

    This subclass of PFDeltaDataset converts the base
    heterogeneous graph representation into the input format expected
    by PFNet. Specifically, it constructs per-bus feature vectors and
    labels, encodes bus types via one-hot representations, applies
    prediction/input masks, and adapts edge attributes for PFNet’s
    parameterization.

    Parameters
    ----------
    root_dir : str, optional
        Path to the dataset root directory. Defaults to "data".
    case_name : str, optional
        Name of the power grid case (e.g., "case118"). Defaults to an empty string.
    split : str, optional
        Dataset split to load ('train', 'val', or 'test'). Defaults to "train".
    model : str, optional
        Model name identifier. Defaults to "PFNet".
    task : float or str, optional
        Task identifier (e.g., 1.1, 3.1). Defaults to 1.3.
    add_bus_type : bool, optional
        Whether to include bus type encodings in the base dataset before
        PFNet-specific processing.
    transform : callable or str, optional
        Transform applied to data objects after loading.
        If set to "pfnet_data_mean0_var1", applies PFNet normalization
        using case-specific statistics from pfnet_pfdata_stats.
    pre_transform : callable or str, optional
        Transform applied before saving the processed dataset.
        If set to "pfnet_data_mean0_var1", applies PFNet normalization
        using case-specific statistics.
    pre_filter : callable, optional
        Optional filter applied to data objects before processing.
    force_reload : bool, optional
        If True, forces dataset reprocessing. Defaults to False.

    Notes
    -----
    - Each bus feature vector concatenates one-hot bus type encoding,
      masked input features, and the prediction mask, resulting in
      a vector of length 16.
    - Edge attributes are reformatted to contain resistance, reactance,
      total susceptance, transformer tap ratio, and phase shift angle
      for each branch.
    """

    def __init__(
        self,
        root_dir="data",
        case_name="",
        split="train",
        model="PFNet",
        task=1.1,
        add_bus_type=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
        normalized_case_name=None,
    ):
        self.normalized_case_name = normalized_case_name

        if pre_transform:
            if pre_transform == "pfnet_data_mean0_var1":
                stats = pfnet_pfdata_stats[case_name]
                pre_transform = partial(pfnet_data_mean0_var1, stats)

        if transform is not None:
            if transform == "pfnet_data_mean0_var1":
                if task in [3.1, 3.2, 3.3]:
                    stats = pfnet_pfdata_stats[self.normalized_case_name]
                else:
                    stats = pfnet_pfdata_stats[case_name]
                transform = partial(pfnet_data_mean0_var1, stats)

        super().__init__(
            root_dir=root_dir,
            case_name=case_name,
            split=split,
            model=model,
            task=task,
            add_bus_type=add_bus_type,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

    def build_heterodata(self, pm_case: dict, is_cpf_sample: bool = False):
        # call base version
        data = super().build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)

        num_buses = data["bus"].x.size(0)
        bus_types = data["bus"].bus_type
        pf_x = data["bus"].x
        pf_y = data["bus"].y
        shunts = data["bus"].shunt
        num_gens = data["gen"].generation.size(0)
        num_loads = data["load"].demand.size(0)

        # New node features for PFNet
        x_pfnet = []
        y_pfnet = []
        for i in range(num_buses):
            bus_type = int(bus_types[i].item())

            # One-hot encode bus type
            one_hot = torch.zeros(4)
            one_hot[bus_type - 1] = 1
            gs, bs = shunts[i]

            # Prediction mask
            if bus_type == 1:  # PQ
                pred_mask = torch.tensor([1, 1, 0, 0, 0, 0])
                va, vm = pf_y[i]
                pd, qd = pf_x[i]
                input_mask = (1 - pred_mask).float()
                input_feats = torch.tensor([vm, va, pd, qd, gs, bs]) * input_mask
                features = torch.cat([one_hot, input_feats])
                y = torch.tensor([vm, va, pd, qd, gs, bs])
            elif bus_type == 2:  # PV
                pred_mask = torch.tensor([0, 1, 0, 1, 0, 0])
                pg_pd, vm = pf_x[i]
                qg_qd, va = pf_y[i]
                input_mask = (1 - pred_mask).float()
                input_feats = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs]) * input_mask
                features = torch.cat([one_hot, input_feats])
                y = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs])
            elif bus_type == 3:  # Slack
                pred_mask = torch.tensor([0, 0, 1, 1, 0, 0])
                va, vm = pf_x[i]
                pg_pd, qg_qd = pf_y[i]
                input_mask = (1 - pred_mask).float()
                input_feats = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs]) * input_mask
                features = torch.cat([one_hot, input_feats])
                y = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs])

            x_pfnet.append(torch.cat([features, pred_mask]))
            y_pfnet.append(y)

        x_pfnet = torch.stack(x_pfnet)  # shape [N, 4+6+6=16]
        y_pfnet = torch.stack(y_pfnet)  # shape [N, 6]

        data["bus"].x = x_pfnet
        data["bus"].y = y_pfnet

        data["gen"].num_nodes = num_gens
        data["load"].num_nodes = num_loads

        edge_attrs = []
        for attr in data["bus", "branch", "bus"].edge_attr:
            r, x = attr[0], attr[1]
            b = attr[3] + attr[5]
            tau, angle = attr[6], attr[7]
            edge_attrs.append(torch.tensor([r, x, b, tau, angle]))

        data["bus", "branch", "bus"].edge_attr = torch.stack(edge_attrs)

        if self.pre_transform:
            data = self.pre_transform(data)

        return data
