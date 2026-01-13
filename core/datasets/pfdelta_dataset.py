import json
import os
import copy
from tqdm import tqdm
from functools import partial
import glob
import torch
from torch_geometric.data import InMemoryDataset, HeteroData, download_url, extract_tar
from typing import Any, Dict
import warnings
from core.datasets.dataset_utils import (
    canos_pf_data_mean0_var1,
    canos_pf_slack_mean0_var1,
    pfnet_data_mean0_var1,
)
from core.datasets.data_stats import pfnet_pfdata_stats
from core.utils.registry import registry


@registry.register_dataset("pfdeltadata")
class PFDeltaDataset(InMemoryDataset):
    """
    PFDeltaDataset: PyTorch Geometric InMemoryDataset for PFDelta benchmark data.

    This class automatically downloads, extracts, and preprocesses data from the
    PFDelta benchmark on HuggingFace 
    (https://huggingface.co/datasets/pfdelta/pfdelta/tree/main). It converts raw
    JSON power flow samples into PyTorch Geometric `HeteroData` objects for use
    with graph-based learning models (e.g., CANOS, GNS, PFNet).

    The dataset supports multiple loading and processing modes:
        - Task-specific datasets: For supervised learning on benchmark tasks
          (e.g., 1.x, 2.x, 3.x, 4.x)
        - Analysis datasets: For per-case data inspection, visualization, and
          analytical performance studies

    Processed datasets are automatically organized by task, case, perturbation
    type, and feasibility regime within the downloaded dataset's directory structure.
    """

    def __init__(
        self,
        root_dir: str = "data",
        case_name: str = "case14",
        perturbation: str = "n",
        feasibility_type: str = "feasible",
        n_samples: int = -1,
        split: str = "train",
        model: str = "",
        task: Any = 1.3,
        add_bus_type: bool = False,
        transform: Any = None,
        pre_transform: Any = None,
        pre_filter: Any = None,
        force_reload: bool = False,
    ):
        """
        Initialize the PFDeltaDataset class. 

        Parameters
        ----------
        root_dir : str
            Root data directory.
        case_name : str
            Power system case folder name (e.g., "case14").
        perturbation : str
            Grid contingency type ("n", "n-1", or "n-2").
        feasibility_type : str
            Feasibility regime ("feasible", "near infeasible", or "approaching infeasible").
        n_samples : int
            Number of samples to load. If < 0, loads all available samples.
        split : str
            Dataset split to load ("train", "val", "test", "all", or 
            "separate_{casename}_{split}_{feasibility}_{grid_type}").
        model : str
            Model shorthand used for processed dataset naming.
        task : float or str
            Benchmark task ID (e.g., 1.3, 3.1, 4.2) or "analysis" for analytical mode.
        add_bus_type : bool
            Whether to include bus-type-specific node sets in HeteroData graphs.
        transform, pre_transform, pre_filter : callable, optional
            PyTorch Geometric dataset hooks for on-the-fly or preprocessing transforms.
        force_reload : bool
            If True, forces the dataset to reprocess even if cache exists.

        Notes
        -----
        This initializer configures the dataset's directory structure and metadata.
        The raw-to-graph conversion occurs within the process() method, automatically
        triggered by the parent InMemoryDataset when required. 
        """
        self.split = split
        self.case_name = case_name
        self.force_reload = force_reload
        self.add_bus_type = add_bus_type
        self.task = task
        self.model = model
        self.root = os.path.join(root_dir)

        self.perturbation = perturbation
        self.feasibility_type = feasibility_type
        self.n_samples = n_samples

        if task in [3.2, 3.3]:
            self._custom_processed_dir = os.path.join(
                self.root, "processed", f"combined_task_{task}_{model}_"
            )

        elif task == "analysis":
            self._custom_processed_dir = os.path.join(
                self.root,
                "processed",
                f"task_{self.task}_{self.case_name}_{self.perturbation}_{self.feasibility_type}_{self.n_samples}",
            )
            self.split = "all"
        else:
            self._custom_processed_dir = os.path.join(
                self.root, "processed", f"combined_task_{task}_{model}_{self.case_name}"
            )

        self.all_case_names = [
            "case14",
            "case30",
            "case57",
            "case118",
            "case500",
            "case2000",
        ]

        self.task_config = {  # values here will have the number of train samples
            1.1: {
                "feasible": {"n": 54000, "n-1": 0, "n-2": 0},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            1.2: {
                "feasible": {"n": 27000, "n-1": 27000, "n-2": 0},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            1.3: {
                "feasible": {"n": 18000, "n-1": 18000, "n-2": 18000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            2.1: {
                "feasible": {"n": 18000, "n-1": 18000, "n-2": 18000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            2.2: {
                "feasible": {"n": 12000, "n-1": 12000, "n-2": 12000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            2.3: {
                "feasible": {"n": 6000, "n-1": 6000, "n-2": 6000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            3.1: {
                "feasible": {"n": 18000, "n-1": 18000, "n-2": 18000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            3.2: {
                "feasible": {"n": 6000, "n-1": 6000, "n-2": 6000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            3.3: {
                "feasible": {"n": 9000, "n-1": 9000, "n-2": 9000},
                "near infeasible": {"n": 0, "n-1": 0, "n-2": 0},
            },
            4.1: {
                "near infeasible": {"n": 1800, "n-1": 1800, "n-2": 1800},
                "feasible": {"n": 16200, "n-1": 16200, "n-2": 16200},
            },
            4.2: {
                "approaching infeasible": {"n": 7200, "n-1": 7200, "n-2": 7200},
                "near infeasible": {"n": 1800, "n-1": 1800, "n-2": 1800},
                "feasible": {"n": 9000, "n-1": 9000, "n-2": 9000},
            },
            4.3: {"near infeasible": {"n": 3600, "n-1": 3600, "n-2": 3600},
                  "approaching infeasible": {"n": 14400, "n-1": 14400, "n-2": 14400},
                  "feasible": {"n": 0, "n-1": 0, "n-2": 0}
            },
            "analysis": {
                "feasible": {"n": 56000, "n-1": 29000, "n-2": 20000},
                "near infeasible": {"n": 2000, "n-1": 2000, "n-2": 2000},
                "approaching infeasible": {"n": 7200, "n-1": 7200, "n-2": 7200},
            },
        }

        self.feasibility_config = {
            "feasible": {
                "n": 56000,
                "n-1": 29000,
                "n-2": 20000,
                "test": {"n": 2000, "n-1": 2000, "n-2": 2000},
            },
            "approaching infeasible": {
                "n": 7200,
                "n-1": 7200,
                "n-2": 7200,
                "test": None,  # no test set for this regime
            },
            "near infeasible": {
                "n": 2000,
                "n-1": 2000,
                "n-2": 2000,
                "test": {"n": 200, "n-1": 200, "n-2": 200},
            },
        }

        self.task_split_config = {
            3.1: {
                "train": [self.case_name],
                "val": self.all_case_names,
                "test": self.all_case_names,
            },
            3.2: {
                "train": ["case14", "case30", "case57"],
                "val": ["case118", "case500", "case2000"],
                "test": ["case118", "case500", "case2000"],
            },
            3.3: {
                "train": ["case118", "case500", "case2000"],
                "val": ["case14", "case30", "case57"],
                "test": ["case14", "case30", "case57"],
            },
        }

        super().__init__(
            self.root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        self.load(self.split)

    def _split_to_idx(self):
        return {"train": 0, "val": 1, "test": 2}[self.split]

    @property
    def raw_file_names(self) -> list:
        # List raw file names required for this dataset

        # Determine which case(s) to download
        if self.task in [3.1, 3.2, 3.3, "analysis"]:
            case_names = self.all_case_names  # a list of strings
        else:
            case_names = [self.case_name]

        # For each case, download all sub-archives
        for case_name in case_names:
            case_raw_dir = os.path.join(self.root, case_name)
            # check if this exists
            if not os.path.exists(case_raw_dir):
                return []  # this triggers download()

        return sorted(
            [
                os.path.join(self.task, self.casename, f)
                for f in os.listdir(case_raw_dir)
                if f.endswith(".json")
            ]
        )

    @property
    def processed_dir(self) -> str:
        # Directory where processed files for this dataset are stored
        return self._custom_processed_dir

    @property
    def processed_file_names(self) -> list:
        # Name of files that must exist in processed_dir
        if self.task == "analysis":
            return ["all.pt"]  # Only require all.pt for analysis task

        return ["train.pt", "val.pt", "test.pt"]

    def download(self) -> None:
        """
        Download and extract raw PFDelta dataset archives. This method is automatically
        triggered when the PFDeltaDataset class is initialized.

        This method retrieves per-case data archives and the shuffle mapping directory 
        from the canonical PFDelta Hugging Face repository:
        https://huggingface.co/datasets/pfdelta/pfdelta/tree/main

        It ensures that all necessary files are available locally under `self.root`
        for downstream dataset processing and loading. Existing directories or files 
        are skipped to prevent redundant downloads.

        For each case, the method downloads the corresponding tar file, extracts its 
        contents into the dataset root directory, and removes the compressed archive. 
        The global `shuffle_files` directory, containing pre-defined train/val/test 
        splits, is downloaded once and reused across all cases.

        Notes
        -----
        - The method supports both per-case downloads (for tasks 1.x–2.x) and 
        multi-case downloads (for tasks 3.x and analysis mode).
        - Downloads are skipped if the corresponding extracted folders already exist.
        - Each tar file is automatically removed after successful extraction.
        """
        print(f"Downloading files for task {self.task}...")

        # Determine which case(s) to download
        if self.task in [3.1, 3.2, 3.3]:
            case_names = self.all_case_names  # a list of strings
        else:
            case_names = [self.case_name]

        # Remote dataset URL (Hugging Face)
        base_url = "https://huggingface.co/datasets/pfdelta/pfdelta/resolve/main"

        # Download the shuffle files if not already present
        shuffle_download_path = self.root
        shuffle_files_dir = os.path.join(shuffle_download_path, "shuffle_files")

        # Only download and extract if shuffle_files directory doesn't exist
        if os.path.exists(shuffle_files_dir):
            print("Shuffle files already exist. Skipping download.")
        else:
            print("Downloading shuffle files...")
            file_url = f"{base_url}/shuffle_files.tar.gz"
            shuffle_files_path = download_url(file_url, shuffle_download_path, log=True)
            extract_tar(shuffle_files_path, shuffle_download_path)

        # For each case, download all sub-archives
        for case_name in case_names:
            data_url = f"{base_url}/{case_name}.tar.gz"
            case_raw_dir = self.root
            os.makedirs(case_raw_dir, exist_ok=True)

            # Skip download if the archive already exists
            if os.path.exists(
                os.path.join(case_raw_dir, case_name.replace(".tar.gz", ""))
            ):
                print(f"{case_name} data already exists. Skipping download.")
                continue

            print(f"Downloading {case_name} data from {data_url} ...")

            try:
                download_path = case_raw_dir
                data_path = download_url(data_url, download_path, log=True)
                extract_tar(data_path, download_path)
                os.unlink(data_path)  # delete the archive after extraction
                print(f"Extracted {case_name} data to {case_raw_dir}")

            except Exception as e:
                print(f"Failed to download or extract {case_name} data: {e}")

    def build_heterodata(
        self, 
        pm_case: Dict[str, Any], 
        is_cpf_sample: bool = False) -> HeteroData:
        """
        Convert a parsed PowerModels case dictionary into a HeteroData graph.

        This method parses a PowerModels.jl-style case JSON (produced by the PFDelta
        data generation pipeline) and converts it into a typed heterogeneous graph
        suitable for PyTorch Geometric. Each case contains detailed electrical
        network data, including buses, generators, loads, and branches, along with
        solved power flow variables.

        The resulting `HeteroData` object includes:
        - Node sets for buses, generators, and loads (and optionally PV, PQ, slack subsets).
        - Edge types linking physical entities (e.g., ("bus","branch","bus"), ("gen","gen_link","bus")).
        - Node- and edge-level attributes for network parameters, limits, and solved quantities.
        - Labels (.y) for supervised learning tasks (e.g., voltage angles and magnitudes).

        Parameters
        ----------
        pm_case : dict
            PowerModels network dictionary parsed from a JSON file.  
            - For continuation power flow (CPF) samples, this contains a "solved_net" entry 
            holding both network and solution data.  
            - For standard power flow samples, the dictionary includes "network" and 
            `"solution"` keys.

        is_cpf_sample : bool, optional (default=False)
            Whether the case is a CPF sample (i.e., derived from a near- or approaching-
            infeasible regime). Enables alternative handling for "solved_net" format.

        Returns
        -------
        torch_geometric.data.HeteroData
            A heterogeneous graph representation of the power network containing node and edge
            types with structured features, physical parameters, and solved variables.

        Notes
        -----
        - Node features include bus demands, generations, shunts, voltage magnitudes/angles, and limits.
        - Edge features include branch impedances, admittances, tap ratios, and flow limits.
        - Supports optional sub-node sets (PV, PQ, slack) and linking edges if 
        self.add_bus_type=True.
        - The resulting graphs are standardized for use across multiple PFDelta tasks (1.x–4.x).
        """
        data = HeteroData()

        if is_cpf_sample:
            # If sample was generated via CPF, solved_net contains both the network and solution information in one dict
            network_data = pm_case["solved_net"]
            solution_data = pm_case["solved_net"]
        else:
            # For standard samples, network and solution are separate dicts
            network_data = pm_case["network"]
            solution_data = pm_case["solution"]["solution"]

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
                            assert gen["pg"] == 0 and gen["qg"] == 0, (
                                f"Expected gen {gen_id} to be off"
                            )
                        else:
                            assert solution_data["gen"].get(gen_id) is None, (
                                f"Expected gen {gen_id} to be off."
                            )

            bus_gen.append(torch.tensor([pg, qg]))

            # Decide final bus type
            bus_type_now = bus["bus_type"]

            if bus_type_now == 2 and pg == 0.0 and qg == 0.0:
                bus_type_now = 1  # PV bus with no gen --> becomes PQ
                warnings.warn(
                    f"Warning: Changed bus {bus_id} type from PV to PQ because it has no generation."
                )  # TODO: I don't think you need to do this anymore, leaving it for now.

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
                    assert (
                        solution_data["gen"][gen_id]["pg"] == 0
                        and solution_data["gen"][gen_id]["qg"] == 0
                    ), f"Expected gen {gen_id} to be off"
                else:
                    assert solution_data["gen"].get(gen_id) is None, (
                        f"Expected gen {gen_id} to be off."
                    )

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
            assert branch_sol is not None, (
                f"Missing solution for active branch {branch_id_str}"
            )

            edge_label.append(
                torch.tensor(
                    [
                        branch_sol["pf"],
                        branch_sol["qf"],
                        branch_sol["pt"],
                        branch_sol["qt"],
                    ]
                )
            )

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

        if self.add_bus_type:
            data["PQ"].x = torch.stack(PQ_bus_x)
            data["PQ"].y = torch.stack(PQ_bus_y)

            data["PV"].x = torch.stack(PV_bus_x)
            data["PV"].y = torch.stack(PV_bus_y)
            data["PV"].generation = torch.stack(PV_generation)
            data["PV"].demand = torch.stack(PV_demand)

            data["slack"].x = torch.stack(slack_x)
            data["slack"].y = torch.stack(slack_y)
            data["slack"].generation = torch.stack(slack_generation)
            data["slack"].demand = torch.stack(slack_demand)

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
                # These correspond to limits in the original pglib case file, limits are not enforced in our dataset
                data[link_name].edge_limits = torch.stack(
                    edge_limits
                ) 

        if self.add_bus_type:
            for link_name, edges in {
                ("PV", "PV_link", "bus"): PV_to_bus,
                ("PQ", "PQ_link", "bus"): PQ_to_bus,
                ("slack", "slack_link", "bus"): slack_to_bus,
            }.items():
                edge_tensor = torch.stack(edges, dim=1)
                data[link_name].edge_index = edge_tensor
                data[
                    (link_name[2], link_name[1], link_name[0])
                ].edge_index = edge_tensor.flip(0)

        return data

    def get_analysis_data(self, case_root: str) -> list[HeteroData]:
        """
        Collect and convert raw JSON files for the 'analysis' task.

        This method gathers solved power flow samples from the appropriate 
        feasibility regime—feasible, near infeasible (nose), or approaching 
        infeasible (around_nose)—and converts each JSON case into a 
        `HeteroData` object suitable for graph-based analysis.

        The number of loaded samples is determined by self.n_samples or 
        the task configuration limits defined in self.task_config.

        Parameters
        ----------
        case_root : str
            Path to the case directory containing subfolders such as 
            raw/, nose/, or around_nose/.

        Returns
        -------
        list of HeteroData
            List of converted power flow samples up to the configured dataset size.

        Notes
        -----
        - Samples are treated as continuation power flow (CPF) cases when 
        the feasibility type is not 'feasible'.
        - File discovery is based on fixed directory naming conventions 
        within each case folder.
        """
        dataset_size = (
            self.n_samples
            if self.n_samples > 0
            else self.task_config[self.task][self.feasibility_type][self.perturbation]
        )
        data_list = []

        if self.feasibility_type == "feasible":
            raw_fnames = glob.glob(
                os.path.join(case_root, self.perturbation, "raw", "*.json")
            )
        elif self.feasibility_type == "near infeasible":
            raw_fnames = glob.glob(
                os.path.join(case_root, self.perturbation, "nose", "train", "*.json")
            )
            raw_fnames.extend(
                glob.glob(
                    os.path.join(case_root, self.perturbation, "nose", "test", "*.json")
                )
            )
        elif self.feasibility_type == "approaching infeasible":
            raw_fnames = glob.glob(
                os.path.join(
                    case_root, self.perturbation, "around_nose", "train", "*.json"
                )
            )

        data_list = []
        is_cpf_sample = True if self.feasibility_type != "feasible" else False
        for file in raw_fnames[:dataset_size]:
            with open(file, "r") as f:
                pm_case = json.load(f)
            data = self.build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)
            data_list.append(data)

        return data_list

    def get_shuffle_file_path(self, grid_type: str, case_root: str) -> str:
        """
        Return the path to the shuffle mapping JSON file for a given grid type.

        This utility determines which shuffle mapping to use when building 
        processed train/validation/test splits. For large-scale systems such 
        as case2000, a dedicated shuffle file is selected to match 
        dataset naming conventions.

        Parameters
        ----------
        grid_type : str
            Grid size category (e.g., 'small', 'medium', 'large') 
            used to locate the correct shuffle file.
        case_root : str
            Path to the case directory, used to detect special naming rules.

        Returns
        -------
        str
            Absolute path to the corresponding shuffle JSON file.
        """
        if "case2000" in case_root:
            return os.path.join(
                self.root, "shuffle_files", grid_type, "raw_shuffle_2000.json"
            )
        else:
            return os.path.join(
                self.root, "shuffle_files", grid_type, "raw_shuffle.json"
            )

    def shuffle_split_and_save_data(self, case_root: str, split: str) -> Dict[str, list]:
        """Create and save processed train/val/test splits for a given case.

        This method constructs processed PyTorch Geometric datasets for one
        power flow case by reading shuffle mappings, loading raw JSON samples,
        and converting each case to a `HeteroData` object. It handles both
        feasible and infeasible regimes and organizes processed data into
        per-split .pt files under structured folders.

        The function also returns in-memory lists of all processed samples,
        enabling downstream concatenation across multiple cases (e.g. for
        cross-case training or evaluation).

        Parameters
        ----------
        case_root : str
            Path to the directory containing all subfolders for a given
            power flow case (e.g. case14/n/raw/, case14/n-1/nose/train/).

        Returns
        -------
        Dict[str, list]
            Dictionary mapping split names ("train", "val", "test")
            to lists of processed `HeteroData` objects.

        Notes
        -----
        - For feasible data, split boundaries are derived from shuffle maps
        located in shuffle_files/ and follow a 90/10 train/val partition.
        The test set uses the last test_size samples.
        - For infeasible and approaching infeasible regimes, samples are
        directly loaded from their respective nose and around_nose folders.
        - Processed datasets are saved under:
        <case_root>/<grid_type>/processed/task_<task>_<feasibility>_<model>/split.pt.
        - The returned data lists can be concatenated to build cross-case datasets.
        """
        # when being combined
        task, model = self.task, self.model
        task_config = self.task_config[task]

        # Identify case name of specific case root ()
        current_case_name = os.path.basename(case_root)
        if current_case_name == "case2000":
            task_config = copy.deepcopy(task_config)
            for feas_values in task_config.values():
                for feas_type, feas_num in feas_values.items():
                    feas_values[feas_type] = feas_num // 2

        for feasibility, train_cfg_dict in task_config.items():
            feasibility_config = self.feasibility_config[feasibility]

            for grid_type in ["n", "n-1", "n-2"]:
                train_size = train_cfg_dict[grid_type]
                test_cfg = feasibility_config.get("test", {})
                test_size = test_cfg.get(grid_type) if test_cfg else 0

                if current_case_name == "case2000":
                    test_size = test_size // 2

                if train_size == 0 and test_size == 0:
                    continue

                # Create split dict with files corresponding to each feas type
                if feasibility == "feasible":
                    # Gather shuffle files
                    shuffle_path = self.get_shuffle_file_path(grid_type, case_root)
                    with open(shuffle_path, "r") as f:
                        shuffle_dict = json.load(f)
                    shuffle_map = {int(k): int(v) for k, v in shuffle_dict.items()}

                    # Parse shuffle file
                    raw_path = os.path.join(case_root, grid_type, "raw")
                    raw_fnames = [
                        os.path.join(raw_path, f"sample_{i + 1}.json")
                        for i in shuffle_map.keys()
                    ]
                    fnames_shuffled = [
                        raw_fnames[shuffle_map[i]] for i in shuffle_map.keys()
                    ]

                    # Verify train and test won't have overlap
                    entire_size = len(fnames_shuffled)
                    assert (train_size + test_size) <= entire_size, \
                        "Test samples overlap with train samples!"

                    # Split files by train, val, test
                    split_dict = {
                        "train": fnames_shuffled[:int(0.9 * train_size)],
                        "val": fnames_shuffled[
                            int(0.9 * train_size):int(train_size)
                        ],
                        "test": fnames_shuffled[-int(test_size):],
                    }  # always takes the last test_size samples for test set
                else:
                    infeasibility_type = (
                        "around_nose"
                        if feasibility == "approaching infeasible"
                        else "nose"
                    )
                    infeasible_train_path = os.path.join(
                        case_root, grid_type, infeasibility_type, "train"
                    )
                    infeasible_test_path = os.path.join(
                        case_root, grid_type, infeasibility_type, "test"
                    )

                    # Collect filenames
                    train_files = sorted(
                        [
                            os.path.join(infeasible_train_path, f)
                            for f in os.listdir(infeasible_train_path)
                            if f.endswith(".json")
                        ]
                    )[:train_size]
                    if infeasibility_type == "nose":
                        test_files = sorted(
                            [
                                os.path.join(infeasible_test_path, f)
                                for f in os.listdir(infeasible_test_path)
                                if f.endswith(".json")
                            ]
                        )[:test_size]
                    else:
                        test_files = None

                    # Create the split dictionary directly
                    split_idx = int(0.9 * len(train_files))
                    split_dict = {
                        "train": train_files[:split_idx],
                        "val": train_files[split_idx:],
                        "test": test_files,
                    }

                # Gather files from split dict
                files = split_dict[split]

                # For tasks that don't load from every folder
                if not files:
                    continue

                # Proces files
                print(
                    f"Processing split: {model} {task} " + \
                    f"{current_case_name} {grid_type} {feasibility} " + \
                    f"{split} ({len(files)} files)"
                )
                data_list = []
                for fname in tqdm(files, desc=f"Building {split} data"):
                    with open(fname, "r") as f:
                        pm_case = json.load(f)
                    data = self.build_heterodata(pm_case, is_cpf_sample=True)
                    data_list.append(data)

                # Collate and save
                data, slices = self.collate(data_list)
                processed_path = os.path.join(
                    case_root,
                    f"{grid_type}/processed/task_{task}_{feasibility}_{model}",
                )
                os.makedirs(processed_path, exist_ok=True)

                torch.save(
                    (data, slices), os.path.join(processed_path, f"{split}.pt")
                )

        return data_list

    def process(self):
        """
        This method serves as the main processing routine invoked by
        `InMemoryDataset`. It organizes how raw PowerModels cases are
        converted into processed `HeteroData` objects, split into
        train/validation/test partitions, and stored under structured
        directories.

        For the 'analysis' task, all samples are processed into a single
        all.pt file. For other tasks, this function either:
        - Processes one case and generates per-split .pt files, or
        - Combines multiple cases (tasks 3.x) into consolidated datasets
        for each split, saving them under combined processed folders.

        Notes
        -----
        - For 'analysis' tasks, processed data is stored under:
        <root>/processed/task_<task>_<case>_<perturbation>_<feasibility>_<n_samples>/all.pt.
        - For combined multi-case tasks (3.x), split datasets are saved under:
        <root>/processed/combined_task_<task>_<model>_<casename>/<split>.pt.
        - Internally uses shuffle_split_and_save_data and get_analysis_data 
        to build per-case datasets before collation.
        """
        task, model = self.task, self.model
        casename = None
        combined_data_lists = {"train": [], "val": [], "test": []}

        # determine roots based on task
        if task in [3.1, 3.2, 3.3]:
            case_roots = [
                os.path.join(self.root, case_name)
                for case_name in self.all_case_names
            ]
            casename = self.case_name if task == 3.1 else ""
        else:
            case_roots = [os.path.join(self.root, self.case_name)]
            casename = self.case_name

        if task == "analysis":  # no need to combine anything here
            print(f"Processing data for task {task}")
            task_data_list = self.get_analysis_data(
                os.path.join(self.root, self.case_name)
            )
            data, slices = self.collate(task_data_list)
            processed_path = os.path.join(
                self.root,
                "processed",
                f"task_{self.task}_{self.case_name}_{self.perturbation}_{self.feasibility_type}_{self.n_samples}",
            )

            if not os.path.exists(processed_path):
                os.makedirs(processed_path)

            torch.save((data, slices), os.path.join(processed_path, "all.pt"))
            return

        # First, process each root and collect all data
        for case_root in case_roots:  # loops over cases
            print(f"Processing combined data for task {task}")

            # Add data from this root to the combined lists
            for split in combined_data_lists.keys():
                if task in [3.1, 3.2, 3.3]:
                    # extract case name from case_root
                    case_name = os.path.basename(case_root)
                    if case_name not in self.task_split_config[task][split]:
                        continue
                data_list = self.shuffle_split_and_save_data(case_root, split)
                combined_data_lists[split].extend(data_list)

        for split, data_list in combined_data_lists.items():
            if data_list:  # Only process if we have data
                print(f"Collating combined {split} data with {len(data_list)} samples")
                combined_data, combined_slices = self.collate(data_list)

                # Create a separate directory for the concatenated data
                concat_path = os.path.join(
                    self.root, f"processed/combined_task_{task}_{model}_{casename}"
                )

                if not os.path.exists(concat_path):
                    os.makedirs(concat_path)

                torch.save(
                    (combined_data, combined_slices),
                    os.path.join(concat_path, f"{split}.pt"),
                )
                print(f"Saved combined {split} data with {len(data_list)} samples")

    def load(self, split: str):
        """
        Loads dataset for the specified split.

        Parameters
        ----------
        split : str
            The split to load. Can be one of:
                - 'train', 'val', 'test'
                - 'separate_{casename}_{split}_{feasibility}_{grid_type}' 
                (used for task 3.1 or 4.x separate per-case loading)

        Raises
        ------
        ValueError
            If the split string is malformed.
        FileNotFoundError
            If the processed file is not found at the computed path.

        Notes
        -----
        For example:
        split="separate_case57_train_feasible_n-1" loads:
        root/case57/n-1/processed/task_{task}_{feasibility}_{model}/train.pt.
        """

        if "separate" in split:
            # split should be of format "separate_{split}_{feasibility}_{grid_type}_{casename}"
            _, casename_str, split_str, feasibility_str, grid_type_str = split.split(
                "_"
            )
            if feasibility_str == "near infeasible":
                processed_path = os.path.join(
                    self.root,
                    f"{casename_str}",
                    f"{grid_type_str}",
                    "processed",
                    f"task_{self.task}_{feasibility_str}_{self.model}",
                    f"{split_str}.pt",
                )
            else:
                processed_path = os.path.join(
                    self.root,
                    f"{casename_str}",
                    f"{grid_type_str}",
                    "processed",
                    f"task_{self.task}_{feasibility_str}_{self.model}",
                    f"{split_str}.pt",
                )
            print(f"Loading {split} dataset from {processed_path}")
            self.data, self.slices = torch.load(processed_path)
        else:
            processed_path = os.path.join(self.processed_dir, f"{split}.pt")
            print(f"Loading {split} dataset from {processed_path}")
            self.data, self.slices = torch.load(processed_path)

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
