import json
import os
import torch
import random
from tqdm import tqdm
import numpy as np
from typing import Callable, Sequence, Any
from collections import defaultdict

from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric.data.collate import collate

from core.utils.registry import registry


# TODO: make sure to implement the logic for all the "nose" cases too
@registry.register_dataset("pfdeltadata")
class PFDeltaDataset(InMemoryDataset):
    def __init__(self, root_dir='data', case_name='', split='train', model='', task=1.1, add_bus_type=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.split = split
        self.case_name = case_name
        self.force_reload = force_reload
        self.add_bus_type = add_bus_type
        self.task = task
        self.model = model
        self.root = os.path.join(root_dir)
        if task in [3.2, 3.3]: # TODO: add a flag to specify the specific processed dir!
            self._custom_processed_dir = os.path.join(self.root, "processed", f"combined_task_{task}_{model}_")
        else: 
            self._custom_processed_dir = os.path.join(self.root, "processed", f"combined_task_{task}_{model}_{self.case_name}")

        self.all_case_names = ["case57_seeds", "case118_seeds", "case500_seeds", "case2000_seeds"]

        self.task_config = {
            1.1: {"feasible":{"none": 54000, "n-1": 0, "n-2": 0}},
            1.2: {"feasible":{"none": 27000, "n-1": 27000, "n-2": 0}},
            1.3: {"feasible":{"none": 18000, "n-1": 18000, "n-2": 18000}}, # FEASIBILITY FOR TEST NOT INCLUDED
            2.1: {"feasible":{"none": 18000, "n-1": 18000, "n-2": 18000}},
            2.2: {"feasible":{"none": 12000, "n-1": 12000, "n-2": 12000}},
            2.3: {"feasible":{"none": 6000,  "n-1": 6000,  "n-2": 6000}},
            3.1: {"feasible":{"none": 18000, "n-1": 18000, "n-2": 18000}},
            3.2: {"feasible":{"none": 18000, "n-1": 18000, "n-2": 18000}},
            3.3: {"feasible":{"none": 18000, "n-1": 18000, "n-2": 18000}},
            4.1: {"feasible": {}, "near infeasible": {}}, 
            4.2: {"feasible": {}, "approaching infeasible": {}, "near infeasible": {}},
        } 
        self.feasibility_config = {
            "feasible": {
                "none": 56000,
                "n-1": 29000,
                "n-2": 20000,
                "test": {"none": 2000, "n-1": 2000, "n-2": 2000}
            },
            "approaching infeasible": {
                "none": 7200,
                "n-1": 7200,
                "n-2": 7200,
                "test": None  # no test set for this regime
            },
            "near infeasible": {
                "none": 2000,
                "n-1": 2000,
                "n-2": 2000,
                "test": {"none": 200, "n-1": 200, "n-2": 200}
            },
        }

        super().__init__(self.root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.split)

    def _split_to_idx(self):
        return {'train': 0, 'val': 1, 'test': 2}[self.split]

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.json')])
    
    @property
    def processed_dir(self):
        return self._custom_processed_dir

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def build_heterodata(self, pm_case):
        data = HeteroData()

        network_data = pm_case['network']
        solution_data = pm_case['solution']['solution']

        # Bus nodes
        pf_x, pf_y = [], []
        bus_voltages = []
        bus_type = []
        bus_shunts = []
        bus_gen, bus_demand = [], []

        PQ_bus_x, PQ_bus_y = [], []
        PV_bus_x, PV_bus_y = [], []
        PV_demand, PV_generation = [], []
        slack_x, slack_y = [], []
        slack_demand, slack_generation = [], []
        PV_to_bus, PQ_to_bus, slack_to_bus = [], [], []
        pq_idx, pv_idx, slack_idx = 0, 0, 0

        for bus_id_str, bus in sorted(network_data['bus'].items(), key=lambda x: int(x[0])):
            bus_id = int(bus_id_str)
            bus_idx = bus_id - 1
            bus_sol = solution_data['bus'][bus_id_str]
            
            va, vm = bus_sol['va'], bus_sol['vm']
            bus_voltages.append(torch.tensor([va, vm]))

            # Shunts 
            gs, bs = 0.0, 0.0
            for shunt in network_data['shunt'].values():
                if int(shunt['shunt_bus']) == bus_id:
                    gs += shunt['gs']
                    bs += shunt['bs']
            bus_shunts.append(torch.tensor([gs, bs]))

            # Load
            pd, qd = 0.0, 0.0
            for load in network_data['load'].values():
                if int(load['load_bus']) == bus_id:
                    pd += load['pd']
                    qd += load['qd']

            bus_demand.append(torch.tensor([pd, qd]))

            # Gen
            pg, qg = 0.0, 0.0
            for gen_id, gen in sorted(network_data['gen'].items(), key=lambda x: int(x[0])):
                if int(gen['gen_bus']) == bus_id: 
                    if gen['gen_status'] == 1:
                        gen_sol = solution_data['gen'][gen_id]
                        pg += gen_sol['pg']
                        qg += gen_sol['qg']
                    else:
                        assert solution_data['gen'].get(gen_id) is None, f"Expected gen {gen_id} to be off."

            bus_gen.append(torch.tensor([pg, qg]))

            # Now decide final bus type
            bus_type_now = bus['bus_type']

            if bus_type_now == 2 and pg == 0.0 and qg == 0.0:
                bus_type_now = 1  # PV bus with no gen --> becomes PQ
                # maybe add an assert here to check if all gens were off.

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

        generation, limits, slack_gen  = [], [], []

        # Generator nodes   
        for gen_id, gen in sorted(network_data['gen'].items(), key=lambda x: int(x[0])):
            if gen['gen_status'] == 1:
                gen_sol = solution_data['gen'][gen_id] 
                pmin, pmax, qmin, qmax = gen['pmin'], gen['pmax'], gen['qmin'], gen['qmax']
                pgen, qgen = gen_sol['pg'], gen_sol['qg']
                limits.append(torch.tensor([pmin, pmax, qmin, qmax]))
                generation.append(torch.tensor([pgen, qgen]))
                is_slack = torch.tensor(
                        1 if network_data['bus'][str(gen['gen_bus'])]['bus_type'] == 3 else 0,
                        dtype=torch.bool
                                )
                slack_gen.append(is_slack)
            else:
                assert solution_data['gen'].get(gen_id) is None, f"Expected gen {gen_id} to be off."

        # Load nodes
        demand = []
        for load_id, load in sorted(network_data['load'].items(), key=lambda x: int(x[0])):
            pd, qd = load['pd'], load['qd']
            demand.append(torch.tensor([pd, qd]))

        # Edges
        # bus to bus edges
        edge_index, edge_attr, edge_label = [], [], []
        for branch_id_str, branch in sorted(network_data['branch'].items(), key=lambda x: int(x[0])):
            if branch['br_status'] == 0:
                continue  # Skip inactive branches

            from_bus = int(branch['f_bus']) - 1 
            to_bus = int(branch['t_bus']) - 1
            edge_index.append(torch.tensor([from_bus, to_bus]))
            edge_attr.append(torch.tensor([
                branch['br_r'], branch['br_x'],
                branch['g_fr'], branch['b_fr'],
                branch['g_to'], branch['b_to'], 
                branch['tap'],  branch['shift']
            ]))

            branch_sol = solution_data['branch'].get(branch_id_str)
            assert branch_sol is not None, f"Missing solution for active branch {branch_id_str}"

            if branch_sol:
                edge_label.append(torch.tensor([
                    branch_sol['pf'], branch_sol['qf'],
                    branch_sol['pt'], branch_sol['qt']
                ]))

        # bus to gen edges
        gen_to_bus_index = []
        for gen_id, gen in sorted(network_data['gen'].items(), key=lambda x: int(x[0])):
            if gen['gen_status'] == 1:
                gen_bus = torch.tensor(gen['gen_bus']) - 1
                gen_to_bus_index.append(torch.tensor([int(gen_id) - 1, gen_bus]))

        # bus to load edges
        load_to_bus_index = []
        for load_id, load in sorted(network_data['load'].items(), key=lambda x: int(x[0])):
            load_bus = torch.tensor(load['load_bus']) - 1
            load_to_bus_index.append(torch.tensor([int(load_id) - 1, load_bus]))

        # Create graph nodes and edges
        data['bus'].x = torch.stack(pf_x)
        data['bus'].y = torch.stack(pf_y)
        data['bus'].bus_gen = torch.stack(bus_gen) # aggregated
        data['bus'].bus_demand = torch.stack(bus_demand) # aggregated
        data['bus'].bus_voltages = torch.stack(bus_voltages)
        data['bus'].bus_type = torch.stack(bus_type)
        data['bus'].shunt = torch.stack(bus_shunts)

        data['gen'].limits = torch.stack(limits)
        data['gen'].generation = torch.stack(generation)
        data['gen'].slack_gen = torch.stack(slack_gen)

        data['load'].demand = torch.stack(demand)

        if self.add_bus_type:
            data['PQ'].x = torch.stack(PQ_bus_x) 
            data['PQ'].y = torch.stack(PQ_bus_y)

            data['PV'].x = torch.stack(PV_bus_x) 
            data['PV'].y = torch.stack(PV_bus_y) 
            data['PV'].generation = torch.stack(PV_generation) 
            data['PV'].demand = torch.stack(PV_demand) 

            data['slack'].x = torch.stack(slack_x) 
            data['slack'].y = torch.stack(slack_y)
            data['slack'].generation = torch.stack(slack_generation) 
            data['slack'].demand = torch.stack(slack_demand)         

        for link_name, edges in {
            ('bus', 'branch', 'bus'): edge_index,
            ('gen', 'gen_link', 'bus'): gen_to_bus_index,
            ('load', 'load_link', 'bus'): load_to_bus_index
        }.items():
            edge_tensor = torch.stack(edges, dim=1) 
            data[link_name].edge_index = edge_tensor
            if link_name != ('bus', 'branch', 'bus'):
                data[(link_name[2], link_name[1], link_name[0])].edge_index = edge_tensor.flip(0)
            if link_name == ('bus', 'branch', 'bus'): 
                data[link_name].edge_attr = torch.stack(edge_attr) 
                data[link_name].edge_label = torch.stack(edge_label) 
        
        if self.add_bus_type:
            for link_name, edges in {
                ('PV', 'PV_link', 'bus'): PV_to_bus,
                ('PQ', 'PQ_link', 'bus'): PQ_to_bus,
                ('slack', 'slack_link', 'bus'): slack_to_bus
            }.items():
                edge_tensor = torch.stack(edges, dim=1) 
                data[link_name].edge_index = edge_tensor
                data[(link_name[2], link_name[1], link_name[0])].edge_index = edge_tensor.flip(0)

        return data

    def shuffle_split_and_save_data(self, root): 
        # TODO: implement logic for what train and test splits to save for a given task so that it gets saved correctly later
        # when being combined
        task, model = self.task, self.model 
        task_config = self.task_config[task]

        # create dicts to store all data lists per task for later concatenation
        all_data_lists = {
            'train': [],
            'val': [],
            'test': []
        }

        for feasibility, train_cfg_dict in task_config.items():
            feasibility_config = self.feasibility_config[feasibility]

            for grid_type in ["none", "n-1", "n-2"]:
                train_size = train_cfg_dict[grid_type]
                test_cfg = feasibility_config.get("test", {})
                test_size = test_cfg.get(grid_type) if test_cfg else 0

                shuffle_path = os.path.join(root, grid_type, "raw_shuffle.json")
                with open(shuffle_path, "r") as f:
                    shuffle_dict = json.load(f)
                shuffle_map = {int(k): int(v) for k, v in shuffle_dict.items()}

                raw_path = os.path.join(root, grid_type, "raw")
                raw_fnames = [os.path.join(raw_path, f"sample_{i+1}.json") for i in shuffle_map.keys()]
                fnames_shuffled = [raw_fnames[shuffle_map[i]] for i in shuffle_map.keys()]

                # extend the lists instead of overwriting
                split_dict = {
                        'train': fnames_shuffled[:int(0.9 * train_size)],
                        'val': fnames_shuffled[int(0.9 * train_size):int(train_size)], # this is optional!
                        'test': fnames_shuffled[-int(test_size):],
                    }

                for split, files in split_dict.items():
                    data_list = []
                    print(f"Processing split: {model} {task} {grid_type} {split} ({len(files)} files)")
                    for fname in tqdm(files, desc=f"Building {split} data"):
                        with open(fname, "r") as f:
                            pm_case = json.load(f)
                        data = self.build_heterodata(pm_case)
                        data_list.append(data)

                    data, slices = self.collate(data_list)
                    processed_path = os.path.join(root, f"{grid_type}/processed/task_{task}_{feasibility}_{model}")
                    
                    if not os.path.exists(os.path.join(root, f"{grid_type}/processed")):
                        os.mkdir(os.path.join(root, f"{grid_type}/processed"))
                    if not os.path.exists(processed_path):
                        os.mkdir(processed_path)

                    torch.save((data, slices), os.path.join(processed_path, f'{split}.pt'))
                    all_data_lists[split].extend(data_list)
        
        return all_data_lists

    def process(self):
        task, model = self.task, self.model
        casename = None
        combined_data_lists = {
            'train': [],
            'val': [],
            'test': []
        }

        # determine roots based on task
        if task in [3.1, 3.2, 3.3]:
            roots = [os.path.join(self.root, case_name) for case_name in self.all_case_names]
            casename = self.case_name if task == 3.1 else ""
            # for tasks with multiple case names, we need to aggregate data from all cases
        else:
            roots = [os.path.join(self.root, self.case_name)]
            casename = self.case_name
        
        # First, process each root and collect all data
        for root in roots:
            print(f"Processing combined data for task {task}")
            task_data_lists = self.shuffle_split_and_save_data(root)
            
            # Add data from this root to the combined lists
            for split in combined_data_lists.keys():
                combined_data_lists[split].extend(task_data_lists[split])
        
        for split, data_list in combined_data_lists.items():
            if data_list:  # Only process if we have data
                print(f"Collating combined {split} data with {len(data_list)} samples")
                combined_data, combined_slices = self.collate(data_list)
                
                # Create a separate directory for the concatenated data
                concat_path = os.path.join(self.root, f"processed/combined_task_{task}_{model}_{casename}")
                
                if not os.path.exists(os.path.join(self.root, "processed")):
                    os.makedirs(os.path.join(self.root, "processed"))
                if not os.path.exists(concat_path):
                    os.makedirs(concat_path)
                    
                torch.save((combined_data, combined_slices), os.path.join(concat_path, f'{split}.pt'))
                print(f"Saved combined {split} data with {len(data_list)} samples")

    def load(self, split):
        """Loads dataset for the specified split.
        
        Args:
            split (str): The split to load ('train', 'val', or 'test')
        """
        processed_path = os.path.join(self.processed_dir, f"{split}.pt")
        print(f"Loading {split} dataset from {processed_path}")
        self.data, self.slices = torch.load(processed_path)


@registry.register_dataset("pfdeltaGNS")
class PFDeltaGNS(PFDeltaDataset): 
    def __init__(self, root_dir='data', case_name='', split='train', model="GNS", task=1.1, add_bus_type=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root_dir, case_name, split, model, task, add_bus_type, transform, pre_transform, pre_filter, force_reload)

    def build_heterodata(self, pm_case):
        # call base version
        data = super().build_heterodata(pm_case)
        num_buses = data['bus'].x.size(0)
        num_gens = data['gen'].generation.size(0)
        num_loads = data['load'].demand.size(0)

        # Init bus-level fields
        v_buses      = torch.zeros(num_buses)
        theta_buses  = torch.zeros(num_buses)
        pd_buses     = torch.zeros(num_buses)
        qd_buses     = torch.zeros(num_buses)
        pg_buses     = torch.zeros(num_buses)
        qg_buses     = torch.zeros(num_buses)

        # Read bus types
        bus_types = data['bus'].bus_type 
        x_gns = torch.zeros((num_buses, 2))

        for bus_idx in range(num_buses):
            bus_type = bus_types[bus_idx].item()
            pf_x = data['bus'].x[bus_idx]
            pf_y = data['bus'].x[bus_idx]
            bus_demand = data['bus'].bus_demand[bus_idx]
            bus_gen = data['bus'].bus_gen[bus_idx]

            if bus_type == 1:  # PQ bus
                # Flat start for PQ bus
                x_gns[bus_idx] = torch.tensor([1.0, 0.0])
                pd = pf_x[0]
                qd = pf_x[1]
                pg = bus_gen[0]
                qg = bus_gen[1]
            elif bus_type == 2:  # PV bus
                v = pf_x[1]
                theta = torch.tensor(0.0)
                x_gns[bus_idx] = torch.stack([v, theta])
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

            v_buses[bus_idx] = x_gns[bus_idx][0]
            theta_buses[bus_idx] = x_gns[bus_idx][1]
            pd_buses[bus_idx] = pd
            qd_buses[bus_idx] = qd
            pg_buses[bus_idx] = pg
            qg_buses[bus_idx] = qg

        # Store in bus
        data['bus'].x_gns = x_gns
        data['bus'].v = v_buses
        data['bus'].theta = theta_buses
        data['bus'].pd = pd_buses
        data['bus'].qd = qd_buses
        data['bus'].pg = pg_buses
        data['bus'].qg = qg_buses
        data['bus'].delta_p = torch.zeros_like(v_buses)
        data['bus'].delta_q = torch.zeros_like(v_buses)
        data['gen'].num_nodes = num_gens
        data['load'].num_nodes = num_loads

        return data


@registry.register_dataset("pfdeltaCANOS")
class PFDeltaCANOS(PFDeltaDataset): 
    def __init__(self, root_dir='data', case_name='', split='train', model="CANOS", task=1.1, add_bus_type=True, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root_dir, case_name, split,  model, task, add_bus_type, transform, pre_transform, pre_filter, force_reload)

    def build_heterodata(self, pm_case):
        # call base version
        data = super().build_heterodata(pm_case)

        # Now prune the data to only keep bus, PV, PQ, slack
        keep_nodes = {"bus", "PV", "PQ", "slack"}

        for node_type in list(data.node_types):
            if node_type not in keep_nodes:
                del data[node_type]

        for edge_type in list(data.edge_types):
            src, _, dst = edge_type
            if src not in keep_nodes or dst not in keep_nodes:
                del data[edge_type]

        return data


@registry.register_dataset("pfdeltaPFNet")
class PFDeltaPFNet(PFDeltaDataset): 
    def __init__(self, root_dir='data', case_name='', split='train', model="PFNet", task=1.1, add_bus_type=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root_dir, case_name, split,  model, task, add_bus_type, transform, pre_transform, pre_filter, force_reload)

    def build_heterodata(self, pm_case):
        # call base version
        data = super().build_heterodata(pm_case)

        num_buses = data['bus'].x.size(0)
        bus_types = data['bus'].bus_type
        pf_x = data['bus'].x
        pf_y = data['bus'].y
        shunts = data['bus'].shunt
        num_gens = data['gen'].generation.size(0)
        num_loads = data['load'].demand.size(0)

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
            if bus_type == 1:   # PQ
                pred_mask = torch.tensor([1, 1, 0, 0, 0, 0])
                va, vm =  pf_y[i] 
                pd, qd = pf_x[i]
                input_mask = (1 - pred_mask).float()
                input_feats = torch.tensor([vm, va, pd, qd, gs, bs]) * input_mask
                features = torch.cat([one_hot, input_feats])
                y = torch.tensor([vm, va, pd, qd, gs, bs])
            elif bus_type == 2: # PV
                pred_mask = torch.tensor([0, 1, 0, 1, 0, 0])
                pg_pd, vm =  pf_x[i]
                qg_qd, va = pf_y[i]
                input_mask = (1 - pred_mask).float()
                input_feats = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs]) * input_mask
                features = torch.cat([one_hot, input_feats])
                y = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs])
            elif bus_type == 3: # Slack
                pred_mask = torch.tensor([0, 0, 1, 1, 0, 0])
                va, vm =  pf_x[i]
                pg_pd, qg_qd = pf_y[i]
                input_mask = (1 - pred_mask).float()
                input_feats = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs]) * input_mask
                features = torch.cat([one_hot, input_feats])
                y = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs])

            x_pfnet.append(torch.cat([features, pred_mask]))
            y_pfnet.append(y)

        x_pfnet = torch.stack(x_pfnet)  # shape [N, 4+6+6=16]
        y_pfnet = torch.stack(y_pfnet)  # shape [N, 6]

        if self.split == 'train':
            # Strip one-hot and pred_mask
            x_cont = x_pfnet[:, 4:10]  # shape [N, 6]
            y_cont = y_pfnet           # shape [N, 6]

            xy = torch.cat([x_cont, y_cont], dim=0)
            mean = xy.mean(dim=0, keepdim=True)
            std = xy.std(dim=0, keepdim=True) + 1e-7

            self.norm_mean = mean
            self.norm_std = std

            x_cont_norm = (x_cont - mean) / std
            y_norm = (y_cont - mean) / std

            x_normalized = torch.cat([x_pfnet[:, :4], x_cont_norm, x_pfnet[:, 10:]], dim=1)

            data['bus'].x = x_normalized
            data['bus'].y = y_norm
        else:
            data['bus'].x = x_pfnet
            data['bus'].y = y_pfnet

        data['gen'].num_nodes = num_gens
        data['load'].num_nodes = num_loads

        edge_attrs = []
        for attr in data['bus', 'branch', 'bus'].edge_attr:
            r, x = attr[0], attr[1]
            b = attr[2] + attr[4]
            tau, angle = attr[6], attr[7]
            edge_attrs.append(torch.tensor([r, x, b, tau, angle]))

        data['bus', 'branch', 'bus'].edge_attr = torch.stack(edge_attrs)

        return data
