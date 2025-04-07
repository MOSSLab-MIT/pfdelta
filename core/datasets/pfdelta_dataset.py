import json
import os
import torch
import random
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, HeteroData

class PFDeltaDataset(InMemoryDataset):
    def __init__(self, root='data', split='train', transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.split = split
        self.force_reload = force_reload
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[self._split_to_idx()]) 

    def _split_to_idx(self):
        return {'train': 0, 'val': 1, 'test': 2}[self.split]

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.json')])

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def build_heterodata(self, pm_case):
        data = HeteroData()

        network_data = pm_case['network']
        solution_data = pm_case['solution']['solution']

        PQ_bus_x, PQ_bus_y = [], []
        PV_bus_x, PV_bus_y = [], []
        PV_demand, PV_generation = [], []
        slack_x, slack_y = [], []
        slack_demand, slack_generation = [], []
        bus_x = []

        PV_to_bus, PQ_to_bus, slack_to_bus = [], [], []
        pq_idx, pv_idx, slack_idx = 0, 0, 0

        for bus_id_str, bus in sorted(network_data['bus'].items(), key=lambda x: int(x[0])):
            bus_id = int(bus_id_str)
            bus_idx = bus_id - 1
            bus_sol = solution_data['bus'][bus_id_str]

            # Shunts 
            gs, bs = 0.0, 0.0
            for shunt in network_data['shunt'].values():
                if int(shunt['shunt_bus']) == bus_id:
                    gs += shunt['gs']
                    bs += shunt['bs']
            bus_x.append(torch.tensor([gs, bs]))

            # Load
            pd, qd = 0.0, 0.0
            for load in network_data['load'].values():
                if int(load['load_bus']) == bus_id:
                    pd += load['pd']
                    qd += load['qd']

            # Gen
            pg, qg = 0.0, 0.0
            for gen_id, gen in network_data['gen'].items():
                if int(gen['gen_bus']) == bus_id:
                    gen_sol = solution_data['gen'].get(gen_id)
                    if gen_sol:
                        pg += gen_sol['pg']
                        qg += gen_sol['qg']
                    else:
                        assert gen['gen_status'] == 0, f"Expected gen {gen_id} to be off."

            # Node features
            va, vm = bus_sol['va'], bus_sol['vm']
            if bus['bus_type'] == 1:
                PQ_bus_x.append(torch.tensor([pd, qd]))
                PQ_bus_y.append(torch.tensor([va, vm]))
                PQ_to_bus.append(torch.tensor([pq_idx, bus_idx]))
                pq_idx += 1
            elif bus['bus_type'] == 2:
                PV_bus_x.append(torch.tensor([pg - pd, vm]))
                PV_bus_y.append(torch.tensor([qg - qd, va]))
                PV_demand.append(torch.tensor([pd, qd]))
                PV_generation.append(torch.tensor([pg, qg]))
                PV_to_bus.append(torch.tensor([pv_idx, bus_idx]))
                pv_idx += 1
            elif bus['bus_type'] == 3:
                slack_x.append(torch.tensor([va, vm]))
                slack_y.append(torch.tensor([pg - pd, qg - qd]))
                slack_demand.append(torch.tensor([pd, qd]))
                slack_generation.append(torch.tensor([pg, qg]))
                slack_to_bus.append(torch.tensor([slack_idx, bus_idx]))
                slack_idx += 1

        # Edges
        edge_index, edge_attr, edge_label = [], [], []
        for branch_id_str, branch in sorted(network_data['branch'].items(), key=lambda x: int(x[0])):
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
            if branch_sol:
                edge_label.append(torch.tensor([
                    branch_sol['pf'], branch_sol['qf'],
                    branch_sol['pt'], branch_sol['qt']
                ]))
            else:
                assert branch['br_status'] == 0, f"Expected branch {branch_id_str} to be outaged."

        # Create graph nodes and edges
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

        data['bus'].x = torch.stack(bus_x) 

        data['bus', 'branch', 'bus'].edge_index = torch.stack(edge_index, dim=1) 
        data['bus', 'branch', 'bus'].edge_attr = torch.stack(edge_attr) 
        data['bus', 'branch', 'bus'].edge_label = torch.stack(edge_label) 

        for link_name, edges in {
            ('PV', 'PV_link', 'bus'): PV_to_bus,
            ('PQ', 'PQ_link', 'bus'): PQ_to_bus,
            ('slack', 'slack_link', 'bus'): slack_to_bus
        }.items():
            edge_tensor = torch.stack(edges, dim=1) 
            data[link_name].edge_index = edge_tensor
            data[(link_name[2], link_name[1], link_name[0])].edge_index = edge_tensor.flip(0)

        return data

    def process(self):
        fnames = self.raw_file_names
        random.shuffle(fnames)
        n = len(fnames)

        split_dict = {
            'train': fnames[:int(0.8 * n)],
            'val': fnames[int(0.8 * n): int(0.9 * n)],
            'test': fnames[int(0.9 * n):]
        }

        for split, files in split_dict.items():
            data_list = []
            print(f"Processing split: {split} ({len(files)} files)")
            for fname in tqdm(files, desc=f"Building {split} data"):
                with open(os.path.join(self.raw_dir, fname)) as f:
                    pm_case = json.load(f)
                data = self.build_heterodata(pm_case)
                data_list.append(data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), os.path.join(self.processed_dir, f'{split}.pt'))
