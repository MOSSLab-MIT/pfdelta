import json
import os
import torch
import random
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, HeteroData

class PFDeltaDataset(InMemoryDataset):
    def __init__(self, root_dir='data', case_name='', split='train', add_bus_type=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.split = split
        self.case_name = case_name
        self.force_reload = force_reload
        self.add_bus_type = add_bus_type
        root = os.path.join(root_dir, case_name)
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[self._split_to_idx()]) 

    def _split_to_idx(self):
        return {'train': 0, 'val': 1, 'test': 2, 'all': 3}[self.split]

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.json')])

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt', 'all.pt']

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

    def process(self):
        fnames = self.raw_file_names
        random.shuffle(fnames)
        n = len(fnames)

        split_dict = {
            'train': fnames[:int(0.8 * n)],
            'val': fnames[int(0.8 * n): int(0.9 * n)],
            'test': fnames[int(0.9 * n):],
            'all': fnames  # 👈 this uses all samples
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

