import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.optim as optim
import json
import os
import torch
import random
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, HeteroData
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import Adam


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


class PFDeltaCANOS(PFDeltaDataset): 
    def __init__(self, root_dir='data', case_name='', split='train', add_bus_type=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root_dir, case_name, split, add_bus_type, transform, pre_transform, pre_filter, force_reload)

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


# Encoder
class Encoder(nn.Module):
    def __init__(self, data, hidden_size: int):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # Linear projection for all node features
        self.node_projections = nn.ModuleDict({
            node_type: nn.Linear(data.num_node_features[node_type], hidden_size)
            for node_type in data.num_node_features.keys()
        })
        # Linear projection for all edge features
        self.edge_projections = nn.ModuleDict({
            str(edge_type): nn.Linear(data.num_edge_features[edge_type], hidden_size)
            for edge_type in data.num_edge_features.keys() if data.num_edge_features[edge_type] != 0
            # not including subnode links which have no attributes.
        })
    

    def forward(self, data):
        device = data["bus"].x.device
        projected_nodes = {
            node_type: self.node_projections[node_type](data[node_type].x)
            for node_type in data.num_node_features.keys()
        }

        projected_edges = {}
        for edge_type in data.edge_types:
            if "edge_attr" in data[edge_type]:
                projected_edges[str(edge_type)] = self.edge_projections[str(edge_type)](data[edge_type].edge_attr)
            elif edge_type[2] != "bus":
                num_edges = data[edge_type]['edge_index'].shape[1]
                projected_edges[str(edge_type)] = torch.zeros((num_edges, self.hidden_size), device=device)

        return projected_nodes, projected_edges


# Interaction Network Module
class InteractionNetwork(nn.Module):
    def __init__(self, edge_type_dict, node_type_dict, edge_dim, node_dim, hidden_dim, include_sent_messages=False):
        """
        PyTorch implementation of the Interaction Network.
        Args:
            projected_edges (dict): Dictionary of projected edge features.
            projected_nodes (dict): Dictionary of projected node features.
            edge_dim (int): Dimension of edge features.
            node_dim (int): Dimension of node features.
            hidden_dim (int): Hidden layer size.
            include_sent_messages (bool): Whether to include messages from sender edges in node update.
        """
        super().__init__()
        self.include_sent_messages = include_sent_messages
        self.edge_update = EdgeUpdate(edge_dim, node_dim, hidden_dim, edge_type_dict)
        self.node_update = NodeUpdate(node_dim, hidden_dim, node_type_dict, self.include_sent_messages)

    def forward(self, nodes, edges, data):
        """
        Forward pass of the Interaction Network.
        Args:
            nodes (Dict): !!!!
            edges (Dict): !!!!
            senders (Tensor): Indices of sender nodes [num_edges].
            receivers (Tensor): Indices of receiver nodes [num_edges].
        Returns:
            Updated nodes and edges.
        """
        # Apply marshalling and relational model phi_r (edge update)
        # phi_r is applied onto src node features, dst node features, and edges
        device = data["bus"].x.device
        edge_hidden_dim = (edges.get("('bus', 'ac_line', 'bus')") or edges.get("('bus', 'branch', 'bus')")).shape[-1]
        sent_received_node_type = {node_type: torch.zeros(n.shape[0], edge_hidden_dim, device=device) for node_type, n in nodes.items()}
        updated_nodes_dict = {}
        updated_edges_dict = {}

        for edge_type, edge_feats in edges.items():
            edge_type_tuple = tuple(edge_type.strip("()").replace("'", "").split(", "))
            sender_type, receiver_type = edge_type_tuple[0], edge_type_tuple[2]
            if sender_type != "bus":
                continue
            senders, receivers = data[edge_type_tuple].edge_index

            # Gather node features
            sender_features = nodes[sender_type][senders]
            receiver_features = nodes[receiver_type][receivers]

            # Calculate edge updates
            updated_edges = self.edge_update(edge_feats, sender_features, receiver_features, edge_type)

            # Pass messages
            sent_received_node_type[receiver_type].scatter_add_(0, receivers.unsqueeze(-1).expand_as(updated_edges), updated_edges)
            if self.include_sent_messages:
                sent_received_node_type[sender_type].scatter_add_(0, senders.unsqueeze(-1).expand_as(updated_edges), updated_edges)

            updated_edges_dict[edge_type] = updated_edges + edge_feats

        # Apply the object model phi_o (node_update)
        # phi_o is applied to the aggregated edge features (with sent and recieved messages)
        for node_type, node_feats in nodes.items():
            updated_nodes = self.node_update(node_feats, sent_received_node_type[node_type], node_type)
            updated_nodes_dict[node_type] = updated_nodes + node_feats

        return updated_nodes_dict, updated_edges_dict


# Relational and Object models (phi_r and phi_o)
class EdgeUpdate(nn.Module):
    def __init__(self, edge_dim, node_dim, hidden_dim, edge_type_dict):
        """
        Edge update function for updating edge features.
        Args:
            edge_dim (int): Dimension of edge features.
            node_dim (int): Dimension of node features.
            hidden_dim (int): Hidden layer size.
            out_dim (int): Output edge feature dimension.
        """
        super().__init__()
        self.mlps = nn.ModuleDict({edge_type: nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        ) for edge_type in edge_type_dict.keys()})

    def forward(self, edges, sender_features, receiver_features, edge_type):
        """
        Compute updated edge features.
        Args:
            edges (Tensor): Shape [num_edges, edge_feat_dim].
            sender_features (Tensor): Shape [num_edges, node_feat_dim].
            receiver_features (Tensor): Shape [num_edges, node_feat_dim].
        Returns:
            Tensor: Updated edge features of shape [num_edges, out_dim].
        """
        x = torch.cat([edges, sender_features, receiver_features], dim=-1)
        return self.mlps[edge_type](x)


class NodeUpdate(nn.Module):
    def __init__(self, node_dim, hidden_dim, node_type_dict, include_sent_messages=False):
        """
        Node update module for updating node features.
        Args:
            input_dim (int): Dimension of node features.
            output_dim (int): Output node feature dimension.
            include_sent_messages (bool): Whether to include messages from sender edges
        """
        super().__init__()
        self.include_sent_messages = include_sent_messages
        self.mlps = nn.ModuleDict({node_type: nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        ) for node_type in node_type_dict.keys()})

    def forward(self, node_features, updated_messages, node_type):
        """
        Compute updated node features.
        Args:
            node_features (Tensor): Shape [num_nodes, node_feat_dim].
            received_messages (Tensor): Shape [num_nodes, node_feat_dim].
            sent_messages (Tensor, optional): Shape [num_nodes, node_feat_dim].
        Returns:
            Tensor: Updated node features of shape [num_nodes, output_dim].
        """
        x = torch.cat([node_features, updated_messages], dim=-1)
        return self.mlps[node_type](x)

#  Decoders 
class DecoderOPF(nn.Module):
    def __init__(self, hidden_size: int):
        super(DecoderOPF, self).__init__()

        # Linear projection for all node features
        self.node_decodings = nn.ModuleDict({
            node_type: nn.Sequential(nn.Linear(hidden_size, 256),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 2)
                                     )
            for node_type in ["bus", "generator"]
        })

    def forward(self, node_dict, data):
        # pmin, pmax = data["generator"].x[:, 2:4].T
        # qmin, qmax = data["generator"].x[:, 5:7].T
        # vmin, vmax = data["bus"].x[:, 2:].T
        pmin, pmax = data["generator"]["p_lims"].T
        qmin, qmax = data["generator"]["q_lims"].T
        vmin, vmax = data["bus"]["v_lims"].T

        output_nodes = {
            node_type: self.node_decodings[node_type](node_dict[node_type])
            for node_type in ["bus", "generator"]
        }

        # Passing vm, pg, qg through the sigmoid layer.
        output_va = output_nodes["bus"][:, 0]
        output_vm = torch.sigmoid(output_nodes["bus"][:, -1]) * (vmax - vmin) + vmin
        output_pg = torch.sigmoid(output_nodes["generator"][:, 0]) * (pmax - pmin) + pmin
        output_qg = torch.sigmoid(output_nodes["generator"][:, -1]) * (qmax - qmin) + qmin

        output_dict = {
            "bus": torch.stack([output_va, output_vm], dim=1),
            "generator": torch.stack([output_pg, output_qg], dim=1)
        }

        return output_dict
    
class DecoderPF(nn.Module):
    def __init__(self, hidden_size: int):
        super(DecoderPF, self).__init__()

        # Linear projection for all node features
        self.node_decodings = nn.ModuleDict({
            node_type: nn.Sequential(nn.Linear(hidden_size, 256),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 2)
                                     )
            for node_type in ["PV", "PQ", "slack"]
        })

    def forward(self, node_dict, data):
        
        device = data["bus"].x.device
        output_dict = {
            node_type: self.node_decodings[node_type](node_dict[node_type])
            for node_type in ["PV", "PQ", "slack"]
        }

        # Reconstructing the bus-level data
        num_buses = data["bus"].num_nodes
        bus_va = torch.zeros(num_buses, device=device)
        bus_vm = torch.zeros(num_buses, device=device)

        # PQ
        pq_idx = data["PQ", "PQ_link", "bus"].edge_index[1]
        pq_outputs = output_dict["PQ"]
        bus_va[pq_idx] = pq_outputs[:, 0]
        bus_vm[pq_idx] = pq_outputs[:, 1]

        # PV
        pv_idx = data["PV", "PV_link", "bus"].edge_index[1]
        pv_outputs = output_dict["PV"]
        bus_va[pv_idx] = pv_outputs[:, 1]
        bus_vm[pv_idx] = data["PV"].x[:, 1]

        # Slack
        slack_idx = data["slack", "slack_link", "bus"].edge_index[1]
        slack_va_vm = data["slack"].x
        bus_va[slack_idx] = slack_va_vm[:, 0]
        bus_vm[slack_idx] = slack_va_vm[:, 1]

        output_dict["bus"] = torch.stack([bus_va, bus_vm], dim=-1)

        return output_dict

class CANOS_PF(nn.Module):
    def __init__(self, dataset, hidden_dim, include_sent_messages, k_steps):
        super().__init__()
        edge_feat_dim = node_feat_dim = hidden_dim

        # Define the encoder to get projected nodes and edges
        self.encoder = Encoder(data=dataset, hidden_size=hidden_dim)

        # Interaction network layers for message passing
        node_type_dict = {
            node_type: True
            for node_type in dataset[0].num_node_features.keys()
        }

        edge_type_dict = {
            str(edge_type): True if "edge_attr" in dataset[0][edge_type] else False
            for edge_type in dataset[0].edge_types
            if "bus" in edge_type[0]  # Only include edges where "bus" is the source
        }

        self.message_passing_layers = nn.ModuleList(
            InteractionNetwork(edge_type_dict=edge_type_dict,
                               node_type_dict=node_type_dict,
                               edge_dim=edge_feat_dim,
                               node_dim=node_feat_dim,
                               hidden_dim=hidden_dim,
                               include_sent_messages=include_sent_messages) for _ in range(k_steps))

        # Define the decoder to get the model outputs
        self.decoder = DecoderPF(hidden_size=hidden_dim)
        self.k_steps = k_steps

    def forward(self, data):

        # Encoding
        projected_nodes, projected_edges = self.encoder(data)

        # Message passing layers with residual connections
        nodes, edges = projected_nodes, projected_edges
        for l in range(self.k_steps):
            nodes, edges = self.message_passing_layers[l](nodes, edges, data)

        # Decoding
        output_dict = self.decoder(nodes, data)

        # Deriving branch flows
        p_fr, q_fr, p_to, q_to = self.derive_branch_flows(output_dict, data)
        output_dict["edge_preds"] = torch.stack([p_to, q_to, p_fr, q_fr], dim=-1) 

        return output_dict

    def derive_branch_flows(self, output_dict, data):

        # Create complex voltage
        va = output_dict["bus"][:, 0]
        vm = output_dict["bus"][:, -1]
        v_complex = vm * torch.exp(1j* va)

        # Extract edge info
        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = data["bus", "branch", "bus"].edge_attr

        # Unpack edge features
        br_r, br_x = edge_attr[:, 0], edge_attr[:, 1]
        b_fr, b_to = edge_attr[:, 3], edge_attr[:, 5]
        g_fr, g_to = edge_attr[:, 2], edge_attr[:, 4]
        tap = edge_attr[:, 6]
        shift = edge_attr[:, 7]

        # Complex tap ratio
        T_complex = tap * torch.exp(1j * shift)

        # Complex admittances
        Y_branch = 1 / (br_r + 1j * br_x)
        Y_c_fr = 1j * b_fr
        Y_c_to = 1j * b_to

        # Node voltages
        i, j = edge_index[0], edge_index[1]
        vi = v_complex[i]
        vj = v_complex[j]
        
        # Compute complex branch flows
        S_fr = (Y_branch + Y_c_fr).conj() * (torch.abs(vi) ** 2) / (torch.abs(T_complex) ** 2) - \
            Y_branch.conj() * (vi * vj.conj()) / T_complex

        S_to = (Y_branch + Y_c_to).conj() * (torch.abs(vj) ** 2) - \
            Y_branch.conj() * (vj * vi.conj()) / T_complex.conj()

        # Real/reactive power flows
        p_fr, q_fr = S_fr.real, S_fr.imag
        p_to, q_to = S_to.real, S_to.imag

        return p_fr, q_fr, p_to, q_to

class constraint_violations_loss_pf:
    def __init__(self, ):
        self.constraint_loss = None
        self.bus_real_mismatch = None
        self.bus_reactive_mismatch = None
        
    def __call__(self, output_dict, data):

        device = data["bus"].x.device

        # Get the predictions and edge features
        bus_pred = output_dict["bus"]
        edge_pred = output_dict["edge_preds"]
        edge_indices = data["bus", "branch", "bus"].edge_index
        edge_features = data["bus", "branch", "bus"].edge_attr
        va, vm = bus_pred.T
        complex_voltage = vm * torch.exp(1j* va)

        # Get the branch flows from the edge predictions
        n = data["bus"].x.shape[0]
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        flows_rev = edge_pred[:, 0] + 1j * edge_pred[:, 1]  
        flows_fwd = edge_pred[:, 2] + 1j * edge_pred[:, 3]  
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)

        # Generator flows (already aggregated per bus)
        bus_gen = data["bus"].bus_gen.to(device) 
        gen_flows = bus_gen[:, 0] + 1j * bus_gen[:, 1]

        # Demand flows (already aggregated per bus)
        bus_demand = data["bus"].bus_demand.to(device) 
        demand_flows = bus_demand[:, 0] + 1j * bus_demand[:, 1]

        # Shunt admittances
        bus_shunts = data["bus"].shunt.to(device)  
        shunt_flows = (torch.abs(vm) ** 2) * (bus_shunts[:, 1] + 1j * bus_shunts[:, 0])  # (b_shunt + j*g_shunt)

        power_balance = gen_flows - demand_flows - shunt_flows - sum_branch_flows
        real_power_mismatch = torch.abs(torch.real(power_balance))
        reactive_power_mismatch = torch.abs(torch.imag(power_balance))

        # power: real and imaginary mismatches
        violation_degree_real_mismatch = real_power_mismatch.mean()
        violation_degree_imag_mismatch = reactive_power_mismatch.mean()

        # branch flows: ground truth mismatch, real
        p_flows_true = data["bus", "branch", "bus"].edge_label[:,-2] # this is from bus flow
        p_flows_mismatch = torch.real(flows_fwd) - p_flows_true
        violation_degree_real_flow_mismatch = torch.abs(p_flows_mismatch).mean()

        # branch flows: ground truth mismatch, reactive
        q_flows_true = data["bus", "branch", "bus"].edge_label[:,-1] # this is from bus flow
        q_flows_mismatch = torch.imag(flows_fwd) - q_flows_true
        violation_degree_imag_flow_mismatch = torch.abs(q_flows_mismatch).mean()

        # loss
        loss_c = (violation_degree_real_mismatch + violation_degree_imag_mismatch + 
                  violation_degree_real_flow_mismatch + violation_degree_imag_flow_mismatch)
        

        self.constraint_loss = loss_c
        self.bus_real_mismatch = violation_degree_real_mismatch
        self.bus_reactive_mismatch = violation_degree_imag_mismatch
        self.real_flow_mismatch_violation = violation_degree_real_flow_mismatch
        self.imag_flow_mismatch_violation = violation_degree_imag_flow_mismatch

        return loss_c

def CANOSMSE(output_dict, data):
    # Gather predictions
    PV_pred, PQ_pred, slack_pred = output_dict["PV"], output_dict["PQ"], output_dict["slack"]
    edge_preds = output_dict["edge_preds"]

    # Gather targets
    PV_target, PQ_target, slack_target = data["PV"].y, data["PQ"].y, data["slack"].y
    branch_target = data["bus", "branch", "bus"].edge_label

    # Calculate L2 loss
    pv_loss = torch.nn.functional.mse_loss(PV_pred, PV_target)
    pq_loss = torch.nn.functional.mse_loss(PQ_pred, PQ_target)
    slack_loss = torch.nn.functional.mse_loss(slack_pred, slack_target)
    edge_loss = torch.nn.functional.mse_loss(edge_preds, branch_target)

    total_loss = pv_loss + pq_loss + slack_loss + edge_loss

    return total_loss


if __name__ == "__main__": 
    case_14_data = PFDeltaCANOS(root_dir='data/gns_data/', case_name='case14', add_bus_type=True, split='train')
    model = CANOS_PF(dataset=case_14_data, hidden_dim=128, include_sent_messages=True, k_steps=5)
    device='cpu'
    epochs = 10
    batch_size = 32
    lr = 1e-4
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Assume data_list is a list of HeteroData objects
    loader = DataLoader(case_14_data, batch_size=batch_size, shuffle=True)

    def register_hooks(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, n=name: print(f"{n}: grad norm = {grad.norm():.4e}"))
            
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        loader_iter = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loader_iter:
            # Send batch to device
            batch = batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            output_dict = model(batch)

            # Compute losses
            loss_mse = CANOSMSE(output_dict, batch)  # MSE-based loss
            loss_constraints = constraint_violations_loss_pf()(output_dict, batch)  # Constraint violation loss
            loss = loss_mse + 0.1*loss_constraints
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Optionally update tqdm with current batch loss
            loader_iter.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")