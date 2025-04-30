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
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TAGConv
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import degree


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


class PFDeltaPFNet(PFDeltaDataset): 
    def __init__(self, root_dir='data', case_name='', split='train', add_bus_type=False, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root_dir, case_name, split, add_bus_type, transform, pre_transform, pre_filter, force_reload)

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
            one_hot = torch.zeros(4) # wanna make sure this works before pushing
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

class EdgeAggregation(MessagePassing):
    """
    Custom MessagePassing module for aggregating edge features
    to compute node-level representations.
    Params:
        nfeature_dim (int): Dimensionality of node features.
        efeature_dim (int): Dimensionality of edge features.
        hidden_dim (int): Hidden dimension of the MLP.
        output_dim (int): Dimensionality of the output node features.
    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim*2 + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def message(self, x_i, x_j, edge_attr):
        '''
        Compute messages passed from source to target nodes in the graph.
        Params:
            x_i (torch.Tensor): Target node features (num_edges, nfeature_dim).
            x_j (torch.Tensor): Source node features (num_edges, nfeature_dim).
            edge_attr (torch.Tensor): Edge features (num_edges, efeature_dim).
        Returns:
            (torch.Tensor): Aggregated features for each edge (num_edges, output_dim).
        '''
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1)) # PNAConv style

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for aggregating edge features and computing node embeddings.
        N is the batch size.
        Params:
            x (torch.Tensor): Node features (N, num_nodes, nfeature_dim).
            edge_index (torch.Tensor): Graph connectivity in COO format (N, 2, num_edges).
            edge_attr (torch.Tensor): Edge features (N, num_edges, efeature_dim).
        Returns:
            torch.Tensor: Node embeddings after aggregating edge features
                        (N, num_nodes, output_dim).
        """

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # no self loop because NO EDGE ATTR FOR SELF LOOP

        # Step 2: Calculate the degree of each node.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 

        # Step 3: Feature transformation.
        # x = self.linear(x) # no feature transformation

        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)

        return out

class PowerFlowNet(nn.Module):
    """
    PowerFlowNet: A Graph Neural Network for power flow approximation in graphs.
    Model combines message passing and convolutions to predict node-level
    outputs (e.g., voltages, angles) in power systems:
    - Mask embedding for selective feature predictions.
    - Multi-step message passing layers combined with convolution layers.
    Params:
        nfeature_dim (int): Dimensionality of node features.
        efeature_dim (int): Dimensionality of edge features.
        output_dim (int): Dimensionality of the output node embeddings.
        hidden_dim (int): Hidden layer dimensionality.
        n_gnn_layers (int): Number of GNN layers in the network.
        K (int): Number of hops for the TAGConv layer.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()

        self.mask_embd = nn.Sequential(
                nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for _ in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        # NO SLACK BUS OPERATIONS INCLUDED
        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        # self.slack_aggr = SlackAggregation(hidden_dim, hidden_dim, 'to_slack')
        # self.slack_propagate = SlackAggregation(hidden_dim, hidden_dim, 'from_slack')
        self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))

        self.dropout = nn.Dropout(self.dropout_rate, inplace=False)

    def is_directed(self, edge_index):
        """
        Determines if a graph is directed by examining the first edge.
        Params:
            edge_index (torch.Tensor): Edge indices of shape (2, num_edges).
        Returns:
            (bool): True if the graph is directed, False otherwise.
        """
        if edge_index.shape[1] == 0:
            # no edge at all, only single nodes. automatically undirected
            return False
        # if there is the reverse of the first edge does not exist, then directed.
        return edge_index[0, 0] not in edge_index[1, edge_index[0, :] == edge_index[1, 0]]

    def undirect_graph(self, edge_index, edge_attr):
        """
        Converts a directed graph into an undirected one by duplicating edges.
        Params:
            edge_index (torch.Tensor): Edge indices (2, num_edges).
            edge_attr (torch.Tensor): Edge features (num_edges, efeature_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated edge indices and edge attributes.
        """
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1, :], edge_index[0, :]],
                dim=0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim=1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim=0
            )   # (2*E, fe)

            return edge_index, edge_attr
        else:
            return edge_index, edge_attr

    def data_gatherer(self, data):
        """
        This method gathers the node features, prediction mask, edge index,
        and edge features for both heterogeneous and homogeneous graph formats.
        """
        if isinstance(data, HeteroData):
            x = data["bus"].x  # (N, 16)
            mask = x[:, -6:]    # (N, 6): prediction mask (last 6 dims)
            x = x[:, :-6]       # (N, 10): remaining features (4 one-hot + 6 real features)

            edge_index = data["bus", "branch", "bus"].edge_index
            edge_features = data["bus", "branch", "bus"].edge_attr

        elif isinstance(data, Data):
            # PowerFlowNet original format
            assert data.x.shape[-1] == self.nfeature_dim * 2 + 4
            x = data.x[:, 4:4 + self.nfeature_dim]
            mask = data.x[:, -self.nfeature_dim:]
            edge_index = data.edge_index
            edge_features = data.edge_attr

        return x, mask, edge_index, edge_features


    def forward(self, data):
        """
        Forward pass of the PowerFlowNet.
        Params:
            data (Data): Input graph data containing:
                - x (torch.Tensor): Node features (num_nodes, nfeature_dim).
                - edge_index (torch.Tensor): Edge indices (2, num_edges).
                - edge_attr (torch.Tensor): Edge features (num_edges, efeature_dim).
                - pred_mask (torch.Tensor): Mask for features to predict.
                - bus_type (torch.Tensor): Node types.
                - batch (torch.Tensor): Batch information.
        Returns:
            (torch.Tensor): Output node embeddings (num_nodes, output_dim).
        """
        x, mask, edge_index, edge_features = self.data_gatherer(data)

        # assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data["bus"].x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying.
        x = self.mask_embd(mask) + x

        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = nn.ReLU()(x)

        # Is this if statement necessary? It must be an EdgeAggregation layer
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)

        # # Mask out known values
        # x = place_known_values(x, data)
        return x

if __name__ == "__main__":
    case_14_data = PFDeltaPFNet(root_dir='data/gns_data/case14_n/', case_name='case14', split='train')
    model = PowerFlowNet(
    nfeature_dim=6,
    efeature_dim=5,
    output_dim=6,
    hidden_dim=64,
    n_gnn_layers=3,
    K=3,
    dropout_rate=0.1
    )
    device='cpu'
    epochs = 10
    batch_size = 32

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    model.train()
    # Assume data_list is a list of HeteroData objects
    loader = DataLoader(case_14_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0

        for data in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)  # shape: [num_nodes, 6]

            # Target values
            y = data['bus'].y  # shape: [num_nodes, 6]
            mask = data['bus'].x[:, -6:]  # prediction mask: [num_nodes, 6]

            # Compute L2 loss only on masked targets
            loss = F.mse_loss(out * mask, y * mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}: Avg MSE Loss = {avg_loss:.6f}")