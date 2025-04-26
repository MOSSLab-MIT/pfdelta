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

class PFDeltaDataset(InMemoryDataset):
    def __init__(self, root_dir='data', case_name='', split='train', transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        self.split = split
        self.force_reload = force_reload
        root = os.path.join(root_dir, case_name)
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
        gen_limits, gen_setpoints, more_gen_data  = [], [], []

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
            for gen_id, gen in sorted(network_data['gen'].items(), key=lambda x: int(x[0])):
                if int(gen['gen_bus']) == bus_id: 
                    if gen['gen_status'] == 1:
                        gen_sol = solution_data['gen'][gen_id]
                        pg += gen_sol['pg']
                        qg += gen_sol['qg']
                    else:
                        assert solution_data['gen'].get(gen_id) is None, f"Expected gen {gen_id} to be off."

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
        
        # Generator
        for gen_id, gen in sorted(network_data['gen'].items(), key=lambda x: int(x[0])):
            if gen['gen_status'] == 1:
                gen_sol = solution_data['gen'][gen_id]
                pmin, pmax, qmin, qmax = gen['pmin'], gen['pmax'], gen['qmin'], gen['qmax']
                pgen, qgen = gen_sol['pg'], gen_sol['qg']
                gen_limits.append(torch.tensor([pmin, pmax, qmin, qmax]))
                gen_setpoints.append(torch.tensor([pgen, qgen]))
                is_slack = torch.tensor(
                        1 if network_data['bus'][str(gen['gen_bus'])]['bus_type'] == 3 else 0,
                        dtype=torch.bool
                                )
                gen_bus = torch.tensor(gen['gen_bus']) - 1  # zero-indexed
                more_gen_data.append(torch.stack([gen_bus, is_slack]))
            else:
                assert solution_data['gen'].get(gen_id) is None, f"Expected gen {gen_id} to be off."

        # Edges
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

        data['gen'].limits = torch.stack(gen_limits)
        data['gen'].setpoints = torch.stack(gen_setpoints)
        data['gen'].more_gen_data = torch.stack(more_gen_data)

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



class PFDeltaGNS(PFDeltaDataset): 
    def __init__(self, root_dir='data', case_name='', split='train', transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root_dir, case_name, split, transform, pre_transform, pre_filter, force_reload)

    def build_heterodata(self, pm_case):
        """ """
        # call base version
        data = super().build_heterodata(pm_case)
        num_buses = data['bus'].x.size(0)

        # Init bus-level fields
        v_buses      = torch.zeros(num_buses)
        theta_buses  = torch.zeros(num_buses)
        pd_buses     = torch.zeros(num_buses)
        qd_buses     = torch.zeros(num_buses)
        pg_buses     = torch.zeros(num_buses)
        qg_buses     = torch.zeros(num_buses)

        for ntype in ['PQ', 'PV', 'slack']:
            node = data[ntype]
            num_nodes = node.x.size(0)

            if ntype == 'PQ':
                # Flat start init
                node.x_gns = torch.tensor([1.0, 0.0]).repeat(num_nodes, 1)
                pd = node.x[:, 0]
                qd = node.x[:, 1]
                pg = torch.zeros_like(pd)
                qg = torch.zeros_like(qd)

            elif ntype == 'PV':
                v = node.x[:, 1:2]
                theta = torch.zeros_like(v)
                node.x_gns = torch.cat([v, theta], dim=1)
                pd = node.demand[:, 0]
                qd = node.demand[:, 1]
                pg = node.generation[:, 0]
                qg = node.generation[:, 1]

            elif ntype == 'slack':
                node.x_gns = node.x.clone()
                pd = node.demand[:, 0]
                qd = node.demand[:, 1]
                pg = node.generation[:, 0]
                qg = node.generation[:, 1]

            # Map to buses
            link_type = ntype + '_link'
            edge_index = data[(ntype, link_type, 'bus')].edge_index
            src, dst = edge_index  # src is local node index, dst is bus index

            x_gns = node.x_gns
            v_dst = x_gns[:, 0]
            theta_dst = x_gns[:, 1]

            v_buses.index_add_(0, dst, v_dst)
            theta_buses.index_add_(0, dst, theta_dst)
            pd_buses.index_add_(0, dst, pd)
            qd_buses.index_add_(0, dst, qd)
            pg_buses.index_add_(0, dst, pg)
            qg_buses.index_add_(0, dst, qg)

        # Store in bus
        data['bus'].v = v_buses
        data['bus'].theta = theta_buses
        data['bus'].pd = pd_buses
        data['bus'].qd = qd_buses
        data['bus'].pg = pg_buses
        data['bus'].qg = qg_buses
        data['bus'].delta_p = torch.zeros_like(v_buses)
        data['bus'].delta_q = torch.zeros_like(v_buses)

        return data



class GraphNeuralSolver(nn.Module):
    def __init__(self, K, hidden_dim, gamma):
        super().__init__()
        self.K = K
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.phi_input_dim = hidden_dim + 5
        self.L_input_dim = 2 * hidden_dim + 4
        self.phi = nn.Sequential(
            nn.Linear(self.phi_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList(
            NNUpdate(self.L_input_dim, hidden_dim) for _ in range(K)
        )
        self.power_balance = LocalPowerBalanceLoss()

    def forward(self, data):  
        """ """
        device = data['bus'].x.device 

        # Instantiate message vectors for each bus
        num_nodes = data['bus'].x.size(0)
        data['bus'].m = torch.zeros((num_nodes, self.hidden_dim), device=device)
        total_layer_loss = 0

        for k in range(self.K): 

            # Update P and Q values for all buses
            self.global_active_compensation(data) 

            # Compute local power imbalance variables and store power imbalance loss 
            layer_loss = self.local_power_imbalance(data, layer_loss=True)
            total_layer_loss += layer_loss * (self.gamma ** (self.K - k))

            # Apply the neural network update block 
            self.apply_nn_update(data, k)
        
        return data, total_layer_loss


    def global_active_compensation(self, data):
        """ """
        # Compute global power demand 
        p_joule = self.compute_p_joule(data)   
        p_global = self.compute_p_global(data, p_joule)

        # Compute pg_slack and assign to relevant buses
        pg_slack = self.compute_pg_slack(p_global, data)
        edge_index = data[('slack', 'slack_link', 'bus')].edge_index
        _, dst = edge_index 
        data['bus'].pg[dst] = pg_slack

        # Compute qg values for each bus
        qg = self.power_balance(data, layer_loss=False)
        data['bus'].qg = qg


    def compute_p_joule(self, data): 
        """ """
        # Extract edge index and attributes
        edge_index = data[('bus', 'branch', 'bus')].edge_index
        edge_attr = data[('bus', 'branch', 'bus')].edge_attr 
        src, dst = edge_index 

        # Edge features
        tau_ij = edge_attr[:, -2]
        shift_ij = edge_attr[:, -1]

        # Line admittance features 
        br_r = edge_attr[:, 0]
        br_x = edge_attr[:, 1]
        y = 1 / (torch.complex(br_r, br_x))
        if torch.any((torch.complex(br_r, br_x)) == 0):
            print("Division by zero detected!")
        y_ij = torch.abs(y)
        delta_ij = torch.angle(y)

        # Node features
        v_i = data['bus'].v[src]
        v_j = data['bus'].v[dst]
        theta_i = data['bus'].theta[src]
        theta_j = data['bus'].theta[dst]

        # Compute p_global
        term1 = v_i * v_j * y_ij / tau_ij * (
            torch.sin(theta_i - theta_j - delta_ij - shift_ij) +  
            torch.sin(theta_j - theta_i - delta_ij + shift_ij))

        term2 = (v_i / tau_ij) ** 2 * y_ij * torch.sin(delta_ij)
        term3 = v_j ** 2 * y_ij * torch.sin(delta_ij)
        p_joule_edge = torch.abs(term1 + term2 + term3)
        
        # Map to individual graphs
        bus_batch = data['bus'].batch  # batch index per bus
        edge_batch = bus_batch[src]    # batch index per edge (via source bus)
        num_graphs = int(edge_batch.max()) + 1
        p_joule_per_graph = torch.zeros(num_graphs, device=p_joule_edge.device)
        p_joule_per_graph = p_joule_per_graph.index_add(0, edge_batch, p_joule_edge)

        return p_joule_per_graph
    

    def compute_p_global(self, data, p_joule): 
        """ """
        # Per-bus data
        pd = data['bus'].pd 
        v = data['bus'].v
        g_shunt = data['bus'].x[:, 0]

        # Graph assignment per bus
        bus_batch = data['bus'].batch
        num_graphs = int(bus_batch.max()) + 1

        # Compute local p_global components
        p_global_local = pd + (v ** 2) * g_shunt

        # Sum per graph
        p_global = torch.zeros(num_graphs, device=p_global_local.device)
        p_global = p_global.index_add(0, bus_batch, p_global_local)

        # Add per-graph Joule losses
        p_global += p_joule

        return p_global


    def compute_pg_slack(self, p_global, data):
        """ """
        pg_setpoints = data['gen'].setpoints[:, 0] 
        pg_max_vals = data['gen'].limits[:, 1]
        pg_min_vals = data['gen'].limits[:, 0]
        is_slack = data['gen'].more_gen_data[:, 1] == 1 
        graph_ids = is_slack.cumsum(dim=0) - 1
        num_graphs = graph_ids.max().item() + 1
        pg_setpoint_slack = pg_setpoints[is_slack]
        pg_setpoints_non_slack = pg_setpoints.clone()
        pg_setpoints_non_slack[is_slack] = 0.0
        pg_max_slack = torch.full_like(pg_max_vals[is_slack], 3.4)
        pg_min_slack = torch.full_like(pg_min_vals[is_slack], 0.0)
        # TODO: ADD THIS BACK LATER pg_max_vals[is_slack] 
        # TODO: ADD THIS BACK LATER pg_min_vals[is_slack]  

        # Sum of total generator setpoints per graph
        pg_setpoints_sum = torch.zeros(num_graphs, device=pg_setpoints.device)
        pg_setpoints_sum = pg_setpoints_sum.index_add(0, graph_ids, pg_setpoints)
        pg_non_slack_setpoints_sum = torch.zeros(num_graphs, device=pg_setpoints.device)
        pg_non_slack_setpoints_sum = pg_non_slack_setpoints_sum.index_add(0, graph_ids, pg_setpoints_non_slack)

        # Compute lambda in a vectorized way
        under = (p_global < pg_setpoints_sum)
        over = ~under
        lamb = torch.zeros(num_graphs, device=pg_setpoints.device)
        lamb[under] = (
            (p_global[under] - pg_non_slack_setpoints_sum[under] - pg_max_slack[under]) /
            (2 * (pg_setpoint_slack[under] - pg_min_slack[under]))
        )
        lamb[over] = (
            (p_global[over] - pg_non_slack_setpoints_sum[over] - 2 * pg_setpoint_slack[over] - pg_max_slack[over]) /
            (2 * (pg_max_slack[over] - pg_setpoint_slack[over]))
        )

        lamb = torch.clamp(lamb, min=0.0)
        pg_slack = torch.zeros_like(lamb)

        # Compute the pg_slack values 
        case1 = lamb < 0.5
        case2 = ~case1
        pg_slack[case1] = pg_min_slack[case1] + 2 * (pg_setpoint_slack[case1] - pg_min_slack[case1]) * lamb[case1]

        pg_slack[case2] = (
            2 * pg_setpoint_slack[case2] - pg_max_slack[case2] +
            2 * (pg_max_slack[case2] - pg_setpoint_slack[case2]) * lamb[case2]
        )
        
        return pg_slack


    def local_power_imbalance(self, data, layer_loss):
        """ """
        delta_p, delta_q, delta_s = self.power_balance(data, layer_loss)
        data['bus'].delta_p = delta_p
        data['bus'].delta_q = delta_q
        return delta_s


    def message_passing_update(self, data): 
        """ """
        edge_index = data[('bus', 'branch', 'bus')].edge_index
        edge_attr = data[('bus', 'branch', 'bus')].edge_attr 
        src, dst = edge_index 

        # Extract edge features
        br_r = edge_attr[:, 0]
        br_x = edge_attr[:, 1]
        b_ij = edge_attr[:, 3]
        shift_ij = edge_attr[:, -1]
        tau_ij = edge_attr[:, -2]
        line_ij = torch.stack([br_r, br_x, b_ij, tau_ij, shift_ij], dim=1)

        # Get source node message vectors
        m_src = data['bus'].m[src] 

        # Compute messages along edges
        edge_input = torch.cat([m_src, line_ij], dim=1) 
        messages = self.phi(edge_input)  

        # Aggregate messages to each destination node
        num_nodes = data['bus'].x.size(0)
        agg_msg_i = torch.zeros((num_nodes, self.hidden_dim), device=messages.device)
        agg_msg_i = agg_msg_i.index_add(0, dst, messages) 

        return agg_msg_i


    def apply_nn_update(self, data, k): 
        """ """
        messages = self.message_passing_update(data)
        self.layers[k](data, messages)



class NNUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.L_theta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.L_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.L_m = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(self, data, messages):
        """ """
        v = data['bus'].v.unsqueeze(1) 
        theta = data['bus'].theta.unsqueeze(1) 
        delta_p = data['bus'].delta_p.unsqueeze(1) 
        delta_q = data['bus'].delta_q.unsqueeze(1) 
        m = data['bus'].m
        feature_vector = torch.cat([v, theta, delta_p, delta_q, m, messages], dim=1)

        theta_update = self.L_theta(feature_vector)
        v_update = self.L_v(feature_vector)
        m_update = self.L_m(feature_vector)

        # theta update 
        data['bus'].theta = data['bus'].theta + theta_update.squeeze(-1)
        
        # v only gets updated if it doesn't have a generator
        _, gen_idx = data[('PV', 'PV_link', 'bus')].edge_index
        v_update[gen_idx] = 0
        data['bus'].v = data['bus'].v + v_update.squeeze(-1)

        # m update 
        data['bus'].m = data['bus'].m + m_update



class LocalPowerBalanceLoss:
    """
    Compute the power balance loss.
    """
    def __init__(self):
        self.power_balance_loss = None
    
    def __call__(self, data, layer_loss=False, training=False):
        edge_index = data[('bus', 'branch', 'bus')].edge_index
        edge_attr = data[('bus', 'branch', 'bus')].edge_attr
        src, dst = edge_index

        # Bus values
        v = data['bus'].v
        theta = data['bus'].theta
        b_s = data['bus'].x[:,1]
        g_s = data['bus'].x[:,0]
        pg = data['bus'].pg
        pd = data['bus'].pd
        qd = data['bus'].qd
        qg = data['bus'].qg

        # Edge values
        br_r = edge_attr[:, 0]
        br_x = edge_attr[:, 1]
        b_ij = edge_attr[:, 3]
        shift_ij = edge_attr[:, -1]
        tau_ij = edge_attr[:, -2]
        y = 1 / (torch.complex(br_r, br_x))
        y_ij = torch.abs(y)
        delta_ij = torch.angle(y)

        # Gather per-branch bus features
        v_i = v[src]
        v_j = v[dst]
        theta_i = theta[src]
        theta_j = theta[dst]

        # Active power flows
        P_flow_src = (
            (v_i * v_j * y_ij / tau_ij) * torch.sin(theta_i - theta_j - delta_ij - shift_ij) 
            + ((v_i / tau_ij) ** 2) * (y_ij * torch.sin(delta_ij))
        ) 

        P_flow_dst = (
            (v_j * v_i * y_ij / tau_ij) * torch.sin(theta_j - theta_i - delta_ij + shift_ij)
            + ((v_j) ** 2) * (y_ij * torch.sin(delta_ij))
        )

        # Reactive power flows
        Q_flow_src = (
            (-v_i * v_j * y_ij / tau_ij) * torch.cos(theta_i - theta_j - delta_ij - shift_ij)
            + ((v_i / tau_ij) ** 2) * (y_ij * torch.cos(delta_ij) - b_ij / 2)
        )

        Q_flow_dst = (
            (-v_j * v_i * y_ij / tau_ij) * torch.cos(theta_j - theta_i - delta_ij + shift_ij)
            + ((v_j) ** 2) * (y_ij * torch.sin(delta_ij) - b_ij / 2)
        ) 

        # Aggregate contributions for all nodes
        Pbus_pred = torch.zeros_like(v).scatter_add_(0, src, P_flow_src)
        Pbus_pred = Pbus_pred.scatter_add_(0, dst, P_flow_dst)
        Qbus_pred = torch.zeros_like(v).scatter_add_(0, src, Q_flow_src)
        Qbus_pred = Qbus_pred.scatter_add_(0, dst, Q_flow_dst)

        if layer_loss: 
            delta_p = (pg - pd - g_s * (v ** 2)) + Pbus_pred
            delta_q = qg - qd + b_s * (v ** 2) + Qbus_pred
            delta_s = (delta_p ** 2 + delta_q ** 2).mean()
            if training: 
                self.power_balance_loss = delta_s
            return delta_p, delta_q, delta_s
        else: 
            qg = qd - b_s * v**2 - Qbus_pred
            return qg

if __name__ == "__main__": 
    model = GraphNeuralSolver(K=5, hidden_dim=10, gamma=0.1)
    device='cpu'
    case_14_data = PFDeltaGNS(root_dir='data/gns_data', case_name='case14', split='train')
    epochs = 10
    batch_size = 32
    lr = 1e-3
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = LocalPowerBalanceLoss()

    # Assume data_list is a list of HeteroData objects
    loader = DataLoader(case_14_data, batch_size=batch_size, shuffle=True)
        
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        loader_iter = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loader_iter:
            # Send batch to device
            batch = batch.to(device)

            # Forward pass
            _, layer_loss = model(batch)
            _, _, final_loss = criterion(batch, layer_loss=True, training=True)
            final_loss += layer_loss
            loss = final_loss / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Optionally update tqdm with current batch loss
            loader_iter.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

