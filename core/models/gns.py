import torch
import torch.nn as nn

from core.utils.pf_losses_utils import GNSPowerBalanceLoss
from core.utils.registry import registry


@registry.register_model("graph_neural_solver")
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
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            NNUpdate(self.L_input_dim, hidden_dim) for _ in range(K)
        )
        self.power_balance = GNSPowerBalanceLoss()

    def forward(self, data, return_data=False):
        """ """
        device = data["bus"].x.device

        # Instantiate message vectors for each bus
        num_nodes = data["bus"].x.size(0)
        data["bus"].m = torch.zeros((num_nodes, self.hidden_dim), device=device)
        total_layer_loss = 0

        for k in range(self.K):
            # Update P and Q values for all buses
            self.global_active_compensation(data)

            # Compute local power imbalance variables and store power imbalance loss
            layer_loss = self.local_power_imbalance(data, layer_loss=True)
            total_layer_loss += layer_loss * (self.gamma ** (self.K - k))

            # Apply the neural network update block
            self.apply_nn_update(data, k)

        last_loss = self.local_power_imbalance(data, layer_loss=True)
        total_layer_loss += last_loss * (self.gamma**0)
        out_dict = {
            "last_loss": last_loss,
            "total_layer_loss": total_layer_loss,
            "data": data,
        }
        return out_dict

    def global_active_compensation(self, data):
        """ """
        # Compute global power demand
        p_joule = self.compute_p_joule(data)
        p_global = self.compute_p_global(data, p_joule)

        # Compute pg_slack and assign to relevant buses
        pg_slack = self.compute_pg_slack(p_global, data)
        slack_idx = (data["bus"].bus_type == 3).nonzero(as_tuple=True)[0]
        data["bus"].pg[slack_idx] = pg_slack

        # Compute qg values for each bus
        qg = self.power_balance(data, layer_loss=False)
        data["bus"].qg = qg

    def compute_p_joule(self, data):
        """ """
        # Extract edge index and attributes
        edge_index = data[("bus", "branch", "bus")].edge_index
        edge_attr = data[("bus", "branch", "bus")].edge_attr
        src, dst = edge_index

        # Edge features
        tau_ij = edge_attr[:, -2]
        shift_ij = edge_attr[:, -1]

        # Line admittance features
        br_r = edge_attr[:, 0]
        br_x = edge_attr[:, 1]
        y = 1 / (torch.complex(br_r, br_x))
        y_ij = torch.abs(y)
        delta_ij = torch.angle(y)

        # Node features
        v_i = data["bus"].v[src]
        v_j = data["bus"].v[dst]
        theta_i = data["bus"].theta[src]
        theta_j = data["bus"].theta[dst]

        # Compute p_global
        term1 = (
            -v_i
            * v_j
            * y_ij
            / tau_ij
            * (
                torch.cos(theta_i - theta_j - delta_ij - shift_ij)
                + torch.cos(theta_j - theta_i - delta_ij + shift_ij)
            )
        )

        term2 = (v_i / tau_ij) ** 2 * y_ij * torch.cos(delta_ij)
        term3 = v_j**2 * y_ij * torch.cos(delta_ij)
        p_joule_edge = torch.abs(term1 + term2 + term3)

        # Map to individual graphs
        bus_batch = data["bus"].batch  # batch index per bus
        edge_batch = bus_batch[src]  # batch index per edge (via source bus)
        num_graphs = int(edge_batch.max()) + 1
        p_joule_per_graph = torch.zeros(num_graphs, device=p_joule_edge.device)
        p_joule_per_graph = p_joule_per_graph.index_add(0, edge_batch, p_joule_edge)

        return p_joule_per_graph

    def compute_p_global(self, data, p_joule):
        """ """
        # Per-bus data
        pd = data["bus"].pd
        v = data["bus"].v
        g_shunt = data["bus"].shunt[:, 0]

        # Graph assignment per bus
        bus_batch = data["bus"].batch
        num_graphs = int(bus_batch.max()) + 1

        # Compute local p_global components
        p_global_local = pd + (v**2) * g_shunt

        # Sum per graph
        p_global = torch.zeros(num_graphs, device=p_global_local.device)
        p_global = p_global.index_add(0, bus_batch, p_global_local)

        # Add per-graph Joule losses
        p_global += p_joule

        return p_global

    def compute_pg_slack(self, p_global, data):
        """ """
        pg_setpoints = data["gen"].generation[:, 0]
        pg_max_vals = data["gen"].limits
        pg_min_vals = data["gen"].limits
        is_slack = data["gen"].slack_gen
        pg_max_slack = pg_max_vals[is_slack][:, 1]
        pg_min_slack = pg_min_vals[is_slack][:, 0]

        gen_idx = torch.arange(pg_setpoints.size(0), device=pg_setpoints.device)
        graph_ids = torch.bucketize(
            gen_idx, data["gen"].ptr[1:]
        )  # gives [num_gen_nodes] → [num_graphs]

        # Initialize output tensors
        num_graphs = data["gen"].ptr.size(0) - 1

        pg_setpoint_slack = pg_setpoints[is_slack]
        pg_setpoints_non_slack = pg_setpoints.clone()
        pg_setpoints_non_slack[is_slack] = 0.0

        # Sum of total generator setpoints per graph
        pg_setpoints_sum = torch.zeros(num_graphs, device=pg_setpoints.device)
        pg_setpoints_sum = pg_setpoints_sum.index_add(0, graph_ids, pg_setpoints)
        pg_non_slack_setpoints_sum = torch.zeros(num_graphs, device=pg_setpoints.device)
        pg_non_slack_setpoints_sum = pg_non_slack_setpoints_sum.index_add(
            0, graph_ids, pg_setpoints_non_slack
        )

        # Compute lambda in a vectorized way
        under = p_global < pg_setpoints_sum
        over = ~under
        lamb = torch.zeros(num_graphs, device=pg_setpoints.device)
        lamb[under] = (
            p_global[under] - pg_non_slack_setpoints_sum[under] - pg_max_slack[under]
        ) / (2 * (pg_setpoint_slack[under] - pg_min_slack[under]))
        lamb[over] = (
            p_global[over]
            - pg_non_slack_setpoints_sum[over]
            - 2 * pg_setpoint_slack[over]
            - pg_max_slack[over]
        ) / (2 * (pg_max_slack[over] - pg_setpoint_slack[over]))

        lamb = torch.clamp(lamb, min=0.0)
        pg_slack = torch.zeros_like(lamb)

        # Compute the pg_slack values
        case1 = lamb < 0.5
        case2 = ~case1
        pg_slack[case1] = (
            pg_min_slack[case1]
            + 2 * (pg_setpoint_slack[case1] - pg_min_slack[case1]) * lamb[case1]
        )

        pg_slack[case2] = (
            2 * pg_setpoint_slack[case2]
            - pg_max_slack[case2]
            + 2 * (pg_max_slack[case2] - pg_setpoint_slack[case2]) * lamb[case2]
        )

        return pg_slack

    def local_power_imbalance(self, data, layer_loss):
        """ """
        delta_p, delta_q, delta_s = self.power_balance(data, layer_loss)
        data["bus"].delta_p = delta_p
        data["bus"].delta_q = delta_q
        return delta_s

    def message_passing_update(self, data):
        """ """
        edge_index = data[("bus", "branch", "bus")].edge_index
        edge_attr = data[("bus", "branch", "bus")].edge_attr
        src, dst = edge_index

        # Extract edge features
        br_r = edge_attr[:, 0]
        br_x = edge_attr[:, 1]
        b_ij = edge_attr[:, 3]
        shift_ij = edge_attr[:, -1]
        tau_ij = edge_attr[:, -2]
        line_ij = torch.stack([br_r, br_x, b_ij, tau_ij, shift_ij], dim=1)

        # Get source node message vectors
        m_src = data["bus"].m[src]

        # Compute messages along edges
        edge_input = torch.cat([m_src, line_ij], dim=1)
        messages = self.phi(edge_input)

        # Aggregate messages to each destination node
        num_nodes = data["bus"].x.size(0)
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
            nn.Linear(hidden_dim, 1),
        )
        self.L_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.L_m = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data, messages):
        """ """
        v = data["bus"].v.unsqueeze(1)
        theta = data["bus"].theta.unsqueeze(1)
        delta_p = data["bus"].delta_p.unsqueeze(1)
        delta_q = data["bus"].delta_q.unsqueeze(1)
        m = data["bus"].m
        feature_vector = torch.cat([v, theta, delta_p, delta_q, m, messages], dim=1)

        theta_update = self.L_theta(feature_vector)
        v_update = self.L_v(feature_vector)
        m_update = self.L_m(feature_vector)

        # theta update
        data["bus"].theta = data["bus"].theta + theta_update.squeeze(-1)

        # v only gets updated at that non-PV buses
        gen_idx = (data["bus"].bus_type == 2).nonzero(as_tuple=True)
        v_update[gen_idx] = 0
        data["bus"].v = data["bus"].v + v_update.squeeze(-1)

        # m update
        data["bus"].m = data["bus"].m + m_update
