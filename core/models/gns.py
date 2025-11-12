# This code is a PyTorch implementation of the Graph Neural Solver
# architecture originally proposed in the following paper: 
# 
#   Balthazar Donon, Rémy Clément, Benjamin Donnot, Antoine Marot, 
#   Isabelle Guyon, et al.. Neural Networks for Power Flow : Graph Neural 
#   Solver. Electric Power Systems Research, 2020, 189, pp.106547. 
#   ff10.1016/j.epsr.2020.106547ff. ffhal-02372741f
# 
# It is compatible with PFDeltaDataset.
# 
# Code was implemented by Anvita Bhagavathula, Alvaro Carbonero, and Ana K. Rivera 
# in April 2025.


import torch
import torch.nn as nn

from core.utils.pf_losses_utils import GNSPowerBalanceLoss
from core.utils.registry import registry


@registry.register_model("graph_neural_solver")
class GraphNeuralSolver(nn.Module):
    """
    Graph Neural Solver (GNS) for AC power flow and optimal power flow tasks.

    This model performs iterative message-passing updates on a power grid graph,
    enforcing power balance through physics-aware constraints while learning
    to correct node voltages and angles.

    At each iteration:
        1. Global active power compensation is applied at the system level.
        2. Local reactive power balance is computed per node.
        3. A neural message-passing block updates node features.

    The architecture couples a learned component (neural updates) with
    physically motivated updates (power balance and slack bus adjustments).
    """
    def __init__(self, K, hidden_dim, gamma, lamb_eps=1e-8):
        """ 
        Initialize the GNS model. Note: to replicate test error performance on
        case 57, use lamb_eps=1e-4.

        Parameters
        ----------
        K: int
            Number of message-passing / update iterations.
        hidden_dim: int 
            Dimension of node and message embeddings.
        gamma: float
            Discount factor for weighting layer-wise imbalance losses.
        lambda_eps: float
            Value used to avoid division by zero when calculating lambda.
        """
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lamb_eps = lamb_eps
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
        """
        Forward pass of the GNS model.

        Performs K iterative updates on the graph, applying global
        power compensation, local imbalance correction, and neural updates.

        Parameters
        ----------
        data: HeteroData 
            Graph data containing node (bus), edge (branch),
            and generator information.
        return_data: bool, optional
            If True, returns modified `data` object.

        Returns
        -------
        out_dict: dict
            dict: {
                "last_loss" (Tensor): Power imbalance loss after final layer.
                "total_layer_loss" (Tensor): Weighted sum of imbalance losses across layers.
                "data" (HeteroData): Updated graph data object (if return_data=True).
            }
        """
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
        """
        Apply global active power balance across the network.

        This step enforces active power conservation by:
            1. Computing per-graph Joule losses.
            2. Aggregating total active demand and shunt losses.
            3. Adjusting the slack bus generation (Pg) to match global demand.
            4. Computing reactive power generation (Qg) via the power balance loss.

        Parameters
        ----------
        data: HeteroData
            Input graph data object (in-place update).
        """
        # Compute global power demand
        p_joule = self.compute_p_joule(data)
        p_global = self.compute_p_global(data, p_joule)

        # Compute pg_slack and assign to relevant buses
        pg_slack = self.compute_pg_slack(p_global, data)
        slack_idx = (data["bus"].bus_type == 3).nonzero(as_tuple=True)[0]
        data["bus"].pg[slack_idx] = pg_slack

        # Compute qg values for each bus
        qg = self.power_balance(data, layer_loss=False)
        bus_types = data["bus"].bus_type
        data["bus"].qg[bus_types != 1] = qg[bus_types != 1]

    def compute_p_joule(self, data):
        """
        Compute per-graph Joule losses due to line resistances.

        Parameters
        ----------
        data: HeteroData
            Graph data with branch parameters and node states.

        Returns:
        -------
        p_joule_per_graph: Tensor
            Per-graph vector of Joule losses [num_graphs] (batched)
        """
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
        """
        Compute per-graph global active power demand including Joule losses.

        Parameters
        ----------
        data: HeteroData
            Input graph data.
        p_joule: torch.Tensor
            Per-graph Joule losses.

        Returns:
        -------
        p_global: torch.Tensor
            Per-graph total active power demand [num_graphs]. (batched)
        """
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
        """
        Compute active power generation (Pg) for slack buses to 
        satisfy power balance.

        The slack bus adjusts its generation within defined limits to match
        total system demand and losses, ensuring that ∑Pg = ∑Pd + losses.

        Parameters
        ----------
        p_global: torch.Tensor
            Per-graph global demand including losses.
        data: HeteroData 
            Input graph data with generator info.

        Returns:
        --------
        pg_slack: torch.Tensor
            Slack bus Pg values [num_graphs].
        """
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
        ) / (2 * (pg_setpoint_slack[under] - pg_min_slack[under]) + self.lamb_eps)
        lamb[over] = (
            p_global[over]
            - pg_non_slack_setpoints_sum[over]
            - 2 * pg_setpoint_slack[over]
            - pg_max_slack[over]
        ) / (2 * (pg_max_slack[over] - pg_setpoint_slack[over]) + self.lamb_eps)

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
        """
        Compute local power imbalance (ΔP, ΔQ) at each bus.

        Parameters
        ----------
        data: HeteroData
            Graph data containing bus-level states.
        layer_loss: bool
            If True, returns the scalar imbalance magnitude
            for use in training loss.

        Returns:
        --------
        delta_s: torch.Tensor
            Total imbalance magnitude (ΔS) or per-bus imbalance values.
        """
        delta_p, delta_q, delta_s = self.power_balance(data, layer_loss)
        data["bus"].delta_p = delta_p
        data["bus"].delta_q = delta_q
        return delta_s

    def message_passing_update(self, data):
        """
        Compute and aggregate messages along network branches.

        For each edge (i,j), a message is generated from the source node i
        based on its embedding and line parameters, and aggregated at the
        destination node j.

        Parameters
        ----------
        data: HeteroData
            Graph data with node and edge features.

        Returns:
        --------
        agg_msg_i: torch.Tensor 
            Aggregated messages for each node [num_nodes, hidden_dim].
        """
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
        """
        Apply the neural update block for iteration k.

        Parameters
        ----------
        data: HeteroData
            Graph data (updated in-place).
        k: int 
            Current iteration index.
        """
        messages = self.message_passing_update(data)
        self.layers[k](data, messages)


class NNUpdate(nn.Module):
    """
    Neural update block applied per iteration in the GNS architecture.

    Given current node states (v, θ), local power imbalances (ΔP, ΔQ),
    and aggregated messages, this module updates:
        - voltage magnitudes (v),
        - voltage angles (θ),
        - node memory embeddings (m).
    """
    def __init__(self, input_dim, hidden_dim):
        """ 
        Parameters
        ----------
        input_dim: int
            Dimensionality of concatenated input vector.
        hidden_dim: int
            Dimension of hidden and output embeddings.
        """
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
        """
        Apply per-node updates for voltage, angle, and embedding.

        Parameters
        ----------
        data: HeteroData
            Graph data containing bus-level states.
        messages: torch.Tensor
            Aggregated node messages [num_nodes, hidden_dim].
        """
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
        gen_idx = (data["bus"].bus_type != 1).nonzero(as_tuple=True)
        v_update[gen_idx] = 0
        data["bus"].v = data["bus"].v + v_update.squeeze(-1)

        # m update
        data["bus"].m = data["bus"].m + m_update
