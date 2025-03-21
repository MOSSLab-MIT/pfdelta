import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch import scatter_add
import torch.optim as optim
import copy

class Encoder(nn.Module):
    def __init__(self, data, hidden_size: int):
        super(Encoder, self).__init__()
        # Linear projection for all node features
        self.node_projections = nn.ModuleDict({
            node_type: nn.Linear(data.num_node_features[node_type], hidden_size)
            for node_type in data.num_node_features.keys()
        })
        # Linear projection for all edge features
        self.edge_projections = nn.ModuleDict({
            str(edge_type): nn.Linear(data.num_edge_features[edge_type], hidden_size)
            for edge_type in data.num_edge_features.keys() if data.num_edge_features[edge_type] != 0
                   # so we’re not including subnode links which have no attributes.
        })
    def forward(self, data):
        projected_nodes = {
            node_type: self.node_projections[node_type](data[node_type].x)
            for node_type in data.num_node_features.keys()
        }

        projected_edges = {
            str(edge_type): self.edge_projections[str(edge_type)](data[edge_type].edge_attr)
            if "edge_attr" in data[edge_type] else None
            for edge_type in data.edge_types
        }

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
        edge_hidden_dim = edges["('bus', 'ac_line', 'bus')"].shape[-1]
        sent_received_node_type = {node_type: torch.zeros(n.shape[0], edge_hidden_dim) for node_type, n in nodes.items()}
        updated_nodes_dict = nodes
        updated_edges_dict = edges

        for edge_type, edge_feats in edges.items():
            edge_type_tuple = tuple(edge_type.strip("()").replace("'", "").split(", "))
            senders, receivers = data[edge_type_tuple].edge_index
            if edge_type == "('bus', 'ac_line', 'bus')" or edge_type == "('bus', 'transformer', 'bus')":
                sender_features, receiver_features = nodes["bus"][senders], nodes["bus"][receivers]
                updated_edges = self.edge_update(edge_feats, sender_features, receiver_features, edge_type)
                sent_received_node_type["bus"].scatter_add_(0, receivers.unsqueeze(-1).expand_as(updated_edges), updated_edges)
                if self.include_sent_messages:
                    sent_received_node_type["bus"].scatter_add_(0, senders.unsqueeze(-1).expand_as(updated_edges), updated_edges)

            elif edge_type == "('bus', 'generator_link', 'generator')":
                sender_features, receiver_features = nodes["bus"][senders], nodes["generator"][receivers]
                updated_edges = self.edge_update(edge_feats, sender_features, receiver_features, edge_type)
                sent_received_node_type["generator"].scatter_add_(0, receivers.unsqueeze(-1).expand_as(updated_edges), updated_edges)
                if self.include_sent_messages:
                    sent_received_node_type["bus"].scatter_add_(0, senders.unsqueeze(-1).expand_as(updated_edges), updated_edges)

            elif edge_type == "('bus', 'load_link', 'load')":
                sender_features, receiver_features = nodes["bus"][senders], nodes["load"][receivers]
                updated_edges = self.edge_update(edge_feats, sender_features, receiver_features, edge_type)
                sent_received_node_type["load"].scatter_add_(0, receivers.unsqueeze(-1).expand_as(updated_edges), updated_edges)
                if self.include_sent_messages:
                    sent_received_node_type["bus"].scatter_add_(0, senders.unsqueeze(-1).expand_as(updated_edges), updated_edges)

            elif edge_type == "('bus', 'shunt_link', 'shunt')":
                sender_features, receiver_features = nodes["bus"][senders], nodes["shunt"][receivers]
                updated_edges = self.edge_update(edge_feats, sender_features, receiver_features, edge_type)
                sent_received_node_type["shunt"].scatter_add_(0, receivers.unsqueeze(-1).expand_as(updated_edges), updated_edges)
                if self.include_sent_messages:
                    sent_received_node_type["bus"].scatter_add_(0, senders.unsqueeze(-1).expand_as(updated_edges), updated_edges)

            updated_edges_dict[edge_type] = updated_edges

        # Apply the object model phi_o (node_update)
        # phi_o is applied to the aggregated edge features (with sent and recieved messages)
        for node_type, node_feats in nodes.items():
            updated_nodes = self.node_update(node_feats, sent_received_node_type[node_type], node_type)
            updated_nodes_dict[node_type] = updated_nodes

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
        ) if feats else nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        ) for edge_type, feats in edge_type_dict.items()})

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
        if edge_type == "('bus', 'ac_line', 'bus')" or edge_type == "('bus', 'transformer', 'bus')":
            x = torch.cat([edges, sender_features, receiver_features], dim=-1)
            return self.mlps[edge_type](x)

        x = torch.cat([sender_features, receiver_features], dim=-1)
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
    
#  Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size: int):
        super(Decoder, self).__init__()

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
        pmin, pmax = data["generator"].x[:, 2:4].T
        qmin, qmax = data["generator"].x[:, 5:7].T
        vmin, vmax = data["bus"].x[:, 2:].T

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

        return output_dict # dict with outputs for voltages and generators.

# CANOS Architecture
class CANOS(nn.Module):
    def __init__(self, data, encoder, interaction_network, edge_feat_dim, node_feat_dim, hidden_dim,
                 decoder, include_sent_messages, k_steps):
        super().__init__()

        # Define the encoder to get projected nodes and edges
        self.encoder = encoder(data=data, hidden_size=hidden_dim)

        # Interaction network layers for message passing
        node_type_dict = {
            node_type: True
            for node_type in data[0].num_node_features.keys()
        }

        edge_type_dict = {
            str(edge_type): True if "edge_attr" in data[0][edge_type] else False
            for edge_type in data[0].edge_types
            if "bus" in edge_type[0]  # Only include edges where "bus" is the source
        }

        self.message_passing_layers = nn.ModuleList(
            interaction_network(edge_type_dict=edge_type_dict,
                                  node_type_dict=node_type_dict,
                                  edge_dim=edge_feat_dim,
                                  node_dim=node_feat_dim,
                                  hidden_dim=hidden_dim,
                                  include_sent_messages=include_sent_messages)
         for _ in range(k_steps))

        # Define the decoder to get the model outputs
        self.decoder = decoder(hidden_size=hidden_dim)
        self.k_steps = k_steps

    def forward(self, data):
        # Encoding
        projected_nodes, projected_edges = self.encoder(data)

        # Message passing layers with residual connections
        nodes, edges = projected_nodes, projected_edges
        for l in range(self.k_steps):
            new_nodes, new_edges = self.message_passing_layers[l](nodes, edges, data)

            # USE MAP REDUCE HERE FOR RESIDUAL CONNECTION ADDING
            # >>> import collections, functools, operator
            # >>> dict1 = { 'dog': np.array([3]), 'cat':np.array([2]) }
            # >>> dict2 = { 'dog': np.array([4]), 'cat':np.array([7]) }
            # >>> result = dict(functools.reduce(operator.add, map(collections.Counter, [dict1, dict2])))
            # >>> result
            # {'dog': array([7]), 'cat': array([9])

            # Residual connection (sum the previous input with the output)
            nodes = {key: torch.add(nodes[key], new_nodes[key]) for key in nodes}
            edges = {key: torch.add(edges[key], new_edges[key]) for key in edges}

        # Decoding
        output_dict = self.decoder(nodes, data)

        # Deriving branch flows
        p_fr, q_fr, p_to, q_to = self.derive_branch_flows(output_dict, data)

        return output_dict, p_fr, q_fr, p_to, q_to

    def derive_branch_flows(self, output_dict, data):

        # Create complex voltage
        va = output_dict["bus"][:, 0]
        vm = output_dict["bus"][:, -1]
        v_complex = vm * torch.exp(1j* va)

        # Edge index matrix
        edge_indices = torch.cat([data["bus", "ac_line", "bus"].edge_index, data["bus", "transformer", "bus"].edge_index], dim=-1)

        # Edge attributes matrix
        mask = torch.ones(9, dtype=torch.bool)
        mask[2] = False
        mask[3] = False
        ac_line_attr_masked = data["bus", "ac_line", "bus"].edge_attr[:, mask]
        tap_shift = torch.cat([torch.ones((ac_line_attr_masked.shape[0], 1)), torch.zeros((ac_line_attr_masked.shape[0], 1)) ], dim=-1)
        ac_line_susceptances = data["bus", "ac_line", "bus"].edge_attr[:, 2:4]
        ac_line_attr = torch.cat([ac_line_attr_masked, tap_shift, ac_line_susceptances], dim=-1)

        edge_attr = torch.cat([ac_line_attr, data["bus", "transformer", "bus"].edge_attr], dim=0)

        # Extract parameters
        br_r, br_x = edge_attr[:, 2], edge_attr[:, 3]
        b_fr, b_to = edge_attr[:, -2], edge_attr[:, -1]
        tap = edge_attr[:, -4]
        shift = edge_attr[:, -3]

        # Compute admittances and tap complex transformation
        Y_branch = 1 / (br_r + 1j * br_x)
        Y_c_fr = 1j * b_fr
        Y_c_to = 1j * b_to
        T_complex = tap * torch.exp(1j * shift)

        # Get sending and receiving buses
        i, j = edge_indices[0], edge_indices[1]
        vi = v_complex[i]
        vj = v_complex[j]

        # Compute complex branch flows
        S_fr = (Y_branch + Y_c_fr).conj() * (torch.abs(vi) ** 2) / (torch.abs(T_complex) ** 2) - \
            Y_branch.conj() * (vi * vj.conj()) / T_complex

        S_to = (Y_branch + Y_c_to).conj() * (torch.abs(vj) ** 2) - \
            Y_branch.conj() * (vj * vi.conj()) / T_complex.conj()

        # Extract real and reactive power flows
        p_fr, q_fr = S_fr.real, S_fr.imag
        p_to, q_to = S_to.real, S_to.imag

        # ARE THE BRANCH FLOWS SUPPOSED TO BE CALCULATED AND UPDATED IN THE EDGE FEATURES THEMSELVES?
        # DOES THIS OUTPUT FORMAT MAKE SENSE?

        return p_fr, q_fr, p_to, q_to
