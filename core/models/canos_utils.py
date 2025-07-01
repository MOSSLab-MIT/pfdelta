import torch
import torch.nn as nn


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
        edge_hidden_dim = edges.get("('bus', 'ac_line', 'bus')", edges.get("('bus', 'branch', 'bus')")).shape[-1]
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
    

#  Decoder
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
