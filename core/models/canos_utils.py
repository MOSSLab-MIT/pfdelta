# This code is a PyTorch implementation of the Encoder, Interaction Network 
# and the Decoder components of the CANOS model originally proposed in the 
# following papers: 
# 
#   Piloto, L., Liguori, S., Madjiheurem, S., Zgubic, M., Lovett, S., 
#   Tomlinson, H., … Witherspoon, S. (2024). CANOS: A Fast and Scalable 
#   Neural AC-OPF Solver Robust To N-1 Perturbations. arXiv [Cs.LG]. 
#   Retrieved from http://arxiv.org/abs/2403.17660
#
#   Battaglia, P. W., Pascanu, R., Lai, M., Rezende, D., & Kavukcuoglu, K. 
#   (2016). Interaction Networks for Learning about Objects, Relations and 
#   Physics. arXiv [Cs.AI]. Retrieved from http://arxiv.org/abs/1612.00222
# 
# Code was implemented by Anvita Bhagavathula, Alvaro Carbonero, and Ana K. Rivera 
# in April 2025.

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder module for projecting raw node and edge features into latent space.
    """
    def __init__(self, data, hidden_size: int):
        """ 
        This component performs type-specific linear projections for each node and edge
        feature set in a heterogeneous graph, preparing them for message passing
        in the Interaction Network.

        Params
        ----------
        data : HeteroData
            A PyTorch Geometric `HeteroData` object defining node/edge feature dimensions.
        hidden_size : int
            Dimensionality of the latent embedding space.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # Linear projection for all node features
        self.node_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(data.num_node_features[node_type], hidden_size)
                for node_type in data.num_node_features.keys()
            }
        )
        # Linear projection for all edge features
        self.edge_projections = nn.ModuleDict(
            {
                str(edge_type): nn.Linear(
                    data.num_edge_features[edge_type], hidden_size
                )
                for edge_type in data.num_edge_features.keys()
                if data.num_edge_features[edge_type] != 0
                # not including subnode links which have no attributes.
            }
        )

    def forward(self, data):
        """
        Forward pass for the encoder.

        Parameters
        ----------
        data : HeteroData
            Input graph containing `x` and `edge_attr` attributes for each node/edge type.

        Returns
        -------
        projected_nodes : dict[str, Tensor]
            Dictionary of each node type mapped to its high dimensional features 
        projected_edges : dict[str, Tensor]
            Dictionary of each edge edge type mapped to its high dimensional features
        """
        device = data["bus"].x.device
        projected_nodes = {
            node_type: self.node_projections[node_type](data[node_type].x)
            for node_type in data.num_node_features.keys()
        }

        projected_edges = {}
        for edge_type in data.edge_types:
            if "edge_attr" in data[edge_type]:
                projected_edges[str(edge_type)] = self.edge_projections[str(edge_type)](
                    data[edge_type].edge_attr
                )
            elif edge_type[2] != "bus":
                num_edges = data[edge_type]["edge_index"].shape[1]
                projected_edges[str(edge_type)] = torch.zeros(
                    (num_edges, self.hidden_size), device=device
                )

        return projected_nodes, projected_edges


class InteractionNetwork(nn.Module):
    """
    Core message-passing module implementing the Interaction Network architecture.
    """
    def __init__(
        self,
        edge_type_dict,
        node_type_dict,
        edge_dim,
        node_dim,
        hidden_dim,
        include_sent_messages=False,
    ):
        """
        Each forward pass updates both edge and node features based on the latent
        representations produced by the `Encoder`. The update functions are defined
        by the `EdgeUpdate` (relation model φ_r) and `NodeUpdate` (object model φ_o)
        submodules.

        Parameters
        ----------
        edge_type_dict : dict
            Mapping of edge types to bool indicating whether to constructing 
            edge-specific MLPs for this edge type.
        node_type_dict : dict
            Mapping of node types to bool indicating whether to constructing 
            node-specific MLPs for this node type.
        edge_dim : int
            Dimensionality of edge embeddings.
        node_dim : int
            Dimensionality of node embeddings.
        hidden_dim : int
            Hidden layer dimensionality for all MLPs.
        include_sent_messages : bool, optional
            Whether to aggregate outgoing edge messages in node updates.
        """
        super().__init__()
        self.include_sent_messages = include_sent_messages
        self.edge_update = EdgeUpdate(edge_dim, node_dim, hidden_dim, edge_type_dict)
        self.node_update = NodeUpdate(
            node_dim, hidden_dim, node_type_dict, self.include_sent_messages
        )

    def forward(self, nodes, edges, data):
        """
        Forward pass of the Interaction Network.

        Parameters
        ----------
        nodes : dict[str, Tensor]
            Node embeddings by type, each of shape [num_nodes, node_dim].
        edges : dict[str, Tensor]
            Edge embeddings by type, each of shape [num_edges, edge_dim].
        data : HeteroData
            PyTorch Geometric graph with `edge_index` mappings between node types.

        Returns
        -------
        updated_nodes_dict : dict[str, Tensor]
            Updated node embeddings for each type.
        updated_edges_dict : dict[str, Tensor]
            Updated edge embeddings for each type.
        """
        # Apply marshalling and relational model phi_r (edge update)
        # phi_r is applied onto src node features, dst node features, and edges
        device = data["bus"].x.device
        edge_hidden_dim = edges.get(
            "('bus', 'ac_line', 'bus')", edges.get("('bus', 'branch', 'bus')")
        ).shape[-1]
        sent_received_node_type = {
            node_type: torch.zeros(n.shape[0], edge_hidden_dim, device=device)
            for node_type, n in nodes.items()
        }
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
            updated_edges = self.edge_update(
                edge_feats, sender_features, receiver_features, edge_type
            )

            # Pass messages
            sent_received_node_type[receiver_type].scatter_add_(
                0, receivers.unsqueeze(-1).expand_as(updated_edges), updated_edges
            )
            if self.include_sent_messages:
                sent_received_node_type[sender_type].scatter_add_(
                    0, senders.unsqueeze(-1).expand_as(updated_edges), updated_edges
                )

            updated_edges_dict[edge_type] = updated_edges + edge_feats

        # Apply the object model phi_o (node_update)
        # phi_o is applied to the aggregated edge features (with sent and recieved messages)
        for node_type, node_feats in nodes.items():
            updated_nodes = self.node_update(
                node_feats, sent_received_node_type[node_type], node_type
            )
            updated_nodes_dict[node_type] = updated_nodes + node_feats
        return updated_nodes_dict, updated_edges_dict


# relational and object models (phi_r and phi_o)
class EdgeUpdate(nn.Module):
    """
    Relational model (φ_r) for updating edge features based 
    on sender and receiver node embeddings.
    """
    def __init__(self, edge_dim, node_dim, hidden_dim, edge_type_dict):
        """
        Edge update function for updating edge features.

        Parameters:
        ----------
        edge_dim: int
            Dimension of edge features.
        node_dim: int 
            Dimension of node features.
        hidden_dim: int 
            Hidden layer size.
        out_dim: int
            Output edge feature dimension.
        """
        super().__init__()
        self.mlps = nn.ModuleDict(
            {
                edge_type: nn.Sequential(
                    nn.Linear(edge_dim + 2 * node_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, edge_dim),
                )
                for edge_type in edge_type_dict.keys()
            }
        )

    def forward(self, edges, sender_features, receiver_features, edge_type):
        """
        Compute updated edge features.

        Parameters:
        ----------
        edges: torch.Tensor
            Shape [num_edges, edge_feat_dim].
        sender_features: torch.Tensor
            Shape [num_edges, node_feat_dim].
        receiver_features: torch.Tensor 
            Shape [num_edges, node_feat_dim].

        Returns:
        -------
        Tensor: 
            Edge-specific MLPs applied to compute updated edge 
            features of shape [num_edges, edge_dim].
        """
        x = torch.cat([edges, sender_features, receiver_features], dim=-1)
        return self.mlps[edge_type](x)


class NodeUpdate(nn.Module):
    """
    Object model (φ_o) for updating node features using aggregated edge messages.
    """
    def __init__(
        self, node_dim, hidden_dim, node_type_dict, include_sent_messages=False
    ):
        """
        Node update module for updating node features.

        Parameters:
        ----------
        input_dim: int
            Dimension of node features.
        output_dim: int 
            Output node feature dimension.
        include_sent_messages: bool 
            Whether to include messages from sender edges
        """
        super().__init__()
        self.include_sent_messages = include_sent_messages
        self.mlps = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(node_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, node_dim),
                )
                for node_type in node_type_dict.keys()
            }
        )

    def forward(self, node_features, updated_messages, node_type):
        """
        Compute updated node features.

        Parameters
        ----------
        node_features (Tensor): 
            Shape [num_nodes, node_feat_dim].
        received_messages (Tensor): 
            Shape [num_nodes, node_feat_dim].
        sent_messages (Tensor, optional): 
            Shape [num_nodes, node_feat_dim].

        Returns:
        -------
        Tensor: 
            Node-specific MLPs applied to compute updated node
            features of shape [num_nodes, node_dim].
        """
        x = torch.cat([node_features, updated_messages], dim=-1)
        return self.mlps[node_type](x)


class DecoderOPF(nn.Module):
    """
    Decoder for the AC-OPF task.
    """
    def __init__(self, hidden_size: int):
        """ 
        Maps latent bus and generator embeddings to 
        physical quantities (voltage magnitude, 
        voltage angle, active/reactive power).

        Parameters
        ----------
        hidden_size : int
            Dimensionality of the latent embedding space.
        """
        super(DecoderOPF, self).__init__()

        # Linear projection for all node features
        self.node_decodings = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Linear(256, 2),
                )
                for node_type in ["bus", "generator"]
            }
        )

    def forward(self, node_dict, data):
        """
        Decode latent node embeddings to OPF quantities.

        Parameters
        ----------
        node_dict : dict[str, Tensor]
            Latent node embeddings by type.
        data : HeteroData
            Graph containing voltage and power bounds.

        Returns
        -------
        output_dict : dict
            Dictionary containing model predictions:
            - output_dict["bus"] : torch.Tensor of shape (n_bus, 2)
              Predicted bus voltage angle [rad] and magnitude [p.u.].
            - output_dict["generator"] torch.Tensor of shape (n_gen, 2)
              Predicted active and reactive power generation values. 
        """
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
        output_pg = (
            torch.sigmoid(output_nodes["generator"][:, 0]) * (pmax - pmin) + pmin
        )
        output_qg = (
            torch.sigmoid(output_nodes["generator"][:, -1]) * (qmax - qmin) + qmin
        )

        output_dict = {
            "bus": torch.stack([output_va, output_vm], dim=1),
            "generator": torch.stack([output_pg, output_qg], dim=1),
        }

        return output_dict


class DecoderPF(nn.Module):
    """
    Decoder for the AC-OPF task.
    """
    def __init__(self, hidden_size: int):
        """ 
        Maps latent PV, PQ, slack embeddings to 
        physical quantities (voltage magnitude, 
        voltage angle, active/reactive power).

        Parameters
        ----------
        hidden_size : int
            Dimensionality of the latent embedding space.
        """
        super(DecoderPF, self).__init__()

        # Linear projection for all node features
        self.node_decodings = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Linear(256, 2),
                )
                for node_type in ["PV", "PQ", "slack"]
            }
        )

    def forward(self, node_dict, data):
        """
        Decode latent node embeddings to reconstruct bus-level state.

        Parameters
        ----------
        node_dict : dict[str, Tensor]
            Latent node embeddings for PV, PQ, and slack buses.
        data : HeteroData
            Graph defining bus-link relations.

        Returns
        -------
        output_dict : dict
            Dictionary containing model predictions:
            - output_dict["bus"] : torch.Tensor of shape (n_bus, 2)
              Predicted bus voltage angle [rad] and magnitude [p.u.].
            - output_dict["PV"] : torch.Tensor of shape (n_PV, 2)
              Predicted bus voltage angle [rad] and reactive power generation
            - output_dict["PQ"] : torch.Tensor of shape (n_PQ, 2)
              Predicted bus voltage angle [rad] and magnitude [p.u.].
            - output_dict["slack"] : torch.Tensor of shape (1, 2)
              Net active/reactive power generation at the slack bus [p.u.].
        """
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
