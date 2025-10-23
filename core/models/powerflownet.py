# This code is adapted and modified from the PowerFlowNet repository:
# https://github.com/StavrosOrf/PowerFlowNet/tree/main (GitHub: StavrosOrf)
#
# Original work is described in the following publication:
#   Nan Lin, Stavros Orfanoudakis, Nathan Ordonez Cardenas, Juan S. Giraldo,
#   and Pedro P. Vergara, "PowerFlowNet: Power flow approximation using
#   message passing Graph Neural Networks," International Journal of Electrical
#   Power & Energy Systems, vol. 160, pp. 110112, 2024.
#   https://doi.org/10.1016/j.ijepes.2024.110112
#
# Code was modified by Anvita Bhagavathula, Alvaro Carbonero, and Ana K. Rivera
# in January 2025.

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TAGConv
from torch_geometric.utils import degree
from torch_geometric.data import Data, HeteroData

from core.utils.registry import registry


def place_known_values(network_values, data):
    """
    network_values is a vector of size n_nodes x 4, where the four node features
    are vm, va, p, q. data contains all sorts of data inputs. In particular, it
    needs to contain a vector for all vm, va, p, and q values, as well as a
    vector with the bus types.
    """
    bus_types = data["bus"].bus_type
    vm = data["bus"].vm
    va = data["bus"].va
    p = data["bus"].p
    q = data["bus"].q

    # Mask in PQ values
    mask = bus_types == 1
    network_values[mask, 2] = p[mask]
    network_values[mask, 3] = q[mask]

    # Mask in PV values
    mask = bus_types == 2
    network_values[mask, 2] = p[mask]
    network_values[mask, 0] = vm[mask]

    # Mask in VTheta values
    mask = bus_types == 3
    network_values[mask, 0] = vm[mask]
    network_values[mask, 1] = va[mask]

    return network_values


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


@registry.register_model("powerflownet")
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
        if (n_gnn_layers != 1):
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
        This method allows us to distinguish between the original dataset on
        which powerflownet was developed, and our proposed benchmark. We use
        Hetero graphs, while the original powerflownet uses Homogeneous graphs.
        """
        if isinstance(data, HeteroData):
            x = data["bus"].x[:, 4:4+self.nfeature_dim] # (N, 16)
            mask = data["bus"].x[:, -self.nfeature_dim:]
            edge_index = data["bus", "branch", "bus"].edge_index
            edge_features = data["bus", "branch", "bus"].edge_attr
        elif isinstance(data, Data):
            assert data.x.shape[-1] == self.nfeature_dim * 2 + 4
            x = data.x[:, 4:4+self.nfeature_dim]
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
        # x = x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying.

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

        # Mask out known values
        # x = place_known_values(x, data)
        return x
