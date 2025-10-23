# import torch
# import torch.nn as nn

# from torch_geometric.nn import GCN
# from core.utils.registry import registry

# # Graph Convolutional Network (GCN) for Power Flow Analysis

# # design choice: (edge weight) either resistance (r) or reactance (x)
# #                choose resistance (r) of the line, then try reactance (x) later

# # in_channels: (TODO)
# # out_channels: (6 for P, Q, V, theta, G, B)
# # hidden_channels: (128)
# # num_layers: (2)
# # dropout: (0.0)
# @registry.register_model("gcn_pf")
# class GCN_PF(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=2, dropout=0.0):
#         super().__init__()
#         # Define the GCN model
#         self.gcn = GCN(
#             in_channels=in_channels,
#             hidden_channels=hidden_channels,
#             out_channels=out_channels,
#             num_layers=num_layers,
#             dropout=dropout
#         )
        
#     def forward(self, data):
#         import ipdb
#         ipdb.set_trace()
#         # Assuming data contains x (node features) and edge_index (graph connectivity)
#         x = data["bus"].x  # Node features
#         edge_index = data["bus", "branch", "bus"].edge_index  # Graph connectivity
#         resistance = data["bus", "branch", "bus"].edge_attr[:,0]  # Edge weights [r, x, b, tau, angle]
#         return self.gcn(x, edge_index, edge_attr=resistance)

# from typing import List

"""
Adapted from Ojas Sanghi.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv

# from core.utils.custom_losses import DirichletEnergyLoss
from core.utils.registry import registry


@registry.register_model("graph_conv")
class GraphConvModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        # self.de = DirichletEnergyLoss()
        # self.energies: List[float] = []

        self.start_conv: GraphConv = GraphConv(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
        )
        self.end_conv: GraphConv = GraphConv(hidden_dim, out_dim)

    def forward(self, data: HeteroData) -> Tensor:
        x = data["bus"].x[:, 4:10]
        edge_index = data["bus", "branch", "bus"].edge_index
        edge_weights = data["bus", "branch", "bus"].edge_attr[:,0]  # Edge weights [r, x, b, tau, angle]
        
        # lap = self.de.get_graph_laplacian(edge_index, x.size(0))
        # self.energies = []
        
        x = self.start_conv(x=x, edge_index=edge_index, edge_weight=edge_weights)

        # energy = self.de.dirichlet_energy(x, lap)
        # self.energies.append(energy)

        x = F.relu(x)

        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_weight=edge_weights)

            # energy = self.de.dirichlet_energy(x, lap)
            # self.energies.append(energy)

            x = F.relu(x)
            
        x = self.end_conv(x=x, edge_index=edge_index, edge_weight=edge_weights)

        # energy = self.de.dirichlet_energy(x, lap)
        # self.energies.append(energy)

        return x