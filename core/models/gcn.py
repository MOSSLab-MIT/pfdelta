import torch
import torch.nn as nn

from torch_geometric.nn import GCN
from core.utils.registry import registry

# Graph Convolutional Network (GCN) for Power Flow Analysis

# design choice: (edge weight) either resistance (r) or reactance (x)
#                choose resistance (r) of the line, then try reactance (x) later

# in_channels: (TODO)
# out_channels: (6 for P, Q, V, theta, G, B)
# hidden_channels: (128)
# num_layers: (2)
# dropout: (0.0)
@registry.register_model("gcn_pf")
class GCN_PF(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=2, dropout=0.0):
        super().__init__()
        # Define the GCN model
        self.gcn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, data):
        # Assuming data contains x (node features) and edge_index (graph connectivity)
        x = data["bus", "branch", "bus"].x  # Node features
        edge_index = data["bus", "branch", "bus"].edge_index  # Graph connectivity
        resistance = data["bus", "branch", "bus"].edge_attr[:,0]  # Edge weights [r, x, b, tau, angle]
        return self.gcn(x, edge_index, resistance)