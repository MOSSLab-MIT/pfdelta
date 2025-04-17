import torch
import torch.nn as nn
from torch import scatter_add
import functools

from core.models.canos_utils import Encoder, InteractionNetwork, Decoder, EdgeUpdate, NodeUpdate
from core.utils.registry import registry

# CANOS Architecture
@registry.register_model("canos_opf")
class CANOS_OPF(nn.Module):
    def __init__(self, dataset, hidden_dim, include_sent_messages, k_steps):
        super().__init__()
        edge_feat_dim = node_feat_dim = hidden_dim

        # Define the encoder to get projected nodes and edges
        self.encoder = Encoder(data=dataset, hidden_size=hidden_dim)

        # Interaction network layers for message passing
        node_type_dict = {
            node_type: True
            for node_type in dataset[0].num_node_features.keys()
        }

        edge_type_dict = {
            str(edge_type): True if "edge_attr" in dataset[0][edge_type] else False
            for edge_type in dataset[0].edge_types
            if "bus" in edge_type[0]  # Only include edges where "bus" is the source
        }

        self.message_passing_layers = nn.ModuleList(
            InteractionNetwork(edge_type_dict=edge_type_dict,
                               node_type_dict=node_type_dict,
                               edge_dim=edge_feat_dim,
                               node_dim=node_feat_dim,
                               hidden_dim=hidden_dim,
                               include_sent_messages=include_sent_messages) for _ in range(k_steps))

        # Define the decoder to get the model outputs
        self.decoder = Decoder(hidden_size=hidden_dim)
        self.k_steps = k_steps

    def forward(self, data):
        # Encoding
        projected_nodes, projected_edges = self.encoder(data)

        # Message passing layers with residual connections
        nodes, edges = projected_nodes, projected_edges
        for l in range(self.k_steps):
            nodes, edges = self.message_passing_layers[l](nodes, edges, data)

        # Decoding
        output_dict = self.decoder(nodes, data)

        # Deriving branch flows
        p_fr, q_fr, p_to, q_to = self.derive_branch_flows(output_dict, data)
        output_dict["edge_preds"] = torch.stack([p_to, q_to, p_fr, q_fr], dim=-1) 

        return output_dict

    def derive_branch_flows(self, output_dict, data):
        
        device = data["x"].device

        # Create complex voltage
        va = output_dict["bus"][:, 0]
        vm = output_dict["bus"][:, -1]
        v_complex = vm * torch.exp(1j* va)

        # Edge index matrix
        edge_indices = torch.cat([data["bus", "ac_line", "bus"].edge_index,
                                  data["bus", "transformer", "bus"].edge_index], dim=-1)

        # Edge attributes matrix
        mask = torch.ones(9, dtype=torch.bool, device=device)
        mask[2] = False
        mask[3] = False
        ac_line_attr_masked = data["bus", "ac_line", "bus"].branch_vals[:, mask]
        # ac_line_attr_masked = data["bus", "ac_line", "bus"].edge_attr[:, mask]
        tap_shift = torch.cat([torch.ones((ac_line_attr_masked.shape[0], 1)), 
                               torch.zeros((ac_line_attr_masked.shape[0], 1))], dim=-1).to(device)
        ac_line_susceptances = data["bus", "ac_line", "bus"].branch_vals[:, 2:4]
        # ac_line_susceptances = data["bus", "ac_line", "bus"].edge_attr[:, 2:4]
        ac_line_attr = torch.cat([ac_line_attr_masked, tap_shift, ac_line_susceptances], dim=-1)

        edge_attr = torch.cat([ac_line_attr, data["bus", "transformer", "bus"].branch_vals], dim=0)
        # edge_attr = torch.cat([ac_line_attr, data["bus", "transformer", "bus"].edge_attr], dim=0)

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

        return p_fr, q_fr, p_to, q_to
