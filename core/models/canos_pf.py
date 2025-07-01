import torch
import torch.nn as nn
from torch import scatter_add
import functools

from core.models.canos_utils import Encoder, InteractionNetwork, DecoderPF, EdgeUpdate, NodeUpdate
from core.utils.registry import registry

# CANOS Architecture
@registry.register_model("canos_pf")
class CANOS_PF(nn.Module):
    def __init__(self, dataset, hidden_dim, include_sent_messages, k_steps):
        super().__init__()
        edge_feat_dim = node_feat_dim = hidden_dim

        # Define the encoder to get projected nodes and edges
        self.case_name = dataset.case_name
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
        self.decoder = DecoderPF(hidden_size=hidden_dim)
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
        output_dict["edge_preds"] = torch.stack([p_fr, q_fr, p_to, q_to], dim=-1) 

        # Deriving slack generation
        p_slack_net, q_slack_net = self.derive_slack_output(output_dict, data)
        output_dict["slack"] = torch.stack([p_slack_net, q_slack_net], dim=-1)

        output_dict["casename"] = self.case_name

        return output_dict

    def derive_branch_flows(self, output_dict, data):

        # Create complex voltage
        va = output_dict["bus"][:, 0]
        vm = output_dict["bus"][:, -1]
        v_complex = vm * torch.exp(1j* va)

        # Extract edge info
        edge_index = data["bus", "branch", "bus"].edge_index
        edge_attr = data["bus", "branch", "bus"].edge_attr

        # Unpack edge features
        br_r, br_x = edge_attr[:, 0], edge_attr[:, 1]
        b_fr, b_to = edge_attr[:, 3], edge_attr[:, 5]
        g_fr, g_to = edge_attr[:, 2], edge_attr[:, 4]
        tap = edge_attr[:, 6]
        shift = edge_attr[:, 7]

        # Complex tap ratio
        T_complex = tap * torch.exp(1j * shift)

        # Complex admittances
        Y_branch = 1 / (br_r + 1j * br_x)
        Y_c_fr = 1j * b_fr
        Y_c_to = 1j * b_to

        # Node voltages
        i, j = edge_index[0], edge_index[1]
        vi = v_complex[i]
        vj = v_complex[j]
        
        # Compute complex branch flows
        S_fr = (Y_branch + Y_c_fr).conj() * (torch.abs(vi) ** 2) / (torch.abs(T_complex) ** 2) - \
            Y_branch.conj() * (vi * vj.conj()) / T_complex

        S_to = (Y_branch + Y_c_to).conj() * (torch.abs(vj) ** 2) - \
            Y_branch.conj() * (vj * vi.conj()) / T_complex.conj()

        # Real/reactive power flows
        p_fr, q_fr = S_fr.real, S_fr.imag
        p_to, q_to = S_to.real, S_to.imag

        return p_fr, q_fr, p_to, q_to

    def derive_slack_output(self, output_dict, data):

        device = data["bus"].x.device
        n = data["bus"].x.shape[0]
        bus_shunts = data["bus"].shunt.to(device)  
        slack_idx = data["slack", "slack_link", "bus"].edge_index[1]
        edge_pred = output_dict["edge_preds"]
        edge_indices = data["bus", "branch", "bus"].edge_index
        flows_rev = edge_pred[:, 2] + 1j * edge_pred[:, 3]  
        flows_fwd = edge_pred[:, 0] + 1j * edge_pred[:, 1]  
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)
        bus_pred = output_dict["bus"]
        va, vm = bus_pred.T

        shunt_flows = (torch.abs(vm) ** 2) * (bus_shunts[:, 1] + 1j * bus_shunts[:, 0])  # (b_shunt + j*g_shunt)
        slack_net_generation = sum_branch_flows[slack_idx] + shunt_flows[slack_idx]
        p_slack_net = torch.real(slack_net_generation)
        q_slack_net = torch.imag(slack_net_generation)

        return p_slack_net, q_slack_net