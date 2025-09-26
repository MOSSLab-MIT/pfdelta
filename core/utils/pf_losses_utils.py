import os
import pickle
import torch

from torch.nn.functional import mse_loss
from torch_geometric.nn import global_add_pool

from core.utils.registry import registry
from core.datasets.dataset_utils import canos_pf_data_mean0_var1
from core.datasets.data_stats import canos_pfdelta_stats, pfnet_pfdata_stats

@registry.register_loss("universal_power_balance")
class PowerBalanceLoss:
    """
    Compute the power balance loss.

    Parameters:
        predictions (torch.Tensor): Model predictions (num_nodes, 4)
                                    where each row represents[V, θ, Pg, Qg]
        targets (torch.Tensor): Ground truth values (num_nodes, 4) where
                                each row represents [V, θ, Pd, Qd]
        bus_type (torch.Tensor): Tensor (num_nodes,) that indicates bus type.
        edge_index (torch.Tensor): Graph connectivity in COO format (2, num_edges).
        edge_attr (torch.Tensor): Edge features (num_edges, edge_dim).

    Returns:
        torch.Tensor: Loss value; sum of squared power mismatches.
    """
    def __init__(self, model): 
        self.power_balance_mean = None 
        self.power_balance_max = None
        self.power_balance_l2 = None
        self.model = model
        self.loss_name = "PBL Mean"

    def __call__(self, outputs, data):
        power_balance_model_preds = self.collect_model_predictions(self.model, data, outputs)
        predictions = power_balance_model_preds["predictions"]
        edge_attr = power_balance_model_preds["edge_attr"]
        edge_index = data["bus", "branch","bus"].edge_index
    
        shunt_g = data["bus"].shunt[:, 0]
        shunt_b = data["bus"].shunt[:, 1]
        # Unpack node feature and edge feature values values
        V_pred, theta_pred, Pnet, Qnet = predictions 
        r, x, bs, tau, theta_shift = edge_attr 

        # Compute bus-level power injections based on predictions
        Pbus_pred = torch.zeros_like(V_pred)
        Qbus_pred = torch.zeros_like(V_pred)
        src, dst = edge_index

        Y_real = torch.real(1/(r + 1j*x))
        Y_imag = torch.imag(1/(r + 1j*x))
        suscept = bs
        delta_theta1 = theta_pred[src] - theta_pred[dst] 
        delta_theta2 = theta_pred[dst] - theta_pred[src] 
        # NOTE: difference in the delta_ij is added to gns loss
        # Active power flows
        P_flow_src = V_pred[src] * V_pred[dst] / tau * (
                    -Y_real * torch.cos(delta_theta1 - theta_shift) - Y_imag * torch.sin(delta_theta1 - theta_shift)
        ) + Y_real * (V_pred[src] / tau)**2

        P_flow_dst = V_pred[dst] * V_pred[src] / tau * (
                    -Y_real * torch.cos(delta_theta2 - theta_shift) - Y_imag * torch.sin(delta_theta2 - theta_shift)
        ) + Y_real * V_pred[dst]**2

        # Reactive power flows
        Q_flow_src = V_pred[src] * V_pred[dst] / tau * (
                    -Y_real * torch.sin(delta_theta1 - theta_shift) + Y_imag * torch.cos(delta_theta1 - theta_shift)
        ) - (Y_imag + suscept / 2) * (V_pred[src] / tau)**2

        Q_flow_dst = V_pred[dst] * V_pred[src] / tau * (
                    -Y_real * torch.sin(delta_theta2 - theta_shift) + Y_imag * torch.cos(delta_theta2 - theta_shift)
        ) - (Y_imag + suscept / 2) * V_pred[dst]**2

        # Aggregate contributions for all nodes
        Pbus_pred = torch.zeros_like(V_pred).scatter_add_(0, src, P_flow_src)
        Pbus_pred = Pbus_pred.scatter_add_(0, dst, P_flow_dst)

        Qbus_pred = torch.zeros_like(V_pred).scatter_add_(0, src, Q_flow_src)
        Qbus_pred = Qbus_pred.scatter_add_(0, dst, Q_flow_dst)

        # Calculate the power mismatches ΔP and ΔQ
        delta_P = Pnet - Pbus_pred - V_pred**2 * shunt_g
        delta_Q = Qnet - Qbus_pred + V_pred**2 * shunt_b 

        # Compute the loss as the sum of squared mismatches
        delta_PQ_2 = delta_P**2 + delta_Q**2

        # Calculate PBL Mean
        delta_PQ_magnitude = torch.sqrt(delta_PQ_2)
        self.power_balance_mean = delta_PQ_magnitude.mean()

        # Calculate PBL L2
        batch_idx = data["bus"].batch
        batched_power_balance_l2 = torch.sqrt(global_add_pool(delta_PQ_2, batch_idx))
        self.power_balance_l2 = batched_power_balance_l2.mean()

        # Calculate PBL Max
        self.power_balance_max = delta_PQ_magnitude.max()

        # new_max = delta_PQ_magnitude.max()
        # if self.power_balance_max is None:
        #     self.power_balance_max = new_max
        # else:
        #     self.power_balance_max = delta_PQ_magnitude.max() # TODO: fix this

        return self.power_balance_mean


    def collect_model_predictions(self, model_name, data, output=None):
        """ """
        power_balance_model_preds = {}
        if model_name == "CANOS":
            device = data["bus"].x.device
            num_buses = data["bus"].num_nodes
            bus_output = output["bus"] 
            theta_pred = bus_output[:, 0] 
            V_pred = bus_output[:, 1] 
            Pnet = torch.zeros(num_buses, device=device)
            Qnet = torch.zeros(num_buses, device=device)

            # PQ
            pq_idx = data["PQ", "PQ_link", "bus"].edge_index[1]
            bus_gen = data['bus'].bus_gen[pq_idx]
            pq_pg = bus_gen[:, 0]
            pq_qg = bus_gen[:, 1]
            Pnet[pq_idx] = pq_pg - data["PQ"].x[:, 0]
            Qnet[pq_idx] = pq_qg - data["PQ"].x[:, 1]

            # PV
            pv_idx = data["PV", "PV_link", "bus"].edge_index[1]
            pv_outputs = output["PV"]
            Pnet[pv_idx] = data["PV"].x[:, 0]
            Qnet[pv_idx] = pv_outputs[:, 0]

            # Slack
            slack_idx = data["slack", "slack_link", "bus"].edge_index[1]
            slack_outputs = output["slack"]
            Pnet[slack_idx] = slack_outputs[:, 0]
            Qnet[slack_idx] = slack_outputs[:, 1]

            # Collect edge attributes
            edge_attr = data["bus", "branch","bus"].edge_attr
            r = edge_attr[:, 0]
            x = edge_attr[:, 1]
            bs = edge_attr[:, 3] + edge_attr[:, 5]
            tau = edge_attr[:, 6]
            theta_shift = edge_attr[:, 7]

            power_balance_model_preds["predictions"] = (V_pred, theta_pred, Pnet, Qnet)
            power_balance_model_preds["edge_attr"] = (r, x, bs, tau, theta_shift)

            return power_balance_model_preds

        elif model_name == "PFNet":
            device = data["bus"].x.device
            # Unnormalize predictions
            casename = data.case_name[0]
            mean = pfnet_pfdata_stats[casename]["mean"]["bus"]["y"].to(device)
            std = pfnet_pfdata_stats[casename]["std"]["bus"]["y"].to(device)
            output = (output * std) + mean
            unnormalized_y = (data["bus"].y * std) + mean

            num_buses = data["bus"].num_nodes
            theta_pred = output[:, 1]
            V_pred = output[:, 0]
            Pnet = torch.zeros(num_buses, device=device)
            Qnet = torch.zeros(num_buses, device=device)

            # PQ
            pq_idx = (data['bus'].bus_type == 1).nonzero(as_tuple=True)[0]
            bus_gen = data['bus'].bus_gen[pq_idx]
            pq_pg = bus_gen[:, 0]
            pq_qg = bus_gen[:, 1]
            pq_outputs = output[pq_idx]
            Pnet[pq_idx] = pq_pg - unnormalized_y[pq_idx, 2] # P fixed in PQ
            Qnet[pq_idx] = pq_qg - unnormalized_y[pq_idx, 3] # Q fixed in PQ

            # PV
            pv_idx = (data['bus'].bus_type == 2).nonzero(as_tuple=True)[0]
            pv_outputs = output[pv_idx]
            Pnet[pv_idx] = unnormalized_y[pv_idx, 2] # P fixed in PV
            Qnet[pv_idx] = pv_outputs[:, 3]

            # Slack
            slack_idx = (data['bus'].bus_type == 3).nonzero(as_tuple=True)[0]
            slack_outputs = output[slack_idx]
            Pnet[slack_idx] = slack_outputs[:, 2]
            Qnet[slack_idx] = slack_outputs[:, 3]

            # Unnormalize edge attributes
            mean = pfnet_pfdata_stats[casename]["mean"][("bus", "branch", "bus")]["edge_attr"].to(device)
            std = pfnet_pfdata_stats[casename]["std"][("bus", "branch", "bus")]["edge_attr"].to(device)
            edge_attr = data["bus", "branch","bus"].edge_attr
            edge_attr = (edge_attr * std) + mean

            # Collect edge attributes
            r = edge_attr[:, 0]
            x = edge_attr[:, 1]
            bs = edge_attr[:, 2]
            tau = edge_attr[:, 3]
            theta_shift = edge_attr[:, 4]

            power_balance_model_preds["predictions"] = (V_pred, theta_pred, Pnet, Qnet)
            power_balance_model_preds["edge_attr"] = (r, x, bs, tau, theta_shift)

            return power_balance_model_preds
    
        elif model_name == "GNS": 
            V_pred = data["bus"].v
            theta_pred = data["bus"].theta
            pg = data['bus'].pg
            pd = data['bus'].pd
            qd = data['bus'].qd
            qg = data['bus'].qg
            Pnet = pg - pd
            Qnet = qg - qd

            # Collect edge attributes
            edge_attr = data["bus", "branch","bus"].edge_attr
            r = edge_attr[:, 0]
            x = edge_attr[:, 1]
            bs = edge_attr[:, 3] + edge_attr[:, 5]
            tau = edge_attr[:, 6]
            theta_shift = edge_attr[:, 7]

            power_balance_model_preds["predictions"] = (V_pred, theta_pred, Pnet, Qnet)
            power_balance_model_preds["edge_attr"] = (r, x, bs, tau, theta_shift)

            return power_balance_model_preds


@registry.register_loss("gns_layer_loss")
def GNS_layer_loss(out_dict, data):
    return out_dict["total_layer_loss"]


@registry.register_loss("gns_last_loss")
def GNS_last_loss(out_dict, data):
    return out_dict["last_loss"]


@registry.register_loss("GNSPowerBalanceLoss")
class GNSPowerBalanceLoss:
    """
    Compute the power balance loss.
    """
    def __init__(self):
        self.power_balance_loss = None
    
    def __call__(self, data, layer_loss=False, training=False):
        edge_index = data[('bus', 'branch', 'bus')].edge_index
        edge_attr = data[('bus', 'branch', 'bus')].edge_attr
        src, dst = edge_index

        # Bus values
        v = data['bus'].v
        theta = data['bus'].theta
        b_s = data['bus'].shunt[:, 1]
        g_s = data['bus'].shunt[:, 0]
        pg = data['bus'].pg
        pd = data['bus'].pd
        qd = data['bus'].qd
        qg = data['bus'].qg

        # Edge values
        br_r = edge_attr[:, 0]
        br_x = edge_attr[:, 1]
        b_ij = edge_attr[:, 3] + edge_attr[:, 5]
        shift_ij = edge_attr[:, -1]
        tau_ij = edge_attr[:, -2]
        y = 1 / (torch.complex(br_r, br_x))
        y_ij = torch.abs(y)
        delta_ij = torch.angle(y)

        # Gather per-branch bus features
        v_i = v[src]
        v_j = v[dst]
        theta_i = theta[src]
        theta_j = theta[dst]

        # Active power flows
        P_flow_src = (
            - (v_i * v_j * y_ij / tau_ij) * torch.cos(theta_i - theta_j - delta_ij - shift_ij) 
            + ((v_i / tau_ij) ** 2) * (y_ij * torch.cos(delta_ij))
        ) 

        P_flow_dst = (
            - (v_i * v_j * y_ij / tau_ij) * torch.cos(theta_j - theta_i - delta_ij + shift_ij)
            + ((v_j) ** 2) * (y_ij * torch.cos(delta_ij))
        )

        # Reactive power flows
        Q_flow_src = (
            (-v_i * v_j * y_ij / tau_ij) * torch.sin(theta_i - theta_j - delta_ij - shift_ij)
            - ((v_i / tau_ij) ** 2) * (y_ij * torch.sin(delta_ij) + b_ij / 2) # this is term is fine
        )

        Q_flow_dst = (
            (-v_j * v_i * y_ij / tau_ij) * torch.sin(theta_j - theta_i - delta_ij + shift_ij)
            - ((v_j) ** 2) * (y_ij * torch.sin(delta_ij) + b_ij / 2) # this is term is fine
        ) 

        # Aggregate contributions for all nodes
        Pbus_pred = torch.zeros_like(v).scatter_add_(0, src, P_flow_src)
        Pbus_pred = Pbus_pred.scatter_add_(0, dst, P_flow_dst)
        Qbus_pred = torch.zeros_like(v).scatter_add_(0, src, Q_flow_src)
        Qbus_pred = Qbus_pred.scatter_add_(0, dst, Q_flow_dst)

        if layer_loss: 
            delta_p = (pg - pd - g_s * (v ** 2)) - Pbus_pred
            delta_q = (qg - qd + b_s * (v ** 2)) - Qbus_pred
            delta_s = (delta_p ** 2 + delta_q ** 2).mean()
            if training: 
                self.power_balance_loss = delta_s
            return delta_p, delta_q, delta_s
        else: 
            qg = qd - b_s * v**2 - Qbus_pred
            return qg


@registry.register_loss("pf_constraint_violation")
class constraint_violations_loss_pf:
    def __init__(self, ):
        self.constraint_loss = None
        self.bus_real_mismatch = None
        self.bus_reactive_mismatch = None
        
    def __call__(self, output_dict, data):
        device = data["bus"].x.device
        casename = output_dict["casename"]
        # Get the predictions and edge features
        bus_pred = output_dict["bus"]
        PV_pred = output_dict["PV"]
        slack_pred = output_dict["slack"]
        slack_demand = data["slack"].demand
        PV_generation = data['PV'].generation
        PV_demand = data['PV'].demand
        edge_pred = output_dict["edge_preds"]
        edge_indices = data["bus", "branch", "bus"].edge_index
        va, vm = bus_pred.T

        # # Unnormalize slack predictions 
        # mean = canos_pfdelta_stats[casename]["mean"]["slack"]["y"].to(device)
        # std = canos_pfdelta_stats[casename]["std"]["slack"]["y"].to(device)
        # unnormalized_slack_pred = (slack_pred * std) + mean

        # Get the branch flows from the edge predictions
        n = data["bus"].x.shape[0]
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        flows_rev = edge_pred[:, 2] + 1j * edge_pred[:, 3]  
        flows_fwd = edge_pred[:, 0] + 1j * edge_pred[:, 1]  
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)

        # Generator flows (already aggregated per bus)
        bus_gen = data["bus"].bus_gen.to(device)
        slack_idx = data["slack", "slack_link", "bus"].edge_index[1]
        pv_idx = data["PV", "PV_link", "bus"].edge_index[1]
        slack_pg = slack_pred[:, 0] + slack_demand[:, 0]
        slack_qg = slack_pred[:, 1] +  slack_demand[:, 1]
        pv_pg = PV_generation[:, 0]
        pv_qg = PV_pred[:, 0] + PV_demand[:, 1]
        bus_gen[slack_idx] = torch.stack([slack_pg, slack_qg], dim=1)
        bus_gen[pv_idx] = torch.stack([pv_pg, pv_qg], dim=1)
        gen_flows = bus_gen[:, 0] + 1j * bus_gen[:, 1]

        # Demand flows (already aggregated per bus)
        bus_demand = data["bus"].bus_demand.to(device) 
        demand_flows = bus_demand[:, 0] + 1j * bus_demand[:, 1]

        # Shunt admittances
        bus_shunts = data["bus"].shunt.to(device)  
        shunt_flows = (torch.abs(vm) ** 2) * (bus_shunts[:, 1] + 1j * bus_shunts[:, 0])  # (b_shunt + j*g_shunt)

        power_balance = gen_flows - demand_flows - shunt_flows - sum_branch_flows
        real_power_mismatch = torch.abs(torch.real(power_balance))
        reactive_power_mismatch = torch.abs(torch.imag(power_balance))

        # power: real and imaginary mismatches
        violation_degree_real_mismatch = real_power_mismatch.mean()
        violation_degree_imag_mismatch = reactive_power_mismatch.mean()

        # branch flows: ground truth mismatch, real
        p_flows_true = data["bus", "branch", "bus"].edge_label[:,-2] # this is from bus flow
        p_flows_mismatch = torch.real(flows_fwd) - p_flows_true
        violation_degree_real_flow_mismatch = torch.abs(p_flows_mismatch).mean()

        # branch flows: ground truth mismatch, reactive
        q_flows_true = data["bus", "branch", "bus"].edge_label[:,-1] # this is from bus flow
        q_flows_mismatch = torch.imag(flows_fwd) - q_flows_true
        violation_degree_imag_flow_mismatch = torch.abs(q_flows_mismatch).mean()

        # loss
        loss_c = (violation_degree_real_mismatch + violation_degree_imag_mismatch + 
                  violation_degree_real_flow_mismatch + violation_degree_imag_flow_mismatch)
        
        self.constraint_loss = loss_c
        self.bus_real_mismatch = violation_degree_real_mismatch
        self.bus_reactive_mismatch = violation_degree_imag_mismatch
        self.real_flow_mismatch_violation = violation_degree_real_flow_mismatch
        self.imag_flow_mismatch_violation = violation_degree_imag_flow_mismatch

        return loss_c


@registry.register_loss("canos_pf_mse")
class CANOS_PF_MSE:
    def __init__(self, ):
        self.loss = None
    
    def __call__(self, output_dict, data):
        PV_pred, PQ_pred, slack_pred = output_dict["PV"], output_dict["PQ"], output_dict["slack"]
        edge_preds = output_dict["edge_preds"]

        # gather targets
        PV_target, PQ_target, slack_target = data["PV"].y, data["PQ"].y, data["slack"].y
        branch_target = data["bus", "branch", "bus"].edge_label

        # calculate L2 loss
        pv_loss = mse_loss(PV_pred, PV_target)
        pq_loss = mse_loss(PQ_pred, PQ_target)
        slack_loss = mse_loss(slack_pred, slack_target)
        edge_loss = mse_loss(edge_preds, branch_target)

        total_loss = pv_loss + pq_loss + slack_loss + edge_loss
        self.loss = total_loss

        return total_loss

