import os
import pickle
import torch

from torch.nn.functional import mse_loss

from core.utils.registry import registry
from core.datasets.dataset_utils import canos_pf_data_mean0_var1
from core.datasets.data_stats import pfdata_stats


def PowerBalanceLoss(predictions, data):
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
    bus_type = data["bus"].bus_type
    edge_index = data["bus", "forward","bus"].edge_index
    edge_feat = data["bus", "forward","bus"].edge_feat
    shunt_g = data["bus"].shunt_g
    shunt_b = data["bus"].shunt_b
    # Unpack node feature values (pred and target)
    V_pred, theta_pred, Pnet, Qnet = predictions.T # Pnet and Qnet should correspond to Pg - Pd for each bus.

    # branch attributes (resistance, reactance, charging_susceptance, tap ratio magnitude, shift angle)
    r, x, bs, tau, theta_shift = edge_feat.T[:5]

    # Compute bus-level power injections based on predictions
    Pbus_pred = torch.zeros_like(V_pred)
    Qbus_pred = torch.zeros_like(V_pred)
    src, dst = edge_index

    Y_real = torch.real(1/(r + 1j*x))
    Y_imag = torch.imag(1/(r + 1j*x))
    suscept = bs
    delta_theta1 = theta_pred[src] - theta_pred[dst] 
    delta_theta2 = theta_pred[dst] - theta_pred[src] 

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
    delta_PQ_magnitude = torch.sqrt(delta_P**2 + delta_Q**2)
    return delta_PQ_magnitude, delta_P, delta_Q


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
        b_ij = edge_attr[:, 3]
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
            (v_i * v_j * y_ij / tau_ij) * torch.sin(theta_i - theta_j - delta_ij - shift_ij) 
            + ((v_i / tau_ij) ** 2) * (y_ij * torch.sin(delta_ij))
        ) 

        P_flow_dst = (
            (v_j * v_i * y_ij / tau_ij) * torch.sin(theta_j - theta_i - delta_ij + shift_ij)
            + ((v_j) ** 2) * (y_ij * torch.sin(delta_ij))
        )

        # Reactive power flows
        Q_flow_src = (
            (-v_i * v_j * y_ij / tau_ij) * torch.cos(theta_i - theta_j - delta_ij - shift_ij)
            + ((v_i / tau_ij) ** 2) * (y_ij * torch.cos(delta_ij) - b_ij / 2)
        )

        Q_flow_dst = (
            (-v_j * v_i * y_ij / tau_ij) * torch.cos(theta_j - theta_i - delta_ij + shift_ij)
            + ((v_j) ** 2) * (y_ij * torch.sin(delta_ij) - b_ij / 2)
        ) 

        # Aggregate contributions for all nodes
        Pbus_pred = torch.zeros_like(v).scatter_add_(0, src, P_flow_src)
        Pbus_pred = Pbus_pred.scatter_add_(0, dst, P_flow_dst)
        Qbus_pred = torch.zeros_like(v).scatter_add_(0, src, Q_flow_src)
        Qbus_pred = Qbus_pred.scatter_add_(0, dst, Q_flow_dst)

        if layer_loss: 
            delta_p = (pg - pd - g_s * (v ** 2)) + Pbus_pred
            delta_q = qg - qd + b_s * (v ** 2) + Qbus_pred
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
        mean = pfdata_stats[output_dict["casename"]]["mean"]
        std = pfdata_stats[output_dict["casename"]]["std"]
        device = data["bus"].x.device
        # Get the predictions and edge features
        bus_pred = output_dict["bus"]
        edge_pred = output_dict["edge_preds"]
        edge_indices = data["bus", "branch", "bus"].edge_index
        va, vm = bus_pred.T
        complex_voltage = vm * torch.exp(1j* va)

        # Get the branch flows from the edge predictions
        n = data["bus"].x.shape[0]
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        flows_rev = edge_pred[:, 0] + 1j * edge_pred[:, 1]  
        flows_fwd = edge_pred[:, 2] + 1j * edge_pred[:, 3]  
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)

        # Generator flows (already aggregated per bus)
        bus_gen = data["bus"].bus_gen.to(device) # TODO: does this use the slack gen? DATA LEAKAGE? 
        # slack y is pg - pd -- we need to add pd to those values before doing the constraints
        # unnormalize right before we use here
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

