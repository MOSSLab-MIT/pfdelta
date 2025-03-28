import os
import pickle

import torch

from core.utils.registry import registry


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

