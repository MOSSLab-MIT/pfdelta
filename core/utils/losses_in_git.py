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
        self.model = model

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
        delta_PQ_magnitude = torch.sqrt(delta_P**2 + delta_Q**2)

        # Keep track of losses
        self.power_balance_mean = delta_PQ_magnitude.mean()
        new_max = delta_PQ_magnitude.max()
        if self.power_balance_max is None:
            self.power_balance_max = new_max
        else:
            self.power_balance_max = delta_PQ_magnitude.max() # TODO: fix this

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
            Pnet[pq_idx] = pq_pg - pq_outputs[:, 2]
            Qnet[pq_idx] = pq_qg - pq_outputs[:, 3]

            # PV
            pv_idx = (data['bus'].bus_type == 2).nonzero(as_tuple=True)[0]
            pv_outputs = output[pv_idx]
            Pnet[pv_idx] = pv_outputs[:, 2]
            Qnet[pv_idx] = pv_outputs[:, 3]

            # Slack
            slack_idx = (data['bus'].bus_type == 3).nonzero(as_tuple=True)[0]
            slack_outputs = output[slack_idx]
            Pnet[slack_idx] = slack_outputs[:, 2]
            Qnet[slack_idx] = slack_outputs[:, 3]

            # Collect edge attributes 
            edge_attr = data["bus", "branch","bus"].edge_attr
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