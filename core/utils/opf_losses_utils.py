import torch
from torch.nn.functional import relu, mse_loss

from core.utils.registry import registry


def setup_single_branch_features(ac_line_attr, trafo_attr):
# Edge attributes matrix
    mask = torch.ones(9, dtype=torch.bool)
    mask[2] = False
    mask[3] = False
    ac_line_attr_masked = ac_line_attr[:, mask]
    tap_shift = torch.cat([torch.ones((ac_line_attr_masked.shape[0], 1)),
                           torch.zeros((ac_line_attr_masked.shape[0], 1))], dim=-1)
    ac_line_susceptances = ac_line_attr[:, 2:4]
    ac_line_attr = torch.cat([ac_line_attr_masked, tap_shift, ac_line_susceptances], dim=-1)
    edge_attr = torch.cat([ac_line_attr, trafo_attr], dim=0)
    return edge_attr


@registry.register_loss("opf_constraint_violation")
def constraint_violations_loss(output_dict, data):
    # Get the predictions
    bus_pred = output_dict["bus"]
    gen_pred = output_dict["generator"]
    edge_pred = output_dict["edge_preds"]

    # Power balance mismatch
    edge_indices = torch.cat((data["bus", "ac_line", "bus"].edge_index,
                              data["bus", "transformer", "bus"].edge_index), dim=-1)
    edge_features = setup_single_branch_features(data["bus", "ac_line", "bus"].edge_attr, 
                                                 data["bus", "transformer", "bus"].edge_attr)

    va, vm = bus_pred.T
    complex_voltage = vm * torch.exp(1j* va)

    n = data["bus"].x.shape[0]
    sum_branch_flows = torch.zeros(n, dtype=torch.cfloat)
    flows_rev = edge_pred[:,0] + 1j*edge_pred[:,1]
    flows_fwd = edge_pred[:,2] + 1j*edge_pred[:,3]
    sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
    sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)

    pg = gen_pred[:, 0]
    qg = gen_pred[:, 1]

    gen_flows = torch.zeros(n, dtype=torch.cfloat).scatter_add_(0, data["bus", "generator_link", "generator"].edge_index[0], pg + 1j*qg)
    demand_flows = torch.zeros(n, dtype=torch.cfloat).scatter_add_(0, data["bus", "load_link", "load"].edge_index[0], data["load"].x[:, 0] + 1j*data["load"].x[:, 1])
    shunt_flows = torch.abs(vm)**2 * torch.zeros(n, dtype=torch.cfloat).scatter_add_(0, data["bus", "shunt_link", "shunt"].edge_index[0], data["shunt"].x[:, 1] + 1j*data["shunt"].x[:, 0]).conj()

    power_balance = gen_flows - demand_flows - shunt_flows - sum_branch_flows
    real_power_mismatch = torch.abs(torch.real(power_balance))
    reactive_power_mismatch = torch.abs(torch.imag(power_balance))
    violation_degree_real_mismatch = real_power_mismatch.mean()
    violation_degree_imag_mismatch = reactive_power_mismatch.mean()

    # Voltage bounds mismatch
    vmin, vmax = data["bus"].x[:, -2:].T

    voltage_left = relu(vmin - vm)
    voltage_right = relu(vm - vmax)
    violation_degree_voltages = (voltage_left + voltage_right).mean()

    # angle difference bounds
    angmin, angmax = edge_features[:, :2].T
    i = edge_indices[0]
    j = edge_indices[1]
    angle_left = relu((va[i] - va[j]) - angmax)
    angle_right = relu((va[j] - va[i]) + angmin)
    violation_degree_angles = (angle_left + angle_right).mean()

    # generation bounds
    pmin, pmax = data["generator"].x[:, 2:4].T
    real_power_right = relu(pmin - pg)
    real_power_left = relu(pg - pmax)
    violation_degree_pg = (real_power_right + real_power_left).mean()

    qmin, qmax = data["generator"].x[:, 5:7].T

    reactive_power_right = relu(qmin - qg)
    reactive_power_left = relu(qg - qmax)
    violation_degree_qg = (reactive_power_right + reactive_power_left).mean()

    # branch flow bounds
    sf = torch.abs(flows_fwd)
    st = torch.abs(flows_rev)
    smax = edge_features[:, 4]
    flow_mismatch_fwd = relu(torch.abs(sf)**2 - smax**2)
    flow_mismatch_rev = relu(torch.abs(st)**2 - smax**2)
    violation_degree_flows = torch.cat([flow_mismatch_fwd, flow_mismatch_rev]).mean()

    # loss
    loss_c = (violation_degree_real_mismatch +  violation_degree_imag_mismatch +
                  violation_degree_voltages + violation_degree_angles +
                    violation_degree_pg + violation_degree_qg + violation_degree_flows)

    return loss_c


@registry.register_loss("canos_mse")
def CANOSMSE(output_dict, data):
    # Gather predictions
    bus_pred, gen_pred = output_dict["bus"], output_dict["generator"]
    edge_preds = output_dict["edge_preds"]

    # Gather targets
    bus_target, gen_target = data["bus"].y, data["generator"].y
    ac_line_target = data["bus", "ac_line", "bus"].edge_label
    transformer_line_target = data["bus", "transformer", "bus"].edge_label    
    edge_target = torch.cat([ac_line_target, transformer_line_target], dim=0)
    
    # Calculate L2 loss
    bus_loss = mse_loss(bus_pred, bus_target)
    gen_loss = mse_loss(gen_pred, gen_target)
    edge_loss = mse_loss(edge_preds, edge_target)

    total_loss = bus_loss + gen_loss + edge_loss

    return total_loss
