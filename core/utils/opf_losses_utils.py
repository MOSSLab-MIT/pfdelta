import torch
from torch.nn.functional import relu, mse_loss

from core.utils.registry import registry


def setup_single_branch_features(ac_line_attr, trafo_attr):
    device = ac_line_attr.device
    # Edge attributes matrix
    mask = torch.ones(9, dtype=torch.bool)
    mask[2] = False
    mask[3] = False
    ac_line_attr_masked = ac_line_attr[:, mask]
    tap_shift = torch.cat(
        [
            torch.ones((ac_line_attr_masked.shape[0], 1)),
            torch.zeros((ac_line_attr_masked.shape[0], 1)),
        ],
        dim=-1,
    ).to(device)
    ac_line_susceptances = ac_line_attr[:, 2:4]
    ac_line_attr = torch.cat(
        [ac_line_attr_masked, tap_shift, ac_line_susceptances], dim=-1
    )
    edge_attr = torch.cat([ac_line_attr, trafo_attr], dim=0)
    return edge_attr


@registry.register_loss("opf_constraint_violation")
class constraint_violations_loss:
    def __init__(
        self,
    ):
        self.constraint_loss = None
        self.bus_real_mismatch = None
        self.bus_reactive_mismatch = None
        self.angle_difference_violations = None
        self.branch_flow_violations = None
        self.voltage_violations = None
        self.generator_pg_violations = None
        self.generator_qg_violations = None

    def __call__(self, output_dict, data):
        device = data["x"].device
        # Get the predictions
        bus_pred = output_dict["bus"]
        gen_pred = output_dict["generator"]
        edge_pred = output_dict["edge_preds"]

        # Power balance mismatch
        edge_indices = torch.cat(
            (
                data["bus", "ac_line", "bus"].edge_index,
                data["bus", "transformer", "bus"].edge_index,
            ),
            dim=-1,
        )
        edge_features = setup_single_branch_features(
            data["bus", "ac_line", "bus"].branch_vals,
            data["bus", "transformer", "bus"].branch_vals,
        )

        va, vm = bus_pred.T
        complex_voltage = vm * torch.exp(1j * va)

        n = data["bus"].x.shape[0]
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        flows_rev = edge_pred[:, 0] + 1j * edge_pred[:, 1]
        flows_fwd = edge_pred[:, 2] + 1j * edge_pred[:, 3]
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)

        pg = gen_pred[:, 0]
        qg = gen_pred[:, 1]

        gen_flows = torch.zeros(n, dtype=torch.cfloat, device=device).scatter_add_(
            0, data["bus", "generator_link", "generator"].edge_index[0], pg + 1j * qg
        )
        demand_flows = torch.zeros(n, dtype=torch.cfloat, device=device).scatter_add_(
            0,
            data["bus", "load_link", "load"].edge_index[0],
            data["load"]["unnormalized"][:, 0]
            + 1j * data["load"]["unnormalized"][:, 1],
        )
        shunt_flows = (
            torch.abs(vm) ** 2
            * torch.zeros(n, dtype=torch.cfloat, device=device)
            .scatter_add_(
                0,
                data["bus", "shunt_link", "shunt"].edge_index[0],
                data["shunt"]["unnormalized"][:, 1]
                + 1j * data["shunt"]["unnormalized"][:, 0],
            )
            .conj()
        )

        power_balance = gen_flows - demand_flows - shunt_flows - sum_branch_flows
        real_power_mismatch = torch.abs(torch.real(power_balance))
        reactive_power_mismatch = torch.abs(torch.imag(power_balance))
        violation_degree_real_mismatch = real_power_mismatch.mean()
        violation_degree_imag_mismatch = reactive_power_mismatch.mean()

        # Voltage bounds mismatch
        vmin, vmax = data["bus"]["v_lims"].T
        # vmin, vmax = data["bus"].x[:, -2:].T

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
        pmin, pmax = data["generator"]["p_lims"].T
        # pmin, pmax = data["generator"].x[:, 2:4].T
        real_power_right = relu(pmin - pg)
        real_power_left = relu(pg - pmax)
        violation_degree_pg = (real_power_right + real_power_left).mean()

        qmin, qmax = data["generator"]["q_lims"].T

        reactive_power_right = relu(qmin - qg)
        reactive_power_left = relu(qg - qmax)
        violation_degree_qg = (reactive_power_right + reactive_power_left).mean()

        # branch flow bounds
        sf = torch.abs(flows_fwd)
        st = torch.abs(flows_rev)
        smax = edge_features[:, 4]
        flow_mismatch_fwd = relu(torch.abs(sf) ** 2 - smax**2)
        flow_mismatch_rev = relu(torch.abs(st) ** 2 - smax**2)
        # violation_degree_flows = torch.cat([flow_mismatch_fwd, flow_mismatch_rev]).mean()
        violation_degree_flow_f = flow_mismatch_fwd.mean()
        violation_degree_flow_r = flow_mismatch_rev.mean()

        # branch flows: ground truth mismatch, real
        p_flows_true_ac = data["bus", "ac_line", "bus"].edge_label[
            :, -2
        ]  # this is from bus flow
        p_flows_true_tr = data["bus", "transformer", "bus"].edge_label[
            :, -2
        ]  # this is from bus flow
        p_flows_true = torch.cat([p_flows_true_ac, p_flows_true_tr])
        p_flows_mismatch = torch.real(flows_fwd) - p_flows_true
        violation_degree_real_flow_mismatch = torch.abs(p_flows_mismatch).mean()

        # branch flows: ground truth mismatch, reactive
        q_flows_true_ac = data["bus", "ac_line", "bus"].edge_label[
            :, -1
        ]  # this is from bus flow
        q_flows_true_tr = data["bus", "transformer", "bus"].edge_label[:, -1]
        q_flows_true = torch.cat([q_flows_true_ac, q_flows_true_tr])
        q_flows_mismatch = torch.imag(flows_fwd) - q_flows_true
        violation_degree_imag_flow_mismatch = torch.abs(q_flows_mismatch).mean()

        # loss
        loss_c = (
            violation_degree_real_mismatch
            + violation_degree_imag_mismatch
            + violation_degree_voltages
            + violation_degree_angles
            + violation_degree_real_flow_mismatch
            + violation_degree_imag_flow_mismatch
            + violation_degree_pg
            + violation_degree_qg
            + violation_degree_flow_f
            + violation_degree_flow_r
        )

        self.constraint_loss = loss_c
        self.bus_real_mismatch = violation_degree_real_mismatch
        self.bus_reactive_mismatch = violation_degree_imag_mismatch
        self.voltage_violations = violation_degree_voltages
        self.angle_difference_violations = violation_degree_angles
        self.real_flow_mismatch_violation = violation_degree_real_flow_mismatch
        self.imag_flow_mismatch_violation = violation_degree_imag_flow_mismatch
        self.generator_pg_violations = violation_degree_pg
        self.generator_qg_violations = violation_degree_qg
        self.branch_flow_violation_f = violation_degree_flow_f
        self.branch_flow_violation_r = violation_degree_flow_r

        return loss_c


@registry.register_loss("recycle_loss")
class RecycleLoss:
    def __init__(self, recycled_parameter, loss_name, keyword):
        self.recycled_parameter = recycled_parameter
        self.loss_name = loss_name
        self.source = None
        self.keyword = keyword

    def __call__(self, output_dict, data):
        recycled_value = getattr(self.source, self.recycled_parameter)
        return recycled_value


@registry.register_loss("canos_mse")
class CANOSMSE:
    def __init__(
        self,
    ):
        self.va = None
        self.vm = None
        self.pg = None
        self.qg = None
        self.pt_ac = None
        self.qt_ac = None
        self.pf_ac = None
        self.qf_ac = None
        self.pt_transformer = None
        self.qf_transformer = None
        self.pf_transformer = None
        self.qt_transformer = None
        self.total_loss = None

    def bus_error(self, pred, target):
        first = mse_loss(pred[:, 0], target[:, 0])
        second = mse_loss(pred[:, 1], target[:, 1])
        total_error = torch.stack([first, second]).mean()
        return first, second, total_error

    def voltage_error(self, bus_pred, bus_target):
        va, vm, v_error = self.bus_error(bus_pred, bus_target)
        self.va, self.vm = va, vm
        return v_error

    def gen_error(self, gen_pred, gen_target):
        pg, qg, g_error = self.bus_error(gen_pred, gen_target)
        self.pg, self.qg = pg, qg
        return g_error

    def line_errors(self, pred, target):
        pt = mse_loss(pred[:, 0], target[:, 0])
        qt = mse_loss(pred[:, 1], target[:, 1])
        pf = mse_loss(pred[:, 2], target[:, 2])
        qf = mse_loss(pred[:, 3], target[:, 3])
        line_error = torch.stack([pt, qt, pf, qf]).mean()
        return pt, qt, pf, qf, line_error

    def ac_line_error(self, pred, target):
        pt, qt, pf, qf, line_error = self.line_errors(pred, target)
        self.pt_ac, self.qt_ac, self.pf_ac, self.qf_ac = pt, qt, pf, qf
        return line_error

    def transformer_line_error(self, pred, target):
        pt, qt, pf, qf, line_error = self.line_errors(pred, target)
        self.pt_transformer, self.qt_transformer = pt, qt
        self.pf_transformer, self.qf_transformer = pf, qf
        return line_error

    def __call__(self, output_dict, data):
        # Gather targets
        bus_target, gen_target = data["bus"].y, data["generator"].y
        ac_line_target = data["bus", "ac_line", "bus"].edge_label
        transformer_line_target = data["bus", "transformer", "bus"].edge_label
        num_ac_lines = ac_line_target.size(0)

        # Gather predictions
        bus_pred, gen_pred = output_dict["bus"], output_dict["generator"]
        edge_preds = output_dict["edge_preds"]
        ac_line_pred = edge_preds[:num_ac_lines]
        transformer_line_pred = edge_preds[num_ac_lines:]

        # Calculate L2 losses
        bus_loss = self.voltage_error(bus_pred, bus_target)
        gen_loss = self.gen_error(gen_pred, gen_target)
        ac_loss = self.ac_line_error(ac_line_pred, ac_line_target)
        transformer_loss = self.transformer_line_error(
            transformer_line_pred, transformer_line_target
        )

        # Normalize to average
        n_bus, n_gen = bus_pred.size(0), gen_pred.size(0)
        n_ac, n_transformer = ac_line_pred.size(0), transformer_line_pred.size(0)
        total_loss = bus_loss + gen_loss + ac_loss + transformer_loss

        self.total_loss = total_loss

        return total_loss
