"""
Tests for constraint_violations_loss_pf and PowerBalanceLoss in
core/utils/pf_losses_utils.py.

The key property: when ground-truth predictions are fed into the loss, each
component should be zero (or near-zero within floating-point precision):

  - Flow mismatch:    derived flows must match stored edge_label exactly
  - Power balance:    KCL must be satisfied at every bus

Tests run over N_SAMPLES samples per dataset to avoid dependence on a single
data point.

Ground truth output_dict assembled as:
  output_dict["bus"]       = data["bus"].bus_voltages   [va, vm]
  output_dict["PV"]        = data["PV"].y               [net_qg, va]
  output_dict["slack"]     = data["slack"].y             [net_pg, net_qg]
  output_dict["edge_preds"]= data["bus","branch","bus"].edge_label  [p_fr, q_fr, p_to, q_to]
  output_dict["casename"]  = dataset.case_name

Note: constraint_violations_loss_pf mutates data["bus"].bus_gen in-place,
but with ground-truth predictions the mutation is idempotent.
"""

import pytest

from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.datasets.pfdelta_variants import PFDeltaCANOS
from core.utils.pf_losses_utils import PowerBalanceLoss, constraint_violations_loss_pf

ATOL = 1e-4
N_SAMPLES = 50
SAMPLE_IDS = list(range(N_SAMPLES))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gt_output_dict(data, case_name):
    return {
        "bus": data["bus"].bus_voltages,
        "PV": data["PV"].y,
        "slack": data["slack"].y,
        "edge_preds": data["bus", "branch", "bus"].edge_label,
        "casename": case_name,
    }


def _pbl_inputs(data):
    vm = data["bus"].bus_voltages[:, 1]
    va = data["bus"].bus_voltages[:, 0]
    Pnet = data["bus"].bus_gen[:, 0] - data["bus"].bus_demand[:, 0]
    Qnet = data["bus"].bus_gen[:, 1] - data["bus"].bus_demand[:, 1]

    edge_attr = data["bus", "branch", "bus"].edge_attr
    r = edge_attr[:, 0]
    x = edge_attr[:, 1]
    bs = edge_attr[:, 3] + edge_attr[:, 5]  # b_fr + b_to
    tau = edge_attr[:, 6]
    theta_shift = edge_attr[:, 7]

    shunt_g = data["bus"].shunt[:, 0]
    shunt_b = data["bus"].shunt[:, 1]
    src, dst = data["bus", "branch", "bus"].edge_index

    return vm, va, Pnet, Qnet, r, x, bs, tau, theta_shift, shunt_g, shunt_b, src, dst


# ---------------------------------------------------------------------------
# Dataset-level fixtures (loaded once per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def case14_canos():
    ds = PFDeltaCANOS(
        root_dir="data", case_name="case14", task="analysis", add_bus_type=True
    )
    return ds, ds.case_name


@pytest.fixture(scope="session")
def case118_canos():
    ds = PFDeltaCANOS(
        root_dir="data", case_name="case118", task="analysis", add_bus_type=True
    )
    return ds, ds.case_name


@pytest.fixture(scope="session")
def case14_pfdelta():
    return PFDeltaDataset(
        root_dir="data",
        case_name="case14",
        task="analysis",
        feasibility_type="feasible",
        n_samples=100,
        add_bus_type=True,
    )


@pytest.fixture(scope="session")
def case118_pfdelta():
    return PFDeltaDataset(
        root_dir="data",
        case_name="case118",
        task="analysis",
        feasibility_type="feasible",
        n_samples=100,
        add_bus_type=True,
    )


# ---------------------------------------------------------------------------
# Integration tests: ground-truth predictions → zero violations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_fixture", ["case14_canos", "case118_canos"])
@pytest.mark.parametrize("sample_idx", SAMPLE_IDS)
class TestConstraintViolationsPFGroundTruth:
    def test_flow_mismatch_real(self, data_fixture, sample_idx, request):
        ds, case_name = request.getfixturevalue(data_fixture)
        loss = constraint_violations_loss_pf()
        loss(_gt_output_dict(ds[sample_idx], case_name), ds[sample_idx])
        assert loss.real_flow_mismatch_violation < ATOL, (
            f"real flow mismatch = {loss.real_flow_mismatch_violation:.2e}"
        )

    def test_flow_mismatch_reactive(self, data_fixture, sample_idx, request):
        ds, case_name = request.getfixturevalue(data_fixture)
        loss = constraint_violations_loss_pf()
        loss(_gt_output_dict(ds[sample_idx], case_name), ds[sample_idx])
        assert loss.imag_flow_mismatch_violation < ATOL, (
            f"reactive flow mismatch = {loss.imag_flow_mismatch_violation:.2e}"
        )

    def test_power_balance_real(self, data_fixture, sample_idx, request):
        ds, case_name = request.getfixturevalue(data_fixture)
        loss = constraint_violations_loss_pf()
        loss(_gt_output_dict(ds[sample_idx], case_name), ds[sample_idx])
        assert loss.bus_real_mismatch < ATOL, (
            f"real power balance mismatch = {loss.bus_real_mismatch:.2e}"
        )

    def test_power_balance_reactive(self, data_fixture, sample_idx, request):
        ds, case_name = request.getfixturevalue(data_fixture)
        loss = constraint_violations_loss_pf()
        loss(_gt_output_dict(ds[sample_idx], case_name), ds[sample_idx])
        assert loss.bus_reactive_mismatch < ATOL, (
            f"reactive power balance mismatch = {loss.bus_reactive_mismatch:.2e}"
        )


# ---------------------------------------------------------------------------
# Tests for PowerBalanceLoss.calculate_PBL (static method)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_fixture", ["case14_pfdelta", "case118_pfdelta"])
@pytest.mark.parametrize("sample_idx", SAMPLE_IDS)
class TestCalculatePBLPFDelta:
    def test_delta_P_near_zero(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        vm, va, Pnet, Qnet, r, x, bs, tau, theta_shift, shunt_g, shunt_b, src, dst = (
            _pbl_inputs(data)
        )
        delta_P, _ = PowerBalanceLoss.calculate_PBL(
            vm, va, Pnet, Qnet, r, x, bs, tau, theta_shift, shunt_g, shunt_b, src, dst
        )
        assert delta_P.abs().max() < ATOL, f"delta_P max = {delta_P.abs().max():.2e}"

    def test_delta_Q_near_zero(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        vm, va, Pnet, Qnet, r, x, bs, tau, theta_shift, shunt_g, shunt_b, src, dst = (
            _pbl_inputs(data)
        )
        _, delta_Q = PowerBalanceLoss.calculate_PBL(
            vm, va, Pnet, Qnet, r, x, bs, tau, theta_shift, shunt_g, shunt_b, src, dst
        )
        assert delta_Q.abs().max() < ATOL, f"delta_Q max = {delta_Q.abs().max():.2e}"
