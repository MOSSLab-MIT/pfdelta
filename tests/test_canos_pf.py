"""
Tests for CANOS_PF.derive_branch_flows and derive_slack_output.

Integration tests load samples from PFDeltaDataset (case14 and case118), with
and without topological perturbations (N-1, N-2) and near-infeasible cases,
feed the ground-truth bus voltages through the function, and compare the
resulting branch flows against the ground-truth edge_label values.

A synthetic test exercises the phase-shift angle parameter, which is zero for
all branches in the real datasets and therefore cannot be validated through the
integration tests alone.

PFDelta edge_attr column layout for ('bus', 'branch', 'bus'):
  col 0: br_r    col 1: br_x
  col 2: g_fr    col 3: b_fr
  col 4: g_to    col 5: b_to
  col 6: tap     col 7: shift

edge_label column layout:
  col 0: p_fr    col 1: q_fr    col 2: p_to    col 3: q_to
"""

import math

import pytest
import torch
from torch_geometric.data import HeteroData

from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.datasets.pfdelta_variants import PFDeltaCANOS
from core.models.canos_pf import CANOS_PF

ATOL = 1e-4
N_SAMPLES = 50
SAMPLE_IDS = list(range(N_SAMPLES))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive(data):
    output_dict = {"bus": data["bus"].bus_voltages}
    return CANOS_PF.derive_branch_flows(None, output_dict, data)


def _gt(data, col):
    return data["bus", "branch", "bus"].edge_label[:, col]


def _kcl_output_dict(data):
    p_fr, q_fr, p_to, q_to = _derive(data)
    return {
        "bus": data["bus"].bus_voltages,
        "edge_preds": torch.stack([p_fr, q_fr, p_to, q_to], dim=-1),
    }


def _kcl_all_buses(data, output_dict):
    """KCL residual at every bus via derive_slack_output with all-bus mask."""
    n = data["bus"].x.shape[0]
    orig_edge_index = data["slack", "slack_link", "bus"].edge_index
    data["slack", "slack_link", "bus"].edge_index = torch.stack(
        [
            torch.zeros(n, dtype=torch.long),
            torch.arange(n),
        ]
    )
    p_net, q_net = CANOS_PF.derive_slack_output(None, output_dict, data)
    data["slack", "slack_link", "bus"].edge_index = orig_edge_index

    net_gen = data["bus"].bus_gen[:, 0] + 1j * data["bus"].bus_gen[:, 1]
    net_demand = data["bus"].bus_demand[:, 0] + 1j * data["bus"].bus_demand[:, 1]
    residual = net_gen - net_demand - (p_net + 1j * q_net)
    return residual.real, residual.imag


# ---------------------------------------------------------------------------
# Fixtures (return datasets, loaded once per session)
# ---------------------------------------------------------------------------


def _pfdelta(case_name, perturbation="n", feasibility_type="feasible"):
    return PFDeltaDataset(
        root_dir="data",
        case_name=case_name,
        task="analysis",
        feasibility_type=feasibility_type,
        n_samples=100,
        add_bus_type=True,
        perturbation=perturbation,
    )


@pytest.fixture(scope="session")
def case14():
    return _pfdelta("case14")


@pytest.fixture(scope="session")
def case118():
    return _pfdelta("case118")


@pytest.fixture(scope="session")
def case14_n1():
    return _pfdelta("case14", perturbation="n-1")


@pytest.fixture(scope="session")
def case118_n1():
    return _pfdelta("case118", perturbation="n-1")


@pytest.fixture(scope="session")
def case14_n2():
    return _pfdelta("case14", perturbation="n-2")


@pytest.fixture(scope="session")
def case118_n2():
    return _pfdelta("case118", perturbation="n-2")


@pytest.fixture(scope="session")
def case14_ni():
    return _pfdelta("case14", feasibility_type="near infeasible")


@pytest.fixture(scope="session")
def case118_ni():
    return _pfdelta("case118", feasibility_type="near infeasible")


@pytest.fixture(scope="session")
def case14_ni_n1():
    return _pfdelta("case14", perturbation="n-1", feasibility_type="near infeasible")


@pytest.fixture(scope="session")
def case118_ni_n1():
    return _pfdelta("case118", perturbation="n-1", feasibility_type="near infeasible")


@pytest.fixture(scope="session")
def case14_ni_n2():
    return _pfdelta("case14", perturbation="n-2", feasibility_type="near infeasible")


@pytest.fixture(scope="session")
def case118_ni_n2():
    return _pfdelta("case118", perturbation="n-2", feasibility_type="near infeasible")


@pytest.fixture(scope="session")
def case14_canos():
    return PFDeltaCANOS(
        root_dir="data", case_name="case14", task="analysis", add_bus_type=True
    )


@pytest.fixture(scope="session")
def case118_canos():
    return PFDeltaCANOS(
        root_dir="data", case_name="case118", task="analysis", add_bus_type=True
    )


# ---------------------------------------------------------------------------
# Integration tests: ground-truth voltages → ground-truth flows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_fixture",
    [
        "case14",
        "case118",
        "case14_n1",
        "case118_n1",
        "case14_n2",
        "case118_n2",
        "case14_ni",
        "case118_ni",
        "case14_ni_n1",
        "case118_ni_n1",
        "case14_ni_n2",
        "case118_ni_n2",
    ],
)
@pytest.mark.parametrize("sample_idx", SAMPLE_IDS)
class TestDeriveBranchFlowsPF:
    def test_p_fr(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        p_fr, _, _, _ = _derive(data)
        gt = _gt(data, 0)
        assert torch.allclose(p_fr, gt, atol=ATOL), (
            f"p_fr mismatch: max error = {(p_fr - gt).abs().max():.2e}"
        )

    def test_q_fr(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        _, q_fr, _, _ = _derive(data)
        gt = _gt(data, 1)
        assert torch.allclose(q_fr, gt, atol=ATOL), (
            f"q_fr mismatch: max error = {(q_fr - gt).abs().max():.2e}"
        )

    def test_p_to(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        _, _, p_to, _ = _derive(data)
        gt = _gt(data, 2)
        assert torch.allclose(p_to, gt, atol=ATOL), (
            f"p_to mismatch: max error = {(p_to - gt).abs().max():.2e}"
        )

    def test_q_to(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        _, _, _, q_to = _derive(data)
        gt = _gt(data, 3)
        assert torch.allclose(q_to, gt, atol=ATOL), (
            f"q_to mismatch: max error = {(q_to - gt).abs().max():.2e}"
        )


# ---------------------------------------------------------------------------
# Integration tests: ground-truth voltages → ground-truth slack / KCL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_fixture", ["case14_canos", "case118_canos"])
@pytest.mark.parametrize("sample_idx", SAMPLE_IDS)
class TestDeriveSlackOutput:
    def test_p_slack(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        output_dict = _kcl_output_dict(data)
        p_slack, _ = CANOS_PF.derive_slack_output(None, output_dict, data)
        gt = data["slack"].y[:, 0]
        assert torch.allclose(p_slack, gt, atol=ATOL), (
            f"p_slack mismatch: max error = {(p_slack - gt).abs().max():.2e}"
        )

    def test_q_slack(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        output_dict = _kcl_output_dict(data)
        _, q_slack = CANOS_PF.derive_slack_output(None, output_dict, data)
        gt = data["slack"].y[:, 1]
        assert torch.allclose(q_slack, gt, atol=ATOL), (
            f"q_slack mismatch: max error = {(q_slack - gt).abs().max():.2e}"
        )

    def test_kcl_real_all_buses(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        output_dict = _kcl_output_dict(data)
        delta_p, _ = _kcl_all_buses(data, output_dict)
        assert delta_p.abs().max() < ATOL, (
            f"KCL real residual max = {delta_p.abs().max():.2e}"
        )

    def test_kcl_reactive_all_buses(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        output_dict = _kcl_output_dict(data)
        _, delta_q = _kcl_all_buses(data, output_dict)
        assert delta_q.abs().max() < ATOL, (
            f"KCL reactive residual max = {delta_q.abs().max():.2e}"
        )


# ---------------------------------------------------------------------------
# Synthetic test: phase shift angle
# ---------------------------------------------------------------------------


def _make_shift_data(br_x, tap, shift, va_j, vm_j):
    data = HeteroData()
    edge_attr = torch.zeros(1, 8)
    edge_attr[0, 1] = br_x
    edge_attr[0, 6] = tap
    edge_attr[0, 7] = shift
    data["bus", "branch", "bus"].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    data["bus", "branch", "bus"].edge_attr = edge_attr
    output_dict = {"bus": torch.tensor([[0.0, 1.0], [va_j, vm_j]])}
    return data, output_dict


@pytest.mark.parametrize("shift", [math.pi / 6, -math.pi / 4, math.pi / 3])
def test_phase_shift_p_fr(shift):
    br_x, va_j, vm_j = 0.1, 0.2, 1.0
    data, output_dict = _make_shift_data(
        br_x=br_x, tap=1.0, shift=shift, va_j=va_j, vm_j=vm_j
    )
    p_fr, _, _, _ = CANOS_PF.derive_branch_flows(None, output_dict, data)
    expected = -vm_j * math.sin(va_j + shift) / br_x
    assert torch.allclose(p_fr, torch.tensor([expected]), atol=1e-5), (
        f"shift={shift:.4f}: got {p_fr.item():.6f}, expected {expected:.6f}"
    )
