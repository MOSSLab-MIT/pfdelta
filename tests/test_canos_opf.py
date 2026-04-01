"""
Tests for CANOS_OPF.derive_branch_flows.

Integration tests load samples from OPFDataset (case14 and case118), with and
without topological perturbations (N-1 contingencies), feed the ground-truth
bus voltages through the function, and compare the resulting branch flows
against the ground-truth edge_label values stored in the dataset.
This verifies the π-model equations are implemented correctly, including on
graphs with a reduced number of branches.

A synthetic test exercises the phase-shift angle parameter, which is zero for
all branches in the real datasets and therefore cannot be validated through the
integration tests alone.
"""

import math

import pytest
import torch
from torch_geometric.datasets import OPFDataset

from core.models.canos_opf import CANOS_OPF

ATOL = 1e-4
N_SAMPLES = 50
SAMPLE_IDS = list(range(N_SAMPLES))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive(data):
    data["bus", "ac_line", "bus"].branch_vals = data["bus", "ac_line", "bus"].edge_attr
    data["bus", "transformer", "bus"].branch_vals = data["bus", "transformer", "bus"].edge_attr
    output_dict = {"bus": data["bus"].y}
    return CANOS_OPF.derive_branch_flows(None, output_dict, data)


def _gt_p_fr(data):
    ac = data["bus", "ac_line", "bus"].edge_label[:, -2]
    tr = data["bus", "transformer", "bus"].edge_label[:, -2]
    return torch.cat([ac, tr])


def _gt_q_fr(data):
    ac = data["bus", "ac_line", "bus"].edge_label[:, -1]
    tr = data["bus", "transformer", "bus"].edge_label[:, -1]
    return torch.cat([ac, tr])


def _gt_p_to(data):
    ac = data["bus", "ac_line", "bus"].edge_label[:, 0]
    tr = data["bus", "transformer", "bus"].edge_label[:, 0]
    return torch.cat([ac, tr])


def _gt_q_to(data):
    ac = data["bus", "ac_line", "bus"].edge_label[:, 1]
    tr = data["bus", "transformer", "bus"].edge_label[:, 1]
    return torch.cat([ac, tr])


# ---------------------------------------------------------------------------
# Fixtures (return datasets, loaded once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def case14():
    return OPFDataset(root="data/opfdata", case_name="pglib_opf_case14_ieee",
                      split="train", num_groups=1)


@pytest.fixture(scope="session")
def case118():
    return OPFDataset(root="data/opfdata", case_name="pglib_opf_case118_ieee",
                      split="train", num_groups=1)


@pytest.fixture(scope="session")
def case14_n1():
    return OPFDataset(root="data/opfdata", case_name="pglib_opf_case14_ieee",
                      split="train", num_groups=1, topological_perturbations=True)


@pytest.fixture(scope="session")
def case118_n1():
    return OPFDataset(root="data/opfdata", case_name="pglib_opf_case118_ieee",
                      split="train", num_groups=1, topological_perturbations=True)


# ---------------------------------------------------------------------------
# Integration tests: ground-truth voltages → ground-truth flows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_fixture", ["case14", "case118", "case14_n1", "case118_n1"])
@pytest.mark.parametrize("sample_idx", SAMPLE_IDS)
class TestDeriveBranchFlowsGroundTruth:

    def test_p_fr(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        p_fr, _, _, _ = _derive(data)
        assert torch.allclose(p_fr, _gt_p_fr(data), atol=ATOL), \
            f"p_fr mismatch: max error = {(p_fr - _gt_p_fr(data)).abs().max():.2e}"

    def test_q_fr(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        _, q_fr, _, _ = _derive(data)
        assert torch.allclose(q_fr, _gt_q_fr(data), atol=ATOL), \
            f"q_fr mismatch: max error = {(q_fr - _gt_q_fr(data)).abs().max():.2e}"

    def test_p_to(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        _, _, p_to, _ = _derive(data)
        assert torch.allclose(p_to, _gt_p_to(data), atol=ATOL), \
            f"p_to mismatch: max error = {(p_to - _gt_p_to(data)).abs().max():.2e}"

    def test_q_to(self, data_fixture, sample_idx, request):
        data = request.getfixturevalue(data_fixture)[sample_idx]
        _, _, _, q_to = _derive(data)
        assert torch.allclose(q_to, _gt_q_to(data), atol=ATOL), \
            f"q_to mismatch: max error = {(q_to - _gt_q_to(data)).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Synthetic test: phase shift angle
# ---------------------------------------------------------------------------

def _make_shift_data(br_x, tap, shift, va_j, vm_j):
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["x"] = torch.tensor([100.0])

    data["bus", "ac_line", "bus"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    data["bus", "ac_line", "bus"].branch_vals = torch.zeros(0, 9)

    tr_bv = torch.zeros(1, 11)
    tr_bv[0, 2] = 0.0
    tr_bv[0, 3] = br_x
    tr_bv[0, 7] = tap
    tr_bv[0, 8] = shift

    data["bus", "transformer", "bus"].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    data["bus", "transformer", "bus"].branch_vals = tr_bv

    output_dict = {"bus": torch.tensor([[0.0, 1.0], [va_j, vm_j]])}
    return data, output_dict


@pytest.mark.parametrize("shift", [math.pi / 6, -math.pi / 4, math.pi / 3])
def test_phase_shift_p_fr(shift):
    br_x, va_j, vm_j = 0.1, 0.2, 1.0
    data, output_dict = _make_shift_data(br_x=br_x, tap=1.0, shift=shift, va_j=va_j, vm_j=vm_j)
    p_fr, _, _, _ = CANOS_OPF.derive_branch_flows(None, output_dict, data)
    expected = -vm_j * math.sin(va_j + shift) / br_x
    assert torch.allclose(p_fr, torch.tensor([expected]), atol=1e-5), \
        f"shift={shift:.4f}: got {p_fr.item():.6f}, expected {expected:.6f}"
