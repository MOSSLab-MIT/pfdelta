"""
Tests for constraint_violations_loss in core/utils/opf_losses_utils.py.

The key property: when ground-truth predictions (bus voltages, generator
outputs, branch flows) are fed into the loss, each component should be
zero (or near-zero within floating-point precision):

  - Flow mismatch:    derived flows must match stored edge_label exactly
  - Power balance:    KCL must be satisfied at every bus
  - Voltage bounds:   feasible OPF solutions must respect v_lims
  - Gen bounds:       feasible OPF solutions must respect p_lims / q_lims

The test also checks that perturbing bus voltages causes violations to
increase above zero.

Attributes set up from raw OPFDataset (mirroring opfdata.py pre_transform):
  branch_vals     = edge_attr  (unnormalized branch parameters)
  v_lims          = bus.x[:, 2:]
  p_lims          = generator.x[:, 2:4]
  q_lims          = generator.x[:, 5:7]
  unnormalized    = load.x / shunt.x
"""

import pytest
import torch
from torch_geometric.datasets import OPFDataset

from core.utils.opf_losses_utils import constraint_violations_loss

ATOL = 1e-4  # tolerance for near-zero violation checks
N_SAMPLES = 50
SAMPLE_IDS = list(range(N_SAMPLES))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup(data):
    """Attach attributes that opfdata pre_transform would normally set."""
    data["bus", "ac_line", "bus"].branch_vals = data["bus", "ac_line", "bus"].edge_attr
    data["bus", "transformer", "bus"].branch_vals = data["bus", "transformer", "bus"].edge_attr
    data["bus"]["v_lims"] = data["bus"]["x"][:, 2:]
    data["generator"]["p_lims"] = data["generator"]["x"][:, 2:4]
    data["generator"]["q_lims"] = data["generator"]["x"][:, 5:7]
    data["load"]["unnormalized"] = data["load"]["x"]
    data["shunt"]["unnormalized"] = data["shunt"]["x"]
    return data


def _gt_output_dict(data):
    """Build output_dict from ground-truth labels."""
    ac_label = data["bus", "ac_line", "bus"].edge_label
    tr_label = data["bus", "transformer", "bus"].edge_label
    return {
        "bus": data["bus"].y,
        "generator": data["generator"].y,
        "edge_preds": torch.cat([ac_label, tr_label], dim=0),
    }


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
# Integration tests: ground-truth predictions → zero violations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_fixture", ["case14", "case118", "case14_n1", "case118_n1"])
@pytest.mark.parametrize("sample_idx", SAMPLE_IDS)
class TestConstraintViolationsGroundTruth:

    def test_flow_mismatch_real(self, data_fixture, sample_idx, request):
        data = _setup(request.getfixturevalue(data_fixture)[sample_idx])
        loss = constraint_violations_loss()
        loss(_gt_output_dict(data), data)
        assert loss.real_flow_mismatch_violation < ATOL, \
            f"real flow mismatch = {loss.real_flow_mismatch_violation:.2e}"

    def test_flow_mismatch_reactive(self, data_fixture, sample_idx, request):
        data = _setup(request.getfixturevalue(data_fixture)[sample_idx])
        loss = constraint_violations_loss()
        loss(_gt_output_dict(data), data)
        assert loss.imag_flow_mismatch_violation < ATOL, \
            f"reactive flow mismatch = {loss.imag_flow_mismatch_violation:.2e}"

    def test_power_balance_real(self, data_fixture, sample_idx, request):
        data = _setup(request.getfixturevalue(data_fixture)[sample_idx])
        loss = constraint_violations_loss()
        loss(_gt_output_dict(data), data)
        assert loss.bus_real_mismatch < ATOL, \
            f"real power balance mismatch = {loss.bus_real_mismatch:.2e}"

    def test_power_balance_reactive(self, data_fixture, sample_idx, request):
        data = _setup(request.getfixturevalue(data_fixture)[sample_idx])
        loss = constraint_violations_loss()
        loss(_gt_output_dict(data), data)
        assert loss.bus_reactive_mismatch < ATOL, \
            f"reactive power balance mismatch = {loss.bus_reactive_mismatch:.2e}"

    def test_voltage_bounds(self, data_fixture, sample_idx, request):
        data = _setup(request.getfixturevalue(data_fixture)[sample_idx])
        loss = constraint_violations_loss()
        loss(_gt_output_dict(data), data)
        assert loss.voltage_violations < ATOL, \
            f"voltage bound violation = {loss.voltage_violations:.2e}"

    def test_generator_pg_bounds(self, data_fixture, sample_idx, request):
        data = _setup(request.getfixturevalue(data_fixture)[sample_idx])
        loss = constraint_violations_loss()
        loss(_gt_output_dict(data), data)
        assert loss.generator_pg_violations < ATOL, \
            f"generator Pg bound violation = {loss.generator_pg_violations:.2e}"

    def test_generator_qg_bounds(self, data_fixture, sample_idx, request):
        data = _setup(request.getfixturevalue(data_fixture)[sample_idx])
        loss = constraint_violations_loss()
        loss(_gt_output_dict(data), data)
        assert loss.generator_qg_violations < ATOL, \
            f"generator Qg bound violation = {loss.generator_qg_violations:.2e}"