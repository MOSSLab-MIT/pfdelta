"""
Tests for GNSPowerBalanceLoss in core/utils/pf_losses_utils.py.

GNSPowerBalanceLoss implements a second, independently-derived formulation of
the AC power flow equations (polar admittance form: y_ij, delta_ij) used by
the GNS (Graph Network Solver) model, distinct from PowerBalanceLoss.calculate_PBL
(tested in test_pf_losses.py) which uses a Cartesian (Y_real/Y_imag) form.

The key property: when ground-truth voltages, generation, and demand are fed
in, the returned power mismatches (delta_p, delta_q) should be zero (or
near-zero within floating-point precision) at every bus (layer_loss=True).

layer_loss=False instead solves the reactive power-balance equation for qg
given the true v/theta/qd/shunt and the derived branch flows; on ground truth
that solved qg should match the true data["bus"].qg.

PFDeltaGNS gives PQ and PV buses a flat start (v=1, theta=0) for the voltage
fields used to initialize GNS training, so `v`/`theta` are overwritten here
with the true solved values stored in bus_voltages ([va, vm]) before calling
the loss. pg/pd/qg/qd/shunt/edge_attr are already ground truth.

Tests run over N_SAMPLES samples per dataset to avoid dependence on a single
data point.
"""

import pytest

from core.datasets.pfdelta_variants import PFDeltaGNS
from core.utils.pf_losses_utils import GNSPowerBalanceLoss

ATOL = 1e-4
N_SAMPLES = 50
SAMPLE_IDS = list(range(N_SAMPLES))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ground_truth_data(data):
    """Overwrite the flat-start v/theta with the true solved values."""
    data["bus"].theta = data["bus"].bus_voltages[:, 0].clone()
    data["bus"].v = data["bus"].bus_voltages[:, 1].clone()
    return data


# ---------------------------------------------------------------------------
# Dataset-level fixtures (loaded once per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def case14_gns():
    return PFDeltaGNS(root_dir="data", case_name="case14", add_bus_type=True)


@pytest.fixture(scope="session")
def case118_gns():
    return PFDeltaGNS(root_dir="data", case_name="case118", add_bus_type=True)


# ---------------------------------------------------------------------------
# Integration tests: ground-truth predictions -> zero power-balance mismatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_fixture", ["case14_gns", "case118_gns"])
@pytest.mark.parametrize("sample_idx", SAMPLE_IDS)
class TestGNSPowerBalanceLossGroundTruth:
    def test_delta_p_near_zero(self, data_fixture, sample_idx, request):
        ds = request.getfixturevalue(data_fixture)
        data = _ground_truth_data(ds[sample_idx])
        loss = GNSPowerBalanceLoss()
        delta_p, _, _ = loss(data, layer_loss=True)
        assert delta_p.abs().max() < ATOL, f"delta_p max = {delta_p.abs().max():.2e}"

    def test_delta_q_near_zero(self, data_fixture, sample_idx, request):
        ds = request.getfixturevalue(data_fixture)
        data = _ground_truth_data(ds[sample_idx])
        loss = GNSPowerBalanceLoss()
        _, delta_q, _ = loss(data, layer_loss=True)
        assert delta_q.abs().max() < ATOL, f"delta_q max = {delta_q.abs().max():.2e}"


@pytest.mark.parametrize("data_fixture", ["case14_gns", "case118_gns"])
@pytest.mark.parametrize("sample_idx", SAMPLE_IDS)
class TestGNSPowerBalanceLossQgFromBalance:
    def test_qg_matches_ground_truth(self, data_fixture, sample_idx, request):
        ds = request.getfixturevalue(data_fixture)
        data = _ground_truth_data(ds[sample_idx])
        true_qg = data["bus"].qg.clone()
        loss = GNSPowerBalanceLoss()
        qg_from_balance = loss(data, layer_loss=False)
        max_err = (qg_from_balance - true_qg).abs().max()
        assert max_err < ATOL, f"qg mismatch max = {max_err:.2e}"
