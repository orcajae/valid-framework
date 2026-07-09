"""Unit tests for metrics."""
import numpy as np
import pytest
from valid.metrics import (
    annualized_sharpe, var_sr_is, wilson_ci, compute_pbo, deflated_sharpe_ratio,
)


def test_sharpe_positive():
    np.random.seed(42)
    r = np.random.normal(0.001, 0.01, 252)  # positive mean, some variance
    assert annualized_sharpe(r) > 0


def test_sharpe_zero():
    assert annualized_sharpe(np.array([0, 0, 0])) == 0.0


def test_var_sr_is():
    mat = np.array([[1.0, 1.5], [0.8, 1.2], [1.1, 1.3]])
    v = var_sr_is(mat)
    assert v > 0


def test_wilson_ci_extreme():
    lo, hi = wilson_ci(0.0, 100)
    assert lo == 0.0
    assert hi > 0


def test_pbo_perfect():
    is_m = np.array([[1.0, 0.5], [0.5, 1.0]])
    oos_m = np.array([[1.0, 0.5], [0.5, 1.0]])
    pbo, _, _ = compute_pbo(is_m, oos_m)
    assert 0 <= pbo <= 1


def test_dsr_decreases_with_more_strategies():
    dsr_few, _, _ = deflated_sharpe_ratio(1.0, 2, 1000, 0.0, 3.0)
    dsr_many, _, _ = deflated_sharpe_ratio(1.0, 100, 1000, 0.0, 3.0)
    assert dsr_many < dsr_few


def test_dsr_half_when_sr_equals_expected_max():
    _, e_max, _ = deflated_sharpe_ratio(1.0, 50, 1000, 0.0, 3.0)
    dsr, _, _ = deflated_sharpe_ratio(e_max, 50, 1000, 0.0, 3.0)
    assert dsr == pytest.approx(0.5, abs=1e-9)


def test_dsr_se_positive():
    _, _, se = deflated_sharpe_ratio(0.8, 20, 500, -0.5, 4.0)
    assert se > 0
