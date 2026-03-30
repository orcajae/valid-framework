"""Unit tests for VALID checklist."""
import numpy as np
import pytest
from valid.checklist import VALIDChecker, VALIDReport


class TestV1:
    def test_extreme_bull_bias(self):
        checker = VALIDChecker()
        y = np.array([1] * 97 + [0] * 3)
        result = checker.check_v1(y)
        assert result.details["bias_warning"] is True

    def test_balanced_predictions(self):
        checker = VALIDChecker()
        y = np.array([1] * 50 + [0] * 50)
        result = checker.check_v1(y)
        assert result.details["bias_warning"] is False


class TestV2:
    def test_detects_bias(self):
        checker = VALIDChecker()
        unbal = np.array([1] * 90 + [0] * 10)
        bal = np.array([1] * 50 + [0] * 50)
        result = checker.check_v2(unbal, bal)
        assert result.passed is False

    def test_no_bias(self):
        checker = VALIDChecker()
        unbal = np.array([1] * 60 + [0] * 40)
        bal = np.array([1] * 50 + [0] * 50)
        result = checker.check_v2(unbal, bal)
        assert result.passed is True


class TestV5:
    def test_flat_landscape(self):
        checker = VALIDChecker()
        result = checker.check_v5(0.005)
        assert result.details["flat_landscape"] is True
        assert result.passed is False

    def test_non_flat(self):
        checker = VALIDChecker()
        result = checker.check_v5(0.15)
        assert result.passed is True


class TestFullReport:
    def test_paper_example(self):
        checker = VALIDChecker()
        report = checker.run_all(
            y_pred_unbal=np.array([1] * 972 + [0] * 28),
            y_pred_bal=np.array([1] * 423 + [0] * 577),
            has_temporal_split=True,
            pbo_value=0.000, n_configs=15, var_sr_is=0.012,
            has_permutation_test=True, permutation_p=0.000,
            gross_sr=0.84, net_sr=0.135, cost_bp=18,
            sr_at_costs={0: 0.640, 18: 0.530, 50: 0.335},
            ml_sr=0.530,
            baseline_srs={"buy_hold": 0.618, "sma200": 0.723},
            has_bear_market_eval=True, trades_per_year=30.8,
            gross_alpha=0.84, cost_drag=0.71, code_available=True,
        )
        assert isinstance(report, VALIDReport)
        assert report.total == 12
        v2 = [r for r in report.results if r.item == "V2"][0]
        assert v2.passed is False  # bull bias
        v5 = [r for r in report.results if r.item == "V5"][0]
        assert v5.passed is False  # flat landscape
        v9 = [r for r in report.results if r.item == "V9"][0]
        assert v9.passed is False  # ML < all baselines

    def test_markdown_output(self, tmp_path):
        checker = VALIDChecker()
        report = checker.run_all(
            y_pred_unbal=np.array([1] * 50 + [0] * 50),
            y_pred_bal=np.array([1] * 50 + [0] * 50),
            has_temporal_split=True, pbo_value=0.1, var_sr_is=0.1,
            has_permutation_test=True, permutation_p=0.01,
            gross_sr=1.0, net_sr=0.8, cost_bp=18,
            sr_at_costs={0: 1.0, 18: 0.8, 50: 0.5},
            ml_sr=0.9, baseline_srs={"bnh": 0.5},
            has_bear_market_eval=True, trades_per_year=20,
            gross_alpha=1.0, cost_drag=0.2, code_available=True,
        )
        out = tmp_path / "report.md"
        report.to_markdown(str(out))
        assert out.exists()
        content = out.read_text()
        assert "VALID" in content
