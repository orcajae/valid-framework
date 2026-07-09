"""Tests for valid.multiple_testing (Bonferroni / Holm / BH / Romano-Wolf)."""
import numpy as np
import pytest

from valid.multiple_testing import (
    sharpe_pvalues,
    bonferroni,
    holm,
    benjamini_hochberg,
    romano_wolf,
)

# Hand-computed expectations for p = [.001, .008, .039, .041, .042, .06]
# (cross-checked against statsmodels.stats.multitest.multipletests).
P6 = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06])


class TestClassicalCorrections:
    def test_bonferroni_known_values(self):
        reject, p_adj = bonferroni(P6, alpha=0.05)
        np.testing.assert_allclose(p_adj, [0.006, 0.048, 0.234, 0.246, 0.252, 0.36])
        assert list(reject) == [True, True, False, False, False, False]

    def test_holm_known_values(self):
        _, p_adj = holm(P6)
        np.testing.assert_allclose(p_adj, [0.006, 0.040, 0.156, 0.156, 0.156, 0.156])

    def test_bh_known_values(self):
        _, p_adj = benjamini_hochberg(P6)
        np.testing.assert_allclose(p_adj, [0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.06])

    def test_holm_rejects_superset_of_bonferroni(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.uniform(0, 0.2, size=15)
            rej_b, _ = bonferroni(p)
            rej_h, _ = holm(p)
            assert set(np.where(rej_b)[0]) <= set(np.where(rej_h)[0])

    def test_adjusted_p_monotone_in_raw_p(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, size=25)
        order = np.argsort(p)
        for fn in (holm, benjamini_hochberg):
            _, p_adj = fn(p)
            assert np.all(np.diff(p_adj[order]) >= -1e-12)


class TestSharpePvalues:
    def test_lo_t_hand_computed(self):
        # r = [.01, .02, -.01, .02]: mean=.01, sd(ddof=1)=.014142 -> SR=.70711
        # t = SR*sqrt(4)/sqrt(1+0.5*SR^2) = 1.41421/1.11803 = 1.26491
        r = np.array([[0.01], [0.02], [-0.01], [0.02]])
        pvals, t = sharpe_pvalues(r)
        assert t[0] == pytest.approx(1.26491, abs=1e-4)
        assert pvals[0] == pytest.approx(0.10295, abs=1e-4)

    def test_drift_vs_noise(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, size=500)
        drift = rng.normal(0.003, 0.01, size=500)
        pvals, _ = sharpe_pvalues(np.column_stack([noise, drift]))
        assert pvals[0] > 0.05
        assert pvals[1] < 0.01

    def test_zero_variance_column_never_significant(self):
        X = np.column_stack([np.zeros(100), np.random.default_rng(0).normal(0, 1, 100)])
        pvals, t = sharpe_pvalues(X)
        assert np.isneginf(t[0]) and pvals[0] == 1.0


class TestRomanoWolf:
    def test_deterministic_with_seed(self):
        X = np.random.default_rng(7).normal(0, 0.01, size=(300, 8))
        r1 = romano_wolf(X, n_boot=200, seed=123)
        r2 = romano_wolf(X, n_boot=200, seed=123)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_allclose(r1[1], r2[1])

    def test_adjusted_p_monotone_in_t(self):
        X = np.random.default_rng(3).normal(0.0005, 0.01, size=(400, 10))
        _, p_adj, t = romano_wolf(X, n_boot=300, seed=0)
        order = np.argsort(-t)
        assert np.all(np.diff(p_adj[order]) >= -1e-12)

    def test_fwer_on_null_panel(self):
        # 50 iid-null panels: family-wise rejection rate should be near alpha.
        alpha, n_rep = 0.05, 50
        fwe = 0
        for s in range(n_rep):
            X = np.random.default_rng(s).normal(0, 0.01, size=(250, 10))
            reject, _, _ = romano_wolf(X, n_boot=200, alpha=alpha, seed=s)
            fwe += reject.any()
        bound = alpha + 3 * np.sqrt(alpha * (1 - alpha) / n_rep)  # ~0.142
        assert fwe / n_rep <= bound

    def test_power_on_planted_signal(self):
        rng = np.random.default_rng(11)
        T, sd = 500, 0.01
        X = rng.normal(0, sd, size=(T, 12))
        X[:, [2, 5, 9]] += 6 * sd / np.sqrt(T)  # t ~ 6
        reject, _, _ = romano_wolf(X, n_boot=500, seed=11)
        assert reject[[2, 5, 9]].all()
        assert reject.sum() <= 4  # at most one null column slips through

    def test_weakly_dominates_bonferroni(self):
        rng = np.random.default_rng(21)
        T, sd = 500, 0.01
        base = rng.normal(0, sd, size=(T, 1))
        X = base + rng.normal(0, sd / 2, size=(T, 12))  # correlated panel
        X[:, [1, 4]] += 5 * sd / np.sqrt(T)
        rej_rw, _, _ = romano_wolf(X, n_boot=500, seed=21)
        pvals, _ = sharpe_pvalues(X)
        rej_bonf, _ = bonferroni(pvals)
        assert rej_rw.sum() >= rej_bonf.sum()

    def test_serial_dependence_block_bootstrap(self):
        # AR(0.5) null panel with block_size=10: FWER must not blow up.
        alpha, n_rep = 0.05, 30
        fwe = 0
        for s in range(n_rep):
            rng = np.random.default_rng(100 + s)
            eps = rng.normal(0, 0.01, size=(300, 8))
            X = np.zeros_like(eps)
            for i in range(1, 300):
                X[i] = 0.5 * X[i - 1] + eps[i]
            reject, _, _ = romano_wolf(X, n_boot=200, alpha=alpha,
                                       block_size=10, seed=s)
            fwe += reject.any()
        assert fwe / n_rep <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / n_rep) + 0.1

    def test_zero_variance_column(self):
        X = np.random.default_rng(5).normal(0.002, 0.01, size=(300, 3))
        X[:, 1] = 0.0
        reject, p_adj, t = romano_wolf(X, n_boot=200, seed=5)
        assert not reject[1] and p_adj[1] == 1.0 and np.isneginf(t[1])
