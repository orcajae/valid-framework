"""VALID 12-item checklist implementation.

Reference: Kim (2026), "Beyond Accuracy: A Validation Framework
for Machine Learning in Cryptocurrency Trading"

Usage:
    from valid.checklist import VALIDChecker
    checker = VALIDChecker()
    report = checker.run_all(y_pred_unbal=preds, ...)
    report.print_summary()
"""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VALIDResult:
    item: str
    name: str
    passed: bool
    details: Dict
    recommendation: str = ""


@dataclass
class VALIDReport:
    results: List[VALIDResult] = field(default_factory=list)

    @property
    def score(self):
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self):
        return len(self.results)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"VALID Framework Assessment: {self.score}/{self.total}")
        print(f"{'='*60}")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {r.item}: {r.name:<40} [{status}]")
            if not r.passed and r.recommendation:
                print(f"         -> {r.recommendation}")
        stage1 = [r for r in self.results
                  if r.item in ["V1","V2","V3","V4","V5","V6","V9","V12"]]
        stage2 = [r for r in self.results
                  if r.item in ["V7","V8","V10","V11"]]
        s1 = sum(1 for r in stage1 if r.passed)
        s2 = sum(1 for r in stage2 if r.passed)
        print(f"\n  Stage 1 (Reporting):  {s1}/{len(stage1)}")
        print(f"  Stage 2 (Deployment): {s2}/{len(stage2)}")

    def to_markdown(self, filepath):
        lines = [
            "# VALID Framework Assessment Report\n",
            f"**Score: {self.score}/{self.total}**\n",
            "| Item | Name | Status | Details |",
            "|------|------|--------|---------|",
        ]
        for r in self.results:
            s = "PASS" if r.passed else "**FAIL**"
            d = json.dumps(r.details, default=str)[:80]
            lines.append(f"| {r.item} | {r.name} | {s} | {d} |")
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    def to_dict(self):
        return {
            "score": f"{self.score}/{self.total}",
            "items": [{"item": r.item, "name": r.name,
                        "passed": r.passed, "details": r.details}
                       for r in self.results],
        }


class VALIDChecker:
    """Run all 12 VALID checks on a financial ML strategy."""

    def __init__(self, bull_bias_threshold=0.75, pbo_threshold=0.50,
                 var_sr_null_threshold=0.02, permutation_alpha=0.05):
        self.bull_bias_threshold = bull_bias_threshold
        self.pbo_threshold = pbo_threshold
        self.var_sr_null_threshold = var_sr_null_threshold
        self.permutation_alpha = permutation_alpha

    def check_v1(self, y_pred) -> VALIDResult:
        if y_pred is None:
            return VALIDResult("V1", "Prediction distribution", False, {},
                               "Provide predictions")
        long_pct = float(np.mean(y_pred == 1))
        flagged = long_pct > self.bull_bias_threshold
        return VALIDResult("V1", "Prediction distribution", True,
                           {"long_pct": f"{long_pct:.1%}", "bias_warning": flagged},
                           "Long% > 75%: bull bias likely" if flagged else "")

    def check_v2(self, y_pred_unbal, y_pred_bal) -> VALIDResult:
        if y_pred_unbal is None or y_pred_bal is None:
            return VALIDResult("V2", "Class balancing test", False, {},
                               "Run with and without class balancing")
        unbal = float(np.mean(y_pred_unbal == 1))
        bal = float(np.mean(y_pred_bal == 1))
        bias = unbal > self.bull_bias_threshold
        return VALIDResult("V2", "Class balancing test", not bias,
                           {"unbal_long": f"{unbal:.1%}", "bal_long": f"{bal:.1%}",
                            "bias_detected": bias},
                           'Apply class_weight="balanced"' if bias else "")

    def check_v3(self, has_temporal_split) -> VALIDResult:
        return VALIDResult("V3", "Temporal splitting", has_temporal_split,
                           {"temporal_split": has_temporal_split},
                           "Never use random k-fold on time series"
                           if not has_temporal_split else "")

    def check_v4(self, pbo_value, n_configs=0) -> VALIDResult:
        if pbo_value is None:
            return VALIDResult("V4", "CPCV/PBO validation", False, {},
                               "Apply CPCV and compute PBO")
        passed = pbo_value < self.pbo_threshold
        return VALIDResult("V4", "CPCV/PBO validation", passed,
                           {"pbo": pbo_value, "n_configs": n_configs,
                            "threshold": self.pbo_threshold},
                           f"PBO={pbo_value:.3f} >= {self.pbo_threshold}"
                           if not passed else "")

    def check_v5(self, var_sr_is) -> VALIDResult:
        if var_sr_is is None:
            return VALIDResult("V5", "Parameter-space variance", False, {},
                               "Compute Var(SR_IS) across CPCV folds")
        flat = var_sr_is < self.var_sr_null_threshold
        return VALIDResult("V5", "Parameter-space variance", not flat,
                           {"var_sr_is": var_sr_is,
                            "threshold": self.var_sr_null_threshold,
                            "flat_landscape": flat},
                           "Flat landscape: PBO may be unreliable" if flat else "")

    def check_v6(self, has_permutation, permutation_p=None) -> VALIDResult:
        if not has_permutation:
            return VALIDResult("V6", "Permutation test", False, {},
                               "Run >=100 permutation shuffles")
        passed = permutation_p is not None and permutation_p < self.permutation_alpha
        return VALIDResult("V6", "Permutation test", passed,
                           {"p_value": permutation_p, "alpha": self.permutation_alpha},
                           f"p={permutation_p:.3f} not significant" if not passed else "")

    def check_v7(self, gross_sr, net_sr, cost_bp) -> VALIDResult:
        if net_sr is None:
            return VALIDResult("V7", "Net performance", False, {},
                               "Report net (cost-adjusted) SR")
        drag = (1 - net_sr / gross_sr) * 100 if gross_sr > 0 else float("inf")
        return VALIDResult("V7", "Net performance", True,
                           {"gross_sr": gross_sr, "net_sr": net_sr,
                            "cost_bp": cost_bp, "cost_drag_pct": f"{drag:.0f}%"})

    def check_v8(self, sr_at_costs: Dict[int, float]) -> VALIDResult:
        if not sr_at_costs or len(sr_at_costs) < 3:
            return VALIDResult("V8", "Cost sensitivity", False, {},
                               "Test at 0bp, realistic, and conservative costs")
        vals = list(sr_at_costs.values())
        sign_flip = vals[0] > 0 and vals[-1] < 0
        return VALIDResult("V8", "Cost sensitivity", True,
                           {"sr_by_cost": sr_at_costs, "sign_flip": sign_flip},
                           "Alpha sign flips at high costs" if sign_flip else "")

    def check_v9(self, ml_sr, baseline_srs: Dict[str, float]) -> VALIDResult:
        if not baseline_srs:
            return VALIDResult("V9", "Baseline comparison", False, {},
                               "Compare vs buy-and-hold, momentum, RSI/SMA")
        beats_any = any(ml_sr > b for b in baseline_srs.values())
        return VALIDResult("V9", "Baseline comparison", beats_any,
                           {"ml_sr": ml_sr, "baselines": baseline_srs},
                           "ML fails to beat simple baselines" if not beats_any else "")

    def check_v10(self, has_bear_eval) -> VALIDResult:
        return VALIDResult("V10", "Bear market evaluation", has_bear_eval,
                           {"bear_market_tested": has_bear_eval},
                           "Evaluate during >=1 bear period" if not has_bear_eval else "")

    def check_v11(self, trades_per_year, gross_alpha, cost_drag) -> VALIDResult:
        cpa = cost_drag / gross_alpha if gross_alpha > 0 else float("inf")
        return VALIDResult("V11", "Trade frequency reporting", True,
                           {"trades_yr": trades_per_year, "cost_per_alpha": f"{cpa:.1%}"})

    def check_v12(self, code_available) -> VALIDResult:
        return VALIDResult("V12", "Reproducibility", code_available,
                           {"code_available": code_available},
                           "Provide GitHub repo with pipeline code"
                           if not code_available else "")

    def run_all(self, **kw) -> VALIDReport:
        report = VALIDReport()
        report.results.append(self.check_v1(kw.get("y_pred_unbal")))
        report.results.append(self.check_v2(kw.get("y_pred_unbal"), kw.get("y_pred_bal")))
        report.results.append(self.check_v3(kw.get("has_temporal_split", False)))
        report.results.append(self.check_v4(kw.get("pbo_value"), kw.get("n_configs", 0)))
        report.results.append(self.check_v5(kw.get("var_sr_is")))
        report.results.append(self.check_v6(kw.get("has_permutation_test", False), kw.get("permutation_p")))
        report.results.append(self.check_v7(kw.get("gross_sr", 0), kw.get("net_sr"), kw.get("cost_bp", 18)))
        report.results.append(self.check_v8(kw.get("sr_at_costs", {})))
        report.results.append(self.check_v9(kw.get("ml_sr", 0), kw.get("baseline_srs", {})))
        report.results.append(self.check_v10(kw.get("has_bear_market_eval", False)))
        report.results.append(self.check_v11(kw.get("trades_per_year", 0), kw.get("gross_alpha", 0), kw.get("cost_drag", 0)))
        report.results.append(self.check_v12(kw.get("code_available", False)))
        return report


if __name__ == "__main__":
    checker = VALIDChecker()
    report = checker.run_all(
        y_pred_unbal=np.array([1]*972 + [0]*28),
        y_pred_bal=np.array([1]*423 + [0]*577),
        has_temporal_split=True, pbo_value=0.000, n_configs=15,
        var_sr_is=0.012, has_permutation_test=True, permutation_p=0.000,
        gross_sr=0.84, net_sr=0.135, cost_bp=18,
        sr_at_costs={0: 0.640, 18: 0.530, 50: 0.335}, ml_sr=0.530,
        baseline_srs={"buy_hold": 0.618, "sma200": 0.723, "rsi": 0.804, "momentum": 0.917},
        has_bear_market_eval=True, trades_per_year=30.8,
        gross_alpha=0.84, cost_drag=0.71, code_available=True,
    )
    report.print_summary()
