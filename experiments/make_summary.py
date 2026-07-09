"""Assemble results/REPRODUCE_SUMMARY.md from the artifacts of a pipeline run.

Reads whatever stages produced (mc_fpr_summary.csv, multiple_testing.csv,
worked_example summary) and, for the MC stage, places the run's numbers
side-by-side with the tracked reference CSVs — with an explicit note that
synthetic-tier numbers verify the mechanism, not the published values.
Every number in the summary is read from a produced artifact; nothing is
typed in by hand.
"""
import argparse
import datetime
import platform

import pandas as pd

try:
    from experiments import config
except ImportError:
    import config

REFERENCE_MAP = {
    # generated artifact (results/)      reference artifact (results/reference/)
    "mc_fpr_btc_daily.csv": "mc_fpr_btc_daily_200.csv",
    "mc_fpr_eth_daily.csv": "mc_fpr_eth_1h_200.csv",   # reference name mislabeled 1h; daily run
    "mc_fpr_sol_daily.csv": "mc_fpr_sol_1h_200.csv",   # reference name mislabeled 1h; daily run
}


def fpr_line(df, n):
    fpr_auc = (df["auc"] > 0.55).mean()
    fpr_pbo = (df["pbo"] < 0.20).mean()
    return f"AUC-gate {fpr_auc * 100:.1f}% / PBO-gate {fpr_pbo * 100:.1f}% (n={n})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="default",
                        help="label recorded in the summary (smoke/default/real)")
    args = parser.parse_args()

    lines = [
        "# Reproduction Run Summary",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Tier: {args.tier}",
        f"- Platform: {platform.platform()} / Python {platform.python_version()}",
        "",
    ]

    summary_path = config.RESULTS_DIR / "mc_fpr_summary.csv"
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        lines += ["## Monte Carlo null-pipeline FPR", "",
                  "| setting | gate | FPR | Wilson 95% CI | n |",
                  "|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lines.append(f"| {r.setting} | {r.gate} | {r.fpr * 100:.1f}% | "
                         f"[{r.wilson_lo * 100:.1f}%, {r.wilson_hi * 100:.1f}%] | {int(r.n)} |")
        lines += ["", "### Side-by-side with tracked reference runs", "",
                  "| setting | this run | reference (real data, CatBoost, 200 iter) |",
                  "|---|---|---|"]
        for gen, ref in REFERENCE_MAP.items():
            gen_p = config.RESULTS_DIR / gen
            ref_p = config.RESULTS_DIR / "reference" / ref
            if gen_p.exists() and ref_p.exists():
                g, r = pd.read_csv(gen_p), pd.read_csv(ref_p)
                lines.append(f"| {gen.replace('mc_fpr_', '').replace('.csv', '')} | "
                             f"{fpr_line(g, len(g))} | {fpr_line(r, len(r))} |")
        lines += ["",
                  "Synthetic/smoke tiers verify the *mechanism* (a signal-free "
                  "pipeline passes naive gates at high rates and CPCV+PBO gates "
                  "at low rates); the reference numbers come from full real-data "
                  "runs. See REPRODUCE.md for tier definitions and deviations.", ""]

    mt_path = config.RESULTS_DIR / "multiple_testing.csv"
    if mt_path.exists():
        mt = pd.read_csv(mt_path)
        lines += ["## Multiple-testing survivors", "",
                  "| method | survivors |", "|---|---|"]
        for name, col in [("raw p<0.05", "rej_raw"), ("Bonferroni", "rej_bonferroni"),
                          ("Holm", "rej_holm"), ("Benjamini-Hochberg", "rej_bh"),
                          ("Romano-Wolf", "rej_romano_wolf"),
                          ("Harvey t>3", "rej_harvey_t3"), ("DSR>0.95", "rej_dsr95")]:
            lines.append(f"| {name} | {int(mt[col].sum())}/{len(mt)} |")
        lines.append("")

    we_path = config.RESULTS_DIR / "worked_example" / "summary.csv"
    if we_path.exists():
        we = pd.read_csv(we_path)
        lines += ["## Worked example (VALID checklist scoreboard)", "",
                  "| strategy | VALID score |", "|---|---|"]
        for _, r in we.iterrows():
            lines.append(f"| {r.strategy} | {int(r.score)}/{int(r.total)} |")
        lines.append("")

    out = config.RESULTS_DIR / "REPRODUCE_SUMMARY.md"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
