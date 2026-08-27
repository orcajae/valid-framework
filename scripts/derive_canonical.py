#!/usr/bin/env python3
"""
Derive canonical.json — the single home for the figures this project prints.

Design rule, and the reason this file exists at all: **no figure is written as
a literal here.** Every value in `figures` is produced by running or parsing an
existing artifact or generator. A hand-written JSON would be one more copy of
the numbers, which is the problem this replaces, not a fix for it. Values that
genuinely cannot be derived from anything in the tree live in `manual`, each
carrying a `source` and a `[VERIFY]` flag so they read as claims, not as facts.

Sources (all in-repo; nothing outside the worktree is read):
  audit/literature_audit_80.csv       via audit/audit_analysis.py     (판정 1)
  audit/audit_analysis.py             AST-extracted: df, empirical, checks
  scripts/gen_fig7_valid_heatmap.py   AST-extracted: code_paper, WEIGHT
  results/reference/variants_340.csv  via scripts/derive_corpus_stats.py
                                      and scripts/derive_tf_ladder.py (stdout)
  results/reference/multiple_testing.json          read directly
  results/reference/monte_carlo_fpr_200.csv        via experiments/make_summary.py
  results/reference/traditional_baselines.csv      read directly
  experiments/config.py                            imported
  docs/paper/Paper18_..._corrected.pdf             hashed

The AST extraction executes only the named top-level nodes of a source file, so
importing a generator does not run its plotting or overwrite its artifacts.

Idempotence: `generated_at` / `generator_commit` are rewritten only when the
derived payload actually changes. An unchanged re-run reproduces the committed
file byte for byte, which is what lets gate_canon.sh CHECK A be a plain diff and
what makes `generated_at` mean "when the numbers last moved" for the cascade
manifest.

Usage:
  python3 scripts/derive_canonical.py            # write canonical.json
  python3 scripts/derive_canonical.py --stdout   # print, write nothing
  python3 scripts/derive_canonical.py --check    # exit 1 if the file is stale
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "canonical.json"

AUDIT_ANALYSIS = ROOT / "audit" / "audit_analysis.py"
AUDIT_CSV = ROOT / "audit" / "literature_audit_80.csv"
AUDIT_CSV_MIRROR = ROOT / "results" / "reference" / "literature_audit_80.csv"
HEATMAP_GEN = ROOT / "scripts" / "gen_fig7_valid_heatmap.py"
CORPUS_STATS = ROOT / "scripts" / "derive_corpus_stats.py"
TF_LADDER = ROOT / "scripts" / "derive_tf_ladder.py"
MULT_TESTING = ROOT / "results" / "reference" / "multiple_testing.json"
MC_FPR = ROOT / "results" / "reference" / "monte_carlo_fpr_200.csv"
BASELINES = ROOT / "results" / "reference" / "traditional_baselines.csv"
CR_PDF = ROOT / "docs" / "paper" / "Paper18_VALID_camera_ready_corrected.pdf"

sys.path.insert(0, str(ROOT))


class DeriveError(RuntimeError):
    """Raised when a source cannot be parsed. Never fall back to a literal."""


# --------------------------------------------------------------- AST extraction
def load_symbols(path: Path, names, ns=None) -> dict:
    """Execute only the top-level nodes of `path` that bind one of `names`.

    Source order is preserved, so a node may depend on one extracted before it.
    Anything else in the file — imports of matplotlib, figure construction,
    artifact writes — is never executed.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(names)
    selected, bound = [], set()
    for node in tree.body:
        names_here = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names_here = {node.name}
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Name):
                        names_here.add(sub.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names_here = {node.target.id}
        if names_here & wanted:
            selected.append(node)
            bound |= names_here & wanted
    missing = wanted - bound
    if missing:
        raise DeriveError(f"{path.name}: symbols not found: {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(ns or {})
    scope.setdefault("__file__", str(path))
    exec(compile(module, str(path), "exec"), scope)  # noqa: S102 — narrowed above
    return scope


def run_captured(path: Path, func_name: str = "main"):
    """Import a generator by path and capture one function's stdout.

    Returns (module, output) so a caller that also needs the module's constants
    does not import it a second time.
    """
    spec = importlib.util.spec_from_file_location(f"_src_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with redirect_stdout(buf):
        getattr(mod, func_name)()
    return mod, buf.getvalue()


def grab(pattern: str, text: str, label: str, group=1):
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise DeriveError(f"could not parse {label} — source output format changed")
    return m.group(group)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------- collection
def collect() -> dict:
    figures: dict[str, dict] = {}

    def fig(key, value, source, print_fmt=None, derived=True, **extra):
        entry = {
            "value": value,
            "source": source,
            "derived": derived,
            "print_precision": (print_fmt.format(value) if print_fmt else str(value)),
        }
        entry.update(extra)
        figures[key] = entry

    # --- corpus figures, from derive_corpus_stats.py stdout --------------------
    dcs, cs = run_captured(CORPUS_STATS)
    src_cs = "scripts/derive_corpus_stats.py:main (results/reference/variants_340.csv)"

    n_full = int(grab(r"^\(a\).*?\n.*?\n\s*n\s*:\s*(\d+)", cs, "n_variants"))
    neg_full = int(grab(r"^\(a\).*?\n(?:.*\n)*?\s*negative net SR\s*:\s*(\d+)/", cs,
                        "n_negative_full"))
    pct_neg_full = float(grab(
        r"^\(a\).*?\n(?:.*\n)*?\s*negative net SR\s*:\s*\d+/\d+ = ([\d.]+)%", cs,
        "pct_negative_full"))
    above_full = int(grab(
        r"^\(a\).*?\n(?:.*\n)*?\s*> [\d.]+ \(rounded\)\s*:\s*(\d+)/", cs,
        "n_above_bench_full"))
    pct_above_full = float(grab(
        r"^\(a\).*?\n(?:.*\n)*?\s*> [\d.]+ \(rounded\)\s*:\s*\d+/\d+ = ([\d.]+)%", cs,
        "pct_above_bench_full"))

    n_cost = int(grab(r"^\(b\) excluding the (\d+) COST_\*", cs, "n_cost_entries"))
    n_ex = int(grab(r"^\(b\).*?\n.*?\n\s*n\s*:\s*(\d+)", cs, "n_variants_ex_cost"))
    neg_ex = int(grab(r"^\(b\).*?\n(?:.*\n)*?\s*negative net SR\s*:\s*(\d+)/", cs,
                      "n_negative_ex_cost"))
    pct_neg_ex = float(grab(
        r"^\(b\).*?\n(?:.*\n)*?\s*negative net SR\s*:\s*\d+/\d+ = ([\d.]+)%", cs,
        "pct_negative_ex_cost"))
    above_ex = int(grab(
        r"^\(b\).*?\n(?:.*\n)*?\s*> [\d.]+ \(rounded\)\s*:\s*(\d+)/", cs,
        "n_above_bench_ex_cost"))

    fig("n_variants", n_full, src_cs, "{}")
    fig("n_cost_entries", n_cost, src_cs, "{}")
    # 322 is never stored as a value of its own — it is an operation on two keys.
    figures["n_variants_ex_cost"] = {
        "value": n_ex,
        "derived_from": ["n_variants", "n_cost_entries"],
        "op": "subtract",
        "source": src_cs,
        "derived": True,
        "print_precision": str(n_ex),
    }
    if n_full - n_cost != n_ex:
        raise DeriveError(
            f"n_variants - n_cost_entries = {n_full - n_cost}, source says {n_ex}")

    fig("n_negative_full", neg_full, src_cs, "{}")
    fig("pct_negative_full", pct_neg_full, src_cs, "{:.0f}%")
    fig("n_above_bench_full", above_full, src_cs, "{}")
    fig("pct_above_bench_full", pct_above_full, src_cs, "{:.1f}%")
    fig("n_negative_ex_cost", neg_ex, src_cs, "{}")
    fig("pct_negative_ex_cost", pct_neg_ex, src_cs, "{:.1f}%")
    fig("n_above_bench_ex_cost", above_ex, src_cs, "{}")

    # --- benchmark net Sharpe --------------------------------------------------
    mt = json.loads(MULT_TESTING.read_text(encoding="utf-8"))
    fig("benchmark_sr_rounded", mt["benchmark_sr"],
        "results/reference/multiple_testing.json:benchmark_sr", "{:.3f}")

    fig("benchmark_sr_full", dcs.BENCH_FULL,
        "scripts/derive_corpus_stats.py:BENCH_FULL "
        "(results/reference/traditional_baselines.csv)", "{:.6f}")

    # --- multiple testing ------------------------------------------------------
    src_mt = "results/reference/multiple_testing.json"
    fig("e_max_sr_null", mt["E_max_sr_null"], f"{src_mt}:E_max_sr_null", "{:.2f}")
    fig("best_observed_sr", mt["best_observed_sr"], f"{src_mt}:best_observed_sr",
        "{:.2f}")
    fig("n_bonferroni_survivors", mt["methods"]["bonferroni"]["n"],
        f"{src_mt}:methods.bonferroni.n", "{}")
    fig("n_dsr_survivors", mt["methods"]["dsr_normal"]["n"],
        f"{src_mt}:methods.dsr_normal.n", "{}")
    fig("n_uncorrected_survivors", mt["methods"]["before_correction"]["n"],
        f"{src_mt}:methods.before_correction.n", "{}")
    fig("pct_uncorrected_survivors", mt["methods"]["before_correction"]["pct"],
        f"{src_mt}:methods.before_correction.pct", "{:.1f}%")

    # --- literature audit ------------------------------------------------------
    import pandas as pd  # noqa: E402 — deferred so --help works without pandas
    from valid.metrics import wilson_ci

    aa = load_symbols(
        AUDIT_ANALYSIS,
        ["CSV", "df", "NON_EMPIRICAL_IDS", "empirical", "N", "checks"],
        ns={"pd": pd, "Path": Path},
    )
    if Path(aa["CSV"]).resolve() != AUDIT_CSV.resolve():
        raise DeriveError(f"audit source drifted to {aa['CSV']} (판정 1: audit/ 고정)")

    src_audit = "audit/audit_analysis.py:checks (audit/literature_audit_80.csv)"
    n_surveyed = int(len(aa["df"]))
    n_emp = int(aa["N"])
    fig("n_papers_surveyed", n_surveyed,
        "audit/literature_audit_80.csv (row count)", "{}")
    fig("n_papers_empirical", n_emp,
        "audit/audit_analysis.py:N (audit/literature_audit_80.csv)", "{}")
    fig("n_papers_non_empirical", len(aa["NON_EMPIRICAL_IDS"]),
        "audit/audit_analysis.py:NON_EMPIRICAL_IDS", "{}")

    audit_key = {
        "D4: CPCV used": "cpcv_used",
        "D5: BnH only": "d5_bnh_only",
        "D7: No code": "d7_no_code",
        "D1: Cost omitted": "d1_cost_omitted",
        "D2: No class balance": "d2_no_class_balance",
        "D3: Random split": "d3_random_split",
        "D4: Weak validation": "d4_weak_validation",
        "D6: No net perf": "d6_no_net_perf",
    }
    for label, mask in aa["checks"].items():
        if label not in audit_key:
            raise DeriveError(f"unmapped audit dimension {label!r}")
        key = audit_key[label]
        count = int(mask.sum())
        rate = count / n_emp
        lo, hi = wilson_ci(rate, n_emp)
        figures[f"n_{key}"] = {
            "value": count,
            "denominator": n_emp,
            "pct": round(100.0 * rate, 1),
            "ci95_pct": [round(100.0 * lo, 1), round(100.0 * hi, 1)],
            "source": src_audit,
            "derived": True,
            "print_precision": f"{count}/{n_emp} ({100.0 * rate:.0f}%) "
                               f"[{100.0 * lo:.0f}%, {100.0 * hi:.0f}%]",
        }

    # --- median VALID score ----------------------------------------------------
    hm = load_symbols(
        HEATMAP_GEN,
        ["FAIL", "PARTIAL", "PASS", "NA", "ITEMS", "CODES",
         "NON_TRIVIAL_BASELINE", "cell", "code_paper", "WEIGHT"],
        ns={"pd": pd},
    )
    # The two generators keep their own copy of the exclusion list. If they ever
    # disagree the median would silently be scored over a different paper set.
    hm_ids = load_symbols(HEATMAP_GEN, ["NON_EMPIRICAL_IDS"])["NON_EMPIRICAL_IDS"]
    if list(hm_ids) != list(aa["NON_EMPIRICAL_IDS"]):
        raise DeriveError(
            f"exclusion lists disagree: gen_fig7 {hm_ids} vs audit_analysis "
            f"{aa['NON_EMPIRICAL_IDS']}")
    scores = [sum(hm["WEIGHT"][v] for v in hm["code_paper"](row))
              for row in aa["empirical"].itertuples()]
    if len(scores) != n_emp:
        raise DeriveError(f"scored {len(scores)} papers, expected {n_emp}")
    median_score = float(pd.Series(scores).median())
    fig("median_valid_score", median_score,
        "scripts/gen_fig7_valid_heatmap.py:code_paper+WEIGHT "
        "(audit/literature_audit_80.csv)", "{:.1f}")
    fig("n_valid_items", len(hm["CODES"]),
        "scripts/gen_fig7_valid_heatmap.py:ITEMS", "{}")

    # --- Monte Carlo false-positive rate --------------------------------------
    from experiments.make_summary import fpr_line  # noqa: E402

    mc = pd.read_csv(MC_FPR)
    line = fpr_line(mc, len(mc))
    fpr_auc = float(grab(r"AUC-gate ([\d.]+)%", line, "mc_fpr_auc_gate"))
    fpr_pbo = float(grab(r"PBO-gate ([\d.]+)%", line, "mc_fpr_pbo_gate"))
    lo, hi = wilson_ci(fpr_auc / 100.0, len(mc))
    src_mc = ("experiments/make_summary.py:fpr_line "
              "(results/reference/monte_carlo_fpr_200.csv)")
    figures["mc_fpr_auc_gate"] = {
        "value": fpr_auc,
        "denominator": int(len(mc)),
        "ci95_pct": [round(100.0 * lo, 1), round(100.0 * hi, 1)],
        "source": src_mc,
        "derived": True,
        "print_precision": f"{fpr_auc:.0f}% [{100.0 * lo:.0f}, {100.0 * hi:.0f}]",
    }
    fig("mc_fpr_pbo_gate", fpr_pbo, src_mc, "{:.0f}%")
    fig("mc_iterations", int(len(mc)),
        "results/reference/monte_carlo_fpr_200.csv (row count)", "{}")

    # --- cost model ------------------------------------------------------------
    from experiments import config  # noqa: E402

    fig("cost_retail_bp", config.COST_RETAIL_BP,
        "experiments/config.py:COST_RETAIL_BP", "{} bp")
    fig("cost_levels_bp", list(config.COST_LEVELS),
        "experiments/config.py:COST_LEVELS",
        derived=True)
    figures["cost_levels_bp"]["print_precision"] = ", ".join(
        str(c) for c in config.COST_LEVELS)

    # --- trading-frequency ladder ---------------------------------------------
    _tfl, tf = run_captured(TF_LADDER)
    ladder = {}
    for m in re.finditer(
            r"^\s{2}(\S+)\s+n=(\d+)\s+median=([-+][\d.]+)\s+best=([-+][\d.]+)\s+"
            r"share<0=(\d+)%", tf, re.MULTILINE):
        ladder[m.group(1)] = {
            "n": int(m.group(2)),
            "median_net_sr": float(m.group(3)),
            "best_net_sr": float(m.group(4)),
            "pct_negative": int(m.group(5)),
        }
    if not ladder:
        raise DeriveError("derive_tf_ladder.py produced no parseable ladder rows")
    figures["tf_ladder"] = {
        "value": ladder,
        "source": "scripts/derive_tf_ladder.py:main "
                  "(results/reference/variants_340.csv)",
        "derived": True,
        "print_precision": "; ".join(
            f"{k} n={v['n']} median={v['median_net_sr']:+.2f}"
            for k, v in ladder.items()),
    }

    # --- artifact hashes -------------------------------------------------------
    # The camera-ready md5 was listed as a manual item, but the PDF it names is
    # tracked, so it derives. Recorded here as a real drift check rather than an
    # attested string.
    fig("camera_ready_md5", md5(CR_PDF),
        f"md5({CR_PDF.relative_to(ROOT)})", "{}")
    fig("audit_csv_md5", md5(AUDIT_CSV), f"md5({AUDIT_CSV.relative_to(ROOT)})", "{}")
    fig("audit_csv_mirror_md5", md5(AUDIT_CSV_MIRROR),
        f"md5({AUDIT_CSV_MIRROR.relative_to(ROOT)})", "{}")

    # --- values with no in-repo origin ----------------------------------------
    manual = {
        "paper_number": {
            "value": 18,
            "source": "KDD-MLF 2026 acceptance notice; echoed in the tracked "
                      "filename docs/paper/Paper18_VALID_camera_ready_corrected.pdf",
            "derived": False,
            "print_precision": "Paper 18",
            "verify": "[VERIFY]",
        },
        "ssrn_id": {
            "value": "6508779",
            "source": "SSRN submission record; echoed in CITATION.cff and "
                      "pyproject.toml but with no generating artifact",
            "derived": False,
            "print_precision": "SSRN 6508779",
            "verify": "[VERIFY]",
        },
        "self_audit_score": {
            "value": "9/12",
            "source": "CHANGELOG_CR.md — manual application of the VALID "
                      "checklist to this paper; no scorer covers own-paper input",
            "derived": False,
            "print_precision": "9 / 12",
            "verify": "[VERIFY]",
        },
        "n_trades_total": {
            "value": None,
            "claimed": 26215,
            "source": "CLC 지시 토큰 목록. GATE 1/1.5 실측 결과 추적 텍스트 88개 "
                      "및 PDF 6개 전부에서 0회 — repo 내 근거 문서 없음",
            "derived": False,
            "print_precision": "26,215",
            "verify": "[VERIFY] unresolved: 근거 산출물 미발견",
        },
        "n_trades_oos": {
            "value": None,
            "claimed": 6518,
            "source": "CLC 지시 토큰 목록. GATE 1/1.5 실측 결과 추적 텍스트 88개 "
                      "및 PDF 6개 전부에서 0회 — repo 내 근거 문서 없음",
            "derived": False,
            "print_precision": "6,518",
            "verify": "[VERIFY] unresolved: 근거 산출물 미발견",
        },
    }

    # Sources named in the directive that this tree does not contain. Recorded
    # so their absence is visible in the artifact instead of inferred from a
    # missing key.
    unavailable_sources = {
        "regime_conditional.json": "repo 밖 (~/jwquant/paper/kdd-mlf/results/) — "
                                   "clone/CI에서 읽을 수 없어 제외",
        "round_a_orderflow/": "이 저장소에 없음",
        "var_sr_is.csv": "이 저장소에 없음 (monte_carlo_fpr_200.csv의 컬럼명으로만 존재)",
        "seed_optimal_grid.csv": "이 저장소에 없음",
    }

    return {"figures": figures, "manual": manual,
            "unavailable_sources": unavailable_sources}


def build() -> dict:
    payload = collect()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    stamp = (datetime.datetime.now(datetime.timezone.utc)
             .replace(microsecond=0).isoformat().replace("+00:00", "Z"))

    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
        same = all(prev.get(k) == payload[k] for k in payload)
        if same and prev.get("generated_at") and prev.get("generator_commit"):
            stamp = prev["generated_at"]
            commit = prev["generator_commit"]

    return {"generated_at": stamp, "generator_commit": commit, **payload}


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stdout", action="store_true", help="print, write nothing")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if canonical.json differs from a fresh derivation")
    args = ap.parse_args()

    doc = build()
    text = render(doc)

    if args.stdout:
        sys.stdout.write(text)
        return 0
    if args.check:
        if not OUT.exists():
            print(f"FAIL canonical.json missing at {OUT}", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != text:
            print("FAIL canonical.json is stale — re-run without --check",
                  file=sys.stderr)
            return 1
        print("OK canonical.json matches a fresh derivation")
        return 0

    OUT.write_text(text, encoding="utf-8")
    n_fig = len(doc["figures"])
    n_man = len(doc["manual"])
    unresolved = sum(1 for v in doc["manual"].values() if v["value"] is None)
    print(f"wrote {OUT.relative_to(ROOT)}: {n_fig} derived, {n_man} manual "
          f"({unresolved} unresolved), generated_at={doc['generated_at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
