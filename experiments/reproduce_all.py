"""One-command reproduction pipeline orchestrator.

Runs the real stages in order and FAILS LOUDLY (nonzero exit) if any stage
fails. Tiers (see REPRODUCE.md):

  python experiments/reproduce_all.py --smoke   # CI tier, minutes
  python experiments/reproduce_all.py           # default: synthetic, full MC
  python experiments/reproduce_all.py --real    # public Binance data via ccxt
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXP = ROOT / "experiments"


def stage_cmds(args):
    passthrough = []
    if args.smoke:
        passthrough.append("--smoke")
    data = ["--data", "real" if args.real else "synthetic"]

    stages = []
    if args.real:
        stages.append(("download", [sys.executable, str(EXP / "download_data.py")]))
    stages += [
        ("mc_null", [sys.executable, str(EXP / "mc_expansion.py"), "--all",
                     "--model", args.model] + data + passthrough),
        ("variant_grid", [sys.executable, str(EXP / "run_variant_grid.py")] + data + passthrough),
        ("multiple_testing", [sys.executable, str(EXP / "run_multiple_testing.py")] + passthrough),
        ("worked_example", [sys.executable, str(ROOT / "examples" / "worked_example.py")]
         + data + passthrough),
        ("summary", [sys.executable, str(EXP / "make_summary.py"), "--tier",
                     "smoke" if args.smoke else ("real" if args.real else "default")]),
    ]
    return [(name, cmd) for name, cmd in stages if name not in args.skip]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="CI tier (minutes)")
    parser.add_argument("--real", action="store_true",
                        help="download + use public Binance OHLCV (needs [data] extra)")
    parser.add_argument("--model", choices=["auto", "catboost", "hgb"], default="auto")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="stage names to skip (e.g. --skip mc_null)")
    args = parser.parse_args()

    print("=" * 70)
    print("VALID REPRODUCTION PIPELINE")
    print("=" * 70)

    t0 = time.time()
    for name, cmd in stage_cmds(args):
        print(f"\n>>> stage: {name}")
        t1 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            print(f"\nFAILED at stage '{name}' (exit {proc.returncode}) "
                  f"after {time.time() - t1:.0f}s", file=sys.stderr)
            sys.exit(proc.returncode)
        print(f"<<< {name} done in {time.time() - t1:.0f}s")

    print(f"\nAll stages complete in {(time.time() - t0) / 60:.1f} min.")
    print("Summary: results/REPRODUCE_SUMMARY.md")


if __name__ == "__main__":
    main()
