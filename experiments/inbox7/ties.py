"""In-sample ties in the PBO variants S1, S2 and S4 of the robustness study (CHANGELOG E15 addendum).

usage (python3 -I, from the repository root):
  python3 -I experiments/inbox7/ties.py [results/inbox7/stage1_arrays.npz [OUT_DIR]]

Writes, to OUT_DIR (default: the directory of the arrays file):
  ties.csv        per variant and arm: splits, splits with a tie for the in-sample best, splits with
                  four or more tied. A tie is two or more configurations with exactly the highest
                  in-sample per-event Sharpe ratio.
  tie_rules.csv   per variant and arm: PBO pass rates (< 0.20, < 0.50) and mean PBO under three tie
                  rules. "first" is the rule of the run: np.argmax in E14's pbo_eq1
                  (experiments/true_pbo/run.py), which S1, S2 and S4 call; it reproduces the published
                  PBO of every series. "last" gives a tie to the last-listed configuration. "mean_rank"
                  uses the mean out-of-sample rank of the tied configurations. The last two are outside
                  the pre-registration.
S3 is not covered: its in-sample values are half-split Sharpe ratios computed from block sums.
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "inbox7", "stage1_arrays.npz")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(src))
z = np.load(src)
RULES = ("first", "last", "mean_rank")


def pbo(is_m, oos_m):
    """PBO of one series under each tie rule, and its tie counts (same split rule as pbo_eq1)."""
    hits = dict.fromkeys(RULES, 0)
    tot = tie = tie4 = 0
    for j in range(is_m.shape[1]):
        a, b = is_m[:, j], oos_m[:, j]
        ok = ~(np.isnan(a) | np.isnan(b))
        M = int(ok.sum())
        if M < 2:
            continue
        a, r = a[ok], rankdata(b[ok])
        tied = np.flatnonzero(a == a.max())
        tie += len(tied) > 1
        tie4 += len(tied) >= 4
        for rule, w in (("first", r[tied[0]]), ("last", r[tied[-1]]), ("mean_rank", r[tied].mean())):
            wbar = w / (M + 1)
            hits[rule] += int(np.log(wbar / (1 - wbar)) <= 0)
        tot += 1
    return {k: v / tot for k, v in hits.items()}, tot, tie, tie4


counts = defaultdict(lambda: [0, 0, 0])
values = defaultdict(list)
for key in sorted(z.files):
    m = re.match(r"(?P<arm>[A-Z0-9p]+)_(?P<asset>[A-Z]+)_(?P<seed>\d+)_(?P<var>S[124])_is$", key)
    if not m:
        continue
    p, tot, tie, tie4 = pbo(z[key], z[key[:-3] + "_oos"])
    arm = m["arm"] + ("-" + m["asset"] if m["arm"] == "A" else "")
    for k in ((m["var"], "all"), (m["var"], arm)):
        c = counts[k]
        c[0] += tot
        c[1] += tie
        c[2] += tie4
    values[(m["var"], arm)].append(p)

os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "ties.csv"), "w") as f:
    f.write("variant,arm,splits,splits_with_tie,share_with_tie,splits_with_4_or_more_tied,share_4_or_more\n")
    for (var, arm), (s, t, t4) in sorted(counts.items()):
        f.write(f"{var},{arm},{s},{t},{t / s:.3f},{t4},{t4 / s:.3f}\n")
with open(os.path.join(out, "tie_rules.csv"), "w") as f:
    f.write("variant,arm,series,rule,pass_lt_0.20_pct,pass_lt_0.50_pct,mean_pbo\n")
    for (var, arm), ps in sorted(values.items()):
        for rule in RULES:
            v = np.array([p[rule] for p in ps])
            f.write(f"{var},{arm},{len(v)},{rule},{100 * np.mean(v < 0.2):.1f},{100 * np.mean(v < 0.5):.1f},{v.mean():.3f}\n")
print(f"wrote ties.csv and tie_rules.csv to {out}")
