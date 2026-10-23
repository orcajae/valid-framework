"""Layer 2 block statistics for publication (CHANGELOG E17).

The Layer 2 per-event return matrices (stage3_R.npz, T x 24 per series) are resampled exchange data and
are not published. This script reduces each of them to the per-block counts, sums and sums of squares from
which every canonical-CSCV PBO of every sub-grid is computed (S contiguous row blocks of sizes differing by
at most one, np.array_split, as in experiments/inbox7/run.py). Arm R (the real BTC series) is left out.

usage (python3 -I): blockstats.py STAGE3_R_NPZ OUT_NPZ
Keys: <arm>_<seed>_S<S>_n (S,), _sum (S, 24), _sumsq (S, 24), for S in {8, 16}.
"""
import sys

import numpy as np


def main(src, dst):
    z = np.load(src)
    out = {}
    for key in z.files:
        arm, seed, _ = key.rsplit("_", 2)
        if arm == "R":
            continue
        R = z[key]
        for S in (8, 16):
            blocks = np.array_split(np.arange(R.shape[0]), S)
            out[f"{arm}_{seed}_S{S}_n"] = np.array([len(b) for b in blocks], dtype=np.int64)
            out[f"{arm}_{seed}_S{S}_sum"] = np.stack([R[b].sum(0) for b in blocks])
            out[f"{arm}_{seed}_S{S}_sumsq"] = np.stack([(R[b] ** 2).sum(0) for b in blocks])
    np.savez_compressed(dst, **out)
    print(f"{len(out) // 6} series written to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
