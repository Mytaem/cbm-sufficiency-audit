"""Exact macro-F1 optimum over gold-concept profiles (Section 4.4).

Reproduces the 0.826 fixed-sample optimum and its bootstrap interval from
test_gold_profile_counts.csv alone. No post text is needed.

    python exact_optimum.py                 # point optimum
    python exact_optimum.py --bootstrap 2000
"""

import argparse

import numpy as np
import pandas as pd


def optimum(pos, neg):
    """Largest macro-F1 over deterministic profile labelings.

    best_tp[q] is the largest true-positive count reachable with exactly q
    false positives (O(M * N-) time, O(N-) memory).
    """
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    best_tp = np.full(n_neg + 1, -1, dtype=np.int64)
    best_tp[0] = 0
    for a, b in zip(pos.astype(int), neg.astype(int)):
        shifted = np.full_like(best_tp, -1)
        if b == 0:
            shifted = np.where(best_tp >= 0, best_tp + a, -1)
        else:
            shifted[b:] = np.where(best_tp[:-b] >= 0, best_tp[:-b] + a, -1)
        best_tp = np.maximum(best_tp, shifted)
    fp = np.flatnonzero(best_tp >= 0)
    tp = best_tp[fp]
    scores = 0.5 * (2 * tp / (n_pos + tp + fp) + 2 * (n_neg - fp) / (2 * n_neg + n_pos - tp - fp))
    k = int(np.argmax(scores))
    return float(scores[k]), int(tp[k]), int(fp[k])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default="test_gold_profile_counts.csv")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    table = pd.read_csv(args.counts)
    pos, neg = table["n_positive"].to_numpy(), table["n_negative"].to_numpy()
    score, tp, fp = optimum(pos, neg)
    print(f"profiles={len(table)}  N+={pos.sum()}  N-={neg.sum()}")
    print(f"optimum macro-F1={score:.4f}  TP={tp}  FP={fp}")

    if args.bootstrap:
        # Resample posts: expand profiles to posts, draw with replacement, regroup.
        profile = np.repeat(np.arange(len(table)), pos + neg)
        label = np.concatenate([np.r_[np.ones(p, int), np.zeros(n, int)] for p, n in zip(pos, neg)])
        rng = np.random.default_rng(args.seed)
        scores = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, len(label), len(label))
            p = np.bincount(profile[idx], weights=label[idx], minlength=len(table))
            n = np.bincount(profile[idx], weights=1 - label[idx], minlength=len(table))
            keep = (p + n) > 0
            scores.append(optimum(p[keep], n[keep])[0])
        lo, hi = np.percentile(scores, [2.5, 97.5])
        print(f"bootstrap 95% interval: [{lo:.4f}, {hi:.4f}]  ({args.bootstrap} replicates)")


if __name__ == "__main__":
    main()
