"""Recompute the headline test results from test_predictions.csv.

Checks Table 4 (macro-F1 of the hard CBM, TF-IDF, PhoBERT), the exact
gold-profile optimum (Sec. 5.2), and the paired bootstrap of the gaps
between the text models and the optimum.

    python verify_results.py
    python verify_results.py --bootstrap 2000
"""

import argparse

import numpy as np
import pandas as pd

from exact_optimum import optimum

THRESHOLDS = {"cbm": 0.41, "tfidf": 0.49, "phobert": 0.18}
CONCEPTS = [f"c{i:02d}" for i in range(1, 11)]


def macro_f1(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = len(y) - tp - fp - fn
    return 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))


def profile_optimum(concepts, y):
    _, profile = np.unique(concepts, axis=0, return_inverse=True)
    profile = profile.reshape(-1)
    pos = np.bincount(profile, weights=y)
    neg = np.bincount(profile, weights=1 - y)
    return optimum(pos, neg)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="test_predictions.csv")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    table = pd.read_csv(args.predictions)
    y = table["y"].to_numpy()
    concepts = table[CONCEPTS].to_numpy()
    preds = {name: (table[f"{name}_probability"].to_numpy() >= t).astype(int)
             for name, t in THRESHOLDS.items()}

    print(f"test posts: {len(y)}  positive: {int(y.sum())}")
    for name, pred in preds.items():
        print(f"{name:8s} macro-F1 = {macro_f1(y, pred):.4f}")
    best = profile_optimum(concepts, y)
    print(f"gold-profile optimum = {best:.4f}")

    if args.bootstrap:
        rng = np.random.default_rng(args.seed)
        rows = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, len(y), len(y))
            opt = profile_optimum(concepts[idx], y[idx])
            cbm = macro_f1(y[idx], preds["cbm"][idx])
            phobert = macro_f1(y[idx], preds["phobert"][idx])
            tfidf = macro_f1(y[idx], preds["tfidf"][idx])
            rows.append((opt, phobert - opt, tfidf - opt, cbm - phobert, cbm - tfidf))
        rows = np.array(rows)
        labels = ["optimum", "PhoBERT - optimum", "TF-IDF - optimum", "CBM - PhoBERT", "CBM - TF-IDF"]
        for col, label in enumerate(labels):
            lo, hi = np.percentile(rows[:, col], [2.5, 97.5])
            print(f"{label:18s} 95% interval [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
