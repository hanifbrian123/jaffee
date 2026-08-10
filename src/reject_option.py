"""Reject option on the JAFFE ensemble: answer only when confident.

Same idea transferred from CASME II: for a deployed classifier an honest "tidak
yakin" beats a confident wrong answer. Confidence = margin between the top two
class probabilities (the measure that won on CASME II).
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="ensemble_all")
    args = parser.parse_args()
    run_dir = os.path.join(HERE, "experiments", args.run)
    raw = np.load(os.path.join(run_dir, "probs.npz"))
    probs = raw["probs"].astype(float)
    labels = raw["label"]
    preds = probs.argmax(1)

    ordered = np.sort(probs, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    order = np.argsort(-margin)

    base_acc = float(accuracy_score(labels, preds))
    rows = []
    print(f"model: {args.run}   tanpa reject: acc={base_acc:.4f} n={len(labels)}\n")
    print(f'{"dijawab":>9s} {"n":>4s} {"acc":>7s}   ambang margin')
    for frac in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(len(labels) * frac)))
        sel = order[:k]
        acc = float(accuracy_score(labels[sel], preds[sel]))
        rows.append({"coverage": k / len(labels), "n": k, "acc": acc,
                     "threshold": float(margin[order[k - 1]])})
        print(f"{frac*100:8.0f}% {k:4d} {acc:7.4f}   {margin[order[k-1]]:.4f}")

    with open(os.path.join(run_dir, "reject_option.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"run": args.run, "base_acc": base_acc, "curve": rows},
                  handle, indent=2)
    print(f"\nlaporan -> {os.path.join(run_dir, 'reject_option.json')}")


if __name__ == "__main__":
    main()
