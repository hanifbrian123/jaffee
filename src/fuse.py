"""Ensemble JAFFE models by averaging softmax probs on the shared test set.

The ensemble was the single idea that robustly won on CASME II. Members must
share the exact test set (guaranteed by split_8020.json), which is checked here.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load(name):
    raw = np.load(os.path.join(HERE, "experiments", name, "probs.npz"))
    order = np.argsort([str(k) for k in raw["keys"]])
    return {"keys": np.asarray([str(k) for k in raw["keys"]])[order],
            "label": raw["label"][order], "probs": raw["probs"][order].astype(float)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--name", default="ensemble")
    args = parser.parse_args()

    members = [load(m) for m in args.members]
    ref = members[0]
    for m in members[1:]:
        if not np.array_equal(m["keys"], ref["keys"]):
            raise ValueError("anggota ensemble punya test set berbeda")
    labels = ref["label"]
    probs = sum(m["probs"] for m in members) / len(members)
    preds = probs.argmax(1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    uar = float(recall_score(labels, preds, average="macro", zero_division=0))
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))

    out_dir = os.path.join(HERE, "experiments", args.name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "probs.npz"), keys=ref["keys"], label=labels,
             probs=probs.astype(np.float32))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv"))
    summary = {"name": args.name, "complete": True, "members": args.members,
               "num_classes": len(CLASSES), "ACC": acc, "macroF1": macro_f1,
               "UAR": uar, "confusion_matrix": cm.tolist()}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)

    ledger = os.path.join(HERE, "experiments", "results_jaffe.csv")
    pd.DataFrame([{"name": args.name, "time": "", "ACC": acc, "macroF1": macro_f1,
                   "UAR": uar, "backbone": "ensemble", "seed": -1,
                   "epochs": -1, "tta": -1, "elapsed_sec": 0}]).to_csv(
        ledger, mode="a", header=not os.path.exists(ledger), index=False)
    print(json.dumps({"name": args.name, "ACC": acc, "macroF1": macro_f1,
                      "UAR": uar, "n_members": len(members)}, indent=2))


if __name__ == "__main__":
    main()
