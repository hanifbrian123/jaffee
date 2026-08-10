"""Build the dev/test split and the 5-fold CV assignment on dev.

TEST is held out (20%, stratified) and never touched during cross-validation.
DEV (80%) is split into 5 stratified folds; each fold serves once as validation.
The assignment is fixed (seeded) and written to disk so every backbone and the
report all use the identical folds and test set.

No image pixels are opened here; only labels from the index are used.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_FOLDS = 5
SEED = 42


def build():
    index = pd.read_csv(os.path.join(HERE, "index.csv"))
    keys = index["key"].tolist()
    labels = index["label"].tolist()

    dev_keys, test_keys, dev_labels, _ = train_test_split(
        keys, labels, test_size=0.20, random_state=SEED, stratify=labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    dev_keys = np.asarray(dev_keys)
    dev_labels = np.asarray(dev_labels)
    folds = []
    for _, val_idx in skf.split(dev_keys, dev_labels):
        folds.append(sorted(dev_keys[val_idx].tolist()))

    split = {
        "test": sorted(test_keys),
        "dev": sorted(dev_keys.tolist()),
        "folds": folds,
        "n_folds": N_FOLDS,
        "test_size": 0.20,
        "random_state": SEED,
        "stratified": True,
    }
    out = os.path.join(HERE, "experiments", "split_dev_test.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    print(f"TEST (dikunci): {len(test_keys)} gambar")
    print(f"DEV           : {len(dev_keys)} gambar -> {N_FOLDS}-fold CV")
    for k, fold in enumerate(folds):
        print(f"   fold {k}: validasi {len(fold)} / latih {len(dev_keys) - len(fold)}")
    print(f"-> {out}")


if __name__ == "__main__":
    build()
