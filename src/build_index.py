"""Build the JAFFE index and the fixed 80/20 split (test doubles as validation).

Labels come from the filename: e.g. ``KA.AN1.39.tiff`` -> subject KA,
expression AN (anger). The split is stratified by label with a fixed seed and
written to disk so every model trains and is evaluated on the exact same test
set — a hard requirement for the ensemble step to be valid.

No image pixels are opened here; only filenames are parsed.
"""
import json
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(HERE, "dataset", "jaffe")

# Alphabetical, stable label ids.
EXPRESSIONS = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]
EXPR_NAME = {"AN": "angry", "DI": "disgust", "FE": "fear", "HA": "happy",
             "NE": "neutral", "SA": "sad", "SU": "surprise"}
LABEL = {code: index for index, code in enumerate(EXPRESSIONS)}
NAME_RE = re.compile(r"^([A-Z]{2})\.([A-Z]{2})\d*\.")


def build():
    rows = []
    for filename in sorted(os.listdir(IMG_DIR)):
        if not filename.lower().endswith(".tiff"):
            continue
        match = NAME_RE.match(filename)
        if not match:
            raise ValueError(f"nama file tak terduga: {filename}")
        subject, expr = match.group(1), match.group(2)
        if expr not in LABEL:
            raise ValueError(f"ekspresi tak dikenal {expr} di {filename}")
        rows.append({
            "key": filename,
            "path": os.path.join("dataset", "jaffe", filename),
            "subject": subject,
            "expr": expr,
            "emotion": EXPR_NAME[expr],
            "label": LABEL[expr],
        })
    table = pd.DataFrame(rows)
    index_path = os.path.join(HERE, "index.csv")
    table.to_csv(index_path, index=False)
    print(f"index: {len(table)} gambar -> {index_path}")
    print("distribusi kelas:")
    for code in EXPRESSIONS:
        n = int((table["expr"] == code).sum())
        print(f"   {code} {EXPR_NAME[code]:9s} {n}")

    # 80/20 stratified split, fixed seed. Stratify keeps all 7 classes present
    # in the tiny (43-image) test set; test is used as validation during
    # training, exactly like the Kaggle reference.
    train_keys, test_keys = train_test_split(
        table["key"].tolist(), test_size=0.20, random_state=42,
        stratify=table["label"].tolist())
    split = {"train": sorted(train_keys), "test": sorted(test_keys),
             "test_size": 0.20, "random_state": 42, "stratified": True}
    split_path = os.path.join(HERE, "experiments", "split_8020.json")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    print(f"\nsplit 80/20: {len(train_keys)} train / {len(test_keys)} test "
          f"-> {split_path}")


if __name__ == "__main__":
    build()
