"""Train one JAFFE model with full logging (CASME II discipline).

Protocol matches the Kaggle reference: fixed 80/20 split, the test set is used
as validation during training (the 80 is NOT split again), fixed number of
epochs, and the reported number is the final epoch. Everything is logged so the
run is reproducible and can be ensembled: per-epoch train/val loss+acc,
confusion matrix, per-sample predictions, softmax probs, a training curve, and a
one-line summary appended to a shared CSV.
"""
import argparse
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score)
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import JaffeDataset, preload, prepare, tta_views
from models import build_model

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def make_samples(index, keys):
    keyset = set(keys)
    return [row for _, row in index.iterrows() if row["key"] in keyset]


@torch.no_grad()
def evaluate(model, samples, arrays, cfg, device, tta=1):
    """Return (probs, loss, acc) averaged over deterministic TTA views."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    views = tta_views(tta)
    probs_sum = None
    labels = np.asarray([int(s["label"]) for s in samples])
    loss_total = 0.0
    for view in views:
        ds = JaffeDataset(samples, arrays, cfg, train=False, view=view)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)
        batch_probs, offset = [], 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x).float()
            loss_total += ce(out, y).item()
            batch_probs.append(torch.softmax(out, dim=1).cpu().numpy())
            offset += len(y)
        p = np.concatenate(batch_probs, axis=0)
        probs_sum = p if probs_sum is None else probs_sum + p
    probs = probs_sum / len(views)
    loss = loss_total / (len(samples) * len(views))
    acc = float(accuracy_score(labels, probs.argmax(1)))
    return probs, loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    name = cfg["name"]
    out_dir = os.path.join(HERE, "experiments", name)
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(str(message) + "\n")
        log_file.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    log(f"=== JAFFE {name} ===")
    log(f"time: {datetime.now().isoformat()} | seed: {seed}")
    log(f"device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")
    log(f"platform: {platform.platform()} | torch {torch.__version__}")

    index = pd.read_csv(os.path.join(HERE, "index.csv"))
    with open(os.path.join(HERE, "experiments", "split_8020.json"),
              encoding="utf-8") as handle:
        split = json.load(handle)
    train_samples = make_samples(index, split["train"])
    test_samples = make_samples(index, split["test"])
    log(f"split: {len(train_samples)} train / {len(test_samples)} test "
        f"(test dipakai sebagai validation)")

    arrays = preload(train_samples + test_samples, cfg["base_size"])
    num_classes = len(CLASSES)

    model = build_model(cfg, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = int(cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    train_loader = DataLoader(
        JaffeDataset(train_samples, arrays, cfg, train=True),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

    history = []
    best_acc, best_epoch = 0.0, -1
    for epoch in range(epochs):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()
        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)
        # Test set as validation, monitored every epoch (Kaggle style).
        _, val_loss, val_acc = evaluate(model, test_samples, arrays, cfg, device,
                                        tta=1)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss,
                        "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]})
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
        log(f"  ep{epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
            f" | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Final evaluation with TTA (faithful: reported number is final epoch model).
    tta = int(cfg.get("tta", 1))
    probs, final_loss, _ = evaluate(model, test_samples, arrays, cfg, device,
                                    tta=tta)
    labels = np.asarray([int(s["label"]) for s in test_samples])
    keys = np.asarray([s["key"] for s in test_samples])
    preds = probs.argmax(1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    uar = float(recall_score(labels, preds, average="macro", zero_division=0))
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    per_class_f1 = f1_score(labels, preds, average=None,
                            labels=list(range(num_classes)), zero_division=0)

    log(f"\nFINAL (epoch terakhir, TTA={tta}): "
        f"acc={acc:.4f} macroF1={macro_f1:.4f} uar={uar:.4f}")
    log(f"best val_acc {best_acc:.4f} @ epoch {best_epoch}")

    # --- artefak logging ---
    pd.DataFrame(history).to_json(
        os.path.join(out_dir, "epoch_history.jsonl"), orient="records",
        lines=True)
    np.savez(os.path.join(out_dir, "probs.npz"), keys=keys, label=labels,
             probs=probs.astype(np.float32))
    pd.DataFrame({"key": keys, "label": labels, "pred": preds,
                  "correct": preds == labels}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv"))
    _plot_curves(history, out_dir)
    _plot_confusion(cm, out_dir)

    summary = {
        "name": name, "complete": True, "time": datetime.now().isoformat(),
        "seed": seed, "elapsed_sec": time.time() - start,
        "n_train": len(train_samples), "n_test": len(test_samples),
        "num_classes": num_classes, "ACC": acc, "macroF1": macro_f1, "UAR": uar,
        "best_val_acc": best_acc, "best_epoch": best_epoch,
        "final_val_loss": final_loss,
        "per_class_f1": [float(v) for v in per_class_f1],
        "confusion_matrix": cm.tolist(), "config": cfg,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)

    ledger = os.path.join(HERE, "experiments", "results_jaffe.csv")
    row = {"name": name, "time": summary["time"], "ACC": acc,
           "macroF1": macro_f1, "UAR": uar, "backbone": cfg["backbone"],
           "seed": seed, "epochs": epochs, "tta": tta,
           "elapsed_sec": round(summary["elapsed_sec"], 1)}
    pd.DataFrame([row]).to_csv(
        ledger, mode="a", header=not os.path.exists(ledger), index=False)
    log(f"ringkasan -> {out_dir}")
    log_file.close()


def _plot_curves(history, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train")
    ax[0].plot(h["epoch"], h["val_loss"], label="val (test)")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["epoch"], h["train_acc"], label="train")
    ax[1].plot(h["epoch"], h["val_acc"], label="val (test)")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=110)
    plt.close(fig)


def _plot_confusion(cm, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("prediksi"); ax.set_ylabel("sebenarnya")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
