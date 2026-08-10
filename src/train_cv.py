"""5-fold CV on DEV + ensemble evaluation on the held-out TEST set.

Per fold, per epoch: train loss/acc, val loss/acc, and val macro-AUC are logged.
After each fold finishes, its ROC curve on its own validation set is saved and
its probabilities on TEST are stored. The 5 fold models are then averaged into a
single ensemble whose TEST accuracy, macro-F1, macro-AUC, per-class AUC,
confusion matrix and ROC curve are reported.

Reuses dataset.py / models.py so preprocessing, augmentation and TTA match every
other JAFFE run.
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
from dataset import JaffeDataset, preload, tta_views
from models import build_model
from metrics_roc import macro_auc, per_class_auc, plot_roc

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def rows_for(index, keys):
    keyset = set(keys)
    return [row for _, row in index.iterrows() if row["key"] in keyset]


@torch.no_grad()
def predict(model, samples, arrays, cfg, device, tta=1):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    labels = np.asarray([int(s["label"]) for s in samples])
    probs_sum, loss_total = None, 0.0
    for view in tta_views(tta):
        ds = JaffeDataset(samples, arrays, cfg, train=False, view=view)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)
        batch = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).float()
            loss_total += ce(out, y).item()
            batch.append(torch.softmax(out, 1).cpu().numpy())
        p = np.concatenate(batch, 0)
        probs_sum = p if probs_sum is None else probs_sum + p
    n_views = len(tta_views(tta))
    return probs_sum / n_views, loss_total / (len(samples) * n_views), labels


def train_one_fold(fold, train_samples, val_samples, arrays, cfg, device, log):
    torch.manual_seed(cfg["seed"] + fold)
    np.random.seed(cfg["seed"] + fold)
    model = build_model(cfg, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = int(cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    loader = DataLoader(JaffeDataset(train_samples, arrays, cfg, train=True),
                        batch_size=cfg["batch_size"], shuffle=True)

    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()
        val_probs, val_loss, val_labels = predict(model, val_samples, arrays,
                                                   cfg, device, tta=1)
        val_acc = float(accuracy_score(val_labels, val_probs.argmax(1)))
        val_auc = macro_auc(val_labels, val_probs, len(CLASSES))
        history.append({"fold": fold, "epoch": epoch,
                        "train_loss": loss_sum / max(1, total),
                        "train_acc": correct / max(1, total),
                        "val_loss": val_loss, "val_acc": val_acc,
                        "val_auc": val_auc,
                        "lr": optimizer.param_groups[0]["lr"]})
        log(f"    fold{fold} ep{epoch:02d} "
            f"train_acc={correct/max(1,total):.3f} "
            f"val_acc={val_acc:.3f} val_auc={val_auc:.3f}")
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
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
    log(f"=== JAFFE CV {name} ===")
    log(f"time: {datetime.now().isoformat()} | seed: {cfg['seed']}")
    log(f"device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")
    log(f"platform: {platform.platform()} | torch {torch.__version__}")

    index = pd.read_csv(os.path.join(HERE, "index.csv"))
    with open(os.path.join(HERE, "experiments", "split_dev_test.json"),
              encoding="utf-8") as handle:
        split = json.load(handle)
    dev_keys, test_keys, folds = split["dev"], split["test"], split["folds"]
    test_samples = rows_for(index, test_keys)
    arrays = preload(rows_for(index, dev_keys) + test_samples, cfg["base_size"])
    log(f"DEV {len(dev_keys)} ({len(folds)} fold) | TEST {len(test_keys)} (dikunci)")

    all_history = []
    fold_summaries = []
    test_probs_per_fold = []
    tta = int(cfg.get("tta", 4))
    for fold, val_keys in enumerate(folds):
        val_set = set(val_keys)
        train_samples = [r for r in rows_for(index, dev_keys)
                         if r["key"] not in val_set]
        val_samples = rows_for(index, val_keys)
        log(f"\n[fold {fold}] latih {len(train_samples)} / validasi {len(val_samples)}")
        model, history = train_one_fold(fold, train_samples, val_samples, arrays,
                                        cfg, device, log)
        all_history.extend(history)

        # Validation metrics with full TTA + ROC curve for this fold.
        val_probs, _, val_labels = predict(model, val_samples, arrays, cfg,
                                            device, tta=tta)
        val_acc = float(accuracy_score(val_labels, val_probs.argmax(1)))
        val_f1 = float(f1_score(val_labels, val_probs.argmax(1), average="macro",
                                zero_division=0))
        val_auc = macro_auc(val_labels, val_probs, len(CLASSES))
        fold_summaries.append({"fold": fold, "val_acc": val_acc, "val_f1": val_f1,
                               "val_auc": val_auc})
        plot_roc(val_labels, val_probs, CLASSES,
                 os.path.join(out_dir, f"roc_val_fold{fold}.png"),
                 f"ROC validasi fold {fold} — {name}")
        log(f"  [fold {fold}] val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"val_auc={val_auc:.4f}")

        # This fold's probabilities on the locked TEST set (for the ensemble).
        tprobs, _, test_labels = predict(model, test_samples, arrays, cfg,
                                         device, tta=tta)
        test_probs_per_fold.append(tprobs)
        del model
        torch.cuda.empty_cache()

    # --- aggregate validation across folds ---
    fold_df = pd.DataFrame(fold_summaries)
    log("\n=== VALIDASI (rata-rata antar-fold) ===")
    log(f"  acc {fold_df['val_acc'].mean():.4f} +- {fold_df['val_acc'].std():.4f}")
    log(f"  f1  {fold_df['val_f1'].mean():.4f} +- {fold_df['val_f1'].std():.4f}")
    log(f"  auc {fold_df['val_auc'].mean():.4f} +- {fold_df['val_auc'].std():.4f}")

    # --- TEST: ensemble of the 5 fold models ---
    test_probs = np.mean(test_probs_per_fold, axis=0)
    preds = test_probs.argmax(1)
    test_acc = float(accuracy_score(test_labels, preds))
    test_f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))
    test_uar = float(recall_score(test_labels, preds, average="macro",
                                  zero_division=0))
    test_auc = macro_auc(test_labels, test_probs, len(CLASSES))
    test_auc_pc = per_class_auc(test_labels, test_probs, len(CLASSES))
    cm = confusion_matrix(test_labels, preds, labels=list(range(len(CLASSES))))
    plot_roc(test_labels, test_probs, CLASSES,
             os.path.join(out_dir, "roc_test.png"), f"ROC TEST — {name}")
    log("\n=== TEST (ensemble 5 fold) ===")
    log(f"  acc={test_acc:.4f} macroF1={test_f1:.4f} uar={test_uar:.4f} "
        f"macroAUC={test_auc:.4f}")

    # --- artefak ---
    pd.DataFrame(all_history).to_json(
        os.path.join(out_dir, "epoch_history.jsonl"), orient="records", lines=True)
    fold_df.to_csv(os.path.join(out_dir, "fold_summary.csv"), index=False)
    np.savez(os.path.join(out_dir, "test_probs.npz"),
             keys=np.asarray([s["key"] for s in test_samples]),
             label=test_labels, probs=test_probs.astype(np.float32))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(out_dir, "confusion_matrix_test.csv"))
    _plot_confusion(cm, out_dir)
    _plot_cv_curve(all_history, out_dir)

    summary = {
        "name": name, "complete": True, "time": datetime.now().isoformat(),
        "seed": cfg["seed"], "elapsed_sec": time.time() - start,
        "n_dev": len(dev_keys), "n_test": len(test_keys), "n_folds": len(folds),
        "num_classes": len(CLASSES),
        "val_acc_mean": float(fold_df["val_acc"].mean()),
        "val_acc_std": float(fold_df["val_acc"].std()),
        "val_auc_mean": float(fold_df["val_auc"].mean()),
        "val_auc_std": float(fold_df["val_auc"].std()),
        "test_ACC": test_acc, "test_macroF1": test_f1, "test_UAR": test_uar,
        "test_macroAUC": test_auc,
        "test_per_class_auc": test_auc_pc,
        "per_fold": fold_summaries,
        "confusion_matrix_test": cm.tolist(), "config": cfg,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)

    ledger = os.path.join(HERE, "experiments", "results_cv.csv")
    pd.DataFrame([{
        "name": name, "backbone": cfg["backbone"], "seed": cfg["seed"],
        "val_acc_mean": summary["val_acc_mean"],
        "val_auc_mean": summary["val_auc_mean"],
        "test_ACC": test_acc, "test_macroF1": test_f1, "test_macroAUC": test_auc,
        "elapsed_sec": round(summary["elapsed_sec"], 1),
    }]).to_csv(ledger, mode="a", header=not os.path.exists(ledger), index=False)
    log(f"\nringkasan -> {out_dir}")
    log_file.close()


def _plot_cv_curve(history, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for fold in sorted(h["fold"].unique()):
        hf = h[h["fold"] == fold]
        ax[0].plot(hf["epoch"], hf["val_acc"], alpha=0.7, label=f"fold {fold}")
        ax[1].plot(hf["epoch"], hf["val_auc"], alpha=0.7, label=f"fold {fold}")
    ax[0].set_title("val accuracy per fold"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8)
    ax[1].set_title("val AUC per fold"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cv_val_curve.png"), dpi=110)
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
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("prediksi"); ax.set_ylabel("sebenarnya")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_test.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
