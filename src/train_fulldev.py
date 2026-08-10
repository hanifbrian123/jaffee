"""Train ONE model on the full DEV set (no CV split), evaluate on TEST.

Complements the 5-fold experiment: instead of ensembling five models each
trained on ~136 images, this trains a single model on all 170 DEV images and
tests it once. TEST is monitored every epoch for a training curve but is never
used for selection — the reported number is the final epoch, evaluated with TTA.
Same ROC/AUC, confusion matrix and ROC curve as the CV run.
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
from dataset import JaffeDataset, preload
from models import build_model
from metrics_roc import macro_auc, per_class_auc, plot_roc
from train_cv import rows_for, predict, _plot_confusion, CLASSES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)
    seed = int(cfg["seed"])
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
    log(f"=== JAFFE FULL-DEV {name} ===")
    log(f"time: {datetime.now().isoformat()} | seed: {seed}")
    log(f"device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")
    log(f"platform: {platform.platform()} | torch {torch.__version__}")

    index = pd.read_csv(os.path.join(HERE, "index.csv"))
    with open(os.path.join(HERE, "experiments", "split_dev_test.json"),
              encoding="utf-8") as handle:
        split = json.load(handle)
    train_samples = rows_for(index, split["dev"])
    test_samples = rows_for(index, split["test"])
    arrays = preload(train_samples + test_samples, cfg["base_size"])
    log(f"latih seluruh DEV {len(train_samples)} / uji TEST {len(test_samples)}")

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
        # TEST monitored each epoch (curve only, not used for selection).
        test_probs, test_loss, test_labels = predict(model, test_samples, arrays,
                                                      cfg, device, tta=1)
        test_acc = float(accuracy_score(test_labels, test_probs.argmax(1)))
        test_auc = macro_auc(test_labels, test_probs, len(CLASSES))
        history.append({"epoch": epoch, "train_loss": loss_sum / max(1, total),
                        "train_acc": correct / max(1, total),
                        "test_loss": test_loss, "test_acc": test_acc,
                        "test_auc": test_auc,
                        "lr": optimizer.param_groups[0]["lr"]})
        log(f"  ep{epoch:02d} train_acc={correct/max(1,total):.3f} "
            f"test_acc={test_acc:.3f} test_auc={test_auc:.3f}")

    # Final evaluation with TTA.
    tta = int(cfg.get("tta", 4))
    test_probs, _, test_labels = predict(model, test_samples, arrays, cfg,
                                         device, tta=tta)
    preds = test_probs.argmax(1)
    acc = float(accuracy_score(test_labels, preds))
    macro_f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))
    uar = float(recall_score(test_labels, preds, average="macro", zero_division=0))
    auc = macro_auc(test_labels, test_probs, len(CLASSES))
    auc_pc = per_class_auc(test_labels, test_probs, len(CLASSES))
    cm = confusion_matrix(test_labels, preds, labels=list(range(len(CLASSES))))
    log(f"\n=== TEST (model full-dev) ===")
    log(f"  acc={acc:.4f} macroF1={macro_f1:.4f} uar={uar:.4f} macroAUC={auc:.4f}")

    pd.DataFrame(history).to_json(
        os.path.join(out_dir, "epoch_history.jsonl"), orient="records", lines=True)
    np.savez(os.path.join(out_dir, "test_probs.npz"),
             keys=np.asarray([s["key"] for s in test_samples]),
             label=test_labels, probs=test_probs.astype(np.float32))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(out_dir, "confusion_matrix_test.csv"))
    _plot_confusion(cm, out_dir)
    plot_roc(test_labels, test_probs, CLASSES,
             os.path.join(out_dir, "roc_test.png"), f"ROC TEST — {name}")
    _plot_train_curve(history, out_dir)

    summary = {
        "name": name, "complete": True, "time": datetime.now().isoformat(),
        "seed": seed, "elapsed_sec": time.time() - start,
        "n_train": len(train_samples), "n_test": len(test_samples),
        "num_classes": len(CLASSES), "test_ACC": acc, "test_macroF1": macro_f1,
        "test_UAR": uar, "test_macroAUC": auc, "test_per_class_auc": auc_pc,
        "confusion_matrix_test": cm.tolist(), "config": cfg,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)
    log(f"ringkasan -> {out_dir}")
    log_file.close()


def _plot_train_curve(history, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_acc"], label="train")
    ax[0].plot(h["epoch"], h["test_acc"], label="test (monitoring)")
    ax[0].set_title("accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["epoch"], h["test_auc"], color="green")
    ax[1].set_title("test AUC (monitoring)"); ax[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
