"""JAFFE experiments: single-stage or two-stage fine-tuning, 5-fold CV, early
stopping, six metrics (accuracy, precision, recall, F1, specificity, AUC).

- single : all layers trainable, one LR, up to max_epochs, early stopping.
- two_stage : Stage 1 freezes the backbone and trains only the head for
  stage1_epochs; Stage 2 unfreezes the selected deeper layers (layer3+layer4+fc)
  and trains for up to stage2_epochs with early stopping.

Per epoch/fold logging, per-fold validation summary, fold-ensemble TEST with ROC
curve and confusion matrix — same logging discipline as the other JAFFE runs.
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
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import JaffeDataset, preload
from models import build_model
from metrics_roc import plot_roc
from metrics_full import all_metrics, METRIC_KEYS
from train_cv import rows_for, predict, _plot_confusion, CLASSES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_trainable(model, mode, layers=None):
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "head_only":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc")
    elif mode == "unfreeze":
        for name, p in model.named_parameters():
            p.requires_grad = any(name.startswith(prefix) for prefix in layers)
    else:
        raise ValueError(mode)


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_stage(model, train_loader, val_samples, arrays, cfg, device, *, epochs,
              lr, stage, fold, es_patience, history, log, gstep):
    from metrics_roc import macro_auc
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr,
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    best_acc, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
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
        val_acc = float((val_probs.argmax(1) == val_labels).mean())
        val_auc = macro_auc(val_labels, val_probs, len(CLASSES))
        history.append({"fold": fold, "stage": stage, "epoch": gstep,
                        "train_loss": loss_sum / max(1, total),
                        "train_acc": correct / max(1, total),
                        "val_loss": val_loss, "val_acc": val_acc,
                        "val_auc": val_auc, "lr": lr})
        gstep += 1
        if es_patience > 0:
            if val_acc > best_acc + 1e-6:
                best_acc, best_state, bad = val_acc, _snapshot(model), 0
            else:
                bad += 1
                if bad >= es_patience:
                    log(f"    fold{fold} {stage}: early stop @gstep{gstep-1} "
                        f"(best val_acc={best_acc:.3f})")
                    break
    if es_patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
    return gstep


def train_fold(fold, train_samples, val_samples, arrays, cfg, device, history, log):
    torch.manual_seed(cfg["seed"] + fold)
    np.random.seed(cfg["seed"] + fold)
    model = build_model(cfg, len(CLASSES)).to(device)
    loader = DataLoader(JaffeDataset(train_samples, arrays, cfg, train=True),
                        batch_size=cfg["batch_size"], shuffle=True)
    gstep = 0
    if cfg["strategy"] == "single":
        set_trainable(model, "all")
        gstep = run_stage(model, loader, val_samples, arrays, cfg, device,
                          epochs=cfg["max_epochs"], lr=cfg["lr"], stage="single",
                          fold=fold, es_patience=cfg["es_patience"],
                          history=history, log=log, gstep=gstep)
    elif cfg["strategy"] == "two_stage":
        set_trainable(model, "head_only")
        gstep = run_stage(model, loader, val_samples, arrays, cfg, device,
                          epochs=cfg["stage1_epochs"], lr=cfg["stage1_lr"],
                          stage="s1_head", fold=fold, es_patience=0,
                          history=history, log=log, gstep=gstep)
        set_trainable(model, "unfreeze", cfg["stage2_unfreeze"])
        gstep = run_stage(model, loader, val_samples, arrays, cfg, device,
                          epochs=cfg["stage2_epochs"], lr=cfg["stage2_lr"],
                          stage="s2_deep", fold=fold, es_patience=cfg["es_patience"],
                          history=history, log=log, gstep=gstep)
    else:
        raise ValueError(cfg["strategy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)
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
    log(f"=== JAFFE EXP {name} ({cfg['strategy']}) ===")
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
    tta = int(cfg.get("tta", 4))

    history, fold_summaries, test_probs_per_fold = [], [], []
    for fold, val_keys in enumerate(folds):
        val_set = set(val_keys)
        train_samples = [r for r in rows_for(index, dev_keys)
                         if r["key"] not in val_set]
        val_samples = rows_for(index, val_keys)
        log(f"\n[fold {fold}] latih {len(train_samples)} / validasi {len(val_samples)}")
        model = train_fold(fold, train_samples, val_samples, arrays, cfg, device,
                           history, log)
        val_probs, _, val_labels = predict(model, val_samples, arrays, cfg,
                                            device, tta=tta)
        m = all_metrics(val_labels, val_probs, len(CLASSES))
        m["fold"] = fold
        fold_summaries.append(m)
        log(f"  [fold {fold}] " + " ".join(
            f"{k}={m[k]:.4f}" for k in METRIC_KEYS))
        tprobs, _, test_labels = predict(model, test_samples, arrays, cfg,
                                         device, tta=tta)
        test_probs_per_fold.append(tprobs)
        del model
        torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_summaries)
    log("\n=== VALIDASI (mean +- sd antar fold) ===")
    val_stats = {}
    for k in METRIC_KEYS:
        val_stats[k] = (float(fold_df[k].mean()), float(fold_df[k].std()))
        log(f"  {k:12s} {val_stats[k][0]:.4f} +- {val_stats[k][1]:.4f}")

    # TEST: ensemble of fold models.
    test_probs = np.mean(test_probs_per_fold, axis=0)
    tm = all_metrics(test_labels, test_probs, len(CLASSES))
    cm = confusion_matrix(test_labels, test_probs.argmax(1),
                          labels=list(range(len(CLASSES))))
    log("\n=== TEST (ensemble 5 fold) ===")
    log("  " + " ".join(f"{k}={tm[k]:.4f}" for k in METRIC_KEYS))

    pd.DataFrame(history).to_json(
        os.path.join(out_dir, "epoch_history.jsonl"), orient="records", lines=True)
    fold_df.to_csv(os.path.join(out_dir, "fold_summary.csv"), index=False)
    np.savez(os.path.join(out_dir, "test_probs.npz"),
             keys=np.asarray([s["key"] for s in test_samples]),
             label=test_labels, probs=test_probs.astype(np.float32))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(out_dir, "confusion_matrix_test.csv"))
    _plot_confusion(cm, out_dir)
    plot_roc(test_labels, test_probs, CLASSES,
             os.path.join(out_dir, "roc_test.png"), f"ROC TEST — {name}")
    _plot_cv_curve(history, out_dir)

    summary = {
        "name": name, "complete": True, "time": datetime.now().isoformat(),
        "strategy": cfg["strategy"], "seed": cfg["seed"],
        "elapsed_sec": time.time() - start, "n_dev": len(dev_keys),
        "n_test": len(test_keys), "n_folds": len(folds),
        "num_classes": len(CLASSES),
        "val_mean": {k: val_stats[k][0] for k in METRIC_KEYS},
        "val_std": {k: val_stats[k][1] for k in METRIC_KEYS},
        "test": {k: tm[k] for k in METRIC_KEYS},
        "test_per_class_auc": tm["per_class_auc"],
        "per_fold": fold_summaries,
        "confusion_matrix_test": cm.tolist(), "config": cfg,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)

    ledger = os.path.join(HERE, "experiments", "results_experiments.csv")
    row = {"name": name, "strategy": cfg["strategy"]}
    row.update({f"val_{k}": val_stats[k][0] for k in METRIC_KEYS})
    row.update({f"test_{k}": tm[k] for k in METRIC_KEYS})
    row["elapsed_sec"] = round(summary["elapsed_sec"], 1)
    pd.DataFrame([row]).to_csv(
        ledger, mode="a", header=not os.path.exists(ledger), index=False)
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
        ax[0].plot(range(len(hf)), hf["val_acc"], alpha=0.7, label=f"fold {fold}")
        ax[1].plot(range(len(hf)), hf["val_auc"], alpha=0.7, label=f"fold {fold}")
    ax[0].set_title("val accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8)
    ax[1].set_title("val AUC"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cv_val_curve.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
