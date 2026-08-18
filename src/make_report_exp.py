"""Comparison report across the 6 JAFFE experiments -> reports/HASIL_EKSPERIMEN.md."""
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "experiments")
OUT = os.path.join(HERE, "reports", "HASIL_EKSPERIMEN.md")
KEYS = ["accuracy", "precision", "recall", "f1", "specificity", "auc"]
LABELS = {"accuracy": "Acc", "precision": "Prec", "recall": "Rec",
          "f1": "F1", "specificity": "Spec", "auc": "AUC"}

RUNS = [
    ("exp1", "Single, LR 0,001", "exp1_single_lr1e3"),
    ("exp2", "Single, LR 0,0001", "exp2_single_lr1e4"),
    ("exp3", "Single, LR 0,00001", "exp3_single_lr1e5"),
    ("exp4", "Two-Stage, S2 LR 0,001", "exp4_twostage_s2lr1e3"),
    ("exp5", "Two-Stage, S2 LR 0,0001", "exp5_twostage_s2lr1e4"),
    ("exp6", "Two-Stage, S2 LR 0,00001", "exp6_twostage_s2lr1e5"),
]


def load(name):
    path = os.path.join(EXP, name, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as h:
        return json.load(h)


def num(v, d=4):
    return "—" if v is None else f"{v:.{d}f}".replace(".", ",")


def main():
    L = ["# Hasil Eksperimen JAFFE — Sweep LR & Two-Stage Fine-Tuning", ""]
    L.append(f"Dibuat otomatis dari `jaffee/experiments/` pada "
             f"**{datetime.now().strftime('%d %B %Y, %H:%M')}**. "
             "Perbarui: `python jaffee/src/make_report_exp.py`")
    L.append("")
    L.append("Semua: ResNet-34, 5-Fold CV, early stopping (patience 10), "
             "dropout 0,5. Metrik = macro one-vs-rest. Test = ensemble 5 fold. "
             "⚠️ Split acak (kebocoran identitas) → akurasi tinggi wajar.")
    L.append("")

    # ---- TEST table (6 metrics) ----
    L.append("## Hasil TEST (ensemble 5 fold)")
    L.append("")
    L.append("| # | Konfigurasi | " + " | ".join(LABELS[k] for k in KEYS) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(KEYS)) + "|")
    best = None
    for tag, label, name in RUNS:
        s = load(name)
        if s is None:
            L.append(f"| {tag} | {label} | " + " | ".join(["⏳"] * len(KEYS)) + " |")
            continue
        L.append(f"| {tag} | {label} | " +
                 " | ".join(num(s["test"][k]) for k in KEYS) + " |")
        if best is None or s["test"]["accuracy"] > best[1]:
            best = (label, s["test"]["accuracy"], name)
    L.append("")
    if best:
        L.append(f"**Konfigurasi terbaik (TEST acc): {best[0]} — {num(best[1])}**")
        L.append("")

    # ---- VALIDATION table (mean +- sd) ----
    L.append("## Hasil VALIDASI (5-fold, mean ± sd)")
    L.append("")
    L.append("| # | Konfigurasi | " + " | ".join(LABELS[k] for k in KEYS) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(KEYS)) + "|")
    for tag, label, name in RUNS:
        s = load(name)
        if s is None:
            L.append(f"| {tag} | {label} | " + " | ".join(["⏳"] * len(KEYS)) + " |")
            continue
        cells = [f"{num(s['val_mean'][k],3)}±{num(s['val_std'][k],3)}" for k in KEYS]
        L.append(f"| {tag} | {label} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- per-experiment detail ----
    L.append("## Detail per eksperimen")
    L.append("")
    for tag, label, name in RUNS:
        s = load(name)
        if s is None:
            continue
        cfg = s["config"]
        L.append(f"### {tag} — {label}")
        L.append("")
        if s["strategy"] == "single":
            L.append(f"- Strategy: full-model fine-tuning, LR {cfg['lr']}, "
                     f"max {cfg['max_epochs']} epoch, early stopping {cfg['es_patience']}")
        else:
            L.append(f"- Strategy: two-stage. S1 freeze backbone (head only) "
                     f"{cfg['stage1_epochs']} ep @ {cfg['stage1_lr']}; "
                     f"S2 unfreeze {cfg['stage2_unfreeze']} ≤{cfg['stage2_epochs']} ep "
                     f"@ {cfg['stage2_lr']}, early stopping {cfg['es_patience']}")
        L.append(f"- Gambar: `experiments/{name}/cv_val_curve.png`, "
                 f"`roc_test.png`, `confusion_matrix_test.png`")
        L.append(f"- AUC per kelas (TEST): " +
                 ", ".join(f"{c} {num(a,3)}" for c, a in
                           zip(["angry", "disgust", "fear", "happy", "neutral",
                                "sad", "surprise"], s["test_per_class_auc"])))
        L.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as h:
        h.write("\n".join(L) + "\n")
    print(f"ditulis: {OUT}")


if __name__ == "__main__":
    main()
