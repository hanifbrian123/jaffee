"""Regenerate jaffee/reports/HASIL_CV.md from the CV runs (one table)."""
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "experiments")
OUT = os.path.join(HERE, "reports", "HASIL_CV.md")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

RUNS = [
    ("cv_resnet34", "ResNet-34 + TTA (5-fold CV)"),
]


def load(name):
    path = os.path.join(EXP, name, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def num(v, d=4):
    return "—" if v is None else f"{v:.{d}f}".replace(".", ",")


def main():
    lines = ["# Hasil JAFFE — protokol dev/test + 5-fold CV (dengan ROC/AUC)", ""]
    lines.append(f"Dibuat otomatis dari `jaffee/experiments/` pada "
                 f"**{datetime.now().strftime('%d %B %Y, %H:%M')}**. "
                 "Perbarui: `python jaffee/src/make_report_cv.py`")
    lines.append("")
    lines.append("**Protokol:** TEST 20% dikunci; DEV 80% di-5-fold cross-validation. "
                 "Validasi & training dicatat tiap epoch tiap fold; TEST dinilai "
                 "sekali dengan ensemble 5 fold. Split acak stratified → ada "
                 "kebocoran identitas subjek (akurasi tinggi wajar).")
    lines.append("")

    lines.append("## Ringkasan")
    lines.append("")
    lines.append("| Model | Val acc (mean±sd) | Val AUC (mean±sd) | "
                 "Test acc | Test macro-F1 | **Test AUC** |")
    lines.append("|---|---|---|---|---|---|")
    for name, label in RUNS:
        s = load(name)
        if s is None:
            lines.append(f"| {label} | ⏳ | | | | |")
            continue
        lines.append(
            f"| {label} | {num(s['val_acc_mean'])}±{num(s['val_acc_std'],3)} | "
            f"{num(s['val_auc_mean'])}±{num(s['val_auc_std'],3)} | "
            f"**{num(s['test_ACC'])}** | {num(s['test_macroF1'])} | "
            f"**{num(s['test_macroAUC'])}** |")
    lines.append("")

    s = load(RUNS[0][0])
    if s is not None:
        lines.append("## Validasi per fold")
        lines.append("")
        lines.append("| fold | val acc | val macro-F1 | val AUC |")
        lines.append("|---|---|---|---|")
        for f in s["per_fold"]:
            lines.append(f"| {f['fold']} | {num(f['val_acc'])} | "
                         f"{num(f['val_f1'])} | {num(f['val_auc'])} |")
        lines.append("")

        lines.append("## Test — AUC per kelas (ensemble 5 fold)")
        lines.append("")
        lines.append("| kelas | AUC |")
        lines.append("|---|---|")
        for cls, auc in zip(CLASSES, s["test_per_class_auc"]):
            lines.append(f"| {cls} | {num(auc)} |")
        lines.append("")

        lines.append("## Gambar (di folder experiments/" + RUNS[0][0] + "/)")
        lines.append("")
        lines.append("- `cv_val_curve.png` — accuracy & AUC validasi per fold, per epoch")
        lines.append("- `roc_val_fold{0..4}.png` — kurva ROC validasi tiap fold")
        lines.append("- `roc_test.png` — kurva ROC pada TEST (ensemble)")
        lines.append("- `confusion_matrix_test.png` — confusion matrix TEST")
        lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"ditulis: {OUT}")


if __name__ == "__main__":
    main()
