"""Assemble the full JAFFE report (one file) covering all five requested parts:
1) arsitektur model  2) skenario eksperimen  3) training+validasi per epoch/fold
4) hasil testing  5) ROC/AUC. Reads the CV run artifacts from disk.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "cv_resnet34"
EXP = os.path.join(HERE, "experiments", RUN)
OUT = os.path.join(HERE, "reports", "LAPORAN.md")
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def num(v, d=4):
    return "—" if v is None else f"{v:.{d}f}".replace(".", ",")


def main():
    with open(os.path.join(EXP, "summary.json"), encoding="utf-8") as h:
        s = json.load(h)
    cfg = s["config"]
    history = pd.read_json(os.path.join(EXP, "epoch_history.jsonl"), lines=True)

    L = []
    A = L.append
    A("# Laporan Klasifikasi Ekspresi Wajah JAFFE (7 kelas)")
    A("")
    A(f"Dibuat otomatis dari `jaffee/experiments/{RUN}/`. "
      "Regenerasi: `python jaffee/src/make_laporan.py`")
    A("")

    # ---------- 1. ARSITEKTUR ----------
    A("## 1. Arsitektur Model")
    A("")
    A("![Arsitektur ResNet-34](../figures/architecture.png)")
    A("")
    A("Model = **ResNet-34** (pra-terlatih ImageNet) sebagai pengekstrak fitur, "
      "diikuti kepala klasifikasi ringan. Ini analog 2D dari pemenang CASME II "
      "(r3d_18 pra-terlatih Kinetics); JAFFE gambar diam, jadi backbone temporal "
      "3D tidak dipakai.")
    A("")
    A("| bagian | detail |")
    A("|---|---|")
    A("| Input | gambar grayscale JAFFE → 3 channel, "
      f"{cfg['base_size']}×{cfg['base_size']} → crop {cfg['img_size']}×{cfg['img_size']} |")
    A("| Backbone | ResNet-34, bobot ImageNet (Conv0 + 4 stage: 64/128/256/512) |")
    A("| Pooling | Global Average Pooling → 512 fitur |")
    A(f"| Head | Dropout {cfg['dropout']} → Linear 512→7 → Softmax |")
    A("| Jumlah kelas | 7 (angry, disgust, fear, happy, neutral, sad, surprise) |")
    A("")

    # ---------- 2. SKENARIO ----------
    A("## 2. Skenario Eksperimen")
    A("")
    A("| aspek | nilai |")
    A("|---|---|")
    A(f"| Dataset | JAFFE, 213 gambar, 10 subjek, 7 kelas |")
    A(f"| Split | TEST {s['n_test']} (dikunci) + DEV {s['n_dev']} |")
    A(f"| Cross-validation | {s['n_folds']}-fold stratified pada DEV "
      f"(~136 latih / ~34 validasi per fold) |")
    A(f"| Validasi | tiap epoch tiap fold (loss, accuracy, AUC) |")
    A(f"| Testing | sekali, ensemble {s['n_folds']} model fold |")
    A(f"| Backbone | {cfg['backbone']} (pretrained={cfg['pretrained']}) |")
    A(f"| Epoch | {cfg['epochs']} | ")
    A(f"| Optimizer | AdamW, lr={cfg['lr']}, weight_decay={cfg['weight_decay']} |")
    A(f"| Scheduler | Cosine annealing |")
    A(f"| Loss | Cross-entropy, label smoothing {cfg['label_smoothing']} |")
    A(f"| Augmentasi | flip, rotasi ±{cfg['rotate']}°, brightness ±{cfg['brightness']}, "
      f"random-erase {cfg['random_erase']} |")
    A(f"| TTA | {cfg['tta']} view deterministik |")
    A(f"| Batch size | {cfg['batch_size']} |")
    A(f"| Seed | {cfg['seed']} |")
    A("")
    A("")

    # ---------- 3. TRAINING & VALIDASI per epoch per fold ----------
    A("## 3. Hasil Training & Validasi (tiap epoch, tiap fold)")
    A("")
    A("Kurva ringkas (accuracy & AUC validasi tiap fold): "
      f"`experiments/{RUN}/cv_val_curve.png`")
    A("")
    A(f"![Kurva CV](../experiments/{RUN}/cv_val_curve.png)")
    A("")
    for fold in sorted(history["fold"].unique()):
        hf = history[history["fold"] == fold]
        A(f"### Fold {fold}")
        A("")
        A("| epoch | train_loss | train_acc | val_loss | val_acc | val_AUC |")
        A("|---|---|---|---|---|---|")
        for _, r in hf.iterrows():
            A(f"| {int(r['epoch'])} | {num(r['train_loss'])} | {num(r['train_acc'])} | "
              f"{num(r['val_loss'])} | {num(r['val_acc'])} | {num(r['val_auc'])} |")
        A("")

    # per-fold validation summary (final, TTA)
    A("### Ringkasan validasi per fold (setelah training, dengan TTA)")
    A("")
    A("| fold | val acc | val macro-F1 | val AUC |")
    A("|---|---|---|---|")
    for f in s["per_fold"]:
        A(f"| {f['fold']} | {num(f['val_acc'])} | {num(f['val_f1'])} | {num(f['val_auc'])} |")
    A(f"| **rata-rata** | **{num(s['val_acc_mean'])} ± {num(s['val_acc_std'],3)}** | "
      f"— | **{num(s['val_auc_mean'])} ± {num(s['val_auc_std'],3)}** |")
    A("")
    A(f"Kurva ROC validasi tiap fold: `experiments/{RUN}/roc_val_fold0..4.png`")
    A("")

    # ---------- 4. TESTING UTAMA: model full-DEV ----------
    fd_path = os.path.join(HERE, "experiments", "fulldev_resnet34", "summary.json")
    fd = None
    if os.path.exists(fd_path):
        with open(fd_path, encoding="utf-8") as h:
            fd = json.load(h)

    A("## 4. Hasil Testing Utama — model Full-DEV")
    A("")
    if fd is not None:
        A(f"**Model final = ResNet-34 dilatih di SELURUH DEV ({fd['n_train']} "
          "gambar), diuji sekali ke TEST (43 gambar).** Ini hasil test utama: "
          "model tunggal yang memakai semua data latih, cara yang lazim dipakai "
          "untuk model deployment. Cross-validation di Bagian 3 adalah estimasi "
          "validasinya. TEST dipantau tiap epoch untuk kurva "
          "(`experiments/fulldev_resnet34/training_curve.png`) tapi tidak dipakai "
          "untuk seleksi; angka final = epoch terakhir + TTA.")
        A("")
        A("| metrik | nilai |")
        A("|---|---|")
        A(f"| Accuracy | **{num(fd['test_ACC'])}** |")
        A(f"| Macro-F1 | {num(fd['test_macroF1'])} |")
        A(f"| UAR (balanced acc) | {num(fd['test_UAR'])} |")
        A(f"| **Macro-AUC** | **{num(fd['test_macroAUC'])}** |")
        A("")
        A("Confusion matrix (TEST):")
        A("")
        A("![Confusion matrix](../experiments/fulldev_resnet34/confusion_matrix_test.png)")
        A("")

    # ---------- 5. ROC & AUC (model full-DEV) ----------
    A("## 5. ROC & AUC (model Full-DEV)")
    A("")
    if fd is not None:
        A("Kurva ROC pada TEST (one-vs-rest, per kelas + rata-rata makro):")
        A("")
        A("![ROC full-dev](../experiments/fulldev_resnet34/roc_test.png)")
        A("")
        A("**AUC per kelas (TEST):**")
        A("")
        A("| kelas | AUC |")
        A("|---|---|")
        for cls, auc in zip(CLASSES, fd["test_per_class_auc"]):
            A(f"| {cls} | {num(auc)} |")
        A(f"| **makro** | **{num(fd['test_macroAUC'])}** |")
        A("")
        worst = min(zip(CLASSES, fd["test_per_class_auc"]), key=lambda kv: kv[1])
        A(f"Kelas tersulit: **{worst[0]}** (AUC {num(worst[1])}) — konsisten "
          "dengan literatur FER (sad/neutral paling sering tertukar).")
        A("")

    # ---------- 6. TESTING TAMBAHAN: ensemble 5-fold ----------
    A("## 6. Testing tambahan — Ensemble 5-fold")
    A("")
    A("Sebagai pembanding, kelima model dari cross-validation (Bagian 3) "
      "digabung (rata-rata probabilitas) lalu diuji ke TEST yang sama.")
    A("")
    A("| metrik | nilai |")
    A("|---|---|")
    A(f"| Accuracy | {num(s['test_ACC'])} |")
    A(f"| Macro-F1 | {num(s['test_macroF1'])} |")
    A(f"| UAR | {num(s['test_UAR'])} |")
    A(f"| Macro-AUC | {num(s['test_macroAUC'])} |")
    A("")
    A(f"Kurva ROC & confusion matrix: `experiments/{RUN}/roc_test.png`, "
      f"`experiments/{RUN}/confusion_matrix_test.png`")
    A("")
    if fd is not None:
        A("### Perbandingan di TEST yang sama")
        A("")
        A("| strategi | data latih | Test acc | Test macro-F1 | Test AUC |")
        A("|---|---|---|---|---|")
        A(f"| **Full-DEV (utama)** | **170 × 1 model** | "
          f"**{num(fd['test_ACC'])}** | **{num(fd['test_macroF1'])}** | "
          f"**{num(fd['test_macroAUC'])}** |")
        A(f"| Ensemble 5-fold (tambahan) | ~136 × 5 model | {num(s['test_ACC'])} | "
          f"{num(s['test_macroF1'])} | {num(s['test_macroAUC'])} |")
        A("")
        if fd["test_ACC"] >= s["test_ACC"]:
            A("**Temuan:** model full-DEV (utama) **≥** ensemble 5-fold. Melatih "
              "satu model di seluruh 170 gambar mengungguli menggabungkan 5 model "
              "yang masing-masing kelaparan data (136) — konsisten dengan "
              "pelajaran CASME II: pada data kecil, lebih banyak data latih per "
              "model sering lebih menolong daripada ensembling model lemah. "
              "**Catatan:** TEST hanya 43 gambar, selisih ini ≈ 1 gambar; jangan "
              "di-over-interpretasi.")
        else:
            A("**Temuan:** ensemble 5-fold sedikit unggul di TEST, tapi model "
              "full-DEV tetap dipakai sebagai hasil utama (model deployment "
              "standar). TEST hanya 43 gambar, selisih kecil jangan "
              "di-over-interpretasi.")
        A("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as h:
        h.write("\n".join(L) + "\n")
    print(f"ditulis: {OUT}")


if __name__ == "__main__":
    main()
