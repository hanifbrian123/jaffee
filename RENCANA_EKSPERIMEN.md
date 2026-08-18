# Rencana Eksperimen JAFFE — sweep LR & Two-Stage fine-tuning

Enam konfigurasi, semua ResNet-34 + 5-Fold CV + early stopping (patience 10),
dengan 6 metrik: **Accuracy, Precision, Recall, F1, Specificity, AUC**.

## Tabel eksperimen

### A. Single-stage (full-model fine-tuning) — sweep learning rate

| # | Model | Validation | Learning Rate | Max Epoch | Dropout | Early Stopping | Strategy |
|---|---|---|---|---|---|---|---|
| exp1 | ResNet-34 | 5-Fold CV | **0,001** | 100 | 0,5 | 10 | Full-model fine-tuning |
| exp2 | ResNet-34 | 5-Fold CV | **0,0001** | 100 | 0,5 | 10 | Full-model fine-tuning |
| exp3 | ResNet-34 | 5-Fold CV | **0,00001** | 100 | 0,5 | 10 | Full-model fine-tuning |

### B. Two-Stage fine-tuning — sweep Stage-2 learning rate

| # | Model | Validation | Stage 1 LR | Stage 2 LR | Epoch (S1+S2) | Dropout | Early Stopping | Strategy |
|---|---|---|---|---|---|---|---|---|
| exp4 | Two-Stage ResNet-34 | 5-Fold CV | 0,001 | **0,001** | 30 + 70 | 0,5 | 10 | S1: freeze backbone; S2: unfreeze deeper layers |
| exp5 | Two-Stage ResNet-34 | 5-Fold CV | 0,001 | **0,0001** | 30 + 70 | 0,5 | 10 | S1: freeze backbone; S2: unfreeze deeper layers |
| exp6 | Two-Stage ResNet-34 | 5-Fold CV | 0,001 | **0,00001** | 30 + 70 | 0,5 | 10 | S1: freeze backbone; S2: unfreeze deeper layers |

## Interpretasi teknis (harap dicek)

| istilah | implementasi saya |
|---|---|
| Full-model fine-tuning | semua lapisan dilatih dari awal (LR seragam) |
| Stage 1: freeze backbone | semua backbone dibekukan; **hanya head (FC) dilatih** 30 epoch @ Stage-1 LR |
| Stage 2: unfreeze deeper layers | buka **layer3 + layer4 + FC** (stem/layer1/layer2 tetap beku), latih ≤70 epoch @ Stage-2 LR |
| Early Stopping 10 | pantau val accuracy tiap epoch; berhenti jika 10 epoch tak membaik; kembalikan bobot terbaik. Untuk two-stage, ES aktif di Stage 2 |
| Validation | 5-Fold CV pada DEV (test dikunci) |

## Protokol & output per eksperimen

| aspek | isinya |
|---|---|
| Split | TEST 43 dikunci + DEV 170 (5-fold, sama untuk semua) |
| Per epoch, per fold | train loss/acc, val loss/acc, val AUC (dicatat) |
| Ringkasan validasi | mean ± sd antar fold untuk **6 metrik** |
| Testing | ensemble 5 fold → **6 metrik + ROC/AUC** |
| Gambar | kurva CV, ROC test, confusion matrix test per eksperimen |

## Metrik (7 kelas, one-vs-rest, makro)

| metrik | rumus |
|---|---|
| Accuracy | benar / total |
| Precision (makro) | rata-rata TP/(TP+FP) per kelas |
| Recall (makro) | rata-rata TP/(TP+FN) per kelas |
| F1 (makro) | rata-rata 2·P·R/(P+R) per kelas |
| Specificity (makro) | rata-rata TN/(TN+FP) per kelas |
| AUC (makro) | rata-rata AUC one-vs-rest per kelas |

## Logging (selengkap sebelumnya)

`epoch_history.jsonl` (fold, stage, epoch, semua metrik), `fold_summary.csv`,
`summary.json`, kurva PNG, prediksi, ledger CSV — semua di `jaffee/experiments/`.

## Output akhir

Laporan perbandingan 6 eksperimen → `jaffee/reports/HASIL_EKSPERIMEN.md`
(tabel: tiap config × validasi mean±sd × test 6 metrik), plus config terbaik
disorot.

## Catatan jujur

Split acak (bukan subject-independent) → akurasi tinggi wajar (kebocoran
identitas). LR 0,00001 kemungkinan terlalu kecil (konvergensi lambat) — hasilnya
tetap dilaporkan apa adanya.
