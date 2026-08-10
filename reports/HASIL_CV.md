# Hasil JAFFE — protokol dev/test + 5-fold CV (dengan ROC/AUC)

Dibuat otomatis dari `jaffee/experiments/` pada **08 August 2026, 11:04**. Perbarui: `python jaffee/src/make_report_cv.py`

**Protokol:** TEST 20% dikunci; DEV 80% di-5-fold cross-validation. Validasi & training dicatat tiap epoch tiap fold; TEST dinilai sekali dengan ensemble 5 fold. Split acak stratified.

## Ringkasan

| Model | Val acc (mean±sd) | Val AUC (mean±sd) | Test acc | Test macro-F1 | **Test AUC** |
|---|---|---|---|---|---|
| ResNet-34 + TTA (5-fold CV) | 0,9176±0,053 | 0,9952±0,005 | **0,9302** | 0,9209 | **0,9833** |

## Validasi per fold

| fold | val acc | val macro-F1 | val AUC |
|---|---|---|---|
| 0 | 0,8824 | 0,8829 | 0,9970 |
| 1 | 0,8824 | 0,8846 | 0,9892 |
| 2 | 1,0000 | 1,0000 | 1,0000 |
| 3 | 0,8824 | 0,8766 | 0,9907 |
| 4 | 0,9412 | 0,9394 | 0,9990 |

## Test — AUC per kelas (ensemble 5 fold)

| kelas | AUC |
|---|---|
| angry | 1,0000 |
| disgust | 1,0000 |
| fear | 1,0000 |
| happy | 0,9955 |
| neutral | 1,0000 |
| sad | 0,8874 |
| surprise | 1,0000 |

## Gambar (di folder experiments/cv_resnet34/)

- `cv_val_curve.png` — accuracy & AUC validasi per fold, per epoch
- `roc_val_fold{0..4}.png` — kurva ROC validasi tiap fold
- `roc_test.png` — kurva ROC pada TEST (ensemble)
- `confusion_matrix_test.png` — confusion matrix TEST

