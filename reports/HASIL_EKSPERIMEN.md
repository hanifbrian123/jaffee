# Hasil Eksperimen JAFFE — Sweep LR & Two-Stage Fine-Tuning

Dibuat otomatis dari `jaffee/experiments/` pada **18 August 2026, 08:57**. Perbarui: `python jaffee/src/make_report_exp.py`

Semua: ResNet-34, 5-Fold CV, early stopping (patience 10), dropout 0,5. Metrik = macro one-vs-rest. Test = ensemble 5 fold. ⚠️ Split acak (kebocoran identitas) → akurasi tinggi wajar.

## Hasil TEST (ensemble 5 fold)

| # | Konfigurasi | Acc | Prec | Rec | F1 | Spec | AUC |
|---|---|---|---|---|---|---|---|
| exp1 | Single, LR 0,001 | 0,8837 | 0,9082 | 0,8810 | 0,8720 | 0,9807 | 0,9794 |
| exp2 | Single, LR 0,0001 | 0,8605 | 0,8713 | 0,8571 | 0,8444 | 0,9766 | 0,9795 |
| exp3 | Single, LR 0,00001 | 0,7907 | 0,7854 | 0,7891 | 0,7828 | 0,9650 | 0,9588 |
| exp4 | Two-Stage, S2 LR 0,001 | 0,8837 | 0,9036 | 0,8810 | 0,8757 | 0,9805 | 0,9723 |
| exp5 | Two-Stage, S2 LR 0,0001 | 0,9070 | 0,9056 | 0,9048 | 0,8984 | 0,9844 | 0,9749 |
| exp6 | Two-Stage, S2 LR 0,00001 | 0,7674 | 0,7957 | 0,7653 | 0,7569 | 0,9613 | 0,9492 |

**Konfigurasi terbaik (TEST acc): Two-Stage, S2 LR 0,0001 — 0,9070**

## Hasil VALIDASI (5-fold, mean ± sd)

| # | Konfigurasi | Acc | Prec | Rec | F1 | Spec | AUC |
|---|---|---|---|---|---|---|---|
| exp1 | Single, LR 0,001 | 0,853±0,069 | 0,895±0,046 | 0,854±0,065 | 0,848±0,068 | 0,976±0,012 | 0,984±0,015 |
| exp2 | Single, LR 0,0001 | 0,906±0,073 | 0,915±0,067 | 0,904±0,075 | 0,905±0,074 | 0,984±0,012 | 0,990±0,010 |
| exp3 | Single, LR 0,00001 | 0,812±0,053 | 0,841±0,044 | 0,809±0,057 | 0,808±0,056 | 0,969±0,009 | 0,967±0,018 |
| exp4 | Two-Stage, S2 LR 0,001 | 0,918±0,120 | 0,932±0,103 | 0,916±0,122 | 0,911±0,135 | 0,986±0,020 | 0,988±0,015 |
| exp5 | Two-Stage, S2 LR 0,0001 | 0,929±0,053 | 0,945±0,036 | 0,927±0,058 | 0,927±0,056 | 0,988±0,009 | 0,993±0,009 |
| exp6 | Two-Stage, S2 LR 0,00001 | 0,776±0,111 | 0,814±0,099 | 0,774±0,113 | 0,770±0,123 | 0,963±0,019 | 0,948±0,041 |

## Detail per eksperimen

### exp1 — Single, LR 0,001

- Strategy: full-model fine-tuning, LR 0.001, max 100 epoch, early stopping 10
- Gambar: `experiments/exp1_single_lr1e3/cv_val_curve.png`, `roc_test.png`, `confusion_matrix_test.png`
- AUC per kelas (TEST): angry 1,000, disgust 1,000, fear 1,000, happy 0,991, neutral 0,991, sad 0,874, surprise 1,000

### exp2 — Single, LR 0,0001

- Strategy: full-model fine-tuning, LR 0.0001, max 100 epoch, early stopping 10
- Gambar: `experiments/exp2_single_lr1e4/cv_val_curve.png`, `roc_test.png`, `confusion_matrix_test.png`
- AUC per kelas (TEST): angry 1,000, disgust 1,000, fear 0,996, happy 0,986, neutral 1,000, sad 0,874, surprise 1,000

### exp3 — Single, LR 0,00001

- Strategy: full-model fine-tuning, LR 1e-05, max 100 epoch, early stopping 10
- Gambar: `experiments/exp3_single_lr1e5/cv_val_curve.png`, `roc_test.png`, `confusion_matrix_test.png`
- AUC per kelas (TEST): angry 0,991, disgust 0,973, fear 1,000, happy 0,991, neutral 0,968, sad 0,797, surprise 0,991

### exp4 — Two-Stage, S2 LR 0,001

- Strategy: two-stage. S1 freeze backbone (head only) 30 ep @ 0.001; S2 unfreeze ['layer3', 'layer4', 'fc'] ≤70 ep @ 0.001, early stopping 10
- Gambar: `experiments/exp4_twostage_s2lr1e3/cv_val_curve.png`, `roc_test.png`, `confusion_matrix_test.png`
- AUC per kelas (TEST): angry 1,000, disgust 1,000, fear 1,000, happy 0,973, neutral 1,000, sad 0,833, surprise 1,000

### exp5 — Two-Stage, S2 LR 0,0001

- Strategy: two-stage. S1 freeze backbone (head only) 30 ep @ 0.001; S2 unfreeze ['layer3', 'layer4', 'fc'] ≤70 ep @ 0.0001, early stopping 10
- Gambar: `experiments/exp5_twostage_s2lr1e4/cv_val_curve.png`, `roc_test.png`, `confusion_matrix_test.png`
- AUC per kelas (TEST): angry 1,000, disgust 0,995, fear 1,000, happy 0,986, neutral 1,000, sad 0,842, surprise 1,000

### exp6 — Two-Stage, S2 LR 0,00001

- Strategy: two-stage. S1 freeze backbone (head only) 30 ep @ 0.001; S2 unfreeze ['layer3', 'layer4', 'fc'] ≤70 ep @ 1e-05, early stopping 10
- Gambar: `experiments/exp6_twostage_s2lr1e5/cv_val_curve.png`, `roc_test.png`, `confusion_matrix_test.png`
- AUC per kelas (TEST): angry 0,986, disgust 0,973, fear 1,000, happy 0,991, neutral 0,946, sad 0,766, surprise 0,982

