# Hasil JAFFE — klasifikasi 7 kelas

Dibuat otomatis dari `jaffee/experiments/` pada **07 August 2026, 09:39**.
Perbarui: `python jaffee/src/make_report.py`

Protokol: split 80/20 acak stratified (170 train / 43 test), test dipakai sebagai validation saat training — meniru notebook Kaggle. ⚠️ Split acak = ada kebocoran identitas subjek, jadi akurasi tinggi wajar dan bukan estimasi ke orang baru.

| Fase | # | Percobaan | Accuracy | macro-F1 | UAR | Status |
|---|---|---|---|---|---|---|
| 1.2 | j01 | CNN sederhana dari nol (kontrol Kaggle) | **0,5581** | 0,4939 | 0,5578 | ✅ |
| 2.1 | j02 | ResNet-18 pretrained + TTA, seed 42 | **0,9302** | 0,9235 | 0,9286 | ✅ |
| 2.1 | j03 | ResNet-18 pretrained + TTA, seed 123 | **0,9302** | 0,9235 | 0,9286 | ✅ |
| 2.1 | j04 | ResNet-18 pretrained + TTA, seed 2024 | **0,9302** | 0,9271 | 0,9286 | ✅ |
| 2.3 | j05 | ResNet-34 pretrained + TTA, seed 42 | **0,9767** | 0,9760 | 0,9762 | ✅ |
| 2.4 | ens | **Ensemble (gabungan 4 ResNet)** | **0,9535** | 0,9509 | 0,9524 | ✅ |

**Terbaik: ResNet-34 pretrained + TTA, seed 42 — accuracy 0,9767**

## Temuan

- **Ensemble (0,9535) TIDAK mengalahkan model tunggal terbaik (0,9767).** ResNet-34 sendirian lebih kuat; menggabungkannya dengan tiga ResNet-18 yang lebih lemah justru menyeret turun. Ini persis pelajaran CASME II: ensemble hanya menolong kalau anggotanya sama-sama kuat (aturan ±0,04). Untuk JAFFE, pakai **ResNet-34 + TTA** saja, atau ensemble beberapa ResNet-34 (bukan campuran).
- Test set hanya **43 gambar** → tiap gambar = 2,3%. Selisih 1–2 gambar menggeser angka banyak; jangan over-interpretasi beda kecil.
- CNN dari nol (55,8%) vs ResNet pretrained (93–98%): **pretraining ImageNet menyumbang ~40 poin** — konsisten dengan temuan CASME II bahwa backbone pretrained itu kunci.

## Reject option (ensemble)

| dijawab | accuracy |
|---|---|
| 100% | 0,9535 |
| 91% | 0,9744 |
| 79% | 0,9706 |
| 70% | 0,9667 |
| 60% | 0,9615 |
| 51% | 0,9545 |

