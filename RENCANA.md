# Rencana Klasifikasi Ekspresi JAFFE — 7 kelas

Menerapkan bagian **metode terbaik dari proyek CASME II** yang bisa ditransfer,
dengan pembagian train/test meniru notebook Kaggle referensi.

## Dataset

| aspek | isinya |
|---|---|
| sumber | JAFFE (Japanese Female Facial Expression), 1997 — `jaffee/dataset/jaffe/` |
| jumlah | 213 gambar diam `.tiff`, grayscale 256×256 |
| subjek | 10 wanita (KA, KL, KM, KR, MK, NA, NM, TM, UY, YM) |
| kelas | 7 ekspresi: AN(30) DI(29) FE(32) HA(31) NE(30) SA(31) SU(30) — seimbang |
| label | dari nama file: `KA.AN1.39.tiff` → subjek KA, ekspresi AN (anger) |

## ⚠️ Perbedaan mendasar dengan CASME II — harus jujur

| | CASME II | JAFFE |
|---|---|---|
| jenis data | **video** (urutan frame) | **gambar diam** |
| metode inti kita | TV-L1 optical flow + r3d_18 (3D CNN) | **tidak bisa dipakai** |

Optical flow butuh ≥2 frame; JAFFE 1 gambar per sampel. Jadi fondasi metode
terbaik kita (flow + 3D CNN temporal) **tidak berlaku**. Yang ditransfer adalah
bagian yang justru terbukti menang di CASME II dan **tidak** bergantung video.

## Protokol evaluasi — meniru Kaggle referensi

| aspek | isinya |
|---|---|
| split | **80/20 acak** (bukan subject-independent), `random_state` tetap |
| validasi | **test = validation** saat training (sesuai notebook Kaggle) — 80 TIDAK dibagi lagi |
| metrik | Accuracy (utama, seperti Kaggle) + macro-F1 + confusion matrix |

⚠️ **Catatan jujur:** split acak 80/20 membuat gambar subjek yang sama bisa
muncul di train DAN test (kebocoran identitas). Akurasi JAFFE dengan split ini
biasanya sangat tinggi (95–100%) dan **tidak sebanding** dengan protokol
subject-independent CASME II. Ini disengaja untuk meniru Kaggle, bukan estimasi
generalisasi ke orang baru.

## Rencana percobaan

| Fase | # | Percobaan | Ditransfer dari CASME II | Status |
|---|---|---|---|---|
| **0** | 0.1 | Index dari nama file (subjek + label 7 kelas) | pola build_index | ⬜ |
| **0** | 0.2 | Loader `.tiff` → 3-channel, resize, normalisasi ImageNet | pola dataset.py | ⬜ |
| **1** | 1.1 | Split 80/20 acak, test = validation | protokol Kaggle | ⬜ |
| **1** | 1.2 | CNN sederhana (kontrol, meniru Kaggle) | — pembanding | ⬜ |
| **2** | 2.1 | **ResNet-18 pretrained ImageNet** + fine-tune | analog r3d_18/Kinetics | ⬜ |
| **2** | 2.2 | + augmentasi (flip, rotasi, erase) + **TTA** | augmentasi & TTA kita | ⬜ |
| **2** | 2.3 | ResNet-34 + backbone lain (keragaman) | — untuk ensemble | ⬜ |
| **2** | 2.4 | **Ensemble/gabungan model** | ✅ satu-satunya yang menang kuat | ⬜ |
| **2** | 2.5 | **Reject option** (boleh bilang "tidak yakin") | reject_option.py kita | ⬜ |
| **3** | 3.1 | Laporan lengkap: tabel + confusion matrix + kurva | make_experiment_report | ⬜ |

## Logging (selengkap CASME II)

Setiap run menyimpan: config + seed + hardware, loss & accuracy per-epoch
(train **dan** val), confusion matrix, kurva training, prediksi per-sampel,
dan baris ringkasan CSV. Semua di `jaffee/experiments/`.

## Ekspektasi jujur

- Baseline CNN sederhana: ~85–95% (JAFFE mudah dengan split acak)
- ResNet pretrained + TTA + ensemble: mendekati atau mencapai ~98–100%
- Ensemble kemungkinan menang tipis — headroom kecil karena baseline sudah tinggi
- **Angka tinggi di sini WAJAR** karena kebocoran identitas, bukan bukti metode
  unggul untuk orang baru. Kalau mau uji jujur ke orang baru, perlu split
  subject-independent (opsional, di luar permintaan sekarang)

## Struktur folder (semua JAFFE di sini)

```
jaffee/
  dataset/jaffe/     213 .tiff (sudah ada)
  src/               kode
  configs/           config per percobaan
  experiments/       log, kurva, prediksi, CSV ringkasan
  reports/           laporan akhir
  RENCANA.md         file ini
```
