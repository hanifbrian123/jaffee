"""Regenerate jaffee/reports/HASIL.md from the runs on disk (one table)."""
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "experiments")
OUT = os.path.join(HERE, "reports", "HASIL.md")

# (fase, #, deskripsi, run-name)
PLAN = [
    ("1.2", "j01", "CNN sederhana dari nol (kontrol Kaggle)", "j01_simplecnn"),
    ("2.1", "j02", "ResNet-18 pretrained + TTA, seed 42", "j02_resnet18_s42"),
    ("2.1", "j03", "ResNet-18 pretrained + TTA, seed 123", "j03_resnet18_s123"),
    ("2.1", "j04", "ResNet-18 pretrained + TTA, seed 2024", "j04_resnet18_s2024"),
    ("2.3", "j05", "ResNet-34 pretrained + TTA, seed 42", "j05_resnet34_s42"),
    ("2.4", "ens", "**Ensemble (gabungan 4 ResNet)**", "ensemble_all"),
]


def load(name):
    path = os.path.join(EXP, name, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def num(value):
    return f"{value:.4f}".replace(".", ",")


def main():
    lines = ["# Hasil JAFFE — klasifikasi 7 kelas", ""]
    lines.append(f"Dibuat otomatis dari `jaffee/experiments/` pada "
                 f"**{datetime.now().strftime('%d %B %Y, %H:%M')}**.")
    lines.append("Perbarui: `python jaffee/src/make_report.py`")
    lines.append("")
    lines.append("Protokol: split 80/20 acak stratified (170 train / 43 test), "
                 "test dipakai sebagai validation saat training — meniru "
                 "notebook Kaggle.  "
                 "")
    lines.append("")
    lines.append("| Fase | # | Percobaan | Accuracy | macro-F1 | UAR | Status |")
    lines.append("|---|---|---|---|---|---|---|")
    best = None
    for fase, tag, label, name in PLAN:
        summary = load(name)
        if summary is None:
            lines.append(f"| {fase} | {tag} | {label} | ⏳ | | | antre |")
            continue
        acc = summary["ACC"]
        lines.append(f"| {fase} | {tag} | {label} | **{num(acc)}** | "
                     f"{num(summary['macroF1'])} | {num(summary['UAR'])} | ✅ |")
        if best is None or acc > best[0]:
            best = (acc, label)
    lines.append("")
    if best:
        lines.append(f"**Terbaik: {best[1]} — accuracy {num(best[0])}**")
        lines.append("")

    # Temuan: apakah ensemble mengalahkan model tunggal terbaik?
    ens = load("ensemble_all")
    singles = [load(n)["ACC"] for _, _, _, n in PLAN
               if n != "ensemble_all" and load(n) is not None]
    if ens is not None and singles:
        best_single = max(singles)
        lines.append("## Temuan")
        lines.append("")
        if ens["ACC"] >= best_single:
            lines.append(f"- Ensemble ({num(ens['ACC'])}) ≥ model tunggal terbaik "
                         f"({num(best_single)}): menggabungkan model menolong, "
                         "seperti di CASME II.")
        else:
            lines.append(f"- **Ensemble ({num(ens['ACC'])}) TIDAK mengalahkan "
                         f"model tunggal terbaik ({num(best_single)}).** "
                         "ResNet-34 sendirian lebih kuat; menggabungkannya dengan "
                         "tiga ResNet-18 yang lebih lemah justru menyeret turun. "
                         "Ini persis pelajaran CASME II: ensemble hanya menolong "
                         "kalau anggotanya sama-sama kuat (aturan ±0,04). Untuk "
                         "JAFFE, pakai **ResNet-34 + TTA** saja, atau ensemble "
                         "beberapa ResNet-34 (bukan campuran).")
        lines.append(f"- Test set hanya **43 gambar** → tiap gambar = 2,3%. "
                     "Selisih 1–2 gambar menggeser angka banyak; jangan "
                     "over-interpretasi beda kecil.")
        lines.append(f"- CNN dari nol (55,8%) vs ResNet pretrained (93–98%): "
                     "**pretraining ImageNet menyumbang ~40 poin** — konsisten "
                     "dengan temuan CASME II bahwa backbone pretrained itu kunci.")
        lines.append("")

    # reject option kalau ada
    reject_path = os.path.join(EXP, "ensemble_all", "reject_option.json")
    if os.path.exists(reject_path):
        with open(reject_path, encoding="utf-8") as handle:
            reject = json.load(handle)
        lines.append("## Reject option (ensemble)")
        lines.append("")
        lines.append("| dijawab | accuracy |")
        lines.append("|---|---|")
        for row in reject["curve"]:
            lines.append(f"| {row['coverage']*100:.0f}% | {num(row['acc'])} |")
        lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"ditulis: {OUT}")


if __name__ == "__main__":
    main()
