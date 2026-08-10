#!/bin/bash
# JAFFE otonom -- seluruh rencana dalam satu rantai.
#
# Menerapkan bagian metode terbaik CASME II yang bisa ditransfer ke gambar diam:
# backbone pretrained ImageNet + augmentasi + TTA + ensemble + reject option.
# Split 80/20 meniru Kaggle (test = validation). Semua di folder jaffee/.
cd "d:/AI-Projects/casmeII-new-from-sequence-model/jaffee" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

run(){   # $1 = config stem
  local stem="$1"
  local out="experiments/${stem}"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"; return
  fi
  log "START ${stem}"
  $CONDA src/train.py --config "configs/${stem}.json" \
      > "experiments/${stem}_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'FINAL' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem} -- lihat experiments/${stem}_out.log"
  fi
}

log "=== JAFFE QUEUE START ==="

# Index + split (idempoten).
$CONDA src/build_index.py >> "$QLOG" 2>&1

# Fase 1-2: latih semua model.
for stem in j01_simplecnn j02_resnet18_s42 j03_resnet18_s123 \
            j04_resnet18_s2024 j05_resnet34_s42; do
  run "$stem"
done

# Fase 2.4: ensemble 4 ResNet (gabungan model -- yang menang di CASME II).
log ">>> ensemble 4 ResNet"
$CONDA src/fuse.py --name ensemble_all \
    --members j02_resnet18_s42 j03_resnet18_s123 j04_resnet18_s2024 \
              j05_resnet34_s42 >> "$QLOG" 2>&1 && log "ensemble selesai"

# Fase 2.5: reject option pada ensemble.
log ">>> reject option"
$CONDA src/reject_option.py --run ensemble_all >> "$QLOG" 2>&1 && log "reject selesai"

# Fase 3: laporan.
log "menyusun laporan..."
$CONDA src/make_report.py >> "$QLOG" 2>&1
log "=== JAFFE QUEUE COMPLETE ==="
