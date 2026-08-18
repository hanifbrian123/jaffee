#!/bin/bash
# JAFFE: 6 eksperimen (sweep LR + two-stage), 5-fold CV, early stopping, 6 metrik.
cd "d:/AI-Projects/casmeII-new-from-sequence-model/jaffee" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_EXP.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }
log "=== JAFFE EXPERIMENTS START ==="
$CONDA src/build_split_cv.py >> "$QLOG" 2>&1
for stem in exp1_single_lr1e3 exp2_single_lr1e4 exp3_single_lr1e5 \
            exp4_twostage_s2lr1e3 exp5_twostage_s2lr1e4 exp6_twostage_s2lr1e5; do
  out="experiments/${stem}"
  if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "SKIP ${stem}"; continue
  fi
  log "START ${stem}"
  $CONDA src/train_exp.py --config "configs/${stem}.json" > "experiments/${stem}_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'TEST' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem} -- lihat experiments/${stem}_out.log"
  fi
done
log "menyusun laporan..."
$CONDA src/make_report_exp.py >> "$QLOG" 2>&1
log "=== JAFFE EXPERIMENTS COMPLETE ==="
