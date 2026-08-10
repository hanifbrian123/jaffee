#!/bin/bash
# JAFFE dev/test + 5-fold CV dengan ROC/AUC. Otonom.
cd "d:/AI-Projects/casmeII-new-from-sequence-model/jaffee" || exit 1
CONDA="conda run -n facesleuth python"
QLOG="experiments/QUEUE_CV.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }
log "=== JAFFE CV QUEUE START ==="
$CONDA src/build_split_cv.py >> "$QLOG" 2>&1
stem="cv_resnet34"
out="experiments/${stem}"
if [ -f "${out}/summary.json" ] && grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
  log "SKIP ${stem}"
else
  log "START ${stem} (5-fold CV + test ensemble)"
  $CONDA src/train_cv.py --config "configs/${stem}.json" > "experiments/${stem}_out.log" 2>&1
  if grep -q '"complete": true' "${out}/summary.json" 2>/dev/null; then
    log "DONE  ${stem}: $(grep 'TEST' "${out}/run.log" | tail -1)"
  else
    log "!! FAILED ${stem} -- lihat experiments/${stem}_out.log"
  fi
fi
log "menyusun laporan..."
$CONDA src/make_report_cv.py >> "$QLOG" 2>&1
log "=== JAFFE CV QUEUE COMPLETE ==="
