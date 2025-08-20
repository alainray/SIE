#!/usr/bin/env bash
set -Eeuo pipefail

# Uso:
#   ./run_sie_all.sh results/baseline
BASE="${1:-results/baseline}"

WORKDIR="/root/SIE"
DATA_ROOT="/root/datasets/3DIEBench"

WEIGHTS_FILE="${BASE}/final_weights.pth"
ROOT_LOG_DIR="${BASE}/logs"  # ojo: sin slash final

# Exp-dirs
EXP_CLASS_EQ="${BASE}/classification_eq"
EXP_CLASS_INV="${BASE}/classification_inv"
EXP_CLASS_FULL="${BASE}/classification_full"

EXP_ANGLE_EQ="${BASE}/angle_eq"
EXP_ANGLE_INV="${BASE}/angle_inv"
EXP_ANGLE_FULL="${BASE}/angle_full"

EXP_COLOR_EQ="${BASE}/color_eq"
EXP_COLOR_INV="${BASE}/color_inv"
EXP_COLOR_FULL="${BASE}/color_full"

# Prep
cd "$WORKDIR"
mkdir -p \
  "$ROOT_LOG_DIR" \
  "$EXP_CLASS_EQ" "$EXP_CLASS_INV" "$EXP_CLASS_FULL" \
  "$EXP_ANGLE_EQ" "$EXP_ANGLE_INV" "$EXP_ANGLE_FULL" \
  "$EXP_COLOR_EQ" "$EXP_COLOR_INV" "$EXP_COLOR_FULL"

TS="$(date +%F_%H-%M-%S)"
PIPELINE_LOG="${ROOT_LOG_DIR}/pipeline_${TS}.log"
METRICS_CSV="${ROOT_LOG_DIR}/pipeline_metrics_${TS}.csv"

# Duplicar salida a log
exec > >(tee -a "$PIPELINE_LOG") 2>&1

# --- utils ---
fmt() { local s=$1; printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60)); }

STEP_NAMES=()
STEP_SECS=()
STEP_RC=()

print_summary() {
  echo ""
  echo "==================== RESUMEN DE TIEMPOS ===================="
  local total=0
  for i in "${!STEP_NAMES[@]}"; do
    local name="${STEP_NAMES[$i]}"
    local sec="${STEP_SECS[$i]}"
    local rc="${STEP_RC[$i]}"
    printf " - %-28s %s  (rc=%s)\n" "$name" "$(fmt "$sec")" "$rc"
    total=$(( total + sec ))
  done
  echo "-------------------------------------------------------------"
  echo " Tiempo total:               $(fmt "$total")"
  echo " CSV: $METRICS_CSV"
  echo " Log: $PIPELINE_LOG"
  echo "============================================================="
}
trap print_summary EXIT

echo "timestamp,name,status_code,seconds" > "$METRICS_CSV"

run_step() {
  local name="$1"; shift
  echo ""
  echo "== [$name] inicio: $(date '+%F %T') =="

  local start end dur rc
  start=$(date +%s)
  set +e
  "$@"
  rc=$?
  set -e
  end=$(date +%s)
  dur=$(( end - start ))

  STEP_NAMES+=("$name"); STEP_SECS+=("$dur"); STEP_RC+=("$rc")
  echo "$(date '+%F %T'),${name},${rc},${dur}" >> "$METRICS_CSV"

  if (( rc != 0 )); then
    echo "== [$name] FALLO rc=$rc en $(fmt "$dur") =="
    return $rc
  fi
  echo "== [$name] OK en $(fmt "$dur") =="
}

echo "== Inicio pipeline: $(date '+%F %T') =="
echo "BASE=${BASE}"
echo "Pesos: $WEIGHTS_FILE"
(rocm-smi || true)

# ====================== COMANDOS ======================

# 1) Clasificación EQ (equi)
run_step "classification_eq" \
python eval_classification.py \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_CLASS_EQ}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 256 --device cuda:0

# 2) Clasificación INV (inv-part)
run_step "classification_inv" \
python eval_classification.py \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_CLASS_INV}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 256 --device cuda:0 --inv-part

# 3) Clasificación FULL (512)
run_step "classification_full" \
python eval_classification.py \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_CLASS_FULL}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 512 --device cuda:0

# 4) Ángulo EQ (deep-end)
run_step "angle_eq" \
python eval_angle_prediction.py \
  --experience quat \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_ANGLE_EQ}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 256 --device cuda:0 --deep-end

# 5) Ángulo INV (deep-end + inv-part)
run_step "angle_inv" \
python eval_angle_prediction.py \
  --experience quat \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_ANGLE_INV}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 256 --device cuda:0 --deep-end --inv-part

# 6) Ángulo FULL (512, deep-end)
run_step "angle_full" \
python eval_angle_prediction.py \
  --experience quat \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_ANGLE_FULL}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 512 --device cuda:0 --deep-end

# 7) Color EQ (deep-end)
run_step "color_eq" \
python eval_color_prediction.py \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_COLOR_EQ}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 256 --device cuda:0 --deep-end

# 8) Color INV (deep-end + inv-part)
run_step "color_inv" \
python eval_color_prediction.py \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_COLOR_INV}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 256 --device cuda:0 --deep-end --inv-part

# 9) Color FULL (512, deep-end)
run_step "color_full" \
python eval_color_prediction.py \
  --weights-file "${WEIGHTS_FILE}" \
  --dataset-root "${DATA_ROOT}" \
  --exp-dir "${EXP_COLOR_FULL}" \
  --root-log-dir "${ROOT_LOG_DIR}/" \
  --epochs 300 --arch resnet18 --batch-size 256 \
  --lr 0.001 --wd 0.00000 --equi-dims 512 --device cuda:0 --deep-end

echo ""
echo "== Fin pipeline: $(date '+%F %T') =="
