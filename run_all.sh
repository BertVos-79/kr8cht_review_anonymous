#!/usr/bin/env bash
# Run the full pipeline of analysis scripts in order.
# Default: run ALL scripts (a_ … f_), skipping any that are missing.
# Usage:
#   scripts/run_all.sh             # run all, continue on errors
#   scripts/run_all.sh --strict    # abort on first failure
#
# Environment knobs (optional):
#   REVIEW_MODE=1   → prefer idempotent/load-from-cache branches where supported
#   CPU_ONLY=1      → hint to use CPU (fine-tuning steps should skip/short-circuit)
#   OMP_NUM_THREADS → bound CPU threading (defaults to 1)
#   TOKENIZERS_PARALLELISM=false   (set by default here to reduce noise)
#
set -euo pipefail

# Resolve repo root (this script lives in ./scripts)
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT"

# Options
STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
  shift || true
fi

# Light environment hygiene
mkdir -p logs
export REVIEW_MODE="${REVIEW_MODE:-1}"
export CPU_ONLY="${CPU_ONLY:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Ordered list of scripts (missing ones are skipped gracefully)
SCRIPTS_ORDER=(
  "scripts/a_static.py"
  "scripts/b_frozen.py"
  "scripts/c_interim_report.py"
  "scripts/d_fine_tuned.py"
  "scripts/e_1_synth_augmentation.py"
  "scripts/e_2_teacher_labeling.py"
  "scripts/e_3_student_scoring.py"
  "scripts/f_1_wilcoxon_heatmap.py"
  "scripts/f_2_pooling_families_cd.py"
  "scripts/f_3_embedding_cd.py"
  "scripts/f_4_frozen_vs_finetuned.py"
  "scripts/f_5_average_diversity.py"
  "scripts/f_6_target_analysis.py"
)

echo "==> Running pipeline from: $ROOT"
echo "==> Python: $(python -V 2>/dev/null || echo 'not found')"
echo "==> REVIEW_MODE=${REVIEW_MODE} CPU_ONLY=${CPU_ONLY} STRICT=${STRICT}"

start_ts=$(date +%s)

for s in "${SCRIPTS_ORDER[@]}"; do
  if [[ -f "$s" ]]; then
    name=$(basename "$s" .py)
    echo "────────────────────────────────────────────────────────"
    echo "▶ Running $s"
    ts=$(date +%s)

    # run and tee logs; capture python's exit code
    set +e
    python -u "$s" 2>&1 | tee "logs/${name}.log"
    code=${PIPESTATUS[0]}
    set -e

    dur=$(( $(date +%s) - ts ))
    if [[ $code -ne 0 ]]; then
      echo "✗ ${s} exited with code ${code} (duration ${dur}s)"
      if [[ $STRICT -eq 1 ]]; then
        echo "Aborting due to --strict."
        exit $code
      else
        echo "Continuing to next script..."
      fi
    else
      echo "✓ Finished ${s} in ${dur}s"
    fi
  else
    echo "• Skipping missing: $s"
  fi
done

total=$(( $(date +%s) - start_ts ))
echo "────────────────────────────────────────────────────────"
echo "All done in ${total}s. Logs in ./logs"
