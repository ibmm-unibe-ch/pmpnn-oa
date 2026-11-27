#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-tmscore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=00:30:00
#SBATCH --array=10

# Sanitize-to-TMscore pipeline for the models produced by Experiment 2
# (finetuning), plus a self-consistency baseline for the native sequences
# themselves ("native" in MODEL_NAMES below: Boltz's prediction from the
# true native sequence, scored against the true native structure). For one
# model/seed (or "native") directory, each array task:
#
#   1. runs scripts/sanitize_boltz.py, reducing every Boltz-predicted
#      structure under eval/structs_boltz/ to a Ca-only mmCIF under
#      eval/structs_boltz_ca/, then
#   2. runs scripts/tmscore.py, which runs USalign between each sanitized
#      prediction and its native structure (models/exp2_finetuning/native/
#      eval/structs/, written beforehand by scripts/extract_native.py) and
#      writes the result to eval/structs_boltz/TMscore.
#
# Both steps are CPU-only (no GPU requested) and fast (a couple of minutes
# per model), so the array isn't throttled.
#
# Requires:
#   - models/exp2_finetuning/native/eval/structs/ already populated, e.g.
#     `python scripts/extract_native.py`
#   - models/exp2_finetuning/native/eval/structs_boltz/ populated by a
#     exp2_finetuning_03_boltz.sbatch run over the "native" entry, for the
#     self-consistency baseline task
#   - the "usalign" environment module (loaded below)
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_04_tmscore.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

module load usalign
source .venv/bin/activate

MODEL_NAMES=(
    oa1_seed0 oa1_seed1 oa1_seed2
    oa3_seed0 oa3_seed1 oa3_seed2
    vanilla3
    gat3_seed0 gat3_seed1 gat3_seed2
    native
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
MODEL_DIR="models/exp2_finetuning/${MODEL_NAME}"
PREDICTIONS_DIR="${MODEL_DIR}/eval/structs_boltz/boltz_results_seqs_boltz/predictions"
TMSCORE_FILE="${MODEL_DIR}/eval/structs_boltz/TMscore"

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}"

if [ ! -d "${PREDICTIONS_DIR}" ] || [ -z "$(ls -A "${PREDICTIONS_DIR}" 2>/dev/null)" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${PREDICTIONS_DIR} does not exist or is empty, skipping." >&2
    exit 1
fi

if [ -s "${TMSCORE_FILE}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${TMSCORE_FILE} already exists, skipping."
    exit 0
fi

python scripts/sanitize_boltz.py "${MODEL_DIR}"
python scripts/tmscore.py "${MODEL_DIR}"
