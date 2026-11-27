#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-geometry
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --array=0-8

# (d, theta, phi) geometry maps for the models produced by Experiment 2
# (finetuning) that record attention — oa1, oa3 and gat3 (vanilla3 has no
# attention and therefore no stats.npz; see scripts/exp2_finetuning_05_stats.sbatch).
#
# Each array task runs pmpnn-geometry on one seed's own
# models/<model_name>/eval/stats.npz, writing its per-layer (d, theta, phi)
# score-map figures to models/exp2_finetuning/<model_name>/eval/geometry/ —
# one figure set per seed checkpoint, not pooled across seeds.
#
# Volumetric (MRC / ChimeraX) export is skipped (--no_vol): only the
# fig_geometry_scatter_layerN.svg score maps are needed here.
#
# CPU-only (no GPU requested) and fast — no model weights or forward passes
# are required, only the pre-computed eval/stats.npz file.
#
# Requires models/exp2_finetuning/<model_name>/eval/stats.npz already
# populated by scripts/exp2_finetuning_05_stats.sbatch.
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_06_geometry.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

MODEL_NAMES=(
    oa1_seed0 oa1_seed1 oa1_seed2
    oa3_seed0 oa3_seed1 oa3_seed2
    gat3_seed0 gat3_seed1 gat3_seed2
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}"

MODEL_DIR="models/exp2_finetuning/${MODEL_NAME}"
STATS_FILE="${MODEL_DIR}/eval/stats.npz"
OUTPUT_DIR="${MODEL_DIR}/eval/geometry"

if [ ! -s "${STATS_FILE}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${STATS_FILE} does not exist, skipping." >&2
    exit 1
fi

if [ -s "${OUTPUT_DIR}/fig_geometry_scatter_layer1.svg" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUTPUT_DIR} already populated, skipping."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

pmpnn-geometry "${STATS_FILE}" --out_dir "${OUTPUT_DIR}" --no_vol
