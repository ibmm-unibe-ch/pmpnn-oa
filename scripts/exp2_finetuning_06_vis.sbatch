#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --array=0-20%12

# Geometric + edge-type attention analysis (pmpnn-vis, see src/vis/__main__.py)
# for the models produced by Experiment 2 (finetuning) that record attention —
# oa1, oa3 and gat3 (vanilla3 has no attention and therefore no stats.npz; see
# scripts/exp2_finetuning_05_stats.sbatch).
#
# Unlike the other exp2_finetuning scripts, each array task is one
# (model, encoder layer) PAIR, not just one model — oa1 has 1 layer, oa3 and
# gat3 have 3, so the flattened array below has 3*1 + 3*3 + 3*3 = 21 tasks
# (indices 0-20). Each task runs pmpnn-vis on one seed's own
# models/<model_name>/eval/stats.npz restricted to one layer (--layers), and
# writes that layer's figures to models/<model_name>/vis_<layer>/ — one
# output directory per (model, layer), not pooled across layers or seeds.
#
# All other options are left at their pmpnn-vis defaults (5000-replicate
# protein-level bootstrap, the default --pairs set, the 20x20 AA-pair
# heatmap, and the spectral-PCA section for OA checkpoints only).
#
# CPU-only (no GPU requested) — only the pre-computed eval/stats.npz file is
# read, no model weights or forward passes. --n_jobs is pinned to the
# SLURM cpu allocation so joblib doesn't oversubscribe the node.
#
# Requires models/exp2_finetuning/<model_name>/eval/stats.npz already
# populated by scripts/exp2_finetuning_05_stats.sbatch.
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_06_vis.sbatch
# Throttled to 8 concurrent tasks (%8) so the array doesn't monopolize the
# CPU partition; adjust to taste.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

# Parallel arrays, flattened over (model, layer) — oa1 has 1 encoder layer,
# oa3 and gat3 have 3 (same convention as scripts/exp2_finetuning_05_stats.sbatch).
MODEL_NAMES=(
    oa1_seed0
    oa1_seed1
    oa1_seed2
    oa3_seed0 oa3_seed0 oa3_seed0
    oa3_seed1 oa3_seed1 oa3_seed1
    oa3_seed2 oa3_seed2 oa3_seed2
    gat3_seed0 gat3_seed0 gat3_seed0
    gat3_seed1 gat3_seed1 gat3_seed1
    gat3_seed2 gat3_seed2 gat3_seed2
)
LAYERS=(
    1
    1
    1
    1 2 3
    1 2 3
    1 2 3
    1 2 3
    1 2 3
    1 2 3
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}, LAYER=${LAYER}"

MODEL_DIR="models/exp2_finetuning/${MODEL_NAME}"
STATS_FILE="${MODEL_DIR}/eval/stats.npz"
OUTPUT_DIR="${MODEL_DIR}/vis_${LAYER}"

if [ ! -s "${STATS_FILE}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${STATS_FILE} does not exist, skipping." >&2
    exit 1
fi

if [ -s "${OUTPUT_DIR}/geometry/fig_score_layer${LAYER}.svg" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUTPUT_DIR} already populated, skipping."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

pmpnn-vis "${STATS_FILE}" --out_dir "${OUTPUT_DIR}" --layers "${LAYER}" \
    --n_jobs "${SLURM_CPUS_PER_TASK:-8}"
