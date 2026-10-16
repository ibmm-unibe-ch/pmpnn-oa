#!/bin/bash
#SBATCH --job-name=pmpnn-exp1-oa1-stats-vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=08:00:00
#SBATCH --array=0-14%6

# Statistics extraction and geometric/attention visualisation for the oa1
# models produced by Experiment 1 (data-fraction scaling sweep), on the
# held-out test monomers (data/test_monomers.txt).
#
# Merges scripts/stats_exp2_finetuning.sbatch and
# scripts/vis_exp2_finetuning.sbatch into a single per-model pipeline: oa1
# is always 1 encoder layer, so there is only ever one pmpnn-vis layer per
# model and no need for the (model, layer)-flattened array
# scripts/vis_exp2_finetuning.sbatch uses for exp2's oa3/gat3 (3 layers
# each). Sequence generation is not included here — it is already covered
# by scripts/gen_exp1_data_fraction.sbatch for all four exp1 configs.
#
# Each array task, for one models/exp1_data_fraction/oa1_frac<fraction>_seed<seed>:
#   1. pmpnn-stats -> eval/stats.npz                       (if missing)
#   2. pmpnn-vis   -> vis_1/ (figures for encoder layer 1) (if missing)
# Each stage is skipped if its output already exists, so a partially
# completed task can be resumed by re-submitting.
#
# Model naming: oa1_frac<fraction>_seed<seed>, following
# scripts/train_exp1_data_fraction.sbatch and
# scripts/gen_exp1_data_fraction.sbatch (mode=oa, num_encoder_layers=1,
# --r_max_orb 12.0, the same default used for oa1 elsewhere).
#
# Submit from the repo root: sbatch scripts/stats_vis_exp1_oa1.sbatch
# The production partition has 6 schedulable GPUs (4x A5000, 2x A4000), so
# the array is throttled to 6 concurrent tasks (%6).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

MODE=oa
LAYERS=1
R_MAX_ORB=12.0

FRACTIONS=(0.1 0.2 0.4 0.8 1.0)
SEEDS=(0 1 2)

N_SEEDS=${#SEEDS[@]}

FRAC_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

FRACTION=${FRACTIONS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

MODEL_NAME="oa1_frac${FRACTION}_seed${SEED}"

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}, MODE=${MODE}, LAYERS=${LAYERS}, R_MAX_ORB=${R_MAX_ORB}"

MODEL_DIR="models/exp1_data_fraction/${MODEL_NAME}"
WEIGHTS="${MODEL_DIR}/model_weights/best.pt"

if [ ! -f "${WEIGHTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${WEIGHTS} does not exist, skipping." >&2
    exit 1
fi

# --- 1. Statistics extraction ------------------------------------------------

STATS_FILE="${MODEL_DIR}/eval/stats.npz"

if [ -s "${STATS_FILE}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${STATS_FILE} already exists, skipping stats."
else
    mkdir -p "${MODEL_DIR}/eval"
    pmpnn-stats \
        data/pdb_2021aug02.h5 data/test_monomers.txt \
        "${WEIGHTS}" \
        "${STATS_FILE}" \
        --mode "${MODE}" \
        --num_encoder_layers "${LAYERS}" \
        --r_max_orb "${R_MAX_ORB}" \
        --device cuda
fi

# --- 2. Visualisation --------------------------------------------------------

VIS_DIR="${MODEL_DIR}/vis_${LAYERS}"

if [ -s "${VIS_DIR}/geometry/fig_score_layer${LAYERS}.svg" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${VIS_DIR} already populated, skipping vis."
else
    mkdir -p "${VIS_DIR}"
    pmpnn-vis "${STATS_FILE}" --out_dir "${VIS_DIR}" --layers "${LAYERS}" \
        --n_jobs "${SLURM_CPUS_PER_TASK:-8}"
fi
