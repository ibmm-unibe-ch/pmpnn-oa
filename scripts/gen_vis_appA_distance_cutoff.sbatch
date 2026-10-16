#!/bin/bash
#SBATCH --job-name=pmpnn-appA-gen-vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=10:00:00
#SBATCH --array=0-11%6

# Sequence generation, statistics extraction and geometric/attention
# visualisation for the models produced by Appendix A (OA distance-cutoff
# sweep), on the held-out test monomers (data/test_monomers.txt).
#
# Merges scripts/gen_exp2_finetuning.sbatch, scripts/stats_exp2_finetuning.sbatch
# and scripts/vis_exp2_finetuning.sbatch into a single per-model pipeline:
# every appA model is oa1 (1 encoder layer), so there is only ever one
# pmpnn-vis layer per model and no need for the (model, layer)-flattened
# array scripts/vis_exp2_finetuning.sbatch uses for exp2's oa3/gat3 (3
# layers each).
#
# Each array task, for one models/appA_distance_cutoff/<model_name>:
#   1. pmpnn-gen   -> eval/seqs/<pdbid_chain>.fa              (if missing)
#   2. pmpnn-stats -> eval/stats.npz                          (if missing)
#   3. pmpnn-vis   -> vis_1/ (figures for encoder layer 1)    (if missing)
# Each stage is skipped if its output already exists, so a partially
# completed task can be resumed by re-submitting.
#
# Model naming: oa1_dist<r_max_orb>_seed<seed>, e.g. oa1_dist8_seed0. Every
# model is mode=oa, num_encoder_layers=1; --r_max_orb is read directly off
# the "dist<N>" component of the model name, since that cutoff is the
# independent variable Appendix A sweeps over.
#
# Submit from the repo root: sbatch scripts/gen_vis_appA_distance_cutoff.sbatch
# The production partition has 6 schedulable GPUs (4x A5000, 2x A4000), so
# the array is throttled to 6 concurrent tasks (%6).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

MODEL_NAMES=(
    oa1_dist8_seed0  oa1_dist8_seed1  oa1_dist8_seed2
    oa1_dist10_seed0 oa1_dist10_seed1 oa1_dist10_seed2
    oa1_dist14_seed0 oa1_dist14_seed1 oa1_dist14_seed2
    oa1_dist16_seed0 oa1_dist16_seed1 oa1_dist16_seed2
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

if [[ "${MODEL_NAME}" =~ ^oa1_dist([0-9]+)_seed[0-9]+$ ]]; then
    MODE=oa
    LAYERS=1
    R_MAX_ORB="${BASH_REMATCH[1]}.0"
else
    echo "Task ${SLURM_ARRAY_TASK_ID}: unrecognized model name ${MODEL_NAME}" >&2
    exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}, MODE=${MODE}, LAYERS=${LAYERS}, R_MAX_ORB=${R_MAX_ORB}"

MODEL_DIR="models/appA_distance_cutoff/${MODEL_NAME}"
WEIGHTS="${MODEL_DIR}/model_weights/best.pt"

if [ ! -f "${WEIGHTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${WEIGHTS} does not exist, skipping." >&2
    exit 1
fi

# --- 1. Sequence generation --------------------------------------------------

SEQS_DIR="${MODEL_DIR}/eval/seqs"

if [ -d "${SEQS_DIR}" ] && [ -n "$(ls -A "${SEQS_DIR}" 2>/dev/null)" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${SEQS_DIR} already populated, skipping generation."
else
    mkdir -p "${SEQS_DIR}"
    pmpnn-gen \
        data/test_monomers.txt \
        "${SEQS_DIR}" \
        "${WEIGHTS}" \
        --h5_file data/pdb_2021aug02.h5 \
        --mode "${MODE}" \
        --num_encoder_layers "${LAYERS}" \
        --r_max_orb "${R_MAX_ORB}"
fi

# --- 2. Statistics extraction ------------------------------------------------

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

# --- 3. Visualisation --------------------------------------------------------

VIS_DIR="${MODEL_DIR}/vis_${LAYERS}"

if [ -s "${VIS_DIR}/geometry/fig_score_layer${LAYERS}.svg" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${VIS_DIR} already populated, skipping vis."
else
    mkdir -p "${VIS_DIR}"
    pmpnn-vis "${STATS_FILE}" --out_dir "${VIS_DIR}" --layers "${LAYERS}" \
        --n_jobs "${SLURM_CPUS_PER_TASK:-8}"
fi
