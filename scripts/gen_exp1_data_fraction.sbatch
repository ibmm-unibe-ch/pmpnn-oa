#!/bin/bash
#SBATCH --job-name=pmpnn-exp1-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=04:00:00
#SBATCH --array=0-59%6

# Sequence generation with the models produced by Experiment 1 (data-fraction
# scaling sweep), for the held-out test monomers (data/test_monomers.txt).
#
# Each array task samples sequences for every structure in
# data/test_monomers.txt with one model's best.pt checkpoint and writes one
# FASTA file per structure (<pdbid_chain>.fa) to
# models/exp1_data_fraction/<name>_frac<fraction>_seed<seed>/eval/seqs/
#
# The (config, fraction, seed) grid and the array-index decomposition follow
# scripts/train_exp1_data_fraction.sbatch exactly (4 configs x 5 fractions x
# 3 seeds = 60 runs):
#   configs:   oa1 (mode=oa, 1 encoder layer), oa3 (mode=oa, 3 encoder layers),
#              vanilla3 (mode=vanilla, 3 layers), gat3 (mode=gat, 3 layers)
#   fractions: 0.1 0.2 0.4 0.8 1.0
#   seeds:     0 1 2
#
# Submit from the repo root: sbatch scripts/gen_exp1_data_fraction.sbatch
# The production partition has 6 schedulable GPUs (4x A5000, 2x A4000), so
# the array is throttled to 6 concurrent tasks (%6).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

CONFIG_NAMES=(oa1 oa3 vanilla3 gat3)
CONFIG_MODES=(oa  oa  vanilla   gat)
CONFIG_LAYERS=(1   3   3         3)

FRACTIONS=(0.1 0.2 0.4 0.8 1.0)
SEEDS=(0 1 2)

N_FRACTIONS=${#FRACTIONS[@]}
N_SEEDS=${#SEEDS[@]}

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / (N_FRACTIONS * N_SEEDS) ))
FRAC_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_FRACTIONS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

NAME=${CONFIG_NAMES[$CONFIG_IDX]}
MODE=${CONFIG_MODES[$CONFIG_IDX]}
LAYERS=${CONFIG_LAYERS[$CONFIG_IDX]}
FRACTION=${FRACTIONS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: NAME=${NAME}, MODE=${MODE}, LAYERS=${LAYERS}, FRACTION=${FRACTION}, SEED=${SEED}"

MODEL_DIR="models/exp1_data_fraction/${NAME}_frac${FRACTION}_seed${SEED}"
WEIGHTS="${MODEL_DIR}/model_weights/best.pt"
OUTPUT_DIR="${MODEL_DIR}/eval/seqs"

if [ ! -f "${WEIGHTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${WEIGHTS} does not exist, skipping." >&2
    exit 1
fi

if [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUTPUT_DIR} already populated, skipping."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
    data/test_monomers.txt
    "${OUTPUT_DIR}"
    "${WEIGHTS}"
    --h5_file data/pdb_2021aug02.h5
    --mode "${MODE}"
    --num_encoder_layers "${LAYERS}"
)

if [ "${MODE}" = "oa" ]; then
    ARGS+=(--r_max_orb 12.0)
fi

pmpnn-gen "${ARGS[@]}"
