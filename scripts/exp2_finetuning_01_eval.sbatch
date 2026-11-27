#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=04:00:00
#SBATCH --array=0-9%6

# Teacher-forced evaluation of the models produced by Experiment 2
# (finetuning), on the held-out test split.
#
# Each array task evaluates one model directory under
# models/exp2_finetuning/<model_name>/model_weights/best.pt and writes its
# metrics to models/exp2_finetuning/<model_name>/eval/results.txt
#
# The model_name -> (mode, num_encoder_layers) mapping follows the same
# naming convention used by scripts/exp1_data_fraction_01_train.sbatch:
#   oa1*      -> mode=oa,      num_encoder_layers=1
#   oa3*      -> mode=oa,      num_encoder_layers=3
#   vanilla3* -> mode=vanilla, num_encoder_layers=3
#   gat3*     -> mode=gat,     num_encoder_layers=3
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_01_eval.sbatch
# The production partition has 6 schedulable GPUs (4x A5000, 2x A4000), so
# the array is throttled to 6 concurrent tasks (%6).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

MODEL_NAMES=(
    oa1_seed0 oa1_seed1 oa1_seed2
    oa3_seed0 oa3_seed1 oa3_seed2
    vanilla3
    gat3_seed0 gat3_seed1 gat3_seed2
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

case "${MODEL_NAME}" in
    oa1*)      MODE=oa;      LAYERS=1 ;;
    oa3*)      MODE=oa;      LAYERS=3 ;;
    vanilla3*) MODE=vanilla; LAYERS=3 ;;
    gat3*)     MODE=gat;     LAYERS=3 ;;
    *) echo "Task ${SLURM_ARRAY_TASK_ID}: unrecognized model name ${MODEL_NAME}" >&2; exit 1 ;;
esac

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}, MODE=${MODE}, LAYERS=${LAYERS}"

MODEL_DIR="models/exp2_finetuning/${MODEL_NAME}"
WEIGHTS="${MODEL_DIR}/model_weights/best.pt"
OUTPUT_DIR="${MODEL_DIR}/eval"
OUTPUT_FILE="${OUTPUT_DIR}/results.txt"

if [ ! -f "${WEIGHTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${WEIGHTS} does not exist, skipping." >&2
    exit 1
fi

if [ -s "${OUTPUT_FILE}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUTPUT_FILE} already exists, skipping."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
    data/pdb_2021aug02.h5 data/list.csv data/valid_clusters.txt data/test_clusters.txt
    "${WEIGHTS}"
    --mode "${MODE}"
    --num_encoder_layers "${LAYERS}"
    --no_tqdm
)

if [ "${MODE}" = "oa" ]; then
    ARGS+=(--r_max_orb 12.0)
fi

pmpnn-eval "${ARGS[@]}" > "${OUTPUT_FILE}"
