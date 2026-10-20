#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-gen-multimers
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=08:00:00
#SBATCH --array=0-9%3

# Sequence generation with the models produced by Experiment 2 (finetuning),
# for the held-out test homomers and heteromers (data/test_homomers.txt,
# data/test_heteromers.txt) — the multi-chain complex counterparts of
# scripts/gen_exp2_finetuning.sbatch, which only covers test_monomers.txt.
#
# Both test sets use the same '<pdbid>_<chain>' entry format as
# test_monomers.txt (one representative chain per structure), so each array
# task runs the same model through pmpnn-gen twice, once per test set, and
# writes one FASTA file per structure (<pdbid_chain>.fa) to
#   models/exp2_finetuning/<model_name>/eval/seqs_homomers/
#   models/exp2_finetuning/<model_name>/eval/seqs_heteromers/
#
# The model_name -> (mode, num_encoder_layers) mapping follows the same
# naming convention used by scripts/train_exp1_data_fraction.sbatch and
# scripts/gen_exp2_finetuning.sbatch:
#   oa1*      -> mode=oa,      num_encoder_layers=1
#   oa3*      -> mode=oa,      num_encoder_layers=3
#   vanilla3* -> mode=vanilla, num_encoder_layers=3
#   gat3*     -> mode=gat,     num_encoder_layers=3
#
# Submit from the repo root: sbatch scripts/gen_exp2_finetuning_multimers.sbatch
# The production partition has 6 schedulable GPUs (4x A5000, 2x A4000), so
# the array is throttled to 3 concurrent tasks (%3), matching
# scripts/gen_exp2_finetuning.sbatch; --time is doubled to 08:00:00 since
# each task now generates for two 512-structure test sets instead of one.

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

if [ ! -f "${WEIGHTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${WEIGHTS} does not exist, skipping." >&2
    exit 1
fi

generate() {
    local test_list="$1"
    local output_dir="$2"

    if [ -d "${output_dir}" ] && [ -n "$(ls -A "${output_dir}" 2>/dev/null)" ]; then
        echo "Task ${SLURM_ARRAY_TASK_ID}: ${output_dir} already populated, skipping."
        return
    fi

    mkdir -p "${output_dir}"

    local args=(
        "${test_list}"
        "${output_dir}"
        "${WEIGHTS}"
        --h5_file data/pdb_2021aug02.h5
        --mode "${MODE}"
        --num_encoder_layers "${LAYERS}"
    )
    if [ "${MODE}" = "oa" ]; then
        args+=(--r_max_orb 12.0)
    fi

    pmpnn-gen "${args[@]}"
}

generate data/test_homomers.txt   "${MODEL_DIR}/eval/seqs_homomers"
generate data/test_heteromers.txt "${MODEL_DIR}/eval/seqs_heteromers"
