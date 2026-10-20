#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-gen-multimers
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=10:00:00
#SBATCH --array=0-3

# Sequence generation with the models produced by Experiment 2 (finetuning),
# for the held-out test homomers and heteromers (data/test_homomers.txt,
# data/test_heteromers.txt) — the multi-chain complex counterparts of
# scripts/gen_exp2_finetuning.sbatch, which only covers test_monomers.txt.
#
# pmpnn-gen is inference-only and lightweight on GPU memory/compute, so
# instead of one model per array task (and one GPU per model), this script
# packs GROUP_SIZE=3 models per array task and runs them as concurrent
# background processes sharing the task's single allocated GPU. 10 models
# / 3 per group = 4 array tasks (0-3; the last group has only 1 model).
#
# Each backgrounded model process runs both pmpnn-gen passes, one after the
# other, and writes to:
#   data/test_homomers.txt   -> models/exp2_finetuning/<model_name>/eval/seqs_homomers/
#   data/test_heteromers.txt -> models/exp2_finetuning/<model_name>/eval/seqs_heteromers/
# A failure in one backgrounded model does not stop its siblings; the task
# as a whole exits non-zero if any model failed.
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
# --cpus-per-task/--mem are scaled up (x2, not x3) from a single-model task
# since 3 concurrent inference processes share the GPU but don't linearly
# multiply CPU/host-memory needs; adjust to taste if a group turns out to
# be CPU- or memory-starved in practice. No array throttle is needed: with
# only 4 tasks total, running them all at once still fits comfortably
# within the production partition's 6 schedulable GPUs.

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

GROUP_SIZE=3
N_MODELS=${#MODEL_NAMES[@]}

START=$(( SLURM_ARRAY_TASK_ID * GROUP_SIZE ))
END=$(( START + GROUP_SIZE ))
if [ "${END}" -gt "${N_MODELS}" ]; then
    END=${N_MODELS}
fi

generate() {
    local test_list="$1"
    local output_dir="$2"
    local weights="$3"
    local mode="$4"
    local layers="$5"

    if [ -d "${output_dir}" ] && [ -n "$(ls -A "${output_dir}" 2>/dev/null)" ]; then
        echo "$(basename "${output_dir}"): already populated, skipping."
        return
    fi

    mkdir -p "${output_dir}"

    local args=(
        "${test_list}"
        "${output_dir}"
        "${weights}"
        --h5_file data/pdb_2021aug02.h5
        --mode "${mode}"
        --num_encoder_layers "${layers}"
    )
    if [ "${mode}" = "oa" ]; then
        args+=(--r_max_orb 12.0)
    fi

    pmpnn-gen "${args[@]}"
}

run_model() {
    local model_name="$1"
    local mode layers

    case "${model_name}" in
        oa1*)      mode=oa;      layers=1 ;;
        oa3*)      mode=oa;      layers=3 ;;
        vanilla3*) mode=vanilla; layers=3 ;;
        gat3*)     mode=gat;     layers=3 ;;
        *) echo "${model_name}: unrecognized model name" >&2; return 1 ;;
    esac

    local model_dir="models/exp2_finetuning/${model_name}"
    local weights="${model_dir}/model_weights/best.pt"

    if [ ! -f "${weights}" ]; then
        echo "${model_name}: ${weights} does not exist, skipping." >&2
        return 1
    fi

    echo "${model_name}: MODE=${mode}, LAYERS=${layers}"
    generate data/test_homomers.txt   "${model_dir}/eval/seqs_homomers"   "${weights}" "${mode}" "${layers}"
    generate data/test_heteromers.txt "${model_dir}/eval/seqs_heteromers" "${weights}" "${mode}" "${layers}"
}

echo "Task ${SLURM_ARRAY_TASK_ID}: running models ${MODEL_NAMES[*]:START:END-START} concurrently on one GPU"

pids=()
for (( i = START; i < END; i++ )); do
    run_model "${MODEL_NAMES[$i]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done

exit "${status}"
