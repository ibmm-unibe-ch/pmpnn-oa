#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-gen-permute-attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --array=0-8

# Permute-attention ablation (src/gen/permute_attn.py) of the Experiment 2
# (finetuning) oa1_seed0, oa3_seed0 and gat3_seed0 models, on the held-out
# test monomers (data/test_monomers.txt). Complements
# scripts/exp2_finetuning_07_ablation_uniform_attn.sbatch: instead of flattening a
# node's attention to its other neighbours to a uniform weight, this keeps
# the self-attention weight AND the learned multiset of non-self weights,
# but randomly reassigns which neighbour each weight lands on — so it
# tests whether the model needs to know specifically which neighbour each
# weight belongs to, as opposed to just needing a non-trivial attention
# profile over its neighbourhood.
#
# Runs the whole per-layer ablation sweep for each model, not just the
# fully-ablated case:
#   oa1_seed0  (1 encoder layer)  -> permute the only layer      -> 1 run
#   oa3_seed0  (3 encoder layers) -> permute each layer alone
#                                    (0, 1, 2) + all three         -> 4 runs
#   gat3_seed0 (3 encoder layers) -> same sweep as oa3_seed0,
#                                    geometry-free attention ablation -> 4 runs
# so the per-layer runs show which individual layer's attention matters,
# and the "all" run shows the fully-ablated model (the oa1 case is already
# "all" since it only has one layer). 1 + 4 + 4 = 9 tasks total (array 0-8).
#
# This is a hooks-based ablation (see register_permute_attention_hooks in
# src/gen/permute_attn.py) — no unablated baseline is run here, since
# that's already covered by models/exp2_finetuning/<model_name>/eval/seqs/
# from scripts/exp2_finetuning_02_gen_monomers.sbatch. The permutation is drawn fresh
# from torch's global RNG (seeded via --seed, left at its default) on
# every forward pass, independently per encoder layer.
#
# Each array task writes one FASTA file per structure to
# models/exp2_finetuning/<model_name>/eval/seqs_permute_<layers>/
# where <layers> is the --permute_attn_layers value used (a 0-indexed
# encoder-layer index, or "all").
#
# Invoked via `python -m gen.permute_attn` rather than a
# pmpnn-gen_permute_attn console script, so this works straight from
# source without needing the package reinstalled after pyproject.toml
# picked up the new entry point.
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_07_ablation_permute_attn.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

# Parallel arrays, flattened over (model, layer spec) — oa1_seed0 has 1
# encoder layer (so only the trivial "permute everything" case exists);
# oa3_seed0 and gat3_seed0 have 3, each swept individually plus "all".
MODEL_NAMES=(
    oa1_seed0
    oa3_seed0  oa3_seed0  oa3_seed0  oa3_seed0
    gat3_seed0 gat3_seed0 gat3_seed0 gat3_seed0
)
LAYER_SPECS=(
    all
    0 1 2 all
    0 1 2 all
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
LAYER_SPEC=${LAYER_SPECS[$SLURM_ARRAY_TASK_ID]}

case "${MODEL_NAME}" in
    oa1*)  MODE=oa;  LAYERS=1 ;;
    oa3*)  MODE=oa;  LAYERS=3 ;;
    gat3*) MODE=gat; LAYERS=3 ;;
    *) echo "Task ${SLURM_ARRAY_TASK_ID}: unrecognized model name ${MODEL_NAME}" >&2; exit 1 ;;
esac

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}, MODE=${MODE}, LAYERS=${LAYERS}, PERMUTE_ATTN_LAYERS=${LAYER_SPEC}"

MODEL_DIR="models/exp2_finetuning/${MODEL_NAME}"
WEIGHTS="${MODEL_DIR}/model_weights/best.pt"
OUTPUT_DIR="${MODEL_DIR}/eval/seqs_permute_${LAYER_SPEC}"

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
    --permute_attn_layers "${LAYER_SPEC}"
)

if [ "${MODE}" = "oa" ]; then
    ARGS+=(--r_max_orb 12.0)
fi

PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" python -m gen.permute_attn "${ARGS[@]}"
