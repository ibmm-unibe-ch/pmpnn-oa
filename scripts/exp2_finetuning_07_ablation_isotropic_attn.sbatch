#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-gen-isotropic-attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --array=0-4

# Isotropic (angular-blind) ablation (src/gen/isotropic_attn.py) of the
# Experiment 2 (finetuning) oa1_seed0 and oa3_seed0 models, on the
# held-out test monomers (data/test_monomers.txt). Unlike
# scripts/exp2_finetuning_07_ablation_uniform_attn.sbatch and
# scripts/exp2_finetuning_07_ablation_permute_attn.sbatch (which rewrite the
# post-softmax attention weights), this hooks OAAttention.degree_gate
# directly and zeroes its output for every spherical-harmonic degree
# l >= 1 — since l=0 is a direction-independent constant, this collapses
# the attention score to a function of Ca-Ca distance alone, with no
# angular/directional dependence. Tests whether the model is actually
# using that angular information, as opposed to only the isotropic,
# distance-only part of it.
#
# Only --mode oa has a degree_gate (OAAttention); gat3_seed0 has none and
# is therefore not included here (unlike the uniform/permute sweeps,
# which both also cover gat3_seed0's edge-driven attention).
#
# Runs the whole per-layer ablation sweep for each model, not just the
# fully-ablated case:
#   oa1_seed0 (1 encoder layer)  -> ablate the only layer      -> 1 run
#   oa3_seed0 (3 encoder layers) -> ablate each layer alone
#                                   (0, 1, 2) + all three       -> 4 runs
# so the per-layer runs show which individual layer's angular
# information matters, and the "all" run shows the fully-ablated model
# (the oa1 case is already "all" since it only has one layer).
# 1 + 4 = 5 tasks total (array 0-4).
#
# This is a hooks-based ablation (see register_isotropic_hooks in
# src/gen/isotropic_attn.py) — no unablated baseline is run here, since
# that's already covered by models/exp2_finetuning/<model_name>/eval/seqs/
# from scripts/exp2_finetuning_02_gen_monomers.sbatch.
#
# Each array task writes one FASTA file per structure to
# models/exp2_finetuning/<model_name>/eval/seqs_isotropic_<layers>/
# where <layers> is the --isotropic_attn_layers value used (a 0-indexed
# encoder-layer index, or "all").
#
# Invoked via `python -m gen.isotropic_attn` rather than a
# pmpnn-gen_isotropic_attn console script, so this works straight from
# source without needing the package reinstalled after pyproject.toml
# picked up the new entry point.
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_07_ablation_isotropic_attn.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

# Parallel arrays, flattened over (model, layer spec) — oa1_seed0 has 1
# encoder layer (so only the trivial "ablate everything" case exists);
# oa3_seed0 has 3, swept individually plus "all".
MODEL_NAMES=(
    oa1_seed0
    oa3_seed0 oa3_seed0 oa3_seed0 oa3_seed0
)
LAYER_SPECS=(
    all
    0 1 2 all
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
LAYER_SPEC=${LAYER_SPECS[$SLURM_ARRAY_TASK_ID]}

case "${MODEL_NAME}" in
    oa1*) MODE=oa; LAYERS=1 ;;
    oa3*) MODE=oa; LAYERS=3 ;;
    *) echo "Task ${SLURM_ARRAY_TASK_ID}: unrecognized model name ${MODEL_NAME}" >&2; exit 1 ;;
esac

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}, MODE=${MODE}, LAYERS=${LAYERS}, ISOTROPIC_ATTN_LAYERS=${LAYER_SPEC}"

MODEL_DIR="models/exp2_finetuning/${MODEL_NAME}"
WEIGHTS="${MODEL_DIR}/model_weights/best.pt"
OUTPUT_DIR="${MODEL_DIR}/eval/seqs_isotropic_${LAYER_SPEC}"

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
    --isotropic_attn_layers "${LAYER_SPEC}"
    --r_max_orb 12.0
)

PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" python -m gen.isotropic_attn "${ARGS[@]}"
