#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-gen-seqdist-attn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3

# Sequence-distance-blind ablation (src/gen/seqdist_attn.py) of the
# Experiment 2 (finetuning) oa1_seed0, oa3_seed0, gat3_seed0 and vanilla3
# models, on the held-out test monomers (data/test_monomers.txt). Hooks
# ProteinFeatures.embeddings (the PositionalEncodings submodule) and
# overwrites every edge's true sequence-distance offset
# (residue_idx[i] - residue_idx[j]) with a fixed constant
# (--seqdist_offset, default 6) before it's clamped/one-hot/linearly
# embedded. Tests whether the model is actually using the relative
# sequence-position information carried by that offset, as opposed to
# only the spatial (distance/orientation) information carried by the
# other edge features.
#
# Unlike scripts/exp2_finetuning_07_ablation_isotropic_attn.sbatch,
# scripts/exp2_finetuning_07_ablation_uniform_attn.sbatch and
# scripts/exp2_finetuning_07_ablation_permute_attn.sbatch, this ablation has
# nothing to do with per-layer attention — PositionalEncodings runs once
# per forward pass, before the encoder stack, and feeds into every
# mode's edge features. So there is no per-layer sweep (no
# --seqdist_attn_layers option exists) and, unlike those three sweeps,
# vanilla3 IS included here: it has no attention module to ablate, but
# it still uses PositionalEncodings for its edge features. 4 tasks total
# (array 0-3), one per model.
#
# This is a hooks-based ablation (see register_seqdist_hook in
# src/gen/seqdist_attn.py) — no unablated baseline is run here, since
# that's already covered by models/exp2_finetuning/<model_name>/eval/seqs/
# from scripts/exp2_finetuning_02_gen_monomers.sbatch.
#
# Each array task writes one FASTA file per structure to
# models/exp2_finetuning/<model_name>/eval/seqs_seqdist_<offset>/
# where <offset> is the --seqdist_offset value used.
#
# Invoked via `python -m gen.seqdist_attn` rather than a
# pmpnn-gen_seqdist_attn console script, so this works straight from
# source without needing the package reinstalled after pyproject.toml
# picked up the new entry point.
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_07_ablation_seqdist_attn.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

SEQDIST_OFFSET=6

MODEL_NAMES=(
    oa1_seed0
    oa3_seed0
    gat3_seed0
    vanilla3
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

case "${MODEL_NAME}" in
    oa1*)      MODE=oa;      LAYERS=1 ;;
    oa3*)      MODE=oa;      LAYERS=3 ;;
    gat3*)     MODE=gat;     LAYERS=3 ;;
    vanilla3*) MODE=vanilla; LAYERS=3 ;;
    *) echo "Task ${SLURM_ARRAY_TASK_ID}: unrecognized model name ${MODEL_NAME}" >&2; exit 1 ;;
esac

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}, MODE=${MODE}, LAYERS=${LAYERS}, SEQDIST_OFFSET=${SEQDIST_OFFSET}"

MODEL_DIR="models/exp2_finetuning/${MODEL_NAME}"
WEIGHTS="${MODEL_DIR}/model_weights/best.pt"
OUTPUT_DIR="${MODEL_DIR}/eval/seqs_seqdist_${SEQDIST_OFFSET}"

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
    --seqdist_offset "${SEQDIST_OFFSET}"
)

if [ "${MODE}" = "oa" ]; then
    ARGS+=(--r_max_orb 12.0)
fi

PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" python -m gen.seqdist_attn "${ARGS[@]}"
