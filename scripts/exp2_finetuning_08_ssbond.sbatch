#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-ssbond
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=08:00:00
#SBATCH --array=0-2

# Cys-Cys SSBOND attention analysis (src/vis/ss_bond_analysis.py) of the
# Experiment 2 (finetuning) oa1_seed0/1/2 models, on the full raw PDB
# entries behind the held-out test monomers/heteromers/homomers splits
# (data/test_pdb_raw — ~1500 structures, every chain and every SSBOND
# record kept, intra- and inter-chain alike; the HDF5 training format
# used by pmpnn-stats/pmpnn-gen drops header/connectivity records
# entirely, and the single-chain-per-file data/test_*_pdb splits — see
# scripts/download_test_monomers.py — keep only ~800 of the ~4000 SSBOND
# records, dropping every inter-chain bond and every chain with none of
# its own).
#
# Each array task runs both subcommands for one seed:
#   compute  - forward pass over every structure, restrict to non-self
#              Cys-Cys edges, and save the per-edge budget-weighted
#              attention score + SSBOND label to
#              models/exp2_finetuning/<model_name>/eval/ssbond.npz
#   analyze  - load that .npz and print the statistical analysis (AUC /
#              effect size of score vs. SSBOND, stratified by Ca-Ca and
#              Cb-Cb distance, with a protein-level cluster bootstrap 95%
#              CI) plus the matched-control figure and the same per-bin
#              tables as a CSV, saved to
#              models/exp2_finetuning/<model_name>/eval/ssbond_matched.svg
#              and .../ssbond_tables.csv
#
# oa1_seed0/1/2 are single-encoder-layer OA models, so --layer 0 is the
# (only) layer analyzed. 'analyze' is CPU-only and cheap, so it always
# re-runs (even if ssbond.npz already existed from a previous submission)
# — only 'compute' (the GPU forward pass) is skipped once its .npz exists.
# The printed per-bin tables and pooled AUC/effect land in this task's
# slurm log.
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_08_ssbond.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs

source .venv/bin/activate

MODEL_NAMES=(oa1_seed0 oa1_seed1 oa1_seed2)
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL_NAME=${MODEL_NAME}"

MODEL_DIR="models/exp2_finetuning/${MODEL_NAME}"
WEIGHTS="${MODEL_DIR}/model_weights/best.pt"
OUTPUT_DIR="${MODEL_DIR}/eval"
NPZ="${OUTPUT_DIR}/ssbond.npz"
FIG="${OUTPUT_DIR}/ssbond_matched.svg"
TABLES="${OUTPUT_DIR}/ssbond_tables.csv"

if [ ! -f "${WEIGHTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${WEIGHTS} does not exist, skipping." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [ -s "${NPZ}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${NPZ} already exists, skipping compute."
else
    pmpnn-ssbond compute \
        data/test_pdb_raw \
        "${WEIGHTS}" "${NPZ}" \
        --mode oa \
        --num_encoder_layers 1 \
        --r_max_orb 12.0 \
        --device cuda
fi

pmpnn-ssbond analyze "${NPZ}" --layer 0 --fig_out "${FIG}" --tables_out "${TABLES}"
