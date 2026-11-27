#!/bin/bash
#SBATCH --job-name=pmpnn-exp2-dist_corr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32g
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=08:00:00
#SBATCH --array=0-2

# Budget-weighted attention score vs. backbone atom-pair distance analysis
# (src/vis/dist_corr.py) of the Experiment 2 (finetuning) oa1_seed0/1/2
# models, on the held-out test monomers (data/test_monomers.txt) — same
# HDF5/pdb_list input pipeline as scripts/exp2_finetuning_05_stats.sbatch
# (pmpnn-stats), not the raw-PDB-directory pipeline ssbond_exp2_finetuning
# uses (this analysis needs no SSBOND header annotations).
#
# Each array task runs both subcommands for one seed:
#   compute  - forward pass over every structure, and save every non-self
#              edge's full 5x5 (N/Ca/C/O/Cb) x (N/Ca/C/O/Cb) pairwise
#              atom-distance matrix plus the per-layer budget-weighted
#              attention score to
#              models/exp2_finetuning/<model_name>/eval/dist_corr.npz
#   analyze  - load that .npz, bin by Ca-Ca distance into the 4 fixed
#              bands (4-6, 6-8, 8-10, 10-12 A), and report the Pearson /
#              Spearman correlation of score vs. every one of the 25
#              atom-pair distances within each band, with a protein-level
#              cluster bootstrap 95% CI — saved to
#              models/exp2_finetuning/<model_name>/eval/dist_corr.csv and
#              .../fig_dist_corr_layer1.svg
#
# oa1_seed0/1/2 are single-encoder-layer OA models, so only layer 1 is
# analyzed (--layers is left at its default: every layer in the .npz).
# 'analyze' is CPU-only and cheap, so it always re-runs (even if
# dist_corr.npz already existed from a previous submission) — only
# 'compute' (the GPU forward pass) is skipped once its .npz exists. The
# printed per-bin report lands in this task's slurm log.
#
# Submit from the repo root: sbatch scripts/exp2_finetuning_08_dist_corr.sbatch

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
NPZ="${OUTPUT_DIR}/dist_corr.npz"
TABLE="${OUTPUT_DIR}/dist_corr.csv"

if [ ! -f "${WEIGHTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${WEIGHTS} does not exist, skipping." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [ -s "${NPZ}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${NPZ} already exists, skipping compute."
else
    pmpnn-dist_corr compute \
        data/pdb_2021aug02.h5 data/test_monomers.txt \
        "${WEIGHTS}" "${NPZ}" \
        --mode oa \
        --num_encoder_layers 1 \
        --r_max_orb 12.0 \
        --device cuda
fi

pmpnn-dist_corr analyze "${NPZ}" --out "${TABLE}" --out_dir "${OUTPUT_DIR}"
