#!/bin/bash
#SBATCH --job-name=pmpnn-exp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH --array=1-59

# Experiment 1: data-fraction scaling sweep.
#
# Grid (4 model configs x 5 data fractions x 3 seeds = 60 runs):
#   configs:   oa1 (mode=oa, 1 encoder layer), oa3 (mode=oa, 3 encoder layers),
#              vanilla3 (mode=vanilla, 3 layers), gat3 (mode=gat, 3 layers)
#   fractions: 0.1 0.2 0.4 0.8 1.0   (train_cluster_fraction)
#   seeds:     0 1 2
#
# All configs train from scratch (no --from_pretrained) for a fair
# data-efficiency comparison. Each run trains for up to 200 epochs with
# the model's default early stopping (patience=20).
#
# Submit from the repo root: sbatch scripts/exp1_data_fraction_01_train.sbatch
# The production partition has 6 schedulable GPUs (4x A5000, 2x A4000), so
# the array is throttled to 6 concurrent tasks (%6).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p logs models/exp1_data_fraction

source .venv/bin/activate

echo "=== System GPU Discovery ==="
if command -v lspci &> /dev/null; then
    lspci | grep -i nvidia || echo "No NVIDIA PCI devices visible."
else
    echo "lspci command not available."
fi

echo -e "\n=== NVIDIA Driver & Device Status (nvidia-smi) ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,uuid --format=csv
    echo ""
    nvidia-smi
else
    echo "ERROR: nvidia-smi not found. NVIDIA drivers may not be installed or CUDA bin is not in PATH."
fi

echo -e "\n=== Slurm Environment Variables ==="
env | grep -E "CUDA_|SLURM_GPU|GPU" || echo "No Slurm/CUDA GPU environment variables set."

echo -e "\n=== PyTorch GPU Test (via uv) ==="
if command -v uv &> /dev/null; then
    uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available in PyTorch: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    try:
        x = torch.ones(1).cuda()
        print('Tensor allocation test: SUCCESS')
    except Exception as e:
        print(f'Tensor allocation test: FAILED ({e})')
else:
    print('PyTorch cannot access GPU.')
" 2>/dev/null || echo "PyTorch execution via uv failed or package is missing."
else
    echo "ERROR: uv command not found in PATH."
fi
echo "=== End GPU Discovery ==="

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

OUTPUT_DIR="models/exp1_data_fraction/${NAME}_frac${FRACTION}_seed${SEED}"

if [ -f "${OUTPUT_DIR}/model_weights/best.pt" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${OUTPUT_DIR}/model_weights/best.pt already exists, skipping."
    exit 0
fi
rm -rf "${OUTPUT_DIR}"

ARGS=(
    data/pdb_2021aug02.h5 data/list.csv data/valid_clusters.txt data/test_clusters.txt
    "${OUTPUT_DIR}"
    --mode "${MODE}"
    --num_encoder_layers "${LAYERS}"
    --train_cluster_fraction "${FRACTION}"
    --seed "${SEED}"
    --num_epochs 200
    --num_examples_per_epoch 10000
    --batch_size 10000
    --mini_batch_size 10000
    --wandb_project protein-mpnn-oa-exp1-data-fraction
)

if [ "${MODE}" = "oa" ]; then
    ARGS+=(--r_max_orb 12.0)
fi

pmpnn-train "${ARGS[@]}"
