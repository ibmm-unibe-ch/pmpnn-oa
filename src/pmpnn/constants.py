"""
constants.py
============
Central registry of constants shared across the pmpnn library.

Categories
----------
1. Protein Alphabet & Sequence   — amino-acid and chain-label alphabets
2. PMPNN Architecture            — default hyper-parameters for ProteinMPNN layers
3. OA / GAT Geometry             — defaults for the Orbital Attention and GAT modules
4. HP Training                   — optimizer, loss, and training-loop defaults
5. DATA                          — structure dataset filtering thresholds
6. DATALOADER                    — DataLoader / streaming-buffer defaults
7. METRICS                       — evaluation matrices (BLOSUM62)
"""

import numpy as np

# =============================================================================
# 1. Protein Alphabet & Sequence
# =============================================================================
AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C",
}
NUM_AA: int = len(AA_ALPHABET)

_CHAIN_INIT  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_CHAIN_EXTRA = [str(i) for i in range(300)]
CHAIN_ALPHABET: list = _CHAIN_INIT + _CHAIN_EXTRA


# =============================================================================
# 2. PMPNN Architecture defaults
# =============================================================================
PMPNN_HIDDEN_DIM: int = 128
PMPNN_NUM_ENCODER_LAYERS: int = 3
PMPNN_NUM_DECODER_LAYERS: int = 3
PMPNN_K_NEIGHBORS: int = 48
PMPNN_DROPOUT: float = 0.1
PMPNN_AUGMENT_EPS: float = 0.05
PMPNN_MAX_RELATIVE_FEATURE: int = 32
PMPNN_NUM_POSITIONAL_EMBEDDINGS: int = 16
PMPNN_NUM_RBF: int = 16
PMPNN_RBF_D_MIN: float = 2.0
PMPNN_RBF_D_MAX: float = 22.0
PMPNN_ENC_LAYER_SCALE: int = 30
PMPNN_FF_EXPANSION: int = 4
PMPNN_AA_EMBED_DIM: int = 16
PMPNN_NUM_INTERFACE_TYPES: int = 5
PMPNN_MAX_CHAINS: int = 352
PMPNN_DIST_EPS: float = 1e-6
PMPNN_CB_COEFF_A: float = -0.58273431
PMPNN_CB_COEFF_B: float =  0.56802827
PMPNN_CB_COEFF_C: float = -0.54067466


# =============================================================================
# 3. OA / GAT Geometry defaults
# =============================================================================
OA_NUM_RADIAL: int = 8
OA_R_MAX: float = 8.0
OA_P_CUTOFF: int = 5
OA_L_MAX: int = 2
OA_WEIGHT_MLP_HIDDEN_DIM: int = 128
OA_GATE_HIDDEN_DIM: int = 32
OA_BESSEL_EPS: float = 1e-8
OA_MASK_FILL_VALUE: float = -1e9
GAT_LEAKY_RELU_SLOPE: float = 0.2


# =============================================================================
# 4. HP — Optimizer & Training defaults
# =============================================================================
HP_WANDB_PROJECT: str = "protein-mpnn-oa"
HP_NOAM_FACTOR: int = 2
HP_NOAM_WARMUP: int = 4000
HP_ADAM_BETAS: tuple = (0.9, 0.98)
HP_ADAM_EPS: float = 1e-9
HP_LABEL_SMOOTHING_WEIGHT: float = 0.1
HP_LOSS_SMOOTHED_NORM: float = 2000.0
HP_FOCAL_GAMMA: float = 2.0
HP_NUM_EPOCHS: int = 200
HP_BATCH_SIZE: int = 10_000
HP_MINI_BATCH_SIZE: int = 10_000
HP_SAVE_EVERY_N_EPOCHS: int = 10
HP_PATIENCE: int = 20
HP_MIN_DELTA: float = 1e-4
HP_NUM_EXAMPLES_PER_EPOCH: int = 1_000_000
HP_SEED: int = 42
HP_BACKBONE_NOISE: float = 0.2
HP_GRADIENT_NORM: float = -1.0
# Default encoder mode. Choices: "vanilla", "oa", "gat".
HP_MODE: str = "oa"
# Default parameter freeze mode. Choices: "none", "freeze_encoder", "freeze_decoder", "freeze".
HP_FREEZE_MODE: str = "none"


# =============================================================================
# 5. DATA — Structure dataset defaults
# =============================================================================
DATA_MAX_PROTEIN_LENGTH: int = 10_000
DATA_RESCUT: float = 3.5
DATA_TRAIN_CLUSTER_FRACTION: float = 1.0
DATA_HOMO_THRESHOLD: float = 0.70
DATA_CA_BOND_MIN: float = 3.6
DATA_CA_BOND_MAX: float = 4.0
DATA_DATCUT: str = "2030-Jan-01"


# =============================================================================
# 6. DATALOADER — DataLoader & streaming-buffer defaults
# =============================================================================
DATALOADER_BATCH_SIZE: int = 32
DATALOADER_NUM_WORKERS: int = 8
DATALOADER_STREAMING_BUFFER_SIZE: int = 2_000


# =============================================================================
# 7. METRICS — Evaluation matrices
# =============================================================================
# Full 24x24 BLOSUM62 matrix in NCBI ordering (A C D E F G H I K L M N P Q R
# S T V W Y B Z X *).  Reindexed to AA_ALPHABET order via BLOSUM62_AA_INDEX.
_BLOSUM62_FULL: np.ndarray = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4],
    [-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4],
    [-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
    [ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4],
    [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1],
], dtype=np.int32)

# Maps each AA_ALPHABET position to the corresponding row/col in _BLOSUM62_FULL.
# A=0 C=4 D=3 E=6 F=13 G=7 H=8 I=9 K=11 L=10 M=12 N=2 P=14 Q=5 R=1 S=15
# T=16 V=19 W=17 Y=18 X=22
BLOSUM62_AA_INDEX: np.ndarray = np.array(
    [0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 22],
    dtype=np.int32,
)

# BLOSUM62 submatrix in AA_ALPHABET order, shape (NUM_AA, NUM_AA).
BLOSUM62: np.ndarray = _BLOSUM62_FULL[np.ix_(BLOSUM62_AA_INDEX, BLOSUM62_AA_INDEX)]
