from __future__ import annotations
import argparse
import os
from dataclasses import dataclass, field

from pmpnn.constants import (
    DATA_MAX_PROTEIN_LENGTH,
    DATA_RESCUT,
    DATA_TRAIN_CLUSTER_FRACTION,
    HP_BACKBONE_NOISE,
    HP_BATCH_SIZE,
    HP_EVAL_SEED,
    HP_FREEZE_MODE,
    HP_GRADIENT_NORM,
    HP_MIN_DELTA,
    HP_MINI_BATCH_SIZE,
    HP_MODE,
    HP_NUM_EPOCHS,
    HP_NUM_EXAMPLES_PER_EPOCH,
    HP_PATIENCE,
    HP_SAVE_EVERY_N_EPOCHS,
    HP_SEED,
    HP_WANDB_PROJECT,
    OA_GATE_HIDDEN_DIM,
    OA_L_MAX,
    OA_NUM_RADIAL,
    OA_P_CUTOFF,
    OA_R_MAX,
    OA_WEIGHT_MLP_HIDDEN_DIM,
    PMPNN_DROPOUT,
    PMPNN_HIDDEN_DIM,
    PMPNN_K_NEIGHBORS,
    PMPNN_NUM_DECODER_LAYERS,
    PMPNN_NUM_ENCODER_LAYERS,
)


@dataclass
class Context:
    # ── data ──────────────────────────────────────────────────────────────────
    h5_file: str
    h5_in_memory: bool
    index: str
    val_clu_index: str
    test_clu_index: str
    out_dir: str
    num_examples_per_epoch: int
    max_protein_length: int
    rescut: float
    train_cluster_fraction: float
    debug: bool

    # ── model ─────────────────────────────────────────────────────────────────
    from_pretrained: str
    hidden_dim: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_neighbors: int
    dropout: float
    backbone_noise: float
    mode: str
    freeze_mode: str
    num_radial_orb: int
    r_max_orb: float
    p_cutoff: int
    l_max: int
    weight_mlp_hidden_dim: int
    gate_hidden_dim: int

    # ── training ──────────────────────────────────────────────────────────────
    num_epochs: int
    save_model_every_n_epochs: int
    batch_size: int
    mini_batch_size: int
    gradient_norm: float
    mixed_precision: bool

    # ── early stopping ────────────────────────────────────────────────────────
    patience: int
    min_delta: float

    # ── logging ───────────────────────────────────────────────────────────────
    wandb_project: str
    use_tqdm: bool

    # ── reproducibility ───────────────────────────────────────────────────────
    seed: int = HP_SEED
    val_seed: int = HP_EVAL_SEED

    # ── runtime state (populated during training, not from CLI) ───────────────
    device:       object = field(default=None, repr=False)
    scaler:       object = field(default=None, repr=False)
    model:        object = field(default=None, repr=False)
    optimizer:    object = field(default=None, repr=False)
    train_loader: object = field(default=None, repr=False)
    valid_loader: object = field(default=None, repr=False)
    base_folder:  str    = field(default="",   repr=False)
    model_name:   str    = field(default="",   repr=False)
    logfile:      str    = field(default="",   repr=False)
    total_step:   int    = field(default=0,    repr=False)
    start_epoch:  int    = field(default=0,    repr=False)

    def __post_init__(self):
        # ── paths ─────────────────────────────────────────────────────────────
        if not os.path.isfile(self.h5_file):
            raise ValueError(f"h5_file does not exist: {self.h5_file!r}")
        if not os.path.isfile(self.index):
            raise ValueError(f"index does not exist: {self.index!r}")
        if not os.path.isfile(self.val_clu_index):
            raise ValueError(f"val_clu_index does not exist: {self.val_clu_index!r}")
        if not os.path.isfile(self.test_clu_index):
            raise ValueError(f"test_clu_index does not exist: {self.test_clu_index!r}")
        if self.from_pretrained and not os.path.isfile(self.from_pretrained):
            raise ValueError(f"from_pretrained does not exist: {self.from_pretrained!r}")

        output_parent = os.path.dirname(os.path.abspath(self.out_dir))
        if not os.path.isdir(output_parent):
            raise ValueError(f"parent directory of out_dir does not exist: {output_parent!r}")
        if os.path.exists(self.out_dir):
            raise ValueError(f"out_dir already exists, refusing to overwrite: {self.out_dir!r}")

        # ── numeric bounds ────────────────────────────────────────────────────
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.backbone_noise < 0.0:
            raise ValueError(f"backbone_noise must be >= 0, got {self.backbone_noise}")
        if self.rescut <= 0.0:
            raise ValueError(f"rescut must be > 0, got {self.rescut}")
        if not (0.0 < self.train_cluster_fraction <= 1.0):
            raise ValueError(
                f"train_cluster_fraction must be in (0, 1], got {self.train_cluster_fraction}")

        if self.mode == "oa" and self.r_max_orb <= 0.0:
            raise ValueError(f"r_max_orb must be > 0, got {self.r_max_orb}")

        for name, val in [
            ("num_examples_per_epoch",    self.num_examples_per_epoch),
            ("max_protein_length",        self.max_protein_length),
            ("hidden_dim",                self.hidden_dim),
            ("num_encoder_layers",        self.num_encoder_layers),
            ("num_decoder_layers",        self.num_decoder_layers),
            ("num_neighbors",             self.num_neighbors),
            ("num_epochs",                self.num_epochs),
            ("save_model_every_n_epochs", self.save_model_every_n_epochs),
            ("batch_size",                self.batch_size),
            ("mini_batch_size",           self.mini_batch_size),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

        if self.batch_size % self.mini_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by "
                f"mini_batch_size ({self.mini_batch_size})"
            )

        if self.mode == "oa":
            for name, val in [
                ("num_radial_orb",        self.num_radial_orb),
                ("p_cutoff",              self.p_cutoff),
                ("l_max",                 self.l_max),
                ("weight_mlp_hidden_dim", self.weight_mlp_hidden_dim),
                ("gate_hidden_dim",       self.gate_hidden_dim),
            ]:
                if val <= 0:
                    raise ValueError(f"{name} must be > 0, got {val}")

    @classmethod
    def from_args(cls) -> "Context":
        p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        p.add_argument("h5_file",        help="Path to the HDF5 structure file produced by pt_to_h5.py")
        p.add_argument("index",          help="Path to the CSV index file (list.csv)")
        p.add_argument("val_clu_index",  help="Path to the validation cluster index file")
        p.add_argument("test_clu_index", help="Path to the test cluster index file")
        p.add_argument("out_dir",        help="Output directory for logs and model weights (must not already exist)")

        data = p.add_argument_group("Data")
        data.add_argument("--h5_in_memory", action="store_true",
                          help="Load the whole HDF5 file into RAM (h5py driver='core', "
                               "backing_store=False, libver='latest') instead of reading it "
                               "from disk on demand. Each DataLoader worker process loads its "
                               "own copy, so RAM use scales with --num_workers.")
        data.add_argument("--num_examples_per_epoch", type=int, default=HP_NUM_EXAMPLES_PER_EPOCH,
                          help="Number of sequence clusters sampled per epoch")
        data.add_argument("--max_protein_length", type=int, default=DATA_MAX_PROTEIN_LENGTH,
                          help="Maximum protein length in residues; longer proteins are discarded")
        data.add_argument("--rescut", type=float, default=DATA_RESCUT,
                          help="Maximum resolution cutoff (Å) for filtering structures")
        data.add_argument("--train_cluster_fraction", type=float, default=DATA_TRAIN_CLUSTER_FRACTION,
                          help="Fraction of training clusters to keep (in (0, 1]). Clusters are "
                               "sorted ascending by number of member entries and the smallest "
                               "ceil(fraction * N) are kept; 1.0 uses the full training set. "
                               "Applied after the val/test split, before sampling.")
        data.add_argument("--debug", action="store_true",
                          help="Enable debug mode: reduces dataset size and protein length for fast iteration")

        model = p.add_argument_group("Model")
        model.add_argument("--from_pretrained", type=str, default="",
                           help="Path to pretrained model weights (.pt) to initialise from")
        model.add_argument("--hidden_dim", type=int, default=PMPNN_HIDDEN_DIM,
                           help="Hidden dimension of node and edge features")
        model.add_argument("--num_encoder_layers", type=int, default=PMPNN_NUM_ENCODER_LAYERS,
                           help="Number of encoder layers")
        model.add_argument("--num_decoder_layers", type=int, default=PMPNN_NUM_DECODER_LAYERS,
                           help="Number of decoder layers")
        model.add_argument("--num_neighbors", type=int, default=PMPNN_K_NEIGHBORS,
                           help="Number of k-nearest neighbours in the protein graph")
        model.add_argument("--dropout", type=float, default=PMPNN_DROPOUT,
                           help="Dropout probability applied throughout the model")
        model.add_argument("--backbone_noise", type=float, default=HP_BACKBONE_NOISE,
                           help="Std of Gaussian noise added to backbone coordinates during training")
        model.add_argument("--mode", type=str, default=HP_MODE,
                           choices=["vanilla", "oa", "gat"],
                           help="vanilla: standard MPNN; oa: OA with degree gating; gat: edge-driven GAT without geometry")
        model.add_argument("--freeze_mode", type=str, default=HP_FREEZE_MODE,
                           choices=["none", "freeze_encoder", "freeze_decoder", "freeze"],
                           help="none: train all; freeze_encoder/freeze_decoder: freeze one side; freeze: freeze both")

        oa = p.add_argument_group("Orbital Attention Architecture")
        oa.add_argument("--num_radial_orb", type=int, default=OA_NUM_RADIAL,
                        help="Number of radial Bessel basis functions in the OA distance embedding")
        oa.add_argument("--r_max_orb", type=float, default=OA_R_MAX,
                        help="Cutoff radius (Å) for the OA neighbourhood")
        oa.add_argument("--p_cutoff", type=int, default=OA_P_CUTOFF,
                        help="Degree of the polynomial envelope for the smooth radial cutoff in OA")
        oa.add_argument("--l_max", type=int, default=OA_L_MAX,
                        help="Maximum spherical harmonic degree l in OA")
        oa.add_argument("--weight_mlp_hidden_dim", type=int, default=OA_WEIGHT_MLP_HIDDEN_DIM,
                        help="Hidden dimension of the MLP that produces OA edge weights")
        oa.add_argument("--gate_hidden_dim", type=int, default=OA_GATE_HIDDEN_DIM,
                        help="Hidden dimension of the degree-gating MLP in OA")

        train = p.add_argument_group("Training")
        train.add_argument("--num_epochs", type=int, default=HP_NUM_EPOCHS,
                           help="Total number of training epochs")
        train.add_argument("--save_model_every_n_epochs", type=int, default=HP_SAVE_EVERY_N_EPOCHS,
                           help="Save a named checkpoint every N epochs")
        train.add_argument("--batch_size", type=int, default=HP_BATCH_SIZE,
                           help="Number of tokens per effective batch (after gradient accumulation)")
        train.add_argument("--mini_batch_size", type=int, default=HP_MINI_BATCH_SIZE,
                           help="Number of tokens per mini-batch; accumulation steps = batch_size // mini_batch_size")
        train.add_argument("--gradient_norm", type=float, default=HP_GRADIENT_NORM,
                           help="Max gradient norm for clipping; negative value disables clipping")
        train.add_argument("--no_mixed_precision", action="store_false", dest="mixed_precision",
                           help="Disable automatic mixed precision (AMP) training")
        p.set_defaults(mixed_precision=True)

        es = p.add_argument_group("Early Stopping")
        es.add_argument("--patience", type=int, default=HP_PATIENCE,
                        help="Stop training after this many epochs without sufficient improvement")
        es.add_argument("--min_delta", type=float, default=HP_MIN_DELTA,
                        help="Minimum relative improvement in val loss to count as progress "
                             "(improvement = (best - current) / best > min_delta)")

        p.add_argument("--seed", type=int, default=HP_SEED,
                       help="Global random seed for reproducibility")
        p.add_argument("--val_seed", type=int, default=HP_EVAL_SEED,
                       help="Seed for the validation-time decoding-order randn draw, "
                            "re-applied at the start of every validation epoch. Deliberately "
                            "independent of --seed, so models trained with different --seed "
                            "values are validated under identical randomness")

        wandb = p.add_argument_group("Logging")
        wandb.add_argument("--wandb_project", type=str, default=HP_WANDB_PROJECT,
                           help="Weights & Biases project name to log runs under")
        wandb.add_argument("--no_tqdm", action="store_false", dest="use_tqdm",
                           help="Disable tqdm progress bars during training")
        p.set_defaults(use_tqdm=True)

        args = p.parse_args()
        return cls(**vars(args))
