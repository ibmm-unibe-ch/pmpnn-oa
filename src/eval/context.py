from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from pmpnn.constants import (
    DATA_MAX_PROTEIN_LENGTH,
    DATA_RESCUT,
    DATALOADER_BATCH_SIZE,
    DATALOADER_NUM_WORKERS,
    HP_MINI_BATCH_SIZE,
    HP_MODE,
    HP_NUM_EXAMPLES_PER_EPOCH,
    HP_SEED,
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
class EvalContext:
    # ── data ──────────────────────────────────────────────────────────────────
    h5_file: str
    index: str
    val_clu_index: str
    test_clu_index: str
    from_pretrained: str
    num_examples_per_epoch: int
    max_protein_length: int
    rescut: float
    debug: bool

    # ── model ─────────────────────────────────────────────────────────────────
    hidden_dim: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_neighbors: int
    dropout: float
    backbone_noise: float
    mode: str
    num_radial_orb: int
    r_max_orb: float
    p_cutoff: int
    l_max: int
    weight_mlp_hidden_dim: int
    gate_hidden_dim: int

    # ── inference ─────────────────────────────────────────────────────────────
    mini_batch_size: int
    use_tqdm: bool

    # ── reproducibility ───────────────────────────────────────────────────────
    seed: int = HP_SEED

    # ── runtime state (populated during evaluation, not from CLI) ─────────────
    device:      object = field(default=None, repr=False)
    model:       object = field(default=None, repr=False)
    test_loader: object = field(default=None, repr=False)

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
        if not os.path.isfile(self.from_pretrained):
            raise ValueError(f"from_pretrained does not exist: {self.from_pretrained!r}")

        # ── numeric bounds ────────────────────────────────────────────────────
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.backbone_noise < 0.0:
            raise ValueError(f"backbone_noise must be >= 0, got {self.backbone_noise}")
        if self.rescut <= 0.0:
            raise ValueError(f"rescut must be > 0, got {self.rescut}")

        if self.mode == "oa" and self.r_max_orb <= 0.0:
            raise ValueError(f"r_max_orb must be > 0, got {self.r_max_orb}")

        for name, val in [
            ("num_examples_per_epoch", self.num_examples_per_epoch),
            ("max_protein_length",     self.max_protein_length),
            ("hidden_dim",             self.hidden_dim),
            ("num_encoder_layers",     self.num_encoder_layers),
            ("num_decoder_layers",     self.num_decoder_layers),
            ("num_neighbors",          self.num_neighbors),
            ("mini_batch_size",        self.mini_batch_size),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

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
    def from_args(cls) -> "EvalContext":
        p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        p.add_argument("h5_file",         help="Path to the HDF5 structure file produced by pt_to_h5.py")
        p.add_argument("index",           help="Path to the CSV index file (list.csv)")
        p.add_argument("val_clu_index",   help="Path to the validation cluster index file")
        p.add_argument("test_clu_index",  help="Path to the test cluster index file")
        p.add_argument("from_pretrained", help="Path to model weights (.pt) to evaluate")

        data = p.add_argument_group("Data")
        data.add_argument("--num_examples_per_epoch", type=int, default=HP_NUM_EXAMPLES_PER_EPOCH,
                          help="Maximum number of test clusters to evaluate")
        data.add_argument("--max_protein_length", type=int, default=DATA_MAX_PROTEIN_LENGTH,
                          help="Maximum protein length in residues; longer proteins are discarded")
        data.add_argument("--rescut", type=float, default=DATA_RESCUT,
                          help="Maximum resolution cutoff (Å) for filtering structures")
        data.add_argument("--debug", action="store_true",
                          help="Enable debug mode: reduces dataset size for fast iteration")

        model = p.add_argument_group("Model")
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
        model.add_argument("--backbone_noise", type=float, default=0.0,
                           help="Std of Gaussian noise added to backbone coordinates "
                                "(set to 0 for clean evaluation)")
        model.add_argument("--mode", type=str, default=HP_MODE,
                           choices=["vanilla", "oa", "gat"],
                           help="vanilla: standard MPNN; oa: OA with degree gating; "
                                "gat: edge-driven GAT without geometry")

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

        inf = p.add_argument_group("Inference")
        inf.add_argument("--mini_batch_size", type=int, default=HP_MINI_BATCH_SIZE,
                         help="Number of tokens per mini-batch for teacher-forced evaluation")
        inf.add_argument("--no_tqdm", action="store_false", dest="use_tqdm",
                         help="Disable tqdm progress bars during evaluation")
        p.set_defaults(use_tqdm=True)

        p.add_argument("--seed", type=int, default=HP_SEED,
                       help="Global random seed for reproducibility")

        args = p.parse_args()
        return cls(**vars(args))
