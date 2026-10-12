from __future__ import annotations
import argparse
import os
from dataclasses import dataclass, field

from pmpnn.constants import (
    DATA_MAX_PROTEIN_LENGTH,
    HP_MODE,
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
class GenerationContext:
    # ── I/O ───────────────────────────────────────────────────────────────────
    pdb_path: str
    out_folder: str
    checkpoint_path: str
    h5_file: str = ""   # when set, pdb_path is interpreted as PDB IDs loaded from the HDF5

    # ── Generation ────────────────────────────────────────────────────────────
    num_seq_per_target: int = 1
    sampling_temp: str = "0.1"
    backbone_noise: float = 0.0
    batch_size: int = 1
    max_length: int = DATA_MAX_PROTEIN_LENGTH
    mixed_precision: bool = True

    # ── Chain / position control ──────────────────────────────────────────────
    pdb_path_chains: str = ""
    chain_id_jsonl: str = ""
    fixed_positions_jsonl: str = ""
    tied_positions_jsonl: str = ""

    # ── Amino acid constraints ────────────────────────────────────────────────
    omit_AAs: str = "X"
    bias_AA_jsonl: str = ""
    bias_by_res_jsonl: str = ""
    omit_AA_jsonl: str = ""

    # ── PSSM ──────────────────────────────────────────────────────────────────
    pssm_jsonl: str = ""
    pssm_multi: float = 0.0
    pssm_threshold: float = 0.0
    pssm_log_odds_flag: bool = False
    pssm_bias_flag: bool = False

    # ── Scoring modes ─────────────────────────────────────────────────────────
    score_only: bool = False
    conditional_probs_only: bool = False
    conditional_probs_only_backbone_noise: float = 0.0
    unconditional_probs_only: bool = False

    # ── Model architecture ────────────────────────────────────────────────────
    hidden_dim: int = PMPNN_HIDDEN_DIM
    num_encoder_layers: int = PMPNN_NUM_ENCODER_LAYERS
    num_decoder_layers: int = PMPNN_NUM_DECODER_LAYERS
    num_neighbors: int = PMPNN_K_NEIGHBORS
    dropout: float = PMPNN_DROPOUT
    mode: str = HP_MODE
    num_radial_orb: int = OA_NUM_RADIAL
    r_max_orb: float = OA_R_MAX
    p_cutoff: int = OA_P_CUTOFF
    l_max: int = OA_L_MAX
    weight_mlp_hidden_dim: int = OA_WEIGHT_MLP_HIDDEN_DIM
    gate_hidden_dim: int = OA_GATE_HIDDEN_DIM

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = HP_SEED

    # ── Runtime state (not from CLI) ──────────────────────────────────────────
    device: object = field(default=None, repr=False)
    model:  object = field(default=None, repr=False)

    def __post_init__(self):
        if self.h5_file:
            if not os.path.isfile(self.h5_file):
                raise ValueError(f"h5_file does not exist: {self.h5_file!r}")
            # pdb_path is a PDB ID or a text file of IDs; validated at runtime
        else:
            if not os.path.exists(self.pdb_path):
                raise ValueError(f"pdb_path does not exist: {self.pdb_path!r}")
        if not os.path.isfile(self.checkpoint_path):
            raise ValueError(f"checkpoint_path does not exist: {self.checkpoint_path!r}")
        for name, path in [
            ("chain_id_jsonl",        self.chain_id_jsonl),
            ("fixed_positions_jsonl", self.fixed_positions_jsonl),
            ("tied_positions_jsonl",  self.tied_positions_jsonl),
            ("bias_AA_jsonl",         self.bias_AA_jsonl),
            ("bias_by_res_jsonl",     self.bias_by_res_jsonl),
            ("omit_AA_jsonl",         self.omit_AA_jsonl),
            ("pssm_jsonl",            self.pssm_jsonl),
        ]:
            if path and not os.path.isfile(path):
                raise ValueError(f"{name} does not exist: {path!r}")

    @classmethod
    def from_args(cls) -> "GenerationContext":
        p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        p.add_argument("pdb_path",
                       help="Input PDB source: a single .pdb file, a directory of .pdb files, "
                            "a .zip or .tar/.tar.gz/.tgz archive containing .pdb files, "
                            "a plain-text file listing one PDB path per line, "
                            "or (when --h5_file is given) a PDB ID (e.g. 1ABC_A) or a "
                            "plain-text file listing one PDB ID per line")
        p.add_argument("out_folder",      help="Output folder for generated sequences")
        p.add_argument("checkpoint_path", help="Path to model checkpoint (.pt)")

        io = p.add_argument_group("HDF5 input")
        io.add_argument("--h5_file", type=str, default="",
                        help="Path to an HDF5 dataset (produced by pt_to_h5.py). "
                             "When given, pdb_path is interpreted as PDB IDs rather than file paths.")

        gen = p.add_argument_group("Generation")
        gen.add_argument("--num_seq_per_target", type=int, default=1,
                         help="Number of sequences to generate per target")
        gen.add_argument("--sampling_temp", type=str, default="0.1",
                         help="Space-separated sampling temperatures")
        gen.add_argument("--backbone_noise", type=float, default=0.0,
                         help="Std of Gaussian noise added to backbone coordinates")
        gen.add_argument("--batch_size", type=int, default=1,
                         help="Number of structures per batch")
        gen.add_argument("--max_length", type=int, default=DATA_MAX_PROTEIN_LENGTH,
                         help="Maximum protein length; longer structures are skipped")
        gen.add_argument("--no_mixed_precision", action="store_false", dest="mixed_precision",
                         help="Disable automatic mixed precision (AMP) during inference")
        p.set_defaults(mixed_precision=True)

        pos = p.add_argument_group("Chain and position control")
        pos.add_argument("--pdb_path_chains", type=str, default="",
                         help="Space-separated chain IDs to design; empty = all chains")
        pos.add_argument("--chain_id_jsonl", type=str, default="",
                         help="JSONL mapping protein name to (masked, visible) chain lists")
        pos.add_argument("--fixed_positions_jsonl", type=str, default="",
                         help="JSONL mapping protein name to per-chain fixed residue positions (1-indexed)")
        pos.add_argument("--tied_positions_jsonl", type=str, default="",
                         help="JSONL mapping protein name to tied position groups for symmetric design")

        aa = p.add_argument_group("Amino acid constraints")
        aa.add_argument("--omit_AAs", type=str, default="X",
                        help="String of amino acids to globally exclude from sampling")
        aa.add_argument("--bias_AA_jsonl", type=str, default="",
                        help="JSONL with global per-amino-acid bias values")
        aa.add_argument("--bias_by_res_jsonl", type=str, default="",
                        help="JSONL with per-residue, per-amino-acid bias values")
        aa.add_argument("--omit_AA_jsonl", type=str, default="",
                        help="JSONL with per-position amino acids to exclude")

        pssm = p.add_argument_group("PSSM")
        pssm.add_argument("--pssm_jsonl", type=str, default="")
        pssm.add_argument("--pssm_multi", type=float, default=0.0)
        pssm.add_argument("--pssm_threshold", type=float, default=0.0)
        pssm.add_argument("--pssm_log_odds_flag", action="store_true")
        pssm.add_argument("--pssm_bias_flag", action="store_true")

        sc = p.add_argument_group("Scoring modes")
        sc.add_argument("--score_only", action="store_true",
                        help="Compute log-prob scores for native sequences only; no sampling")
        sc.add_argument("--conditional_probs_only", action="store_true",
                        help="Compute per-position conditional probabilities for native sequences")
        sc.add_argument("--conditional_probs_only_backbone_noise", type=float, default=0.0)
        sc.add_argument("--unconditional_probs_only", action="store_true",
                        help="Compute unconditional (sequence-blind) per-position probabilities")

        mdl = p.add_argument_group("Model architecture")
        mdl.add_argument("--hidden_dim", type=int, default=PMPNN_HIDDEN_DIM)
        mdl.add_argument("--num_encoder_layers", type=int, default=PMPNN_NUM_ENCODER_LAYERS)
        mdl.add_argument("--num_decoder_layers", type=int, default=PMPNN_NUM_DECODER_LAYERS)
        mdl.add_argument("--num_neighbors", type=int, default=PMPNN_K_NEIGHBORS)
        mdl.add_argument("--dropout", type=float, default=PMPNN_DROPOUT)
        mdl.add_argument("--mode", type=str, default=HP_MODE,
                         choices=["vanilla", "oa", "gat"])
        mdl.add_argument("--num_radial_orb", type=int, default=OA_NUM_RADIAL)
        mdl.add_argument("--r_max_orb", type=float, default=OA_R_MAX)
        mdl.add_argument("--p_cutoff", type=int, default=OA_P_CUTOFF)
        mdl.add_argument("--l_max", type=int, default=OA_L_MAX)
        mdl.add_argument("--weight_mlp_hidden_dim", type=int, default=OA_WEIGHT_MLP_HIDDEN_DIM)
        mdl.add_argument("--gate_hidden_dim", type=int, default=OA_GATE_HIDDEN_DIM)

        p.add_argument("--seed", type=int, default=HP_SEED)

        args = p.parse_args()
        return cls(**vars(args))
