"""
stats.py — Extract and save meaningful statistics from ProteinMPNN-OA / GAT.

This script must be run before any visualisation component.  It performs
forward passes over a list of proteins and accumulates all per-residue and
per-edge statistics into a single .npz file that every downstream
visualisation module can load without re-running the model.

Both encoder modes are supported:
  - ``oa``  — Orbital Attention (geometry-aware, with degree gating)
  - ``gat`` — Graph Attention (edge-driven, no geometry)

Usage
-----
python stats.py  path/to/data.h5  path/to/pdbs.txt  path/to/checkpoint.pt  \\
    out_stats.npz  [options]

Key outputs saved in the .npz
------------------------------
Scalar / global
  mode   str — encoder mode used ("oa" | "gat")
  L      int — number of encoder layers
  n_sh   int — number of spherical-harmonic channels  (0 for GAT)

Per-residue arrays  [total_residues]
  valid_k           — number of valid neighbours (incl. self-edge)
  neff_{l}          — effective number of attended non-self neighbours (layer l)
  alpha_self_{l}    — self-attention weight (layer l)
  argmax_seq_dist_{l} — seq. dist. of highest-α non-self neighbour
  argmax_dist_3d_{l}  — 3-D dist.  of highest-α non-self neighbour
  mean_score_{l}    — mean budget-weighted score over non-self valid edges

Per-edge arrays  [total_edges]  (all edges, incl. self)
  seq_dist          — |pos_i − pos_j|
  dist_3d           — Cα–Cα Euclidean distance (Å)
  is_self           — bool, True when edge is a self-loop
  res_idx           — global residue id of source residue
  res_type_i        — amino-acid index of source residue
  res_type_j        — amino-acid index of neighbour residue

Per-edge, per-layer arrays  [total_edges]
  attn_{l}          — attention weight α_ij
  rel_attn_{l}      — relative attention (α_ij / per-residue non-self baseline)
  budget_{l}        — non-self budget (1 − α_self_i) broadcast to edge
  scored_attn_{l}   — budget-weighted score: rel_α ^ budget

OA only — per *non-self* edge arrays  [total_non_self_edges]
  spectral_{l}      — spherical-harmonic spectral coefficients [E, S]
  scored_attn_ns_{l}— scored_attn restricted to non-self edges
  theta             — polar angle from backbone e₁ axis  (rad, [0, π])
  phi               — azimuthal angle in the e₂–e₃ plane (rad, [−π, π])
  seq_dist_ns       — seq. dist. for non-self edges
  dist_3d_ns        — 3-D dist.  for non-self edges
  res_type_i_ns     — amino-acid index of source  (non-self edges only)
  res_type_j_ns     — amino-acid index of neighbour (non-self edges only)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from pmpnn.constants import (
    DATA_MAX_PROTEIN_LENGTH,
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
from pmpnn.data import _loader_h5, _to_pdb_dict
from pmpnn.model import ProteinMPNN
from pmpnn.preprocess import featurize


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class StatsContext:
    """
    All configuration needed to run the statistics extraction pass.

    CLI-facing fields are set by :meth:`from_args`; runtime fields
    (``device``, ``model``) are populated in :func:`setup`.
    """

    # ── I/O ───────────────────────────────────────────────────────────────────
    h5_file:          str
    pdb_list:         str
    checkpoint:       str
    out_npz:          str

    # ── data ──────────────────────────────────────────────────────────────────
    max_protein_length: int

    # ── model ─────────────────────────────────────────────────────────────────
    mode:                  str   # "oa" | "gat"
    hidden_dim:            int
    num_encoder_layers:    int
    num_decoder_layers:    int
    num_neighbors:         int
    dropout:               float

    # OA-specific (ignored when mode == "gat")
    num_radial_orb:        int
    r_max_orb:             float
    p_cutoff:              int
    l_max:                 int
    weight_mlp_hidden_dim: int
    gate_hidden_dim:       int


    # ── runtime ───────────────────────────────────────────────────────────────
    torch_device: str = "cpu"
    device: object = field(default=None, repr=False)
    model:  object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # validate paths
        for attr, label in [
            ("h5_file",   "HDF5 structure file"),
            ("pdb_list",  "PDB list file"),
            ("checkpoint","model checkpoint"),
        ]:
            path = getattr(self, attr)
            if not os.path.isfile(path):
                raise ValueError(f"{label} not found: {path!r}")

        out_parent = os.path.dirname(os.path.abspath(self.out_npz)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(
                f"Output directory does not exist: {out_parent!r}")

        # validate mode
        valid_modes = {"oa", "gat"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got {self.mode!r}")

        # validate positive integers / floats
        for name, val in [
            ("max_protein_length",    self.max_protein_length),
            ("hidden_dim",            self.hidden_dim),
            ("num_encoder_layers",    self.num_encoder_layers),
            ("num_decoder_layers",    self.num_decoder_layers),
            ("num_neighbors",         self.num_neighbors),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}")

        if self.is_oa_mode:
            if self.r_max_orb <= 0.0:
                raise ValueError(
                    f"r_max_orb must be > 0 for mode={self.mode!r}, "
                    f"got {self.r_max_orb}")
            for name, val in [
                ("num_radial_orb",        self.num_radial_orb),
                ("p_cutoff",              self.p_cutoff),
                ("l_max",                 self.l_max),
                ("weight_mlp_hidden_dim", self.weight_mlp_hidden_dim),
                ("gate_hidden_dim",       self.gate_hidden_dim),
            ]:
                if val <= 0:
                    raise ValueError(f"{name} must be > 0, got {val}")

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def is_oa_mode(self) -> bool:
        return self.mode == "oa"

    @property
    def is_gat_mode(self) -> bool:
        return self.mode == "gat"

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_args(cls) -> "StatsContext":
        p = argparse.ArgumentParser(
            description="Extract ProteinMPNN-OA / GAT statistics into a .npz file",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # positional
        p.add_argument("h5_file",    help="HDF5 structure file")
        p.add_argument("pdb_list",   help="Text file with PDBID_CHAIN entries (one per line)")
        p.add_argument("checkpoint", help="Model checkpoint (.pt)")
        p.add_argument("out_npz",    help="Output .npz file path")

        # data
        d = p.add_argument_group("Data")
        d.add_argument("--max_protein_length", type=int, default=DATA_MAX_PROTEIN_LENGTH,
                       help="Maximum protein length in residues; longer proteins are discarded")

        # model
        m = p.add_argument_group("Model  (must match checkpoint)")
        m.add_argument("--mode", default="oa",
                       choices=["oa", "gat"],
                       help="Encoder mode: 'oa' (orbital attention with degree gating), "
                            "'gat' (edge-driven graph attention, no geometry)")
        m.add_argument("--hidden_dim",          type=int,   default=PMPNN_HIDDEN_DIM)
        m.add_argument("--num_encoder_layers",  type=int,   default=PMPNN_NUM_ENCODER_LAYERS)
        m.add_argument("--num_decoder_layers",  type=int,   default=PMPNN_NUM_DECODER_LAYERS)
        m.add_argument("--num_neighbors",       type=int,   default=PMPNN_K_NEIGHBORS)
        m.add_argument("--dropout",             type=float, default=PMPNN_DROPOUT)

        # OA hypers
        oa = p.add_argument_group("Orbital Attention hypers  (ignored for --mode gat)")
        oa.add_argument("--num_radial_orb",        type=int,   default=OA_NUM_RADIAL)
        oa.add_argument("--r_max_orb",             type=float, default=OA_R_MAX)
        oa.add_argument("--p_cutoff",              type=int,   default=OA_P_CUTOFF)
        oa.add_argument("--l_max",                 type=int,   default=OA_L_MAX)
        oa.add_argument("--weight_mlp_hidden_dim", type=int,   default=OA_WEIGHT_MLP_HIDDEN_DIM)
        oa.add_argument("--gate_hidden_dim",       type=int,   default=OA_GATE_HIDDEN_DIM)

        # runtime
        p.add_argument("--device", default="cpu",
                       dest="torch_device",
                       help="Torch device (cpu / cuda / cuda:N)")

        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

_MODE_KWARGS = {
    "oa":  dict(use_orbital_attn=True,  use_gat=False),
    "gat": dict(use_orbital_attn=False, use_gat=True),
}


def setup(ctx: StatsContext) -> None:
    """
    Instantiate the device and the model, store them on *ctx*.
    Must be called before :func:`accumulate`.
    """
    ctx.device = torch.device(ctx.torch_device)

    oa_kwargs = (
        dict(
            num_radial_orb        = ctx.num_radial_orb,
            r_max_orb             = ctx.r_max_orb,
            p_cutoff              = ctx.p_cutoff,
            l_max                 = ctx.l_max,
            weight_mlp_hidden_dim = ctx.weight_mlp_hidden_dim,
            gate_hidden_dim       = ctx.gate_hidden_dim,
        )
        if ctx.is_oa_mode else {}
    )

    ctx.model = ProteinMPNN.from_pretrained(
        ctx.checkpoint,
        map_location       = ctx.device,
        node_features      = ctx.hidden_dim,
        edge_features      = ctx.hidden_dim,
        hidden_dim         = ctx.hidden_dim,
        num_encoder_layers = ctx.num_encoder_layers,
        num_decoder_layers = ctx.num_decoder_layers,
        k_neighbors        = ctx.num_neighbors,
        dropout            = ctx.dropout,
        **oa_kwargs,
        **_MODE_KWARGS[ctx.mode],
    )
    ctx.model.to(ctx.device).eval()
    print(f"[setup] mode={ctx.mode!r}  device={ctx.device}  "
          f"encoder_layers={len(ctx.model.encoder_layers)}")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_forward(model, pdb_dict, device, is_oa_mode: bool):
    """
    Run a single protein through the model.

    Returns
    -------
    X             : [1, N, 4, 3]
    S             : [1, N]
    mask          : [1, N]
    E_idx         : [1, N, K]
    mask_attend   : [1, N, K]
    attn_weights  : list of L tensors [1, N, K]
    spectral_layers  : list of L tensors [1, N, K, S]  or empty list for GAT
    Y_out         : [1, N, K, S] raw spherical harmonics          or None for GAT
    """
    X, S, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize(
        [pdb_dict], device)
    randn = torch.zeros_like(chain_M)
    out   = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
    return (
        X, S, mask,
        out.E_idx, out.mask_attend, out.attn_weights,
        out.spectral_layers if is_oa_mode else [],
        out.Y if is_oa_mode else None,
    )


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def accumulate(ctx: StatsContext, pdb_chains: List[str]) -> dict:
    """
    Iterate over *pdb_chains* and collect per-residue and per-edge statistics.

    The returned dict maps string keys to numpy arrays.  Layer-indexed arrays
    carry a ``_{l}`` suffix (0-indexed).  Two separate edge pools are
    maintained:

    * **all-edge pool** — every valid edge (self included); the ``attn_*``,
      ``rel_attn_*``, ``budget_*``, ``scored_attn_*`` arrays live here.
    * **non-self-edge pool** — only ``non_self_valid`` edges; spectral, theta,
      phi, and ``scored_attn_ns_*`` arrays live here (OA only).

    Parameters
    ----------
    ctx : StatsContext
        Fully set-up context (must have called :func:`setup` first).
    pdb_chains : list of str
        List of ``PDBID_CHAIN`` strings to process.

    Returns
    -------
    dict
    """
    model     = ctx.model
    device    = ctx.device
    L         = len(model.encoder_layers)
    is_oa     = ctx.is_oa_mode

    # number of SH channels — 0 for GAT so downstream code can check it
    n_sh = (
        model.encoder_layers[0].orbital_attn.n_sh
        if is_oa else 0
    )

    # ── per-residue accumulators ───────────────────────────────────────────────
    all_valid_k          = []
    all_neff             = [[] for _ in range(L)]
    all_alpha_self       = [[] for _ in range(L)]
    all_argmax_seq_dist  = [[] for _ in range(L)]
    all_argmax_dist_3d   = [[] for _ in range(L)]
    all_mean_score       = [[] for _ in range(L)]

    # ── per-edge (all valid edges, incl. self) ─────────────────────────────────
    all_attn             = [[] for _ in range(L)]
    all_rel_attn         = [[] for _ in range(L)]
    all_budget           = [[] for _ in range(L)]
    all_scored_attn      = [[] for _ in range(L)]
    all_seq_dist         = []
    all_dist_3d          = []
    all_is_self          = []
    all_res_idx          = []
    all_res_type_i       = []
    all_res_type_j       = []

    # ── per-edge (non-self only, OA mode) ─────────────────────────────────────
    all_spectral         = [[] for _ in range(L)]
    all_scored_attn_ns   = [[] for _ in range(L)]
    all_theta            = []
    all_phi              = []
    all_seq_dist_ns      = []
    all_dist_3d_ns       = []
    all_res_type_i_ns    = []
    all_res_type_j_ns    = []

    global_res_id = 0
    n_proteins    = 0

    for pdb_chain in pdb_chains:
        print(f"  {pdb_chain} …", end=" ", flush=True)

        raw      = _loader_h5([pdb_chain], ctx.h5_file)
        pdb_dict = _to_pdb_dict(raw, ctx.max_protein_length)
        if pdb_dict is None:
            print("skipped (filtered by _to_pdb_dict)")
            continue

        try:
            (X, S, mask,
             E_idx, mask_attend,
             attn_weights, spectral_layers, Y_out) = _run_forward(
                model, pdb_dict, device, is_oa)
        except Exception as exc:
            print(f"skipped ({exc})")
            continue

        valid_n   = mask[0].bool()               # [N]
        n_indices = torch.where(valid_n)[0]      # [N_v]
        N_v       = int(valid_n.sum())

        E_idx_v       = E_idx[0][valid_n]        # [N_v, K]
        mask_attend_v = mask_attend[0][valid_n]  # [N_v, K]
        valid_edges   = mask_attend_v.bool()     # [N_v, K]

        is_self_vk     = (E_idx_v == n_indices.unsqueeze(-1))   # [N_v, K]
        non_self_valid = valid_edges & ~is_self_vk              # [N_v, K]

        # Geometric quantities
        Ca         = X[0, :, 1, :]
        Ca_j       = Ca[E_idx_v]                                           # [N_v, K, 3]
        dist_3d_vk = (Ca_j - Ca[n_indices].unsqueeze(1)).norm(dim=-1)     # [N_v, K]
        seq_dist_vk = (
            E_idx_v.long() - n_indices.unsqueeze(-1).long()
        ).abs().float()                                                     # [N_v, K]

        # ── OA-specific: angles from l=1 SH channels ──────────────────────────
        if is_oa:
            Y_vk   = Y_out[0][valid_n]                         # [N_v, K, S]
            inv_s3 = 1.0 / (3.0 ** 0.5)
            # e3nn component-normalised: Y[1]=√3·e₂, Y[2]=√3·e₃, Y[3]=√3·e₁
            d_e1      = (Y_vk[..., 3] * inv_s3).clamp(-1.0, 1.0)
            d_e2      =  Y_vk[..., 1] * inv_s3
            d_e3      =  Y_vk[..., 2] * inv_s3
            theta_vk  = torch.acos(d_e1)           # polar angle [0, π]
            phi_vk    = torch.atan2(d_e2, d_e3)    # azimuthal  [−π, π]

        # K_non_self is topology-only: it does not change across layers
        K_non_self = non_self_valid.float().sum(-1).clamp(min=1.0)  # [N_v]

        # ── per-residue neighbourhood size ────────────────────────────────────
        all_valid_k.append(mask_attend_v.sum(-1).cpu().numpy())

        # ── per-layer per-residue statistics ──────────────────────────────────
        for l in range(L):
            alpha_vk  = attn_weights[l][0][valid_n]          # [N_v, K]

            # --- self-attention weight ---
            alpha_self = (alpha_vk * is_self_vk.float()).sum(-1)  # [N_v]
            all_alpha_self[l].append(alpha_self.cpu().numpy())

            # --- non-self budget and relative attention ---
            budget     = (1.0 - alpha_self).clamp(min=1e-8)       # [N_v]
            baseline   = budget / K_non_self                        # [N_v]
            rel_alpha  = alpha_vk / baseline.unsqueeze(-1)         # [N_v, K]
            budget_vk  = budget.unsqueeze(-1).expand_as(alpha_vk) # [N_v, K]
            scored_vk  = rel_alpha ** budget_vk                    # [N_v, K]

            # --- Neff (entropy of non-self α, renormalised) ---
            alpha_ns   = alpha_vk * non_self_valid.float()
            alpha_ns   = alpha_ns / alpha_ns.sum(-1, keepdim=True).clamp(min=1e-8)
            entropy    = -(alpha_ns * (alpha_ns + 1e-10).log()
                           * non_self_valid.float()).sum(-1)
            all_neff[l].append(entropy.exp().cpu().numpy())

            # --- argmax non-self neighbour ---
            scores_ns  = alpha_vk.masked_fill(~non_self_valid, -1.0)
            am         = scores_ns.argmax(dim=-1)
            all_argmax_seq_dist[l].append(
                seq_dist_vk[torch.arange(N_v), am].cpu().numpy())
            all_argmax_dist_3d[l].append(
                dist_3d_vk[torch.arange(N_v), am].cpu().numpy())

            # --- per-residue mean score ---
            scored_ns   = scored_vk * non_self_valid.float()
            mean_score  = scored_ns.sum(-1) / K_non_self             # [N_v]
            all_mean_score[l].append(mean_score.cpu().numpy())

            # --- per-edge (all valid edges) ---
            all_attn[l].append(alpha_vk[valid_edges].cpu().numpy())
            all_rel_attn[l].append(rel_alpha[valid_edges].cpu().numpy())
            all_budget[l].append(budget_vk[valid_edges].cpu().numpy())
            all_scored_attn[l].append(scored_vk[valid_edges].cpu().numpy())

            # --- OA: non-self spectral and scored_attn ---
            if is_oa:
                sp_vk = spectral_layers[l][0][valid_n]          # [N_v, K, S]
                all_spectral[l].append(sp_vk[non_self_valid].cpu().numpy())
                all_scored_attn_ns[l].append(scored_vk[non_self_valid].cpu().numpy())

        # ── per-edge (all valid edges, layer-independent) ─────────────────────
        all_seq_dist.append(seq_dist_vk[valid_edges].cpu().numpy())
        all_dist_3d.append(dist_3d_vk[valid_edges].cpu().numpy())
        all_is_self.append(is_self_vk[valid_edges].cpu().numpy())

        type_i_vk = S[0][n_indices].unsqueeze(-1).expand_as(E_idx_v)  # [N_v, K]
        type_j_vk = S[0][E_idx_v]                                      # [N_v, K]
        all_res_type_i.append(type_i_vk[valid_edges].cpu().numpy())
        all_res_type_j.append(type_j_vk[valid_edges].cpu().numpy())

        res_id_vk = (
            torch.arange(global_res_id, global_res_id + N_v, device=device)
            .unsqueeze(-1).expand_as(E_idx_v)
        )
        all_res_idx.append(res_id_vk[valid_edges].cpu().numpy())

        # ── OA: non-self edge geometry ─────────────────────────────────────────
        if is_oa:
            all_theta.append(theta_vk[non_self_valid].cpu().numpy())
            all_phi.append(phi_vk[non_self_valid].cpu().numpy())
            all_seq_dist_ns.append(seq_dist_vk[non_self_valid].cpu().numpy())
            all_dist_3d_ns.append(dist_3d_vk[non_self_valid].cpu().numpy())
            all_res_type_i_ns.append(type_i_vk[non_self_valid].cpu().numpy())
            all_res_type_j_ns.append(type_j_vk[non_self_valid].cpu().numpy())

        global_res_id += N_v
        n_proteins    += 1

        n_edges     = int(valid_edges.sum())
        n_non_self  = int(non_self_valid.sum())
        print(f"{N_v} residues  {n_edges} edges  ({n_non_self} non-self)")

    # ── assemble output dict ───────────────────────────────────────────────────
    if n_proteins == 0:
        raise RuntimeError("No proteins were successfully processed.")

    out: dict = {
        # metadata
        "mode":            np.array(ctx.mode),
        "L":               np.array(L, dtype=np.int32),
        "n_sh":            np.array(n_sh, dtype=np.int32),
        "n_proteins":      np.array(n_proteins, dtype=np.int32),

        # per-residue
        "valid_k":         np.concatenate(all_valid_k).astype(np.float32),

        # per-edge (all valid, incl. self)
        "seq_dist":        np.concatenate(all_seq_dist).astype(np.float32),
        "dist_3d":         np.concatenate(all_dist_3d).astype(np.float32),
        "is_self":         np.concatenate(all_is_self).astype(bool),
        "res_idx":         np.concatenate(all_res_idx).astype(np.int64),
        "res_type_i":      np.concatenate(all_res_type_i).astype(np.int8),
        "res_type_j":      np.concatenate(all_res_type_j).astype(np.int8),
    }

    for l in range(L):
        # per-residue, per-layer
        out[f"neff_{l}"]            = np.concatenate(all_neff[l]).astype(np.float32)
        out[f"alpha_self_{l}"]      = np.concatenate(all_alpha_self[l]).astype(np.float32)
        out[f"argmax_seq_dist_{l}"] = np.concatenate(all_argmax_seq_dist[l]).astype(np.float32)
        out[f"argmax_dist_3d_{l}"]  = np.concatenate(all_argmax_dist_3d[l]).astype(np.float32)
        out[f"mean_score_{l}"]      = np.concatenate(all_mean_score[l]).astype(np.float32)

        # per-edge (all valid), per-layer
        out[f"attn_{l}"]            = np.concatenate(all_attn[l]).astype(np.float32)
        out[f"rel_attn_{l}"]        = np.concatenate(all_rel_attn[l]).astype(np.float32)
        out[f"budget_{l}"]          = np.concatenate(all_budget[l]).astype(np.float32)
        out[f"scored_attn_{l}"]     = np.concatenate(all_scored_attn[l]).astype(np.float32)

        if is_oa:
            # per non-self edge, per-layer
            out[f"spectral_{l}"]        = np.concatenate(all_spectral[l], axis=0).astype(np.float32)
            out[f"scored_attn_ns_{l}"]  = np.concatenate(all_scored_attn_ns[l]).astype(np.float32)

    if is_oa:
        # per non-self edge, layer-independent
        out["theta"]         = np.concatenate(all_theta).astype(np.float32)
        out["phi"]           = np.concatenate(all_phi).astype(np.float32)
        out["seq_dist_ns"]   = np.concatenate(all_seq_dist_ns).astype(np.float32)
        out["dist_3d_ns"]    = np.concatenate(all_dist_3d_ns).astype(np.float32)
        out["res_type_i_ns"] = np.concatenate(all_res_type_i_ns).astype(np.int8)
        out["res_type_j_ns"] = np.concatenate(all_res_type_j_ns).astype(np.int8)

    return out


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_summary(stats: dict) -> None:
    """Print a brief summary of the accumulated statistics."""
    L        = int(stats["L"])
    mode     = str(stats["mode"])
    n_res    = len(stats["valid_k"])
    n_edges  = len(stats["is_self"])
    n_self   = int(stats["is_self"].sum())
    n_non    = n_edges - n_self

    print(f"\n{'─' * 60}")
    print(f"  mode            : {mode}")
    print(f"  encoder layers  : {L}")
    print(f"  proteins done   : {int(stats['n_proteins'])}")
    print(f"  total residues  : {n_res:,}")
    print(f"  total edges     : {n_edges:,}  "
          f"(self={n_self:,}  non-self={n_non:,})")

    for l in range(L):
        mean_neff   = float(stats[f"neff_{l}"].mean())
        mean_aself  = float(stats[f"alpha_self_{l}"].mean())
        mean_score  = float(stats[f"mean_score_{l}"].mean())
        print(f"  layer {l + 1}  "
              f"mean_neff={mean_neff:.2f}  "
              f"mean_α_self={mean_aself:.3f}  "
              f"mean_score={mean_score:.3f}")

    if "theta" in stats:
        n_ns = len(stats["theta"])
        print(f"  non-self edges  : {n_ns:,}  (spectral arrays)")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = StatsContext.from_args()

    print("[stats] Loading model …")
    setup(ctx)

    print("[stats] Reading PDB list …")
    with open(ctx.pdb_list) as fh:
        pdb_chains = [ln.strip() for ln in fh if ln.strip()]
    print(f"  {len(pdb_chains)} entries")

    print("[stats] Running forward passes …")
    stats = accumulate(ctx, pdb_chains)

    _print_summary(stats)

    print(f"[stats] Saving → {ctx.out_npz}")
    np.savez_compressed(ctx.out_npz, **stats)
    print(f"[stats] Done.  Keys saved: {sorted(stats.keys())}")


if __name__ == "__main__":
    main()