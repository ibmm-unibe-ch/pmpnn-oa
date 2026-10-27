"""
ss_bond_analysis.py — Budget-weighted attention score for Cys-Cys edges,
annotated against real SSBOND (disulfide-bond) records.

Computes the exact same per-edge budget-weighted score as stats.py —
``scored_attn = rel_alpha ** budget``, with ``budget = 1 - alpha_self`` and
``rel_alpha = alpha_ij / (budget_i / K_non_self_i)`` (see stats.py's
``accumulate``) — but:

  - restricts every per-edge array to non-self edges where BOTH the source
    and the neighbour residue are Cys, and
  - adds a boolean ``is_ssbond`` label per edge: True iff that exact
    (source, neighbour) residue pair is recorded as an SSBOND in the
    structure's own PDB header.

Unlike stats.py (which reads structures from the training/eval HDF5 — a
format that drops all header/connectivity records), this script needs the
SSBOND annotation, so it parses raw PDB files directly — by default
data/test_pdb_raw, the full, as-downloaded, multi-chain PDB entries
(scripts/download_test_monomers.py's --cache_dir) — and keeps each
residue's original author-assigned residue id for exact matching against
the SSBOND seqNum fields.

Every structure is read whole (all its chains together, not split one
chain at a time), so every SSBOND record in the file is captured — intra-
AND inter-chain — giving full disulfide-bond coverage. This matters:
data/test_pdb_raw's ~1500 structures carry ~4000 SSBOND records in total,
vs. ~800 survivng in the single-chain-per-file splits (data/test_monomers_pdb
etc., written by scripts/download_test_monomers.py) — that script keeps
only entries with an *intra*-chain disulfide on one designated chain, so
it drops every inter-chain bond outright (the partner chain isn't even
in the file) along with every chain that happens to have zero disulfides
of its own (losing non-bonded Cys-Cys statistics from those chains too).
Those split directories remain usable as --pdb_dirs (e.g. to reproduce a
single-chain-only run), just with that reduced, biased coverage.

Two subcommands:

  compute   Run the forward passes and write the per-edge .npz described
            below (GPU work — the expensive step).
  analyze   Load a .npz written by ``compute`` and report the statistical
            analysis of budget-weighted score vs. SSBOND, plus the
            matched-control figure (CPU-only; ported from the exploratory
            analysis in test.ipynb — see "Statistical analysis" below).

Usage
-----
python ss_bond_analysis.py  compute  data/test_pdb_raw  checkpoint.pt  out.npz  --mode oa
python ss_bond_analysis.py  analyze  out.npz  --layer 0  --fig_out ssbond_matched.svg \\
    --tables_out ssbond_tables.csv

Output .npz keys  (compute)
----------------------------
Scalar
  L            int   — number of encoder layers
  n_proteins   int   — structures successfully processed
  n_skipped    int   — structures skipped (parse/forward failure, too long,
                       or no resolved residues)

Per Cys-Cys edge  [total_edges]  (non-self only; both endpoints are Cys)
  name         structure name (.pdb filename stem — "PDBID" for
                       data/test_pdb_raw, "PDBID_CHAIN" for the
                       single-chain split dirs) the edge came from
  protein_idx  int64 — 0-indexed structure id, in processing order
  res_idx_i    int64 — global residue id of the source residue (monotonic
                       across the whole run; does NOT reset per structure)
  res_idx_j    int64 — global residue id of the neighbour residue, in the
                       same id space as res_idx_i
  seq_dist     |pos_i - pos_j| within the structure
  dist_3d      Cα-Cα distance (Å)
  dist_cb      virtual Cβ-Cβ distance (Å) — see pmpnn.model.virtual_cbeta
  is_ssbond    bool  — True iff this residue pair is an annotated SSBOND

Per Cys-Cys edge, per layer  [total_edges]
  attn_{l}         attention weight α_ij
  rel_attn_{l}     relative attention (α_ij / per-residue non-self baseline)
  budget_{l}       non-self budget (1 - α_self) broadcast to the edge
  scored_attn_{l}  budget-weighted score: rel_α ^ budget

Per non-Cys/non-Cys control edge  [total_ctrl_edges]  (same fields as
above, "ctrl_" prefixed: ctrl_name, ctrl_protein_idx, ctrl_res_idx_i,
ctrl_res_idx_j, ctrl_seq_dist, ctrl_dist_3d, ctrl_dist_cb, and per layer
ctrl_attn_{l}/ctrl_rel_attn_{l}/ctrl_budget_{l}/ctrl_scored_attn_{l}; no
ctrl_is_ssbond — these edges are never bonded by construction). A random,
--ctrl_edges_per_protein-capped (default 500) subsample of every
structure's non-self edges where NEITHER endpoint is Cys, seeded by
--ctrl_seed. This is ``analyze``'s candidate pool for a distance-matched
comparison against the real (non-bonded) Cys-Cys population — see
"Statistical analysis" below.

Statistical analysis  (analyze)
--------------------------------
Edges are first restricted to |seq_dist| >= --sep_min (default 6, matching
stats.py's medium/long-range convention) then collapsed from directed
(i->j and j->i) rows into one row per undirected residue pair, averaging
dist_3d / dist_cb / scored_attn_{layer} across the (up to 2) directed
observations seen for that pair.

Three analyses are then run over those undirected pairs, each stratifying
by a distance into adaptive quantile bins (chosen so the minority class —
usually SSBOND — has >= --min_minority edges per bin, within the
--support_q quantile range shared by both classes), then reporting, per
bin and pooled across bins (inverse-variance-style weighting):
  - AUC: the probability a random bonded pair's score exceeds a random
    non-bonded pair's, at matched distance (a Mann-Whitney U statistic).
  - Effect size: mean difference in log2(score) (or, for B, in -dist_cb)
    between bonded and non-bonded pairs, at matched distance.
A 95% CI on both pooled quantities comes from a protein-level cluster
bootstrap (--n_boot resamples of whole proteins, not edges, since edges
within one protein aren't independent draws).

  A  main              stratify by dist_3d,  score = scored_attn_{layer}
                        — does the model's budget-weighted attention track
                        SSBOND once Cα-Cα distance is matched?
  B  Cb reference       stratify by dist_3d,  score = -dist_cb
                        — how strongly does SSBOND track Cβ-Cβ distance
                        alone at the same Cα-Cα strata, i.e. the geometric
                        effect size the model's score is implicitly being
                        compared against.
  C  Cb-matched control stratify by dist_cb,  score = scored_attn_{layer}
                        — repeats A but matched on the finer Cβ-Cβ
                        distance (the more direct disulfide-geometry
                        proxy), as a stricter control.

More analyses (D-I), run only when the .npz has the ctrl_* fields (see
"Output .npz keys" above — older .npz files lack them and these are
skipped with a warning), repeat A/C but swap out the comparison class:
instead of the real non-bonded Cys-Cys population, they use a distance-
matched sample of non-Cys/non-Cys control edges — same total size as
A/C's non-bonded group, and the same per-bin distribution over dist_3d
(D), dist_cb (E) or both jointly (F, via --joint_match; C2 is F's
Cys-Cys-vs-Cys-Cys reference at the same joint strata) — drawn
(--match_seed) from the ctrl_* candidate pool via _sample_matched_control.
This asks a stricter question than A/C: is scored_attn elevated at
SSBOND pairs specifically, or just at ANY close-range contact regardless
of residue identity (which non-bonded Cys-Cys pairs, being Cys
themselves, might already share with SSBOND pairs for non-bond-specific
chemical reasons)?

  D  non-Cys control (Ca-matched)        stratify by dist_3d,        score = scored_attn_{layer}
                        — SSBOND vs. a dist_3d-matched non-Cys/non-Cys sample.
  E  non-Cys control (Cb-matched)        stratify by dist_cb,        score = scored_attn_{layer}
                        — SSBOND vs. a dist_cb-matched non-Cys/non-Cys sample.
  F  non-Cys control (joint-matched)     stratify by --joint_match,  score = scored_attn_{layer}
                        — SSBOND vs. a jointly-matched non-Cys/non-Cys sample.

G/H/I repeat D/E/F again, but with the comparison's roles reversed: the
NON-annotated (non-bonded) Cys-Cys pairs now stand in as the "positive"
class, compared against the exact same matched non-Cys/non-Cys samples
D/E/F already drew (those were matched to the non-bonded Cys-Cys group's
own per-bin counts, so they're already the right control for this). This
asks: is scored_attn elevated at ANY Cys-Cys pair relative to a distance-
matched non-Cys/non-Cys pair, even when that Cys-Cys pair isn't an
annotated disulfide — i.e. is the model's score partly tracking Cys-Cys
chemistry/geometry on its own, separately from the SSBOND-specific signal
A-F measure?

  G  non-Cys control (Ca-matched, non-annotated Cys-Cys)    stratify by dist_3d
  H  non-Cys control (Cb-matched, non-annotated Cys-Cys)    stratify by dist_cb
  I  non-Cys control (joint-matched, non-annotated Cys-Cys) stratify by --joint_match

A two-panel figure (--fig_out, default ssbond_matched.svg) then plots
log2(scored_attn_{layer}) against dist_3d (A's pairs) and dist_cb (C's
pairs), split into SSBOND / not-annotated, each with a per-bin median step
line. (D-I are not plotted — tables/CSV only.)

All per-bin tables are also written as a single combined CSV
(--tables_out, default ssbond_tables.csv): columns analysis/label/by/score
identify which of A-E a row belongs to, bin_lo/bin_hi/n_bond/n_non/
prev/auc/eff are that analysis's per-bin table, and a trailing row per
analysis (bin_lo=bin_hi="POOLED") carries the pooled auc/eff plus their
95% CI (auc_lo/auc_hi/eff_lo/eff_hi — NaN on the per-bin rows, where no
CI is computed).
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import warnings
from dataclasses import dataclass
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel
from scipy.stats import rankdata

from ._bootstrap import BootData, build_layout, run_bootstrap
from pmpnn.constants import (
    AA_3TO1,
    AA_ALPHABET,
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
from pmpnn.model import ProteinMPNN, virtual_cbeta
from .stats import _MODE_KWARGS, _run_forward

_CYS_IDX = AA_ALPHABET.index("C")


# ---------------------------------------------------------------------------
# PDB + SSBOND parsing
# ---------------------------------------------------------------------------

def _parse_chain_pdb(pdb_file: str):
    """
    Parse a PDB file into (pdb_dict, res_ids_by_chain, ssbond_pairs):

      pdb_dict          — ready for pmpnn.preprocess.featurize(), same shape
                           as src.gen.__main__.parse_PDB's output.
      res_ids_by_chain   — {chain: [author residue id, ...]}, same order and
                           length as pdb_dict[f"seq_chain_{chain}"], for
                           mapping a model position back to its PDB identity.
      ssbond_pairs       — set of frozenset({(chain1, res_id1), (chain2, res_id2)})
                           built from the file's own SSBOND header records.

    Every SSBOND record in the file is kept, however many chains it spans
    — including inter-chain disulfides — not just ones fully contained in
    a single chain. A file with multiple MODEL records (NMR ensembles)
    contributes only its first model's coordinates; SSBOND records live
    in the file's single shared header, so they're unaffected.

    Returns None if the file has no resolved (CA-bearing) residues.
    """
    name = os.path.splitext(os.path.basename(pdb_file))[0]

    chain_ids: list[str] = []
    res_order: dict[str, list] = {}
    res_atoms: dict[str, dict] = {}
    res_seq:   dict[str, dict] = {}
    ssbond_pairs: set = set()

    with open(pdb_file) as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "ENDMDL":
                break
            if rec == "SSBOND":
                c1, r1 = line[15].strip(), line[17:22].strip()
                c2, r2 = line[29].strip(), line[31:36].strip()
                ssbond_pairs.add(frozenset({(c1, r1), (c2, r2)}))
                continue
            if rec not in ("ATOM", "HETATM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C", "O"):
                continue
            res_name = line[17:20].strip()
            chain_id = line[21]
            res_id   = line[22:27].strip()
            x, y, z  = float(line[30:38]), float(line[38:46]), float(line[46:54])
            aa = AA_3TO1.get(res_name, "X")

            if chain_id not in chain_ids:
                chain_ids.append(chain_id)
                res_order[chain_id] = []
                res_atoms[chain_id] = {}
                res_seq[chain_id]   = {}

            if res_id not in res_atoms[chain_id]:
                res_atoms[chain_id][res_id] = {}
                res_order[chain_id].append(res_id)
                res_seq[chain_id][res_id] = aa

            res_atoms[chain_id][res_id][atom_name] = [x, y, z]

    my_dict: dict = {"name": name}
    res_ids_by_chain: dict[str, list] = {}

    for chain in chain_ids:
        N_list, CA_list, C_list, O_list, seq, res_ids = [], [], [], [], "", []
        for res_id in res_order[chain]:
            atoms = res_atoms[chain][res_id]
            if "CA" not in atoms:
                continue
            N_list.append(atoms.get("N",  [0., 0., 0.]))
            CA_list.append(atoms.get("CA", [0., 0., 0.]))
            C_list.append(atoms.get("C",  [0., 0., 0.]))
            O_list.append(atoms.get("O",  [0., 0., 0.]))
            seq += res_seq[chain][res_id]
            res_ids.append(res_id)

        if not seq:
            continue

        my_dict[f"seq_chain_{chain}"] = seq
        my_dict[f"coords_chain_{chain}"] = {
            f"N_chain_{chain}":  np.array(N_list,  dtype=np.float32),
            f"CA_chain_{chain}": np.array(CA_list, dtype=np.float32),
            f"C_chain_{chain}":  np.array(C_list,  dtype=np.float32),
            f"O_chain_{chain}":  np.array(O_list,  dtype=np.float32),
        }
        res_ids_by_chain[chain] = res_ids

    resolved_chains = list(res_ids_by_chain)
    if not resolved_chains:
        return None

    my_dict["num_of_chains"] = len(resolved_chains)
    my_dict["masked_list"]   = resolved_chains
    my_dict["visible_list"]  = []
    my_dict["seq"] = "".join(my_dict[f"seq_chain_{c}"] for c in resolved_chains)

    return my_dict, res_ids_by_chain, ssbond_pairs


def _collect_pdb_files(pdb_dirs: List[str]) -> List[str]:
    files = []
    for d in pdb_dirs:
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, "*.pdb"))))
        elif d.lower().endswith(".pdb"):
            files.append(d)
        else:
            raise ValueError(f"Not a directory or .pdb file: {d!r}")
    return files


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def _run_forward_stable_chain_order(model, pdb_dict, device, is_oa: bool):
    """Call stats._run_forward with pmpnn.preprocess.featurize's internal
    chain-order randomization (``random.shuffle(all_chains)``, a training-
    time augmentation) disabled, so the model's flattened residue order
    exactly matches pdb_dict["masked_list"] + pdb_dict["visible_list"] —
    i.e. the same chain order _parse_chain_pdb built res_ids_by_chain in.
    Needed for correctness whenever pdb_dict has more than one chain
    (e.g. data/test_pdb_raw's multi-chain entries): this script tracks
    each model position's (chain, author res_id) identity purely by that
    fixed order (see the chain_of_pos/res_id_of_pos construction below),
    and a shuffled order would silently scramble it (while leaving
    geometry — distances, which depend only on actual 3D coordinates —
    unaffected, so the bug would not be obviously visible from distances
    alone). Single-chain pdb_dicts are unaffected either way (nothing to
    shuffle)."""
    real_shuffle = random.shuffle
    random.shuffle = lambda seq: None
    try:
        return _run_forward(model, pdb_dict, device, is_oa)
    finally:
        random.shuffle = real_shuffle


def accumulate(
    model, device, is_oa: bool, pdb_files: List[str], max_length: int,
    ctrl_edges_per_protein: int = 500, ctrl_seed: int = 0,
) -> dict:
    """*ctrl_edges_per_protein* non-Cys/non-Cys non-self edges are randomly
    subsampled (without replacement, seeded by *ctrl_seed*) per structure
    and accumulated in parallel under "ctrl_*" keys — a candidate pool for
    `analyze`'s distance-matched non-Cys control comparison (see module
    docstring). Cheap: it only *indexes* the attention/geometry tensors
    already computed for the Cys-Cys edges, with a different boolean mask."""
    L = len(model.encoder_layers)

    all_attn        = [[] for _ in range(L)]
    all_rel_attn    = [[] for _ in range(L)]
    all_budget      = [[] for _ in range(L)]
    all_scored_attn = [[] for _ in range(L)]
    all_seq_dist    = []
    all_dist_3d     = []
    all_dist_cb     = []
    all_is_ssbond   = []
    all_res_idx_i   = []
    all_res_idx_j   = []
    all_protein_idx = []
    all_name        = []

    ctrl_attn        = [[] for _ in range(L)]
    ctrl_rel_attn    = [[] for _ in range(L)]
    ctrl_budget      = [[] for _ in range(L)]
    ctrl_scored_attn = [[] for _ in range(L)]
    ctrl_seq_dist    = []
    ctrl_dist_3d     = []
    ctrl_dist_cb     = []
    ctrl_res_idx_i   = []
    ctrl_res_idx_j   = []
    ctrl_protein_idx = []
    ctrl_name        = []

    ctrl_rng = torch.Generator().manual_seed(ctrl_seed)

    n_proteins    = 0
    n_skipped     = 0
    global_res_id = 0

    for pdb_file in pdb_files:
        name = os.path.splitext(os.path.basename(pdb_file))[0]
        print(f"  {name} …", end=" ", flush=True)

        parsed = _parse_chain_pdb(pdb_file)
        if parsed is None:
            print("skipped (no resolved residues)")
            n_skipped += 1
            continue
        pdb_dict, res_ids_by_chain, ssbond_pairs = parsed

        if len(pdb_dict["seq"]) > max_length:
            print(f"skipped (length {len(pdb_dict['seq'])} > max_length)")
            n_skipped += 1
            continue

        # (chain, author res_id) for every position in pdb_dict["seq"], in order.
        chain_of_pos: list[str] = []
        res_id_of_pos: list[str] = []
        for c in pdb_dict["masked_list"] + pdb_dict["visible_list"]:
            chain_of_pos.extend([c] * len(res_ids_by_chain[c]))
            res_id_of_pos.extend(res_ids_by_chain[c])

        try:
            X, S, mask, E_idx, mask_attend, attn_weights, _ = _run_forward_stable_chain_order(
                model, pdb_dict, device, is_oa)
        except Exception as exc:
            print(f"skipped ({exc})")
            n_skipped += 1
            continue

        valid_n   = mask[0].bool()               # [N]
        n_indices = torch.where(valid_n)[0]      # [N_v]
        N_v       = int(valid_n.sum())

        E_idx_v       = E_idx[0][valid_n]        # [N_v, K]
        mask_attend_v = mask_attend[0][valid_n]  # [N_v, K]
        valid_edges   = mask_attend_v.bool()     # [N_v, K]
        is_self_vk     = (E_idx_v == n_indices.unsqueeze(-1))
        non_self_valid = valid_edges & ~is_self_vk

        type_i_vk = S[0][n_indices].unsqueeze(-1).expand_as(E_idx_v)  # [N_v, K]
        type_j_vk = S[0][E_idx_v]                                      # [N_v, K]
        cys_edges   = non_self_valid & (type_i_vk == _CYS_IDX) & (type_j_vk == _CYS_IDX)
        noncys_edges = non_self_valid & (type_i_vk != _CYS_IDX) & (type_j_vk != _CYS_IDX)

        # global residue id of the source (res_idx_i) and neighbour (res_idx_j),
        # in the same, run-wide monotonic id space — mirrors stats.py's convention.
        rank_of_padded_idx = torch.full((X.shape[1],), -1, dtype=torch.long, device=device)
        rank_of_padded_idx[n_indices] = torch.arange(N_v, device=device)
        res_idx_i_vk = (global_res_id + torch.arange(N_v, device=device)).unsqueeze(-1).expand_as(E_idx_v)
        res_idx_j_vk = global_res_id + rank_of_padded_idx[E_idx_v]

        n_proteins += 1
        has_cys_edges = bool(cys_edges.any())

        Ca          = X[0, :, 1, :]
        Cb          = virtual_cbeta(X)[0]
        Ca_j        = Ca[E_idx_v]
        Cb_j        = Cb[E_idx_v]
        dist_3d_vk  = (Ca_j - Ca[n_indices].unsqueeze(1)).norm(dim=-1)        # [N_v, K]
        dist_cb_vk  = (Cb_j - Cb[n_indices].unsqueeze(1)).norm(dim=-1)        # [N_v, K]
        seq_dist_vk = (E_idx_v.long() - n_indices.unsqueeze(-1).long()).abs().float()

        K_non_self = non_self_valid.float().sum(-1).clamp(min=1.0)  # [N_v]

        n_cys_edges = 0
        is_ssbond_edge = None
        if has_cys_edges:
            # is_ssbond per cys-cys edge, via the (chain, author res_id) identity.
            rows, cols = torch.where(cys_edges)
            is_ssbond_edge = np.zeros(rows.shape[0], dtype=bool)
            for e, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
                pos_i = int(n_indices[r])
                pos_j = int(E_idx_v[r, c])
                pair = frozenset({
                    (chain_of_pos[pos_i], res_id_of_pos[pos_i]),
                    (chain_of_pos[pos_j], res_id_of_pos[pos_j]),
                })
                is_ssbond_edge[e] = pair in ssbond_pairs

            n_cys_edges = int(rows.shape[0])
            all_seq_dist.append(seq_dist_vk[cys_edges].cpu().numpy())
            all_dist_3d.append(dist_3d_vk[cys_edges].cpu().numpy())
            all_dist_cb.append(dist_cb_vk[cys_edges].cpu().numpy())
            all_is_ssbond.append(is_ssbond_edge)
            all_res_idx_i.append(res_idx_i_vk[cys_edges].cpu().numpy())
            all_res_idx_j.append(res_idx_j_vk[cys_edges].cpu().numpy())
            all_protein_idx.append(np.full(n_cys_edges, n_proteins - 1, dtype=np.int64))
            all_name.extend([name] * n_cys_edges)

        # Random, capped subsample of non-Cys/non-Cys edges (candidate pool
        # for analyze's distance-matched control comparison).
        flat_valid = noncys_edges.reshape(-1)
        idx_pool   = torch.where(flat_valid)[0]
        if len(idx_pool) > ctrl_edges_per_protein:
            perm      = torch.randperm(len(idx_pool), generator=ctrl_rng)[:ctrl_edges_per_protein]
            idx_keep  = idx_pool[perm.to(idx_pool.device)]
        else:
            idx_keep  = idx_pool
        ctrl_mask = torch.zeros_like(flat_valid)
        ctrl_mask[idx_keep] = True
        ctrl_mask = ctrl_mask.reshape(noncys_edges.shape)
        n_ctrl_edges = int(idx_keep.shape[0])

        if n_ctrl_edges > 0:
            ctrl_seq_dist.append(seq_dist_vk[ctrl_mask].cpu().numpy())
            ctrl_dist_3d.append(dist_3d_vk[ctrl_mask].cpu().numpy())
            ctrl_dist_cb.append(dist_cb_vk[ctrl_mask].cpu().numpy())
            ctrl_res_idx_i.append(res_idx_i_vk[ctrl_mask].cpu().numpy())
            ctrl_res_idx_j.append(res_idx_j_vk[ctrl_mask].cpu().numpy())
            ctrl_protein_idx.append(np.full(n_ctrl_edges, n_proteins - 1, dtype=np.int64))
            ctrl_name.extend([name] * n_ctrl_edges)

        for l in range(L):
            alpha_vk   = attn_weights[l][0][valid_n]               # [N_v, K]
            alpha_self = (alpha_vk * is_self_vk.float()).sum(-1)    # [N_v]
            budget     = (1.0 - alpha_self).clamp(min=1e-8)         # [N_v]
            baseline   = budget / K_non_self                        # [N_v]
            rel_alpha  = alpha_vk / baseline.unsqueeze(-1)         # [N_v, K]
            budget_vk  = budget.unsqueeze(-1).expand_as(alpha_vk)  # [N_v, K]
            scored_vk  = rel_alpha ** budget_vk                     # [N_v, K]

            if has_cys_edges:
                all_attn[l].append(alpha_vk[cys_edges].cpu().numpy())
                all_rel_attn[l].append(rel_alpha[cys_edges].cpu().numpy())
                all_budget[l].append(budget_vk[cys_edges].cpu().numpy())
                all_scored_attn[l].append(scored_vk[cys_edges].cpu().numpy())

            if n_ctrl_edges > 0:
                ctrl_attn[l].append(alpha_vk[ctrl_mask].cpu().numpy())
                ctrl_rel_attn[l].append(rel_alpha[ctrl_mask].cpu().numpy())
                ctrl_budget[l].append(budget_vk[ctrl_mask].cpu().numpy())
                ctrl_scored_attn[l].append(scored_vk[ctrl_mask].cpu().numpy())

        n_ssbond = int(is_ssbond_edge.sum()) if is_ssbond_edge is not None else 0
        print(f"{n_cys_edges} cys-cys edges  ({n_ssbond} ssbond), {n_ctrl_edges} control edges")
        global_res_id += N_v

    if n_proteins == 0:
        raise RuntimeError("No structures were successfully processed.")
    if not all_seq_dist:
        raise RuntimeError("No Cys-Cys edges found in any processed structure.")
    if not ctrl_seq_dist:
        raise RuntimeError("No non-Cys/non-Cys control edges found in any processed structure.")

    out: dict = {
        "L":           np.array(L, dtype=np.int32),
        "n_proteins":  np.array(n_proteins, dtype=np.int32),
        "n_skipped":   np.array(n_skipped, dtype=np.int32),
        "name":        np.array(all_name, dtype=object),
        "protein_idx": np.concatenate(all_protein_idx).astype(np.int64),
        "res_idx_i":   np.concatenate(all_res_idx_i).astype(np.int64),
        "res_idx_j":   np.concatenate(all_res_idx_j).astype(np.int64),
        "seq_dist":    np.concatenate(all_seq_dist).astype(np.float32),
        "dist_3d":     np.concatenate(all_dist_3d).astype(np.float32),
        "dist_cb":     np.concatenate(all_dist_cb).astype(np.float32),
        "is_ssbond":   np.concatenate(all_is_ssbond).astype(bool),
        "ctrl_name":        np.array(ctrl_name, dtype=object),
        "ctrl_protein_idx": np.concatenate(ctrl_protein_idx).astype(np.int64),
        "ctrl_res_idx_i":   np.concatenate(ctrl_res_idx_i).astype(np.int64),
        "ctrl_res_idx_j":   np.concatenate(ctrl_res_idx_j).astype(np.int64),
        "ctrl_seq_dist":    np.concatenate(ctrl_seq_dist).astype(np.float32),
        "ctrl_dist_3d":     np.concatenate(ctrl_dist_3d).astype(np.float32),
        "ctrl_dist_cb":     np.concatenate(ctrl_dist_cb).astype(np.float32),
    }
    for l in range(L):
        out[f"attn_{l}"]        = np.concatenate(all_attn[l]).astype(np.float32)
        out[f"rel_attn_{l}"]    = np.concatenate(all_rel_attn[l]).astype(np.float32)
        out[f"budget_{l}"]      = np.concatenate(all_budget[l]).astype(np.float32)
        out[f"scored_attn_{l}"] = np.concatenate(all_scored_attn[l]).astype(np.float32)
        out[f"ctrl_attn_{l}"]        = np.concatenate(ctrl_attn[l]).astype(np.float32)
        out[f"ctrl_rel_attn_{l}"]    = np.concatenate(ctrl_rel_attn[l]).astype(np.float32)
        out[f"ctrl_budget_{l}"]      = np.concatenate(ctrl_budget[l]).astype(np.float32)
        out[f"ctrl_scored_attn_{l}"] = np.concatenate(ctrl_scored_attn[l]).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def _load_model(args: argparse.Namespace):
    device = torch.device(args.torch_device)
    is_oa  = args.mode == "oa"

    oa_kwargs = (
        dict(
            num_radial_orb        = args.num_radial_orb,
            r_max_orb              = args.r_max_orb,
            p_cutoff                = args.p_cutoff,
            l_max                   = args.l_max,
            weight_mlp_hidden_dim   = args.weight_mlp_hidden_dim,
            gate_hidden_dim         = args.gate_hidden_dim,
        )
        if is_oa else {}
    )

    model = ProteinMPNN.from_pretrained(
        args.checkpoint,
        map_location       = device,
        node_features      = args.hidden_dim,
        edge_features      = args.hidden_dim,
        hidden_dim         = args.hidden_dim,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        k_neighbors        = args.num_neighbors,
        dropout            = args.dropout,
        **oa_kwargs,
        **_MODE_KWARGS[args.mode],
    )
    model.to(device).eval()
    return model, device, is_oa


def _print_summary(stats: dict) -> None:
    L        = int(stats["L"])
    is_ssbond = stats["is_ssbond"]
    n_edges  = len(is_ssbond)
    n_ss     = int(is_ssbond.sum())

    n_ctrl = len(stats["ctrl_dist_3d"])

    print(f"\n{'─' * 60}")
    print(f"  proteins processed : {int(stats['n_proteins'])}  (skipped {int(stats['n_skipped'])})")
    print(f"  cys-cys edges       : {n_edges:,}  (ssbond={n_ss:,}  non_ssbond={n_edges - n_ss:,})")
    print(f"  control edges       : {n_ctrl:,}  (non-Cys/non-Cys candidate pool)")
    for l in range(L):
        scored = stats[f"scored_attn_{l}"]
        m_ss  = float(scored[is_ssbond].mean())  if n_ss > 0        else float("nan")
        m_non = float(scored[~is_ssbond].mean()) if n_ss < n_edges else float("nan")
        print(f"  layer {l + 1}  mean_scored_attn  ssbond={m_ss:.3f}  non_ssbond={m_non:.3f}")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# Statistical analysis — ported from test.ipynb (see module docstring)
# ---------------------------------------------------------------------------

@dataclass
class _AnalysisConfig:
    sep_min:      int   = 6
    n_boot:       int   = 5000
    boot_seed:    int   = 0
    support_q:    tuple = (0.005, 0.995)
    min_minority: int   = 25
    bin_mode:     str   = "adaptive"   # "adaptive" | "fixed"
    bin_width:    float = 0.5
    match_seed:   int   = 0
    joint_match:  tuple = ("dist_cb", "dist")   # axes for the jointly-matched analysis F
    n_jobs:       int   = -1   # joblib workers for the bootstrap (-1 = all cores)


def _collapse_directed_edges(d: "pd.DataFrame", n_dir: int, lo: "pd.Series", hi: "pd.Series") -> "pd.DataFrame":
    """One row per undirected Cys-Cys pair, averaging dist/dist_cb/score
    across the (up to 2) directed i->j / j->i rows accumulate() stores for
    it. *lo*/*hi* are elementwise min/max(res_idx_i, res_idx_j), so both
    directions of the same pair land on the same (prot, lo, hi) key."""
    d = d.assign(pair=d.prot + "|" + lo.astype(str) + "|" + hi.astype(str))
    g = d.groupby("pair", as_index=False).agg(
        prot=("prot", "first"), bonded=("bonded", "max"),
        dist=("dist", "mean"), dist_cb=("dist_cb", "mean"), score=("score", "mean"),
        sep=("sep", "first"), n_dir=("score", "size"))
    print(f"  directed edges (|i-j|>=sep_min): {n_dir}  ->  undirected pairs: {len(g)}")
    print(f"    {int(g.bonded.sum())} bonded, {int((~g.bonded).sum())} non-bonded; "
          f"{int((g.n_dir == 1).sum())} pairs seen in one direction only")
    return g.drop(columns="n_dir")


def _shared_support(d: "pd.DataFrame", by: str, cfg: _AnalysisConfig) -> tuple:
    """Quantile range (--support_q) shared by the bonded and non-bonded
    distributions of column *by*, so bins are only formed where both
    classes actually have data."""
    qlo, qhi = cfg.support_q
    b, nb = d.loc[d.bonded, by], d.loc[~d.bonded, by]
    return float(max(b.quantile(qlo), nb.quantile(qlo))), \
           float(min(b.quantile(qhi), nb.quantile(qhi)))


def _make_bin_edges(d: "pd.DataFrame", by: str, cfg: _AnalysisConfig) -> tuple:
    """Bin edges over column *by*, restricted to _shared_support. In
    "adaptive" mode (default), edges are quantiles of the minority class
    (whichever of bonded/non-bonded is rarer) chosen so each bin holds
    >= --min_minority minority-class examples; in "fixed" mode, edges are
    a uniform --bin_width grid."""
    lo, hi = _shared_support(d, by, cfg)
    if cfg.bin_mode == "fixed":
        return np.arange(lo, hi + cfg.bin_width / 2, cfg.bin_width), lo, hi
    m = d[(d[by] >= lo) & (d[by] <= hi)]
    minor = m.bonded.to_numpy() if m.bonded.sum() < (~m.bonded).sum() else ~m.bonded.to_numpy()
    vals = np.sort(m.loc[minor, by].to_numpy())
    k = max(2, len(vals) // cfg.min_minority)
    e = np.unique(np.round(np.quantile(vals, np.linspace(0, 1, k + 1)), 2))
    e[0], e[-1] = lo, hi + 1e-9
    return e, lo, hi


def _as_axes(by) -> tuple:
    """Normalise *by* (a column name, or a sequence of them) to a tuple."""
    return (by,) if isinstance(by, str) else tuple(by)


def _bins_per_axis(d: "pd.DataFrame", by, cfg: _AnalysisConfig) -> tuple:
    """Bin edges for each matching axis, as {axis: edges}, plus
    {axis: (lo, hi)} shared-support bounds.

    One axis reproduces _make_bin_edges exactly (unchanged behaviour). With
    D > 1 axes the strata are the cross-product of the per-axis bins, so the
    number of bins *per axis* is reduced to k = (n_minority / min_minority)
    ** (1/D): that keeps roughly --min_minority minority-class pairs per
    CELL rather than per axis-slice, which is what the pooled statistic
    actually needs. Edges are quantiles of the minority class computed on
    the region where every axis is inside its shared support."""
    axes = _as_axes(by)
    if len(axes) == 1:
        e, lo, hi = _make_bin_edges(d, axes[0], cfg)
        return {axes[0]: e}, {axes[0]: (lo, hi)}

    sup = {a: _shared_support(d, a, cfg) for a in axes}
    m = d
    for a in axes:
        m = m[(m[a] >= sup[a][0]) & (m[a] <= sup[a][1])]
    n_minor = int(min(m.bonded.sum(), (~m.bonded).sum()))
    k = max(2, int((n_minor / cfg.min_minority) ** (1.0 / len(axes))))
    minor = m.bonded.to_numpy() if m.bonded.sum() < (~m.bonded).sum() else ~m.bonded.to_numpy()
    edges = {}
    for a in axes:
        vals = np.sort(m.loc[minor, a].to_numpy())
        e = np.unique(np.round(np.quantile(vals, np.linspace(0, 1, k + 1)), 2))
        e[0], e[-1] = sup[a][0], sup[a][1] + 1e-9
        edges[a] = e
    return edges, sup


def _assign_bins(d: "pd.DataFrame", by, edges: dict) -> "pd.DataFrame":
    """Restrict *d* to the joint support of *edges* and add a "bin" column.

    With one axis "bin" is the pandas Interval categorical the single-axis
    code (and _plot_matched_control) already expects, so nothing changes.
    With several it is a string label naming the cell on every axis, and
    per-axis "bin_<axis>" columns are kept alongside for the CSV."""
    axes = _as_axes(by)
    keep = np.ones(len(d), bool)
    for a in axes:
        v = d[a].to_numpy()
        keep &= (v >= edges[a][0]) & (v < edges[a][-1])
    d = d[keep].copy()
    cats = {a: pd.cut(d[a], edges[a], right=False) for a in axes}
    if len(axes) == 1:
        d["bin"] = cats[axes[0]]
        return d
    for a in axes:
        d[f"bin_{a}"] = cats[a]
    # Interval Categorical .astype(str) raises a spurious "invalid value
    # encountered in cast" RuntimeWarning even with zero actual NaNs (a
    # pandas quirk: it unconditionally casts the dtype's NA placeholder
    # too) — same harmless-warning situation vis._bootstrap.double_center
    # already suppresses around its own nanmean calls.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        str_cats = [cats[a].astype(str) for a in axes]
    d["bin"] = pd.Categorical(
        [" x ".join(f"{a}{iv}" for a, iv in zip(axes, t)) for t in zip(*str_cats)])
    return d


def _bin_table(d: "pd.DataFrame", by, score: str, edges: dict, log_effect: bool = True) -> tuple:
    """Per-bin (n_bond, n_non, prevalence, AUC, effect size) over *edges*.
    AUC is the Mann-Whitney U statistic normalized to [0, 1] (probability
    a random bonded edge outscores a random non-bonded one); effect is the
    mean log2(score) difference (log_effect=True) or mean score difference
    (log_effect=False) between bonded and non-bonded edges in that bin.
    Bins with no edges of one class get NaN AUC/effect (still counted)."""
    d = _assign_bins(d, by, edges)
    rows = []
    for b, g in d.groupby("bin", observed=True):
        n1, n0 = int(g.bonded.sum()), int((~g.bonded).sum())
        if n1 == 0 or n0 == 0:
            rows.append(dict(bin=b, n_bond=n1, n_non=n0, prev=np.nan,
                             auc=np.nan, eff=np.nan)); continue
        y, sc = g.bonded.to_numpy(), g[score].to_numpy()
        U = rankdata(sc)[y].sum() - n1 * (n1 + 1) / 2
        if log_effect:
            lg = np.log2(np.clip(sc, 1e-6, None)); eff = lg[y].mean() - lg[~y].mean()
        else:
            eff = sc[y].mean() - sc[~y].mean()
        rows.append(dict(bin=b, n_bond=n1, n_non=n0, prev=n1 / (n1 + n0),
                         auc=U / (n1 * n0), eff=eff))
    return d, pd.DataFrame(rows)


def _pooled(tab: "pd.DataFrame") -> tuple:
    """Pool per-bin AUC/effect across bins: AUC is weighted by n_bond*n_non
    (its variance scale), effect by the harmonic-mean-of-counts weight
    (the standard fixed-effect meta-analysis weight for a mean difference
    with per-group variance ~ 1/n)."""
    t = tab.dropna(subset=["auc"])
    if t.empty:
        return np.nan, np.nan
    w, nh = t.n_bond * t.n_non, 2 / (1 / t.n_bond + 1 / t.n_non)
    return float((w * t.auc).sum() / w.sum()), float((nh * t.eff).sum() / nh.sum())


@dataclass
class _BootExtra:
    """One bootstrap replicate's recipe, bundled into one explicit argument
    so joblib's loky backend can memory-map the large arrays instead of
    re-pickling them per task (same convention as vis._bootstrap's own
    MainBootExtra). ``columns`` holds exactly the plain numpy arrays
    _bin_table needs (one per matching axis, the score column, and
    "bonded"), aligned to the row universe _bootstrap_ci resamples over —
    a DataFrame is cheap to rebuild from these inside each worker, and
    _bin_table/_pooled then run completely unchanged, single- or
    multi-axis alike."""
    columns:    dict
    by:         tuple
    score:      str
    edges:      dict
    log_effect: bool


def _bootstrap_compute(idx_pool: np.ndarray, extra: _BootExtra) -> dict:
    """One replicate: rebuild the resampled rows as a small DataFrame, bin
    + pool exactly as the point estimate does. Used both inside the
    parallel bootstrap and (trivially, with idx_pool = every row) nowhere
    else — the point estimate keeps calling _bin_table/_pooled directly in
    _analyse, since it also needs the per-bin table to print/plot."""
    resampled = pd.DataFrame({k: v[idx_pool] for k, v in extra.columns.items()})
    _, tab = _bin_table(resampled, extra.by, extra.score, extra.edges, extra.log_effect)
    auc, eff = _pooled(tab)
    return {"auc": auc, "eff": eff}


def _bootstrap_ci(
    d: "pd.DataFrame", by, score: str, edges: dict, cfg: _AnalysisConfig, parallel: Parallel,
    log_effect: bool = True,
) -> np.ndarray:
    """95% CI (2.5th/97.5th percentile) on pooled (AUC, effect) via a
    protein-level cluster bootstrap: each of --n_boot resamples draws whole
    proteins with replacement (not individual edges), since edges from the
    same protein aren't independent observations. Resampling + the per-
    replicate bin/pool both run in *parallel* (a joblib worker pool shared
    across every _analyse call in one ``analyze`` run — see vis._bootstrap,
    the same machinery vis.__main__/vis.persistence use), batched so each
    task runs several replicates rather than dispatching one per task."""
    axes = _as_axes(by)
    n = len(d)
    if n == 0:
        return np.full((2, 2), np.nan)

    protein_ids, _ = pd.factorize(d["prot"].to_numpy())
    protein_ids = protein_ids.astype(np.int64)
    n_local = int(protein_ids.max()) + 1
    mask = np.ones(n, dtype=bool)
    flat_idx, starts, lengths = build_layout(mask, protein_ids, n_local)

    columns = {a: d[a].to_numpy(dtype=np.float64) for a in axes}
    columns[score] = d[score].to_numpy(dtype=np.float64)
    columns["bonded"] = d["bonded"].to_numpy(dtype=bool)
    extra = _BootExtra(columns=columns, by=axes, score=score, edges=edges, log_effect=log_effect)
    bd = BootData(flat_idx=flat_idx, starts=starts, lengths=lengths,
                  n_proteins=n_local, extra=extra)

    seeds = np.random.SeedSequence(cfg.boot_seed).spawn(cfg.n_boot)
    n_tasks = min(cfg.n_boot, 4 * (os.cpu_count() or 1))
    batch_size = -(-cfg.n_boot // n_tasks)
    results, n_none = run_bootstrap(parallel, seeds, bd, batch_size, _bootstrap_compute)
    if n_none:
        print(f"    ({n_none}/{cfg.n_boot} empty bootstrap replicates)")
    if not results:
        return np.full((2, 2), np.nan)

    out = np.array([[r["auc"], r["eff"]] for r in results], dtype=float)
    return np.nanpercentile(out, [2.5, 97.5], axis=0)


def _sample_matched_control(
    pool: "pd.DataFrame", by: str, edges: np.ndarray, target_counts: "pd.Series",
    rng: np.random.Generator,
) -> "pd.DataFrame":
    """Sample rows from *pool* (the collapsed non-Cys/non-Cys control
    population, "bonded" already False for every row) so the result has
    the same per-bin counts as *target_counts* — typically a bin_table's
    n_non column, i.e. the real Cys-Cys non-bonded group's distribution —
    over the same *by*/*edges* another analysis already binned on. This
    makes the sampled control group "distance-matched" to whatever
    non-bonded population it's standing in for: same total size, same
    distance distribution. Samples without replacement when the pool has
    enough edges in a bin; otherwise takes every available edge in that
    bin and prints a warning (the matched count falls short there)."""
    cols = ["prot", "bonded", "dist", "dist_cb", "score", "sep"]
    pool = _assign_bins(pool, by, edges)
    sampled = []
    for b, n in target_counts.items():
        n = int(n)
        if n == 0:
            continue
        # Select cols (dropping "bin"/"bin_<axis>") BEFORE .sample(): a
        # Categorical column (Interval for one axis, string labels for
        # several) makes pandas' internal hashing choke on its NA category
        # placeholder, cast to float — harmless in the end (sampling still
        # works) but noisy (RuntimeWarning per call, one per bin).
        avail = pool.loc[pool["bin"] == b, cols]
        if len(avail) < n:
            print(f"    WARNING: control pool bin {b} has only {len(avail)} < {n} "
                  f"edges needed for matching — using all available")
            sampled.append(avail)
        else:
            sampled.append(avail.sample(n=n, replace=False, random_state=rng))
    if not sampled:
        return pool.iloc[0:0][cols]
    return pd.concat(sampled, ignore_index=True)


def _analyse(
    d: "pd.DataFrame", by, score: str, label: str, unit: str, cfg: _AnalysisConfig,
    parallel: Parallel, log_effect: bool = True, edges: dict | None = None,
) -> tuple:
    """Run one full stratified analysis (bin, pool, bootstrap) and print
    its per-bin table plus the pooled AUC/effect with 95% CI.

    *by* is one column name or a sequence of them; with several, the strata
    are the cross-product of the per-axis bins (see _bins_per_axis), which
    is how a comparison is matched on more than one geometric quantity at
    once. Pass *edges* to reuse another analysis's strata verbatim — needed
    when a sampled control group must land in the same cells as the group
    it stands in for. *parallel* is a joblib worker pool (shared across
    every _analyse call in one ``analyze`` run) used for the bootstrap.
    Returns (sub, tab, stats, edges); *sub* is *d* restricted to the
    occupied cells and tagged with its "bin" column, for the figure;
    *stats* is {auc, auc_lo, auc_hi, eff, eff_lo, eff_hi}."""
    axes = _as_axes(by)
    if edges is None:
        edges, sup = _bins_per_axis(d, axes, cfg)
    else:
        sup = {a: (float(edges[a][0]), float(edges[a][-1])) for a in axes}
    sub, tab = _bin_table(d, axes, score, edges, log_effect)
    auc, eff = _pooled(tab)
    (al, el), (ah, eh) = _bootstrap_ci(sub, axes, score, edges, cfg, parallel, log_effect)
    strata = ", ".join(f"{a} ({sup[a][0]:.2f}-{sup[a][1]:.2f} A, {len(edges[a])-1} bins)"
                       for a in axes)
    n_occ = int(tab.auc.notna().sum())
    print(f"\n  [{label}]  strata: {strata}"
          + (f" -> {len(tab)} cells, {n_occ} with both classes" if len(axes) > 1 else "")
          + f", score: {score}")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"    stratified AUC  {auc:.3f}  [{al:.3f}, {ah:.3f}]")
    print(f"    effect ({unit})  {eff:+.2f}  [{el:+.2f}, {eh:+.2f}]")
    stats = dict(auc=auc, auc_lo=al, auc_hi=ah, eff=eff, eff_lo=el, eff_hi=eh)
    return sub, tab, stats, edges


def _plot_matched_control(dA: "pd.DataFrame", dC: "pd.DataFrame", path: str) -> str:
    """Two-panel figure: log2(score) vs. dist_3d (analysis A's pairs, left)
    and vs. dist_cb (analysis C's pairs, right), split SSBOND / not
    annotated, each with a per-bin median step line. Saved as SVG (vector
    axes/legend/step lines, with the dense per-edge scatter rasterized at
    its own dpi — otherwise the SVG would embed one path per edge)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, d, by, xl in [(axes[0], dA, "dist", r"C$\alpha$–C$\alpha$ distance ($\mathrm{\AA}$)"),
                          (axes[1], dC, "dist_cb", r"C$\beta$–C$\beta$ distance ($\mathrm{\AA}$)")]:
        for flag, col, lab in [(False, "#4a3b8c", "not annotated"), (True, "#e8b60b", "SSBOND")]:
            g = d[d.bonded == flag]
            ax.scatter(g[by], np.log2(np.clip(g.score, 1e-6, None)), s=3, c=col,
                       alpha=.55, lw=0, label=lab, rasterized=True)
            med = (g.assign(l=np.log2(np.clip(g.score, 1e-6, None)))
                    .groupby("bin", observed=True).l.median())
            if len(med):
                ax.step([iv.mid for iv in med.index], med.values, where="mid",
                        color=col, lw=1.8)
        ax.axhline(0, color="k", lw=.8, ls="--")
        ax.set_xlabel(xl); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"budget-weighted score (log$_2$)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _table_rows(analysis: str, label: str, by: str, score: str, tab: "pd.DataFrame", stats: dict) -> list:
    """One row per bin of *tab* (bin_lo/bin_hi split out of its "bin"
    Interval column; auc_lo/auc_hi/eff_lo/eff_hi left blank — those are
    only defined for the pooled estimate), plus one trailing "POOLED" row
    carrying *stats*'s pooled AUC/effect and 95% CI."""
    rows = []
    for r in tab.itertuples(index=False):
        iv = isinstance(r.bin, pd.Interval)
        rows.append(dict(
            analysis=analysis, label=label, by=by, score=score,
            bin_lo=r.bin.left if iv else str(r.bin), bin_hi=r.bin.right if iv else "",
            n_bond=r.n_bond, n_non=r.n_non, prev=r.prev, auc=r.auc, eff=r.eff,
            auc_lo=np.nan, auc_hi=np.nan, eff_lo=np.nan, eff_hi=np.nan,
        ))
    rows.append(dict(
        analysis=analysis, label=label, by=by, score=score,
        bin_lo="POOLED", bin_hi="POOLED",
        n_bond=int(tab.n_bond.sum()), n_non=int(tab.n_non.sum()), prev=np.nan,
        auc=stats["auc"], eff=stats["eff"],
        auc_lo=stats["auc_lo"], auc_hi=stats["auc_hi"],
        eff_lo=stats["eff_lo"], eff_hi=stats["eff_hi"],
    ))
    return rows


def _save_tables(analyses: list, path: str) -> str:
    """Write one combined CSV from *analyses* — a list of
    (analysis, label, by, score, tab, stats) tuples, one per call to
    _analyse — with one row per bin plus a trailing POOLED row per
    analysis (see _table_rows). Saves to *path*."""
    rows = []
    for analysis, label, by, score, tab, stats in analyses:
        rows.extend(_table_rows(analysis, label, by, score, tab, stats))
    df = pd.DataFrame(rows, columns=[
        "analysis", "label", "by", "score", "bin_lo", "bin_hi",
        "n_bond", "n_non", "prev", "auc", "eff", "auc_lo", "auc_hi", "eff_lo", "eff_hi",
    ])
    df.to_csv(path, index=False)
    return path


def _load_control_pool(obj, layer: int, cfg: _AnalysisConfig) -> "pd.DataFrame | None":
    """Load + collapse the ctrl_* (non-Cys/non-Cys) fields from an open
    .npz into the same (prot, bonded=False, dist, dist_cb, score) shape as
    the Cys-Cys collapsed frame, for _sample_matched_control to draw from.
    Returns None (with a warning) if the .npz predates the ctrl_* fields
    (written by an older ``compute``)."""
    score_key = f"ctrl_scored_attn_{layer}"
    cols = dict(dist="ctrl_dist_3d", dist_cb="ctrl_dist_cb", score=score_key,
                sep="ctrl_seq_dist", protein="ctrl_protein_idx",
                res_i="ctrl_res_idx_i", res_j="ctrl_res_idx_j")
    avail = list(obj.files)
    if missing := [v for v in cols.values() if v not in avail]:
        print(f"  WARNING: non-Cys control comparison skipped — missing {missing} "
              f"(re-run 'compute' with the current script version to enable it)")
        return None

    raw = pd.DataFrame({k: np.asarray(obj[v]).ravel() for k, v in cols.items()})
    raw = raw.loc[raw["sep"] >= cfg.sep_min].copy()
    raw["bonded"] = False
    raw["prot"] = raw.pop("protein").astype(str)
    n_dir = len(raw)
    i, j = raw.pop("res_i").astype(np.int64), raw.pop("res_j").astype(np.int64)
    print("  [control pool]", end=" ")
    return _collapse_directed_edges(raw, n_dir, np.minimum(i, j), np.maximum(i, j))


def analyze_npz(
    npz_path: str, layer: int, cfg: _AnalysisConfig, fig_out: str, tables_out: str,
) -> None:
    """Load *npz_path* (written by ``compute``) and run the three Cys-Cys
    analyses (A/B/C, see module docstring) against ``scored_attn_{layer}``,
    plus — when the .npz has the ctrl_* (non-Cys/non-Cys) fields — six
    more comparing a Cys-Cys class against a matched sample of non-Cys
    control edges, each matched on Ca-Ca (D), virtual Cb-Cb (E) or both
    jointly (F, via --joint_match, which removes the residual Ca-Ca
    difference that E leaves free; C2 is F's Cys-Cys reference on the same
    joint strata) — with D/E/F using SSBOND as the Cys-Cys class and
    G/H/I repeating the same three comparisons with the non-annotated
    (non-bonded) Cys-Cys pairs as the class instead, reusing D/E/F's exact
    matched control samples. Prints each one's per-bin table and pooled
    AUC/effect, saves the matched-control figure (A/C only) to *fig_out*,
    and all per-bin tables (plus pooled AUC/effect + 95% CI) as a combined
    CSV to *tables_out*."""
    obj = np.load(npz_path, allow_pickle=True)
    score_key = f"scored_attn_{layer}"
    cols = dict(dist="dist_3d", dist_cb="dist_cb", score=score_key,
                bond="is_ssbond", sep="seq_dist", protein="protein_idx",
                res_i="res_idx_i", res_j="res_idx_j")

    avail = list(obj.files)
    print("  fields:", ", ".join(avail))
    if missing := [v for v in cols.values() if v not in avail]:
        raise SystemExit(
            f"not found in {npz_path!r}: {missing} "
            f"(--layer {layer} needs {score_key!r} — is it within [0, L)?)"
        )

    raw = pd.DataFrame({k: np.asarray(obj[v]).ravel() for k, v in cols.items()})
    raw = raw.loc[raw["sep"] >= cfg.sep_min].copy()
    raw["bonded"] = raw.pop("bond").astype(bool)
    raw["prot"] = raw.pop("protein").astype(str)
    n_dir = len(raw)
    i, j = raw.pop("res_i").astype(np.int64), raw.pop("res_j").astype(np.int64)
    d = _collapse_directed_edges(raw, n_dir, np.minimum(i, j), np.maximum(i, j))

    # One joblib worker pool, shared by every _analyse call below (A-F) —
    # avoids paying worker-startup cost per analysis (see vis._bootstrap /
    # vis.__main__'s identical single-pool-for-the-whole-run convention).
    with Parallel(n_jobs=cfg.n_jobs, return_as="generator_unordered", max_nbytes="1M") as parallel:
        dA, tabA, statsA, edgesA = _analyse(
            d, "dist", "score", "A  main", "dlog2 score", cfg, parallel)
        d = d.assign(neg_cb=-d.dist_cb)
        _, tabB, statsB, _ = _analyse(d, "dist", "neg_cb", "B  Cb reference", "dCb (A)", cfg,
                                       parallel, log_effect=False)
        dC, tabC, statsC, edgesC = _analyse(d, "dist_cb", "score", "C  Cb-matched control",
                                             "dlog2 score", cfg, parallel)
        print(f"\n  summary: attention|Ca {statsA['auc']:.3f}   Cb alone|Ca {statsB['auc']:.3f}   "
              f"attention|Cb {statsC['auc']:.3f}")

        analyses = [
            ("A", "main",               "dist",    "score",  tabA, statsA),
            ("B", "Cb reference",       "dist",    "neg_cb", tabB, statsB),
            ("C", "Cb-matched control", "dist_cb", "score",  tabC, statsC),
        ]

        ctrl_pool = _load_control_pool(obj, layer, cfg)
        if ctrl_pool is not None:
            bonded_rows = d.loc[d.bonded, ["prot", "bonded", "dist", "dist_cb", "score", "sep"]]
            # Non-annotated Cys-Cys pairs, relabeled "bonded" (the positive
            # class) for G/H/I below — same rows D/E/F treat as the real
            # non-bonded reference, now standing in for SSBOND itself.
            nonbond_rows = d.loc[~d.bonded, ["prot", "bonded", "dist", "dist_cb", "score", "sep"]] \
                .assign(bonded=True)
            match_rng = np.random.default_rng(cfg.match_seed)

            # D/E reuse A/C's bin edges (edges=edgesA / edgesC below) rather than
            # re-deriving them from the combined frame, as this previously did:
            # the control is SAMPLED to match A/C's per-bin counts, so analysing it
            # in different strata partly undoes the matching. The fix is a
            # consistency one and barely moves the estimate — on a synthetic check
            # it changed the pooled AUC in the third decimal only (D 0.764 -> 0.767,
            # E 0.754 -> 0.753), via slightly different shared-support bounds.
            sampled_ca = _sample_matched_control(
                ctrl_pool, "dist", edgesA, tabA.set_index("bin")["n_non"], match_rng)
            dD = pd.concat([bonded_rows, sampled_ca], ignore_index=True)
            _, tabD, statsD, _ = _analyse(
                dD, "dist", "score", "D  non-Cys control (Ca-matched)", "dlog2 score", cfg,
                parallel, edges=edgesA)

            sampled_cb = _sample_matched_control(
                ctrl_pool, "dist_cb", edgesC, tabC.set_index("bin")["n_non"], match_rng)
            dE = pd.concat([bonded_rows, sampled_cb], ignore_index=True)
            _, tabE, statsE, _ = _analyse(
                dE, "dist_cb", "score", "E  non-Cys control (Cb-matched)", "dlog2 score", cfg,
                parallel, edges=edgesC)

            print(f"\n  summary: attention|nonCys(Ca) {statsD['auc']:.3f}   "
                  f"attention|nonCys(Cb) {statsE['auc']:.3f}")
            analyses += [
                ("D", "non-Cys control (Ca-matched)", "dist",    "score", tabD, statsD),
                ("E", "non-Cys control (Cb-matched)", "dist_cb", "score", tabE, statsE),
            ]

            # G/H — the same non-Cys comparison as D/E, but with the
            # NON-annotated Cys-Cys pairs standing in as the "bonded" class
            # instead of SSBOND: does the model's score still separate any
            # Cys-Cys pair from a distance-matched non-Cys/non-Cys pair, or
            # is that separation specific to actual disulfides? Reuses
            # sampled_ca/sampled_cb verbatim (same control sample, same
            # edges) since they were already matched to the per-bin COUNT
            # of non-bonded Cys-Cys pairs — exactly this group.
            _, tabG, statsG, _ = _analyse(
                pd.concat([nonbond_rows, sampled_ca], ignore_index=True),
                "dist", "score", "G  non-Cys control (Ca-matched, non-annotated Cys-Cys)",
                "dlog2 score", cfg, parallel, edges=edgesA)

            _, tabH, statsH, _ = _analyse(
                pd.concat([nonbond_rows, sampled_cb], ignore_index=True),
                "dist_cb", "score", "H  non-Cys control (Cb-matched, non-annotated Cys-Cys)",
                "dlog2 score", cfg, parallel, edges=edgesC)

            print(f"\n  summary: attention|nonCys(Ca, non-annotated) {statsG['auc']:.3f}   "
                  f"attention|nonCys(Cb, non-annotated) {statsH['auc']:.3f}")
            analyses += [
                ("G", "non-Cys control (Ca-matched, non-annotated Cys-Cys)",
                 "dist", "score", tabG, statsG),
                ("H", "non-Cys control (Cb-matched, non-annotated Cys-Cys)",
                 "dist_cb", "score", tabH, statsH),
            ]

            # F — the same non-Cys comparison, matched JOINTLY on several axes
            # (default Cb and Ca). E matches on Cb alone, which leaves Ca free:
            # Cys-Cys pairs inside the close-Cb window are overwhelmingly
            # disulfides, with Ca tightly constrained and the Cb vectors facing
            # each other, whereas a non-Cys pair whose *virtual* Cb atoms happen
            # to be that close can sit at almost any Ca separation. One class is
            # then a tight cluster in geometry space and the other diffuse, so
            # E's separation can be the residual Ca difference rather than
            # anything about the pair. F closes that route; add "sep" to
            # --joint_match to close the sequence-separation route too.
            axes_f = tuple(cfg.joint_match)
            if missing_ax := [a for a in axes_f if a not in d.columns]:
                print(f"  WARNING: joint analysis F skipped — no such column(s) {missing_ax}")
            else:
                _, tabCj, statsCj, edgesF = _analyse(
                    d, axes_f, "score", "C2 Cb-matched control (joint strata)",
                    "dlog2 score", cfg, parallel)
                sampled_j = _sample_matched_control(
                    ctrl_pool, axes_f, edgesF, tabCj.set_index("bin")["n_non"], match_rng)
                dF = pd.concat([bonded_rows, sampled_j], ignore_index=True)
                _, tabF, statsF, _ = _analyse(
                    dF, axes_f, "score", f"F  non-Cys control ({'+'.join(axes_f)}-matched)",
                    "dlog2 score", cfg, parallel, edges=edgesF)
                print(f"\n  summary: attention|nonCys({'+'.join(axes_f)}) {statsF['auc']:.3f}"
                      f"   (vs {statsE['auc']:.3f} matched on dist_cb alone)")
                analyses += [
                    ("C2", "Cb-matched control (joint strata)",
                     "+".join(axes_f), "score", tabCj, statsCj),
                    ("F", f"non-Cys control ({'+'.join(axes_f)}-matched)",
                     "+".join(axes_f), "score", tabF, statsF),
                ]

                # I — G/H's "non-annotated Cys-Cys vs non-Cys" comparison,
                # matched jointly (--joint_match, "both levels" by default)
                # the same way F tightens E. Reuses sampled_j verbatim.
                _, tabI, statsI, _ = _analyse(
                    pd.concat([nonbond_rows, sampled_j], ignore_index=True),
                    axes_f, "score",
                    f"I  non-Cys control ({'+'.join(axes_f)}-matched, non-annotated Cys-Cys)",
                    "dlog2 score", cfg, parallel, edges=edgesF)
                print(f"\n  summary: attention|nonCys({'+'.join(axes_f)}, non-annotated) "
                      f"{statsI['auc']:.3f}   (vs {statsH['auc']:.3f} matched on dist_cb alone)")
                analyses += [
                    ("I", f"non-Cys control ({'+'.join(axes_f)}-matched, non-annotated Cys-Cys)",
                     "+".join(axes_f), "score", tabI, statsI),
                ]

    print(f"  figure -> {_plot_matched_control(dA, dC, fig_out)}")
    print(f"  tables -> {_save_tables(analyses, tables_out)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "compute", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Run forward passes over Cys-Cys edges and save the per-edge .npz.",
    )
    c.add_argument("pdb_dirs", nargs="+",
                    help="Directory (or directories) of .pdb files with embedded "
                         "SSBOND records — typically data/test_pdb_raw (the full "
                         "multi-chain structures, every SSBOND captured) — or "
                         "individual .pdb file paths. The single-chain split dirs "
                         "(data/test_monomers_pdb etc.) also work, with reduced "
                         "SSBOND coverage (see module docstring)")
    c.add_argument("checkpoint", help="Model checkpoint (.pt)")
    c.add_argument("out_npz",    help="Output .npz file path")

    d = c.add_argument_group("Data")
    d.add_argument("--max_protein_length", type=int, default=DATA_MAX_PROTEIN_LENGTH,
                   help="Maximum protein length in residues; longer structures are skipped")
    d.add_argument("--ctrl_edges_per_protein", type=int, default=500,
                   help="Max number of non-Cys/non-Cys non-self edges randomly "
                        "subsampled per structure, forming analyze's "
                        "distance-matched control candidate pool (ctrl_* keys)")
    d.add_argument("--ctrl_seed", type=int, default=0,
                   help="Seed for the per-structure control-edge subsampling")

    m = c.add_argument_group("Model  (must match checkpoint)")
    m.add_argument("--mode", default="oa", choices=["oa", "gat"],
                   help="Encoder mode: 'oa' (orbital attention with degree gating), "
                        "'gat' (edge-driven graph attention, no geometry)")
    m.add_argument("--hidden_dim",          type=int,   default=PMPNN_HIDDEN_DIM)
    m.add_argument("--num_encoder_layers",  type=int,   default=PMPNN_NUM_ENCODER_LAYERS)
    m.add_argument("--num_decoder_layers",  type=int,   default=PMPNN_NUM_DECODER_LAYERS)
    m.add_argument("--num_neighbors",       type=int,   default=PMPNN_K_NEIGHBORS)
    m.add_argument("--dropout",             type=float, default=PMPNN_DROPOUT)

    oa = c.add_argument_group("Orbital Attention hypers  (ignored for --mode gat)")
    oa.add_argument("--num_radial_orb",        type=int,   default=OA_NUM_RADIAL)
    oa.add_argument("--r_max_orb",             type=float, default=OA_R_MAX)
    oa.add_argument("--p_cutoff",              type=int,   default=OA_P_CUTOFF)
    oa.add_argument("--l_max",                 type=int,   default=OA_L_MAX)
    oa.add_argument("--weight_mlp_hidden_dim", type=int,   default=OA_WEIGHT_MLP_HIDDEN_DIM)
    oa.add_argument("--gate_hidden_dim",       type=int,   default=OA_GATE_HIDDEN_DIM)

    c.add_argument("--device", default="cpu", dest="torch_device",
                   help="Torch device (cpu / cuda / cuda:N)")

    a = sub.add_parser(
        "analyze", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Load a .npz written by 'compute' and report the SSBOND-vs-score "
             "statistical analysis + matched-control figure.",
    )
    a.add_argument("npz", help="Input .npz file path, written by 'compute'")
    a.add_argument("--layer", type=int, default=0,
                   help="Encoder layer whose scored_attn_{layer} to analyze")
    a.add_argument("--fig_out", default="ssbond_matched.svg",
                   help="Output path for the matched-control figure")
    a.add_argument("--tables_out", default="ssbond_tables.csv",
                   help="Output path for the combined per-bin tables CSV "
                        "(one row per bin plus a pooled POOLED row, per analysis)")

    s = a.add_argument_group("Binning / bootstrap")
    s.add_argument("--sep_min", type=int, default=6,
                   help="Minimum |seq_dist| for an edge to be included")
    s.add_argument("--n_boot", type=int, default=5000,
                   help="Number of protein-level cluster-bootstrap resamples")
    s.add_argument("--boot_seed", type=int, default=0,
                   help="Seed for the bootstrap's random resampling")
    s.add_argument("--support_q", type=float, nargs=2, default=(0.005, 0.995),
                   metavar=("LO", "HI"),
                   help="Quantile range shared by the bonded/non-bonded "
                        "distributions used to form bins")
    s.add_argument("--min_minority", type=int, default=25,
                   help="Minimum minority-class (usually SSBOND) edges per "
                        "bin in --bin_mode adaptive")
    s.add_argument("--bin_mode", choices=["adaptive", "fixed"], default="adaptive",
                   help="'adaptive': quantile bins sized by --min_minority; "
                        "'fixed': uniform --bin_width grid")
    s.add_argument("--bin_width", type=float, default=0.5,
                   help="Bin width (A) in --bin_mode fixed")
    s.add_argument("--joint_match", nargs="+", default=["dist_cb", "dist"],
                   metavar="COL",
                   help="Columns the jointly-matched non-Cys comparison (analysis F) "
                        "matches on simultaneously; strata are the cross-product of "
                        "the per-axis bins, with bins per axis reduced so each CELL "
                        "keeps ~--min_minority minority pairs. Choose from dist "
                        "(Ca-Ca), dist_cb (virtual Cb-Cb), sep (|i-j|). Default "
                        "'dist_cb dist' isolates the comparison from the Ca-Ca "
                        "difference that matching on Cb alone leaves free")
    s.add_argument("--match_seed", type=int, default=0,
                   help="Seed for sampling the distance-matched non-Cys "
                        "control comparison (analyses D/E)")
    s.add_argument("--n_jobs", type=int, default=-1,
                   help="joblib worker processes for the bootstrap (-1 = all cores)")

    return p.parse_args()


def _main_compute(args: argparse.Namespace) -> None:
    print("[ss_bond_analysis] Loading model …")
    model, device, is_oa = _load_model(args)
    print(f"  mode={args.mode!r}  device={device}  encoder_layers={len(model.encoder_layers)}")

    pdb_files = _collect_pdb_files(args.pdb_dirs)
    print(f"[ss_bond_analysis] {len(pdb_files)} structures")

    print("[ss_bond_analysis] Running forward passes …")
    stats = accumulate(
        model, device, is_oa, pdb_files, args.max_protein_length,
        ctrl_edges_per_protein=args.ctrl_edges_per_protein, ctrl_seed=args.ctrl_seed,
    )

    _print_summary(stats)

    print(f"[ss_bond_analysis] Saving → {args.out_npz}")
    np.savez_compressed(args.out_npz, **stats)
    print(f"[ss_bond_analysis] Done.  Keys saved: {sorted(stats.keys())}")


def _main_analyze(args: argparse.Namespace) -> None:
    cfg = _AnalysisConfig(
        sep_min      = args.sep_min,
        n_boot       = args.n_boot,
        boot_seed    = args.boot_seed,
        support_q    = tuple(args.support_q),
        min_minority = args.min_minority,
        bin_mode     = args.bin_mode,
        bin_width    = args.bin_width,
        match_seed   = args.match_seed,
        joint_match  = tuple(args.joint_match),
        n_jobs       = args.n_jobs,
    )
    print(f"[ss_bond_analysis] Analyzing {args.npz} (layer={args.layer}) …")
    analyze_npz(args.npz, args.layer, cfg, args.fig_out, args.tables_out)


def main() -> None:
    args = parse_args()
    if args.command == "compute":
        _main_compute(args)
    else:
        _main_analyze(args)


if __name__ == "__main__":
    main()
