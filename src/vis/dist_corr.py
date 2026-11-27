"""
vis.dist_corr — budget-weighted attention score vs. backbone atom-pair
distance, stratified by Cα-Cα distance.

Two subcommands, same compute/analyze split as ``vis.ss_bond_analysis``:

  compute   Run the forward passes and write the per-edge .npz described
            below (GPU work — the expensive step). Same input pipeline as
            ``vis.stats`` — an HDF5 structure file plus a text file of
            ``PDBID_CHAIN`` entries (one per line, e.g. data/test_monomers.txt)
            selecting which monomers to process — reusing
            ``vis.stats.StatsContext``/``setup``/``_run_forward`` directly
            rather than re-deriving the model-loading/data-loading
            machinery.
  analyze   Load a .npz written by ``compute`` and report the bootstrapped
            correlation analysis (CPU-only).

Every non-self, valid edge of every successfully-processed structure is
kept (no Cys-only / control-sampling restriction — this is a general
geometry-vs-attention question, not an SSBOND-specific one).

Output .npz keys  (compute)
----------------------------
Scalar
  L            int   — number of encoder layers
  mode         str   — "oa" | "gat"
  n_proteins   int   — structures successfully processed
  n_skipped    int   — structures skipped (parse/forward failure, too long,
                       or no resolved residues)

Per non-self edge  [total_edges]
  protein_idx  int64 — 0-indexed structure id, in processing order (the
                       clustering unit for the protein-level bootstrap)
  seq_dist     |pos_i - pos_j| within the structure
  dist_{a}_{b} pairwise distance (Å) between atom *a* of the attending
                       (source, i) residue and atom *b* of the attended
                       (neighbour, j) residue, for every ordered pair
                       (a, b) in {n, ca, c, o, cb}^2 — 25 fields in total
                       (e.g. ``dist_ca_ca`` is the usual Cα-Cα distance,
                       the stratification variable below; ``dist_n_cb`` is
                       the distance from residue i's N atom to residue j's
                       virtual Cβ atom, etc.). "cb" is the virtual Cβ, see
                       pmpnn.model.virtual_cbeta.

Per non-self edge, per layer  [total_edges]
  scored_attn_{l}  budget-weighted score: rel_α ^ budget (see stats.py's
                   ``accumulate`` / vis._bootstrap._load_stats for the exact
                   definition: ``budget = 1 - alpha_self``, ``rel_alpha =
                   alpha_ij / (budget_i / K_non_self_i)``)

Statistical analysis  (analyze)
--------------------------------
Edges are first restricted to |seq_dist| >= --seq_dist_min (default 6,
matching stats.py's / ss_bond_analysis's medium/long-range convention),
then binned by dist_ca_ca into 4 fixed 2 Å bands: 4-6, 6-8, 8-10, 10-12 Å
(--dist_edges). Within each (layer, Cα-Cα bin), this reports the Pearson
correlation between log2(budget-weighted score) and EVERY one of the 25
dist_{a}_{b} atom-pair distances — i.e. whether residual geometric
variation *within* a coarse Cα-Cα band (where distance and score trivially
co-vary across bands for the usual reason: closer residues get more
attention) still predicts attention, for any pair of backbone atoms, not
just matching ones — plus the Spearman rank correlation (monotonic,
transform-invariant; unaffected by the log2 choice, unlike Pearson). The
log2 transform is used for Pearson because every other analysis in this
project treats ``scored_attn`` as a multiplicative quantity, so log-space
is where a *linear* correlation is the natural question.

Both correlations are reported with a 95% CI from a protein-level cluster
bootstrap (--n_bootstraps, default 5000: resampling whole proteins with
replacement, not individual edges, since edges from the same protein
aren't independent observations), parallelized with joblib (--n_jobs),
reusing vis._bootstrap's generic machinery (BootData/build_layout/
run_bootstrap) the same way vis.__main__/vis.persistence/
vis.seqdist_anisotropy/vis.ss_bond_analysis do.

Usage
-----
python -m vis.dist_corr compute  data/pdb_2021aug02.h5  data/test_monomers.txt \\
    checkpoint.pt  out.npz  --mode oa
python -m vis.dist_corr analyze  out.npz  --out dist_corr.csv

Output  (analyze)
------------------
--out (default dist_corr.csv): one row per (layer, dist_bin, atom_i, atom_j):
  layer (1-indexed), dist_bin ("4-6 A", ...), atom_i, atom_j (each in
  "N"/"Ca"/"C"/"O"/"Cb"), n_edges, pearson_r, pearson_lo, pearson_hi,
  pearson_coverage, pearson_reliable, pearson_signif, pearson_signif_neg,
  spearman_rho, spearman_lo, spearman_hi, spearman_coverage,
  spearman_reliable, spearman_signif, spearman_signif_neg
  ("coverage" is the fraction of bootstrap replicates in which the cell
  was estimable; "reliable" requires coverage >= --min_coverage; "signif"/
  "signif_neg" flag a 95% CI entirely above/below zero, among reliable
  cells only.)

One figure per analyzed layer (--out_dir, default --out's directory):
fig_dist_corr_layer<N>.svg — one row per Cα-Cα bin, one column per
statistic (Pearson / Spearman), each cell a 5x5 (atom_i x atom_j) heatmap.
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel
from scipy import stats as sp_stats

from ._bootstrap import BootData, bin_index, build_layout, run_bootstrap, _stack_tree
from .stats import StatsContext, setup, _run_forward
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
from pmpnn.model import virtual_cbeta

_ATOMS = ("N", "Ca", "C", "O", "Cb")
_ATOM_PAIRS = [(a, b) for a in _ATOMS for b in _ATOMS]   # 25 ordered (atom_i, atom_j) pairs


def _field(a: str, b: str) -> str:
    return f"dist_{a.lower()}_{b.lower()}"


# ---------------------------------------------------------------------------
# compute — forward passes -> per-edge .npz
# ---------------------------------------------------------------------------

def _atom_coords(X: torch.Tensor) -> dict:
    """*X* : [1, N, 4, 3], atom order N/Cα/C/O (see pmpnn.model.backbone_frame).
    Returns {atom name: [N, 3]} for every name in _ATOMS, "Cb" the virtual
    Cβ (pmpnn.model.virtual_cbeta)."""
    return {
        "N":  X[0, :, 0, :],
        "Ca": X[0, :, 1, :],
        "C":  X[0, :, 2, :],
        "O":  X[0, :, 3, :],
        "Cb": virtual_cbeta(X)[0],
    }


def accumulate(ctx: StatsContext, pdb_chains: list) -> dict:
    """Run every entry in *pdb_chains* through *ctx.model* (same HDF5
    loading as vis.stats.accumulate: ``_loader_h5`` + ``_to_pdb_dict``) and
    accumulate, for every non-self valid edge, all 25 ordered
    atom-pair-x-atom-pair distances (_ATOM_PAIRS) between the attending
    (source) and attended (neighbour) residue, plus the per-layer
    budget-weighted score."""
    model  = ctx.model
    device = ctx.device
    is_oa  = ctx.is_oa_mode
    L      = len(model.encoder_layers)

    all_scored_attn = [[] for _ in range(L)]
    all_seq_dist = []
    all_protein_idx = []
    all_dist = {(a, b): [] for (a, b) in _ATOM_PAIRS}

    n_proteins = 0
    n_skipped = 0

    for pdb_chain in pdb_chains:
        print(f"  {pdb_chain} …", end=" ", flush=True)

        raw_entry = _loader_h5([pdb_chain], ctx.h5_file)
        pdb_dict = _to_pdb_dict(raw_entry, ctx.max_protein_length)
        if pdb_dict is None:
            print("skipped (filtered by _to_pdb_dict)")
            n_skipped += 1
            continue

        try:
            X, S, mask, E_idx, mask_attend, attn_weights, _ = _run_forward(
                model, pdb_dict, device, is_oa)
        except Exception as exc:
            print(f"skipped ({exc})")
            n_skipped += 1
            continue

        valid_n = mask[0].bool()                 # [N]
        n_indices = torch.where(valid_n)[0]      # [N_v]
        N_v = int(valid_n.sum())

        E_idx_v = E_idx[0][valid_n]               # [N_v, K]
        mask_attend_v = mask_attend[0][valid_n]   # [N_v, K]
        valid_edges = mask_attend_v.bool()        # [N_v, K]
        is_self_vk = (E_idx_v == n_indices.unsqueeze(-1))
        non_self_valid = valid_edges & ~is_self_vk

        coords = _atom_coords(X)
        atom_i = {a: coords[a][n_indices].unsqueeze(1) for a in _ATOMS}   # [N_v, 1, 3]
        atom_j = {a: coords[a][E_idx_v] for a in _ATOMS}                  # [N_v, K, 3]

        seq_dist_vk = (E_idx_v.long() - n_indices.unsqueeze(-1).long()).abs().float()
        K_non_self = non_self_valid.float().sum(-1).clamp(min=1.0)  # [N_v]

        n_proteins += 1
        n_ns = int(non_self_valid.sum())

        all_seq_dist.append(seq_dist_vk[non_self_valid].cpu().numpy())
        all_protein_idx.append(np.full(n_ns, n_proteins - 1, dtype=np.int64))

        for (a, b) in _ATOM_PAIRS:
            d_vk = (atom_j[b] - atom_i[a]).norm(dim=-1)   # [N_v, K]
            all_dist[(a, b)].append(d_vk[non_self_valid].cpu().numpy())

        for l in range(L):
            alpha_vk   = attn_weights[l][0][valid_n]               # [N_v, K]
            alpha_self = (alpha_vk * is_self_vk.float()).sum(-1)    # [N_v]
            budget     = (1.0 - alpha_self).clamp(min=1e-8)         # [N_v]
            baseline   = budget / K_non_self                        # [N_v]
            rel_alpha  = alpha_vk / baseline.unsqueeze(-1)         # [N_v, K]
            budget_vk  = budget.unsqueeze(-1).expand_as(alpha_vk)  # [N_v, K]
            scored_vk  = rel_alpha ** budget_vk                     # [N_v, K]
            all_scored_attn[l].append(scored_vk[non_self_valid].cpu().numpy())

        print(f"{N_v} residues  {n_ns} non-self edges")

    if n_proteins == 0:
        raise RuntimeError("No structures were successfully processed.")

    out: dict = {
        "L":          np.array(L, dtype=np.int32),
        "mode":       np.array(ctx.mode),
        "n_proteins": np.array(n_proteins, dtype=np.int32),
        "n_skipped":  np.array(n_skipped, dtype=np.int32),
        "protein_idx": np.concatenate(all_protein_idx).astype(np.int64),
        "seq_dist":   np.concatenate(all_seq_dist).astype(np.float32),
    }
    for (a, b) in _ATOM_PAIRS:
        out[_field(a, b)] = np.concatenate(all_dist[(a, b)]).astype(np.float32)
    for l in range(L):
        out[f"scored_attn_{l}"] = np.concatenate(all_scored_attn[l]).astype(np.float32)
    return out


def _print_summary(stats: dict) -> None:
    L = int(stats["L"])
    n_edges = len(stats["seq_dist"])
    print(f"\n{'─' * 60}")
    print(f"  proteins processed : {int(stats['n_proteins'])}  (skipped {int(stats['n_skipped'])})")
    print(f"  non-self edges      : {n_edges:,}")
    print(f"  mean dist_ca_ca={stats['dist_ca_ca'].mean():.2f} A  "
         f"(see all 25 dist_{{a}}_{{b}} fields for the full atom-pair set)")
    for l in range(L):
        print(f"  layer {l + 1}  mean_scored_attn={float(stats[f'scored_attn_{l}'].mean()):.3f}")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# analyze — per-replicate compute, the ``extra``/``compute_fn`` pair fed
# into ``vis._bootstrap.BootData``/``run_bootstrap``
# ---------------------------------------------------------------------------

@dataclass
class CorrExtra:
    """One file's bootstrap replicate recipe, bundled into one explicit
    argument so joblib's loky backend can memory-map the large arrays
    instead of re-pickling them per task (same convention as
    ``vis.seqdist_anisotropy.AnisoExtra`` / ``vis.ss_bond_analysis._BootExtra``).
    ``bin_id`` (the Cα-Cα stratification) and ``atom_dist`` are shared
    across every layer; only ``v_layers`` is keyed per layer."""
    v_layers:  dict             # layer -> log2(scored_attn), full edges
    atom_dist: dict             # (atom_i, atom_j) -> distance array, full edges
    bin_id:    np.ndarray       # Ca-Ca distance-bin id in [0, n_bins), or -1
    min_count: int
    layers:    list
    n_bins:    int


def _corr_bootstrap_compute(idx_pool: np.ndarray, extra: CorrExtra) -> dict:
    """One replicate's {layer: {bin: {(atom_i, atom_j): {"pearson": r,
    "spearman": rho}}}} — used both for the point estimate (idx_pool =
    every valid edge) and for every bootstrap replicate. A bin with fewer
    than ``extra.min_count`` edges in this replicate gets NaN for every
    atom pair rather than a noisy correlation from a handful of points."""
    bid = extra.bin_id[idx_pool]
    pair_vals = {pair: extra.atom_dist[pair][idx_pool] for pair in _ATOM_PAIRS}
    out = {}
    for l in extra.layers:
        v = extra.v_layers[l][idx_pool]
        bin_out = {}
        for b in range(extra.n_bins):
            m = bid == b
            n = int(m.sum())
            pair_out = {}
            if n < extra.min_count:
                for pair in _ATOM_PAIRS:
                    pair_out[pair] = {"pearson": float("nan"), "spearman": float("nan")}
            else:
                vb = v[m]
                vb_degenerate = np.std(vb) == 0
                for pair in _ATOM_PAIRS:
                    db = pair_vals[pair][m]
                    degenerate = vb_degenerate or np.std(db) == 0
                    pearson = (float("nan") if degenerate
                              else float(sp_stats.pearsonr(db, vb).statistic))
                    spearman = (float("nan") if degenerate
                               else float(sp_stats.spearmanr(db, vb).statistic))
                    pair_out[pair] = {"pearson": pearson, "spearman": spearman}
            bin_out[b] = pair_out
        out[l] = bin_out
    return out


def _summarize(point_tree, boot_tree, n_requested: int, min_coverage: float):
    """Recurse into a (point, bootstrap) pair of matching nested dict
    trees down to scalar leaves, returning at each leaf {point, lo, hi,
    coverage, reliable, signif, signif_neg} — lo/hi are the 2.5/97.5
    percentiles of the bootstrap distribution, reliable requires the cell
    to be estimable (non-NaN) in at least min_coverage of n_requested
    replicates, and signif/signif_neg flag a 95% CI entirely above/below
    zero (a correlation significantly different from zero), among
    reliable cells only. A local, scalar-leaf variant of
    ``vis._bootstrap._summarize_tree`` (array leaves) /
    ``vis.seqdist_anisotropy._summarize`` (bounded-fraction leaves, no
    signif) — this statistic's leaves are plain floats, not arrays, and a
    zero baseline IS meaningful here (unlike the anisotropic fraction)."""
    if isinstance(point_tree, dict):
        return {k: _summarize(point_tree[k], boot_tree[k], n_requested, min_coverage)
                for k in point_tree}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = float(np.nanpercentile(boot_tree, 2.5))
        hi = float(np.nanpercentile(boot_tree, 97.5))
    coverage = float(np.sum(~np.isnan(boot_tree)) / n_requested)
    reliable = coverage >= min_coverage
    signif = bool(reliable and lo > 0)
    signif_neg = bool(reliable and hi < 0)
    return dict(point=float(point_tree), lo=lo, hi=hi, coverage=coverage,
               reliable=bool(reliable), signif=signif, signif_neg=signif_neg)


def _run_analysis(npz_path: str, layers: list, cfg: "AnalysisConfig", parallel: Parallel) -> dict:
    """Load *npz_path* (written by ``compute``) and compute the point
    estimate + protein-level cluster-bootstrap summary of the
    score/atom-distance correlation per Cα-Cα bin and atom pair, for every
    layer in *layers*. Returns {"summary": {layer: {bin: {(atom_i,
    atom_j): {...}}}}, "n_edges": (n_bins,) int array, "mode": str}."""
    raw = np.load(npz_path, allow_pickle=False)
    L = int(raw["L"])
    bad = [l for l in layers if l < 0 or l >= L]
    if bad:
        raise ValueError(f"{npz_path!r} has L={L} encoder layers — layer index(es) "
                         f"{bad} out of range")

    seq_dist = raw["seq_dist"].astype(np.float32)
    dist_ca_ca = raw[_field("Ca", "Ca")].astype(np.float32)
    atom_dist = {pair: raw[_field(*pair)].astype(np.float32) for pair in _ATOM_PAIRS}
    protein_idx = raw["protein_idx"].astype(np.int64)

    # A handful of edges can have scored_attn == 0 (alpha_ij underflows to
    # exactly 0.0 in float32 for a sufficiently peaked attention head), which
    # would otherwise silently turn into -inf under log2 and corrupt every
    # correlation that touches it (via np.std/pearsonr -> NaN). Clip to a
    # small positive floor before the log2, same convention as
    # vis.ss_bond_analysis's own log-effect computation (``np.log2(np.clip(sc,
    # 1e-6, None))``).
    v_layers = {}
    for l in layers:
        score = raw[f"scored_attn_{l}"].astype(np.float64)
        n_bad = int(np.sum(~(score > 0)))
        if n_bad:
            print(f"  layer {l + 1}: {n_bad:,} edge(s) with non-positive budget-weighted "
                 f"score (alpha_ij underflowed to 0?) — clipped to 1e-6 before log2")
        v_layers[l] = np.log2(np.clip(score, 1e-6, None))

    bin_id = bin_index(dist_ca_ca, cfg.dist_edges)
    bin_id[seq_dist < cfg.seq_dist_min] = -1
    n_bins = len(cfg.dist_edges) - 1

    _, protein_ids = np.unique(protein_idx, return_inverse=True)
    n_proteins = int(protein_ids.max()) + 1
    base_mask = bin_id >= 0
    flat_idx, starts, lengths = build_layout(base_mask, protein_ids, n_proteins)
    if flat_idx.size == 0:
        raise RuntimeError(
            f"{npz_path!r}: no edges survive --seq_dist_min/--dist_edges")

    extra = CorrExtra(v_layers=v_layers, atom_dist=atom_dist, bin_id=bin_id,
                      min_count=cfg.min_count, layers=layers, n_bins=n_bins)
    bd = BootData(flat_idx=flat_idx, starts=starts, lengths=lengths,
                  n_proteins=n_proteins, extra=extra)

    point = _corr_bootstrap_compute(flat_idx, extra)

    seeds = np.random.SeedSequence(cfg.root_seed).spawn(cfg.n_bootstraps)
    n_tasks = min(cfg.n_bootstraps, 4 * (os.cpu_count() or 1))
    batch_size = -(-cfg.n_bootstraps // n_tasks)
    results, n_none = run_bootstrap(parallel, seeds, bd, batch_size, _corr_bootstrap_compute)
    if not results:
        raise RuntimeError(f"{npz_path!r}: every bootstrap replicate was empty")
    if n_none:
        print(f"  {n_none}/{cfg.n_bootstraps} empty replicates")

    boot = _stack_tree(results)
    summary = _summarize(point, boot, cfg.n_bootstraps, cfg.min_coverage)

    n_edges = np.array([int((bin_id == b).sum()) for b in range(n_bins)])

    print(f"  proteins={n_proteins:,}  edges in window={flat_idx.size:,}  "
         f"usable replicates={len(results):,}/{cfg.n_bootstraps:,}  mode={str(raw['mode'])!r}")

    return dict(summary=summary, n_edges=n_edges, mode=str(raw["mode"]))


# ---------------------------------------------------------------------------
# Report + figure
# ---------------------------------------------------------------------------

def _bin_labels(edges: np.ndarray) -> list:
    return [f"{edges[b]:g}-{edges[b + 1]:g} A" for b in range(len(edges) - 1)]


def _print_report(run: dict, layers_idx: list, labels: list) -> None:
    for l in layers_idx:
        s, ne = run["summary"][l], run["n_edges"]
        print(f"\n  layer {l + 1}  (mode={run['mode']!r})")
        for b, lab in enumerate(labels):
            print(f"    {lab}  n_edges={ne[b]:,}")
            print(f"      {'i':>3} {'j':>3}  {'pearson r':>9}  {'95% CI':>17}  "
                 f"{'spearman rho':>12}  {'95% CI':>17}")
            for (a, bb) in _ATOM_PAIRS:
                p, sp = s[b][(a, bb)]["pearson"], s[b][(a, bb)]["spearman"]
                print(f"      {a:>3} {bb:>3}  {p['point']:9.4f}  "
                     f"[{p['lo']:7.4f}, {p['hi']:7.4f}]  {sp['point']:12.4f}  "
                     f"[{sp['lo']:7.4f}, {sp['hi']:7.4f}]")


def _save_table(run: dict, layers_idx: list, labels: list, out_csv: str) -> str:
    rows = []
    s, ne = run["summary"], run["n_edges"]
    for l in layers_idx:
        for b, lab in enumerate(labels):
            for (a, bb) in _ATOM_PAIRS:
                p, sp = s[l][b][(a, bb)]["pearson"], s[l][b][(a, bb)]["spearman"]
                rows.append(dict(
                    layer=l + 1, dist_bin=lab, atom_i=a, atom_j=bb, n_edges=int(ne[b]),
                    pearson_r=p["point"], pearson_lo=p["lo"], pearson_hi=p["hi"],
                    pearson_coverage=p["coverage"], pearson_reliable=p["reliable"],
                    pearson_signif=p["signif"], pearson_signif_neg=p["signif_neg"],
                    spearman_rho=sp["point"], spearman_lo=sp["lo"], spearman_hi=sp["hi"],
                    spearman_coverage=sp["coverage"], spearman_reliable=sp["reliable"],
                    spearman_signif=sp["signif"], spearman_signif_neg=sp["signif_neg"],
                ))
    df = pd.DataFrame(rows, columns=[
        "layer", "dist_bin", "atom_i", "atom_j", "n_edges",
        "pearson_r", "pearson_lo", "pearson_hi", "pearson_coverage",
        "pearson_reliable", "pearson_signif", "pearson_signif_neg",
        "spearman_rho", "spearman_lo", "spearman_hi", "spearman_coverage",
        "spearman_reliable", "spearman_signif", "spearman_signif_neg",
    ])
    df.to_csv(out_csv, index=False)
    return out_csv


def _plot_layer(run: dict, layer: int, labels: list, path: str) -> None:
    """One row per Cα-Cα bin, one column per statistic (Pearson /
    Spearman); each cell a 5x5 (atom_i x atom_j) heatmap, annotated with
    the point estimate and bolded when the cell is reliable AND
    significantly different from zero."""
    n_bins = len(labels)
    s = run["summary"][layer]
    n_atoms = len(_ATOMS)

    fig, axes = plt.subplots(n_bins, 2, figsize=(7.5, 3.1 * n_bins), squeeze=False)
    im = None
    for row, lab in enumerate(labels):
        for col, key, title in [(0, "pearson", "Pearson r"), (1, "spearman", "Spearman rho")]:
            ax = axes[row, col]
            mat = np.full((n_atoms, n_atoms), np.nan)
            sig = np.zeros((n_atoms, n_atoms), dtype=bool)
            for i, a in enumerate(_ATOMS):
                for j, b in enumerate(_ATOMS):
                    leaf = s[row][(a, b)][key]
                    mat[i, j] = leaf["point"]
                    sig[i, j] = leaf["reliable"] and (leaf["signif"] or leaf["signif_neg"])
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(n_atoms), _ATOMS, fontsize=7)
            ax.set_yticks(range(n_atoms), _ATOMS, fontsize=7)
            for i in range(n_atoms):
                for j in range(n_atoms):
                    val = mat[i, j]
                    txt = "n/a" if np.isnan(val) else f"{val:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                           weight="bold" if sig[i, j] else "normal")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{lab}\nattending atom (i)", fontsize=8)
            if row == n_bins - 1:
                ax.set_xlabel("attended atom (j)", fontsize=8)
    fig.suptitle(f"Layer {layer + 1} — log2(budget-weighted score) vs. atom-pair "
                f"distance correlation, by Ca-Ca bin\n(bold = reliable and "
                f"significantly different from zero)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label="correlation")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# analyze — configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    dist_edges:   np.ndarray
    seq_dist_min: int
    min_count:    int
    n_bootstraps: int
    root_seed:    int
    min_coverage: float
    n_jobs:       int

    def __post_init__(self) -> None:
        edges = np.asarray(sorted(self.dist_edges), dtype=np.float64)
        if len(edges) < 2:
            raise ValueError(f"--dist_edges needs at least 2 values, got {list(edges)}")
        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"--dist_edges must be strictly increasing, got {list(self.dist_edges)}")
        self.dist_edges = edges

        if self.seq_dist_min < 1:
            raise ValueError(f"seq_dist_min must be >= 1, got {self.seq_dist_min}")
        if self.min_count < 3:
            raise ValueError(f"min_count must be >= 3 (need >=3 points for a correlation), "
                             f"got {self.min_count}")
        if self.n_bootstraps < 100:
            raise ValueError(f"n_bootstraps must be >= 100, got {self.n_bootstraps}")
        if not (0.0 < self.min_coverage <= 1.0):
            raise ValueError(f"min_coverage must be in (0, 1], got {self.min_coverage}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "compute", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Run forward passes over every residue pair and save the per-edge .npz.",
    )
    c.add_argument("h5_file",  help="HDF5 structure file (same as vis.stats)")
    c.add_argument("pdb_list", help="Text file with PDBID_CHAIN entries (one per line), "
                               "e.g. data/test_monomers.txt (same as vis.stats)")
    c.add_argument("checkpoint", help="Model checkpoint (.pt)")
    c.add_argument("out_npz",    help="Output .npz file path")

    d = c.add_argument_group("Data")
    d.add_argument("--max_protein_length", type=int, default=DATA_MAX_PROTEIN_LENGTH,
                   help="Maximum protein length in residues; longer structures are skipped")

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
        help="Load a .npz written by 'compute' and report the "
             "score-vs-atom-distance correlation analysis + figures.",
    )
    a.add_argument("npz", help="Input .npz file path, written by 'compute'")
    a.add_argument("--layers", type=int, nargs="+", default=None, metavar="N",
                   help="1-indexed encoder layer(s) to analyze. Default: all layers.")
    a.add_argument("--dist_edges", type=float, nargs="+", default=[4.0, 6.0, 8.0, 10.0, 12.0],
                   help="Ca-Ca distance bin edges in Angstrom (strictly increasing); "
                        "N edges give N-1 bins. Default: 4-6, 6-8, 8-10, 10-12.")
    a.add_argument("--seq_dist_min", type=int, default=6,
                   help="Minimum sequential separation |i-j|; edges below are excluded "
                        "(matches the project's medium/long-range convention).")
    a.add_argument("--min_count", type=int, default=20,
                   help="Minimum edges in a (layer, bin) cell, in a given bootstrap "
                        "replicate, to report a correlation (else NaN)")
    a.add_argument("--n_bootstraps", type=int, default=5000,
                   help="Protein-level cluster-bootstrap replicates")
    a.add_argument("--root_seed", type=int, default=12345,
                   help="Seed for the bootstrap's resampling stream")
    a.add_argument("--min_coverage", type=float, default=0.95,
                   help="Minimum fraction of bootstrap replicates in which a cell must "
                        "be estimable for its CI to be flagged 'reliable'")
    a.add_argument("--n_jobs", type=int, default=-1, help="joblib worker processes")
    a.add_argument("--out", dest="out_csv", default="dist_corr.csv",
                   help="Output CSV path")
    a.add_argument("--out_dir", default=None,
                   help="Output directory for the comparison figures "
                        "(default: --out's directory)")

    return p.parse_args()


def _main_compute(args: argparse.Namespace) -> None:
    ctx = StatsContext(
        h5_file=args.h5_file, pdb_list=args.pdb_list, checkpoint=args.checkpoint,
        out_npz=args.out_npz, max_protein_length=args.max_protein_length,
        mode=args.mode, hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
        num_neighbors=args.num_neighbors, dropout=args.dropout,
        num_radial_orb=args.num_radial_orb, r_max_orb=args.r_max_orb, p_cutoff=args.p_cutoff,
        l_max=args.l_max, weight_mlp_hidden_dim=args.weight_mlp_hidden_dim,
        gate_hidden_dim=args.gate_hidden_dim, torch_device=args.torch_device,
    )

    print("[dist_corr] Loading model …")
    setup(ctx)

    print("[dist_corr] Reading PDB list …")
    with open(ctx.pdb_list) as fh:
        pdb_chains = [ln.strip() for ln in fh if ln.strip()]
    print(f"  {len(pdb_chains)} entries")

    print("[dist_corr] Running forward passes …")
    stats = accumulate(ctx, pdb_chains)

    _print_summary(stats)

    print(f"[dist_corr] Saving → {ctx.out_npz}")
    np.savez_compressed(ctx.out_npz, **stats)
    print(f"[dist_corr] Done.  Keys saved: {sorted(stats.keys())}")


def _main_analyze(args: argparse.Namespace) -> None:
    cfg = AnalysisConfig(
        dist_edges=np.asarray(args.dist_edges), seq_dist_min=args.seq_dist_min,
        min_count=args.min_count, n_bootstraps=args.n_bootstraps,
        root_seed=args.root_seed, min_coverage=args.min_coverage, n_jobs=args.n_jobs,
    )

    out_parent = os.path.dirname(os.path.abspath(args.out_csv)) or "."
    if not os.path.isdir(out_parent):
        raise SystemExit(f"Parent directory of --out does not exist: {out_parent!r}")
    out_dir = args.out_dir or out_parent
    os.makedirs(out_dir, exist_ok=True)

    with np.load(args.npz, allow_pickle=False) as raw:
        L = int(raw["L"])
    if args.layers is None:
        layers_idx = list(range(L))
    else:
        bad = [l for l in args.layers if l < 1 or l > L]
        if bad:
            raise SystemExit(f"--layers must be in [1, {L}], got out-of-range value(s) {bad}")
        layers_idx = sorted({l - 1 for l in args.layers})

    labels = _bin_labels(cfg.dist_edges)
    print(f"[dist_corr] Analyzing {args.npz}  layers={[l + 1 for l in layers_idx]}  "
         f"n_bootstraps={cfg.n_bootstraps:,}  seq_dist_min={cfg.seq_dist_min}  bins={labels}")

    with Parallel(n_jobs=cfg.n_jobs, return_as="generator_unordered", max_nbytes="1M") as parallel:
        run = _run_analysis(args.npz, layers_idx, cfg, parallel)

    _print_report(run, layers_idx, labels)

    print(f"\n  table -> {_save_table(run, layers_idx, labels, args.out_csv)}")
    for l in layers_idx:
        path = os.path.join(out_dir, f"fig_dist_corr_layer{l + 1}.svg")
        _plot_layer(run, l, labels, path)
        print(f"  saved {path}")


def main() -> None:
    args = parse_args()
    if args.command == "compute":
        _main_compute(args)
    else:
        _main_analyze(args)


if __name__ == "__main__":
    main()
