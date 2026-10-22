"""
vis._bootstrap — shared protein-level cluster bootstrap + grid-construction
primitives, used by both ``vis.__main__`` and ``vis.persistence``.

Everything here is data-agnostic: binning scalars onto a (dist, theta, phi)
grid, removing additive row/column effects by double-centering, and running
a protein-level cluster bootstrap over an arbitrary per-replicate
``compute_fn``. Nothing here knows about edge-type contrasts, residue-pair
specs, or plotting — those stay in ``vis.__main__``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from joblib import delayed


# ---------------------------------------------------------------------------
# Generic binning / grid helpers
# ---------------------------------------------------------------------------

def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in [0, len(edges)-2], or -1 if outside [edges[0], edges[-1]]."""
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == edges[-1]] = len(edges) - 2
    idx[(idx < 0) | (idx > len(edges) - 2)] = -1
    return idx.astype(np.int64)


def flat_index(ia: np.ndarray, ib: np.ndarray, nb: int) -> np.ndarray:
    out = ia * nb + ib
    out[(ia < 0) | (ib < 0)] = -1
    return out


def grid_mean(ids: np.ndarray, v: np.ndarray, shape: tuple, min_count: int) -> np.ndarray:
    n = int(np.prod(shape))
    ok = ids >= 0
    s = np.bincount(ids[ok], weights=v[ok], minlength=n)
    c = np.bincount(ids[ok], minlength=n)
    m = np.full(n, np.nan)
    good = c >= min_count
    m[good] = s[good] / c[good]
    return m.reshape(shape)


def grid_sums(ids: np.ndarray, v: np.ndarray, shape: tuple):
    n = int(np.prod(shape))
    ok = ids >= 0
    s = np.bincount(ids[ok], weights=v[ok], minlength=n).reshape(shape)
    c = np.bincount(ids[ok], minlength=n).reshape(shape)
    return s, c


def prune_sparse_lines(x: np.ndarray, min_cells: int) -> np.ndarray:
    """Set rows/columns with < min_cells valid cells to NaN, until stable."""
    x = x.copy()
    while True:
        valid = ~np.isnan(x)
        bad_r = valid.sum(axis=1) < min_cells
        bad_c = valid.sum(axis=0) < min_cells
        drop = (bad_r[:, None] | bad_c[None, :]) & valid
        if not drop.any():
            return x
        x[drop] = np.nan


def double_center(x: np.ndarray, n_iter: int = 100, tol: float = 1e-9,
                  prune: bool = False, min_cells_line: int = 3) -> np.ndarray:
    """Remove additive row (x-variable) and column (y-variable) effects by iterative
    mean polish; exact with missing (NaN) cells."""
    if prune:
        x = prune_sparse_lines(x, min_cells_line)
    if np.all(np.isnan(x)):
        return x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resid = x - np.nanmean(x)
        for _ in range(n_iter):
            r = np.nan_to_num(np.nanmean(resid, axis=1, keepdims=True))
            resid = resid - r
            c = np.nan_to_num(np.nanmean(resid, axis=0, keepdims=True))
            resid = resid - c
            if max(np.abs(r).max(), np.abs(c).max()) < tol:
                break
    return resid


def build_layout(mask: np.ndarray, protein_ids: np.ndarray, n_proteins: int):
    """CSR layout of masked edge indices grouped by protein. Proteins without masked
    edges keep length 0 (still sampling units for the bootstrap)."""
    sel = np.flatnonzero(mask)
    pid = protein_ids[sel]
    flat_idx = sel[np.argsort(pid, kind="stable")]
    lengths = np.bincount(pid, minlength=n_proteins)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return flat_idx, starts, lengths


def gather(sampled: np.ndarray, flat_idx: np.ndarray, starts: np.ndarray,
          lengths: np.ndarray) -> np.ndarray:
    lens = lengths[sampled]
    total = lens.sum()
    if total == 0:
        return np.empty(0, dtype=np.int64)
    cum = np.cumsum(lens)
    pos = (np.arange(total) - np.repeat(cum - lens, lens)
           + np.repeat(starts[sampled], lens))
    return flat_idx[pos]


def _build_cell_ids(dist: np.ndarray, theta: np.ndarray, phi: np.ndarray, edges: dict):
    bi = {
        "dist": bin_index(dist, edges["dist"]),
        "theta": bin_index(theta, edges["theta"]),
        "phi": bin_index(phi, edges["phi"]),
    }
    panels = {"dist_theta": ("dist", "theta"), "dist_phi": ("dist", "phi"),
             "theta_phi": ("theta", "phi")}
    cell_ids = {name: flat_index(bi[a], bi[b], len(edges[b]) - 1)
               for name, (a, b) in panels.items()}
    cell_ids["dist_1d"] = bi["dist"]
    shapes = {name: (len(edges[a]) - 1, len(edges[b]) - 1) for name, (a, b) in panels.items()}
    shapes["dist_1d"] = (len(edges["dist"]) - 1,)
    return cell_ids, shapes


# ---------------------------------------------------------------------------
# Per-replicate grid construction (plain, picklable top-level functions — kept free
# of closures so joblib's automatic array memmapping applies to the big arrays)
# ---------------------------------------------------------------------------

def _compute_geom_grids(idx, v_layers, cell_ids, shapes, min_count, layers):
    """Per selected layer: raw geometric-mean score grids (3 panels + 1-D distance
    profile), plus a double-centered variant of each 2-D panel (key suffix '_c')."""
    out = {}
    for l in layers:
        vv = v_layers[l][idx]
        layer_out = {}
        for k, ids_full in cell_ids.items():
            g = grid_mean(ids_full[idx], vv, shapes[k], min_count)
            layer_out[k] = g
            if g.ndim == 2:
                layer_out[k + "_c"] = double_center(g)
        out[l] = layer_out
    return out


def _compute_aa_grids(idx, v_layers, type_i, type_j, dist, dist_edges, min_count, layers):
    """Per distance bin, per selected layer: the full 20x20 (attending type, attended
    type) mean log2-score grid, with additive row (attending type) and column
    (attended type) effects removed by iterative double-centering — the generic
    counterpart of the pair-specific edge-type contrast for the full AA alphabet
    rather than a specific residue-pair spec. Grids are returned in the natural
    amino-acid encoding order; any biochemical display order is applied only when
    plotting."""
    n_bins = len(dist_edges) - 1
    db = bin_index(dist[idx], dist_edges)
    ti, tj = type_i[idx], type_j[idx]
    n_aa = 20  # fixed size of the amino-acid alphabet (res_type_i/j encoding)
    out = {}
    for l in layers:
        vv = v_layers[l][idx]
        bin_out = {}
        for b in range(n_bins):
            flat = np.where(db == b, ti * n_aa + tj, -1)
            g = grid_mean(flat, vv, (n_aa, n_aa), min_count)
            bin_out[b] = double_center(g)
        out[l] = bin_out
    return out


# ---------------------------------------------------------------------------
# Statistics loading
# ---------------------------------------------------------------------------

def _load_stats(path: str, layers: list) -> dict:
    """Load a .npz file produced by stats.py and derive the non-self-edge geometry
    pool (local-frame theta/phi/dist) plus the budget-weighted score and, for OA
    checkpoints, the spherical-harmonic spectral coefficients — for ``layers`` (a
    list of 0-indexed encoder-layer ids) only, so an unselected layer's per-edge
    arrays are never read off disk."""
    raw = np.load(path, allow_pickle=False)
    L = int(raw["L"])
    mode = str(raw["mode"])

    required = ("is_self", "nbr_Ca", "seq_dist", "res_type_i", "res_type_j",
               "protein_idx", "res_idx")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"stats file {path!r} is missing required key(s) {missing} "
                         f"— re-run stats.py")
    for l in layers:
        if f"scored_attn_{l}" not in raw:
            raise ValueError(f"stats file {path!r} is missing 'scored_attn_{l}' "
                             f"— re-run stats.py")

    is_self = raw["is_self"].astype(bool)
    non_self = ~is_self

    nbr_Ca = raw["nbr_Ca"][non_self]
    dist = np.linalg.norm(nbr_Ca, axis=-1)
    d_hat = nbr_Ca / np.clip(dist, 1e-8, None)[:, None]
    theta = np.arccos(np.clip(d_hat[:, 0], -1.0, 1.0))
    phi = np.arctan2(d_hat[:, 1], d_hat[:, 2])

    ANGLE_TOL = 1e-5
    if not (theta.min() >= -ANGLE_TOL and theta.max() <= np.pi + ANGLE_TOL):
        raise ValueError(f"theta outside [0, π]: [{theta.min():.4f}, {theta.max():.4f}]")
    if not (phi.min() >= -np.pi - ANGLE_TOL and phi.max() <= np.pi + ANGLE_TOL):
        raise ValueError(f"phi outside [-π, π]: [{phi.min():.4f}, {phi.max():.4f}]")
    theta = np.clip(theta, 0.0, np.pi)
    phi = np.clip(phi, -np.pi, np.pi)

    scored_attn = {}
    for l in layers:
        score = raw[f"scored_attn_{l}"][non_self].astype(np.float64)
        if np.any(~(score > 0)):
            raise ValueError(f"stats file {path!r}, layer {l}: non-positive "
                             f"budget-weighted scores (alpha_ij = 0?)")
        scored_attn[l] = score

    stats = {
        "L": L, "mode": mode,
        "n_proteins": int(raw.get("n_proteins", 0)),
        "dist_3d": dist.astype(np.float32),
        "theta": theta.astype(np.float32),
        "phi": phi.astype(np.float32),
        "seq_dist": raw["seq_dist"][non_self].astype(np.float32),
        "res_type_i": raw["res_type_i"][non_self].astype(np.int64),
        "res_type_j": raw["res_type_j"][non_self].astype(np.int64),
        "protein_idx": raw["protein_idx"][non_self].astype(np.int64),
        "res_idx": raw["res_idx"][non_self].astype(np.int64),
        "scored_attn": scored_attn,
    }

    if mode == "oa" and "spectral_0" in raw:
        n_sh = int(raw["n_sh"]) if "n_sh" in raw else raw["spectral_0"].shape[1]
        for l in layers:
            if f"spectral_{l}" not in raw:
                raise ValueError(f"stats file {path!r} is missing 'spectral_{l}' "
                                 f"— re-run stats.py in 'oa' mode")
        stats["n_sh"] = n_sh
        stats["spectral"] = {l: raw[f"spectral_{l}"].astype(np.float32) for l in layers}

    return stats


# ---------------------------------------------------------------------------
# Protein-level cluster bootstrap — generic over an arbitrary per-replicate
# ``compute_fn(idx_pool, extra) -> dict`` payload
# ---------------------------------------------------------------------------

@dataclass
class BootData:
    """Everything a bootstrap worker needs, bundled into one explicit argument so
    joblib's loky backend can memory-map the large arrays instead of re-pickling
    them per task. ``extra`` is an opaque, caller-defined payload (e.g. a small
    dataclass of the arrays/config a particular ``compute_fn`` needs) — this
    module only ever resamples proteins and hands ``(idx_pool, extra)`` off."""
    flat_idx: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    n_proteins: int
    extra: Any = field(default=None, repr=False)


def _bootstrap_batch(seed_batch, bd: BootData, compute_fn: Callable):
    results, n_none = [], 0
    for ss in seed_batch:
        rng = np.random.default_rng(ss)
        sampled = rng.integers(0, bd.n_proteins, size=bd.n_proteins)
        idx_pool = gather(sampled, bd.flat_idx, bd.starts, bd.lengths)
        if idx_pool.size == 0:
            n_none += 1
            continue
        results.append(compute_fn(idx_pool, bd.extra))
    return results, n_none


def run_bootstrap(parallel, seeds, bd: BootData, batch_size: int, compute_fn: Callable):
    batches = [seeds[k:k + batch_size] for k in range(0, len(seeds), batch_size)]
    all_results, n_none_total = [], 0
    for results, n_none in parallel(delayed(_bootstrap_batch)(b, bd, compute_fn) for b in batches):
        all_results.extend(results)
        n_none_total += n_none
    return all_results, n_none_total


def _stack_tree(items):
    if isinstance(items[0], dict):
        return {k: _stack_tree([it[k] for it in items]) for k in items[0]}
    return np.stack(items, axis=0)


def _summarize_tree(point_tree, boot_tree, n_requested: int, min_coverage: float):
    """Recurse into a (point, bootstrap) pair of matching nested dict/array trees,
    returning at each leaf ``{point, lo, hi, reliable, signif, signif_neg}``:
    ``lo``/``hi`` are the 2.5/97.5 percentiles of the bootstrap distribution,
    ``reliable`` requires the cell to be estimable in at least ``min_coverage`` of
    the ``n_requested`` replicates, ``signif`` flags cells whose CI is entirely
    above baseline (score > 1, i.e. log2-score > 0) and ``signif_neg`` flags cells
    whose CI is entirely below it — the two-sided counterpart used to tell a
    positive bump from a negative dip when comparing conditions."""
    if isinstance(point_tree, dict):
        return {k: _summarize_tree(point_tree[k], boot_tree[k], n_requested, min_coverage)
                for k in point_tree}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(boot_tree, 2.5, axis=0)
        hi = np.nanpercentile(boot_tree, 97.5, axis=0)
    coverage = np.sum(~np.isnan(boot_tree), axis=0) / n_requested
    reliable = coverage >= min_coverage
    signif = reliable & (lo > 0)
    signif_neg = reliable & (hi < 0)
    return dict(point=point_tree, lo=lo, hi=hi, reliable=reliable,
               signif=signif, signif_neg=signif_neg)
