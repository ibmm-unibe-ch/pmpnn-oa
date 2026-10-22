"""
vis.__main__ — geometric and edge-type attention analysis for ProteinMPNN-OA / GAT.

Self-contained CLI replacement for the exploratory analysis in ``enrichment.ipynb``
(formerly split across ``aa_heatmaps.py``, ``enrichment.py``, ``geometry.py`` and
``oa_spectral.py``). Reads a single ``stats.npz`` produced by ``stats.py`` and reports
the budget-weighted attention score as a function of local geometry, using ONE
protein-level cluster bootstrap computed once at the top of the script — every figure
below is built from that same set of bootstrap replicates, so a given replicate index
always corresponds to the same resampling of proteins everywhere it is used.

Two families of results are produced, over the (d, theta, phi) local-frame geometry of
each non-self edge (d = Cα–Cα distance, theta = polar angle, phi = azimuthal angle):

  Geometric results  (``--geom_seq_dist_min``, default 1 — all non-self edges)
    fig_score_<layer>.svg             budget-weighted score, geometric mean per bin
    fig_score_background_<layer>.svg  score relative to the additive (x, y) background
                                       of each 2-D panel (row/column effects removed by
                                       iterative double-centering)
    fig_spectral_*_<layer>.svg        PCA of the spherical-harmonic spectral
                                       coefficients (OA only — GAT has no spectral
                                       channel)

  Edge-type results  (``--edge_seq_dist_min``, default 6 — medium/long range only)
    fig_score_<pair>_<layer>.svg             pair-specific score relative to the
                                              interaction background
                                              (A->B - A->notB - notA->B + notA->notB)
    fig_score_background_<pair>_<layer>.svg  same contrast, additionally centered for
                                              pair-specific (x, y) background within
                                              each 2-D panel
    fig_aa_heatmap_<layer>.svg                full 20x20 (attending type, attended
                                               type) score grid per distance bin,
                                               row/column effects removed by double-
                                               centering (``--aa_n_dist_bins``)

Significance (black dots) is a 95% protein-level bootstrap CI entirely above the
baseline (score = 1, i.e. log2-score > 0); panels with too few edges or too low
bootstrap coverage are shown in grey.

Usage
-----
python -m vis  stats.npz  [options]

Examples
--------
# Minimal — all defaults
python -m vis  stats.npz

# Custom output directory, fewer bootstrap replicates for a quick look
python -m vis  stats.npz  --out_dir figs  --n_bootstraps 500

# Only two residue-pair types, pooled background instead of interaction contrast
python -m vis  stats.npz  --pairs C:C KR:DE  --background pooled

# Skip the spectral PCA (e.g. for a GAT checkpoint, where it is unavailable anyway)
python -m vis  stats.npz  --no_spectral
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from joblib import Parallel

from vis._bootstrap import (
    BootData, _build_cell_ids, _compute_aa_grids, _compute_geom_grids,
    _load_stats, _stack_tree, _summarize_tree, bin_index, build_layout,
    double_center, grid_sums, run_bootstrap,
)


# ---------------------------------------------------------------------------
# Amino-acid alphabet (encoding used in stats res_type_i / res_type_j)
# ---------------------------------------------------------------------------

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Display order for the 20x20 AA-pair heatmap (grouped by biochemical property,
# not the raw encoding order) and the permutation that maps AA_ALPHABET -> it.
AA_ALPHABET_SORTED = list("AVLIMFWYSTNQKRHDECGP")
_AA_PERM = np.array([AA_ALPHABET.index(aa) for aa in AA_ALPHABET_SORTED])


def _parse_aa_set(spec: str) -> np.ndarray:
    """'AILV' -> sorted unique type indices; validates letters."""
    letters = sorted(set(spec.upper()))
    unknown = [c for c in letters if c not in _AA_INDEX]
    if unknown or not letters:
        raise ValueError(f"invalid residue set {spec!r}: unknown letters {unknown}")
    return np.array([_AA_INDEX[c] for c in letters], dtype=np.int64)


def _parse_pairs(specs: list[str]):
    """Parse a list of 'SRC:NBR' specs into (pairs_meta, pairs_info).

    pairs_meta : list of (A, B) index arrays, for computation.
    pairs_info : list of dicts with display labels and a filename-safe tag.
    """
    pairs_meta, pairs_info = [], []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"invalid --pairs entry {spec!r}; expected 'SRC:NBR', "
                             f"e.g. 'AILV:AILV'")
        a_spec, b_spec = spec.split(":", 1)
        A, B = _parse_aa_set(a_spec), _parse_aa_set(b_spec)
        aa_a = "".join(AA_ALPHABET[k] for k in A)
        aa_b = "".join(AA_ALPHABET[k] for k in B)
        lab_a = aa_a if len(A) == 1 else f"[{aa_a}]"
        lab_b = aa_b if len(B) == 1 else f"[{aa_b}]"
        pairs_meta.append((A, B))
        pairs_info.append(dict(aa_a=aa_a, aa_b=aa_b, lab_a=lab_a, lab_b=lab_b,
                               tag=f"{aa_a}-{aa_b}"))
    return pairs_meta, pairs_info


# ---------------------------------------------------------------------------
# Edge-type contrast (main-specific — kept here, not in vis._bootstrap)
# ---------------------------------------------------------------------------

def contrast(s: np.ndarray, c: np.ndarray, background: str, min_count: int) -> np.ndarray:
    """s, c: (..., 4) sums/counts per group [A->B, A->notB, notA->B, notA->notB].
    Returns the log2 contrast per cell."""
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
        m[c < min_count] = np.nan
        if background == "interaction":
            return m[..., 0] - m[..., 1] - m[..., 2] + m[..., 3]
        if background == "pooled":
            cb = c[..., 1] + c[..., 2]
            bg = (s[..., 1] + s[..., 2]) / cb
            bg[cb < min_count] = np.nan
            return m[..., 0] - bg
    raise ValueError(f"unknown background: {background!r}")


def _compute_edge_grids(idx, v_layers, type_i, type_j, cell_ids, shapes, min_count,
                        pairs_meta, background, min_cells_line, layers):
    """Per pair, per selected layer: the A->B contrast grid vs the interaction/pooled
    background (3 panels + 1-D profile), plus a pair-specific double-centered
    variant of each 2-D panel (key suffix '_c')."""
    ti, tj = type_i[idx], type_j[idx]
    out = {}
    for p, (A, B) in enumerate(pairs_meta):
        is_a, is_b = np.isin(ti, A), np.isin(tj, B)
        group = (~is_a).astype(np.int64) * 2 + (~is_b).astype(np.int64)
        layer_out = {}
        for l in layers:
            vv = v_layers[l][idx]
            panel_out = {}
            for k, ids_full in cell_ids.items():
                ids = ids_full[idx]
                gid = np.where(ids >= 0, ids * 4 + group, -1)
                s, c = grid_sums(gid, vv, (*shapes[k], 4))
                g = contrast(s, c, background, min_count)
                panel_out[k] = g
                if g.ndim == 2:
                    panel_out[k + "_c"] = double_center(g, prune=True, min_cells_line=min_cells_line)
            layer_out[l] = panel_out
        out[p] = layer_out
    return out


# ---------------------------------------------------------------------------
# Protein-level cluster bootstrap — run ONCE, shared by every figure below.
# ``MainBootExtra`` + ``_main_bootstrap_compute`` is the ``extra``/``compute_fn``
# pair fed into ``vis._bootstrap.BootData``/``run_bootstrap`` (which only know how
# to resample proteins — this module supplies what to compute per replicate).
# ---------------------------------------------------------------------------

@dataclass
class MainBootExtra:
    """Everything ``_main_bootstrap_compute`` needs for one replicate, bundled into
    one explicit argument so joblib's loky backend can memory-map the large arrays
    instead of re-pickling them per task (this is why these are passed as an
    argument rather than captured in a closure)."""
    seq_dist: np.ndarray
    dist: np.ndarray
    geom_seq_min: int
    geom_cell_ids: dict
    geom_shapes: dict
    geom_min_count: int
    edge_seq_min: int
    edge_cell_ids: dict
    edge_shapes: dict
    edge_min_count: int
    v_layers: list
    type_i: np.ndarray
    type_j: np.ndarray
    pairs_meta: list
    background: str
    edge_min_cells_line: int
    aa_dist_edges: np.ndarray
    aa_min_count: int
    layers: list


def _pool_to_views(idx_pool, extra: MainBootExtra):
    """One protein-resampled edge pool -> the (geometric, edge-type) sub-selections.
    Both are boolean subsets of the SAME idx_pool, so a replicate's resampling of
    proteins is identical across both result families."""
    sd = extra.seq_dist[idx_pool]
    idx_geom = idx_pool if extra.geom_seq_min <= 1 else idx_pool[sd >= extra.geom_seq_min]
    idx_edge = idx_pool[sd >= extra.edge_seq_min]
    return idx_geom, idx_edge


def _main_bootstrap_compute(idx_pool, extra: MainBootExtra) -> dict:
    """One replicate's (geometric, edge-type, AA-heatmap) result tree — used both
    for the point estimate (``idx_pool = flat_idx``) and for every bootstrap
    replicate, so the two code paths can never drift apart."""
    idx_geom, idx_edge = _pool_to_views(idx_pool, extra)
    geom = _compute_geom_grids(idx_geom, extra.v_layers, extra.geom_cell_ids, extra.geom_shapes,
                               extra.geom_min_count, extra.layers)
    edge = _compute_edge_grids(idx_edge, extra.v_layers, extra.type_i, extra.type_j,
                               extra.edge_cell_ids, extra.edge_shapes, extra.edge_min_count,
                               extra.pairs_meta, extra.background, extra.edge_min_cells_line,
                               extra.layers)
    aa = _compute_aa_grids(idx_edge, extra.v_layers, extra.type_i, extra.type_j, extra.dist,
                           extra.aa_dist_edges, extra.aa_min_count, extra.layers)
    return {"geom": geom, "edge": edge, "aa": aa}


# ---------------------------------------------------------------------------
# Plotting — shared 2x2 panel figure (3 geometric panels + 1-D distance profile)
# ---------------------------------------------------------------------------

_PANEL_NAMES = ["dist_theta", "dist_phi", "theta_phi"]
_PANEL_POS = {"dist_theta": (0, 0), "dist_phi": (1, 0), "theta_phi": (1, 1)}
_PANEL_AXES = {"dist_theta": ("dist", "theta"), "dist_phi": ("dist", "phi"),
              "theta_phi": ("theta", "phi")}
_AXIS_LABELS = {
    "dist": r"Distance $d_{\mathrm{3D}}$ (Å)",
    "theta": r"Polar angle $\theta$ (rad)",
    "phi": r"Azimuthal angle $\phi$ (rad)",
}
_PANEL_TITLES = {
    "dist_theta": "Distance vs polar angle",
    "dist_phi": "Distance vs azimuthal angle",
    "theta_phi": "Polar vs azimuthal angle",
}
_ANGLE_TICKS = {
    "theta": ([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
              ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]),
    "phi": ([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]),
}


def _cbar_ticks(clip: float):
    ticks = [clip * x for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    labels = [f"{2.0 ** t:.3g}×" for t in ticks]
    return ticks, labels


def plot_four_panel_figure(
    summary: dict, edges: dict, dist_min: float, dist_max: float, suffix: str,
    cb_label: str, dist_ylabel: str, dist_title: str, suptitle: str, path: str,
    clip: float, dist_tick_step: float = 1.0,
) -> None:
    """2x2 figure: the three geometric 2-D panels (using ``summary[<panel>+suffix]``)
    plus the 1-D distance profile (``summary["dist_1d"]`` — never suffixed, since it is
    never (x, y)-centered in either variant)."""
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")
    norm = mcolors.Normalize(vmin=-clip, vmax=clip)
    ticks, ticklabels = _cbar_ticks(clip)
    dist_range = (dist_min, dist_max)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    for name in _PANEL_NAMES:
        ax = axes[_PANEL_POS[name]]
        s = summary[name + suffix]
        shown = np.where(s["reliable"], np.clip(s["point"], -clip, clip), np.nan)
        a, b = _PANEL_AXES[name]
        xe, ye = edges[a], edges[b]
        mesh = ax.pcolormesh(xe, ye, shown.T, cmap=cmap, norm=norm, shading="flat", rasterized=True)

        ia, ib = np.nonzero(s["signif"])
        xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
        ax.scatter(xc[ia], yc[ib], s=6, c="k", linewidths=0)

        ax.set_title(_PANEL_TITLES[name])
        ax.set_xlabel(_AXIS_LABELS[a])
        ax.set_ylabel(_AXIS_LABELS[b])
        if a == "dist":
            ax.set_xlim(*dist_range)
            ax.set_xticks(np.arange(dist_range[0], dist_range[1] + 1, dist_tick_step))
        if a in _ANGLE_TICKS:
            ax.set_xticks(*_ANGLE_TICKS[a])
        if b in _ANGLE_TICKS:
            ax.set_yticks(*_ANGLE_TICKS[b])

        cb = fig.colorbar(mesh, ax=ax, extend="both")
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklabels)
        cb.set_label(cb_label)

    ax = axes[0, 1]
    s = summary["dist_1d"]
    xc = 0.5 * (edges["dist"][:-1] + edges["dist"][1:])
    y = np.where(s["reliable"], s["point"], np.nan)
    ax.fill_between(xc, np.where(s["reliable"], s["lo"], np.nan),
                    np.where(s["reliable"], s["hi"], np.nan), alpha=0.3, lw=0)
    ax.plot(xc, y, lw=1.5)
    ax.scatter(xc[s["signif"]], y[s["signif"]], s=12, c="k", zorder=3)
    ax.axhline(0, c="k", lw=0.8, ls="--")
    ax.set_xlim(*dist_range)
    ax.set_xticks(np.arange(dist_range[0], dist_range[1] + 1, dist_tick_step))
    ax.set_xlabel(_AXIS_LABELS["dist"])
    ax.set_ylabel(dist_ylabel)
    ax.set_yticks(ticks, ticklabels)
    ax.set_ylim(-clip, clip)
    ax.set_title(dist_title)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_aa_heatmap_figure(
    summary_layer: dict, bin_edges: np.ndarray, bin_counts: list, seq_dist_min: int,
    min_count: int, suptitle_prefix: str, path: str, clip: float,
) -> None:
    """One figure per layer: a row of 20x20 (attending type i, attended type j)
    double-centered score heatmaps, one panel per distance bin — the full-alphabet
    counterpart of the pair-specific edge-type panels above."""
    n_bins = len(bin_edges) - 1
    ncols = min(n_bins, 2)
    nrows = -(-n_bins // ncols)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")
    norm = mcolors.Normalize(vmin=-clip, vmax=clip)
    ticks, ticklabels = _cbar_ticks(clip)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6.0 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    im = None
    for b in range(n_bins):
        ax = axes.flat[b]
        s = summary_layer[b]
        shown = np.where(s["reliable"], np.clip(s["point"], -clip, clip), np.nan)
        shown = shown[np.ix_(_AA_PERM, _AA_PERM)]
        sig = s["signif"][np.ix_(_AA_PERM, _AA_PERM)]

        im = ax.imshow(shown, cmap=cmap, norm=norm, aspect="auto", origin="lower",
                       rasterized=True)
        rows, cols = np.nonzero(sig)
        ax.scatter(cols, rows, s=6, c="k", marker="o", linewidths=0)

        ax.set_xticks(range(len(AA_ALPHABET_SORTED)), AA_ALPHABET_SORTED, fontsize=7)
        ax.set_yticks(range(len(AA_ALPHABET_SORTED)), AA_ALPHABET_SORTED, fontsize=7)
        ax.set_title(f"{bin_edges[b]:.1f}–{bin_edges[b + 1]:.1f} Å  "
                    f"(n = {bin_counts[b]:,})", fontsize=9)
        ax.set_xlabel("Attended residue type (j)", fontsize=8)
        ax.set_ylabel("Attending residue type (i)", fontsize=8)
    for b in range(n_bins, nrows * ncols):
        axes.flat[b].axis("off")

    fig.suptitle(
        f"{suptitle_prefix}pair-specific budget-weighted attention score "
        f"(double-centered log2), |i−j| ≥ {seq_dist_min}\n"
        f"dots: 95% protein-bootstrap CI entirely above 1×; grey: <{min_count} "
        f"edges in a cell",
        fontsize=10,
    )
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax, extend="both")
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.set_label("Score relative to additive row/column background")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Spectral components — PCA over the spherical-harmonic coefficients (OA only)
#
# Not bootstrapped (it is a deterministic PCA of the data, not a per-bin test), so
# this section runs independently of the protein-level bootstrap above.
# ---------------------------------------------------------------------------

_SEQ_CLASSES = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"),
               (6, np.inf, "≥6")]

# (lon, lat) of each backbone/sidechain atom's direction from Cα, under this
# module's (e1, e2, e3) -> (lon, lat) convention: lon = atan2(e2, e1),
# lat = arcsin(e3) (see the sphere-grid code in run_spectral: x/y/z =
# sinΘcosΦ/sinΘsinΦ/cosΘ with (x, y, z) identified with (e1, e2, e3), lon = Φ,
# lat = π/2 - Θ). Since e1 = normalize(Cα - N) points FROM N TOWARD Cα
# (model.py backbone_frame / Ingraham et al.), N itself sits at lon = π, not 0
# — confirmed exactly against a real stats.npz's self-edge nbr_N/nbr_C arrays
# (the source residue's own N/C position in its own local frame): median
# (lon, lat) over 166k residues = (180.0°, 0.0°) for N and (0.0°, -68.8°) for
# C, bond lengths 1.46 / 1.52 Å — matching N and C below to within rounding.
# O and Cβ are cross-checked the same way against geometry.py's idealized ALA
# reference PDB (_write_reference_pdb): O=(1.856,0.001,-1.688) -> (0, -42.3°);
# Cβ=(0.508,-1.245,0.720) -> (-67.8°, 28.2°). Hα has no PDB entry; its value
# is the original generic tetrahedral estimate, sign-corrected the same way
# (lon += π) — it now mirrors Cβ's (e1, e3) in-plane position with an
# opposite-sign e2, exactly as expected for Cα's other out-of-plane
# substituent.
_BACKBONE_ATOMS = {
    "N":  (np.pi - 1e-9,      0.0),
    "Hα": (np.radians(69.8),  np.radians(27.0)),
    "C":  (0.0,               np.radians(-68.8)),
    "O":  (0.0,               np.radians(-42.7)),
    "Cβ": (np.radians(-66.3), np.radians(30.2)),
}
_ATOM_STYLE = {
    "N":  dict(color="steelblue",   marker="o", s=60, zorder=5),
    "Hα": dict(color="black",       marker="o", s=40, zorder=5),
    "C":  dict(color="dimgray",     marker="s", s=60, zorder=5),
    "O":  dict(color="tomato",      marker="o", s=60, zorder=5),
    "Cβ": dict(color="forestgreen", marker="^", s=60, zorder=5),
}


def _l_max_from_n_sh(n_sh: int) -> int:
    l_max = int(round(n_sh ** 0.5)) - 1
    if (l_max + 1) ** 2 != n_sh:
        raise ValueError(f"n_sh={n_sh} is not a perfect square — cannot infer l_max; "
                         f"pass --l_max explicitly")
    return l_max


def _seq_class(sd: np.ndarray) -> np.ndarray:
    out = np.full(sd.shape, -1, dtype=np.int64)
    for k, (lo, hi, _) in enumerate(_SEQ_CLASSES):
        out[(sd >= lo) & (sd <= hi)] = k
    return out


def _standardize_by_degree(x: np.ndarray, idx_map: dict):
    """Center each coefficient; scale each degree block by one scalar (its RMS),
    which preserves the rotational structure within the block."""
    mean = x.mean(axis=0)
    out = x - mean
    scale = {}
    for l, idx in idx_map.items():
        scale[l] = np.sqrt((out[:, idx] ** 2).mean())
        out[:, idx] /= scale[l]
    return out, mean, scale


def _to_raw_delta(d_std: np.ndarray, scale: dict, idx_map: dict) -> np.ndarray:
    d = d_std.copy()
    for l, idx in idx_map.items():
        d[idx] *= scale[l]
    return d


def _add_atoms(ax, with_labels: bool = False) -> None:
    for name, (lon_a, lat_a) in _BACKBONE_ATOMS.items():
        st = _ATOM_STYLE[name]
        ax.scatter(lon_a, lat_a, label=name if with_labels else None, **st)
        ax.annotate(name, xy=(lon_a, lat_a), xytext=(5, 5), textcoords="offset points",
                   fontsize=7, color=st["color"], zorder=6)


def run_spectral(stats: dict, out_dir: str, seq_dist_min: int, l_max: int | None,
                 sh_normalization: str, n_components: int, n_corner: int,
                 n_scatter: int, seed: int, layers: list) -> None:
    if stats["mode"] != "oa":
        print(f"[spectral] skipped — mode={stats['mode']!r}, spherical-harmonic "
             f"spectral coefficients are only recorded in 'oa' mode")
        return
    if "spectral" not in stats:
        print("[spectral] skipped — stats.npz has no 'spectral_*' arrays "
             "(re-run stats.py in 'oa' mode)")
        return

    import torch
    from e3nn.o3 import spherical_harmonics
    from sklearn.decomposition import PCA

    n_sh = stats["n_sh"]
    l_max = _l_max_from_n_sh(n_sh) if l_max is None else l_max
    if (l_max + 1) ** 2 != n_sh:
        raise ValueError(f"--l_max={l_max} is inconsistent with n_sh={n_sh} "
                         f"(expected n_sh=(l_max+1)^2)")
    degrees = [l for l in range(l_max + 1) for _ in range(2 * l + 1)]
    degree_idx = {l: np.flatnonzero(np.array(degrees) == l) for l in range(l_max + 1)}
    degree_colors = plt.get_cmap("tab10", max(l_max + 1, 1))
    degree_labels = {l: rf"$\ell={l}$" for l in range(l_max + 1)}

    # sphere grid, shared across layers (geometry-only, independent of the data)
    n_theta, n_phi = 100, 200
    theta_g = np.linspace(0, np.pi, n_theta)
    phi_g = np.linspace(-np.pi, np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta_g, phi_g)
    xyz_grid = np.stack([np.sin(THETA) * np.cos(PHI), np.sin(THETA) * np.sin(PHI),
                        np.cos(THETA)], axis=-1).reshape(-1, 3)
    y_grid = spherical_harmonics(list(range(l_max + 1)),
                                 torch.tensor(xyz_grid, dtype=torch.float64),
                                 normalize=True, normalization=sh_normalization).numpy()
    lon, lat = PHI, np.pi / 2 - THETA

    def reconstruct(s):
        return (y_grid @ s).reshape(PHI.shape)

    # l=1 harmonics are linear in xyz: Y_l=1(u) = u @ A, exact — fit A once
    Y1 = y_grid[:, degree_idx[1]] if l_max >= 1 else None
    A_l1 = None
    if Y1 is not None:
        A_l1, *_ = np.linalg.lstsq(xyz_grid, Y1, rcond=None)
        resid = np.abs(xyz_grid @ A_l1 - Y1).max()
        if resid >= 1e-6:
            print(f"[spectral] WARNING: l=1 harmonics not linear in xyz as expected "
                 f"(residual {resid:.2e}) — dipole-direction figure may be unreliable")

    seq_dist = stats["seq_dist"]
    mask = seq_dist >= seq_dist_min
    classes = _seq_class(seq_dist[mask])
    n_cls = len(_SEQ_CLASSES)
    seq_cmap = plt.get_cmap("tab10", n_cls)

    for l in layers:
        tag = f"layer{l + 1}"
        c_all = stats["spectral"][l]
        if c_all.shape[1] != n_sh:
            raise ValueError(f"spectral_{l} has {c_all.shape[1]} coefficients, "
                             f"expected n_sh={n_sh}")
        c = c_all[mask].astype(np.float64)

        c_std, c_mean, c_scale = _standardize_by_degree(c, degree_idx)
        n_comp = min(n_components, n_sh)
        pca = PCA(n_components=n_comp).fit(c_std)
        pcs = pca.transform(c_std)
        sq = pca.components_ ** 2
        sq /= sq.sum(axis=1, keepdims=True)
        contrib = {l_: sq[:, idx].sum(axis=1) for l_, idx in degree_idx.items()}

        ev = pca.explained_variance_ratio_
        print(f"  [{tag}] top-{n_comp} explained var = {ev.sum() * 100:.1f}%  "
             f"(PC1={ev[0] * 100:.1f}%" + (f", PC2={ev[1] * 100:.1f}%" if n_comp > 1 else "") + ")")

        # ── mean field (l >= 1) ──────────────────────────────────────────────
        s_mean = c_mean.copy()
        s_mean[degree_idx[0]] = 0.0
        f_mean = reconstruct(s_mean)
        vmax = np.abs(f_mean).max()
        fig, ax = plt.subplots(figsize=(7, 4), subplot_kw={"projection": "mollweide"})
        mesh = ax.pcolormesh(lon, lat, f_mean, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             shading="auto", rasterized=True)
        _add_atoms(ax, with_labels=True)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.6)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_title(f"{tag} — mean field (l ≥ 1)")
        fig.colorbar(mesh, ax=ax, shrink=0.7)
        path = os.path.join(out_dir, f"fig_spectral_mean_field_{tag}.svg")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {path}")

        # ── PC trajectories: field change along each PC, raw coefficient units ──
        pc_indices = tuple(range(min(4, n_comp)))
        sigma_range, n_steps = 2.0, 5
        scores = np.linspace(-sigma_range, sigma_range, n_steps)
        base = {}
        for k in pc_indices:
            d_std = pca.components_[k] * np.sqrt(pca.explained_variance_[k])
            d_raw = _to_raw_delta(d_std, c_scale, degree_idx)
            d_raw[degree_idx[0]] = 0.0
            base[k] = reconstruct(d_raw)
        vmax = sigma_range * max(np.abs(b).max() for b in base.values())
        fig, axes = plt.subplots(len(pc_indices), n_steps,
                                 figsize=(4 * n_steps, 3.2 * len(pc_indices)),
                                 subplot_kw={"projection": "mollweide"}, squeeze=False)
        for row, k in enumerate(pc_indices):
            for col, sc in enumerate(scores):
                ax = axes[row, col]
                mesh = ax.pcolormesh(lon, lat, sc * base[k], cmap="RdBu_r",
                                     vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
                _add_atoms(ax, with_labels=(row == 0 and col == 0))
                ax.set_title(f"PC{k + 1} = {sc:+.1f}σ")
                ax.set_xticklabels([]); ax.set_yticklabels([])
        axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.6, markerscale=0.8)
        fig.colorbar(mesh, ax=axes, shrink=0.6, label="Δ field relative to mean (raw units)")
        fig.suptitle(f"{tag} — PC trajectories", fontsize=10)
        path = os.path.join(out_dir, f"fig_spectral_pc_trajectories_{tag}.svg")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {path}")

        # ── PC corner plot, coloured by sequential-distance class ──────────────
        n_c = min(n_corner, n_comp)
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(pcs), size=min(n_scatter, len(pcs)), replace=False)
        p_sub, cl_sub = pcs[sel], classes[sel]
        norm = mcolors.BoundaryNorm(np.arange(-0.5, n_cls + 0.5), n_cls)
        fig, axes = plt.subplots(n_c, n_c, figsize=(12, 12))
        for i in range(n_c):
            for j in range(n_c):
                ax = axes[i, j]
                if i > j:
                    ax.scatter(p_sub[:, j], p_sub[:, i], c=cl_sub, cmap=seq_cmap, norm=norm,
                              s=1, alpha=0.5, rasterized=True)
                elif i == j:
                    bins = np.linspace(*np.percentile(pcs[:, i], [0.5, 99.5]), 60)
                    for k in range(n_cls):
                        ax.hist(pcs[classes == k, i], bins=bins, density=True,
                               histtype="step", color=seq_cmap(k), lw=1.2)
                    ax.set_yticks([])
                else:
                    ax.axis("off"); continue
                if i == n_c - 1:
                    ax.set_xlabel(f"PC{j + 1}")
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(f"PC{i + 1}")
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        sm = plt.cm.ScalarMappable(cmap=seq_cmap, norm=norm)
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=range(n_cls))
        cbar.set_ticklabels([lab for *_, lab in _SEQ_CLASSES])
        cbar.set_label("Sequence distance |i−j|")
        fig.suptitle(f"{tag} — PC corner plot", fontsize=10)
        fig.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.08, wspace=0.1, hspace=0.1)
        path = os.path.join(out_dir, f"fig_spectral_pc_corner_{tag}.svg")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {path}")

        # ── rotation-invariant power per degree, by sequence-distance class ─────
        P = {l_: (c[:, idx] ** 2).sum(axis=1) for l_, idx in degree_idx.items()}
        total_P = sum(P.values())
        panels = []
        if l_max >= 1:
            panels.append(("P₁ (dipole)", P[1], True))
        if l_max >= 2:
            panels.append(("P₂ (quadrupole)", P[2], True))
        aniso_num = sum(P[l_] for l_ in range(1, l_max + 1)) if l_max >= 1 else np.zeros_like(total_P)
        panels.append(("Anisotropic fraction (l≥1)/total", aniso_num / total_P, False))
        labels = [lab for *_, lab in _SEQ_CLASSES]
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False)
        for ax, (title, vals, logy) in zip(axes[0], panels):
            data = [vals[classes == k] for k in range(n_cls)]
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            if logy:
                ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Sequence distance |i−j|")
        fig.suptitle(f"{tag} — spectral power by degree", fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig_spectral_degree_power_{tag}.svg")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {path}")

        # ── dipole-direction density relative to uniform, by seq-dist class ─────
        if A_l1 is not None:
            d = c[:, degree_idx[1]] @ A_l1.T
            norm_d = np.linalg.norm(d, axis=1)
            ok = norm_d > 0
            u = d[ok] / norm_d[ok, None]
            lon_d = np.arctan2(u[:, 1], u[:, 0])
            lat_d = np.arcsin(np.clip(u[:, 2], -1, 1))
            cl_d = classes[ok]

            n_lon, n_lat, clip_d = 72, 36, 2.0
            lon_e = np.linspace(-np.pi, np.pi, n_lon + 1)
            lat_e = np.arcsin(np.linspace(-1, 1, n_lat + 1))
            LON_E, LAT_E = np.meshgrid(lon_e, lat_e)
            norm = mcolors.Normalize(vmin=-clip_d, vmax=clip_d)
            fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), subplot_kw={"projection": "mollweide"})
            for k, ax in enumerate(axes.flat[:n_cls]):
                sel_k = cl_d == k
                H, _, _ = np.histogram2d(lat_d[sel_k], lon_d[sel_k], bins=[lat_e, lon_e])
                expected = sel_k.sum() / H.size
                with np.errstate(divide="ignore"):
                    ratio = np.log2(H / expected)
                mesh = ax.pcolormesh(LON_E, LAT_E, np.clip(ratio, -clip_d, clip_d),
                                     cmap="RdBu_r", norm=norm, shading="flat", rasterized=True)
                _add_atoms(ax, with_labels=(k == 0))
                ax.set_title(f"|i−j| {_SEQ_CLASSES[k][2]}  (n = {sel_k.sum():,})")
                ax.set_xticklabels([]); ax.set_yticklabels([])
            for ax in axes.flat[n_cls:]:
                ax.axis("off")
            axes.flat[0].legend(loc="lower left", fontsize=7, framealpha=0.6)
            cb = fig.colorbar(mesh, ax=axes, shrink=0.6, extend="both")
            cb.set_ticks([-2, -1, 0, 1, 2])
            cb.set_ticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
            cb.set_label("Dipole-direction density relative to uniform")
            fig.suptitle(f"{tag} — dipole direction density", fontsize=10)
            path = os.path.join(out_dir, f"fig_spectral_dipole_direction_{tag}.svg")
            fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class VisContext:
    """All configuration needed to run the geometric + edge-type attention analysis.

    CLI-facing fields are validated in :meth:`__post_init__`; the class can also be
    instantiated programmatically (e.g. for tests) by passing keyword arguments
    directly.
    """

    # ── I/O ───────────────────────────────────────────────────────────────────
    stats_npz: str
    out_dir: str

    # ── encoder layer selection ─────────────────────────────────────────────────
    layers: list   # 1-indexed layer numbers from the CLI, or None for all layers

    # ── shared distance-range filter ────────────────────────────────────────────
    dist_min: float
    dist_max: float

    # ── geometric results (seq_dist_min default 1) ─────────────────────────────
    geom_seq_dist_min: int
    geom_n_dist: int
    geom_n_theta: int
    geom_n_phi: int
    geom_min_count: int

    # ── edge-type results (seq_dist_min default 6) ──────────────────────────────
    edge_seq_dist_min: int
    edge_n_dist: int
    edge_n_theta: int
    edge_n_phi: int
    edge_min_count: int
    edge_min_cells_line: int
    pairs: list
    background: str

    # ── 20x20 AA-pair heatmap (edge-type result; seq_dist_min default 6) ────────
    aa_heatmap_enabled: bool
    aa_n_dist_bins: int
    aa_min_count: int

    # ── protein-level cluster bootstrap (single pass, shared by every figure) ───
    n_bootstraps: int
    root_seed: int
    min_coverage: float
    clip: float
    n_jobs: int

    # ── spectral components (OA only) ───────────────────────────────────────────
    spectral_enabled: bool
    l_max: int
    sh_normalization: str
    spectral_n_components: int
    spectral_n_corner: int
    spectral_n_scatter: int
    spectral_seed: int

    # ── runtime (populated in __post_init__) ────────────────────────────────────
    stats: dict = field(default=None, repr=False)
    pairs_meta: list = field(default=None, repr=False)
    pairs_info: list = field(default=None, repr=False)
    layers_idx: list = field(default=None, repr=False)   # resolved, 0-indexed
    total_layers: int = field(default=None, repr=False)  # L in the stats file

    # ─────────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        if not os.path.isfile(self.stats_npz):
            raise ValueError(f"stats_npz not found: {self.stats_npz!r}")
        out_parent = os.path.dirname(os.path.abspath(self.out_dir)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(f"Parent directory of out_dir does not exist: {out_parent!r}")

        with np.load(self.stats_npz, allow_pickle=False) as _raw:
            self.total_layers = int(_raw["L"])
        if self.layers is None:
            self.layers_idx = list(range(self.total_layers))
        else:
            bad = [l for l in self.layers if l < 1 or l > self.total_layers]
            if bad:
                raise ValueError(
                    f"--layers must be in [1, {self.total_layers}] (this stats file "
                    f"has {self.total_layers} encoder layer(s)), got out-of-range "
                    f"value(s) {bad}")
            self.layers_idx = sorted({l - 1 for l in self.layers})

        if self.dist_min < 0:
            raise ValueError(f"dist_min must be >= 0, got {self.dist_min}")
        if self.dist_max <= self.dist_min:
            raise ValueError(f"dist_max must be > dist_min, got "
                             f"{self.dist_min} / {self.dist_max}")

        if self.geom_seq_dist_min < 1:
            raise ValueError(f"geom_seq_dist_min must be >= 1, got {self.geom_seq_dist_min}")
        if self.edge_seq_dist_min < 1:
            raise ValueError(f"edge_seq_dist_min must be >= 1, got {self.edge_seq_dist_min}")
        for name, val in [("geom_n_dist", self.geom_n_dist), ("geom_n_theta", self.geom_n_theta),
                          ("geom_n_phi", self.geom_n_phi), ("edge_n_dist", self.edge_n_dist),
                          ("edge_n_theta", self.edge_n_theta), ("edge_n_phi", self.edge_n_phi)]:
            if val < 2:
                raise ValueError(f"{name} must be >= 2, got {val}")
        if self.geom_min_count < 1:
            raise ValueError(f"geom_min_count must be >= 1, got {self.geom_min_count}")
        if self.edge_min_count < 1:
            raise ValueError(f"edge_min_count must be >= 1, got {self.edge_min_count}")
        if self.edge_min_cells_line < 1:
            raise ValueError(f"edge_min_cells_line must be >= 1, got {self.edge_min_cells_line}")
        if self.background not in ("interaction", "pooled"):
            raise ValueError(f"background must be 'interaction' or 'pooled', got "
                             f"{self.background!r}")
        if not self.pairs:
            raise ValueError("at least one --pairs entry is required")
        self.pairs_meta, self.pairs_info = _parse_pairs(self.pairs)

        if self.aa_n_dist_bins < 1:
            raise ValueError(f"aa_n_dist_bins must be >= 1, got {self.aa_n_dist_bins}")
        if self.aa_min_count < 1:
            raise ValueError(f"aa_min_count must be >= 1, got {self.aa_min_count}")

        if self.n_bootstraps < 100:
            raise ValueError(f"n_bootstraps must be >= 100, got {self.n_bootstraps}")
        if not (0.0 < self.min_coverage <= 1.0):
            raise ValueError(f"min_coverage must be in (0, 1], got {self.min_coverage}")
        if self.clip <= 0:
            raise ValueError(f"clip must be > 0, got {self.clip}")

        if self.spectral_n_components < 1:
            raise ValueError(f"spectral_n_components must be >= 1, got "
                             f"{self.spectral_n_components}")
        if self.spectral_n_corner < 1:
            raise ValueError(f"spectral_n_corner must be >= 1, got {self.spectral_n_corner}")
        if self.spectral_n_scatter < 1:
            raise ValueError(f"spectral_n_scatter must be >= 1, got {self.spectral_n_scatter}")
        if self.l_max is not None and self.l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {self.l_max}")

        self.stats = _load_stats(self.stats_npz, layers=self.layers_idx)

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_args(cls) -> "VisContext":
        p = argparse.ArgumentParser(
            description=(
                "Geometric and edge-type budget-weighted attention score analysis for "
                "ProteinMPNN-OA / GAT, with a single protein-level cluster bootstrap "
                "shared across every figure. See the module docstring for the full "
                "list of outputs."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument("stats_npz", help=".npz statistics file produced by stats.py")
        p.add_argument("--out_dir", default="vis_out",
                       help="Output directory (created if absent)")
        p.add_argument("--layers", type=int, nargs="+", default=None, metavar="N",
                       help="1-indexed encoder layer(s) to analyse, e.g. '--layers 2' "
                            "for the middle layer of an oa3/gat3 checkpoint, or "
                            "'--layers 1 3' for the first and last. Default: all "
                            "layers in the stats file.")

        d = p.add_argument_group("Shared distance-range filter")
        d.add_argument("--dist_min", type=float, default=4.0, help="Minimum Cα–Cα distance (Å)")
        d.add_argument("--dist_max", type=float, default=12.0, help="Maximum Cα–Cα distance (Å)")

        g = p.add_argument_group("Geometric results (score map, background, spectral)")
        g.add_argument("--geom_seq_dist_min", type=int, default=1,
                       help="Minimum sequential separation |i−j| for geometric results")
        g.add_argument("--geom_n_dist", type=int, default=32)
        g.add_argument("--geom_n_theta", type=int, default=24)
        g.add_argument("--geom_n_phi", type=int, default=36)
        g.add_argument("--geom_min_count", type=int, default=30,
                       help="Minimum edges per grid cell for it to be shown")

        e = p.add_argument_group("Edge-type results (pair-specific score, background)")
        e.add_argument("--edge_seq_dist_min", type=int, default=6,
                       help="Minimum sequential separation |i−j| for edge-type results")
        e.add_argument("--edge_n_dist", type=int, default=16)
        e.add_argument("--edge_n_theta", type=int, default=12)
        e.add_argument("--edge_n_phi", type=int, default=18)
        e.add_argument("--edge_min_count", type=int, default=20,
                       help="Minimum edges per group (A->B, A->notB, notA->B, notA->notB) "
                            "required in every group used by a cell")
        e.add_argument("--edge_min_cells_line", type=int, default=3,
                       help="Minimum valid cells per row/column before pair-specific "
                            "(x, y) double-centering (background variant only)")
        e.add_argument("--pairs", nargs="+", metavar="SRC:NBR",
                       default=["AILV:AILV", "KR:DE", "DE:KR", "C:C", "ST:FWY"],
                       help="Residue-pair types to analyse, as 'SRC:NBR' (e.g. 'C:C', "
                            "'AILV:AILV' for any-of-AILV -> any-of-AILV)")
        e.add_argument("--background", choices=["interaction", "pooled"], default="interaction",
                       help="interaction: A->B - A->notB - notA->B + notA->notB.  "
                            "pooled: A->B - mean(A->notB U notA->B)")

        aa = p.add_argument_group("20x20 AA-pair heatmap (edge-type result)")
        aa.add_argument("--no_aa_heatmap", action="store_false", dest="aa_heatmap_enabled",
                        help="Skip the full 20x20 (attending, attended) AA-pair heatmap")
        p.set_defaults(aa_heatmap_enabled=True)
        aa.add_argument("--aa_n_dist_bins", type=int, default=4,
                        help="Number of equal-width distance bins between dist_min/dist_max, "
                             "one heatmap panel per bin")
        aa.add_argument("--aa_min_count", type=int, default=10,
                        help="Minimum edges per (attending type, attended type) cell "
                             "for it to be shown")

        b = p.add_argument_group("Protein-level cluster bootstrap (single pass)")
        b.add_argument("--n_bootstraps", type=int, default=5000)
        b.add_argument("--root_seed", type=int, default=12345)
        b.add_argument("--min_coverage", type=float, default=0.95,
                       help="Minimum fraction of replicates in which a cell must be "
                            "estimable for it to be shown")
        b.add_argument("--clip", type=float, default=1.0,
                       help="Display range in log2 units (1.0 -> 0.5x-2x)")
        b.add_argument("--n_jobs", type=int, default=-1, help="joblib worker processes")

        s = p.add_argument_group("Spectral components (OA only)")
        s.add_argument("--no_spectral", action="store_false", dest="spectral_enabled",
                       help="Skip the spectral-PCA section entirely")
        p.set_defaults(spectral_enabled=True)
        s.add_argument("--l_max", type=int, default=None,
                       help="Spherical-harmonic max degree; inferred from n_sh if omitted")
        s.add_argument("--sh_normalization", default="component",
                       help="Must match how stats['spectral'] was computed")
        s.add_argument("--spectral_n_components", type=int, default=5)
        s.add_argument("--spectral_n_corner", type=int, default=4,
                       help="Number of PCs shown in the corner plot")
        s.add_argument("--spectral_n_scatter", type=int, default=50_000,
                       help="Random subsample size for the PC corner scatter")
        s.add_argument("--spectral_seed", type=int, default=0)

        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = VisContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)
    geom_dir = os.path.join(ctx.out_dir, "geometry")
    edge_dir = os.path.join(ctx.out_dir, "edge_type")
    os.makedirs(geom_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    stats = ctx.stats
    mode = stats["mode"]
    layers = ctx.layers_idx
    dist, theta, phi = stats["dist_3d"], stats["theta"], stats["phi"]
    seq_dist = stats["seq_dist"]
    type_i, type_j = stats["res_type_i"], stats["res_type_j"]

    _, protein_ids = np.unique(stats["protein_idx"], return_inverse=True)
    n_proteins = int(protein_ids.max()) + 1
    v_layers = {l: np.log2(s) for l, s in stats["scored_attn"].items()}

    layers_tag = (f"{len(layers)}/{ctx.total_layers}" if len(layers) != ctx.total_layers
                 else str(ctx.total_layers))
    print(f"[vis] mode={mode}  layers={layers_tag} (selected: "
         f"{[l + 1 for l in layers]})  proteins={n_proteins:,}  "
         f"non-self edges={dist.size:,}")

    base_mask = (dist >= ctx.dist_min) & (dist < ctx.dist_max)
    flat_idx, starts, lengths = build_layout(base_mask, protein_ids, n_proteins)
    print(f"[vis] edges in [{ctx.dist_min:g}, {ctx.dist_max:g}) Å: "
         f"{flat_idx.size:,} / {dist.size:,}")
    if flat_idx.size == 0:
        raise RuntimeError("no edges survive --dist_min/--dist_max — nothing to analyse")

    GEOM_EDGES = {
        "dist": np.linspace(ctx.dist_min, ctx.dist_max, ctx.geom_n_dist + 1),
        "theta": np.linspace(0.0, np.pi, ctx.geom_n_theta + 1),
        "phi": np.linspace(-np.pi, np.pi, ctx.geom_n_phi + 1),
    }
    EDGE_EDGES = {
        "dist": np.linspace(ctx.dist_min, ctx.dist_max, ctx.edge_n_dist + 1),
        "theta": np.linspace(0.0, np.pi, ctx.edge_n_theta + 1),
        "phi": np.linspace(-np.pi, np.pi, ctx.edge_n_phi + 1),
    }
    geom_cell_ids, geom_shapes = _build_cell_ids(dist, theta, phi, GEOM_EDGES)
    edge_cell_ids, edge_shapes = _build_cell_ids(dist, theta, phi, EDGE_EDGES)
    AA_DIST_EDGES = np.linspace(ctx.dist_min, ctx.dist_max, ctx.aa_n_dist_bins + 1)

    extra = MainBootExtra(
        seq_dist=seq_dist, dist=dist, geom_seq_min=ctx.geom_seq_dist_min,
        geom_cell_ids=geom_cell_ids, geom_shapes=geom_shapes, geom_min_count=ctx.geom_min_count,
        edge_seq_min=ctx.edge_seq_dist_min, edge_cell_ids=edge_cell_ids,
        edge_shapes=edge_shapes, edge_min_count=ctx.edge_min_count,
        v_layers=v_layers, type_i=type_i, type_j=type_j, pairs_meta=ctx.pairs_meta,
        background=ctx.background, edge_min_cells_line=ctx.edge_min_cells_line,
        aa_dist_edges=AA_DIST_EDGES, aa_min_count=ctx.aa_min_count, layers=layers,
    )
    bd = BootData(flat_idx=flat_idx, starts=starts, lengths=lengths,
                  n_proteins=n_proteins, extra=extra)

    # ── point estimate (unresampled population) ─────────────────────────────────
    point = _main_bootstrap_compute(flat_idx, extra)
    idx_edge_point = _pool_to_views(flat_idx, extra)[1]

    pair_counts = []
    ti_pt, tj_pt = type_i[idx_edge_point], type_j[idx_edge_point]
    for A, B in ctx.pairs_meta:
        is_a, is_b = np.isin(ti_pt, A), np.isin(tj_pt, B)
        group = (~is_a).astype(np.int64) * 2 + (~is_b).astype(np.int64)
        pair_counts.append(np.bincount(group, minlength=4))

    aa_db_point = bin_index(dist[idx_edge_point], AA_DIST_EDGES)
    aa_bin_counts = [int((aa_db_point == b).sum()) for b in range(ctx.aa_n_dist_bins)]

    # ── single protein-level cluster bootstrap, shared by every figure below ────
    print(f"[vis] running {ctx.n_bootstraps:,} protein-level bootstrap replicates "
         f"(root_seed={ctx.root_seed}) ...")
    seeds = np.random.SeedSequence(ctx.root_seed).spawn(ctx.n_bootstraps)
    n_tasks = min(ctx.n_bootstraps, 4 * (os.cpu_count() or 1))
    batch_size = -(-ctx.n_bootstraps // n_tasks)
    with Parallel(n_jobs=ctx.n_jobs, return_as="generator_unordered", max_nbytes="1M") as parallel:
        results, n_none = run_bootstrap(parallel, seeds, bd, batch_size, _main_bootstrap_compute)
    if not results:
        raise RuntimeError("every bootstrap replicate was empty — check --dist_min/"
                           "--dist_max and the seq_dist_min filters")
    if n_none:
        print(f"[vis]   {n_none}/{ctx.n_bootstraps} empty replicates")

    boot = _stack_tree(results)
    summary = _summarize_tree(point, boot, ctx.n_bootstraps, ctx.min_coverage)

    # ── geometric figures ────────────────────────────────────────────────────────
    print("[vis] writing geometric figures ...")
    dist_note = f"{ctx.dist_min:g}–{ctx.dist_max:g} Å"
    for l in layers:
        s = summary["geom"][l]
        plot_four_panel_figure(
            s, GEOM_EDGES, ctx.dist_min, ctx.dist_max, suffix="",
            cb_label="Budget-weighted score", dist_ylabel="Budget-weighted score",
            dist_title="Distance profile (95% protein-bootstrap CI)",
            suptitle=(f"Layer {l + 1} — budget-weighted attention score, geometric mean "
                     f"per bin; |i−j| ≥ {ctx.geom_seq_dist_min}, {dist_note}\n"
                     f"dots: 95% CI entirely above 1× (B={ctx.n_bootstraps:,}); "
                     f"grey: <{ctx.geom_min_count} edges or estimable in "
                     f"<{ctx.min_coverage:.0%} of replicates"),
            path=os.path.join(geom_dir, f"fig_score_layer{l + 1}.svg"), clip=ctx.clip,
        )
        plot_four_panel_figure(
            s, GEOM_EDGES, ctx.dist_min, ctx.dist_max, suffix="_c",
            cb_label="Score relative to additive x/y background",
            dist_ylabel="Budget-weighted score",
            dist_title="Distance profile, uncentered (95% protein-bootstrap CI)",
            suptitle=(f"Layer {l + 1} — budget-weighted attention score relative to "
                     f"additive x/y background (2D panels); geometric mean per bin; "
                     f"|i−j| ≥ {ctx.geom_seq_dist_min}, {dist_note}\n"
                     f"dots: 95% CI entirely above 1× (B={ctx.n_bootstraps:,}); "
                     f"grey: <{ctx.geom_min_count} edges or estimable in "
                     f"<{ctx.min_coverage:.0%} of replicates"),
            path=os.path.join(geom_dir, f"fig_score_background_layer{l + 1}.svg"), clip=ctx.clip,
        )

    # ── edge-type figures ────────────────────────────────────────────────────────
    print("[vis] writing edge-type figures ...")
    BG_TEXT = {
        "interaction": "relative to additive background (A→¬B, ¬A→B, ref. ¬A→¬B)",
        "pooled": "relative to pooled background (A→¬B ∪ ¬A→B)",
    }
    for p, info in enumerate(ctx.pairs_info):
        lab_a, lab_b = info["lab_a"], info["lab_b"]
        npg = pair_counts[p]
        print(f"  {lab_a}→{lab_b}: edges per group [A→B, A→¬B, ¬A→B, ¬A→¬B] = {npg.tolist()}")
        if npg[0] == 0:
            print(f"    no {lab_a}→{lab_b} edges in window — skipping")
            continue
        for l in layers:
            s = summary["edge"][p][l]
            plot_four_panel_figure(
                s, EDGE_EDGES, ctx.dist_min, ctx.dist_max, suffix="",
                cb_label=f"{lab_a}→{lab_b} score relative to background",
                dist_ylabel=f"{lab_a}→{lab_b} score relative to background",
                dist_title="Distance profile (95% protein-bootstrap CI)",
                suptitle=(f"Layer {l + 1} — {lab_a}→{lab_b}: budget-weighted attention "
                         f"score {BG_TEXT[ctx.background]}; |i−j| ≥ {ctx.edge_seq_dist_min}, "
                         f"{dist_note}, n({lab_a}→{lab_b})={npg[0]:,}\n"
                         f"dots: 95% CI entirely above 1× (B={ctx.n_bootstraps:,}); "
                         f"grey: <{ctx.edge_min_count} edges in a group or estimable in "
                         f"<{ctx.min_coverage:.0%} of replicates"),
                path=os.path.join(edge_dir, f"fig_score_{info['tag']}_layer{l + 1}.svg"),
                clip=ctx.clip,
            )
            plot_four_panel_figure(
                s, EDGE_EDGES, ctx.dist_min, ctx.dist_max, suffix="_c",
                cb_label=f"{lab_a}→{lab_b} relative to background\nand pair-specific x/y effects",
                dist_ylabel=f"{lab_a}→{lab_b} score relative to background",
                dist_title="Distance profile, not x/y-centered (95% protein-bootstrap CI)",
                suptitle=(f"Layer {l + 1} — {lab_a}→{lab_b}: budget-weighted attention "
                         f"score {BG_TEXT[ctx.background]}; 2D panels also centered on "
                         f"pair-specific x/y effects; |i−j| ≥ {ctx.edge_seq_dist_min}, "
                         f"{dist_note}, n({lab_a}→{lab_b})={npg[0]:,}\n"
                         f"dots: 95% CI entirely above 1× (B={ctx.n_bootstraps:,}); "
                         f"grey: <{ctx.edge_min_count} edges in a group, "
                         f"<{ctx.edge_min_cells_line} cells per line, or estimable in "
                         f"<{ctx.min_coverage:.0%} of replicates"),
                path=os.path.join(edge_dir, f"fig_score_background_{info['tag']}_layer{l + 1}.svg"),
                clip=ctx.clip,
            )

    # ── 20x20 AA-pair heatmap (edge-type result) ────────────────────────────────
    if ctx.aa_heatmap_enabled:
        print("[vis] writing 20x20 AA-pair heatmap figures ...")
        for l in layers:
            plot_aa_heatmap_figure(
                summary["aa"][l], AA_DIST_EDGES, aa_bin_counts, ctx.edge_seq_dist_min,
                ctx.aa_min_count, suptitle_prefix=f"Layer {l + 1} — ",
                path=os.path.join(edge_dir, f"fig_aa_heatmap_layer{l + 1}.svg"),
                clip=ctx.clip,
            )
    else:
        print("[vis] AA-pair heatmap skipped (--no_aa_heatmap)")

    # ── spectral components (OA only) ───────────────────────────────────────────
    if ctx.spectral_enabled:
        print("[vis] writing spectral-component figures ...")
        run_spectral(
            stats, geom_dir, seq_dist_min=ctx.geom_seq_dist_min, l_max=ctx.l_max,
            sh_normalization=ctx.sh_normalization, n_components=ctx.spectral_n_components,
            n_corner=ctx.spectral_n_corner, n_scatter=ctx.spectral_n_scatter,
            seed=ctx.spectral_seed, layers=layers,
        )
    else:
        print("[vis] spectral section skipped (--no_spectral)")

    print(f"[vis] Done — outputs written to {ctx.out_dir!r}")


if __name__ == "__main__":
    main()
