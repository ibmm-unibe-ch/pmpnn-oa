"""
oa_spectral.py — PCA over the spherical-harmonic spectral coefficients
produced by stats.py.

Reads the .npz file written by stats.py and performs PCA on the per-edge
SH spectral coefficient vectors, independently for each encoder layer.

For each encoder layer the script produces one figure (spectral_pca.svg):
  Left panel  — lower-triangular PC pairs plot coloured by sequential-distance
                bin (1, 2, 3, 4-5, 6+).
                Diagonal cells: stacked histogram per bin.
                Off-diagonal cells: scatter coloured by bin.
  Right panel — stacked bar of squared-loading contributions per SH degree.

Usage
-----
python oa_spectral.py  stats.npz  [options]

Examples
--------
# Minimal — all defaults
python oa_spectral.py  stats.npz

# Custom output directory, exclude local contacts
python oa_spectral.py  stats.npz  --out_dir figs/spectral  --seq_dist_min 6

# Distance-range filter and more PCs
python oa_spectral.py  stats.npz  --d_max 15.0  --n_components 8
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sequential-distance bins: 1, 2, 3, 4-5, 6+
_SEQ_UPPER  = [2, 3, 4, 6, np.inf]   # searchsorted right edges
_SEQ_EDGES  = [1, 2, 3, 4, 6]        # lower bounds (for legend filtering)
_SEQ_LABELS = ["1", "2", "3", "4-5", "6+"]
_N_SEQ_BINS = len(_SEQ_LABELS)
_CMAP_SEQ_DIST = plt.get_cmap("tab10", _N_SEQ_BINS)

# SH degree grouping (supports up to l_max=2; extend lists for higher l_max)
_DEGREE_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
_DEGREE_LABELS = ["$\\ell=0$", "$\\ell=1$", "$\\ell=2$"]


# ---------------------------------------------------------------------------
# Stats loading
# ---------------------------------------------------------------------------

def load_stats(path: str) -> dict:
    """
    Load a .npz file produced by stats.py (mode=oa) and return the arrays
    needed for the spectral PCA.

    ``spectral_{l}`` is a learned, per-layer OA-only quantity with no generic
    counterpart, so it still lives in its own non-self-edge pool.  The
    distance / sequential-separation filters, however, are derived from the
    generic, mode-independent ``nbr_Ca`` / ``seq_dist`` arrays restricted to
    non-self edges via ``~is_self`` — both pools are built by boolean-masking
    the same per-protein [N_v, K] tensor, so the edge order lines up exactly.

    Returns
    -------
    dict with keys
      L           — int
      n_sh        — int, number of SH channels S = (l_max+1)^2
      spectral    — list of L float32 arrays [E_ns, S]
      seq_dist    — float32 [E_ns]
      dist_3d     — float32 [E_ns]  (used for filtering only)
    """
    raw  = np.load(path, allow_pickle=False)
    keys = set(raw.files)

    L    = int(raw["L"])
    n_sh = int(raw["n_sh"]) if "n_sh" in keys else _infer_n_sh(raw, L)

    if "spectral_0" not in keys:
        raise KeyError(
            f"Key 'spectral_0' not found in {path}.  "
            "Make sure the .npz was produced by stats.py in 'oa' mode."
        )
    if "nbr_Ca" not in keys:
        raise KeyError(
            f"Key 'nbr_Ca' not found in {path}.  "
            "This script requires a stats.py output produced after the "
            "local-frame neighbour-coordinate rework — re-run stats.py."
        )

    non_self = ~raw["is_self"].astype(bool)
    dist_3d  = np.linalg.norm(raw["nbr_Ca"][non_self], axis=-1)
    seq_dist = raw["seq_dist"][non_self]

    spectral = [raw[f"spectral_{l}"].astype(np.float32) for l in range(L)]

    return {
        "L":        L,
        "n_sh":     n_sh,
        "spectral": spectral,
        "dist_3d":  dist_3d.astype(np.float32),
        "seq_dist": seq_dist.astype(np.float32),
    }


def _infer_n_sh(raw, L: int) -> int:
    key = "spectral_0"
    if key in set(raw.files):
        return raw[key].shape[1]
    raise KeyError(
        "Cannot determine n_sh: 'n_sh' not in file and 'spectral_0' is missing."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _degree_slices(l_max: int):
    """Return list of (label, color, slice) for each SH degree 0 ... l_max."""
    slices = []
    start  = 0
    for l in range(l_max + 1):
        n = 2 * l + 1
        slices.append((_DEGREE_LABELS[l], _DEGREE_COLORS[l], slice(start, start + n)))
        start += n
    return slices


def _l_max_from_n_sh(n_sh: int) -> int:
    """Infer l_max from S = (l_max+1)^2."""
    l_max = int(round(n_sh ** 0.5)) - 1
    assert (l_max + 1) ** 2 == n_sh, (
        f"n_sh={n_sh} is not a perfect square — cannot infer l_max."
    )
    return l_max


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filters(stats: dict,
                  d_min: float,
                  d_max: float,
                  seq_dist_min: int) -> dict:
    """
    Apply distance and sequential-separation filters.
    Returns a new dict with filtered arrays; the input dict is not mutated.
    """
    L    = stats["L"]
    mask = np.ones(len(stats["dist_3d"]), dtype=bool)

    if d_min > 0.0 or d_max < float("inf"):
        d_mask = (stats["dist_3d"] >= d_min) & (stats["dist_3d"] <= d_max)
        mask  &= d_mask
        print(f"  distance filter [{d_min}, {d_max}] A: "
              f"{d_mask.sum():,} / {len(d_mask):,} edges kept")

    if seq_dist_min > 1:
        sd_mask = stats["seq_dist"] >= seq_dist_min
        mask   &= sd_mask
        print(f"  seq_dist filter [>= {seq_dist_min}]: "
              f"{sd_mask.sum():,} / {len(sd_mask):,} edges kept")

    if not mask.all():
        filtered = dict(stats)
        filtered["dist_3d"]  = stats["dist_3d"][mask]
        filtered["seq_dist"] = stats["seq_dist"][mask]
        filtered["spectral"] = [stats["spectral"][l][mask] for l in range(L)]
        print(f"  total after filters: {int(mask.sum()):,} edges")
        return filtered

    return stats


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(stats: dict, n_components: int) -> list[dict]:
    """
    Fit PCA independently per encoder layer on the SH spectral coefficients.

    Parameters
    ----------
    stats        : dict returned by load_stats (possibly after apply_filters)
    n_components : number of PCA components (capped at n_sh)

    Returns
    -------
    list of L dicts, each with:
      components      : [n_components, S]  eigenvectors (loadings)
      explained_var   : [n_components]     explained variance ratio
      projections     : [total_edges, n_components]
      sq_loadings     : [n_components, S]  squared loadings
    """
    n_comp  = min(n_components, stats["n_sh"])
    results = []

    for l, sp in enumerate(stats["spectral"]):
        sp   = sp - sp.mean(axis=0, keepdims=True)   # explicit demean per SH channel
        pca  = PCA(n_components=n_comp)
        proj = pca.fit_transform(sp)
        results.append({
            "components":    pca.components_,
            "explained_var": pca.explained_variance_ratio_,
            "projections":   proj,
            "sq_loadings":   pca.components_ ** 2,
        })
        ev = pca.explained_variance_ratio_
        print(f"  Layer {l + 1}: top-{n_comp} explained var = "
              f"{ev.sum() * 100:.1f}%  "
              f"(PC1={ev[0] * 100:.1f}%"
              + (f", PC2={ev[1] * 100:.1f}%" if n_comp > 1 else "")
              + ")")

    return results


# ---------------------------------------------------------------------------
# Figure — PC pairs plot + SH-degree stacked bar
# ---------------------------------------------------------------------------

def plot_pca(stats: dict,
             pca_results: list[dict],
             l_max: int,
             out_dir: str,
             n_pairs: int = 3,
             scatter_max_edges: int = 50_000,
             seq_dist_min: int = 1) -> None:
    """
    One figure with L row-blocks, each block containing:
      Left  — lower-triangular pairs plot of the first n_pairs PCs,
              coloured by sequential-distance bin (1, 2, 3, 4-5, 6+).
              Diagonal: stacked histogram per bin.
              Off-diagonal: scatter coloured by bin.
      Right — stacked bar of squared-loading contributions per SH degree.

    Saved as spectral_pca.svg in out_dir.
    """
    import matplotlib.patches as mpatches

    # Sequential-distance bin index and per-edge RGBA colours
    sd             = stats["seq_dist"].astype(float)
    seq_bin_idx    = np.searchsorted(np.array(_SEQ_UPPER), sd, side="right"
                                     ).clip(0, _N_SEQ_BINS - 1)
    seq_bin_colors = np.array([_CMAP_SEQ_DIST(i) for i in seq_bin_idx])  # [E, 4]

    L             = stats["L"]
    degree_slices = _degree_slices(l_max)
    n_comp        = pca_results[0]["components"].shape[0]
    n_pairs       = min(n_pairs, n_comp)

    # Shared subsample index (consistent across all layers and panels)
    n_edges = pca_results[0]["projections"].shape[0]
    if n_edges > scatter_max_edges:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_edges, scatter_max_edges, replace=False)
    else:
        idx = np.arange(n_edges)

    # Legend handles — only show bins whose lower bound >= seq_dist_min
    legend_handles = [
        mpatches.Patch(color=_CMAP_SEQ_DIST(i), label=label)
        for i, (label, lower) in enumerate(zip(_SEQ_LABELS, _SEQ_EDGES))
        if lower >= seq_dist_min
    ]

    col_ratios = [1] * n_pairs + [1.1]
    fig = plt.figure(figsize=(3.0 * (n_pairs + 1.1), 3.0 * n_pairs * L))
    gs  = fig.add_gridspec(
        L * n_pairs, n_pairs + 1,
        hspace=0.08, wspace=0.08,
        width_ratios=col_ratios,
    )

    for l, res in enumerate(pca_results):
        proj = res["projections"]   # [E, n_comp]
        ev   = res["explained_var"] # [n_comp]
        sql  = res["sq_loadings"]   # [n_comp, S]

        row0    = l * n_pairs
        pair_ax = {}

        # ── lower-triangular pairs plot ──────────────────────────────────────
        for r in range(n_pairs):
            for c in range(n_pairs):
                if c > r:
                    continue

                ax = fig.add_subplot(gs[row0 + r, c])
                pair_ax[(r, c)] = ax

                if r == c:
                    # Diagonal: stacked histograms per seq-dist bin
                    for i in range(_N_SEQ_BINS - 1, -1, -1):
                        bin_mask = seq_bin_idx[idx] == i
                        if bin_mask.any():
                            ax.hist(proj[idx[bin_mask], r], bins=60,
                                    color=_CMAP_SEQ_DIST(i),
                                    edgecolor="none", density=False,
                                    alpha=0.6)
                    ax.set_yticks([])
                else:
                    ax.scatter(
                        proj[idx, c], proj[idx, r],
                        c=seq_bin_colors[idx],
                        s=1, alpha=0.3, rasterized=True,
                    )

                if r == n_pairs - 1:
                    ax.set_xlabel(f"PC{c + 1} ({ev[c] * 100:.1f}%)", fontsize=8)
                else:
                    ax.tick_params(labelbottom=False)

                if c == 0 and r != 0:
                    ax.set_ylabel(f"PC{r + 1} ({ev[r] * 100:.1f}%)", fontsize=8)
                else:
                    ax.tick_params(labelleft=False)

                ax.tick_params(labelsize=7)

        pair_ax[(0, 0)].set_title(f"Layer {l + 1}", fontsize=10, loc="left", pad=4)
        pair_ax[(0, 0)].legend(
            handles=legend_handles, fontsize=6, loc="upper right",
            title="Seq. separation", title_fontsize=7,
            framealpha=0.7, markerscale=1.5,
        )

        # ── stacked bar: squared-loading contributions per SH degree ─────────
        ax_bar = fig.add_subplot(gs[row0: row0 + n_pairs, n_pairs])
        x      = np.arange(n_comp)
        bottom = np.zeros(n_comp)

        for label, color, sl in degree_slices:
            contrib = sql[:, sl].sum(axis=1)
            ax_bar.bar(x, contrib, bottom=bottom, color=color,
                       label=label, width=0.6)
            bottom += contrib

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"PC{i + 1}" for i in range(n_comp)], fontsize=7)
        ax_bar.set_ylabel("Sum of squared loadings", fontsize=8)
        ax_bar.set_title(f"Layer {l + 1}  —  SH degree contributions",
                         fontsize=9, pad=4)
        ax_bar.set_ylim(0, 1.15)
        ax_bar.tick_params(labelsize=7)
        ax_bar.legend(fontsize=7, loc="upper right")

        for i, e in enumerate(ev):
            ax_bar.text(i, bottom[i] + 0.02, f"{e * 100:.1f}%",
                        ha="center", va="bottom", fontsize=6)

    path = os.path.join(out_dir, "spectral_pca.svg")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Context / CLI
# ---------------------------------------------------------------------------

@dataclass
class SpectralContext:
    """All configuration for the spectral PCA analysis."""
    stats_npz:         str
    out_dir:           str        = "spectral"
    n_components:      int        = 5
    n_pairs:           int        = 3
    d_min:             float      = 0.0
    d_max:             float      = float("inf")
    seq_dist_min:      int        = 1
    scatter_max_edges: int        = 50_000
    save_npz:          str | None = None

    def __post_init__(self):
        if not os.path.isfile(self.stats_npz):
            raise FileNotFoundError(f"stats_npz not found: {self.stats_npz}")
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        if self.n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1, got {self.n_pairs}")
        if self.d_max <= self.d_min:
            raise ValueError(
                f"d_max must be > d_min, got {self.d_min} / {self.d_max}"
            )
        if self.seq_dist_min < 1:
            raise ValueError(
                f"seq_dist_min must be >= 1 (self-edges excluded by stats.py), "
                f"got {self.seq_dist_min}"
            )

    @classmethod
    def from_args(cls) -> "SpectralContext":
        p = argparse.ArgumentParser(
            description=(
                "PCA over spherical-harmonic spectral coefficients from "
                "a stats.py .npz file.  "
                "Scatter plots are coloured by sequential-distance bin "
                "(1, 2, 3, 4-5, 6+)."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        p.add_argument("stats_npz",
                       help=".npz statistics file produced by stats.py (OA mode)")
        p.add_argument("--out_dir", default="spectral",
                       help="Output directory (created if absent)")

        pca_g = p.add_argument_group("PCA")
        pca_g.add_argument("--n_components", type=int, default=5,
                           help="Number of PCA components to compute")
        pca_g.add_argument("--n_pairs", type=int, default=3,
                           help="Number of PCs in the lower-triangular pairs plot")

        vis = p.add_argument_group("Visualisation")
        vis.add_argument("--scatter_max_edges", type=int, default=50_000,
                         help="Max edges in the scatter (random subsample if exceeded)")

        flt = p.add_argument_group("Edge filters")
        flt.add_argument("--d_min", type=float, default=0.0,
                         help="Minimum Ca-Ca distance (A) for edges included in PCA")
        flt.add_argument("--d_max", type=float, default=float("inf"),
                         help="Maximum Ca-Ca distance (A) for edges included in PCA")
        flt.add_argument("--seq_dist_min", type=int, default=1,
                         help="Minimum sequential separation |i-j|")

        p.add_argument("--save_npz", default=None, metavar="PATH",
                       help="If given, save PCA results (projections, loadings, "
                            "explained variance) to this .npz file")

        return cls(**vars(p.parse_args()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = SpectralContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)

    print(f"[oa_spectral] Loading stats from {ctx.stats_npz} ...")
    stats = load_stats(ctx.stats_npz)
    L     = stats["L"]
    n_sh  = stats["n_sh"]
    l_max = _l_max_from_n_sh(n_sh)
    print(f"  {L} encoder layer(s)  |  n_sh={n_sh}  (l_max={l_max})  "
          f"|  {len(stats['dist_3d']):,} non-self edges")

    print("[oa_spectral] Applying filters ...")
    stats = apply_filters(stats, ctx.d_min, ctx.d_max, ctx.seq_dist_min)
    n_edges = len(stats["dist_3d"])
    print(f"  {n_edges:,} edges after filtering")

    if n_edges == 0:
        print("  ERROR: no edges remain after filtering — check filter parameters.")
        return

    print("[oa_spectral] Running PCA ...")
    pca_results = run_pca(stats, ctx.n_components)

    if ctx.save_npz is not None:
        save = {"seq_dist": stats["seq_dist"]}
        for l, res in enumerate(pca_results):
            for key, val in res.items():
                save[f"{key}_l{l}"] = val
        np.savez(ctx.save_npz, **save)
        print(f"  PCA results saved -> {ctx.save_npz}")

    print("[oa_spectral] Generating figure ...")
    plot_pca(
        stats, pca_results, l_max, ctx.out_dir,
        n_pairs           = ctx.n_pairs,
        scatter_max_edges = ctx.scatter_max_edges,
        seq_dist_min      = ctx.seq_dist_min,
    )

    print("[oa_spectral] Done.")


if __name__ == "__main__":
    main()