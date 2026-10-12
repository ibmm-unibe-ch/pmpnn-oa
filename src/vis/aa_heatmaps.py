"""
pmpnn-oa-heatmaps.py — Amino-acid heatmap visualisations for ProteinMPNN-OA / GAT.

Reads the .npz statistics file produced by stats.py and generates:

  fig_residue_type_heatmap       — 20×20 AA-pair score heatmap (all distances)
  fig_residue_type_heatmap_by_d3 — 20×20 AA-pair score heatmap per Cα–Cα distance bin

No model weights or forward passes are required.

Usage
-----
python pmpnn-oa-heatmaps.py  stats.npz  [options]

Examples
--------
# Minimal — all defaults
python pmpnn-oa-heatmaps.py  stats.npz

# Custom output directory and distance bins
python pmpnn-oa-heatmaps.py  stats.npz  --out_dir figs/heatmaps  --d3_bins_aa 6

# Exclude local contacts — only show edges with |i−j| >= 6 (medium-/long-range)
python pmpnn-oa-heatmaps.py  stats.npz  --seq_dist_min 6

# Tight score range and no bin-level normalisation
python pmpnn-oa-heatmaps.py  stats.npz  \\
    --aa_score_vmin 0.8  --aa_score_vmax 1.25  \\
    --strat_aa_score_vmin 0.6  --strat_aa_score_vmax 1.6  \\
    --no_normalize_d3_bins
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------

# ProteinMPNN alphabet — index 0-19
_AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class HeatmapContext:
    """All configuration needed to generate the AA heatmaps.

    CLI-facing fields are validated in :meth:`__post_init__`; this makes it
    easy to instantiate the class programmatically in tests or notebooks by
    passing keyword arguments directly instead of going through the CLI.
    """

    # ── I/O ───────────────────────────────────────────────────────────────────
    stats_npz: str
    out_dir:   str

    # ── global heatmap — score colorscale ─────────────────────────────────────
    aa_score_vmin: float        # lower bound in score units
    aa_score_vmax: float        # upper bound in score units

    # ── global heatmap — edge-density colorscale ──────────────────────────────
    aa_ec_min: float            # lower bound (mean edges per residue)
    aa_ec_max: float            # upper bound

    # ── binned heatmap — distance range ───────────────────────────────────────
    d3_min:      float          # Cα–Cα distance range start (Å)
    d3_max:      float          # Cα–Cα distance range end   (Å)
    d3_bins_aa:  int            # number of equally-spaced distance bins

    # ── binned heatmap — score colorscale ─────────────────────────────────────
    strat_aa_score_vmin: float
    strat_aa_score_vmax: float

    # ── binned heatmap — normalisation ────────────────────────────────────────
    normalize_d3_bins: bool     # divide each bin by its nanmean

    # ── seq-vs-3d heatmap ────────────────────────────────────────────────────────
    seq_vs_3d_score_vmin:   float  # score colorscale lower bound (score units)
    seq_vs_3d_score_vmax:   float  # score colorscale upper bound (score units)
    seq_vs_3d_ec_min:       float  # edge-count colorscale lower bound
    seq_vs_3d_ec_max:       float  # edge-count colorscale upper bound

    # ── sequence-distance filter ───────────────────────────────────────────────
    seq_dist_min: int           # minimum |i − j|; mirrors spectral analysis default of 1

    # ── runtime (populated in __post_init__) ──────────────────────────────────
    stats: dict = field(default=None, repr=False)

    # ─────────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # --- path validation --------------------------------------------------
        if not os.path.isfile(self.stats_npz):
            raise ValueError(f"stats_npz not found: {self.stats_npz!r}")
        if self.stats_npz.split(".")[-1] not in ("npz",):
            # soft warning — numpy will raise on load if the file is bad
            print(f"[warn] stats_npz does not have a .npz extension: "
                  f"{self.stats_npz!r}")

        out_parent = os.path.dirname(os.path.abspath(self.out_dir)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(
                f"Parent directory of out_dir does not exist: {out_parent!r}")

        # --- numeric validation -----------------------------------------------
        if self.aa_score_vmin <= 0:
            raise ValueError(
                f"aa_score_vmin must be > 0 (score units), "
                f"got {self.aa_score_vmin}")
        if self.aa_score_vmax <= self.aa_score_vmin:
            raise ValueError(
                f"aa_score_vmax must be > aa_score_vmin, "
                f"got vmin={self.aa_score_vmin}  vmax={self.aa_score_vmax}")

        if self.aa_ec_min < 0:
            raise ValueError(
                f"aa_ec_min must be >= 0, got {self.aa_ec_min}")
        if self.aa_ec_max <= self.aa_ec_min:
            raise ValueError(
                f"aa_ec_max must be > aa_ec_min, "
                f"got {self.aa_ec_min} / {self.aa_ec_max}")

        if self.d3_min < 0:
            raise ValueError(f"d3_min must be >= 0, got {self.d3_min}")
        if self.d3_max <= self.d3_min:
            raise ValueError(
                f"d3_max must be > d3_min, "
                f"got d3_min={self.d3_min}  d3_max={self.d3_max}")
        if self.d3_bins_aa < 1:
            raise ValueError(
                f"d3_bins_aa must be >= 1, got {self.d3_bins_aa}")

        if self.strat_aa_score_vmin <= 0:
            raise ValueError(
                f"strat_aa_score_vmin must be > 0, "
                f"got {self.strat_aa_score_vmin}")
        if self.strat_aa_score_vmax <= self.strat_aa_score_vmin:
            raise ValueError(
                f"strat_aa_score_vmax must be > strat_aa_score_vmin, "
                f"got {self.strat_aa_score_vmin} / {self.strat_aa_score_vmax}")

        if self.seq_vs_3d_score_vmin <= 0:
            raise ValueError(
                f"seq_vs_3d_score_vmin must be > 0, got {self.seq_vs_3d_score_vmin}")
        if self.seq_vs_3d_score_vmax <= self.seq_vs_3d_score_vmin:
            raise ValueError(
                f"seq_vs_3d_score_vmax must be > seq_vs_3d_score_vmin, "
                f"got {self.seq_vs_3d_score_vmin} / {self.seq_vs_3d_score_vmax}")
        if self.seq_vs_3d_ec_min < 0:
            raise ValueError(
                f"seq_vs_3d_ec_min must be >= 0, got {self.seq_vs_3d_ec_min}")
        if self.seq_vs_3d_ec_max <= self.seq_vs_3d_ec_min:
            raise ValueError(
                f"seq_vs_3d_ec_max must be > seq_vs_3d_ec_min, "
                f"got {self.seq_vs_3d_ec_min} / {self.seq_vs_3d_ec_max}")

        if self.seq_dist_min < 1:
            raise ValueError(
                f"seq_dist_min must be >= 1 (self-edges are always excluded), "
                f"got {self.seq_dist_min}")

        # --- load statistics --------------------------------------------------
        self.stats = _load_stats(self.stats_npz)

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_args(cls) -> "HeatmapContext":
        """Parse sys.argv and return a validated :class:`HeatmapContext`."""
        p = argparse.ArgumentParser(
            description=(
                "Generate amino-acid pair heatmaps from ProteinMPNN-OA / GAT "
                "statistics produced by stats.py"
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # ── positional ────────────────────────────────────────────────────────
        p.add_argument(
            "stats_npz",
            help=".npz statistics file produced by stats.py",
        )

        # ── output ────────────────────────────────────────────────────────────
        p.add_argument(
            "--out_dir", default="heatmaps",
            help="Output directory for figure files (created if absent)",
        )

        # ── global heatmap — score colorscale ─────────────────────────────────
        g = p.add_argument_group("Global heatmap — score colorscale")
        g.add_argument(
            "--aa_score_vmin", type=float, default=0.769,
            help="Lower bound for the AA-pair score colorscale (score units)",
        )
        g.add_argument(
            "--aa_score_vmax", type=float, default=1.3,
            help="Upper bound for the AA-pair score colorscale (score units)",
        )

        # ── global heatmap — edge-density colorscale ──────────────────────────
        ed = p.add_argument_group("Global heatmap — edge-density colorscale")
        ed.add_argument(
            "--aa_ec_min", type=float, default=0.0,
            help="Lower bound for the edge-density colorscale "
                 "(mean non-self edges per residue)",
        )
        ed.add_argument(
            "--aa_ec_max", type=float, default=3.0,
            help="Upper bound for the edge-density colorscale",
        )

        # ── binned heatmap ─────────────────────────────────────────────────────
        b = p.add_argument_group("Binned heatmap — distance range and bins")
        b.add_argument(
            "--d3_min", type=float, default=2.0,
            help="Minimum Cα–Cα distance (Å) for the distance-binned heatmap",
        )
        b.add_argument(
            "--d3_max", type=float, default=12.0,
            help="Maximum Cα–Cα distance (Å) for the distance-binned heatmap",
        )
        b.add_argument(
            "--d3_bins_aa", type=int, default=5,
            help="Number of equally-spaced Cα–Cα distance bins",
        )

        bs = p.add_argument_group("Binned heatmap — score colorscale")
        bs.add_argument(
            "--strat_aa_score_vmin", type=float, default=0.5,
            help="Lower colorscale bound for the per-bin AA heatmap (score "
                 "units, or ratio to bin mean when --normalize_d3_bins is set)",
        )
        bs.add_argument(
            "--strat_aa_score_vmax", type=float, default=2.0,
            help="Upper colorscale bound for the per-bin AA heatmap",
        )

        # ── normalisation ─────────────────────────────────────────────────────
        n = p.add_argument_group("Binned heatmap — normalisation")
        n.add_argument(
            "--no_normalize_d3_bins", action="store_false",
            dest="normalize_d3_bins",
            help="Disable per-bin normalisation (show raw scores instead of "
                 "ratio to each bin's mean score)",
        )
        p.set_defaults(normalize_d3_bins=True)

        # ── seq-vs-3d heatmap ─────────────────────────────────────────────────
        sv = p.add_argument_group("Seq-vs-3D heatmap")
        sv.add_argument(
            "--seq_vs_3d_score_vmin", type=float, default=0.5,
            help="Lower bound for the seq-vs-3D score colorscale (score units).",
        )
        sv.add_argument(
            "--seq_vs_3d_score_vmax", type=float, default=2.0,
            help="Upper bound for the seq-vs-3D score colorscale (score units).",
        )
        sv.add_argument(
            "--seq_vs_3d_ec_min", type=float, default=0.0,
            help="Lower bound for the seq-vs-3D edge-count colorscale "
                 "(mean edges per residue).",
        )
        sv.add_argument(
            "--seq_vs_3d_ec_max", type=float, default=10.0,
            help="Upper bound for the seq-vs-3D edge-count colorscale "
                 "(mean edges per residue).",
        )

        # ── sequence-distance filter ──────────────────────────────────────────
        sd = p.add_argument_group("Sequence-distance filter")
        sd.add_argument(
            "--seq_dist_min", type=int, default=1,
            help="Minimum sequential separation |i − j| for edges included in "
                 "the heatmaps.  Default 1 keeps all non-self edges.  "
                 "Set to e.g. 6 to exclude local backbone contacts and focus "
                 "on medium- and long-range interactions "
                 "(mirrors --seq_dist_min in pmpnn-oa-spectral-analysis.py).",
        )

        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# Statistics loading  (mirrors load_stats from pmpnn-oa-analysis-pairs.py)
# ---------------------------------------------------------------------------

def _load_stats(path: str) -> dict:
    """
    Load a .npz file produced by stats.py and reconstruct the list-indexed
    per-layer arrays that the plotting functions expect.

    Returns
    -------
    dict with keys:
      ``L``            — int, number of encoder layers
      ``is_self``      — bool array [total_edges]
      ``dist_3d``      — float32 [total_edges]
      ``seq_dist``     — float32 [total_edges]
      ``res_idx``      — int64 [total_edges]
      ``res_type_i``   — int8 [total_edges]
      ``res_type_j``   — int8 [total_edges]
      ``scored_attn``  — list of L float32 arrays [total_edges]
      (and all other flat keys from the .npz)
    """
    raw = np.load(path, allow_pickle=False)
    L   = int(raw["L"])

    stats: dict = dict(raw)

    # Rebuild the per-layer list arrays
    for key in (
        "neff", "alpha_self", "argmax_seq_dist", "argmax_dist_3d",
        "attn", "rel_attn", "budget", "scored_attn", "mean_score",
    ):
        layer_arrays = []
        for l in range(L):
            tag = f"{key}_{l}"
            if tag in raw:
                layer_arrays.append(raw[tag])
        if layer_arrays:
            stats[key] = layer_arrays

    stats["L"]       = L
    stats["is_self"] = raw["is_self"].astype(bool)
    return stats


# ---------------------------------------------------------------------------
# Shared colorscale helper
# ---------------------------------------------------------------------------

def _log_norm_and_ticks(
    log_grids: list[np.ndarray],
    vmin: float | None,
    vmax: float | None,
) -> tuple[mcolors.TwoSlopeNorm, np.ndarray, np.ndarray]:
    """
    Build a log-space TwoSlopeNorm centred at score = 1 (log = 0).

    Parameters
    ----------
    log_grids : sequence of np.ndarray
        Already log-transformed grid(s).  NaN values are ignored.
    vmin, vmax : float or None
        Colorscale bounds in **score** units.  None → data-driven.

    Returns
    -------
    norm       : TwoSlopeNorm
    raw_ticks  : tick positions in log space
    tick_scores: tick labels in score space
    """
    all_log  = np.concatenate([g[~np.isnan(g)] for g in log_grids])
    log_vmin = np.log(max(vmin, 1e-9)) if vmin is not None else float(all_log.min())
    log_vmax = np.log(vmax)            if vmax is not None else float(all_log.max())
    # Guarantee that the neutral point (0 = log(1)) lies strictly inside the range
    log_vmin = min(log_vmin, -1e-6)
    log_vmax = max(log_vmax,  1e-6)

    norm        = mcolors.TwoSlopeNorm(vmin=log_vmin, vcenter=0.0, vmax=log_vmax)
    raw_ticks   = np.linspace(log_vmin, log_vmax, 7)
    tick_scores = np.exp(raw_ticks)
    return norm, raw_ticks, tick_scores


def _add_score_colorbar(fig, im, axes_list, raw_ticks, tick_scores,
                        label="Score  (1 = uniform baseline)",
                        shrink=0.6) -> None:
    """Attach a shared score colorbar with a dashed line at score = 1."""
    cb = fig.colorbar(im, ax=axes_list, shrink=shrink,
                      label=label, ticks=raw_ticks)
    cb.ax.set_yticklabels([f"{s:.2g}" for s in tick_scores])
    cb.ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")


# ---------------------------------------------------------------------------
# Figure 1 — global residue-type heatmap
# ---------------------------------------------------------------------------

def fig_residue_type_heatmap(
    stats: dict,
    out_dir: str,
    vmin: float | None = None,
    vmax: float | None = None,
    ec_min: float = 0.0,
    ec_max: float = 5.0,
    seq_dist_min: int = 1,
) -> None:
    """
    (L + 1)-panel figure.

    Left L panels  — mean budget-weighted score for non-self edges grouped by
                     (source AA, neighbour AA), one panel per encoder layer.
                     A one-row marginal strip above each panel shows the mean
                     score averaged over all source types.  Colorscale is
                     log-transformed with TwoSlopeNorm centred at score = 1.
    Right panel    — mean number of non-self edges per source residue for each
                     (type_i, type_j) pair (Blues palette, layer-independent).

    Parameters
    ----------
    stats  : dict produced by :func:`_load_stats`.
    out_dir: output directory.
    vmin, vmax : colorscale bounds in score units; ``None`` → data-driven.
    ec_min, ec_max : edge-density colorscale bounds.
    seq_dist_min : int
        Minimum sequential separation |i − j|.  Edges with ``seq_dist <
        seq_dist_min`` are excluded from both the score and density grids.
        Default 1 keeps all non-self edges; set to e.g. 6 to restrict to
        medium- and long-range contacts only.
    """
    non_self = (~stats["is_self"]) & (stats["seq_dist"] >= seq_dist_min)
    ti       = stats["res_type_i"][non_self].astype(int)
    tj       = stats["res_type_j"][non_self].astype(int)
    res_ns   = stats["res_idx"][non_self]
    L        = stats["L"]
    N_AA     = len(_AA_ALPHABET)

    # ── build score grids [L × N_AA × N_AA] ──────────────────────────────────
    grids = []
    for l in range(L):
        scored_l = stats["scored_attn"][l][non_self]
        grid = np.full((N_AA, N_AA), np.nan)
        for i in range(N_AA):
            for j in range(N_AA):
                m = (ti == i) & (tj == j)
                if m.sum() > 0:
                    grid[i, j] = float(scored_l[m].mean())
        grids.append(grid)

    # ── marginal strips: mean score per neighbour type j (averaged over all i)
    marginals = []
    for l in range(L):
        scored_l = stats["scored_attn"][l][non_self]
        row = np.full((1, N_AA), np.nan)
        for j in range(N_AA):
            m = tj == j
            if m.sum() > 0:
                row[0, j] = float(scored_l[m].mean())
        marginals.append(row)

    # ── edge-density grid: mean non-self edges per source residue of type i ──
    count_grid = np.full((N_AA, N_AA), np.nan)
    for i in range(N_AA):
        n_res_i = len(np.unique(res_ns[ti == i]))
        if n_res_i == 0:
            continue
        for j in range(N_AA):
            m = (ti == i) & (tj == j)
            if m.sum() > 0:
                count_grid[i, j] = m.sum() / n_res_i

    # ── log-transform (same convention as fig5 in pmpnn-oa-analysis.py) ───────
    log_grids    = [np.where(np.isnan(g), np.nan,
                             np.log(np.clip(g, 1e-9, None))) for g in grids]
    log_marginals = [np.where(np.isnan(m), np.nan,
                              np.log(np.clip(m, 1e-9, None))) for m in marginals]

    norm, raw_ticks, tick_scores = _log_norm_and_ticks(log_grids, vmin, vmax)

    kw_main  = dict(aspect="equal", origin="upper",
                    extent=[-0.5, N_AA - 0.5, N_AA - 0.5, -0.5])
    kw_strip = dict(aspect="auto",  origin="upper",
                    extent=[-0.5, N_AA - 0.5, 0.5, -0.5])

    # ── layout: thin strip on top of each score panel ─────────────────────────
    fig = plt.figure(figsize=(4 * (L + 1), 5.5))
    gs  = gridspec.GridSpec(
        2, L + 1, height_ratios=[1, 9],
        hspace=0.04, wspace=0.35, figure=fig,
    )

    strip_axes = [fig.add_subplot(gs[0, l]) for l in range(L)]
    main_axes  = [fig.add_subplot(gs[1, l], sharex=strip_axes[l]) for l in range(L)]
    ax_c       = fig.add_subplot(gs[:, L])   # density panel spans both rows

    im = None  # will be set in the loop — needed for the shared colorbar
    for l in range(L):
        # strip — marginal mean over source types
        ax_s = strip_axes[l]
        ax_s.imshow(log_marginals[l], cmap="RdBu_r", norm=norm, **kw_strip)
        ax_s.set_yticks([0])
        ax_s.set_yticklabels(["mean\n(all i)"], fontsize=6)
        ax_s.tick_params(axis="x", labelbottom=False)
        ax_s.set_title(f"Layer {l + 1}", pad=14)

        # main 20×20 heatmap
        ax  = main_axes[l]
        im  = ax.imshow(log_grids[l], cmap="RdBu_r", norm=norm, **kw_main)
        ax.set_xticks(range(N_AA))
        ax.set_xticklabels(_AA_ALPHABET, fontsize=7)
        ax.set_yticks(range(N_AA))
        ax.set_yticklabels(_AA_ALPHABET, fontsize=7)
        ax.set_xlabel("Neighbour AA type (j)")
        if l == 0:
            ax.set_ylabel("Source AA type (i)")

    _add_score_colorbar(
        fig, im, main_axes + strip_axes, raw_ticks, tick_scores,
        label="Score  (1 = uniform baseline)", shrink=0.6,
    )

    # ── density panel ─────────────────────────────────────────────────────────
    im_c = ax_c.imshow(count_grid, cmap="Blues",
                       vmin=ec_min, vmax=ec_max, **kw_main)
    ax_c.set_xticks(range(N_AA))
    ax_c.set_xticklabels(_AA_ALPHABET, fontsize=7)
    ax_c.set_yticks(range(N_AA))
    ax_c.set_yticklabels(_AA_ALPHABET, fontsize=7)
    ax_c.set_xlabel("Neighbour AA type (j)")
    ax_c.set_title("Mean edges\nper residue")
    fig.colorbar(im_c, ax=ax_c, shrink=0.6, label="Mean edges per residue")

    fig.suptitle(
        r"Mean budget-weighted score by residue-type pair  "
        r"score$_{ij}$ = $\mathrm{rel}\text{-}\alpha_{ij}^{\,\mathrm{budget}_i}$"
        r"  ·  colorscale: $\log(\mathrm{score})$",
        fontsize=10,
    )

    path = os.path.join(out_dir, "fig_residue_type_heatmap.svg")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 — residue-type heatmap stratified by Cα–Cα distance bin
# ---------------------------------------------------------------------------

def fig_residue_type_heatmap_by_d3(
    stats: dict,
    out_dir: str,
    n_d3_bins: int = 8,
    d3_min: float = 2.0,
    d3_max: float = 12.0,
    vmin: float | None = None,
    vmax: float | None = None,
    normalize: bool = True,
    seq_dist_min: int = 1,
) -> None:
    """
    L-row × n_d3_bins-column figure.

    Each cell shows the 20×20 residue-type score heatmap for non-self edges
    whose Cα–Cα distance falls within that bin, for one encoder layer.

    Parameters
    ----------
    n_d3_bins : int
        Number of equally-spaced distance bins between *d3_min* and *d3_max*.
    d3_min, d3_max : float
        Distance range (Å); edges outside are excluded.
    vmin, vmax : float or None
        Colorscale bounds in score units.  ``None`` → data-driven.
    normalize : bool
        If True, divide each (layer, bin) grid by its nanmean before
        log-transforming.  This removes the distance-driven baseline shift so
        the colorscale reflects AA preference *relative to the bin average*
        rather than the absolute score.
    seq_dist_min : int
        Minimum sequential separation |i − j|.  Edges with ``seq_dist <
        seq_dist_min`` are excluded before distance binning.  Default 1 keeps
        all non-self edges; set to e.g. 6 for medium-/long-range contacts only.
    """
    non_self = (~stats["is_self"]) & (stats["seq_dist"] >= seq_dist_min)
    d3       = stats["dist_3d"][non_self]
    ti       = stats["res_type_i"][non_self].astype(int)
    tj       = stats["res_type_j"][non_self].astype(int)
    L        = stats["L"]
    N_AA     = len(_AA_ALPHABET)

    d3_edges = np.linspace(d3_min, d3_max, n_d3_bins + 1)

    # ── build score grids [L, n_d3_bins, N_AA, N_AA] ─────────────────────────
    grids = np.full((L, n_d3_bins, N_AA, N_AA), np.nan)
    for l in range(L):
        scored_l = stats["scored_attn"][l][non_self]
        for di in range(n_d3_bins):
            in_bin = (d3 >= d3_edges[di]) & (d3 < d3_edges[di + 1])
            for i in range(N_AA):
                for j in range(N_AA):
                    m = in_bin & (ti == i) & (tj == j)
                    if m.sum() > 0:
                        grids[l, di, i, j] = float(scored_l[m].mean())

    # ── optional bin-level normalisation ─────────────────────────────────────
    if normalize:
        for l in range(L):
            for di in range(n_d3_bins):
                bin_mean = np.nanmean(grids[l, di])
                if bin_mean > 0:
                    grids[l, di] /= bin_mean

    # ── log-transform + shared colorscale ────────────────────────────────────
    log_grids = np.where(np.isnan(grids), np.nan,
                         np.log(np.clip(grids, 1e-9, None)))

    flat_log   = log_grids[~np.isnan(log_grids)]
    log_vmin   = np.log(max(vmin, 1e-9)) if vmin is not None else float(flat_log.min())
    log_vmax   = np.log(vmax)            if vmax is not None else float(flat_log.max())
    log_vmin   = min(log_vmin, -1e-6)
    log_vmax   = max(log_vmax,  1e-6)
    norm       = mcolors.TwoSlopeNorm(vmin=log_vmin, vcenter=0.0, vmax=log_vmax)
    raw_ticks  = np.linspace(log_vmin, log_vmax, 7)
    tick_scores = np.exp(raw_ticks)

    kw = dict(aspect="equal", origin="upper",
              extent=[-0.5, N_AA - 0.5, N_AA - 0.5, -0.5])

    # ── layout ────────────────────────────────────────────────────────────────
    cell_size = 2.2
    fig, axes = plt.subplots(
        L, n_d3_bins,
        figsize=(cell_size * n_d3_bins, cell_size * L + 0.8),
        squeeze=False,
    )

    im = None
    for l in range(L):
        for di in range(n_d3_bins):
            ax = axes[l, di]
            im = ax.imshow(log_grids[l, di], cmap="RdBu_r", norm=norm, **kw)

            # AA tick labels on bottom row and left column only
            if l == L - 1:
                ax.set_xticks(range(N_AA))
                ax.set_xticklabels(_AA_ALPHABET, fontsize=5)
            else:
                ax.set_xticks([])

            if di == 0:
                ax.set_yticks(range(N_AA))
                ax.set_yticklabels(_AA_ALPHABET, fontsize=5)
                ax.set_ylabel(f"Layer {l + 1}", fontsize=8)
            else:
                ax.set_yticks([])

            if l == 0:
                lo, hi = d3_edges[di], d3_edges[di + 1]
                ax.set_title(f"{lo:.1f}–{hi:.1f} Å", fontsize=7)

    cb_label = (
        "Score / bin mean  (1 = bin average)" if normalize
        else "Score  (1 = uniform baseline)"
    )
    cb = fig.colorbar(im, ax=axes, shrink=0.6, label=cb_label, ticks=raw_ticks)
    cb.ax.set_yticklabels([f"{s:.2g}" for s in tick_scores])
    cb.ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    norm_tag = "  ·  bin-mean normalised" if normalize else ""
    fig.suptitle(
        r"Residue-type score by Cα–Cα distance bin  "
        r"score$_{ij}$ = $\mathrm{rel}\text{-}\alpha_{ij}^{\,\mathrm{budget}_i}$"
        r"  ·  colorscale: $\log(\mathrm{score})$" + norm_tag,
        fontsize=10,
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "fig_residue_type_heatmap_by_d3.svg")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")



# ---------------------------------------------------------------------------
# Figure 3 — sequence distance vs 3D distance heatmap
# ---------------------------------------------------------------------------

# Fixed sequence-distance bin edges and labels (mirrors pmpnn-oa-analysis.py fig5).
# The upper edge of the last bin is set dynamically from the data.
# Sequence-distance bin edges (lower-inclusive bounds; last bin = 6+).
# Covers: |i−j|=1, =2, =3–4, =5, ≥6
_SEQ_BIN_EDGES  = [1, 2, 3, 5, 6]   # lower bounds; last bin is open-ended
_SEQ_BIN_LABELS = ["1", "2", "3–4", "5", "6+"]


def fig_seq_vs_3d_heatmap(
    stats: dict,
    out_dir: str,
    n_3d_bins: int = 15,
    d3_min: float = 2.0,
    d3_max: float = 12.0,
    vmin: float | None = None,
    vmax: float | None = None,
    ec_min: float = 0.0,
    ec_max: float = 10.0,
    seq_dist_min: int = 1,
) -> None:
    """
    (L + 1)-panel figure: sequence distance |i−j| (y-axis) vs Cα–Cα distance
    (x-axis), one score panel per encoder layer plus a shared edge-count panel.

    Score panels
    ------------
    Each cell shows the mean budget-weighted score for non-self edges whose
    (sequence distance, 3D distance) falls in that bin.  The colorscale is
    log-transformed with TwoSlopeNorm centred at score = 1 (uniform baseline),
    matching the convention used in all other heatmaps in this script.

    Edge-count panel
    ----------------
    Shows mean edges per source residue in each bin (Blues palette).  This is
    layer-independent and serves as a data-density guide for the score panels.

    Sequence bins (fixed, matching pmpnn-oa-analysis.py fig5)
    ---------------------------------------------------------
    |i−j|  1 | 2 | 3 | 4–5 | 6–7 | 8–12 | 13–20 | 21–40 | 41+

    Parameters
    ----------
    n_3d_bins : int
        Number of equally-spaced 3D distance bins between *d3_min* and *d3_max*.
    d3_min, d3_max : float
        Distance range (Å); edges outside are excluded.
    vmin, vmax : float or None
        Score colorscale bounds in score units.  ``None`` → data-driven.
    ec_min, ec_max : float
        Edge-count colorscale bounds (mean edges per residue).
    seq_dist_min : int
        Minimum sequential separation |i − j|.  Edges below this threshold are
        excluded; the corresponding seq-distance rows will appear as NaN (white).
    """
    non_self = (~stats["is_self"]) & (stats["seq_dist"] >= seq_dist_min)
    d3       = stats["dist_3d"][non_self]
    sd       = stats["seq_dist"][non_self]
    res_ns   = stats["res_idx"][non_self]
    L        = stats["L"]

    # ── sequence bin edges ────────────────────────────────────────────────────
    sd_max     = int(sd.max()) if len(sd) > 0 else 41
    seq_edges  = _SEQ_BIN_EDGES + [sd_max + 1]
    seq_labels = _SEQ_BIN_LABELS[:len(seq_edges) - 1]
    n_seq      = len(seq_edges) - 1

    # ── 3D distance bin edges ─────────────────────────────────────────────────
    d3_edges = np.linspace(d3_min, d3_max, n_3d_bins + 1)

    # ── build score grids [L, n_seq, n_3d_bins] ───────────────────────────────
    grids = []
    for l in range(L):
        scored_l = stats["scored_attn"][l][non_self]
        grid = np.full((n_seq, n_3d_bins), np.nan)
        for si in range(n_seq):
            for di in range(n_3d_bins):
                m = (
                    (sd >= seq_edges[si]) & (sd < seq_edges[si + 1]) &
                    (d3 >= d3_edges[di])  & (d3 < d3_edges[di + 1])
                )
                if m.sum() > 0:
                    grid[si, di] = float(scored_l[m].mean())
        grids.append(grid)

    # ── edge-count grid: mean edges per source residue ────────────────────────
    count_grid = np.full((n_seq, n_3d_bins), np.nan)
    for si in range(n_seq):
        for di in range(n_3d_bins):
            m = (
                (sd >= seq_edges[si]) & (sd < seq_edges[si + 1]) &
                (d3 >= d3_edges[di])  & (d3 < d3_edges[di + 1])
            )
            n_edges = int(m.sum())
            if n_edges > 0:
                n_res = len(np.unique(res_ns[m]))
                count_grid[si, di] = n_edges / n_res

    # ── log-transform + shared colorscale ────────────────────────────────────
    log_grids = [np.where(np.isnan(g), np.nan, np.log(np.clip(g, 1e-9, None)))
                 for g in grids]
    norm, raw_ticks, tick_scores = _log_norm_and_ticks(log_grids, vmin, vmax)

    kw = dict(
        aspect="auto", origin="lower",
        extent=[d3_edges[0], d3_edges[-1], -0.5, n_seq - 0.5],
    )

    # ── layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, L + 1, figsize=(5 * (L + 1), 4), sharey=True)
    if L == 0:
        axes = [axes]

    # ── score panels ──────────────────────────────────────────────────────────
    im = None
    for l in range(L):
        ax = axes[l]
        im = ax.imshow(log_grids[l], cmap="RdBu_r", norm=norm, **kw)
        ax.set_yticks(range(n_seq))
        ax.set_yticklabels(seq_labels)
        ax.set_xlabel("3D Cα–Cα distance (Å)")
        if l == 0:
            ax.set_ylabel("Sequence distance |i−j|")
        ax.set_title(f"Layer {l + 1}")

    if im is not None:
        cb = fig.colorbar(im, ax=list(axes[:L]), shrink=0.8,
                          label="Score  (1 = uniform baseline)",
                          ticks=raw_ticks)
        cb.ax.set_yticklabels([f"{s:.2g}" for s in tick_scores])
        cb.ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    # ── edge-count panel ──────────────────────────────────────────────────────
    ax_c = axes[L]
    im_c = ax_c.imshow(count_grid, cmap="Blues",
                       vmin=ec_min, vmax=ec_max, **kw)
    ax_c.set_yticks(range(n_seq))
    ax_c.set_yticklabels(seq_labels)
    ax_c.set_xlabel("3D Cα–Cα distance (Å)")
    ax_c.set_title("Mean edges\nper residue")
    fig.colorbar(im_c, ax=ax_c, shrink=0.8, label="Mean edges per residue")

    sd_note = (
        "  ·  |i−j| ≥ " + str(seq_dist_min)
        if seq_dist_min > 1 else ""
    )
    fig.suptitle(
        "Budget-weighted score by sequence distance and Cα–Cα distance  "
        + r"score$_{ij}$ = rel-$\alpha_{ij}^{\,\mathrm{budget}_i}$"
        + "  ·  colorscale: "
        + r"$\log(\mathrm{score})$" + sd_note,
        fontsize=10,
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "fig_seq_vs_3d_heatmap.svg")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = HeatmapContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)

    stats = ctx.stats
    L     = stats["L"]
    mode  = str(stats.get("mode", "unknown"))
    n_res = len(stats["valid_k"])  if "valid_k" in stats else "?"
    n_edges = len(stats["is_self"])
    n_self  = int(stats["is_self"].sum())
    print(
        f"[heatmaps] mode={mode}  layers={L}  "
        f"residues={n_res:,}  edges={n_edges:,}  "
        f"(self={n_self:,}  non-self={n_edges - n_self:,})"
    )
    if ctx.seq_dist_min > 1:
        n_pass = int((~stats["is_self"] & (stats["seq_dist"] >= ctx.seq_dist_min)).sum())
        print(
            f"[heatmaps] seq_dist_min={ctx.seq_dist_min}  "
            f"non-self edges after filter: {n_pass:,} "
            f"({100 * n_pass / max(n_edges - n_self, 1):.1f} % of non-self)"
        )

    print("[heatmaps] Generating global residue-type heatmap …")
    fig_residue_type_heatmap(
        stats, ctx.out_dir,
        vmin         = ctx.aa_score_vmin,
        vmax         = ctx.aa_score_vmax,
        ec_min       = ctx.aa_ec_min,
        ec_max       = ctx.aa_ec_max,
        seq_dist_min = ctx.seq_dist_min,
    )

    print("[heatmaps] Generating distance-binned residue-type heatmap …")
    fig_residue_type_heatmap_by_d3(
        stats, ctx.out_dir,
        n_d3_bins    = ctx.d3_bins_aa,
        d3_min       = ctx.d3_min,
        d3_max       = ctx.d3_max,
        vmin         = ctx.strat_aa_score_vmin,
        vmax         = ctx.strat_aa_score_vmax,
        normalize    = ctx.normalize_d3_bins,
        seq_dist_min = ctx.seq_dist_min,
    )

    print("[heatmaps] Generating sequence-distance vs 3D-distance heatmap …")
    fig_seq_vs_3d_heatmap(
        stats, ctx.out_dir,
        n_3d_bins    = ctx.d3_bins_aa,
        d3_min       = ctx.d3_min,
        d3_max       = ctx.d3_max,
        vmin         = ctx.seq_vs_3d_score_vmin,
        vmax         = ctx.seq_vs_3d_score_vmax,
        ec_min       = ctx.seq_vs_3d_ec_min,
        ec_max       = ctx.seq_vs_3d_ec_max,
        seq_dist_min = ctx.seq_dist_min,
    )

    print(f"[heatmaps] Done — figures written to {ctx.out_dir!r}")


if __name__ == "__main__":
    main()