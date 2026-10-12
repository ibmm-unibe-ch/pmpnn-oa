"""
oa_budget.py — Budget distribution and neighbourhood-size analysis for
ProteinMPNN-OA.

Reads the .npz statistics file produced by stats.py and generates:

  fig_budget_violin.svg         — per-layer violin plot of the non-self budget
                                    (1 − α_self) on a log scale with Q1 / Q2 / Q3
                                    quartile annotations.

  fig_neighbors_bar.svg         — bar chart of the total neighbourhood size
                                    (valid_k − 1, i.e. non-self neighbours) per
                                    residue, with quartile annotations and the
                                    fraction of residues whose neighbourhood falls
                                    below 10, 20, 30 and 40 neighbours.

  fig_neighbors_by_layer_bar.svg— per-layer bar chart of the number of neighbours
                                    whose budget-weighted score exceeds a threshold,
                                    with the same quartile and fraction annotations.

No model weights or forward passes are required — all statistics are read from
the pre-computed .npz file.

This script works for OA-mode statistics files.  The neighbourhood figures also
work for GAT-mode files (they do not require theta / phi arrays).

Usage
-----
python oa_budget.py  stats.npz  [options]

Examples
--------
# Minimal — all defaults
python oa_budget.py  stats.npz

# Custom output directory and score threshold
python oa_budget.py  stats.npz  --out_dir figs/budget  --score_thresh 1.2

# Change fraction breakpoints
python oa_budget.py  stats.npz  --frac_thresholds 5 10 20 30

# Exclude local contacts
python oa_budget.py  stats.npz  --seq_dist_min 6
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class BudgetContext:
    """All configuration needed to generate the budget and neighbourhood plots.

    CLI-facing fields are validated in :meth:`__post_init__`; the class can
    also be instantiated programmatically for use in notebooks or tests.
    """

    # ── I/O ───────────────────────────────────────────────────────────────────
    stats_npz: str
    out_dir:   str


    # ── neighbourhood bar ─────────────────────────────────────────────────────
    score_thresh:    float  # score threshold for "active neighbours" plot
    frac_thresholds: list   # neighbourhood sizes for cumulative-fraction report
    max_neighbors:   int    # maximum possible neighbours (vertical reference line)
    rc_min:          float  # lower y-axis bound for residue count (log scale)
    rc_max:          float  # upper y-axis bound for residue count (log scale)

    # ── sequence-distance filter ───────────────────────────────────────────────
    seq_dist_min: int       # minimum |i − j|; mirrors spectral analysis default

    # ── runtime (populated in __post_init__) ──────────────────────────────────
    stats: dict = field(default=None, repr=False)

    # ─────────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # --- path validation --------------------------------------------------
        if not os.path.isfile(self.stats_npz):
            raise ValueError(f"stats_npz not found: {self.stats_npz!r}")
        if not self.stats_npz.endswith(".npz"):
            print(f"[warn] stats_npz does not have a .npz extension: "
                  f"{self.stats_npz!r}")

        out_parent = os.path.dirname(os.path.abspath(self.out_dir)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(
                f"Parent directory of out_dir does not exist: {out_parent!r}")

        # --- numeric validation -----------------------------------------------
        if self.score_thresh <= 0:
            raise ValueError(
                f"score_thresh must be > 0, got {self.score_thresh}")

        if self.max_neighbors < 1:
            raise ValueError(
                f"max_neighbors must be >= 1, got {self.max_neighbors}")

        if self.rc_min < 0:
            raise ValueError(
                f"rc_min must be >= 0, got {self.rc_min}")
        if self.rc_max <= max(self.rc_min, 0):
            raise ValueError(
                f"rc_max must be > rc_min, "
                f"got rc_min={self.rc_min}  rc_max={self.rc_max}")

        if len(self.frac_thresholds) == 0:
            raise ValueError("frac_thresholds must contain at least one value")
        for t in self.frac_thresholds:
            if t <= 0:
                raise ValueError(
                    f"All frac_thresholds must be > 0, got {t}")

        if self.seq_dist_min < 1:
            raise ValueError(
                f"seq_dist_min must be >= 1 (self-edges always excluded), "
                f"got {self.seq_dist_min}")

        # --- load and validate statistics ------------------------------------
        self.stats = _load_stats(self.stats_npz)

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_args(cls) -> "BudgetContext":
        """Parse sys.argv and return a validated :class:`BudgetContext`."""
        p = argparse.ArgumentParser(
            description=(
                "Generate budget distribution and neighbourhood-size figures "
                "from ProteinMPNN-OA statistics produced by stats.py."
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
            "--out_dir", default="budget",
            help="Output directory for figure files (created if absent)",
        )

        # ── neighbourhood bar ──────────────────────────────────────────────────
        nb = p.add_argument_group("Neighbourhood bar")
        nb.add_argument(
            "--score_thresh", type=float, default=1.0,
            help="Budget-weighted score threshold for the 'active neighbours' "
                 "per-layer plot (score units; 1 = uniform baseline)",
        )
        nb.add_argument(
            "--frac_thresholds", type=int, nargs="+",
            default=[10, 20, 30, 40],
            help="Neighbourhood sizes at which to report the cumulative fraction "
                 "of residues with fewer neighbours (space-separated integers)",
        )
        nb.add_argument(
            "--max_neighbors", type=int, default=48,
            help="Maximum number of neighbours (model hard limit); drawn as a "
                 "vertical reference line in the neighbourhood bar charts",
        )
        nb.add_argument(
            "--rc_min", type=float, default=0.0,
            help="Lower bound for the residue-count y-axis (log scale). "
                 "0.0 maps to 0.5 internally to keep the log scale valid.",
        )
        nb.add_argument(
            "--rc_max", type=float, default=1e6,
            help="Upper bound for the residue-count y-axis (log scale).",
        )

        # ── sequence-distance filter ──────────────────────────────────────────
        sd = p.add_argument_group("Sequence-distance filter")
        sd.add_argument(
            "--seq_dist_min", type=int, default=1,
            help="Minimum sequential separation |i − j| for edges included in "
                 "the neighbourhood counts.  Default 1 keeps all non-self edges.",
        )

        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# Statistics loading  (mirrors _load_stats from aa_heatmaps.py)
# ---------------------------------------------------------------------------

def _load_stats(path: str) -> dict:
    """
    Load a .npz file produced by stats.py and reconstruct the list-indexed
    per-layer arrays that the plotting functions expect.

    Returns
    -------
    dict with at minimum:
      ``L``            — int, number of encoder layers
      ``valid_k``      — float32 [total_residues]
      ``is_self``      — bool [total_edges]
      ``seq_dist``     — float32 [total_edges]
      ``res_idx``      — int64 [total_edges]
      ``alpha_self``   — list of L float32 arrays [total_residues]
      ``budget``       — list of L float32 arrays [total_edges]
      ``scored_attn``  — list of L float32 arrays [total_edges]
      (and all other flat keys from the .npz)
    """
    raw = np.load(path, allow_pickle=False)
    L   = int(raw["L"])

    stats: dict = dict(raw)

    # Rebuild per-layer list arrays
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
# Shared quartile helper
# ---------------------------------------------------------------------------

def _quartile_stats(data: np.ndarray) -> tuple[float, float, float]:
    """Return (Q1, Q2/median, Q3) for a 1-D array, ignoring NaN."""
    d = data[~np.isnan(data)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.percentile(d, 25)),
        float(np.percentile(d, 50)),
        float(np.percentile(d, 75)),
    )


def _add_quartile_annotation(
    ax: plt.Axes,
    x: float,
    q1: float,
    q2: float,
    q3: float,
    color: str = "black",
    fontsize: int = 7,
    log_y: bool = False,
) -> None:
    """Draw Q1 / Q2 / Q3 tick marks and a text label on the given axes."""
    for q, ls in [(q1, "--"), (q2, "-"), (q3, "--")]:
        ax.axhline(q, color=color, linewidth=0.8, linestyle=ls, alpha=0.55,
                   zorder=3)
    # Text label to the right of the violin
    if log_y:
        label_y = q3 * 1.5
    else:
        label_y = q3 + (q3 - q1) * 0.05
    ax.text(
        x, label_y,
        f"Q1={q1:.3g}\nQ2={q2:.3g}\nQ3={q3:.3g}",
        fontsize=fontsize, va="bottom", ha="left", color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )


# ---------------------------------------------------------------------------
# Figure 1 — budget violin plot (per layer, linear [0, 1] scale)
# ---------------------------------------------------------------------------

def fig_budget_violin(
    stats: dict,
    out_dir: str,
) -> None:
    """
    Violin plot of the non-self budget (1 − α_self) per encoder layer on a
    plain linear y-axis from 0 to 1.

    Each violin shows the full distribution of (1 − α_self) across all
    residues for that layer.  Quartile lines (Q1 dashed, Q2 solid, Q3 dashed)
    are overlaid on every violin and their values are annotated to the right.

    Parameters
    ----------
    stats   : dict produced by :func:`_load_stats`.
    out_dir : output directory.
    """
    L = stats["L"]

    if "alpha_self" not in stats:
        print("[budget] WARNING: alpha_self arrays not found — "
              "skipping budget violin plot")
        return

    # ── collect per-layer budget distributions ────────────────────────────────
    budgets = []
    for l in range(L):
        b = 1.0 - stats["alpha_self"][l].astype(np.float64)
        budgets.append(b)

    # ── layout ────────────────────────────────────────────────────────────────
    fig_width = max(4.0, 2.0 * L + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    positions = list(range(1, L + 1))
    palette   = plt.cm.tab10(np.linspace(0, 0.9, L))

    parts = ax.violinplot(
        budgets,
        positions=positions,
        showmedians=False,
        showextrema=False,
        widths=0.6,
    )

    for body, color in zip(parts["bodies"], palette):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(0.75)

    # ── quartile lines and annotations ────────────────────────────────────────
    for b, pos in zip(budgets, positions):
        q1, q2, q3 = _quartile_stats(b)

        for q, ls in [(q1, "--"), (q2, "-"), (q3, "--")]:
            ax.hlines(q, pos - 0.28, pos + 0.28,
                      colors="black", linewidths=1.2, linestyles=ls, zorder=4)

        ax.text(
            pos + 0.32, q3 + 0.01,
            f"Q1={q1:.3g}\nQ2={q2:.3g}\nQ3={q3:.3g}",
            fontsize=7, va="bottom", ha="left", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Layer {l + 1}" for l in range(L)], fontsize=9)
    ax.set_ylabel("Non-self budget  (1 − α_self)", fontsize=9)
    ax.set_title(
        "Distribution of non-self attention budget per encoder layer\n"
        "Q1/Q2(median)/Q3 shown as dashed/solid/dashed horizontal lines",
        fontsize=9,
    )
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_budget_violin.svg")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# Layer colour palette — matches _LAYER_COLORS in pmpnn-oa-analysis.py
_LAYER_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

def _lc(l: int) -> str:
    return _LAYER_COLORS[l % len(_LAYER_COLORS)]


# ---------------------------------------------------------------------------
# Figure 2 — combined neighbourhood chart (total + per-layer, single axis)
# ---------------------------------------------------------------------------

def fig_neighbors_combined(
    stats: dict,
    out_dir: str,
    score_thresh: float = 1.0,
    frac_thresholds: list | None = None,
    max_neighbors: int = 48,
    rc_min: float = 0.0,
    rc_max: float = 1e6,
    seq_dist_min: int = 1,
) -> None:
    """
    Single-figure, **two-panel** neighbourhood chart where each panel uses
    exactly **one axis** with all series overlaid as semi-transparent filled
    step-histogram surfaces, matching the style of the original analysis
    scripts (``pmpnn-oa-analysis.py``).

    **Top panel (single axis)** — total non-self neighbourhood size per
      residue (grey surface) overlaid with one surface per encoder layer
      showing the count of active neighbours (``scored_attn > score_thresh``).
      All distributions share the same axis so they can be compared directly.

    **Bottom panel (single axis)** — same per-layer active-neighbour surfaces
      as the top panel but shown alone, without the total-neighbourhood
      reference, to give more vertical detail for the per-layer comparison.

    Both panels share:
      - Log-scaled y-axis (residue count) with bounds *rc_min* / *rc_max*.
        ``rc_min = 0`` maps to ``0.5`` internally so the log scale stays
        valid while still reading as "0" on the axis label.
      - A vertical dotted dark-green line at *max_neighbors*.
      - Per-series quartile verticals: Q1 (dashed), Q2 (solid), Q3 (dashed)
        drawn in the series colour.
      - A monospaced annotation box (one per series) listing the fraction of
        residues whose count falls below each value in *frac_thresholds*.

    Output: ``fig_neighbors_bar.svg``.

    Parameters
    ----------
    stats           : dict produced by :func:`_load_stats`.
    out_dir         : output directory.
    score_thresh    : budget-weighted score threshold (score units;
                      1 = uniform baseline).
    frac_thresholds : neighbourhood sizes for cumulative-fraction annotations.
    max_neighbors   : hard neighbour limit drawn as a vertical reference line.
    rc_min          : lower y-axis bound (residue count); 0 → 0.5 floor.
    rc_max          : upper y-axis bound (residue count).
    seq_dist_min    : minimum |i − j|; edges below are excluded.
    """
    if frac_thresholds is None:
        frac_thresholds = [10, 20, 30, 40]
    frac_thresholds = sorted(frac_thresholds)

    # log scale floor: treat 0 as 0.5 (keeps axis valid, reads as "0")
    y_min_log = 0.5 if rc_min <= 0 else rc_min

    L = stats["L"]

    # ── compute total neighbourhood size per residue ──────────────────────────
    if seq_dist_min <= 1:
        k_total = stats["valid_k"].astype(np.float32) - 1.0
        k_total = np.clip(k_total, 0, None)
    else:
        is_self_r = stats["is_self"]
        seq_dist  = stats["seq_dist"]
        res_idx_r = stats["res_idx"]
        mask      = (~is_self_r) & (seq_dist >= seq_dist_min)
        n_res     = len(stats["valid_k"])
        k_total   = np.bincount(res_idx_r[mask].astype(int),
                                minlength=n_res).astype(np.float32)

    n_res_total = len(k_total)

    # ── compute per-layer active-neighbour counts ──────────────────────────────
    all_k_active: list[np.ndarray] = []
    if "scored_attn" in stats:
        is_self_e     = stats["is_self"]
        seq_d_e       = stats["seq_dist"]
        res_idx_e     = stats["res_idx"]
        non_self_mask = (~is_self_e) & (seq_d_e >= seq_dist_min)
        for l in range(L):
            scored_l = stats["scored_attn"][l]
            above    = non_self_mask & (scored_l > score_thresh)
            k_active = np.bincount(res_idx_e[above].astype(int),
                                   minlength=n_res_total).astype(np.float32)
            all_k_active.append(k_active)
    else:
        print("[budget] WARNING: scored_attn arrays not found — "
              "per-layer surfaces will be empty")
        all_k_active = [np.zeros(n_res_total, dtype=np.float32)] * L

    # ── shared x-axis range ───────────────────────────────────────────────────
    k_max_total  = int(k_total.max()) if len(k_total) > 0 else max_neighbors
    k_max_active = max((int(k.max()) for k in all_k_active), default=0)
    x_max = max(k_max_total, k_max_active, max_neighbors) + 1

    # ── inner helpers ─────────────────────────────────────────────────────────

    def _filled_step(ax, k, color, label, alpha=0.38, lw=1.5):
        """Filled semi-transparent step-histogram surface."""
        bins           = np.arange(0, x_max + 2)
        counts, edges  = np.histogram(k.astype(int), bins=bins)
        # Replace zeros with NaN so fill_between doesn't drag the log floor down
        y = counts.astype(float)
        y[y == 0] = np.nan
        x_step = np.repeat(edges, 2)[1:-1]
        y_step = np.repeat(y, 2)
        # fill_between needs finite values; use y_min_log as the baseline
        y_fill = np.where(np.isnan(y_step), y_min_log, y_step)
        ax.fill_between(x_step, y_min_log, y_fill,
                        color=color, alpha=alpha, linewidth=0, zorder=2)
        # Outline: draw step with NaN gaps so missing bins vanish cleanly
        ax.step(edges[:-1], np.where(np.isnan(y), y_min_log * 0.1, y),
                where="post", color=color, linewidth=lw,
                alpha=min(alpha + 0.40, 1.0), label=label, zorder=3)

    def _quartile_vlines(ax, k, color, annot_x, annot_y):
        """Quartile vertical lines + compact text annotation."""
        q1, q2, q3 = _quartile_stats(k)
        for q, ls in [(q1, "--"), (q2, "-"), (q3, "--")]:
            ax.axvline(q, color=color, linewidth=1.1, linestyle=ls,
                       alpha=0.70, zorder=4)
        ax.text(
            annot_x, annot_y,
            f"Q1={q1:.1f}  Q2={q2:.1f}  Q3={q3:.1f}",
            transform=ax.transAxes,
            fontsize=6.5, va="top", ha="right", color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="none", alpha=0.72),
        )

    def _frac_box(ax, k, color, ax_x, ax_y):
        """Fraction-below-threshold annotation box.

        Note: k_total = valid_k - 1, where valid_k includes the self-edge
        occupying one of the K slots.  Therefore the maximum possible non-self
        neighbour count is max_neighbors - 1, and the threshold for a full
        neighbourhood is (max_neighbors - 1).
        """
        n = len(k)
        # The self-edge occupies one K slot, so the max non-self count is
        # max_neighbors - 1.  Replace max_neighbors with that corrected value.
        max_ns = max_neighbors - 1
        all_thresholds = sorted(set(frac_thresholds) | {max_ns})
        lines = [
            f"< {ft:3d}{'*' if ft == max_ns else ' '}: "
            f"{100 * float((k < ft).sum()) / max(n, 1):5.1f}%"
            for ft in all_thresholds
        ]
        ax.text(
            ax_x, ax_y, "\n".join(lines),
            transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left", family="monospace",
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=color, alpha=0.80, linewidth=0.6),
        )

    def _finish_ax(ax, title):
        """Apply shared axis formatting."""
        ax.axvline(max_neighbors, color="darkgreen", linewidth=1.8,
                   linestyle=":", zorder=5,
                   label=f"max k = {max_neighbors}")
        ax.set_yscale("log")
        ax.set_ylim(y_min_log, rc_max)
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.set_xlabel("Contacts per residue", fontsize=9)
        ax.set_ylabel("Number of residues  (log scale)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right",
                  handlelength=1.8, labelspacing=0.25)
        ax.grid(axis="y", which="major", linewidth=0.5, alpha=0.45)
        ax.grid(axis="y", which="minor", linewidth=0.2, alpha=0.20)
        ax.set_axisbelow(True)

    # ── annotation position helpers ───────────────────────────────────────────
    # Quartile boxes stack from top-right; fraction boxes stack from top-left.
    # Each box occupies ~(n_frac_lines + 1) × 0.038 axes-fraction height.
    # +1 if max_neighbors is not already in frac_thresholds (extra line per box)
    _n_frac_lines = len(set(frac_thresholds) | {max_neighbors})
    box_h      = (_n_frac_lines + 0.8) * 0.038
    q_box_h    = 0.038                            # quartile line is one row
    n_series   = 1 + L                            # total + L layers
    annot_ys   = [0.97 - i * (box_h + 0.01) for i in range(n_series)]
    q_annot_ys = [0.97 - i * (q_box_h + 0.005) for i in range(n_series)]

    # ── layout ────────────────────────────────────────────────────────────────
    sd_note = f"  ·  |i−j| ≥ {seq_dist_min}" if seq_dist_min > 1 else ""
    fig_w   = max(9.0, min(16.0, x_max * 0.22 + 3.0))
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fig_w, 9.0),
        gridspec_kw=dict(hspace=0.40),
    )

    # ── top panel: total neighbourhood + all per-layer on ONE axis ────────────
    _q1_t, _q2_t, _q3_t = _quartile_stats(k_total)
    _filled_step(ax_top, k_total, color="#455A64",
                 label=(f"Total non-self neighbours"
                        f"  mean={k_total.mean():.1f}"
                        f"  Q1={_q1_t:.1f}  Q2={_q2_t:.1f}  Q3={_q3_t:.1f}"),
                 alpha=0.42)
    _quartile_vlines(ax_top, k_total, color="#455A64",
                     annot_x=0.985, annot_y=q_annot_ys[0])
    _frac_box(ax_top, k_total, color="#455A64",
              ax_x=0.015, ax_y=annot_ys[0])

    for l, k_active in enumerate(all_k_active):
        color = _lc(l)
        _q1_a, _q2_a, _q3_a = _quartile_stats(k_active)
        _filled_step(ax_top, k_active, color=color,
                     label=(f"Layer {l + 1}  score>{score_thresh:.2g}"
                            f"  mean={k_active.mean():.1f}"
                            f"  Q1={_q1_a:.1f}  Q2={_q2_a:.1f}  Q3={_q3_a:.1f}"),
                     alpha=0.32)
        _quartile_vlines(ax_top, k_active, color=color,
                         annot_x=0.985, annot_y=q_annot_ys[l + 1])
        _frac_box(ax_top, k_active, color=color,
                  ax_x=0.015, ax_y=annot_ys[l + 1])

    _finish_ax(ax_top,
               f"Neighbourhood size: total + active per layer{sd_note}")

    # ── bottom panel: per-layer active neighbours only on ONE axis ─────────────
    for l, k_active in enumerate(all_k_active):
        color = _lc(l)
        _q1_b, _q2_b, _q3_b = _quartile_stats(k_active)
        _filled_step(ax_bot, k_active, color=color,
                     label=(f"Layer {l + 1}  score>{score_thresh:.2g}"
                            f"  mean={k_active.mean():.1f}"
                            f"  Q1={_q1_b:.1f}  Q2={_q2_b:.1f}  Q3={_q3_b:.1f}"),
                     alpha=0.42)
        _quartile_vlines(ax_bot, k_active, color=color,
                         annot_x=0.985, annot_y=q_annot_ys[l])
        _frac_box(ax_bot, k_active, color=color,
                  ax_x=0.015, ax_y=annot_ys[l])

    _finish_ax(ax_bot,
               f"Active neighbours per layer  (score > {score_thresh:.2g})"
               f"{sd_note}")

    fig.suptitle(
        "Neighbourhood size distribution\n"
        f"Top: total + active  ·  Bottom: active per layer  "
        f"(score > {score_thresh:.2g})  ·  y-axis: log scale",
        fontsize=9,
    )

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_neighbors_bar.svg")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = BudgetContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)

    stats  = ctx.stats
    L      = stats["L"]
    mode   = str(stats.get("mode", "unknown"))
    n_res  = len(stats["valid_k"]) if "valid_k" in stats else "?"
    n_edges = len(stats["is_self"])
    n_self  = int(stats["is_self"].sum())

    print(
        f"[budget] mode={mode}  layers={L}  "
        f"residues={n_res:,}  edges={n_edges:,}  "
        f"(self={n_self:,}  non-self={n_edges - n_self:,})"
    )

    if ctx.seq_dist_min > 1:
        n_pass = int(
            (~stats["is_self"] & (stats["seq_dist"] >= ctx.seq_dist_min)).sum()
        )
        print(
            f"[budget] seq_dist_min={ctx.seq_dist_min}  "
            f"non-self edges after filter: {n_pass:,} "
            f"({100 * n_pass / max(n_edges - n_self, 1):.1f} % of non-self)"
        )

    print("[budget] Generating budget violin plot …")
    fig_budget_violin(stats, ctx.out_dir)

    print("[budget] Generating neighbourhood bar charts …")
    fig_neighbors_combined(
        stats, ctx.out_dir,
        score_thresh    = ctx.score_thresh,
        frac_thresholds = ctx.frac_thresholds,
        max_neighbors   = ctx.max_neighbors,
        rc_min          = ctx.rc_min,
        rc_max          = ctx.rc_max,
        seq_dist_min    = ctx.seq_dist_min,
    )

    print(f"[budget] Done — outputs written to {ctx.out_dir!r}")


if __name__ == "__main__":
    main()