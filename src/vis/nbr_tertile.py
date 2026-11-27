"""
vis.nbr_tertile — effect of local neighbourhood size (crowding) on the
budget-weighted attention score, via a within-file tertile stratification.

"Number of neighbours" here is the attending (source) residue's non-self
valid-neighbour count K_non_self — the same per-residue crowding quantity
``vis.stats``/``vis._bootstrap._load_stats`` use as the attention budget's
per-neighbour baseline denominator (``budget / K_non_self``), and the same
one ``vis.ss_bond_analysis`` already reports per edge as ``n_neighbors``
(see its module docstring). It is derived here from a plain ``stats.npz``
(written by ``vis.stats``) with no extra forward pass needed: per-residue
``valid_k`` (valid neighbours INCLUDING the self-edge, which every valid
residue has exactly one of) indexed by each edge's source ``res_idx`` gives
``n_neighbors = valid_k[res_idx] - 1`` for every non-self edge.

Every non-self edge in the shared --dist_min/--dist_max window is assigned
to one of 3 tertiles of its OWN n_neighbors value — separately for the
geometric-score pool (--geom_seq_dist_min, default 1: every edge) and the
AA-pair pool (--aa_seq_dist_min, default 6: medium/long-range only), since
these are different edge populations and the project's convention (see
``vis.persistence``) already treats them as such. Tertile cutpoints
(quantiles 1/3, 2/3) are computed ONCE from the full point-estimate sample
and then applied as FIXED thresholds inside every bootstrap replicate —
the same convention ``vis.persistence``/``vis.dist_corr`` use for their own
fixed bin edges — so a replicate's tertile membership never shifts with
its resampled population; only which edges of a given tertile are present
can change. Because n_neighbors is a per-residue quantity broadcast across
all of a residue's edges, and because most residues in a protein longer
than --num_neighbors simply have the maximal K_non_self (a point-mass at
the model's own k-NN cap), the 3 tertiles need not come out equal-sized —
expected, not a bug.

For every (layer, map family, panel-or-bin) cell, exactly like
``vis.persistence``, three map families are built with ``vis._bootstrap``'s
shared binning/double-centering machinery:

  score      raw geometric-mean budget-weighted score, per 2-D panel
             (dist x theta, dist x phi, theta x phi)
  score_c    the same panels, double-centered (additive row/column effects
             removed)
  aa_c       the 20x20 (attending type, attended type) double-centered
             score grid, one per distance bin

— but instead of comparing BASE against SEED/RANDOM/TEST files, here the
three TERTILE subsets of the SAME file are compared pairwise. Each tertile
is, in turn, treated as the "target" and reported against each of the
"two other tertiles" (so each unique pair appears twice in the report,
once from each tertile's point of view — the underlying statistic is
symmetric and computed only once per pair). The two bootstrap-aware
quantities reported per comparison are exactly ``vis.persistence``'s own
(reusing its ``compare()`` unchanged):

  - Pearson r of the two tertiles' point-estimate grids, restricted to
    cells reliably estimable in both, with a 95% CI from the SAME
    bootstrap stream (one resample of proteins per replicate, split three
    ways into tertile subsets within each replicate — i.e. the three
    tertiles' replicate-k grids are naturally PAIRED, not independent,
    which is in fact the more appropriate design here: all three tertiles
    partition the same sampled edge pool, so pairing them removes the
    sampling noise common to all three rather than adding spurious
    between-stream independence). ``vis.persistence.compare()``'s
    pivotal/basic bootstrap CI (correcting for the per-replicate
    regression-dilution bias described there) applies unchanged, since it
    only needs two aligned [B, *shape] replicate arrays, paired or not.
  - A sign-matched Jaccard overlap of each tertile's "marked regions"
    (cells whose 95% CI is entirely above, or entirely below, baseline),
    reported as a point estimate only (see ``vis.persistence`` for why a
    further CI would need a double bootstrap).

Usage
-----
python -m vis.nbr_tertile  STATS.npz  [options]

Output
------
--out (default nbr_tertile_report.md): one section per layer, one
subsection per map-family cell (geometric panel, or AA-pair distance bin),
each with 3 small tables (one per target tertile) of the usual
Comparison / N joint cells / Pearson r [95% CI] / Cosine sim [95% CI] /
Jaccard (sign-matched) columns.

One comparison figure per (map family, panel-or-bin, layer) (--out_dir,
default --out's directory): fig_nbr_tertile_{geom|aa}_<key>_layer<N>.svg,
a small forest plot of the 3 unique pairwise Pearson r [95% CI].

One additional grid figure per layer: fig_nbr_tertile_aa_grid_layer<N>.svg
— a 3 (tertile) x --aa_n_dist_bins (distance bin) panel of the actual
double-centered 20x20 AA-pair score heatmaps (not a comparison statistic,
the maps themselves), the tertile-stratified counterpart of
``vis.__main__.plot_aa_heatmap_figure``, so every tertile x bin cell can be
eyeballed side by side.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel

from ._bootstrap import (
    BootData, _build_cell_ids, _compute_aa_grids, _compute_geom_grids,
    _load_stats, _stack_tree, _summarize_tree, bin_index, build_layout, run_bootstrap,
)
from .persistence import _SCORE_KEYS, _row, _table, compare

_TERT_LABELS = ("tertile 1 (low)", "tertile 2 (mid)", "tertile 3 (high)")
_PAIRS = [(0, 1), (0, 2), (1, 2)]

# Amino-acid display alphabet/order and colorbar tick helper — duplicated
# (not imported) from vis.__main__, which keeps its own plotting constants
# local rather than exposing them via vis._bootstrap; identical definitions
# so this figure matches vis.__main__.plot_aa_heatmap_figure's look exactly.
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
AA_ALPHABET_SORTED = list("AVLIMFWYSTNQKRHDECGP")
_AA_PERM = np.array([AA_ALPHABET.index(aa) for aa in AA_ALPHABET_SORTED])


def _cbar_ticks(clip: float):
    ticks = [clip * x for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    labels = [f"{2.0 ** t:.3g}×" for t in ticks]
    return ticks, labels


# ---------------------------------------------------------------------------
# Per-replicate grid construction — the ``extra``/``compute_fn`` pair fed into
# ``vis._bootstrap.BootData``/``run_bootstrap``
# ---------------------------------------------------------------------------

@dataclass
class NbrExtra:
    """Everything ``_nbr_bootstrap_compute`` needs for one replicate.
    ``geom_tert_edges``/``aa_tert_edges`` are the FIXED [q33, q67] tertile
    cutpoints computed once from the full (non-bootstrap) sample of each
    pool — see module docstring."""
    v_layers: dict
    seq_dist: np.ndarray
    dist: np.ndarray
    n_neighbors: np.ndarray
    geom_seq_min: int
    geom_cell_ids: dict
    geom_shapes: dict
    geom_min_count: int
    geom_tert_edges: np.ndarray
    aa_seq_min: int
    type_i: np.ndarray
    type_j: np.ndarray
    aa_dist_edges: np.ndarray
    aa_min_count: int
    aa_tert_edges: np.ndarray
    layers: list


def _tertile_id(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """0/1/2 tertile id from the 2 fixed quantile cutpoints *edges* =
    [q33, q67]: x <= q33 -> 0, q33 < x <= q67 -> 1, x > q67 -> 2."""
    return np.searchsorted(edges, x, side="right")


def _nbr_pool_to_views(idx_pool, extra: NbrExtra):
    """One protein-resampled edge pool -> the (geometric, AA) sub-selections
    — same medium/long-range split as vis.persistence._persist_pool_to_views."""
    sd = extra.seq_dist[idx_pool]
    idx_geom = idx_pool if extra.geom_seq_min <= 1 else idx_pool[sd >= extra.geom_seq_min]
    idx_aa = idx_pool[sd >= extra.aa_seq_min]
    return idx_geom, idx_aa


def _nbr_bootstrap_compute(idx_pool, extra: NbrExtra) -> dict:
    idx_geom, idx_aa = _nbr_pool_to_views(idx_pool, extra)
    geom_tert = _tertile_id(extra.n_neighbors[idx_geom], extra.geom_tert_edges)
    aa_tert = _tertile_id(extra.n_neighbors[idx_aa], extra.aa_tert_edges)

    geom_out, aa_out = {}, {}
    for t in range(3):
        geom_out[t] = _compute_geom_grids(
            idx_geom[geom_tert == t], extra.v_layers, extra.geom_cell_ids,
            extra.geom_shapes, extra.geom_min_count, extra.layers)
        aa_out[t] = _compute_aa_grids(
            idx_aa[aa_tert == t], extra.v_layers, extra.type_i, extra.type_j,
            extra.dist, extra.aa_dist_edges, extra.aa_min_count, extra.layers)
    return {"geom": geom_out, "aa": aa_out}


# ---------------------------------------------------------------------------
# One input file -> its per-layer, per-tertile bootstrap point estimate +
# raw bootstrap-replicate stack
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    total_layers: int
    mode: str
    n_bootstraps_actual: int
    summary: dict   # summary["geom"][t][l][panel(+"_c")] / summary["aa"][t][l][bin] -> leaf dict
    boot: dict       # same key structure, leaf -> ndarray [n_bootstraps_actual, *shape]
    geom_tert_edges: np.ndarray
    aa_tert_edges: np.ndarray
    geom_tert_counts: np.ndarray
    aa_tert_counts: np.ndarray
    aa_bin_counts: np.ndarray   # (3, aa_n_dist_bins) int — edges per (tertile, bin) cell


def _run(path: str, layers: list, ctx: "NbrContext", parallel: Parallel) -> RunResult:
    stats = _load_stats(path, layers=layers)
    dist, theta, phi = stats["dist_3d"], stats["theta"], stats["phi"]
    seq_dist = stats["seq_dist"]
    type_i, type_j = stats["res_type_i"], stats["res_type_j"]
    v_layers = {l: np.log2(s) for l, s in stats["scored_attn"].items()}

    with np.load(path, allow_pickle=False) as raw:
        valid_k = raw["valid_k"].astype(np.float64)
    # Per non-self edge: K_non_self of the ATTENDING (source) residue —
    # valid_k counts valid neighbours INCLUDING the self-edge every valid
    # residue has exactly one of, hence the "- 1".
    n_neighbors = valid_k[stats["res_idx"]] - 1.0

    _, protein_ids = np.unique(stats["protein_idx"], return_inverse=True)
    n_proteins = int(protein_ids.max()) + 1

    base_mask = (dist >= ctx.dist_min) & (dist < ctx.dist_max)
    flat_idx, starts, lengths = build_layout(base_mask, protein_ids, n_proteins)
    if flat_idx.size == 0:
        raise RuntimeError(f"{path!r}: no edges survive --dist_min/--dist_max")

    geom_cell_ids, geom_shapes = _build_cell_ids(dist, theta, phi, ctx.geom_edges)

    idx_geom_full = (flat_idx if ctx.geom_seq_dist_min <= 1
                     else flat_idx[seq_dist[flat_idx] >= ctx.geom_seq_dist_min])
    idx_aa_full = flat_idx[seq_dist[flat_idx] >= ctx.aa_seq_dist_min]
    if idx_geom_full.size == 0:
        raise RuntimeError(f"{path!r}: no edges survive --geom_seq_dist_min")
    if idx_aa_full.size == 0:
        raise RuntimeError(f"{path!r}: no edges survive --aa_seq_dist_min")

    geom_tert_edges = np.quantile(n_neighbors[idx_geom_full], [1 / 3, 2 / 3])
    aa_tert_edges = np.quantile(n_neighbors[idx_aa_full], [1 / 3, 2 / 3])
    geom_tert_counts = np.bincount(
        _tertile_id(n_neighbors[idx_geom_full], geom_tert_edges), minlength=3)
    aa_tert_full = _tertile_id(n_neighbors[idx_aa_full], aa_tert_edges)
    aa_tert_counts = np.bincount(aa_tert_full, minlength=3)

    n_aa_bins = len(ctx.aa_dist_edges) - 1
    aa_bin_counts = np.zeros((3, n_aa_bins), dtype=np.int64)
    for t in range(3):
        db_t = bin_index(dist[idx_aa_full[aa_tert_full == t]], ctx.aa_dist_edges)
        aa_bin_counts[t] = np.bincount(db_t[db_t >= 0], minlength=n_aa_bins)

    extra = NbrExtra(
        v_layers=v_layers, seq_dist=seq_dist, dist=dist, n_neighbors=n_neighbors,
        geom_seq_min=ctx.geom_seq_dist_min, geom_cell_ids=geom_cell_ids, geom_shapes=geom_shapes,
        geom_min_count=ctx.geom_min_count, geom_tert_edges=geom_tert_edges,
        aa_seq_min=ctx.aa_seq_dist_min, type_i=type_i, type_j=type_j,
        aa_dist_edges=ctx.aa_dist_edges, aa_min_count=ctx.aa_min_count,
        aa_tert_edges=aa_tert_edges, layers=layers,
    )
    bd = BootData(flat_idx=flat_idx, starts=starts, lengths=lengths,
                  n_proteins=n_proteins, extra=extra)

    point = _nbr_bootstrap_compute(flat_idx, extra)

    seeds = np.random.SeedSequence(ctx.root_seed).spawn(ctx.n_bootstraps)
    n_tasks = min(ctx.n_bootstraps, 4 * (os.cpu_count() or 1))
    batch_size = -(-ctx.n_bootstraps // n_tasks)
    results, n_none = run_bootstrap(parallel, seeds, bd, batch_size, _nbr_bootstrap_compute)
    if not results:
        raise RuntimeError(f"{path!r}: every bootstrap replicate was empty")
    if n_none:
        print(f"  {n_none}/{ctx.n_bootstraps} empty replicates")

    boot = _stack_tree(results)
    summary = _summarize_tree(point, boot, ctx.n_bootstraps, ctx.min_coverage)

    print(f"[nbr_tertile] proteins={n_proteins:,}  edges in window={flat_idx.size:,}  "
         f"usable replicates={len(results):,}/{ctx.n_bootstraps:,}")
    print(f"  geom pool: n={idx_geom_full.size:,}  q33/q67={geom_tert_edges.tolist()}  "
         f"tertile counts={geom_tert_counts.tolist()}")
    print(f"  aa pool:   n={idx_aa_full.size:,}  q33/q67={aa_tert_edges.tolist()}  "
         f"tertile counts={aa_tert_counts.tolist()}")

    return RunResult(total_layers=stats["L"], mode=stats["mode"],
                     n_bootstraps_actual=len(results), summary=summary, boot=boot,
                     geom_tert_edges=geom_tert_edges, aa_tert_edges=aa_tert_edges,
                     geom_tert_counts=geom_tert_counts, aa_tert_counts=aa_tert_counts,
                     aa_bin_counts=aa_bin_counts)


# ---------------------------------------------------------------------------
# Pairwise comparison + report assembly
# ---------------------------------------------------------------------------

def _compare_pairs(summary_by_t: dict, boot_by_t: dict, min_joint_cells: int) -> dict:
    """{(a, b): SimilarityResult|None} for the 3 unique tertile pairs
    (0, 1), (0, 2), (1, 2) — computed once, reused for both directions'
    "target vs other" display rows (the statistic is symmetric)."""
    return {(a, b): compare(boot_by_t[a], summary_by_t[a], boot_by_t[b], summary_by_t[b],
                            min_joint_cells)
           for a, b in _PAIRS}


def _target_tables(pair_results: dict) -> str:
    """One small table per tertile taken as the "target", each with 2 rows
    (vs each of the other two tertiles) — see module docstring for why
    each unique pair's SimilarityResult is reused rather than recomputed."""
    blocks = []
    for t in range(3):
        rows = []
        for o in range(3):
            if o == t:
                continue
            pair = (t, o) if (t, o) in pair_results else (o, t)
            rows.append(_row(f"vs {_TERT_LABELS[o]}", pair_results[pair]))
        blocks.append(f"**Target: {_TERT_LABELS[t]}**\n\n" + _table(rows, with_verdict=False))
    return "\n\n".join(blocks)


def _plot_pairs(map_name: str, key: str, pair_results: dict, path: str) -> None:
    """Forest plot of the 3 unique pairwise Pearson r [95% CI]."""
    rows = [(f"{_TERT_LABELS[a]} vs {_TERT_LABELS[b]}", res)
           for (a, b), res in pair_results.items() if res is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    for yi, (label, res) in zip(y, rows):
        lo, hi = res.pearson_ci
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color="k", lw=1.5)
        ax.scatter([res.pearson_r], [yi], color="tab:blue", zorder=3, s=25)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title(f"{map_name} — {key}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _plot_aa_grid(run: RunResult, layer: int, ctx: "NbrContext", path: str) -> None:
    """3 (tertile) x --aa_n_dist_bins (distance bin) grid of the actual
    double-centered 20x20 AA-pair score heatmap — not a comparison
    statistic, the maps themselves, laid out so every tertile x bin cell
    can be compared by eye at once. One row per tertile (low/mid/high
    n_neighbors), one column per Ca-Ca distance bin; same colormap,
    amino-acid ordering, and significance-dot convention as
    vis.__main__.plot_aa_heatmap_figure."""
    n_bins = ctx.aa_n_dist_bins
    bin_edges = ctx.aa_dist_edges
    clip = ctx.aa_clip

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")
    norm = mcolors.Normalize(vmin=-clip, vmax=clip)
    ticks, ticklabels = _cbar_ticks(clip)

    fig, axes = plt.subplots(3, n_bins, figsize=(4.6 * n_bins, 5.0 * 3),
                             squeeze=False, sharex=True, sharey=True)
    im = None
    for t in range(3):
        for b in range(n_bins):
            ax = axes[t, b]
            s = run.summary["aa"][t][layer][b]
            shown = np.where(s["reliable"], np.clip(s["point"], -clip, clip), np.nan)
            shown = shown[np.ix_(_AA_PERM, _AA_PERM)]
            sig = s["signif"][np.ix_(_AA_PERM, _AA_PERM)]

            im = ax.imshow(shown, cmap=cmap, norm=norm, aspect="auto", origin="lower",
                           rasterized=True)
            rows, cols = np.nonzero(sig)
            ax.scatter(cols, rows, s=5, c="k", marker="o", linewidths=0)

            title = f"n={run.aa_bin_counts[t, b]:,}"
            if t == 0:
                title = f"{bin_edges[b]:.1f}–{bin_edges[b + 1]:.1f} Å\n" + title
            ax.set_title(title, fontsize=9)
            if b == 0:
                ax.set_ylabel(f"{_TERT_LABELS[t]}\nAttending type (i)", fontsize=8)
            ax.set_xticks(range(len(AA_ALPHABET_SORTED)), AA_ALPHABET_SORTED, fontsize=6)
            ax.set_yticks(range(len(AA_ALPHABET_SORTED)), AA_ALPHABET_SORTED, fontsize=6)
            if t == 2:
                ax.set_xlabel("Attended type (j)", fontsize=8)

    fig.suptitle(
        f"Layer {layer + 1} — double-centered budget-weighted score (log2), by "
        f"neighbour-count tertile × Ca-Ca distance bin, |i−j| ≥ "
        f"{ctx.aa_seq_dist_min}\n"
        f"dots: 95% protein-bootstrap CI entirely above 1×; grey: "
        f"<{ctx.aa_min_count} edges or low bootstrap coverage",
        fontsize=10,
    )
    fig.subplots_adjust(right=0.90, top=0.88)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cb = fig.colorbar(im, cax=cbar_ax, extend="both")
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.set_label("Score relative to additive row/column background")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class NbrContext:
    npz: str
    layers: list

    dist_min: float
    dist_max: float

    geom_seq_dist_min: int
    geom_n_dist: int
    geom_n_theta: int
    geom_n_phi: int
    geom_min_count: int

    aa_seq_dist_min: int
    aa_n_dist_bins: int
    aa_min_count: int
    aa_clip: float

    n_bootstraps: int
    root_seed: int
    min_coverage: float
    min_joint_cells: int
    n_jobs: int

    out: str
    out_dir: str

    layers_idx: list = field(default=None, repr=False)
    total_layers: int = field(default=None, repr=False)
    geom_edges: dict = field(default=None, repr=False)
    aa_dist_edges: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not os.path.isfile(self.npz):
            raise ValueError(f"input file not found: {self.npz!r}")

        with np.load(self.npz, allow_pickle=False) as raw:
            if "valid_k" not in raw:
                raise ValueError(
                    f"{self.npz!r} is missing 'valid_k' (per-residue valid-neighbour "
                    f"count) — re-run vis.stats")
            self.total_layers = int(raw["L"])

        if self.layers is None:
            self.layers_idx = list(range(self.total_layers))
        else:
            bad = [l for l in self.layers if l < 1 or l > self.total_layers]
            if bad:
                raise ValueError(f"--layers must be in [1, {self.total_layers}], got "
                                 f"out-of-range value(s) {bad}")
            self.layers_idx = sorted({l - 1 for l in self.layers})

        if self.dist_min < 0:
            raise ValueError(f"dist_min must be >= 0, got {self.dist_min}")
        if self.dist_max <= self.dist_min:
            raise ValueError(f"dist_max must be > dist_min, got {self.dist_min} / {self.dist_max}")
        if self.geom_seq_dist_min < 1:
            raise ValueError(f"geom_seq_dist_min must be >= 1, got {self.geom_seq_dist_min}")
        if self.aa_seq_dist_min < 1:
            raise ValueError(f"aa_seq_dist_min must be >= 1, got {self.aa_seq_dist_min}")
        for name, val in [("geom_n_dist", self.geom_n_dist), ("geom_n_theta", self.geom_n_theta),
                          ("geom_n_phi", self.geom_n_phi)]:
            if val < 2:
                raise ValueError(f"{name} must be >= 2, got {val}")
        if self.geom_min_count < 1:
            raise ValueError(f"geom_min_count must be >= 1, got {self.geom_min_count}")
        if self.aa_n_dist_bins < 1:
            raise ValueError(f"aa_n_dist_bins must be >= 1, got {self.aa_n_dist_bins}")
        if self.aa_min_count < 1:
            raise ValueError(f"aa_min_count must be >= 1, got {self.aa_min_count}")
        if self.aa_clip <= 0:
            raise ValueError(f"aa_clip must be > 0, got {self.aa_clip}")
        if self.n_bootstraps < 100:
            raise ValueError(f"n_bootstraps must be >= 100, got {self.n_bootstraps}")
        if not (0.0 < self.min_coverage <= 1.0):
            raise ValueError(f"min_coverage must be in (0, 1], got {self.min_coverage}")
        if self.min_joint_cells < 1:
            raise ValueError(f"min_joint_cells must be >= 1, got {self.min_joint_cells}")

        out_parent = os.path.dirname(os.path.abspath(self.out)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(f"Parent directory of --out does not exist: {out_parent!r}")
        if self.out_dir is None:
            self.out_dir = out_parent

        self.geom_edges = {
            "dist": np.linspace(self.dist_min, self.dist_max, self.geom_n_dist + 1),
            "theta": np.linspace(0.0, np.pi, self.geom_n_theta + 1),
            "phi": np.linspace(-np.pi, np.pi, self.geom_n_phi + 1),
        }
        self.aa_dist_edges = np.linspace(self.dist_min, self.dist_max, self.aa_n_dist_bins + 1)

    @classmethod
    def from_args(cls) -> "NbrContext":
        p = argparse.ArgumentParser(
            description=(
                "Effect of local neighbourhood size (number of neighbours) on the "
                "budget-weighted attention score: edges are split into 3 tertiles by "
                "their attending residue's non-self neighbour count, and each tertile "
                "is compared, in turn, against the two other tertiles via a "
                "protein-level cluster bootstrap (Pearson r + sign-matched Jaccard, "
                "same statistics as vis.persistence)."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument("npz", help="stats.npz file written by vis.stats")
        p.add_argument("--layers", type=int, nargs="+", default=None, metavar="N",
                       help="1-indexed encoder layer(s) to analyse. Default: all layers.")

        d = p.add_argument_group("Distance window")
        d.add_argument("--dist_min", type=float, default=4.0)
        d.add_argument("--dist_max", type=float, default=12.0)

        g = p.add_argument_group("Geometric score map")
        g.add_argument("--geom_seq_dist_min", type=int, default=1)
        g.add_argument("--geom_n_dist", type=int, default=32)
        g.add_argument("--geom_n_theta", type=int, default=24)
        g.add_argument("--geom_n_phi", type=int, default=36)
        g.add_argument("--geom_min_count", type=int, default=30)

        aa = p.add_argument_group("Double-centered AA-pair map")
        aa.add_argument("--aa_seq_dist_min", type=int, default=6)
        aa.add_argument("--aa_n_dist_bins", type=int, default=4)
        aa.add_argument("--aa_min_count", type=int, default=10)
        aa.add_argument("--aa_clip", type=float, default=1.0,
                       help="Colorbar clip (log2 score units) for the tertile x "
                            "distance-bin AA-pair heatmap grid")

        b = p.add_argument_group("Protein-level cluster bootstrap")
        b.add_argument("--n_bootstraps", type=int, default=5000)
        b.add_argument("--root_seed", type=int, default=12345)
        b.add_argument("--min_coverage", type=float, default=0.95)
        b.add_argument("--min_joint_cells", type=int, default=5,
                       help="Minimum cells reliably estimable in BOTH tertiles of a "
                            "pair for a comparison to be reported")
        b.add_argument("--n_jobs", type=int, default=-1)

        o = p.add_argument_group("Output")
        o.add_argument("--out", default="nbr_tertile_report.md")
        o.add_argument("--out_dir", default=None,
                       help="Figure output directory (default: next to --out)")

        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = NbrContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)

    print(f"[nbr_tertile] {ctx.npz}  layers={[l + 1 for l in ctx.layers_idx]}  "
         f"n_bootstraps={ctx.n_bootstraps:,}  min_coverage={ctx.min_coverage:.0%}  "
         f"min_joint_cells={ctx.min_joint_cells}")

    with Parallel(n_jobs=ctx.n_jobs, return_as="generator_unordered", max_nbytes="1M") as parallel:
        run = _run(ctx.npz, ctx.layers_idx, ctx, parallel)

    md_sections = [
        "# Neighbour-count tertile report\n",
        f"Input: `{ctx.npz}`  (mode={run.mode!r})  \n"
        f"n_bootstraps={ctx.n_bootstraps:,} (paired across tertiles — one resampling "
        f"stream, split 3 ways per replicate), min_coverage={ctx.min_coverage:.0%}, "
        f"min_joint_cells={ctx.min_joint_cells}\n",
        f"Geometric pool (|i-j| >= {ctx.geom_seq_dist_min}): n={int(run.geom_tert_counts.sum()):,}, "
        f"tertile cutpoints (number of non-self neighbours) q33/q67="
        f"{run.geom_tert_edges.tolist()}, per-tertile edge counts={run.geom_tert_counts.tolist()}  \n"
        f"AA-pair pool (|i-j| >= {ctx.aa_seq_dist_min}): n={int(run.aa_tert_counts.sum()):,}, "
        f"tertile cutpoints q33/q67={run.aa_tert_edges.tolist()}, "
        f"per-tertile edge counts={run.aa_tert_counts.tolist()}\n",
    ]

    aa_labels = [(b, f"{ctx.aa_dist_edges[b]:.1f}–{ctx.aa_dist_edges[b + 1]:.1f} Å")
                for b in range(ctx.aa_n_dist_bins)]

    for l in ctx.layers_idx:
        layer_md = [f"## Layer {l + 1}"]

        for key in _SCORE_KEYS:
            summary_by_t = {t: run.summary["geom"][t][l][key] for t in range(3)}
            boot_by_t = {t: run.boot["geom"][t][l][key] for t in range(3)}
            pair_results = _compare_pairs(summary_by_t, boot_by_t, ctx.min_joint_cells)
            layer_md.append(f"### Geometric score map — {key}\n\n"
                           + _target_tables(pair_results))
            fig_path = os.path.join(
                ctx.out_dir, f"fig_nbr_tertile_geom_{key}_layer{l + 1}.svg")
            _plot_pairs("Geometric score map", key, pair_results, fig_path)

        for b, lab in aa_labels:
            summary_by_t = {t: run.summary["aa"][t][l][b] for t in range(3)}
            boot_by_t = {t: run.boot["aa"][t][l][b] for t in range(3)}
            pair_results = _compare_pairs(summary_by_t, boot_by_t, ctx.min_joint_cells)
            layer_md.append(f"### AA-pair map — {lab}\n\n" + _target_tables(pair_results))
            fig_path = os.path.join(
                ctx.out_dir, f"fig_nbr_tertile_aa_bin{b}_layer{l + 1}.svg")
            _plot_pairs("AA-pair map", lab, pair_results, fig_path)

        grid_path = os.path.join(ctx.out_dir, f"fig_nbr_tertile_aa_grid_layer{l + 1}.svg")
        _plot_aa_grid(run, l, ctx, grid_path)

        md_sections.append("\n\n".join(layer_md))

    document = "\n\n".join(md_sections) + "\n"
    with open(ctx.out, "w") as f:
        f.write(document)
    print(f"[nbr_tertile] saved {ctx.out}")


if __name__ == "__main__":
    main()
