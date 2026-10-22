"""
vis.persistence — falsifiable cross-condition persistence test for
ProteinMPNN-OA / GAT geometric attention patterns.

Qualitative claims like "the attention pattern persists across condition X"
(e.g. a different seed, an ablation, a different architecture) are only
falsifiable once there is a statistic, a protein-bootstrap CI on it, and two
calibration references: how similar the SAME condition is to itself across
seeds (an upper bound on expected agreement) and how similar an untrained /
random baseline is (a lower bound). This script computes exactly that, for
three map families built identically to ``vis.__main__`` (same binning, same
double-centering, same protein-level cluster bootstrap machinery, imported
from :mod:`vis._bootstrap`):

  score      raw geometric-mean budget-weighted score, per 2-D panel
             (dist x theta, dist x phi, theta x phi)
  score_c    the same panels, double-centered (additive row/column effects
             removed) — the actual object of most persistence claims, since
             it isolates the geometric bump/dip from the distance-driven
             background
  aa_c       the 20x20 (attending type, attended type) double-centered score
             grid, one per distance bin

For every (layer, map family, panel-or-bin) cell, every ``--seed`` file and
the ``--random`` file are compared against ``base_npz`` to build the two
calibration references, and every ``--test`` file is compared the same way
and given a verdict: "persists" (clearly above the random baseline and
within/above the between-seed band), "indistinguishable from random", or
"intermediate" (above random, below the seed band).

Two bootstrap-aware quantities are reported per comparison:
  - Pearson r and cosine similarity of the two conditions' point-estimate
    grids, restricted to cells reliably estimable in BOTH (95% CI from an
    UNPAIRED protein-level bootstrap: each file gets its own independent
    resampling stream, and replicate k of A is compared to replicate k of B
    — valid because the two streams are independent, so index-pairing them
    is equivalent to the convolution of their sampling distributions). The
    CI uses the basic/pivotal bootstrap (reflecting the replicate
    distribution's spread around the point, see :func:`_pivotal_ci`), not
    the plain percentile method — the replicate distribution is
    systematically biased low relative to the point whenever there is real
    signal (each replicate's per-cell grid means are noisier than the
    full-sample point's, so a derived statistic like Pearson r over many
    such cells is attenuated), most visibly for the AA-pair map where real
    amino-acid frequency skew puts many cells right at --aa_min_count.
  - A sign-matched Jaccard overlap of each condition's "marked regions"
    (cells whose 95% CI is entirely above baseline, or entirely below it —
    the two-sided counterpart of ``vis.__main__``'s single-sided
    significance dots), reported as a POINT estimate only: it is already a
    second-order statistic of the bootstrap-derived significance masks, so
    a further CI on it would require a double bootstrap, out of scope here.

Usage
-----
python -m vis.persistence  BASE.npz  --seeds SEED1.npz [SEED2.npz ...] \\
    --random RANDOM.npz  --test TEST1.npz [TEST2.npz ...]  [options]
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel
from scipy import stats as sp_stats

from vis._bootstrap import (
    BootData, _build_cell_ids, _compute_aa_grids, _compute_geom_grids,
    _load_stats, _stack_tree, _summarize_tree, build_layout, run_bootstrap,
)

_GEOM_PANELS = ("dist_theta", "dist_phi", "theta_phi")
_SCORE_KEYS = list(_GEOM_PANELS) + [p + "_c" for p in _GEOM_PANELS]


# ---------------------------------------------------------------------------
# Per-replicate grid construction — the ``extra``/``compute_fn`` pair fed into
# ``vis._bootstrap.BootData``/``run_bootstrap``
# ---------------------------------------------------------------------------

@dataclass
class PersistExtra:
    """Everything ``_persist_bootstrap_compute`` needs for one replicate."""
    seq_dist: np.ndarray
    dist: np.ndarray
    geom_seq_min: int
    geom_cell_ids: dict
    geom_shapes: dict
    geom_min_count: int
    aa_seq_min: int
    v_layers: dict
    type_i: np.ndarray
    type_j: np.ndarray
    aa_dist_edges: np.ndarray
    aa_min_count: int
    layers: list


def _persist_pool_to_views(idx_pool, extra: PersistExtra):
    """One protein-resampled edge pool -> the (geometric, AA) sub-selections —
    the AA map uses the same medium/long-range filter as vis.__main__'s
    edge-type results (default |i-j| >= 6), the geometric score map its own
    (default |i-j| >= 1, i.e. unfiltered)."""
    sd = extra.seq_dist[idx_pool]
    idx_geom = idx_pool if extra.geom_seq_min <= 1 else idx_pool[sd >= extra.geom_seq_min]
    idx_aa = idx_pool[sd >= extra.aa_seq_min]
    return idx_geom, idx_aa


def _persist_bootstrap_compute(idx_pool, extra: PersistExtra) -> dict:
    idx_geom, idx_aa = _persist_pool_to_views(idx_pool, extra)
    geom = _compute_geom_grids(idx_geom, extra.v_layers, extra.geom_cell_ids, extra.geom_shapes,
                               extra.geom_min_count, extra.layers)
    aa = _compute_aa_grids(idx_aa, extra.v_layers, extra.type_i, extra.type_j, extra.dist,
                           extra.aa_dist_edges, extra.aa_min_count, extra.layers)
    return {"geom": geom, "aa": aa}


# ---------------------------------------------------------------------------
# One resolved input file -> its per-layer bootstrap point estimate + raw
# bootstrap-replicate stack (the summary's lo/hi/reliable are not enough to
# build the similarity statistic's own CI — that needs the per-replicate
# grids themselves, see ``_bootstrap_similarity_ci``).
# ---------------------------------------------------------------------------

@dataclass
class FileRun:
    label: str
    path: str
    total_layers: int
    mode: str
    n_bootstraps_actual: int
    summary: dict   # summary["geom"][l][panel(+"_c")] / summary["aa"][l][bin] -> leaf dict
    boot: dict      # same key structure, leaf -> ndarray [n_bootstraps_actual, *shape]


def _run_file(label: str, path: str, layers: list, ctx: "PersistenceContext",
             parallel, file_index: int, n_files: int) -> FileRun:
    stats = _load_stats(path, layers=layers)
    dist, theta, phi = stats["dist_3d"], stats["theta"], stats["phi"]
    seq_dist = stats["seq_dist"]
    type_i, type_j = stats["res_type_i"], stats["res_type_j"]
    v_layers = {l: np.log2(s) for l, s in stats["scored_attn"].items()}

    _, protein_ids = np.unique(stats["protein_idx"], return_inverse=True)
    n_proteins = int(protein_ids.max()) + 1

    base_mask = (dist >= ctx.dist_min) & (dist < ctx.dist_max)
    flat_idx, starts, lengths = build_layout(base_mask, protein_ids, n_proteins)
    if flat_idx.size == 0:
        raise RuntimeError(f"{label} ({path!r}): no edges survive --dist_min/--dist_max")

    geom_cell_ids, geom_shapes = _build_cell_ids(dist, theta, phi, ctx.geom_edges)

    extra = PersistExtra(
        seq_dist=seq_dist, dist=dist, geom_seq_min=ctx.geom_seq_dist_min,
        geom_cell_ids=geom_cell_ids, geom_shapes=geom_shapes, geom_min_count=ctx.geom_min_count,
        aa_seq_min=ctx.aa_seq_dist_min, v_layers=v_layers, type_i=type_i, type_j=type_j,
        aa_dist_edges=ctx.aa_dist_edges, aa_min_count=ctx.aa_min_count, layers=layers,
    )
    bd = BootData(flat_idx=flat_idx, starts=starts, lengths=lengths,
                  n_proteins=n_proteins, extra=extra)

    point = _persist_bootstrap_compute(flat_idx, extra)

    # Independent resampling stream per file (unpaired bootstrap): a child
    # SeedSequence per file index, each spawning its own n_bootstraps draws —
    # reproducible from one --root_seed, but NOT the same draws across files.
    file_seed_seq = np.random.SeedSequence(ctx.root_seed).spawn(n_files)[file_index]
    seeds = file_seed_seq.spawn(ctx.n_bootstraps)
    n_tasks = min(ctx.n_bootstraps, 4 * (os.cpu_count() or 1))
    batch_size = -(-ctx.n_bootstraps // n_tasks)
    results, n_none = run_bootstrap(parallel, seeds, bd, batch_size, _persist_bootstrap_compute)
    if not results:
        raise RuntimeError(f"{label} ({path!r}): every bootstrap replicate was empty")
    if n_none:
        print(f"  [{label}]   {n_none}/{ctx.n_bootstraps} empty replicates")

    boot = _stack_tree(results)
    summary = _summarize_tree(point, boot, ctx.n_bootstraps, ctx.min_coverage)
    print(f"  [{label}] proteins={n_proteins:,}  edges in window={flat_idx.size:,}  "
         f"usable replicates={len(results):,}/{ctx.n_bootstraps:,}")

    return FileRun(label=label, path=path, total_layers=stats["L"], mode=stats["mode"],
                   n_bootstraps_actual=len(results), summary=summary, boot=boot)


# ---------------------------------------------------------------------------
# Map-similarity statistic
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


@dataclass
class SimilarityResult:
    n_joint: int
    pearson_r: float
    pearson_ci: tuple
    cosine: float
    cosine_ci: tuple
    jaccard: float


def _pivotal_ci(point: float, vals: np.ndarray) -> tuple:
    """Basic/pivotal bootstrap 95% CI: reflects the replicate distribution's
    spread AROUND THE POINT instead of reporting the replicate distribution's
    own percentiles.

    Why not the plain percentile method (``[nanpercentile(vals, 2.5),
    nanpercentile(vals, 97.5)]``): each bootstrap replicate recomputes every
    cell's grid mean (and, for the AA-pair map, its double-centering) from a
    resampled set of proteins. For a sparse/skewed grid — the AA-pair map
    especially, since real amino-acid frequencies are skewed and many of its
    400 cells sit right at ``--aa_min_count`` — a single replicate's per-cell
    estimate is noisier than the full-sample point estimate. Correlating many
    such independently-noisy cells attenuates the per-replicate statistic
    (regression dilution): the replicate distribution of r/cosine sits
    systematically BELOW the point whenever there is real signal, so a plain
    percentile CI can exclude the point entirely — confirmed empirically
    (synthetic AA-frequency-skewed data with injected signal reproduces this
    exactly). The basic/pivotal method is the standard correction for a
    bootstrap distribution biased relative to its point estimate. When the
    bias is so large that even this overshoots the valid [-1, 1] range, the
    interval is widened just enough to still contain the point — an interval
    that excludes the very estimate it is a CI for is not a usable CI."""
    if not np.any(np.isfinite(vals)):
        return (float("nan"), float("nan"))
    lo_p, hi_p = np.nanpercentile(vals, [2.5, 97.5])
    lo = np.clip(2 * point - hi_p, -1.0, 1.0)
    hi = np.clip(2 * point - lo_p, -1.0, 1.0)
    lo, hi = min(lo, hi), max(lo, hi)
    lo, hi = min(lo, point), max(hi, point)
    return (float(lo), float(hi))


def _bootstrap_similarity_ci(boot_a: np.ndarray, boot_b: np.ndarray, M: np.ndarray,
                             min_joint_cells: int, point_r: float, point_cos: float) -> tuple:
    """95% CI on Pearson r and cosine similarity from two INDEPENDENT bootstrap
    replicate stacks (shape ``[B, *grid_shape]`` each), restricted to the fixed
    joint-domain mask ``M`` computed once from the point estimates. Pairing
    replicate k of A with replicate k of B is valid because the two streams
    are independent (see module docstring). See :func:`_pivotal_ci` for why
    the replicate distribution is reflected around the point rather than
    used directly."""
    n_rep = min(len(boot_a), len(boot_b))
    r_vals = np.full(n_rep, np.nan)
    cos_vals = np.full(n_rep, np.nan)
    for k in range(n_rep):
        a_k, b_k = boot_a[k][M], boot_b[k][M]
        finite = np.isfinite(a_k) & np.isfinite(b_k)
        if finite.sum() < min_joint_cells:
            continue
        a_f, b_f = a_k[finite], b_k[finite]
        if a_f.size >= 2 and np.std(a_f) > 0 and np.std(b_f) > 0:
            r_vals[k] = sp_stats.pearsonr(a_f, b_f).statistic
        cos_vals[k] = _cosine(a_f, b_f)
    with np.errstate(invalid="ignore"):
        r_ci = _pivotal_ci(point_r, r_vals) if np.isfinite(point_r) else (float("nan"), float("nan"))
        cos_ci = _pivotal_ci(point_cos, cos_vals) if np.isfinite(point_cos) else (float("nan"), float("nan"))
    return r_ci, cos_ci


def _sign_mask(leaf: dict) -> np.ndarray:
    """+1 where positive-marked (CI entirely above baseline), -1 where
    negative-marked (CI entirely below baseline), 0 elsewhere."""
    s = np.zeros(leaf["signif"].shape, dtype=np.int8)
    s[leaf["signif"]] = 1
    s[leaf["signif_neg"]] = -1
    return s


def _jaccard(leaf_a: dict, leaf_b: dict, M: np.ndarray) -> float:
    sa, sb = _sign_mask(leaf_a)[M], _sign_mask(leaf_b)[M]
    union = (sa != 0) | (sb != 0)
    if not union.any():
        return float("nan")
    same_sign = (sa == sb) & union
    return float(same_sign.sum() / union.sum())


def compare(boot_a: np.ndarray, leaf_a: dict, boot_b: np.ndarray, leaf_b: dict,
           min_joint_cells: int) -> SimilarityResult | None:
    """Full comparison (point + CI + Jaccard) between the same panel/bin's
    summary leaf and raw bootstrap-replicate array in two files. Returns
    ``None`` if fewer than ``min_joint_cells`` cells are reliably estimable
    in both — the falsifiability floor: a map comparison with too little
    shared domain cannot support a persistence claim either way."""
    M = leaf_a["reliable"] & leaf_b["reliable"]
    n_joint = int(M.sum())
    if n_joint < min_joint_cells:
        return None
    a_pt, b_pt = leaf_a["point"][M], leaf_b["point"][M]
    finite = np.isfinite(a_pt) & np.isfinite(b_pt)
    if finite.sum() < min_joint_cells:
        return None
    a_f, b_f = a_pt[finite], b_pt[finite]
    pearson_r = (float(sp_stats.pearsonr(a_f, b_f).statistic)
                if a_f.size >= 2 and np.std(a_f) > 0 and np.std(b_f) > 0 else float("nan"))
    cosine = _cosine(a_f, b_f)
    r_ci, cos_ci = _bootstrap_similarity_ci(boot_a, boot_b, M, min_joint_cells, pearson_r, cosine)
    jaccard = _jaccard(leaf_a, leaf_b, M)
    return SimilarityResult(n_joint=n_joint, pearson_r=pearson_r, pearson_ci=r_ci,
                            cosine=cosine, cosine_ci=cos_ci, jaccard=jaccard)


# ---------------------------------------------------------------------------
# Calibration verdict
# ---------------------------------------------------------------------------

def _verdict(test: SimilarityResult | None, seed_points: list,
            random_res: SimilarityResult | None) -> str:
    if test is None:
        return "n/a (too little shared domain)"
    if random_res is None or not np.isfinite(random_res.pearson_ci).all():
        return "n/a (no random baseline)"
    if not np.isfinite(test.pearson_ci).all():
        return "n/a (no test CI)"
    random_hi = random_res.pearson_ci[1]
    test_lo, test_hi = test.pearson_ci
    above_random = test_lo > random_hi
    seed_lo = min(seed_points) if seed_points else float("nan")
    in_seed_band = np.isfinite(seed_lo) and test.pearson_r >= seed_lo
    if above_random and in_seed_band:
        return "persists"
    if test_hi <= random_hi:
        return "indistinguishable from random"
    return "intermediate (above random, below seed-band)"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _fmt_ci(point: float, ci: tuple) -> str:
    if not np.isfinite(point):
        return "n/a"
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{point:.3f} [n/a]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def _row(label: str, res: SimilarityResult | None, verdict: str | None = None) -> str:
    if res is None:
        cells = [label, "n/a", "n/a", "n/a", "n/a"]
    else:
        cells = [
            label, str(res.n_joint),
            _fmt_ci(res.pearson_r, res.pearson_ci),
            _fmt_ci(res.cosine, res.cosine_ci),
            f"{res.jaccard:.3f}" if np.isfinite(res.jaccard) else "n/a",
        ]
    if verdict is not None:
        cells.append(verdict)
    return "| " + " | ".join(cells) + " |"


def _table(rows: list, with_verdict: bool) -> str:
    header = ["Comparison", "N joint cells", "Pearson r [95% CI]",
              "Cosine sim [95% CI]", "Jaccard (sign-matched)"]
    if with_verdict:
        header.append("Verdict")
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend(rows)
    return "\n".join(lines)


def _disambiguate_labels(paths: list) -> list:
    """Default label per path: the file stem (e.g. 'stats'), or
    '<parent_dir>/<stem>' when the bare stem collides with another path in
    *paths* — the common case here, since every run's stats file is named
    identically (stats.npz) inside its own per-run directory. Still not
    guaranteed unique (two paths could share both stem and parent dir name);
    the caller is responsible for a final hard uniqueness check."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{os.path.basename(os.path.dirname(os.path.abspath(p)))}/{s}"
           for p, s in zip(paths, stems)]


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class PersistenceContext:
    base_npz: str
    seeds: list
    random_npz: str
    test: list
    seed_labels: list
    test_labels: list

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

    n_bootstraps: int
    root_seed: int
    min_coverage: float
    min_joint_cells: int
    n_jobs: int

    out: str
    out_dir: str

    layers_idx: list = field(default=None, repr=False)
    total_layers: dict = field(default=None, repr=False)  # label -> L
    geom_edges: dict = field(default=None, repr=False)
    aa_dist_edges: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        all_paths = {"base": self.base_npz, "random": self.random_npz}
        for i, p in enumerate(self.seeds):
            all_paths[f"seed_{i}"] = p
        for i, p in enumerate(self.test):
            all_paths[f"test_{i}"] = p
        for name, p in all_paths.items():
            if not os.path.isfile(p):
                raise ValueError(f"{name} file not found: {p!r}")

        if not self.seeds:
            raise ValueError("at least one --seeds file is required")
        if not self.test:
            raise ValueError("at least one --test file is required")
        if self.seed_labels is not None and len(self.seed_labels) != len(self.seeds):
            raise ValueError("--seed_labels must match --seeds in count")
        if self.test_labels is not None and len(self.test_labels) != len(self.test):
            raise ValueError("--test_labels must match --test in count")
        # Default labels are derived from the file basename, which commonly
        # collides across runs in this project (every run's output is named
        # stats.npz, one per run directory) — fall back to "<parent_dir>/<stem>"
        # for exactly the paths whose bare stem is not unique within seeds+test.
        if self.seed_labels is None or self.test_labels is None:
            combined_paths = self.seeds + self.test
            combined_labels = _disambiguate_labels(combined_paths)
            if self.seed_labels is None:
                self.seed_labels = combined_labels[:len(self.seeds)]
            if self.test_labels is None:
                self.test_labels = combined_labels[len(self.seeds):]

        # Hard uniqueness check across EVERY label (base/random included, and
        # whether default- or user-supplied): `runs[label] = ...` is a dict
        # keyed by this label, so a collision would silently make one file's
        # results stand in for another's instead of failing loudly.
        all_labels = ["base"] + self.seed_labels + ["random"] + self.test_labels
        dupes = sorted({l for l in all_labels if all_labels.count(l) > 1})
        if dupes:
            raise ValueError(
                f"duplicate label(s) {dupes} among base/--seeds/--random/--test — "
                f"each input file needs a distinct label (results are keyed by "
                f"label, so a collision would silently overwrite one file's "
                f"results with another's). Pass explicit --seed_labels/"
                f"--test_labels to disambiguate.")

        out_parent = os.path.dirname(os.path.abspath(self.out)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(f"Parent directory of --out does not exist: {out_parent!r}")
        if self.out_dir is None:
            self.out_dir = out_parent

        self.total_layers = {}
        for name, p in all_paths.items():
            with np.load(p, allow_pickle=False) as raw:
                self.total_layers[name] = int(raw["L"])
        min_L = min(self.total_layers.values())
        if len(set(self.total_layers.values())) > 1:
            detail = ", ".join(f"{n}={l}" for n, l in self.total_layers.items())
            print(f"[persistence] note: input files disagree on encoder layer count "
                 f"({detail}) — defaulting/validating against the minimum ({min_L})")

        if self.layers is None:
            self.layers_idx = list(range(min_L))
        else:
            bad = [l for l in self.layers if l < 1 or l > min_L]
            if bad:
                raise ValueError(
                    f"--layers must be in [1, {min_L}] (the smallest encoder-layer "
                    f"count across every input file), got out-of-range value(s) {bad}")
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
        if self.n_bootstraps < 100:
            raise ValueError(f"n_bootstraps must be >= 100, got {self.n_bootstraps}")
        if not (0.0 < self.min_coverage <= 1.0):
            raise ValueError(f"min_coverage must be in (0, 1], got {self.min_coverage}")
        if self.min_joint_cells < 1:
            raise ValueError(f"min_joint_cells must be >= 1, got {self.min_joint_cells}")

        self.geom_edges = {
            "dist": np.linspace(self.dist_min, self.dist_max, self.geom_n_dist + 1),
            "theta": np.linspace(0.0, np.pi, self.geom_n_theta + 1),
            "phi": np.linspace(-np.pi, np.pi, self.geom_n_phi + 1),
        }
        self.aa_dist_edges = np.linspace(self.dist_min, self.dist_max, self.aa_n_dist_bins + 1)

    @classmethod
    def from_args(cls) -> "PersistenceContext":
        p = argparse.ArgumentParser(
            description=(
                "Falsifiable cross-condition persistence test: compares the "
                "geometric score map (raw and double-centered) and the "
                "double-centered AA-pair map against a reference condition, "
                "calibrated by between-seed similarity (upper bound) and "
                "trained-vs-untrained similarity (lower bound)."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument("base_npz", help="Reference condition .npz (produced by stats.py)")
        p.add_argument("--seeds", nargs="+", required=True, metavar="SEED.npz",
                       help="One or more same-condition replicate .npz files "
                            "(upper-bound calibration)")
        p.add_argument("--random", dest="random_npz", required=True, metavar="RANDOM.npz",
                       help="Untrained/random-init .npz file (lower-bound calibration)")
        p.add_argument("--test", nargs="+", required=True, metavar="TEST.npz",
                       help="One or more .npz files whose persistence is being tested")
        p.add_argument("--seed_labels", nargs="+", default=None,
                       help="Display label per --seeds file (default: basename)")
        p.add_argument("--test_labels", nargs="+", default=None,
                       help="Display label per --test file (default: basename)")
        p.add_argument("--layers", type=int, nargs="+", default=None, metavar="N",
                       help="1-indexed encoder layer(s) to analyse. Default: all "
                            "layers common to every input file.")

        d = p.add_argument_group("Shared distance-range filter")
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

        b = p.add_argument_group("Protein-level cluster bootstrap (independent per file)")
        b.add_argument("--n_bootstraps", type=int, default=2000)
        b.add_argument("--root_seed", type=int, default=12345)
        b.add_argument("--min_coverage", type=float, default=0.95)
        b.add_argument("--min_joint_cells", type=int, default=5,
                       help="Minimum cells reliably estimable in BOTH conditions "
                            "for a comparison to be reported (falsifiability floor)")
        b.add_argument("--n_jobs", type=int, default=-1)

        o = p.add_argument_group("Output")
        o.add_argument("--out", default="persistence_report.md")
        o.add_argument("--out_dir", default=None,
                       help="Figure output directory (default: next to --out)")

        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# Forest plot — Pearson r point + CI per condition vs the seed-band / random
# ---------------------------------------------------------------------------

def _plot_forest(map_name: str, key: str, seed_entries: list,
                 random_res, test_entries: list, path: str) -> None:
    rows = list(seed_entries) + [("random", random_res)] + list(test_entries)
    rows = [(label, res) for label, res in rows if res is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    seed_vals = [res.pearson_r for _, res in seed_entries if res is not None]
    if seed_vals:
        ax.axvspan(min(seed_vals), max(seed_vals), color="tab:green", alpha=0.15,
                  label="seed band")
    if random_res is not None:
        ax.axvline(random_res.pearson_r, color="tab:red", ls="--", lw=1.2, label="random")
    for yi, (label, res) in zip(y, rows):
        lo, hi = res.pearson_ci
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color="k", lw=1.5)
        ax.scatter([res.pearson_r], [yi], color="tab:blue", zorder=3, s=25)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Pearson r (vs. base)")
    ax.set_title(f"{map_name} — {key}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = PersistenceContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)

    labels = ["base"] + ctx.seed_labels + ["random"] + ctx.test_labels
    paths = [ctx.base_npz] + ctx.seeds + [ctx.random_npz] + ctx.test
    n_files = len(labels)

    print(f"[persistence] layers={[l + 1 for l in ctx.layers_idx]}  "
         f"files: base=1 seeds={len(ctx.seeds)} random=1 test={len(ctx.test)}")

    runs: dict[str, FileRun] = {}
    with Parallel(n_jobs=ctx.n_jobs, return_as="generator_unordered", max_nbytes="1M") as parallel:
        for i, (label, path) in enumerate(zip(labels, paths)):
            runs[label] = _run_file(label, path, ctx.layers_idx, ctx, parallel, i, n_files)

    md_sections = [
        "# Persistence report\n",
        f"Base: `{ctx.base_npz}`  \n"
        f"Seeds: {', '.join(f'`{p}`' for p in ctx.seeds)}  \n"
        f"Random: `{ctx.random_npz}`  \n"
        f"Test: {', '.join(f'`{p}`' for p in ctx.test)}  \n"
        f"n_bootstraps={ctx.n_bootstraps:,} (independent stream per file), "
        f"min_coverage={ctx.min_coverage:.0%}, "
        f"min_joint_cells={ctx.min_joint_cells}\n",
    ]

    def _section(layer_md: list, map_title: str, fig_tag: str, l: int,
                lookup_labels: list, summary_at: callable, boot_at: callable) -> None:
        for lookup_key, label_text in lookup_labels:
            summaries_by_label = {rl: summary_at(runs[rl], l, lookup_key) for rl in labels}
            boots_by_label = {rl: boot_at(runs[rl], l, lookup_key) for rl in labels}
            rows = []
            seed_entries = []
            for s_label in ctx.seed_labels:
                res = compare(boots_by_label["base"], summaries_by_label["base"],
                             boots_by_label[s_label], summaries_by_label[s_label],
                             ctx.min_joint_cells)
                rows.append(_row(f"base vs {s_label}", res))
                seed_entries.append((s_label, res))
            seed_points = [res.pearson_r for _, res in seed_entries if res is not None]
            random_res = compare(boots_by_label["base"], summaries_by_label["base"],
                                 boots_by_label["random"], summaries_by_label["random"],
                                 ctx.min_joint_cells)
            rows.append(_row("base vs random", random_res))
            test_entries = []
            for t_label in ctx.test_labels:
                test_res = compare(boots_by_label["base"], summaries_by_label["base"],
                                   boots_by_label[t_label], summaries_by_label[t_label],
                                   ctx.min_joint_cells)
                verdict = _verdict(test_res, seed_points, random_res)
                rows.append(_row(f"base vs {t_label}", test_res, verdict))
                test_entries.append((t_label, test_res))
            layer_md.append(f"### {map_title} — {label_text}\n\n"
                           + _table(rows, with_verdict=True) + "\n")
            fig_path = os.path.join(
                ctx.out_dir, f"fig_persistence_{fig_tag}_{lookup_key}_layer{l + 1}.svg")
            _plot_forest(map_title, label_text, seed_entries, random_res,
                        test_entries, fig_path)

    for l in ctx.layers_idx:
        layer_md = [f"## Layer {l + 1}"]
        _section(layer_md, "Geometric score map", "geom", l,
                 [(k, k) for k in _SCORE_KEYS],
                 summary_at=lambda run, l, key: run.summary["geom"][l][key],
                 boot_at=lambda run, l, key: run.boot["geom"][l][key])
        aa_labels = [(b, f"{ctx.aa_dist_edges[b]:.1f}–{ctx.aa_dist_edges[b + 1]:.1f} Å")
                    for b in range(ctx.aa_n_dist_bins)]
        _section(layer_md, "AA-pair map", "aa", l, aa_labels,
                 summary_at=lambda run, l, b: run.summary["aa"][l][b],
                 boot_at=lambda run, l, b: run.boot["aa"][l][b])
        md_sections.append("\n".join(layer_md))

    document = "\n\n".join(md_sections) + "\n"
    with open(ctx.out, "w") as f:
        f.write(document)
    print(f"[persistence] saved {ctx.out}")


if __name__ == "__main__":
    main()
