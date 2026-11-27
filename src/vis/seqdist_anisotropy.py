"""
vis.seqdist_anisotropy — bootstrapped sequence-separation anisotropy
comparison across one or more models (OA checkpoints only).

For each input ``stats.npz`` (written by ``vis.stats``, tagged via
``--tags``) and each encoder layer, computes the per-edge rotation-
invariant anisotropic fraction

    aniso = sum_{l>=1} P_l / total_P,   P_l = sum_m c_{l,m}^2

— the fraction of an edge's spherical-harmonic spectral power (the OA
encoder's ``spectral_{l}`` coefficients) NOT carried by the direction-
independent l=0 term; same definition as ``vis.__main__.run_spectral``'s
"Anisotropic fraction (l>=1)/total" panel, but reported here as a mean +
95% CI per sequence-separation bin instead of a boxplot.

Edges are grouped into 6 sequence-separation bins — |i-j| = 1, 2, 3, 4, 5,
>=6 (the same classes as ``vis.__main__._SEQ_CLASSES``) — and, within each
bin, the mean anisotropic fraction is reported with a 95% CI from a
protein-level cluster bootstrap (``--n_bootstraps``, default 5000:
resampling whole proteins with replacement, not individual edges, since
edges from the same protein aren't independent observations). Every
layer of a given file shares one bootstrap (the same ``--n_bootstraps``
resampled protein sets), while different files each get their own
independent resampling stream (seeded off ``--root_seed``) — matching
``vis.__main__``/``vis.persistence``'s conventions, including the shared
``vis._bootstrap`` machinery (``BootData``/``build_layout``/
``run_bootstrap``) and joblib (``--n_jobs``) parallelization of the
bootstrap replicates.

Input files in 'gat'/'vanilla' mode (no spherical-harmonic coefficients)
are skipped with a warning, not an error — so a mixed set of ``--tags``
still runs over whichever files qualify.

Usage
-----
python -m vis.seqdist_anisotropy  oa1/stats.npz oa3/stats.npz gat3/stats.npz \\
    --tags oa1 oa3 gat3  --out seqdist_anisotropy.csv

Output
------
--out (default seqdist_anisotropy.csv): one row per (tag, layer, bin):
  tag, layer (1-indexed), seq_dist_bin ("1".."5"/"≥6"), n_edges,
  mean, ci_lo, ci_hi, coverage (fraction of bootstrap replicates in which
  the cell was estimable), reliable (coverage >= --min_coverage)

One comparison figure per analyzed layer (--out_dir, default --out's
directory): fig_seqdist_anisotropy_layer<N>.svg — mean +/- 95% CI per
sequence-separation bin, one point/errorbar series per tag.
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
from joblib import Parallel

from ._bootstrap import BootData, build_layout, grid_mean, run_bootstrap, _load_stats, _stack_tree

# Same 6 classes as vis.__main__._SEQ_CLASSES (lo, hi, label).
_SEQ_CLASSES = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 5, "5"),
               (6, np.inf, "≥6")]
_N_BINS = len(_SEQ_CLASSES)


def _seq_class_ids(seq_dist: np.ndarray) -> np.ndarray:
    """0-indexed _SEQ_CLASSES bin id per edge, or -1 if none match (never
    happens for a non-self edge's |i-j| >= 1, since the last class is
    [6, inf))."""
    out = np.full(seq_dist.shape, -1, dtype=np.int64)
    for k, (lo, hi, _) in enumerate(_SEQ_CLASSES):
        out[(seq_dist >= lo) & (seq_dist <= hi)] = k
    return out


def _l_max_from_n_sh(n_sh: int) -> int:
    l_max = int(round(n_sh ** 0.5)) - 1
    if (l_max + 1) ** 2 != n_sh:
        raise ValueError(f"n_sh={n_sh} is not a perfect square — cannot infer l_max")
    return l_max


def _anisotropic_fraction(spectral: np.ndarray, degree_idx: dict, l_max: int) -> np.ndarray:
    """Per-edge anisotropic fraction sum_{l>=1} P_l / total_P (see module
    docstring). NaN/inf for the (expected to be vanishingly rare, if ever)
    edge whose total spectral power is exactly 0."""
    P = {l: (spectral[:, idx] ** 2).sum(axis=1) for l, idx in degree_idx.items()}
    total_P = sum(P.values())
    aniso_num = sum(P[l] for l in range(1, l_max + 1)) if l_max >= 1 else np.zeros_like(total_P)
    with np.errstate(invalid="ignore", divide="ignore"):
        return aniso_num / total_P


# ---------------------------------------------------------------------------
# Per-replicate compute — the ``extra``/``compute_fn`` pair fed into
# ``vis._bootstrap.BootData``/``run_bootstrap``
# ---------------------------------------------------------------------------

@dataclass
class AnisoExtra:
    """One file's bootstrap replicate recipe, bundled into one explicit
    argument so joblib's loky backend can memory-map the large arrays
    instead of re-pickling them per task. ``cls_ids``/``aniso`` are keyed
    per layer (not just one shared array) so a layer whose anisotropic
    fraction happens to be non-finite for a few edges (see
    _anisotropic_fraction) excludes exactly those edges for THAT layer,
    not every layer."""
    cls_ids: dict
    aniso:   dict
    min_count: int
    layers:    list


def _aniso_bootstrap_compute(idx_pool: np.ndarray, extra: AnisoExtra) -> dict:
    """One replicate's {layer: (_N_BINS,) mean-anisotropic-fraction-per-bin}
    — used both for the point estimate (idx_pool = every edge) and for
    every bootstrap replicate."""
    return {
        l: grid_mean(extra.cls_ids[l][idx_pool], extra.aniso[l][idx_pool],
                    (_N_BINS,), extra.min_count)
        for l in extra.layers
    }


def _summarize(point_tree, boot_tree, n_requested: int, min_coverage: float):
    """Recurse into a (point, bootstrap) pair of matching {layer: array}
    dicts, returning at each leaf {point, lo, hi, coverage, reliable} —
    lo/hi are the 2.5/97.5 percentiles of the bootstrap distribution,
    reliable requires the cell to be estimable in at least min_coverage of
    n_requested replicates. (The data-agnostic vis._bootstrap._summarize_tree
    also flags "signif"/"signif_neg" against a baseline of 0 — meaningless
    for a fraction confined to [0, 1], so this is a local, simpler variant
    rather than that one reused as-is.)"""
    if isinstance(point_tree, dict):
        return {k: _summarize(point_tree[k], boot_tree[k], n_requested, min_coverage)
                for k in point_tree}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(boot_tree, 2.5, axis=0)
        hi = np.nanpercentile(boot_tree, 97.5, axis=0)
    coverage = np.sum(~np.isnan(boot_tree), axis=0) / n_requested
    reliable = coverage >= min_coverage
    return dict(point=point_tree, lo=lo, hi=hi, coverage=coverage, reliable=reliable)


# ---------------------------------------------------------------------------
# Per-file run
# ---------------------------------------------------------------------------

def _run_file(
    tag: str, path: str, layers: list, cfg: "AnisoContext", parallel: Parallel,
    file_index: int, n_files: int,
) -> dict:
    """Load one stats.npz and compute the point estimate + protein-level
    cluster-bootstrap summary of the mean anisotropic fraction per
    sequence-separation bin, for every layer in *layers*. Returns
    {"summary": {layer: {...}}, "n_edges": {layer: (_N_BINS,) int array},
    "mode": str, "l_max": int}."""
    stats = _load_stats(path, layers=layers)
    seq_dist = stats["seq_dist"]
    base_cls_ids = _seq_class_ids(seq_dist)

    n_sh = stats["n_sh"]
    l_max = _l_max_from_n_sh(n_sh)
    degrees = [l for l in range(l_max + 1) for _ in range(2 * l + 1)]
    degree_idx = {l: np.flatnonzero(np.array(degrees) == l) for l in range(l_max + 1)}

    aniso, cls_ids = {}, {}
    for l in layers:
        a = _anisotropic_fraction(stats["spectral"][l], degree_idx, l_max)
        bad = ~np.isfinite(a)
        if bad.any():
            print(f"  [{tag}] layer {l + 1}: {int(bad.sum())} edge(s) with non-finite "
                  f"anisotropic fraction (zero total spectral power) — excluded")
        aniso[l] = np.where(bad, 0.0, a)
        cls_ids[l] = np.where(bad, -1, base_cls_ids)

    _, protein_ids = np.unique(stats["protein_idx"], return_inverse=True)
    n_proteins = int(protein_ids.max()) + 1
    mask = np.ones(len(seq_dist), dtype=bool)
    flat_idx, starts, lengths = build_layout(mask, protein_ids, n_proteins)
    if flat_idx.size == 0:
        raise RuntimeError(f"{tag} ({path!r}): no edges to analyze")

    extra = AnisoExtra(cls_ids=cls_ids, aniso=aniso, min_count=cfg.min_count, layers=layers)
    bd = BootData(flat_idx=flat_idx, starts=starts, lengths=lengths,
                  n_proteins=n_proteins, extra=extra)

    point = _aniso_bootstrap_compute(flat_idx, extra)

    # Independent resampling stream per file: a child SeedSequence per file
    # index, each spawning its own --n_bootstraps draws — reproducible from
    # one --root_seed, but not the same draws across files (every layer of
    # THIS file does share its one stream — see module docstring).
    file_seed_seq = np.random.SeedSequence(cfg.root_seed).spawn(n_files)[file_index]
    seeds = file_seed_seq.spawn(cfg.n_bootstraps)
    n_tasks = min(cfg.n_bootstraps, 4 * (os.cpu_count() or 1))
    batch_size = -(-cfg.n_bootstraps // n_tasks)
    results, n_none = run_bootstrap(parallel, seeds, bd, batch_size, _aniso_bootstrap_compute)
    if not results:
        raise RuntimeError(f"{tag} ({path!r}): every bootstrap replicate was empty")
    if n_none:
        print(f"  [{tag}]   {n_none}/{cfg.n_bootstraps} empty replicates")

    boot = _stack_tree(results)
    summary = _summarize(point, boot, cfg.n_bootstraps, cfg.min_coverage)

    n_edges = {l: np.array([int((cls_ids[l] == k).sum()) for k in range(_N_BINS)])
              for l in layers}

    print(f"  [{tag}] proteins={n_proteins:,}  edges={len(seq_dist):,}  "
          f"usable replicates={len(results):,}/{cfg.n_bootstraps:,}  "
          f"mode={stats['mode']!r}  l_max={l_max}")

    return dict(summary=summary, n_edges=n_edges, mode=stats["mode"], l_max=l_max)


# ---------------------------------------------------------------------------
# Report + figure
# ---------------------------------------------------------------------------

def _print_report(runs: dict, layers_idx: list) -> None:
    labels = [lab for *_, lab in _SEQ_CLASSES]
    for tag, res in runs.items():
        for l in layers_idx:
            s, ne = res["summary"][l], res["n_edges"][l]
            print(f"\n  [{tag}]  layer {l + 1}  (mode={res['mode']!r}, l_max={res['l_max']})")
            print(f"    {'bin':>4}  {'n_edges':>9}  {'mean':>7}  {'95% CI':>17}   reliable")
            for k, lab in enumerate(labels):
                mean, lo, hi = s["point"][k], s["lo"][k], s["hi"][k]
                print(f"    {lab:>4}  {ne[k]:>9,}  {mean:7.4f}  [{lo:7.4f}, {hi:7.4f}]   "
                      f"{bool(s['reliable'][k])!s:>5}")


def _save_table(runs: dict, layers_idx: list, out_csv: str) -> str:
    labels = [lab for *_, lab in _SEQ_CLASSES]
    rows = []
    for tag, res in runs.items():
        for l in layers_idx:
            s, ne = res["summary"][l], res["n_edges"][l]
            for k, lab in enumerate(labels):
                rows.append(dict(
                    tag=tag, layer=l + 1, seq_dist_bin=lab,
                    n_edges=int(ne[k]), mean=float(s["point"][k]),
                    ci_lo=float(s["lo"][k]), ci_hi=float(s["hi"][k]),
                    coverage=float(s["coverage"][k]), reliable=bool(s["reliable"][k]),
                ))
    df = pd.DataFrame(rows, columns=[
        "tag", "layer", "seq_dist_bin", "n_edges", "mean", "ci_lo", "ci_hi",
        "coverage", "reliable",
    ])
    df.to_csv(out_csv, index=False)
    return out_csv


def _plot_comparison(runs: dict, layer: int, path: str) -> None:
    """Mean +/- 95% CI anisotropic fraction per sequence-separation bin,
    one errorbar series per tag that has this layer."""
    labels = [lab for *_, lab in _SEQ_CLASSES]
    x = np.arange(_N_BINS, dtype=float)
    tags = [t for t, res in runs.items() if layer in res["summary"]]
    width = 0.8 / max(len(tags), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, tag in enumerate(tags):
        s = runs[tag]["summary"][layer]
        offs = x + (i - (len(tags) - 1) / 2) * width
        err = np.stack([s["point"] - s["lo"], s["hi"] - s["point"]])
        ax.errorbar(offs, s["point"], yerr=err, fmt="o", capsize=3, label=tag, markersize=5)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Sequence separation |i−j|")
    ax.set_ylabel(r"Anisotropic fraction  ($\ell \geq 1$ power / total)")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"Layer {layer + 1} — sequence-separation anisotropy")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class AnisoContext:
    npz_files: list
    tags:      list
    layers:    list

    min_count:     int
    n_bootstraps:  int
    root_seed:     int
    min_coverage:  float
    n_jobs:        int

    out_csv: str
    out_dir: str = None

    def __post_init__(self) -> None:
        if not self.npz_files:
            raise ValueError("at least one stats.npz file is required")
        if len(self.tags) != len(self.npz_files):
            raise ValueError(
                f"--tags must have the same length as the input files "
                f"({len(self.npz_files)}), got {len(self.tags)}")
        dupes = sorted({t for t in self.tags if self.tags.count(t) > 1})
        if dupes:
            raise ValueError(f"--tags must be unique, duplicate(s): {dupes}")
        for path in self.npz_files:
            if not os.path.isfile(path):
                raise ValueError(f"file not found: {path!r}")

        out_parent = os.path.dirname(os.path.abspath(self.out_csv)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(f"Parent directory of --out does not exist: {out_parent!r}")
        if self.out_dir is None:
            self.out_dir = out_parent
        elif not os.path.isdir(self.out_dir):
            raise ValueError(f"--out_dir does not exist: {self.out_dir!r}")

    @classmethod
    def from_args(cls) -> "AnisoContext":
        p = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        p.add_argument("npz_files", nargs="+",
                       help="stats.npz file(s) written by vis.stats (OA-mode checkpoints "
                            "only; 'gat'/'vanilla' files are skipped with a warning)")
        p.add_argument("--tags", nargs="+", required=True,
                       help="Display label per input file, same order and count as "
                            "npz_files (e.g. --tags oa1 oa3 gat3)")
        p.add_argument("--layers", type=int, nargs="+", default=None, metavar="N",
                       help="1-indexed encoder layer(s) to analyze. Default: all layers "
                            "common to every applicable (OA-mode) input file.")
        p.add_argument("--min_count", type=int, default=20,
                       help="Minimum edges in a (tag, layer, bin) cell to report a mean "
                            "(else NaN)")
        p.add_argument("--n_bootstraps", type=int, default=5000,
                       help="Protein-level cluster-bootstrap replicates")
        p.add_argument("--root_seed", type=int, default=12345,
                       help="Seed for the (per-file independent) bootstrap resampling streams")
        p.add_argument("--min_coverage", type=float, default=0.95,
                       help="Minimum fraction of bootstrap replicates in which a cell "
                            "must be estimable for its CI to be flagged 'reliable'")
        p.add_argument("--n_jobs", type=int, default=-1, help="joblib worker processes")
        p.add_argument("--out", dest="out_csv", default="seqdist_anisotropy.csv",
                       help="Output CSV path")
        p.add_argument("--out_dir", default=None,
                       help="Output directory for the comparison figures "
                            "(default: --out's directory)")
        args = p.parse_args()
        return cls(
            npz_files=args.npz_files, tags=args.tags, layers=args.layers,
            min_count=args.min_count, n_bootstraps=args.n_bootstraps,
            root_seed=args.root_seed, min_coverage=args.min_coverage, n_jobs=args.n_jobs,
            out_csv=args.out_csv, out_dir=args.out_dir,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = AnisoContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)

    # Peek mode/L/spectral-presence per file without loading the big arrays,
    # so a 'gat'/'vanilla' input (no spherical-harmonic coefficients) can be
    # skipped with a clear message instead of failing deep inside _load_stats.
    valid = []
    for tag, path in zip(ctx.tags, ctx.npz_files):
        with np.load(path, allow_pickle=False) as raw:
            mode = str(raw["mode"])
            L = int(raw["L"])
            has_spectral = "spectral_0" in raw
        if mode != "oa" or not has_spectral:
            print(f"[seqdist_anisotropy] skipping {tag!r} ({path!r}) — mode={mode!r}, "
                  f"no spherical-harmonic spectral coefficients to analyze")
            continue
        valid.append((tag, path, L))
    if not valid:
        raise SystemExit("no input file has spherical-harmonic spectral coefficients "
                         "(mode='oa' checkpoints only) — nothing to analyze")

    total_layers = {tag: L for tag, _, L in valid}
    min_L = min(total_layers.values())
    if len(set(total_layers.values())) > 1:
        detail = ", ".join(f"{t}={l}" for t, l in total_layers.items())
        print(f"[seqdist_anisotropy] note: input files disagree on encoder layer count "
              f"({detail}) — defaulting/validating against the minimum ({min_L})")

    if ctx.layers is None:
        layers_idx = list(range(min_L))
    else:
        bad = [l for l in ctx.layers if l < 1 or l > min_L]
        if bad:
            raise SystemExit(
                f"--layers must be in [1, {min_L}] (the smallest encoder-layer count "
                f"across every applicable input file), got out-of-range value(s) {bad}")
        layers_idx = sorted({l - 1 for l in ctx.layers})

    print(f"[seqdist_anisotropy] layers={[l + 1 for l in layers_idx]}  "
          f"files={len(valid)}/{len(ctx.npz_files)}  n_bootstraps={ctx.n_bootstraps:,}  "
          f"bins={[lab for *_, lab in _SEQ_CLASSES]}")

    n_files = len(valid)
    runs: dict = {}
    with Parallel(n_jobs=ctx.n_jobs, return_as="generator_unordered", max_nbytes="1M") as parallel:
        for i, (tag, path, _) in enumerate(valid):
            runs[tag] = _run_file(tag, path, layers_idx, ctx, parallel, i, n_files)

    _print_report(runs, layers_idx)

    print(f"\n  table -> {_save_table(runs, layers_idx, ctx.out_csv)}")
    for l in layers_idx:
        path = os.path.join(ctx.out_dir, f"fig_seqdist_anisotropy_layer{l + 1}.svg")
        _plot_comparison(runs, l, path)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
