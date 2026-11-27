"""
enrichment_figures.py
======================
Regenerates the three figures that enrichment.ipynb was hand-rolling, by
calling the project's existing, already-tested vis.aa_heatmaps / vis.geometry
figure functions instead of re-deriving the pair/background ratio from
scratch:

  1. AA-pair score heatmap per Cα-Cα distance bin
     (vis.aa_heatmaps.fig_residue_type_heatmap_by_d3)
     — for every (res_type_i, res_type_j), ratio of that pair's mean score to
       its distance bin's overall mean score, log-scaled.
  2. Raw geometric score map over (d, theta, phi)
     (vis.geometry.fig_geometry_scatter)
     — descriptive: binned mean score, no residue-type filter.
  3. Edge-specific enrichment over (d, theta, phi)
     (vis.geometry.fig_geometry_enrichment)
     — log(pair mean score / background mean score), pair = the
       --source_aa/--neighbour_aa filter (default Cys->Cys), background = all
       edges passing the same seq_dist/distance filters.

All three use the same definition — log(group mean / reference mean) — with
no bootstrap or significance testing.

vis.aa_heatmaps._load_stats was fixed here to derive dist_3d from nbr_Ca:
the stats.py local-frame rework (see src/vis/geometry.py) dropped the flat
dist_3d/theta/phi arrays that aa_heatmaps.py used to read directly, so it
was raising KeyError on any current stats.npz before that fix.

Usage
-----
    python scripts/enrichment_figures.py \\
        models/exp2_finetuning/oa1_seed0/eval/stats.npz

    python scripts/enrichment_figures.py STATS.npz --out_dir enrichment \\
        --seq_dist_min 6 --source_aa C --neighbour_aa C
"""
from __future__ import annotations

import argparse
import os

from vis.aa_heatmaps import _load_stats as _load_aa_stats
from vis.aa_heatmaps import fig_residue_type_heatmap_by_d3
from vis.geometry import _load_stats as _load_geometry_stats
from vis.geometry import fig_geometry_enrichment, fig_geometry_scatter


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("stats_npz", help=".npz statistics file produced by stats.py")
    p.add_argument("--out_dir", default="enrichment", help="Output directory for figures")
    p.add_argument("--seq_dist_min", type=int, default=6,
                    help="Minimum |i-j| for edges included in every figure")

    g1 = p.add_argument_group("AA-pair heatmap (figure 1)")
    g1.add_argument("--d3_min", type=float, default=4.0)
    g1.add_argument("--d3_max", type=float, default=12.0)
    g1.add_argument("--d3_bins_aa", type=int, default=4,
                     help="Number of equally-spaced distance bins, e.g. 4 -> "
                          "[4,6), [6,8), [8,10), [10,12) A")

    g2 = p.add_argument_group("Geometric score/enrichment maps (figures 2-3)")
    g2.add_argument("--n_bins", type=int, default=80)
    g2.add_argument("--sigma", type=float, default=1.5,
                     help="Gaussian smoothing sigma (bins); 0 disables")
    g2.add_argument("--source_aa", nargs="+", default=["C"],
                     help="Source-residue one-letter codes for the edge-specific panel")
    g2.add_argument("--neighbour_aa", nargs="+", default=["C"],
                     help="Neighbour-residue one-letter codes for the edge-specific panel")

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/3] AA-pair heatmap by distance bin -> {args.out_dir}")
    aa_stats = _load_aa_stats(args.stats_npz)
    fig_residue_type_heatmap_by_d3(
        aa_stats, args.out_dir,
        n_d3_bins=args.d3_bins_aa, d3_min=args.d3_min, d3_max=args.d3_max,
        seq_dist_min=args.seq_dist_min,
    )

    geometry_stats = _load_geometry_stats(args.stats_npz)

    print(f"[2/3] Raw score map -> {args.out_dir}")
    fig_geometry_scatter(
        geometry_stats, args.out_dir,
        n_bins=args.n_bins, sigma=args.sigma, seq_dist_min=args.seq_dist_min,
    )

    print(f"[3/3] Edge-specific enrichment "
          f"({'/'.join(args.source_aa)} -> {'/'.join(args.neighbour_aa)}) -> {args.out_dir}")
    fig_geometry_enrichment(
        geometry_stats, args.out_dir,
        source_aa=args.source_aa, neighbour_aa=args.neighbour_aa,
        n_bins=args.n_bins, sigma=args.sigma, seq_dist_min=args.seq_dist_min,
    )


if __name__ == "__main__":
    main()
