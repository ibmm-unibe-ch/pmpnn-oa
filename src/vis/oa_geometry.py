"""
pmpnn-oa-geometry.py — (d, θ, φ) geometry maps and volumetric export for
ProteinMPNN-OA.

Reads the .npz statistics file produced by stats.py (OA mode required) and
generates:

  fig_geometry_scatter_layerN.svg  — lower-triangular 3×3 pairs plot of the
                                      geometric axes (d, θ, φ).  Off-diagonal
                                      cells show the mean budget-weighted score
                                      per 2-D bin (log scale); diagonal cells
                                      show marginal histograms.  One figure per
                                      encoder layer.

  geometry_layerN_density.mrc      — voxelised edge-count in the local backbone
  geometry_layerN_score.mrc          frame [−r_max, +r_max]³.
  geometry_layerN_score_high.mrc   — score restricted to regions where score > 1
                                      (above-baseline attention).
  geometry_layerN_score_low.mrc    — score restricted to regions where score < 1
                                      (below-baseline attention).
  geometry_layerN_chimerax.cxc     — ChimeraX session script that opens all four
                                      volumes for the layer and applies surface
                                      rendering.
  geometry_reference_residue.pdb   — canonical ALA backbone in the local frame
                                      (origin = source Cα, x → e₁ backbone
                                      direction, y → e₂, z → e₃).  Open
                                      together with the volumes in ChimeraX for
                                      orientation.

When ``--source_aa`` / ``--neighbour_aa`` are supplied the script additionally
filters all outputs to the specified (source AA, neighbour AA) pair and produces:

  fig_geometry_scatter_<src>_<nbr>_layerN.svg
                                   — same scatter figure restricted to the pair.
  fig_geometry_enrichment_<src>_<nbr>_layerN.svg
                                   — enrichment scatter: log(pair score / background
                                     score) per bin; diagonal panels overlay pair
                                     and background marginal distributions.

  geometry_<src><nbr>_layerN_density.mrc
  geometry_<src><nbr>_layerN_score.mrc
  geometry_<src><nbr>_layerN_score_high.mrc
  geometry_<src><nbr>_layerN_score_low.mrc
  geometry_<src><nbr>_layerN_enrichment.mrc
                                   — volumetric files for the pair subset;
                                     enrichment = log(mean_score_pair /
                                     mean_score_background) per voxel.
  geometry_<src><nbr>_layerN_enrichment_high.mrc
                                   — enrichment restricted to voxels where
                                     enrichment > 0 (pair is enriched vs
                                     background).
  geometry_<src><nbr>_layerN_enrichment_low.mrc
                                   — enrichment restricted to voxels where
                                     enrichment < 0 (pair is depleted vs
                                     background).
  geometry_<src><nbr>_layerN_chimerax.cxc
                                   — ChimeraX session script including all
                                     enrichment volumes.

No model weights or forward passes are required — all geometry is read from the
pre-computed .npz file.

This script only processes OA-mode statistics files.  If the .npz was produced
with ``mode=gat`` (no theta / phi arrays), the script exits with a clear error.

Usage
-----
python oa_geometry.py  stats.npz  [options]

Examples
--------
# Minimal — all defaults
python oa_geometry.py  stats.npz

# Custom output directory and resolution
python oa_geometry.py  stats.npz  --out_dir figs/geometry  --vol_n_bins 80

# Exclude local contacts (|i−j| < 6)
python oa_geometry.py  stats.npz  --seq_dist_min 6

# Tight score range and fine grid
python oa_geometry.py  stats.npz  \\
    --score_vmin 0.6  --score_vmax 1.8  \\
    --vol_n_bins 96  --r_max 18.0

# Skip volumetric export (figures only)
python oa_geometry.py  stats.npz  --no_vol

# Pair-specific map and enrichment (Cys → Cys)
python oa_geometry.py  stats.npz  --source_aa C  --neighbour_aa C

# Pair-specific, figures only
python oa_geometry.py  stats.npz  --source_aa C  --neighbour_aa C  --no_vol
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


# ---------------------------------------------------------------------------
# Amino-acid alphabet
# ---------------------------------------------------------------------------

_AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX    = {aa: i for i, aa in enumerate(_AA_ALPHABET)}


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class GeometryContext:
    """All configuration needed to generate the (d, θ, φ) geometry outputs.

    CLI-facing fields are validated in :meth:`__post_init__`; the class can
    also be instantiated programmatically for use in notebooks or tests.
    """

    # ── I/O ───────────────────────────────────────────────────────────────────
    stats_npz: str
    out_dir:   str

    # ── geometry scatter — score colorscale ───────────────────────────────────
    score_vmin: float       # lower bound in score units
    score_vmax: float       # upper bound in score units

    # ── geometry scatter — binning ────────────────────────────────────────────
    n_bins: int             # number of 2-D histogram bins per axis
    sigma:  float           # Gaussian smoothing σ (bins) for score maps

    # ── geometry scatter — distance range filter ──────────────────────────────
    d_min: float            # minimum Cα–Cα distance to include (Å)
    d_max: float            # maximum Cα–Cα distance to include (Å)

    # ── sequence-distance filter ───────────────────────────────────────────────
    seq_dist_min: int       # minimum |i − j|; mirrors spectral analysis default

    # ── residue-pair filter (optional) ────────────────────────────────────────
    source_aa:    str | None   # one-letter code for the source residue, or None
    neighbour_aa: str | None   # one-letter code for the neighbour residue, or None

    # ── volumetric export ─────────────────────────────────────────────────────
    export_vol: bool        # whether to write MRC + CXC files
    r_max:      float       # half-edge of the output cube in Å
    vol_n_bins: int         # voxels per axis

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
        if self.score_vmin <= 0:
            raise ValueError(
                f"score_vmin must be > 0 (score units), got {self.score_vmin}")
        if self.score_vmax <= self.score_vmin:
            raise ValueError(
                f"score_vmax must be > score_vmin, "
                f"got vmin={self.score_vmin}  vmax={self.score_vmax}")

        if self.n_bins < 4:
            raise ValueError(f"n_bins must be >= 4, got {self.n_bins}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")

        if self.d_min < 0:
            raise ValueError(f"d_min must be >= 0, got {self.d_min}")
        if self.d_max <= self.d_min:
            raise ValueError(
                f"d_max must be > d_min, "
                f"got d_min={self.d_min}  d_max={self.d_max}")

        if self.seq_dist_min < 1:
            raise ValueError(
                f"seq_dist_min must be >= 1 (self-edges always excluded), "
                f"got {self.seq_dist_min}")

        # --- residue-pair validation ------------------------------------------
        if self.source_aa is not None:
            self.source_aa = self.source_aa.upper()
            if self.source_aa not in _AA_INDEX:
                raise ValueError(
                    f"source_aa must be a one-letter amino-acid code, "
                    f"got {self.source_aa!r}.  "
                    f"Valid codes: {''.join(_AA_ALPHABET)}"
                )
        if self.neighbour_aa is not None:
            self.neighbour_aa = self.neighbour_aa.upper()
            if self.neighbour_aa not in _AA_INDEX:
                raise ValueError(
                    f"neighbour_aa must be a one-letter amino-acid code, "
                    f"got {self.neighbour_aa!r}.  "
                    f"Valid codes: {''.join(_AA_ALPHABET)}"
                )

        if self.r_max <= 0:
            raise ValueError(f"r_max must be > 0, got {self.r_max}")
        if self.vol_n_bins < 8:
            raise ValueError(f"vol_n_bins must be >= 8, got {self.vol_n_bins}")

        # --- load and validate statistics ------------------------------------
        self.stats = _load_stats(self.stats_npz)

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_args(cls) -> "GeometryContext":
        """Parse sys.argv and return a validated :class:`GeometryContext`."""
        p = argparse.ArgumentParser(
            description=(
                "Generate (d, θ, φ) geometry maps and volumetric files from "
                "ProteinMPNN-OA statistics produced by stats.py.  "
                "OA-mode .npz required (theta / phi arrays must be present)."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # ── positional ────────────────────────────────────────────────────────
        p.add_argument(
            "stats_npz",
            help=".npz statistics file produced by stats.py (mode=oa)",
        )

        # ── output ────────────────────────────────────────────────────────────
        p.add_argument(
            "--out_dir", default="geometry",
            help="Output directory for figure and volume files (created if absent)",
        )

        # ── geometry scatter ──────────────────────────────────────────────────
        g = p.add_argument_group("Geometry scatter — score colorscale")
        g.add_argument(
            "--score_vmin", type=float, default=0.5,
            help="Lower bound for the score colorscale in the scatter figures "
                 "(score units; 1 = uniform baseline)",
        )
        g.add_argument(
            "--score_vmax", type=float, default=2.0,
            help="Upper bound for the score colorscale (score units)",
        )

        gb = p.add_argument_group("Geometry scatter — binning")
        gb.add_argument(
            "--n_bins", type=int, default=80,
            help="Number of uniform bins per axis in the 2-D score maps",
        )
        gb.add_argument(
            "--sigma", type=float, default=1.5,
            help="Gaussian smoothing σ (in bins) applied to the 2-D score maps "
                 "and density contours.  Set to 0 to disable smoothing.",
        )

        gd = p.add_argument_group("Geometry scatter — distance range filter")
        gd.add_argument(
            "--d_min", type=float, default=0.0,
            help="Minimum Cα–Cα distance (Å) for edges included in the figures "
                 "and volumes",
        )
        gd.add_argument(
            "--d_max", type=float, default=float("inf"),
            help="Maximum Cα–Cα distance (Å) for edges included.  "
                 "Default: no upper limit",
        )

        # ── sequence-distance filter ──────────────────────────────────────────
        sd = p.add_argument_group("Sequence-distance filter")
        sd.add_argument(
            "--seq_dist_min", type=int, default=1,
            help="Minimum sequential separation |i − j| for edges included in "
                 "all outputs.  Default 1 keeps all non-self edges.  "
                 "Set to e.g. 6 to restrict to medium- and long-range contacts "
                 "(mirrors --seq_dist_min in pmpnn-oa-spectral-analysis.py).",
        )

        # ── residue-pair filter ───────────────────────────────────────────────
        pr = p.add_argument_group(
            "Residue-pair filter",
            "When source_aa and/or neighbour_aa are given, all scatter figures "
            "and volumes are additionally restricted to the specified "
            "(source AA → neighbour AA) pair.  Pair-specific scatter figures "
            "and an enrichment map vs the full background are produced alongside "
            "the unfiltered outputs.",
        )
        pr.add_argument(
            "--source_aa", default=None, metavar="AA",
            help="One-letter code for the source (central) residue "
                 "(e.g. C for cysteine).  Case-insensitive.",
        )
        pr.add_argument(
            "--neighbour_aa", default=None, metavar="AA",
            help="One-letter code for the neighbour residue.  "
                 "Case-insensitive.",
        )

        # ── volumetric export ─────────────────────────────────────────────────
        v = p.add_argument_group("Volumetric export (MRC + ChimeraX CXC + PDB)")
        v.add_argument(
            "--no_vol", action="store_false", dest="export_vol",
            help="Skip volumetric export; generate scatter figures only",
        )
        p.set_defaults(export_vol=True)
        v.add_argument(
            "--r_max", type=float, default=20.0,
            help="Half-edge of the cubic [−r_max, +r_max]³ volumetric box (Å).  "
                 "Should match r_max_orb used when training the model.",
        )
        v.add_argument(
            "--vol_n_bins", type=int, default=64,
            help="Number of uniform voxels per axis "
                 "(voxel ≈ 2 · r_max / vol_n_bins Å)",
        )

        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# Statistics loading  (mirrors _load_stats from pmpnn-oa-heatmaps.py)
# ---------------------------------------------------------------------------

def _load_stats(path: str) -> dict:
    """
    Load a .npz file produced by stats.py and reconstruct per-layer lists.

    Raises
    ------
    ValueError
        If the file was produced in GAT mode (no theta / phi arrays present).

    Returns
    -------
    dict with keys (OA mode):
      ``L``                — int, number of encoder layers
      ``mode``             — str, "oa"
      ``theta``            — float32 [total_non_self_edges]
      ``phi``              — float32 [total_non_self_edges]
      ``dist_3d_ns``       — float32 [total_non_self_edges]
      ``seq_dist_ns``      — float32 [total_non_self_edges]
      ``res_type_i_ns``    — int8    [total_non_self_edges]
      ``res_type_j_ns``    — int8    [total_non_self_edges]
      ``scored_attn_ns``   — list of L float32 arrays [total_non_self_edges]
      (plus all other flat keys from the .npz)
    """
    raw = np.load(path, allow_pickle=False)
    L   = int(raw["L"])
    mode = str(raw["mode"])

    if "theta" not in raw:
        raise ValueError(
            f"stats file {path!r} does not contain 'theta' / 'phi' arrays.  "
            f"Only OA-mode statistics are supported by this script "
            f"(mode found: {mode!r}).  "
            f"Re-run stats.py with --mode oa."
        )

    stats: dict = dict(raw)

    # Rebuild the per-layer list arrays
    for key in (
        "neff", "alpha_self", "argmax_seq_dist", "argmax_dist_3d",
        "attn", "rel_attn", "budget", "scored_attn",
        "mean_score", "spectral", "scored_attn_ns",
    ):
        layer_arrays = []
        for l in range(L):
            tag = f"{key}_{l}"
            if tag in raw:
                layer_arrays.append(raw[tag])
        if layer_arrays:
            stats[key] = layer_arrays

    stats["L"]    = L
    stats["mode"] = mode

    # Boolean flag for downstream convenience
    if "is_self" in raw:
        stats["is_self"] = raw["is_self"].astype(bool)

    return stats


# ---------------------------------------------------------------------------
# Cartesian conversion  (mirrors pmpnn-oa-spectral-analysis.py)
# ---------------------------------------------------------------------------

def _to_cartesian(
    dist: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical local-frame coordinates to Cartesian.

    Convention (matches stats.py accumulate()):
        theta = acos(d · e₁)       polar angle   [0, π]
        phi   = atan2(d · e₂, d · e₃)  azimuthal angle  [−π, π]

    Returns
    -------
    x, y, z : Cartesian arrays in Å (same length as inputs)
        x → e₁  (backbone direction)
        y → e₂
        z → e₃
    """
    sin_t = np.sin(theta)
    x = dist * np.cos(theta)
    y = dist * sin_t * np.sin(phi)
    z = dist * sin_t * np.cos(phi)
    return x, y, z


# ---------------------------------------------------------------------------
# Volumetric helpers  (mirrors pmpnn-oa-spectral-analysis.py)
# ---------------------------------------------------------------------------

def _cartesian_edges(r_max: float, n_bins: int) -> list[np.ndarray]:
    """Uniform bin edges for a symmetric cube [−r_max, +r_max]³."""
    e = np.linspace(-r_max, r_max, n_bins + 1)
    return [e, e, e]


def _bin_cartesian(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    scores: np.ndarray,
    edges: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin Cartesian edge positions onto a pre-computed edge grid.

    Returns
    -------
    count      : float32 [nx, ny, nz]  — edges per voxel
    mean_score : float32 [nx, ny, nz]  — mean score (NaN where empty)
    """
    from scipy.stats import binned_statistic_dd

    sample = np.column_stack([x, y, z])
    count, _, _  = binned_statistic_dd(sample, scores,
                                       statistic="count", bins=edges)
    mean_s, _, _ = binned_statistic_dd(sample, scores,
                                       statistic="mean",  bins=edges)
    return count.astype(np.float32), mean_s.astype(np.float32)


def _apply_log_score_band(
    mean_score: np.ndarray,
    above_baseline: bool,
) -> np.ndarray:
    """
    Return a score band masked and transformed onto a common **positive**
    log scale, mirroring the convention used for the enrichment bands.

    Both the high and low volumes are stored as positive values so that a
    single surface level (``eps``) renders both at a comparable intensity in
    ChimeraX / VMD:

      above_baseline=True  (score > 1):
          stored value = log(score)           > 0
      above_baseline=False (0 < score < 1):
          stored value = −log(score) = log(1/score)  > 0

    Parameters
    ----------
    mean_score     : raw linear score grid; NaN where empty.
    above_baseline : selects the high (> 1) or low (< 1) band.

    Returns
    -------
    float32 array with the same shape as mean_score.  Masked voxels are 0.
    """
    score = np.nan_to_num(mean_score, nan=0.0)
    if above_baseline:
        return np.where(score > 1.0,
                        np.log(np.where(score > 1.0, score, 1.0)),
                        0.0).astype(np.float32)
    else:
        return np.where((score > 0.0) & (score < 1.0),
                        -np.log(np.where((score > 0.0) & (score < 1.0), score, 1.0)),
                        0.0).astype(np.float32)


def _write_mrc(
    path: str,
    grid: np.ndarray,
    r_max: float,
    label: str = "volume",
) -> None:
    """
    Write a 3-D float array to MRC2014 format (ChimeraX / VMD).

    ``grid`` is expected as [nx, ny, nz].  MRC2014 stores data in [nz, ny, nx]
    order, so we transpose before writing.  The voxel_size and origin are in Å.

    Parameters
    ----------
    path   : output file path
    grid   : [nx, ny, nz] float array (NaN treated as 0)
    r_max  : half-edge of the cube (Å); origin = −r_max on each axis
    label  : informational string stored in the MRC extended header / title
    """
    try:
        import mrcfile
    except ImportError:
        raise ImportError(
            "mrcfile is required for MRC export.  "
            "Install it with:  pip install mrcfile"
        )

    n      = grid.shape[0]          # cube: nx == ny == nz
    voxel  = 2.0 * float(r_max) / n
    origin = -float(r_max)

    # MRC expects [nz, ny, nx]; our grid is [nx, ny, nz]
    data_mrc = np.nan_to_num(grid, nan=0.0).astype(np.float32).transpose(2, 1, 0)

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data_mrc)
        mrc.voxel_size = (voxel, voxel, voxel)   # isotropic Å spacing
        mrc.header.origin.x = origin
        mrc.header.origin.y = origin
        mrc.header.origin.z = origin
        mrc.update_header_stats()


def _write_reference_pdb(path: str) -> None:
    """
    Write a PDB file containing the canonical ALA backbone positioned in the
    ProteinMPNN-OA local backbone reference frame used by all volumetric files.

    Frame convention (mirrors the computation in stats.py / model.py):
        e1 = normalize(Cα − N)      [x-axis, pointing toward C=O face]
        e2 = normalize(e1 × (C − Cα))
        e3 = e1 × e2
        Cα at origin;  θ = 0 points toward the carbonyl face.
    """
    _PDB_CONTENT = """\
REMARK    Alanine in the ProteinMPNN-OA local backbone reference frame
REMARK    Frame convention (script): e1 = normalize(Ca - N)  [x, toward C=O]
REMARK                               e2 = normalize(e1 x (C - Ca))
REMARK                               e3 = e1 x e2
REMARK    Ca at origin. theta=0 points toward carbonyl face.
HETATM    1  N   ALA A   1      -1.471  -0.000   0.000  1.00  0.00           N
HETATM    2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    3  C   ALA A   1       0.509  -0.000  -1.438  1.00  0.00           C
HETATM    4  O   ALA A   1       1.856   0.001  -1.688  1.00  0.00           O
HETATM    5  CB  ALA A   1       0.508  -1.245   0.720  1.00  0.00           C
CONECT    2    1    3    5
CONECT    3    4
END
"""
    with open(path, "w") as f:
        f.write(_PDB_CONTENT)


# ---------------------------------------------------------------------------
# Figure — (d, θ, φ) geometry scatter
# ---------------------------------------------------------------------------

def fig_geometry_scatter(
    stats: dict,
    out_dir: str,
    score_vmin: float = 0.5,
    score_vmax: float = 2.0,
    n_bins: int = 80,
    sigma: float = 1.5,
    seq_dist_min: int = 1,
    d_min: float = 0.0,
    d_max: float = float("inf"),
) -> None:
    """
    Lower-triangular 3×3 pairs plot of the geometric axes (d, θ, φ).

    Off-diagonal cells show the binned mean budget-weighted score on a
    log-scale diverging colormap centred at 1 (= uniform-baseline).  Density
    contours at the 50th, 75th, 90th and 99th percentiles are overlaid in
    gold.  Diagonal cells show marginal histograms.

    One figure per encoder layer is saved as:
        ``<out_dir>/fig_geometry_scatter_layer{N}.svg``

    Parameters
    ----------
    stats       : dict produced by :func:`_load_stats`.
    out_dir     : output directory.
    score_vmin  : colorscale lower bound in score units.
    score_vmax  : colorscale upper bound in score units.
    n_bins      : number of uniform 2-D bins per axis.
    sigma       : Gaussian smoothing σ (bins); 0 disables smoothing.
    seq_dist_min: minimum |i − j| for edges included in the figures.
    d_min, d_max: distance range filter (Å).
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import binned_statistic_2d

    L = stats["L"]

    # ── apply filters ─────────────────────────────────────────────────────────
    dist = stats["dist_3d_ns"].astype(np.float32)
    seq  = stats["seq_dist_ns"].astype(np.float32)

    mask = (seq >= seq_dist_min) & (dist >= d_min) & (dist <= d_max)

    dist  = dist[mask]
    theta = stats["theta"][mask]
    phi   = stats["phi"][mask]

    if len(dist) == 0:
        print("[geometry] WARNING: no edges survive the filters — "
              "skipping scatter figures")
        return

    axes_vals   = [dist, theta, phi]
    axes_labels = [
        "Cα–Cα distance $d$ (Å)",
        "Polar angle $\\theta$ (rad)  $[0, \\pi]$",
        "Azimuthal angle $\\phi$ (rad)  $[-\\pi, \\pi]$",
    ]
    n_axes = 3

    # ── shared log-score colorscale ───────────────────────────────────────────
    log_vmin = min(np.log(max(score_vmin, 1e-9)), -1e-6)
    log_vmax = max(np.log(max(score_vmax, 1e-9)),  1e-6)
    score_norm      = mcolors.TwoSlopeNorm(vmin=log_vmin, vcenter=0.0, vmax=log_vmax)
    score_raw_ticks = np.linspace(log_vmin, log_vmax, 7)

    # ── one figure per layer ──────────────────────────────────────────────────
    for l in range(L):
        scored_l = stats["scored_attn_ns"][l][mask]
        log_score = np.log(np.clip(scored_l, 1e-9, None))

        fig, axes_grid = plt.subplots(
            n_axes, n_axes,
            figsize=(3.8 * n_axes, 3.8 * n_axes),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.08, wspace=0.08)

        mesh_ref = None

        for r in range(n_axes):
            for c in range(n_axes):
                ax = axes_grid[r, c]
                # hide upper triangle
                if c > r:
                    ax.set_visible(False)
                    continue

                if r == c:
                    # diagonal: marginal histogram
                    ax.hist(axes_vals[r], bins=60,
                            color="#888888", edgecolor="none", density=True)
                    ax.set_yticks([])
                else:
                    # off-diagonal: 2-D binned mean score map
                    x_data = axes_vals[c]
                    y_data = axes_vals[r]

                    mean_stat, x_edges, y_edges, _ = binned_statistic_2d(
                        x_data, y_data, log_score,
                        statistic="mean", bins=n_bins,
                    )
                    count_stat, _, _, _ = binned_statistic_2d(
                        x_data, y_data, log_score,
                        statistic="count", bins=n_bins,
                    )

                    # smooth score map over populated bins only
                    empty = np.isnan(mean_stat)
                    mean_stat[empty] = 0.0
                    if sigma > 0:
                        mean_stat = gaussian_filter(mean_stat, sigma=sigma)
                    mean_stat = np.ma.masked_where(empty, mean_stat)
                    # transpose so axes match (x → col, y → row)
                    mean_stat = mean_stat.T

                    # density contours
                    count_smooth = (
                        gaussian_filter(count_stat, sigma=sigma)
                        if sigma > 0 else count_stat
                    ).T
                    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                    populated  = count_smooth[count_smooth > 0]
                    if len(populated) > 4:
                        levels     = np.percentile(populated, [50, 75, 90, 99])
                        linewidths = [0.4, 0.7, 1.1, 1.6]
                        ax.contour(
                            x_centers, y_centers, count_smooth,
                            levels=levels, colors="gold",
                            linewidths=linewidths, alpha=0.65,
                        )

                    mesh = ax.pcolormesh(
                        x_edges, y_edges, mean_stat,
                        cmap="RdBu_r", norm=score_norm,
                        rasterized=True,
                    )
                    mesh_ref = mesh

                # axis labels on outer edges only
                if r == n_axes - 1:
                    ax.set_xlabel(axes_labels[c], fontsize=8)
                else:
                    ax.tick_params(labelbottom=False)

                if c == 0 and r != 0:
                    ax.set_ylabel(axes_labels[r], fontsize=8)
                else:
                    ax.tick_params(labelleft=False)

                ax.tick_params(labelsize=7)

        sd_note = (
            f"  ·  |i−j| ≥ {seq_dist_min}" if seq_dist_min > 1 else ""
        )
        d_note = (
            f"  ·  d ∈ [{d_min:.4g}, {d_max:.4g}] Å"
            if d_max < float("inf") or d_min > 0 else ""
        )
        axes_grid[0, 0].set_title(
            f"Layer {l + 1}{sd_note}{d_note}",
            fontsize=9, loc="left", pad=4,
        )

        if mesh_ref is not None:
            cb = fig.colorbar(mesh_ref, ax=axes_grid, shrink=0.55, pad=0.02)
            cb.set_ticks(score_raw_ticks)
            cb.ax.set_yticklabels(
                [f"{s:.2g}" for s in np.exp(score_raw_ticks)], fontsize=7)
            cb.ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
            cb.set_label(
                f"Mean budget-weighted score — layer {l + 1}  "
                r"(log scale; 1 = uniform baseline)",
                fontsize=7,
            )

        path = os.path.join(out_dir, f"fig_geometry_scatter_layer{l + 1}.svg")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


# ---------------------------------------------------------------------------
# ChimeraX session script helper
# ---------------------------------------------------------------------------

def _apply_enrichment_band(
    enrichment: np.ndarray,
    enriched: bool,
) -> np.ndarray:
    """
    Return a copy of a log-enrichment grid masked to a single enrichment band,
    with values transformed onto a common positive scale.

    This mirrors the convention used by :func:`_apply_log_score_band`: both the
    enriched and depleted volumes are stored as **positive** values so that a
    single surface level (``eps``) can be used for both in ChimeraX.

    Parameters
    ----------
    enrichment : raw log-enrichment grid (log(pair/background)); NaN where
                 empty.  Positive values = pair enriched; negative = depleted.
    enriched   : if True, keep voxels where enrichment > 0 and return them
                 as-is (already positive).
                 If False, keep voxels where enrichment < 0 and return their
                 **negated** values (so the stored grid is still positive and
                 the surface level in ChimeraX is identical to the high band).

    Returns
    -------
    float32 array with the same shape as enrichment.  Masked voxels are 0.
    """
    grid = np.nan_to_num(enrichment, nan=0.0)
    if enriched:
        return np.where(grid > 0.0,  grid, 0.0).astype(np.float32)
    else:
        return np.where(grid < 0.0, -grid, 0.0).astype(np.float32)


def _write_cxc(
    cxc_path: str,
    layer_files: dict,
    pdb_path: str | None = None,
    src_label: str = "all",
    nbr_label: str = "all",
) -> None:
    """
    Write a ChimeraX command script (.cxc) that opens the score-band volumes
    for one layer and applies surface rendering.

    Parameters
    ----------
    cxc_path    : output path for the .cxc file.
    layer_files : dict with keys ``"density"``, ``"score"``, ``"high"``,
                  ``"low"``, and optionally ``"enrichment"``,
                  ``"enrichment_high"``, ``"enrichment_low"``.
    pdb_path    : path to the reference PDB; if given, the script will open
                  and display it.
    src_label   : source residue label for the header comment.
    nbr_label   : neighbour residue label for the header comment.
    """
    def rel(p: str) -> str:
        return os.path.relpath(p, start=os.path.dirname(cxc_path))

    eps     = 1e-4
    has_enr = "enrichment" in layer_files

    # Model numbers assigned in open order:
    #   #1 score_high, #2 score_low, #3 density
    #   [#4 enrichment]  [#5 enrichment_high]  [#6 enrichment_low]
    #   #N pdb (last)
    next_model = 4
    enr_model  = enr_high_model = enr_low_model = None
    if has_enr:
        enr_model       = next_model;     next_model += 1
        enr_high_model  = next_model;     next_model += 1
        enr_low_model   = next_model;     next_model += 1
    pdb_model = f"#{next_model}" if pdb_path else None

    lines = [
        "# ChimeraX session script generated by oa_geometry.py",
        f"# Pair: {src_label} → {nbr_label}",
        "# Axes (local backbone frame, origin = source Cα):",
        "#   x → e₁ (backbone direction)   y → e₂   z → e₃",
        "#",
        "# High-score band      : log(score),    score > 1          (blue surface)",
        "# Low-score  band      : −log(score),   0 < score < 1      (red surface)",
        "# Enrichment high band : enrichment > 0  (pair enriched)   (teal surface)",
        "# Enrichment low  band : −enrichment < 0  (pair depleted)  (orange surface)",
        "# All band volumes stored as positive log values — same surface level (eps) for all.",
        "",
        "# ── open volumes ─────────────────────────────────────────────",
        f'open "{rel(layer_files["high"])}"',
        f'open "{rel(layer_files["low"])}"',
        f'open "{rel(layer_files["density"])}"',
    ]

    if has_enr:
        lines += [
            f'open "{rel(layer_files["enrichment"])}"',
            f'open "{rel(layer_files["enrichment_high"])}"',
            f'open "{rel(layer_files["enrichment_low"])}"',
        ]

    if pdb_path:
        lines.append(f'open "{rel(pdb_path)}"')

    lines += [
        "",
        "# ── high-score surface  (log(score), score > 1)  ────────────",
        f"volume #1 style surface level {eps} color cornflowerblue transparency 0.35",
        "",
        "# ── low-score surface  (−log(score), 0 < score < 1)  ────────",
        f"volume #2 style surface level {eps} color salmon transparency 0.35",
        "",
        "# ── density (hidden by default) ──────────────────────────────",
        "volume #3 style surface level 10 color #888888 transparency 0.80",
        "volume #3 hide",
    ]

    if has_enr:
        lines += [
            "",
            "# ── enrichment full map — hidden by default ──────────────",
            f"volume #{enr_model} hide",
            "",
            "# ── enrichment high  (log(pair/bg) > 0, pair enriched)  ─",
            f"volume #{enr_high_model} style surface level {eps} "
            f"color teal transparency 0.40",
            "",
            "# ── enrichment low   (−log(pair/bg), pair depleted)  ────",
            f"volume #{enr_low_model} style surface level {eps} "
            f"color darkorange transparency 0.40",
        ]

    if pdb_path:
        lines += [
            "",
            "# ── reference ALA backbone in the local frame ────────────",
            "#    N at -x (behind Cα), C at +x/-z, O further -z, Cb at -y",
            f"show {pdb_model} atoms",
            f"hide {pdb_model} cartoons",
            f"style {pdb_model} ball",
            f"color {pdb_model} byatom",
            f"size {pdb_model} atomRadius 0.35 bondRadius 0.12",
        ]

    lines += [
        "",
        "# ── scene setup ──────────────────────────────────────────────",
        "lighting soft",
        "set bgColor white",
    ]

    with open(cxc_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Volumetric export
# ---------------------------------------------------------------------------

def export_volumes(
    stats: dict,
    out_dir: str,
    r_max: float = 20.0,
    vol_n_bins: int = 64,
    seq_dist_min: int = 1,
    d_min: float = 0.0,
    d_max: float = float("inf"),
    source_aa: str | None = None,
    neighbour_aa: str | None = None,
) -> None:
    """
    Convert (d, θ, φ) → Cartesian (x, y, z) in the local backbone frame, bin
    onto a uniform [−r_max, +r_max]³ Å grid, and save per encoder layer.

    When ``source_aa`` / ``neighbour_aa`` are given the function operates in
    **pair mode**: it filters edges to the specified residue pair and writes a
    second set of volumes with an additional enrichment map
    (log(score_pair / score_background) per voxel).

    Background mode (no pair specified)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      geometry_layerN_density.mrc      — edge count per voxel
      geometry_layerN_score.mrc        — mean budget-weighted score
      geometry_layerN_score_high.mrc   — score in voxels where score > 1
      geometry_layerN_score_low.mrc    — score in voxels where score < 1
      geometry_layerN_chimerax.cxc     — ChimeraX session script
      geometry_reference_residue.pdb   — canonical ALA in the local frame
                                         (written once, shared by all layers)

    Pair mode (source_aa / neighbour_aa given)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      geometry_<src><nbr>_layerN_density.mrc
      geometry_<src><nbr>_layerN_score.mrc
      geometry_<src><nbr>_layerN_score_high.mrc
      geometry_<src><nbr>_layerN_score_low.mrc
      geometry_<src><nbr>_layerN_enrichment.mrc      — log(pair / background) per voxel
      geometry_<src><nbr>_layerN_enrichment_high.mrc — enrichment > 0 (pair enriched)
      geometry_<src><nbr>_layerN_enrichment_low.mrc  — enrichment < 0 (pair depleted)
      geometry_<src><nbr>_layerN_chimerax.cxc        — includes all enrichment volumes

    Parameters
    ----------
    stats        : dict produced by :func:`_load_stats`.
    out_dir      : output directory (must exist).
    r_max        : half-edge of the cubic box (Å).
    vol_n_bins   : voxels per axis.
    seq_dist_min : minimum |i − j|; edges below are excluded.
    d_min, d_max : distance range filter (Å).
    source_aa    : optional one-letter source residue code (already upper-cased).
    neighbour_aa : optional one-letter neighbour residue code (already upper-cased).
    """
    L     = stats["L"]
    voxel = 2.0 * r_max / vol_n_bins
    edges = _cartesian_edges(r_max, vol_n_bins)

    # ── apply global filters ──────────────────────────────────────────────────
    dist_all = stats["dist_3d_ns"].astype(np.float32)
    seq_all  = stats["seq_dist_ns"].astype(np.float32)
    bg_mask  = (seq_all >= seq_dist_min) & (dist_all >= d_min) & (dist_all <= d_max)

    # ── optional pair filter ──────────────────────────────────────────────────
    pair_mode = (source_aa is not None) or (neighbour_aa is not None)
    if pair_mode:
        ti = stats["res_type_i_ns"].astype(int)
        tj = stats["res_type_j_ns"].astype(int)
        pair_mask = bg_mask.copy()
        if source_aa is not None:
            pair_mask &= (ti == _AA_INDEX[source_aa])
        if neighbour_aa is not None:
            pair_mask &= (tj == _AA_INDEX[neighbour_aa])

        src_label = source_aa   if source_aa   else "all"
        nbr_label = neighbour_aa if neighbour_aa else "all"
        pair_tag  = f"_{src_label}{nbr_label}"

        n_pair = int(pair_mask.sum())
        n_bg   = int(bg_mask.sum())
        print(f"  Pair filter {src_label}→{nbr_label}: "
              f"{n_pair:,} edges  ({100 * n_pair / max(n_bg, 1):.1f} % of background)")
        if n_pair == 0:
            print("[geometry] WARNING: no pair edges survive the filters — "
                  "skipping pair volumetric export")
            return

        dist_pair  = dist_all[pair_mask]
        theta_pair = stats["theta"][pair_mask]
        phi_pair   = stats["phi"][pair_mask]
        x_pair, y_pair, z_pair = _to_cartesian(dist_pair, theta_pair, phi_pair)

        # background Cartesian (for enrichment denominator)
        dist_bg  = dist_all[bg_mask]
        theta_bg = stats["theta"][bg_mask]
        phi_bg   = stats["phi"][bg_mask]
        x_bg, y_bg, z_bg = _to_cartesian(dist_bg, theta_bg, phi_bg)

    else:
        src_label = "all"
        nbr_label = "all"
        pair_tag  = ""
        n_bg = int(bg_mask.sum())
        if n_bg == 0:
            print("[geometry] WARNING: no edges survive the filters — "
                  "skipping volumetric export")
            return
        dist_bg  = dist_all[bg_mask]
        theta_bg = stats["theta"][bg_mask]
        phi_bg   = stats["phi"][bg_mask]
        x_bg, y_bg, z_bg = _to_cartesian(dist_bg, theta_bg, phi_bg)

    print(f"  Exporting MRC volumes  "
          f"box=[±{r_max:.1f} Å]  voxel={voxel:.3f} Å  "
          f"grid={vol_n_bins}³  {'pair' if pair_mode else 'background'} "
          f"edges={n_pair if pair_mode else n_bg:,}")

    # ── reference PDB (written once) ─────────────────────────────────────────
    pdb_path = os.path.join(out_dir, "geometry_reference_residue.pdb")
    _write_reference_pdb(pdb_path)
    print(f"  reference backbone → {pdb_path}")

    prefix = os.path.join(out_dir, f"geometry{pair_tag}")

    # ── per-layer volumes ─────────────────────────────────────────────────────
    for l in range(L):
        if pair_mode:
            scores_pair = stats["scored_attn_ns"][l][pair_mask]
            scores_bg   = stats["scored_attn_ns"][l][bg_mask]
            count, mean_pair = _bin_cartesian(x_pair, y_pair, z_pair, scores_pair, edges)
            _,     mean_bg   = _bin_cartesian(x_bg,   y_bg,   z_bg,   scores_bg,   edges)

            with np.errstate(divide="ignore", invalid="ignore"):
                enrichment = np.where(
                    (mean_pair > 0) & (mean_bg > 0),
                    np.log(mean_pair / mean_bg),
                    np.nan,
                ).astype(np.float32)

            mean_score = mean_pair
        else:
            scores_bg   = stats["scored_attn_ns"][l][bg_mask]
            count, mean_score = _bin_cartesian(x_bg, y_bg, z_bg, scores_bg, edges)

        sfx       = f"_layer{l + 1}"
        den_path  = f"{prefix}{sfx}_density.mrc"
        scr_path  = f"{prefix}{sfx}_score.mrc"
        high_path = f"{prefix}{sfx}_score_high.mrc"
        low_path  = f"{prefix}{sfx}_score_low.mrc"
        cxc_path  = f"{prefix}{sfx}_chimerax.cxc"

        _write_mrc(den_path,  count,      r_max, label=f"layer{l + 1}_density")
        _write_mrc(scr_path,  mean_score, r_max, label=f"layer{l + 1}_score")
        _write_mrc(high_path, _apply_log_score_band(mean_score, above_baseline=True),
                   r_max, label=f"layer{l + 1}_score_high")
        _write_mrc(low_path,  _apply_log_score_band(mean_score, above_baseline=False),
                   r_max, label=f"layer{l + 1}_score_low")

        layer_files = {
            "density": den_path,
            "score":   scr_path,
            "high":    high_path,
            "low":     low_path,
        }

        if pair_mode:
            enr_path      = f"{prefix}{sfx}_enrichment.mrc"
            enr_high_path = f"{prefix}{sfx}_enrichment_high.mrc"
            enr_low_path  = f"{prefix}{sfx}_enrichment_low.mrc"
            _write_mrc(enr_path,      enrichment, r_max,
                       label=f"layer{l + 1}_enrichment_log")
            _write_mrc(enr_high_path, _apply_enrichment_band(enrichment, enriched=True),
                       r_max, label=f"layer{l + 1}_enrichment_high")
            _write_mrc(enr_low_path,  _apply_enrichment_band(enrichment, enriched=False),
                       r_max, label=f"layer{l + 1}_enrichment_low")
            layer_files["enrichment"]      = enr_path
            layer_files["enrichment_high"] = enr_high_path
            layer_files["enrichment_low"]  = enr_low_path
            print(f"    layer {l + 1}  enrichment      → {enr_path}")
            print(f"    layer {l + 1}  enrichment_high → {enr_high_path}")
            print(f"    layer {l + 1}  enrichment_low  → {enr_low_path}")

        _write_cxc(cxc_path, layer_files, pdb_path=pdb_path,
                   src_label=src_label, nbr_label=nbr_label)

        print(f"    layer {l + 1}  density    → {den_path}")
        print(f"    layer {l + 1}  score      → {scr_path}")
        print(f"    layer {l + 1}  score_high → {high_path}")
        print(f"    layer {l + 1}  score_low  → {low_path}")
        print(f"    layer {l + 1}  ChimeraX   → {cxc_path}")


# ---------------------------------------------------------------------------
# Figure — (d, θ, φ) geometry scatter for a residue pair
# ---------------------------------------------------------------------------

def fig_geometry_scatter_pair(
    stats: dict,
    out_dir: str,
    source_aa: str | None,
    neighbour_aa: str | None,
    score_vmin: float = 0.5,
    score_vmax: float = 2.0,
    n_bins: int = 80,
    sigma: float = 1.5,
    seq_dist_min: int = 1,
    d_min: float = 0.0,
    d_max: float = float("inf"),
) -> None:
    """
    Pair-filtered version of :func:`fig_geometry_scatter`.

    Applies the global filters (seq_dist_min, d_min, d_max) and then further
    restricts to the (source_aa → neighbour_aa) pair.  Both ``source_aa`` and
    ``neighbour_aa`` may be ``None`` (match all), in which case this function
    is equivalent to the unfiltered scatter.

    Output files (one per encoder layer):
        ``<out_dir>/fig_geometry_scatter_<src>_<nbr>_layer{N}.svg``
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import binned_statistic_2d

    L = stats["L"]

    # ── build pair mask ───────────────────────────────────────────────────────
    dist_all = stats["dist_3d_ns"].astype(np.float32)
    seq_all  = stats["seq_dist_ns"].astype(np.float32)
    mask     = (seq_all >= seq_dist_min) & (dist_all >= d_min) & (dist_all <= d_max)

    if source_aa is not None:
        mask &= (stats["res_type_i_ns"].astype(int) == _AA_INDEX[source_aa])
    if neighbour_aa is not None:
        mask &= (stats["res_type_j_ns"].astype(int) == _AA_INDEX[neighbour_aa])

    src_label = source_aa   if source_aa   else "all"
    nbr_label = neighbour_aa if neighbour_aa else "all"

    dist  = dist_all[mask]
    theta = stats["theta"][mask]
    phi   = stats["phi"][mask]
    n_edges = int(mask.sum())

    if n_edges == 0:
        print(f"[geometry] WARNING: no {src_label}→{nbr_label} edges survive "
              "the filters — skipping pair scatter figures")
        return

    print(f"  Pair scatter {src_label}→{nbr_label}: {n_edges:,} edges")

    axes_vals   = [dist, theta, phi]
    axes_labels = [
        "Cα–Cα distance $d$ (Å)",
        "Polar angle $\\theta$ (rad)  $[0, \\pi]$",
        "Azimuthal angle $\\phi$ (rad)  $[-\\pi, \\pi]$",
    ]
    n_axes = 3

    log_vmin = min(np.log(max(score_vmin, 1e-9)), -1e-6)
    log_vmax = max(np.log(max(score_vmax, 1e-9)),  1e-6)
    score_norm      = mcolors.TwoSlopeNorm(vmin=log_vmin, vcenter=0.0, vmax=log_vmax)
    score_raw_ticks = np.linspace(log_vmin, log_vmax, 7)

    for l in range(L):
        scored_l  = stats["scored_attn_ns"][l][mask]
        log_score = np.log(np.clip(scored_l, 1e-9, None))

        fig, axes_grid = plt.subplots(
            n_axes, n_axes,
            figsize=(3.8 * n_axes, 3.8 * n_axes),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.08, wspace=0.08)
        mesh_ref = None

        for r in range(n_axes):
            for c in range(n_axes):
                ax = axes_grid[r, c]
                if c > r:
                    ax.set_visible(False)
                    continue

                if r == c:
                    ax.hist(axes_vals[r], bins=60,
                            color="#888888", edgecolor="none", density=True)
                    ax.set_yticks([])
                else:
                    x_data = axes_vals[c]
                    y_data = axes_vals[r]
                    mean_stat, x_edges, y_edges, _ = binned_statistic_2d(
                        x_data, y_data, log_score,
                        statistic="mean", bins=n_bins,
                    )
                    count_stat, _, _, _ = binned_statistic_2d(
                        x_data, y_data, log_score,
                        statistic="count", bins=n_bins,
                    )
                    empty = np.isnan(mean_stat)
                    mean_stat[empty] = 0.0
                    if sigma > 0:
                        mean_stat = gaussian_filter(mean_stat, sigma=sigma)
                    mean_stat = np.ma.masked_where(empty, mean_stat).T

                    count_smooth = (
                        gaussian_filter(count_stat, sigma=sigma)
                        if sigma > 0 else count_stat
                    ).T
                    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                    populated  = count_smooth[count_smooth > 0]
                    if len(populated) > 4:
                        levels     = np.percentile(populated, [50, 75, 90, 99])
                        ax.contour(
                            x_centers, y_centers, count_smooth,
                            levels=levels, colors="gold",
                            linewidths=[0.4, 0.7, 1.1, 1.6], alpha=0.65,
                        )
                    mesh = ax.pcolormesh(
                        x_edges, y_edges, mean_stat,
                        cmap="RdBu_r", norm=score_norm, rasterized=True,
                    )
                    mesh_ref = mesh

                if r == n_axes - 1:
                    ax.set_xlabel(axes_labels[c], fontsize=8)
                else:
                    ax.tick_params(labelbottom=False)
                if c == 0 and r != 0:
                    ax.set_ylabel(axes_labels[r], fontsize=8)
                else:
                    ax.tick_params(labelleft=False)
                ax.tick_params(labelsize=7)

        axes_grid[0, 0].set_title(
            f"Layer {l + 1}  ·  {src_label}→{nbr_label}  (n={n_edges:,})",
            fontsize=9, loc="left", pad=4,
        )
        if mesh_ref is not None:
            cb = fig.colorbar(mesh_ref, ax=axes_grid, shrink=0.55, pad=0.02)
            cb.set_ticks(score_raw_ticks)
            cb.ax.set_yticklabels(
                [f"{s:.2g}" for s in np.exp(score_raw_ticks)], fontsize=7)
            cb.ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
            cb.set_label(
                f"Mean budget-weighted score — layer {l + 1}  "
                r"(log scale; 1 = uniform baseline)",
                fontsize=7,
            )

        path = os.path.join(
            out_dir,
            f"fig_geometry_scatter_{src_label}_{nbr_label}_layer{l + 1}.svg",
        )
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure — (d, θ, φ) enrichment scatter for a residue pair
# ---------------------------------------------------------------------------

def fig_geometry_enrichment(
    stats: dict,
    out_dir: str,
    source_aa: str | None,
    neighbour_aa: str | None,
    n_bins: int = 80,
    sigma: float = 1.5,
    seq_dist_min: int = 1,
    d_min: float = 0.0,
    d_max: float = float("inf"),
) -> None:
    """
    Enrichment scatter: log(pair mean score / background mean score) per 2-D bin.

    Diagonal panels overlay the pair and background marginal distributions so
    the geometric selectivity of the pair is immediately visible.  Off-diagonal
    cells show the enrichment on a symmetric RdBu_r colormap centred at 0
    (= no enrichment).

    Output files (one per encoder layer):
        ``<out_dir>/fig_geometry_enrichment_<src>_<nbr>_layer{N}.svg``

    Parameters
    ----------
    stats        : dict produced by :func:`_load_stats`.
    out_dir      : output directory.
    source_aa    : one-letter source residue code (already upper-cased), or None.
    neighbour_aa : one-letter neighbour residue code, or None.
    n_bins       : number of uniform 2-D bins per axis.
    sigma        : Gaussian smoothing σ (bins); 0 disables smoothing.
    seq_dist_min : minimum |i − j|.
    d_min, d_max : distance range filter (Å).
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import binned_statistic_2d

    L = stats["L"]

    # ── background mask ───────────────────────────────────────────────────────
    dist_all = stats["dist_3d_ns"].astype(np.float32)
    seq_all  = stats["seq_dist_ns"].astype(np.float32)
    bg_mask  = (seq_all >= seq_dist_min) & (dist_all >= d_min) & (dist_all <= d_max)

    # ── pair mask ─────────────────────────────────────────────────────────────
    pair_mask = bg_mask.copy()
    if source_aa is not None:
        pair_mask &= (stats["res_type_i_ns"].astype(int) == _AA_INDEX[source_aa])
    if neighbour_aa is not None:
        pair_mask &= (stats["res_type_j_ns"].astype(int) == _AA_INDEX[neighbour_aa])

    src_label = source_aa   if source_aa   else "all"
    nbr_label = neighbour_aa if neighbour_aa else "all"
    n_pair    = int(pair_mask.sum())
    n_bg      = int(bg_mask.sum())

    if n_pair == 0:
        print(f"[geometry] WARNING: no {src_label}→{nbr_label} edges survive "
              "the filters — skipping enrichment figures")
        return

    dist_pair  = dist_all[pair_mask];  dist_bg  = dist_all[bg_mask]
    theta_pair = stats["theta"][pair_mask]; theta_bg = stats["theta"][bg_mask]
    phi_pair   = stats["phi"][pair_mask];   phi_bg   = stats["phi"][bg_mask]

    axes_pair = [dist_pair, theta_pair, phi_pair]
    axes_bg   = [dist_bg,   theta_bg,   phi_bg]
    axes_labels = [
        "Cα–Cα distance $d$ (Å)",
        "Polar angle $\\theta$ (rad)  $[0, \\pi]$",
        "Azimuthal angle $\\phi$ (rad)  $[-\\pi, \\pi]$",
    ]
    n_axes = 3

    # symmetric log enrichment colorscale (±log(2) = 2-fold)
    enr_vmax = np.log(2.0)
    enr_vmin = -enr_vmax
    enr_norm  = mcolors.TwoSlopeNorm(vmin=enr_vmin, vcenter=0.0, vmax=enr_vmax)
    enr_ticks = np.linspace(enr_vmin, enr_vmax, 7)

    for l in range(L):
        scores_pair = stats["scored_attn_ns"][l][pair_mask]
        scores_bg   = stats["scored_attn_ns"][l][bg_mask]

        fig, axes_grid = plt.subplots(
            n_axes, n_axes,
            figsize=(3.8 * n_axes, 3.8 * n_axes),
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.08, wspace=0.08)
        mesh_ref = None

        for r in range(n_axes):
            for c in range(n_axes):
                ax = axes_grid[r, c]
                if c > r:
                    ax.set_visible(False)
                    continue

                if r == c:
                    # diagonal: overlay pair vs background marginals
                    ax.hist(axes_bg[r],   bins=60, color="#aaaaaa",
                            edgecolor="none", density=True,
                            label="background", alpha=0.6)
                    ax.hist(axes_pair[r], bins=60, color="#e05c2d",
                            edgecolor="none", density=True,
                            label=f"{src_label}→{nbr_label}", alpha=0.7)
                    ax.set_yticks([])
                    if r == 0 and c == 0:
                        ax.legend(fontsize=6, loc="upper right",
                                  framealpha=0.7, handlelength=1.0)
                else:
                    x_pair_data = axes_pair[c];  y_pair_data = axes_pair[r]
                    x_bg_data   = axes_bg[c];    y_bg_data   = axes_bg[r]

                    # shared bin edges derived from the background range
                    x_edges = np.linspace(
                        float(x_bg_data.min()), float(x_bg_data.max()), n_bins + 1)
                    y_edges = np.linspace(
                        float(y_bg_data.min()), float(y_bg_data.max()), n_bins + 1)

                    mean_pair_stat, _, _, _ = binned_statistic_2d(
                        x_pair_data, y_pair_data, scores_pair,
                        statistic="mean", bins=[x_edges, y_edges],
                    )
                    mean_bg_stat, _, _, _ = binned_statistic_2d(
                        x_bg_data, y_bg_data, scores_bg,
                        statistic="mean", bins=[x_edges, y_edges],
                    )
                    count_pair_stat, _, _, _ = binned_statistic_2d(
                        x_pair_data, y_pair_data, scores_pair,
                        statistic="count", bins=[x_edges, y_edges],
                    )

                    with np.errstate(divide="ignore", invalid="ignore"):
                        enr = np.where(
                            (mean_pair_stat > 0) & (mean_bg_stat > 0),
                            np.log(mean_pair_stat / mean_bg_stat),
                            np.nan,
                        )

                    empty = np.isnan(enr)
                    enr[empty] = 0.0
                    if sigma > 0:
                        enr = gaussian_filter(enr, sigma=sigma)
                    enr = np.ma.masked_where(empty, enr).T

                    # density contours of the pair distribution
                    count_smooth = (
                        gaussian_filter(count_pair_stat, sigma=sigma)
                        if sigma > 0 else count_pair_stat
                    ).T
                    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                    populated  = count_smooth[count_smooth > 0]
                    if len(populated) > 4:
                        levels = np.percentile(populated, [50, 75, 90, 99])
                        ax.contour(
                            x_centers, y_centers, count_smooth,
                            levels=levels, colors="black",
                            linewidths=[0.4, 0.7, 1.1, 1.6], alpha=0.55,
                        )

                    mesh = ax.pcolormesh(
                        x_edges, y_edges, enr,
                        cmap="RdBu_r", norm=enr_norm, rasterized=True,
                    )
                    mesh_ref = mesh

                if r == n_axes - 1:
                    ax.set_xlabel(axes_labels[c], fontsize=8)
                else:
                    ax.tick_params(labelbottom=False)
                if c == 0 and r != 0:
                    ax.set_ylabel(axes_labels[r], fontsize=8)
                else:
                    ax.tick_params(labelleft=False)
                ax.tick_params(labelsize=7)

        axes_grid[0, 0].set_title(
            f"Layer {l + 1}  ·  {src_label}→{nbr_label} vs background"
            f"  (n={n_pair:,})",
            fontsize=9, loc="left", pad=4,
        )
        if mesh_ref is not None:
            cb = fig.colorbar(mesh_ref, ax=axes_grid, shrink=0.55, pad=0.02)
            cb.set_ticks(enr_ticks)
            cb.ax.set_yticklabels(
                [f"{np.exp(t):.2g}×" for t in enr_ticks], fontsize=7)
            cb.ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
            cb.set_label(
                f"Score enrichment vs background — layer {l + 1}  "
                r"(log scale; 1× = no enrichment)",
                fontsize=7,
            )

        path = os.path.join(
            out_dir,
            f"fig_geometry_enrichment_{src_label}_{nbr_label}_layer{l + 1}.svg",
        )
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = GeometryContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)

    stats   = ctx.stats
    L       = stats["L"]
    mode    = str(stats.get("mode", "oa"))
    n_prots = int(stats.get("n_proteins", 0))
    n_ns    = len(stats["theta"])

    pair_mode = (ctx.source_aa is not None) or (ctx.neighbour_aa is not None)
    src_label = ctx.source_aa   if ctx.source_aa   else "all"
    nbr_label = ctx.neighbour_aa if ctx.neighbour_aa else "all"

    print(
        f"[geometry] mode={mode}  layers={L}  "
        f"proteins={n_prots}  non-self edges (total)={n_ns:,}"
    )
    if pair_mode:
        print(f"[geometry] pair filter: {src_label} → {nbr_label}")

    if ctx.seq_dist_min > 1 or ctx.d_min > 0 or ctx.d_max < float("inf"):
        dist = stats["dist_3d_ns"].astype(np.float32)
        seq  = stats["seq_dist_ns"].astype(np.float32)
        mask = (seq >= ctx.seq_dist_min) & (dist >= ctx.d_min) & (dist <= ctx.d_max)
        n_pass = int(mask.sum())
        filters = []
        if ctx.seq_dist_min > 1:
            filters.append(f"|i−j| ≥ {ctx.seq_dist_min}")
        if ctx.d_min > 0 or ctx.d_max < float("inf"):
            filters.append(f"d ∈ [{ctx.d_min:.4g}, {ctx.d_max:.4g}] Å")
        print(
            f"[geometry] filter ({', '.join(filters)}): "
            f"{n_pass:,} edges  "
            f"({100 * n_pass / max(n_ns, 1):.1f} % of non-self)"
        )

    # ── background scatter (always) ───────────────────────────────────────────
    print("[geometry] Generating (d, θ, φ) geometry scatter figures …")
    fig_geometry_scatter(
        stats, ctx.out_dir,
        score_vmin   = ctx.score_vmin,
        score_vmax   = ctx.score_vmax,
        n_bins       = ctx.n_bins,
        sigma        = ctx.sigma,
        seq_dist_min = ctx.seq_dist_min,
        d_min        = ctx.d_min,
        d_max        = ctx.d_max,
    )

    # ── pair scatter + enrichment scatter (when pair specified) ───────────────
    if pair_mode:
        print(f"[geometry] Generating pair scatter figures "
              f"({src_label}→{nbr_label}) …")
        fig_geometry_scatter_pair(
            stats, ctx.out_dir,
            source_aa    = ctx.source_aa,
            neighbour_aa = ctx.neighbour_aa,
            score_vmin   = ctx.score_vmin,
            score_vmax   = ctx.score_vmax,
            n_bins       = ctx.n_bins,
            sigma        = ctx.sigma,
            seq_dist_min = ctx.seq_dist_min,
            d_min        = ctx.d_min,
            d_max        = ctx.d_max,
        )
        print(f"[geometry] Generating enrichment scatter figures "
              f"({src_label}→{nbr_label} vs background) …")
        fig_geometry_enrichment(
            stats, ctx.out_dir,
            source_aa    = ctx.source_aa,
            neighbour_aa = ctx.neighbour_aa,
            n_bins       = ctx.n_bins,
            sigma        = ctx.sigma,
            seq_dist_min = ctx.seq_dist_min,
            d_min        = ctx.d_min,
            d_max        = ctx.d_max,
        )

    # ── volumetric export ─────────────────────────────────────────────────────
    if ctx.export_vol:
        print("[geometry] Exporting background volumetric MRC files …")
        export_volumes(
            stats, ctx.out_dir,
            r_max        = ctx.r_max,
            vol_n_bins   = ctx.vol_n_bins,
            seq_dist_min = ctx.seq_dist_min,
            d_min        = ctx.d_min,
            d_max        = ctx.d_max,
        )
        if pair_mode:
            print(f"[geometry] Exporting pair volumetric MRC files "
                  f"({src_label}→{nbr_label}) …")
            export_volumes(
                stats, ctx.out_dir,
                r_max        = ctx.r_max,
                vol_n_bins   = ctx.vol_n_bins,
                seq_dist_min = ctx.seq_dist_min,
                d_min        = ctx.d_min,
                d_max        = ctx.d_max,
                source_aa    = ctx.source_aa,
                neighbour_aa = ctx.neighbour_aa,
            )
    else:
        print("[geometry] Volumetric export skipped (--no_vol)")

    print(f"[geometry] Done — outputs written to {ctx.out_dir!r}")


if __name__ == "__main__":
    main()