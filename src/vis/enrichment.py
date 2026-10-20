"""
enrichment.py — type-specific geometric preference (γ) for residue-pair types.

Implements the statistical procedure specified in ``aa_pair_enrichment_statistics_v2.md``
(read that file for the full rationale — this is the executable counterpart):

  * interaction log-enrichment
        γ(t, b) = log[ m(t, b) / m(t, ·) ]  −  log[ m(·, b) / m(·, ·) ]
    for a residue-pair type t in a geometric bin b, where m(t, ·) / m(·, ·) are marginals
    over the *entire* non-self-edge pool (not restricted to the panel's bins).
  * a protein-level cluster bootstrap for the standard error / CI of γ̂ (residues within one
    protein, and edges of one residue, are correlated — the protein is the independent unit).
  * a per-bin test of H0: γ ≤ δ vs H1: γ > δ for a pre-specified minimum effect of interest δ
    (default log(1.2), i.e. "at least 20% more attention than expected"), Benjamini-Hochberg
    FDR corrected over the eligible bins of one panel.
  * a pair-level summary via split-half reproducibility: correlation (r) and magnitude (G) of
    the γ map across random halves of the protein set, answering "does this pair type show a
    geometric preference at all" independently of any one bin.

Each input stats.npz is analysed **independently** — seeds are never pooled (unlike
``pmpnn-geometry``, which pools seeds by design). Passing multiple files runs the full
procedure on each and additionally reports a cross-seed comparison (correlation of γ maps;
fraction of seeds in which each bin is significant) as a robustness check.

Requires a stats.py output that includes the ``protein_idx`` per-edge array (protein identity,
the clustering unit here). Older stats.npz files predating this field will be rejected with a
clear error asking for a re-run of stats.py.

Usage
-----
python enrichment.py  stats1.npz [stats2.npz ...]  --source_aa ... --neighbour_aa ...  [options]

Examples
--------
# Minimal — all defaults, one seed
python enrichment.py  stats.npz  --source_aa A I L V  --neighbour_aa A I L V

# Several seeds, analysed independently + cross-seed robustness check
python enrichment.py  oa1_seed0/stats.npz oa1_seed1/stats.npz oa1_seed2/stats.npz \\
    --source_aa H K R  --neighbour_aa D E

# Custom minimum effect of interest (30% instead of the 20% default) and finer grid
python enrichment.py  stats.npz  --source_aa C  --neighbour_aa C  \\
    --delta_fold 1.3  --d_bins 40  --theta_bins 40  --phi_bins 40
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
from scipy.stats import norm, false_discovery_control

from vis.geometry import _AA_ALPHABET, _AA_INDEX, _aa_display, _aa_tag


# ---------------------------------------------------------------------------
# Context — configuration + validated CLI parsing
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentContext:
    """All configuration needed to run the γ-enrichment procedure.

    CLI-facing fields are validated in :meth:`__post_init__`; the class can also be
    instantiated programmatically for use in notebooks or tests.
    """

    # ── I/O ───────────────────────────────────────────────────────────────────
    stats_npz: list   # one or more .npz paths — each analysed INDEPENDENTLY (never pooled)
    out_dir:   str

    # ── residue-pair type ─────────────────────────────────────────────────────
    source_aa:    list   # one-letter codes for the source residue
    neighbour_aa: list   # one-letter codes for the neighbour residue

    # ── minimum effect of interest ─────────────────────────────────────────────
    delta_fold: float    # fold-change threshold (e.g. 1.2 = "at least 20% more attention")

    # ── eligibility ────────────────────────────────────────────────────────────
    min_count:    int    # minimum type-t edges in a bin to test it
    min_proteins: int    # minimum distinct proteins contributing type-t edges to a bin

    # ── auto-relax eligibility for sparse panels ───────────────────────────────
    auto_relax_eligibility: bool   # if a panel is too sparse, retry with relaxed thresholds
    relax_factor:           float  # multiply min_count/min_proteins by this when relaxing
    min_count_floor:        int    # never relax min_count below this
    min_proteins_floor:     int    # never relax min_proteins below this

    # ── multiple testing ───────────────────────────────────────────────────────
    fdr_alpha: float

    # ── bootstrap / permutation sizes ─────────────────────────────────────────
    n_boot:   int    # protein-level cluster bootstrap replicates for γ SE/CI
    n_splits: int    # random protein splits for split-half reproducibility
    n_perm:   int    # AA-label permutations for the formal pair-level test (§8)

    # ── binning ────────────────────────────────────────────────────────────────
    d_bins:     int
    theta_bins: int
    phi_bins:   int
    d_max:      float    # upper edge of the distance axis (Å); lower edge is the data min

    # ── sequence-distance filter ──────────────────────────────────────────────
    seq_dist_min: int

    # ── reference population ──────────────────────────────────────────────────
    exclude_type_from_background: bool   # see "self-inclusion in the reference" caveat

    # ── reproducibility ────────────────────────────────────────────────────────
    seed: int

    # ─────────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # --- path validation --------------------------------------------------
        if len(self.stats_npz) == 0:
            raise ValueError("At least one stats_npz file must be given")
        for path in self.stats_npz:
            if not os.path.isfile(path):
                raise ValueError(f"stats_npz not found: {path!r}")
            if not path.endswith(".npz"):
                print(f"[warn] stats_npz does not have a .npz extension: {path!r}")

        out_parent = os.path.dirname(os.path.abspath(self.out_dir)) or "."
        if not os.path.isdir(out_parent):
            raise ValueError(f"Parent directory of out_dir does not exist: {out_parent!r}")

        # --- residue-pair validation -------------------------------------------
        for attr in ("source_aa", "neighbour_aa"):
            aas = [aa.upper() for aa in getattr(self, attr)]
            for aa in aas:
                if aa not in _AA_INDEX:
                    raise ValueError(
                        f"{attr} must be one-letter amino-acid codes, got {aa!r}.  "
                        f"Valid codes: {''.join(_AA_ALPHABET)}")
            setattr(self, attr, aas)

        # --- numeric validation -------------------------------------------------
        if self.delta_fold <= 1.0:
            raise ValueError(
                f"delta_fold must be > 1.0 (it is a minimum fold-change of interest, "
                f"e.g. 1.2 = at least 20% more attention), got {self.delta_fold}")
        if self.min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {self.min_count}")
        if self.min_proteins < 2:
            raise ValueError(f"min_proteins must be >= 2, got {self.min_proteins}")
        if not (0.0 < self.relax_factor < 1.0):
            raise ValueError(f"relax_factor must be in (0, 1), got {self.relax_factor}")
        if self.min_count_floor < 1:
            raise ValueError(f"min_count_floor must be >= 1, got {self.min_count_floor}")
        if self.min_proteins_floor < 2:
            raise ValueError(f"min_proteins_floor must be >= 2, got {self.min_proteins_floor}")
        if not (0.0 < self.fdr_alpha < 1.0):
            raise ValueError(f"fdr_alpha must be in (0, 1), got {self.fdr_alpha}")
        if self.n_boot < 100:
            raise ValueError(f"n_boot must be >= 100, got {self.n_boot}")
        if self.n_splits < 10:
            raise ValueError(f"n_splits must be >= 10, got {self.n_splits}")
        if self.n_perm < 10:
            raise ValueError(f"n_perm must be >= 10, got {self.n_perm}")
        for name, val in [("d_bins", self.d_bins), ("theta_bins", self.theta_bins),
                          ("phi_bins", self.phi_bins)]:
            if val < 4:
                raise ValueError(f"{name} must be >= 4, got {val}")
        if self.d_max <= 0:
            raise ValueError(f"d_max must be > 0, got {self.d_max}")
        if self.seq_dist_min < 1:
            raise ValueError(
                f"seq_dist_min must be >= 1 (self-edges always excluded), "
                f"got {self.seq_dist_min}")

    @property
    def delta(self) -> float:
        """Minimum effect of interest, on the log scale."""
        return float(np.log(self.delta_fold))

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_args(cls) -> "EnrichmentContext":
        """Parse sys.argv and return a validated :class:`EnrichmentContext`."""
        p = argparse.ArgumentParser(
            description=(
                "Type-specific geometric preference (gamma) for a residue-pair type, with "
                "a protein-level cluster bootstrap, a per-bin test against a minimum effect "
                "of interest, and a pair-level split-half reproducibility summary.  "
                "See aa_pair_enrichment_statistics_v2.md for the full procedure."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        p.add_argument(
            "stats_npz", nargs="+",
            help=(
                ".npz statistics file(s) produced by stats.py (requires the protein_idx "
                "field).  Each file is analysed INDEPENDENTLY — seeds are never pooled.  "
                "Passing several files additionally reports a cross-seed comparison."
            ),
        )
        p.add_argument("--out_dir", default="enrichment",
                       help="Output directory (created if absent)")

        pr = p.add_argument_group("Residue-pair type")
        pr.add_argument("--source_aa", nargs="+", required=True, metavar="AA",
                        help="One or more one-letter codes for the source residue "
                             "(matched as 'any of').")
        pr.add_argument("--neighbour_aa", nargs="+", required=True, metavar="AA",
                        help="One or more one-letter codes for the neighbour residue.")

        d = p.add_argument_group("Minimum effect of interest")
        d.add_argument(
            "--delta_fold", type=float, default=1.2,
            help="Minimum fold-change for H1 (e.g. 1.2 = test for at least 20%% more "
                 "attention than expected, not just != 0). Fix before looking at the maps.",
        )

        e = p.add_argument_group("Eligibility")
        e.add_argument("--min_count", type=int, default=30,
                       help="Minimum type-t edges in a bin for it to be tested")
        e.add_argument("--min_proteins", type=int, default=20,
                       help="Minimum distinct proteins contributing a type-t edge to a bin "
                            "for it to be tested (a cluster bootstrap over too few clusters "
                            "gives unreliable SEs)")
        e.add_argument(
            "--no_auto_relax_eligibility", action="store_false", dest="auto_relax_eligibility",
            help="Disable auto-relaxing min_count/min_proteins for a sparse panel (see "
                 "--relax_factor). Off by default means: if a panel has too few eligible "
                 "bins to say anything, it just stays mostly blank with a warning.",
        )
        p.set_defaults(auto_relax_eligibility=True)
        e.add_argument(
            "--relax_factor", type=float, default=0.5,
            help="When a panel is too sparse (see the sparse-panel warning threshold), "
                 "retry it with min_count/min_proteins multiplied by this factor (floored "
                 "at --min_count_floor / --min_proteins_floor). Relaxed bins carry a "
                 "visibly weaker evidentiary standard than the requested thresholds — the "
                 "warning says so and the figure title flags which panels were relaxed.",
        )
        e.add_argument("--min_count_floor", type=int, default=10,
                       help="Never relax min_count below this, no matter how sparse")
        e.add_argument("--min_proteins_floor", type=int, default=10,
                       help="Never relax min_proteins below this (per the methods doc, a "
                            "cluster bootstrap below ~10 proteins is already shaky)")

        f = p.add_argument_group("Multiple testing")
        f.add_argument("--fdr_alpha", type=float, default=0.05,
                       help="Benjamini-Hochberg FDR threshold, applied per panel")

        bs = p.add_argument_group("Bootstrap / split-half sizes")
        bs.add_argument("--n_boot", type=int, default=1000,
                        help="Protein-level cluster bootstrap replicates for gamma SE/CI")
        bs.add_argument("--n_splits", type=int, default=200,
                        help="Random protein splits for split-half reproducibility (r, G)")
        bs.add_argument("--n_perm", type=int, default=200,
                        help="AA-label permutations for the formal pair-level test (§8): "
                             "is the observed magnitude explainable by chance association "
                             "between amino-acid identity and geometry?")

        bn = p.add_argument_group("Binning")
        bn.add_argument("--d_bins", type=int, default=32)
        bn.add_argument("--theta_bins", type=int, default=32)
        bn.add_argument("--phi_bins", type=int, default=32)
        bn.add_argument("--d_max", type=float, default=12.0,
                        help="Upper edge of the distance axis (A)")

        sd = p.add_argument_group("Sequence-distance filter")
        sd.add_argument("--seq_dist_min", type=int, default=1,
                        help="Minimum |i-j| for edges included in all outputs")

        rf = p.add_argument_group("Reference population")
        rf.add_argument(
            "--exclude_type_from_background", action="store_true",
            help="Compute the all-edge background/marginal from edges NOT of type t "
                 "(instead of all edges, which include type t itself). See the "
                 "'self-inclusion in the reference' caveat in the methods doc — the "
                 "default (include) shrinks gamma toward 0 for abundant types, which is "
                 "conservative.",
        )

        p.add_argument("--seed", type=int, default=0, help="RNG seed")

        args = p.parse_args()
        return cls(**vars(args))


# ---------------------------------------------------------------------------
# Statistics loading
# ---------------------------------------------------------------------------

def _load_stats(path: str) -> dict:
    """
    Load a .npz file produced by stats.py and derive the non-self-edge geometry pool,
    including protein identity (the clustering unit for this procedure) and residue
    identity (source AND neighbour — needed to look a residue up consistently wherever
    it appears, e.g. for the amino-acid-label permutation test in §8).

    Raises
    ------
    ValueError
        If the file predates the ``protein_idx`` or ``nbr_res_idx`` per-edge arrays.

    Returns
    -------
    dict with keys: L, mode, n_proteins, theta, phi, dist_3d_ns, seq_dist_ns,
    res_type_i_ns, res_type_j_ns, protein_idx_ns, res_idx_ns, nbr_res_idx_ns,
    scored_attn_ns (list of L arrays), plus the per-residue lookup tables
    res_protein / res_type (dense, indexed directly by global residue id — built from
    the FULL edge pool, self-edges included, so every residue is covered even if it
    happens to have zero non-self edges).
    """
    raw = np.load(path, allow_pickle=False)
    L    = int(raw["L"])
    mode = str(raw["mode"])

    if "nbr_Ca" not in raw:
        raise ValueError(
            f"stats file {path!r} does not contain the 'nbr_Ca' per-edge array.  This "
            f"script requires a stats.py output produced after the local-frame "
            f"neighbour-coordinate rework — re-run stats.py (mode found: {mode!r}).")
    if "protein_idx" not in raw:
        raise ValueError(
            f"stats file {path!r} does not contain the 'protein_idx' per-edge array.  "
            f"This script requires a stats.py output that tracks protein identity "
            f"(the clustering unit for the protein-level bootstrap) — re-run stats.py.")
    if "nbr_res_idx" not in raw:
        raise ValueError(
            f"stats file {path!r} does not contain the 'nbr_res_idx' per-edge array.  "
            f"This script requires a stats.py output that tracks the neighbour's own "
            f"residue identity (needed for the pair-level permutation test) — "
            f"re-run stats.py.")

    is_self  = raw["is_self"].astype(bool)
    non_self = ~is_self

    nbr_Ca = raw["nbr_Ca"][non_self]
    dist   = np.linalg.norm(nbr_Ca, axis=-1)
    d_hat  = nbr_Ca / np.clip(dist, 1e-8, None)[:, None]
    theta  = np.arccos(np.clip(d_hat[:, 0], -1.0, 1.0))
    phi    = np.arctan2(d_hat[:, 1], d_hat[:, 2])

    res_idx     = raw["res_idx"]
    protein_idx = raw["protein_idx"]
    res_type_i  = raw["res_type_i"]
    n_res = int(res_idx.max()) + 1
    res_protein = np.zeros(n_res, dtype=np.int64)
    res_type    = np.zeros(n_res, dtype=np.int8)
    res_protein[res_idx] = protein_idx   # idempotent: every row for a residue agrees
    res_type[res_idx]    = res_type_i

    stats = {
        "L": L, "mode": mode, "n_proteins": int(raw.get("n_proteins", 0)),
        "theta": theta.astype(np.float32), "phi": phi.astype(np.float32),
        "dist_3d_ns": dist.astype(np.float32),
        "seq_dist_ns": raw["seq_dist"][non_self].astype(np.float32),
        "res_type_i_ns": raw["res_type_i"][non_self].astype(np.int8),
        "res_type_j_ns": raw["res_type_j"][non_self].astype(np.int8),
        "protein_idx_ns": raw["protein_idx"][non_self].astype(np.int64),
        "res_idx_ns": raw["res_idx"][non_self].astype(np.int64),
        "nbr_res_idx_ns": raw["nbr_res_idx"][non_self].astype(np.int64),
        "scored_attn_ns": [raw[f"scored_attn_{l}"][non_self] for l in range(L)],
        "res_protein": res_protein,   # (n_res,) protein id, indexed by global residue id
        "res_type": res_type,         # (n_res,) true AA type, indexed by global residue id
    }
    return stats


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per value, -1 for anything outside [edges[0], edges[-1])."""
    idx = np.digitize(values, edges) - 1
    idx[(idx < 0) | (idx >= len(edges) - 1)] = -1
    return idx


def panel_bin_id(axes_vals: list[np.ndarray], axes_edges: list[np.ndarray]) -> tuple[np.ndarray, tuple]:
    """Flatten an N-axis (1, 2 or 3 of d/theta/phi) binning scheme into a single bin-id
    array (row-major over the axes, in the order given), -1 where any axis is out of range.

    Returns (bin_id, shape) where shape is the per-axis bin counts.
    """
    idxs  = [_bin_index(v, e) for v, e in zip(axes_vals, axes_edges)]
    shape = tuple(len(e) - 1 for e in axes_edges)
    valid = np.ones_like(idxs[0], dtype=bool)
    for idx in idxs:
        valid &= idx >= 0
    flat = np.full(idxs[0].shape, -1, dtype=np.int64)
    multi = idxs[0].copy()
    for idx, n in zip(idxs[1:], shape[1:]):
        multi = multi * n + idx
    flat[valid] = multi[valid]
    return flat, shape


# ---------------------------------------------------------------------------
# Sufficient-statistic tables (§3)
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentTables:
    """Per-protein x per-bin sufficient statistics, plus per-protein marginal totals
    (over the entire non-self-edge pool, not restricted to the panel's bins).

    All arrays are dense (P x B for the per-bin tables; P for the marginals) — the number
    of proteins (the clustering unit here) is small enough (hundreds) that sparsity isn't
    needed, unlike a residue-level clustering which would have hundreds of thousands of rows.
    """
    S_t:   np.ndarray   # (P, B) sum of scores, type-t edges
    N_t:   np.ndarray   # (P, B) count of edges, type-t
    S_all: np.ndarray   # (P, B) sum of scores, all edges (reference population)
    N_all: np.ndarray   # (P, B) count of edges, all edges
    s_t:   np.ndarray   # (P,) per-protein total score sum, type-t, whole neighbour set
    n_t:   np.ndarray   # (P,) per-protein total edge count, type-t, whole neighbour set
    s_all: np.ndarray   # (P,) per-protein total score sum, all edges, whole neighbour set
    n_all: np.ndarray   # (P,) per-protein total edge count, all edges, whole neighbour set
    n_proteins: int
    n_bins: int
    bin_shape: tuple


def build_tables(
    stats: dict,
    source_aa: list[str],
    neighbour_aa: list[str],
    bin_id: np.ndarray,
    bin_shape: tuple,
    layer: int = 0,
    seq_dist_min: int = 1,
    exclude_type_from_background: bool = False,
) -> EnrichmentTables:
    """Build the four sufficient-statistic tables of §3, for one (seed, layer, panel).

    ``bin_id`` assigns every non-self edge to a bin (-1 = outside the panel's range) —
    build it with :func:`panel_bin_id`. The marginals (``s_t``/``n_t``/``s_all``/``n_all``)
    are computed over the WHOLE seq_dist_min-filtered neighbour set, independent of
    ``bin_id`` — so every panel built from the same stats/pair-type/layer shares identical
    marginals, as required by the procedure.
    """
    score = stats["scored_attn_ns"][layer]
    seq  = stats["seq_dist_ns"]
    ti   = stats["res_type_i_ns"].astype(int)
    tj   = stats["res_type_j_ns"].astype(int)
    pidx = stats["protein_idx_ns"]

    si = [_AA_INDEX[aa] for aa in source_aa]
    ni = [_AA_INDEX[aa] for aa in neighbour_aa]

    keep    = seq >= seq_dist_min
    type_m  = keep & np.isin(ti, si) & np.isin(tj, ni)
    all_m   = keep & (~type_m if exclude_type_from_background else np.ones_like(keep))

    P = int(pidx.max()) + 1
    B = int(np.prod(bin_shape))

    def _accumulate_2d(mask):
        rows = pidx[mask]
        cols = bin_id[mask]
        valid = cols >= 0
        rows, cols, vals = rows[valid], cols[valid], score[mask][valid]
        S = np.zeros((P, B), dtype=np.float64)
        N = np.zeros((P, B), dtype=np.float64)
        np.add.at(S, (rows, cols), vals)
        np.add.at(N, (rows, cols), 1.0)
        return S, N

    def _accumulate_1d(mask):
        s = np.zeros(P, dtype=np.float64)
        n = np.zeros(P, dtype=np.float64)
        np.add.at(s, pidx[mask], score[mask])
        np.add.at(n, pidx[mask], 1.0)
        return s, n

    S_t,   N_t   = _accumulate_2d(type_m)
    S_all, N_all = _accumulate_2d(all_m)
    s_t,   n_t   = _accumulate_1d(type_m)
    s_all, n_all = _accumulate_1d(all_m)

    return EnrichmentTables(
        S_t=S_t, N_t=N_t, S_all=S_all, N_all=N_all,
        s_t=s_t, n_t=n_t, s_all=s_all, n_all=n_all,
        n_proteins=P, n_bins=B, bin_shape=bin_shape,
    )


# ---------------------------------------------------------------------------
# Protein-level cluster bootstrap for gamma (§3)
# ---------------------------------------------------------------------------

@dataclass
class GammaResult:
    gamma_hat: np.ndarray     # (B,) point estimate
    se:        np.ndarray     # (B,) bootstrap SE
    ci_lo:     np.ndarray     # (B,) 2.5th percentile
    ci_hi:     np.ndarray     # (B,) 97.5th percentile
    count_t:   np.ndarray     # (B,) type-t edge count per bin (observed)
    n_prot_t:  np.ndarray     # (B,) distinct proteins contributing a type-t edge per bin
    vs_bg:     np.ndarray     # (B,) descriptive: log[m(t,b)/m(.,b)]  (not tested)
    vs_type:   np.ndarray     # (B,) descriptive: log[m(t,b)/m(t,.)]  (not tested)


def _gamma_point_estimate(tab: EnrichmentTables, w: np.ndarray | None = None):
    """gamma(b), vs_bg(b), vs_type(b) for one set of protein weights (w=None => observed,
    unweighted totals).

    The type/background marginals used as each bin's reference are LEAVE-ONE-BIN-OUT:
    m(t,.) excludes bin b's own contribution to the type-t total when testing bin b (and
    likewise m(.,.) excludes bin b's contribution to the all-edge total). Without this, a
    bin holding a large share of a type's edges inflates its own reference and dilutes the
    very effect being measured — the bigger that bin's share, the worse the self-
    contamination. E.g. for Cys->Cys, the disulfide-range (d, theta) bin alone holds ~1/3
    of all Cys->Cys edges; using the plain (non-leave-one-out) type marginal there
    understated its fold-enrichment by about 25% relative to leave-one-out (1.51x vs the
    correct 2.03x) in a worked check on real data. `st`/`sa` etc. are the GRAND totals
    over the type's/background's entire non-self-edge pool (every bin the panel covers,
    not just the populated ones), so `st - St` is "everywhere this bin isn't", including
    geometry outside this panel's own bin range — a stable, panel-independent reference.
    """
    if w is None:
        St, Nt = tab.S_t.sum(0), tab.N_t.sum(0)
        Sa, Na = tab.S_all.sum(0), tab.N_all.sum(0)
        st, nt = tab.s_t.sum(), tab.n_t.sum()
        sa, na = tab.s_all.sum(), tab.n_all.sum()
    else:
        St, Nt = w @ tab.S_t, w @ tab.N_t
        Sa, Na = w @ tab.S_all, w @ tab.N_all
        st, nt = w @ tab.s_t, w @ tab.n_t
        sa, na = w @ tab.s_all, w @ tab.n_all

    with np.errstate(invalid="ignore", divide="ignore"):
        m_tb    = St / Nt
        m_ab    = Sa / Na
        m_t_loo = (st - St) / (nt - Nt)   # type marginal, excluding bin b itself
        m_a_loo = (sa - Sa) / (na - Na)   # background marginal, excluding bin b itself
        vs_bg   = np.log(m_tb / m_ab)
        vs_type = np.log(m_tb / m_t_loo)
        gamma   = np.log(m_tb / m_t_loo) - np.log(m_ab / m_a_loo)
    return gamma, vs_bg, vs_type


def interaction_bootstrap(tab: EnrichmentTables, n_boot: int = 1000, seed: int = 0) -> GammaResult:
    """Protein-level cluster bootstrap for gamma, per §3. Each replicate draws a
    multinomial resample of the P proteins and recomputes ALL FOUR terms of gamma from the
    same weights, so numerator and reference variability are shared correctly.
    """
    P = tab.n_proteins
    rng = np.random.default_rng(seed)
    boot_gamma = np.empty((n_boot, tab.n_bins))
    for r in range(n_boot):
        w = np.bincount(rng.integers(0, P, size=P), minlength=P).astype(np.float64)
        boot_gamma[r], _, _ = _gamma_point_estimate(tab, w)

    gamma_hat, vs_bg, vs_type = _gamma_point_estimate(tab, w=None)
    with np.errstate(invalid="ignore"):
        se = np.nanstd(boot_gamma, axis=0)
        ci_lo, ci_hi = np.nanpercentile(boot_gamma, [2.5, 97.5], axis=0)

    count_t  = tab.N_t.sum(0)
    n_prot_t = (tab.N_t > 0).sum(0)

    return GammaResult(
        gamma_hat=gamma_hat, se=se, ci_lo=ci_lo, ci_hi=ci_hi,
        count_t=count_t, n_prot_t=n_prot_t, vs_bg=vs_bg, vs_type=vs_type,
    )


# ---------------------------------------------------------------------------
# Eligibility + per-bin preference test (§4)
# ---------------------------------------------------------------------------

def eligibility_mask(gr: GammaResult, min_count: int, min_proteins: int) -> np.ndarray:
    """Bins with enough type-t data to trust the bootstrap SE at all. Independent of the
    scores themselves, so filtering on it does not bias the test."""
    return (gr.count_t >= min_count) & (gr.n_prot_t >= min_proteins)


def preference_test(gr: GammaResult, delta: float, fdr_alpha: float, eligible: np.ndarray):
    """Per-bin test of H0: gamma <= delta vs H1: gamma > delta (§4), BH-FDR corrected over
    the eligible bins of this one panel.

    Returns (p, q, significant) — all length n_bins; p/q are NaN outside `eligible`.
    """
    n_bins = gr.gamma_hat.shape[0]
    p = np.full(n_bins, np.nan)
    q = np.full(n_bins, np.nan)
    significant = np.zeros(n_bins, dtype=bool)

    if eligible.sum() == 0:
        return p, q, significant

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (gr.gamma_hat[eligible] - delta) / gr.se[eligible]
    p_elig = norm.sf(z)   # one-sided: P(Z > z) = 1 - Phi(z)
    q_elig = false_discovery_control(np.nan_to_num(p_elig, nan=1.0), method="bh")

    idx = np.where(eligible)[0]
    p[idx] = p_elig
    q[idx] = q_elig
    significant[idx] = q_elig <= fdr_alpha
    return p, q, significant


# ---------------------------------------------------------------------------
# Split-half reproducibility (§5)
# ---------------------------------------------------------------------------

@dataclass
class ReproducibilityResult:
    r_values: np.ndarray   # (n_splits,)
    g_values: np.ndarray   # (n_splits,)

    @property
    def r_median(self) -> float:
        return float(np.nanmedian(self.r_values))

    @property
    def r_range(self) -> tuple:
        return tuple(np.nanpercentile(self.r_values, [2.5, 97.5]))

    @property
    def g_median(self) -> float:
        return float(np.nanmedian(self.g_values))

    @property
    def g_range(self) -> tuple:
        return tuple(np.nanpercentile(self.g_values, [2.5, 97.5]))

    def shows_preference(self, delta: float) -> bool:
        """Decision rule (§5): reproducible (2.5th pct of r > 0) AND large enough
        (median G >= delta)."""
        return (self.r_range[0] > 0) and (self.g_median >= delta)


def _weighted_pearson(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    wsum = w.sum()
    if wsum <= 0:
        return np.nan
    ma, mb = (w * a).sum() / wsum, (w * b).sum() / wsum
    cov  = (w * (a - ma) * (b - mb)).sum() / wsum
    va   = (w * (a - ma) ** 2).sum() / wsum
    vb   = (w * (b - mb) ** 2).sum() / wsum
    denom = np.sqrt(va * vb)
    return float(cov / denom) if denom > 0 else np.nan


def split_half_reproducibility(
    tab: EnrichmentTables, n_splits: int = 200, min_count: int = 30,
    min_proteins: int = 20, seed: int = 0,
) -> ReproducibilityResult:
    """Pair-level reproducibility (§5): for S random halves of the protein set, compute
    gamma on each half (point estimate only, no nested bootstrap), restrict to bins
    eligible in BOTH halves (§4 thresholds halved), and compute the count-weighted
    correlation r and magnitude G between the two halves' gamma maps.
    """
    P = tab.n_proteins
    rng = np.random.default_rng(seed)
    half_min_count    = max(1, min_count // 2)
    half_min_proteins = max(1, min_proteins // 2)

    # full-data weights: total type-t edge count per bin, used consistently across splits
    w_full = tab.N_t.sum(0)

    r_values = np.empty(n_splits)
    g_values = np.empty(n_splits)

    for s in range(n_splits):
        perm = rng.permutation(P)
        half = P // 2
        idx_A, idx_B = perm[:half], perm[half:]
        w_A = np.zeros(P); w_A[idx_A] = 1.0
        w_B = np.zeros(P); w_B[idx_B] = 1.0

        gamma_A, _, _ = _gamma_point_estimate(tab, w_A)
        gamma_B, _, _ = _gamma_point_estimate(tab, w_B)

        count_A = w_A @ tab.N_t
        count_B = w_B @ tab.N_t
        nprot_A = ((tab.N_t > 0) * w_A[:, None]).sum(0)
        nprot_B = ((tab.N_t > 0) * w_B[:, None]).sum(0)

        elig = (
            (count_A >= half_min_count) & (nprot_A >= half_min_proteins) &
            (count_B >= half_min_count) & (nprot_B >= half_min_proteins)
        )
        if elig.sum() < 2:
            r_values[s] = np.nan
            g_values[s] = np.nan
            continue

        gA, gB, w = gamma_A[elig], gamma_B[elig], w_full[elig]
        r_values[s] = _weighted_pearson(gA, gB, w)
        cross = (w * gA * gB).sum() / w.sum() if w.sum() > 0 else np.nan
        g_values[s] = np.sqrt(max(0.0, cross))

    return ReproducibilityResult(r_values=r_values, g_values=g_values)


# ---------------------------------------------------------------------------
# Pair-level formal test: amino-acid-label permutation (§8, optional)
# ---------------------------------------------------------------------------

def _permute_types_within_protein(res_protein: np.ndarray, res_type: np.ndarray, rng) -> np.ndarray:
    """Random permutation of `res_type`, confined within each protein — same per-protein
    composition (same multiset of amino-acid types), same structural graph and geometry
    (nothing about `res_protein`/edges changes), only WHICH residue position gets WHICH
    type is scrambled. `res_protein`/`res_type` are dense arrays indexed by global
    residue id (as built in :func:`_load_stats`).

    Vectorized via a grouped lexsort: `group_order` visits residues grouped by protein in
    a fixed (stable) order; `random_order` visits the SAME groups (same protein, same
    block sizes) but randomly ordered within each block. Assigning
    ``permuted[group_order] = res_type[random_order]`` therefore swaps types only among
    residues sharing a protein.
    """
    n_res = res_protein.shape[0]
    group_order  = np.argsort(res_protein, kind="stable")
    random_order = np.lexsort((rng.random(n_res), res_protein))
    permuted = np.empty_like(res_type)
    permuted[group_order] = res_type[random_order]
    return permuted


def _weighted_rms(gamma: np.ndarray, weight: np.ndarray) -> float:
    """sqrt(weighted mean of gamma^2) over eligible bins of a single (non-split) gamma
    map — the magnitude statistic used for the permutation test.

    This is deliberately NOT the cross-half-product estimator used for G in §5 (which
    avoids inflating the estimate with squaring noise by using two independent halves).
    Here both the observed statistic and every permutation replicate use this same
    single-map estimator, so whatever upward bias per-bin sampling noise adds to gamma^2
    is present in both and cancels out of the p-value comparison — it would NOT cancel
    if this were compared against the §5 cross-product G instead, so the two G's must
    not be used interchangeably.
    """
    wsum = weight.sum()
    if wsum <= 0:
        return np.nan
    return float(np.sqrt(max(0.0, (weight * gamma ** 2).sum() / wsum)))


@dataclass
class PermutationTestResult:
    g_obs:  float        # observed magnitude (single-map weighted RMS of gamma)
    g_perm: np.ndarray   # (n_perm,) null distribution
    p:      float        # P(G_permuted >= G_observed), one-sided, +1/+1 smoothed


def pair_level_permutation_test(
    stats: dict, source_aa: list[str], neighbour_aa: list[str],
    bin_id: np.ndarray, bin_shape: tuple, layer: int = 0, seq_dist_min: int = 1,
    min_count: int = 30, min_proteins: int = 20, n_perm: int = 200,
    exclude_type_from_background: bool = False, seed: int = 0,
) -> PermutationTestResult:
    """Formal pair-level test (§8): is the observed geometric preference explainable by
    chance association between amino-acid identity and geometry, rather than a real
    property of this pair type?

    Null: randomly permute amino-acid labels to residue POSITIONS within each protein —
    same per-protein composition, same structural graph, same geometry for every edge —
    and recompute gamma. This destroys type specificity while keeping everything else
    fixed, exactly as specified in the methods doc. One full (non-bootstrap, non-split)
    gamma computation per permutation; see :func:`_weighted_rms` for the magnitude
    statistic and its relationship to the descriptive G of §5.
    """
    score   = stats["scored_attn_ns"][layer]
    seq     = stats["seq_dist_ns"]
    pidx    = stats["protein_idx_ns"]
    r_idx   = stats["res_idx_ns"]
    nbr_idx = stats["nbr_res_idx_ns"]
    res_protein = stats["res_protein"]
    res_type    = stats["res_type"]

    keep   = seq >= seq_dist_min
    n_bins = int(np.prod(bin_shape))
    P      = int(pidx.max()) + 1
    si = [_AA_INDEX[aa] for aa in source_aa]
    ni = [_AA_INDEX[aa] for aa in neighbour_aa]

    def _accumulate_2d(mask):
        rows, cols, vals = pidx[mask], bin_id[mask], score[mask]
        valid = cols >= 0
        rows, cols, vals = rows[valid], cols[valid], vals[valid]
        S = np.zeros((P, n_bins)); N = np.zeros((P, n_bins))
        np.add.at(S, (rows, cols), vals)
        np.add.at(N, (rows, cols), 1.0)
        return S, N

    def _accumulate_1d(mask):
        s = np.zeros(P); n = np.zeros(P)
        np.add.at(s, pidx[mask], score[mask])
        np.add.at(n, pidx[mask], 1.0)
        return s, n

    # Background doesn't depend on which labels are "type t" UNLESS it must explicitly
    # exclude type-t edges — compute it once upfront in the common (default) case.
    background_is_fixed = not exclude_type_from_background
    if background_is_fixed:
        S_all, N_all = _accumulate_2d(keep)
        s_all, n_all = _accumulate_1d(keep)

    def _gamma_and_g(ti, tj):
        type_m = keep & np.isin(ti, si) & np.isin(tj, ni)
        S_t, N_t = _accumulate_2d(type_m)
        s_t, n_t = _accumulate_1d(type_m)
        if background_is_fixed:
            Sa, Na, sa, na = S_all, N_all, s_all, n_all
        else:
            all_m = keep & ~type_m
            Sa, Na = _accumulate_2d(all_m)
            sa, na = _accumulate_1d(all_m)

        St_bin, Nt_bin = S_t.sum(0), N_t.sum(0)
        Sa_bin, Na_bin = Sa.sum(0), Na.sum(0)
        st_tot, nt_tot = s_t.sum(), n_t.sum()
        sa_tot, na_tot = sa.sum(), na.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            # leave-one-bin-out marginals, matching _gamma_point_estimate exactly — the
            # permutation null (g_perm) and the per-bin gamma must use the identical
            # formula, or G_obs vs G_perm would not be comparing the same quantity.
            m_tb    = St_bin / Nt_bin
            m_ab    = Sa_bin / Na_bin
            m_t_loo = (st_tot - St_bin) / (nt_tot - Nt_bin)
            m_a_loo = (sa_tot - Sa_bin) / (na_tot - Na_bin)
            gamma = np.log(m_tb / m_t_loo) - np.log(m_ab / m_a_loo)

        count_t  = Nt_bin
        n_prot_t = (N_t > 0).sum(0)
        eligible = (count_t >= min_count) & (n_prot_t >= min_proteins)
        return _weighted_rms(np.nan_to_num(gamma[eligible], nan=0.0), count_t[eligible])

    g_obs = _gamma_and_g(stats["res_type_i_ns"], stats["res_type_j_ns"])

    rng = np.random.default_rng(seed)
    g_perm = np.empty(n_perm)
    for k in range(n_perm):
        permuted_type = _permute_types_within_protein(res_protein, res_type, rng)
        g_perm[k] = _gamma_and_g(permuted_type[r_idx], permuted_type[nbr_idx])

    p = (1.0 + float((g_perm >= g_obs).sum())) / (1.0 + n_perm)
    return PermutationTestResult(g_obs=g_obs, g_perm=g_perm, p=p)


# ---------------------------------------------------------------------------
# Plotting (§6)
# ---------------------------------------------------------------------------

_ENR_VMAX  = float(np.log(2.0))
_ENR_NORM  = mcolors.TwoSlopeNorm(vmin=-_ENR_VMAX, vcenter=0.0, vmax=_ENR_VMAX)
_ENR_TICKS = np.linspace(-_ENR_VMAX, _ENR_VMAX, 7)


def _add_gamma_colorbar(fig, mesh, ax):
    cb = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
    cb.set_ticks(_ENR_TICKS)
    cb.ax.set_yticklabels([f"{np.exp(t):.2g}×" for t in _ENR_TICKS], fontsize=7)
    cb.ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    cb.set_label("exp(γ)  (log scale; 1× = no type-specific preference)", fontsize=7)
    return cb


_SPARSE_ELIGIBLE_FRACTION = 0.10   # warn below this fraction of eligible bins in a panel


def _warn_sparse_panel(
    label: str, axis_name_x: str, axis_name_y: str, src_tag: str, nbr_tag: str,
    n_elig: int, n_bins: int, frac: float,
) -> None:
    """Print a warning + concrete suggestion for a panel that stayed sparse (either
    --no_auto_relax_eligibility was passed, or relaxing didn't actually help) — this
    happens for rare pair types (e.g. Cys->Cys) whose edges get fragmented across more
    bins than the data can support, not because there's no signal (see the "binning
    resolution trades bias against power" caveat in the methods doc).
    """
    flag_x = f"--{axis_name_x}_bins"
    flag_y = f"--{axis_name_y}_bins"
    print(
        f"[enrichment] WARNING: {label}{src_tag}→{nbr_tag}  "
        f"({axis_name_x},{axis_name_y}) panel: only {n_elig}/{n_bins} bins eligible "
        f"({frac:.1%}) — likely too few edges for this grid resolution, not an "
        f"absence of signal. Try halving {flag_x} and/or {flag_y} (coarser bins pool "
        f"more edges/proteins per bin), or judge this pair type from the pair-level "
        f"summary (r, G, permutation p) instead of the per-bin map."
    )


def _eligibility_with_auto_relax(
    gr: "GammaResult", ctx: "EnrichmentContext", axis_name_x: str, axis_name_y: str,
    label: str, src_tag: str, nbr_tag: str,
) -> tuple:
    """Eligibility mask for one panel, auto-relaxing min_count/min_proteins (see
    --relax_factor / --*_floor) when too few bins are eligible at the requested
    thresholds and --auto_relax_eligibility is on (the default).

    Relaxing trades reliability for coverage: bins that only clear the RELAXED
    thresholds have a weaker evidentiary standard (fewer contributing proteins means a
    less reliable bootstrap SE, per the methods doc) — this is always printed loudly
    when it happens, never done silently.

    Returns (eligible, min_count_used, min_proteins_used, relaxed).
    """
    elig = eligibility_mask(gr, ctx.min_count, ctx.min_proteins)
    n_bins = len(elig)
    frac = elig.sum() / n_bins if n_bins > 0 else 0.0
    if frac >= _SPARSE_ELIGIBLE_FRACTION:
        return elig, ctx.min_count, ctx.min_proteins, False

    if not ctx.auto_relax_eligibility:
        _warn_sparse_panel(label, axis_name_x, axis_name_y, src_tag, nbr_tag,
                           int(elig.sum()), n_bins, frac)
        return elig, ctx.min_count, ctx.min_proteins, False

    relaxed_min_count    = max(ctx.min_count_floor,    int(ctx.min_count    * ctx.relax_factor))
    relaxed_min_proteins = max(ctx.min_proteins_floor, int(ctx.min_proteins * ctx.relax_factor))
    elig_relaxed = eligibility_mask(gr, relaxed_min_count, relaxed_min_proteins)

    if elig_relaxed.sum() <= elig.sum():
        # relaxing didn't actually buy anything (truly almost no data at all) —
        # say so plainly rather than claiming a relax that had no effect.
        _warn_sparse_panel(label, axis_name_x, axis_name_y, src_tag, nbr_tag,
                           int(elig.sum()), n_bins, frac)
        return elig, ctx.min_count, ctx.min_proteins, False

    print(
        f"[enrichment] WARNING: {label}{src_tag}→{nbr_tag}  "
        f"({axis_name_x},{axis_name_y}) panel: only {int(elig.sum())}/{n_bins} bins "
        f"eligible ({frac:.1%}) at min_count={ctx.min_count}/min_proteins={ctx.min_proteins} "
        f"— auto-relaxing to min_count={relaxed_min_count}/min_proteins={relaxed_min_proteins} "
        f"({int(elig_relaxed.sum())}/{n_bins} bins eligible now). NOTE: bins only "
        f"eligible after relaxing carry a weaker evidentiary standard (fewer "
        f"proteins → less reliable bootstrap SE) — treat them with extra caution, or "
        f"pass --no_auto_relax_eligibility to turn this off."
    )
    return elig_relaxed, relaxed_min_count, relaxed_min_proteins, True


def plot_gamma_panel(
    ax, gamma_grid: np.ndarray, significant_grid: np.ndarray, eligible_grid: np.ndarray,
    x_edges: np.ndarray, y_edges: np.ndarray,
):
    """One 2-D gamma panel: effect shown everywhere eligible (§6 — non-significant bins are
    NOT blanked, only ineligible ones), significant bins marked with a contour."""
    masked = np.ma.masked_where(~eligible_grid, gamma_grid)
    mesh = ax.pcolormesh(x_edges, y_edges, masked.T, cmap="RdBu_r", norm=_ENR_NORM,
                         rasterized=True)
    if significant_grid.any():
        x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax.contour(x_c, y_c, significant_grid.T.astype(float), levels=[0.5],
                  colors="black", linewidths=1.1)
    return mesh


def plot_gamma_triangle(
    stats: dict, source_aa: list[str], neighbour_aa: list[str], ctx: "EnrichmentContext",
    layer: int = 0, title_prefix: str = "",
):
    """Lower-triangular (d, theta, phi) grid of gamma panels, matching geometry.py's own
    fig_geometry_scatter / fig_geometry_enrichment layout (diagonal = marginal background
    histograms, off-diagonal = gamma heatmaps, shared colorbar).
    """
    d_edges     = np.linspace(stats["dist_3d_ns"][stats["seq_dist_ns"] >= ctx.seq_dist_min].min(),
                              ctx.d_max, ctx.d_bins + 1)
    theta_edges = np.linspace(0, np.pi, ctx.theta_bins + 1)
    phi_edges   = np.linspace(-np.pi, np.pi, ctx.phi_bins + 1)

    axes_vals   = [stats["dist_3d_ns"], stats["theta"], stats["phi"]]
    axes_edges  = [d_edges, theta_edges, phi_edges]
    axes_labels = [
        "Cα–Cα distance $d$ (Å)",
        r"Polar angle $\theta$ (rad)  $[0, \pi]$",
        r"Azimuthal angle $\phi$ (rad)  $[-\pi, \pi]$",
    ]
    axis_names = ["d", "theta", "phi"]   # matches --d_bins / --theta_bins / --phi_bins
    n_axes = 3
    bg = stats["seq_dist_ns"] >= ctx.seq_dist_min

    fig, axes_grid = plt.subplots(n_axes, n_axes, figsize=(4.2 * n_axes, 4.2 * n_axes), squeeze=False)
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    mesh_ref = None
    panel_results = {}

    for r in range(n_axes):
        for c in range(n_axes):
            ax = axes_grid[r, c]
            if c > r:
                ax.set_visible(False)
                continue
            if r == c:
                ax.hist(axes_vals[r][bg], bins=60, color="#888888", edgecolor="none", density=True)
                ax.set_yticks([])
            else:
                bin_id, shape = panel_bin_id([axes_vals[c], axes_vals[r]], [axes_edges[c], axes_edges[r]])
                tab = build_tables(stats, source_aa, neighbour_aa, bin_id, shape,
                                   layer=layer, seq_dist_min=ctx.seq_dist_min,
                                   exclude_type_from_background=ctx.exclude_type_from_background)
                gr = interaction_bootstrap(tab, n_boot=ctx.n_boot, seed=ctx.seed)
                elig, mc_used, mp_used, relaxed = _eligibility_with_auto_relax(
                    gr, ctx, axis_names[c], axis_names[r], title_prefix,
                    _aa_tag(source_aa), _aa_tag(neighbour_aa),
                )
                _, _, sig = preference_test(gr, ctx.delta, ctx.fdr_alpha, elig)

                gamma_grid = gr.gamma_hat.reshape(shape)
                elig_grid  = elig.reshape(shape)
                sig_grid   = sig.reshape(shape)
                mesh = plot_gamma_panel(ax, gamma_grid, sig_grid, elig_grid,
                                        axes_edges[c], axes_edges[r])
                mesh_ref = mesh
                panel_results[(c, r)] = (gr, elig, sig)

                if relaxed:
                    ax.text(0.03, 0.97, f"relaxed\n(n≥{mc_used}, p≥{mp_used})",
                           transform=ax.transAxes, fontsize=6, va="top", ha="left",
                           color="#8a4b00",
                           bbox=dict(boxstyle="round,pad=0.2", fc="#fff3e0", ec="#8a4b00",
                                     linewidth=0.6, alpha=0.9))

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
        f"{title_prefix}{_aa_display(source_aa)}→{_aa_display(neighbour_aa)}  "
        f"(black contour = significant at q≤{ctx.fdr_alpha}, δ={ctx.delta_fold:.2g}×)",
        fontsize=9, loc="left", pad=4,
    )
    if mesh_ref is not None:
        _add_gamma_colorbar(fig, mesh_ref, axes_grid)
    return fig, axes_grid, panel_results


# ---------------------------------------------------------------------------
# Cross-seed comparison (§7)
# ---------------------------------------------------------------------------

def cross_seed_summary(per_seed_gamma: dict, per_seed_sig: dict) -> dict:
    """Light robustness check across seeds (§7): pairwise correlation of gamma maps
    (over bins eligible in both seeds) and the fraction of seeds in which each bin is
    significant. `per_seed_gamma`/`per_seed_sig` map seed label -> (gamma, eligible) /
    significant arrays, all for the SAME panel/binning.
    """
    labels = list(per_seed_gamma.keys())
    n = len(labels)
    corr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g_i, e_i = per_seed_gamma[labels[i]]
            g_j, e_j = per_seed_gamma[labels[j]]
            both = e_i & e_j
            if both.sum() >= 2:
                corr[i, j] = np.corrcoef(g_i[both], g_j[both])[0, 1]

    sig_stack = np.stack([per_seed_sig[l] for l in labels], axis=0)
    frac_sig = sig_stack.mean(axis=0)

    return {"labels": labels, "pairwise_correlation": corr, "fraction_seeds_significant": frac_sig}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _seed_label(path: str) -> str:
    """Best-effort checkpoint/seed name for *path*, for labelling outputs."""
    abspath = os.path.abspath(path)
    parent  = os.path.basename(os.path.dirname(abspath))
    if parent.lower() in {"eval", "stats", "output", "out", "results"}:
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(abspath)))
        if grandparent:
            return grandparent
    return parent or os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    ctx = EnrichmentContext.from_args()
    os.makedirs(ctx.out_dir, exist_ok=True)
    src_tag = _aa_tag(ctx.source_aa)
    nbr_tag = _aa_tag(ctx.neighbour_aa)

    print(f"[enrichment] pair type: {_aa_display(ctx.source_aa)} → "
          f"{_aa_display(ctx.neighbour_aa)}   delta={ctx.delta_fold:.2g}× "
          f"(log={ctx.delta:.3g})   fdr_alpha={ctx.fdr_alpha}")

    per_seed_gamma_dtheta = {}
    per_seed_sig_dtheta   = {}
    summary_lines = ["| seed | n_pair_edges | r (median, 95% range) | G (median, 95% range) "
                     "| shows preference | permutation p (G_obs) |",
                     "|---|---|---|---|---|---|"]

    for path in ctx.stats_npz:
        label = _seed_label(path)
        print(f"\n[enrichment] === seed {label} ({path}) ===")
        stats = _load_stats(path)

        for l in range(stats["L"]):
            layer_tag = f"layer{l + 1}"
            seed_out = os.path.join(ctx.out_dir, label)
            os.makedirs(seed_out, exist_ok=True)

            fig, axes_grid, panel_results = plot_gamma_triangle(
                stats, ctx.source_aa, ctx.neighbour_aa, ctx, layer=l,
                title_prefix=f"{label}  {layer_tag}  ·  ",
            )
            fig_path = os.path.join(seed_out, f"fig_gamma_{src_tag}_{nbr_tag}_{layer_tag}.svg")
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {fig_path}")

            # pair-level reproducibility on the primary (d, theta) panel
            d_edges     = np.linspace(
                stats["dist_3d_ns"][stats["seq_dist_ns"] >= ctx.seq_dist_min].min(),
                ctx.d_max, ctx.d_bins + 1)
            theta_edges = np.linspace(0, np.pi, ctx.theta_bins + 1)
            bin_id, shape = panel_bin_id([stats["dist_3d_ns"], stats["theta"]], [d_edges, theta_edges])
            tab = build_tables(stats, ctx.source_aa, ctx.neighbour_aa, bin_id, shape,
                               layer=l, seq_dist_min=ctx.seq_dist_min,
                               exclude_type_from_background=ctx.exclude_type_from_background)
            repro = split_half_reproducibility(tab, n_splits=ctx.n_splits,
                                               min_count=ctx.min_count,
                                               min_proteins=ctx.min_proteins, seed=ctx.seed)
            decision = "YES" if repro.shows_preference(ctx.delta) else "no"

            perm = pair_level_permutation_test(
                stats, ctx.source_aa, ctx.neighbour_aa, bin_id, shape, layer=l,
                seq_dist_min=ctx.seq_dist_min, min_count=ctx.min_count,
                min_proteins=ctx.min_proteins, n_perm=ctx.n_perm,
                exclude_type_from_background=ctx.exclude_type_from_background, seed=ctx.seed,
            )

            print(f"  [{layer_tag}] (d,θ) panel: n_pair_edges={int(tab.N_t.sum()):,}  "
                  f"r={repro.r_median:.3f} [{repro.r_range[0]:.3f}, {repro.r_range[1]:.3f}]  "
                  f"G={repro.g_median:.3f} [{repro.g_range[0]:.3f}, {repro.g_range[1]:.3f}]  "
                  f"shows_preference={decision}")
            print(f"  [{layer_tag}] AA-label permutation test (n_perm={ctx.n_perm}): "
                  f"G_obs={perm.g_obs:.3f}  p={perm.p:.4g}"
                  f"{'  (floored — try a larger --n_perm)' if perm.p <= 1.0 / (ctx.n_perm + 1) else ''}")
            summary_lines.append(
                f"| {label} {layer_tag} | {int(tab.N_t.sum()):,} | "
                f"{repro.r_median:.3f} [{repro.r_range[0]:.3f}, {repro.r_range[1]:.3f}] | "
                f"{repro.g_median:.3f} [{repro.g_range[0]:.3f}, {repro.g_range[1]:.3f}] | "
                f"{decision} | "
                f"p={perm.p:.4g} (G_obs={perm.g_obs:.3f}) |")

            if l == 0:
                gr, elig, sig = panel_results[(0, 1)]  # (d, theta) panel
                per_seed_gamma_dtheta[f"{label}_{layer_tag}"] = (gr.gamma_hat, elig)
                per_seed_sig_dtheta[f"{label}_{layer_tag}"] = sig

    summary_path = os.path.join(ctx.out_dir, f"summary_{src_tag}_{nbr_tag}.md")
    with open(summary_path, "w") as fh:
        fh.write(f"# {_aa_display(ctx.source_aa)} → {_aa_display(ctx.neighbour_aa)}  "
                 f"(δ={ctx.delta_fold:.2g}×, FDR≤{ctx.fdr_alpha})\n\n")
        fh.write("\n".join(summary_lines) + "\n")
    print(f"\n[enrichment] saved {summary_path}")

    if len(per_seed_gamma_dtheta) > 1:
        cross = cross_seed_summary(per_seed_gamma_dtheta, per_seed_sig_dtheta)
        print("\n[enrichment] cross-seed comparison, (d,θ) panel, layer1:")
        labels = cross["labels"]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                print(f"  corr({labels[i]}, {labels[j]}) = {cross['pairwise_correlation'][i, j]:.3f}")
        print(f"  fraction of seeds significant per bin: "
              f"mean={np.nanmean(cross['fraction_seeds_significant']):.3f}  "
              f"max={np.nanmax(cross['fraction_seeds_significant']):.3f}")

    print(f"\n[enrichment] Done — outputs written to {ctx.out_dir!r}")


if __name__ == "__main__":
    main()
