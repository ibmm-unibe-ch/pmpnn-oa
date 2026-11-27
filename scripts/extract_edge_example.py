"""
extract_edge_example.py
========================
For each requested residue-pair type (e.g. AILV -> AILV), find the
--n_examples (default 3) most illustrative example edges in a stats.npz's
edge-type population and, for each, write out its attending residue plus
its medium/long-range attention
neighbourhood (every edge the model actually looked at from that residue
with |i-j| >= --edge_seq_dist_min, any neighbour type — not just edges of
the requested pair; same sequence-distance window as the bin selection
below, so the displayed environment isn't diluted by trivially-close
backbone neighbours the edge-type analysis itself excludes) as a
backbone-only mmCIF structure, so it can be opened and colored by
attention score (B-factor) in ChimeraX / PyMOL.

Selection (two-step, as specified)
-----------------------------------
  1. Restrict to non-self edges with |i-j| >= --edge_seq_dist_min and d in
     [--dist_min, --dist_max). Bin them into a (d, theta, phi) grid
     (--n_dist/--n_theta/--n_phi) and, for the requested pair type A->B,
     take the bin with the highest score RELATIVE TO BACKGROUND — the
     same interaction/pooled contrast as pmpnn-vis's edge-type panels
     (A->B - A->notB - notA->B + notA->notB in log2 space, see
     vis.__main__.contrast), a point estimate only (no bootstrap — this
     picks one illustrative example, it isn't a statistical test), among
     bins where every group has >= --min_count edges (falls back to the
     most A->B-populated bin, ranked by raw score, if none qualify).
  2. Within that bin, take the --n_examples A->B edges with the highest
     individual raw score (fewer if the bin doesn't have that many). Each
     edge's SOURCE (attending) residue is its own structure's center —
     one mmCIF file per edge.

Recovering real coordinates
----------------------------
stats.npz only stores derived per-edge quantities (local-frame relative
vectors, scores — no absolute atomic coordinates), so each example's real
backbone is re-loaded from the original structure dataset (--h5_file) via
the same pmpnn.data loaders stats.py itself used. --ids_file must be the
EXACT list stats.npz was built from (e.g. data/test_monomers.txt): this
script replays stats.py's own accept/reject filtering over it to recover
which structure a given stats.npz protein_idx refers to (deterministic —
only the random assembly pose inside _loader_h5 isn't reproduced, and that
doesn't affect accept/reject or a chain's own internal geometry).

For a protein with more than one chain, stats.py concatenates its chains
in a RANDOM order per run (random.shuffle in pmpnn.preprocess.featurize),
so that order can't be read off directly either. Instead, every chain
permutation's candidate residue-residue distances are cross-checked
against stats.npz's own recorded distances for that protein; the
permutation that matches is the one used in the original run.

Usage
-----
python scripts/extract_edge_example.py  stats.npz  [options]

Examples
--------
# Default 5 pairs, encoder layer 1, written to edge_examples/
python scripts/extract_edge_example.py models/exp2_finetuning/oa3_seed0/eval/stats.npz

# Just two pairs, layer 2
python scripts/extract_edge_example.py stats.npz --layer 2 --pairs C:C KR:DE

# 5 examples per pair instead of the default 3
python scripts/extract_edge_example.py stats.npz --n_examples 5
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from _cif_utils import write_backbone_cif
from pmpnn.constants import (
    AA_3TO1, DATA_MAX_PROTEIN_LENGTH,
    PMPNN_CB_COEFF_A, PMPNN_CB_COEFF_B, PMPNN_CB_COEFF_C,
)
from pmpnn.data import _loader_h5, _to_pdb_dict
from vis.__main__ import _parse_pairs, bin_index, contrast, grid_sums

_AA_1TO3 = {one: three for three, one in AA_3TO1.items() if three not in ("MSE", "SEC")}
_AA_1TO3["X"] = "UNK"


# ---------------------------------------------------------------------------
# Step 1 + 2: highest-scoring bin, then highest-scoring edge within it
# ---------------------------------------------------------------------------

def find_top_edges(stats: dict, A: np.ndarray, B: np.ndarray, seq_dist_min: int,
                   dist_min: float, dist_max: float, n_dist: int, n_theta: int,
                   n_phi: int, min_count: int, background: str,
                   n_examples: int) -> list[dict]:
    """Highest-background-corrected-score (d, theta, phi) bin for edges of
    type A->B, then the `n_examples` highest RAW-scoring A->B edges within
    it (fewer if the bin doesn't have that many A->B edges).

    The bin score is the same point-estimate interaction/pooled contrast as
    vis.__main__._compute_edge_grids (A->B - A->notB - notA->B + notA->notB
    in log2 space) — no bootstrap, this only needs a handful of illustrative
    examples, not a confidence interval. Returns [] if no A->B edges fall in
    [dist_min, dist_max) with |i-j| >= seq_dist_min."""
    seq_dist, dist = stats["seq_dist"], stats["dist_3d"]
    theta, phi = stats["theta"], stats["phi"]
    ti, tj, score = stats["res_type_i"], stats["res_type_j"], stats["score"]

    window = (seq_dist >= seq_dist_min) & (dist >= dist_min) & (dist < dist_max)
    idx = np.flatnonzero(window)
    is_a, is_b = np.isin(ti[idx], A), np.isin(tj[idx], B)
    type_mask = is_a & is_b
    if not type_mask.any():
        return []

    d_edges = np.linspace(dist_min, dist_max, n_dist + 1)
    t_edges = np.linspace(0.0, np.pi, n_theta + 1)
    p_edges = np.linspace(-np.pi, np.pi, n_phi + 1)
    db = bin_index(dist[idx], d_edges)
    tb = bin_index(theta[idx], t_edges)
    pb = bin_index(phi[idx], p_edges)
    flat = (db * n_theta + tb) * n_phi + pb
    n_bins = n_dist * n_theta * n_phi

    # group 0=A->B, 1=A->notB, 2=notA->B, 3=notA->notB, matching
    # vis.__main__._compute_edge_grids's own group encoding exactly.
    group = (~is_a).astype(np.int64) * 2 + (~is_b).astype(np.int64)
    gid = np.where(flat >= 0, flat * 4 + group, -1)
    v = np.log2(score[idx])
    s, c = grid_sums(gid, v, (n_bins, 4))
    bin_score = contrast(s, c, background, min_count)   # (n_bins,), NaN where ineligible

    count_ab = c[:, 0]
    eligible = ~np.isnan(bin_score)
    relaxed = False
    if eligible.any():
        best_bin = int(np.nanargmax(bin_score))
    else:
        # background correction isn't reliably estimable anywhere in this
        # window — fall back to the most A->B-populated bin, raw score.
        relaxed = True
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_mean = s[:, 0] / c[:, 0]
        best_bin = int(np.argmax(np.where(count_ab == count_ab.max(), raw_mean, -np.inf)))

    in_bin = idx[(flat == best_bin) & type_mask]
    ranked = in_bin[np.argsort(score[in_bin])[::-1][:n_examples]]

    return [
        dict(
            protein_idx=int(stats["protein_idx"][e]),
            res_idx=int(stats["res_idx"][e]),
            nbr_res_idx=int(stats["nbr_res_idx"][e]),
            score=float(score[e]),
            dist=float(dist[e]), theta=float(theta[e]), phi=float(phi[e]),
            bin_score=float(bin_score[best_bin]) if not relaxed else float("nan"),
            bin_count=int(count_ab[best_bin]),
            relaxed=relaxed,
            rank=rank,
        )
        for rank, e in enumerate(ranked, start=1)
    ]


def collect_environment(stats: dict, protein_idx: int, res_idx_global: int, seq_dist_min: int):
    """Non-self edges FROM this residue (any neighbour type) with
    |i-j| >= seq_dist_min — the model's own attention neighbourhood,
    restricted to the same medium/long-range window as the edge-type bin
    selection itself, as (nbr_res_idx, score)."""
    m = (
        (stats["protein_idx"] == protein_idx) & (stats["res_idx"] == res_idx_global)
        & (stats["seq_dist"] >= seq_dist_min)
    )
    rows = np.flatnonzero(m)
    return stats["nbr_res_idx"][rows], stats["score"][rows]


# ---------------------------------------------------------------------------
# protein_idx -> real structure
# ---------------------------------------------------------------------------

def resolve_protein_chains(ids_file: str, h5_file: str, max_length: int,
                           needed_protein_idx: set) -> dict:
    """Replay stats.py's own accumulate()-order filtering over ids_file to map
    each needed (0-indexed, in-processing-order) protein_idx back to its
    PDBID_CHAIN string."""
    with open(ids_file) as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    needed = set(needed_protein_idx)
    max_needed = max(needed)
    mapping = {}
    n_proteins = 0
    for pdb_chain in ids:
        pdb_dict = _to_pdb_dict(_loader_h5([pdb_chain], h5_file), max_length)
        if pdb_dict is None:
            continue
        if n_proteins in needed:
            mapping[n_proteins] = pdb_chain
        n_proteins += 1
        if n_proteins > max_needed:
            break

    missing = needed - mapping.keys()
    if missing:
        raise RuntimeError(
            f"could not resolve protein_idx {sorted(missing)} against {ids_file!r} "
            f"(wrong --ids_file / --h5_file / --max_protein_length for this "
            f"stats.npz, or the dataset changed since it was generated?)")
    return mapping


def _load_chains(pdb_chain: str, h5_file: str, max_length: int) -> list:
    pdb_dict = _to_pdb_dict(_loader_h5([pdb_chain], h5_file), max_length)
    if pdb_dict is None:
        raise RuntimeError(f"{pdb_chain} no longer loads from {h5_file!r} "
                           f"(dataset changed since the stats.npz was generated?)")
    chains = []
    for letter in pdb_dict["masked_list"] + pdb_dict["visible_list"]:
        seq = pdb_dict[f"seq_chain_{letter}"]
        c = pdb_dict[f"coords_chain_{letter}"]
        N = np.asarray(c[f"N_chain_{letter}"], dtype=np.float64)
        CA = np.asarray(c[f"CA_chain_{letter}"], dtype=np.float64)
        C = np.asarray(c[f"C_chain_{letter}"], dtype=np.float64)
        O = np.asarray(c[f"O_chain_{letter}"], dtype=np.float64)
        resolved = (np.isfinite(N).all(axis=1) & np.isfinite(CA).all(axis=1)
                   & np.isfinite(C).all(axis=1) & np.isfinite(O).all(axis=1))
        chains.append(dict(letter=letter, seq=seq, N=N, CA=CA, C=C, O=O, resolved=resolved))
    return chains


def _concat_resolved(chains: list, order: tuple):
    """Concatenate chains in `order`, each restricted to its resolved residues
    — exactly mirroring pmpnn.preprocess.featurize's "mask out unresolved,
    keep resolved in place, concatenate chains" residue numbering."""
    by_letter = {c["letter"]: c for c in chains}
    seqs, Ns, CAs, Cs, Os = [], [], [], [], []
    for letter in order:
        c = by_letter[letter]
        m = c["resolved"]
        seqs.append("".join(a for a, keep in zip(c["seq"], m) if keep))
        Ns.append(c["N"][m]); CAs.append(c["CA"][m]); Cs.append(c["C"][m]); Os.append(c["O"][m])
    return "".join(seqs), np.concatenate(Ns), np.concatenate(CAs), np.concatenate(Cs), np.concatenate(Os)


def pick_chain_order(chains: list, stats: dict, protein_idx: int, offset: int,
                     rng: np.random.Generator, sample_size: int = 200):
    """Resolve which permutation of `chains` stats.py concatenated for this
    protein, by brute-force matching candidate Cα-Cα distances against
    stats.npz's own recorded dist_3d for a sample of this protein's edges.
    Robust even for identical-sequence homodimers (pure geometry, no
    reliance on sequence identity).

    Always validated geometrically, even for a single candidate chain:
    _loader_h5 picks a random assembly per call (see module docstring), so
    even the SET of chains `chains` holds can differ from what the
    original stats.py run saw (e.g. a homodimer assembly vs. the lone
    asymmetric-unit chain) — not just their order. A single-chain `chains`
    that's missing a chain the original run actually used would otherwise
    go undetected here and silently misassign indices in build_example.
    Raises RuntimeError if no permutation matches; the caller
    (resolve_structure) retries with a freshly (re-)drawn `chains`."""
    letters = [c["letter"] for c in chains]

    m = stats["protein_idx"] == protein_idx
    sample = np.flatnonzero(m)
    if sample.size > sample_size:
        sample = sample[rng.choice(sample.size, sample_size, replace=False)]
    res_i = stats["res_idx"][sample] - offset
    res_j = stats["nbr_res_idx"][sample] - offset
    dist_ij = stats["dist_3d"][sample]

    for order in itertools.permutations(letters):
        seq, N, CA, C, O = _concat_resolved(chains, order)
        if CA.shape[0] <= max(res_i.max(), res_j.max()):
            continue
        computed = np.linalg.norm(CA[res_i] - CA[res_j], axis=-1)
        if np.allclose(computed, dist_ij, atol=1e-2):
            return list(order), seq, N, CA, C, O

    raise RuntimeError(
        f"could not determine chain concatenation order for protein_idx="
        f"{protein_idx} (chains {letters}) — no permutation's geometry "
        f"matches stats.npz's recorded distances")


def resolve_structure(pdb_chain: str, h5_file: str, max_length: int, stats: dict,
                      protein_idx: int, offset: int, rng: np.random.Generator,
                      max_attempts: int = 20):
    """_load_chains + pick_chain_order, retried: _loader_h5 draws a random
    assembly per call (see module docstring), so a single attempt can come
    back with a different SET of chains than the original stats.py run
    used, not just a different order. Retrying re-draws that assembly each
    time; with a small number of real candidate assemblies, a handful of
    attempts finds the one matching stats.npz's own recorded geometry."""
    last_err = None
    for _ in range(max_attempts):
        chains = _load_chains(pdb_chain, h5_file, max_length)
        try:
            return (chains, *pick_chain_order(chains, stats, protein_idx, offset, rng))
        except RuntimeError as exc:
            last_err = exc
    raise RuntimeError(
        f"{pdb_chain}: could not match any assembly/chain-order to stats.npz's "
        f"recorded geometry for protein_idx={protein_idx} after {max_attempts} "
        f"attempts (last error: {last_err})")


def virtual_cbeta_np(N: np.ndarray, CA: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Numpy mirror of pmpnn.model.virtual_cbeta (same PMPNN_CB_COEFF_* formula)."""
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    return PMPNN_CB_COEFF_A * a + PMPNN_CB_COEFF_B * b + PMPNN_CB_COEFF_C * c + CA


# ---------------------------------------------------------------------------
# Build + write one example
# ---------------------------------------------------------------------------

def build_example(stats: dict, hit: dict, pdb_chain: str, h5_file: str,
                  max_length: int, rng: np.random.Generator, out_path: str,
                  label: str, edge_seq_dist_min: int) -> None:
    p = hit["protein_idx"]
    offset = int(stats["res_idx"][stats["protein_idx"] == p].min())

    _, order, seq, N, CA, C, O = resolve_structure(
        pdb_chain, h5_file, max_length, stats, p, offset, rng)
    CB = virtual_cbeta_np(N, CA, C)

    i_local = hit["res_idx"] - offset
    nbrs, nbr_scores = collect_environment(stats, p, hit["res_idx"], edge_seq_dist_min)
    nbrs_local = nbrs - offset

    residues = []

    def add(local_idx: int, b_iso: float):
        aa = seq[local_idx]
        resname = _AA_1TO3.get(aa, "UNK")
        atoms = {"N": N[local_idx], "CA": CA[local_idx], "C": C[local_idx],
                 "O": O[local_idx], "CB": CB[local_idx]}
        residues.append((local_idx + 1, resname, atoms, b_iso))

    add(i_local, 1.0)  # attending residue: neutral/baseline B-factor
    for nl, sc in zip(nbrs_local, nbr_scores):
        add(int(nl), float(sc))
    residues.sort(key=lambda r: r[0])

    i_aa = _AA_1TO3.get(seq[i_local], "UNK")
    j_local = hit["nbr_res_idx"] - offset
    j_aa = _AA_1TO3.get(seq[j_local], "UNK")
    if hit["relaxed"]:
        bin_note = "raw score, no bin met --min_count for all 4 groups"
    else:
        bin_note = f"{2.0 ** hit['bin_score']:.2f}x vs background"

    comment = (
        f"extract_edge_example.py: {label} attention environment "
        f"(rank {hit['rank']} edge in its hotspot bin)\n"
        f"source structure: {pdb_chain} (chains {''.join(order)})\n"
        f"ATTENDING residue: chain A, auth_seq_id {i_local + 1} ({i_aa}) "
        f"-- B-factor = 1.0 (neutral baseline), all other residues are its "
        f"neighbours, B-factor = that edge's budget-weighted score\n"
        f"rank {hit['rank']} edge: attending -> chain A auth_seq_id {j_local + 1} "
        f"({j_aa}), score={hit['score']:.3f}\n"
        f"hotspot bin: {bin_note}, n(A->B)={hit['bin_count']}, "
        f"d={hit['dist']:.2f} A, theta={np.degrees(hit['theta']):.1f} deg, "
        f"phi={np.degrees(hit['phi']):.1f} deg"
    )
    write_backbone_cif(f"{pdb_chain}_{label}", [("A", residues)], out_path, comment=comment)

    print(f"  {label}: {pdb_chain} (chains {''.join(order)})  "
         f"attending res {i_local + 1} {i_aa}  "
         f"rank {hit['rank']} neighbour res {j_local + 1} {j_aa}  score={hit['score']:.3f}")
    print(f"    bin: {bin_note}  (n_A->B={hit['bin_count']})  "
         f"d={hit['dist']:.2f} Å  theta={np.degrees(hit['theta']):.1f}°  "
         f"phi={np.degrees(hit['phi']):.1f}°  "
         f"environment={len(residues) - 1} neighbour(s)")
    print(f"    saved {out_path}")


# ---------------------------------------------------------------------------
# Stats loading (mirrors vis.__main__._load_stats, one layer only)
# ---------------------------------------------------------------------------

def load_stats_for_layer(path: str, layer: int) -> tuple[dict, int]:
    raw = np.load(path, allow_pickle=False)
    L = int(raw["L"])
    if not (0 <= layer < L):
        raise ValueError(f"--layer must be in [1, {L}] for this stats file, got {layer + 1}")
    if f"scored_attn_{layer}" not in raw:
        raise ValueError(f"stats file {path!r} is missing 'scored_attn_{layer}'")

    is_self = raw["is_self"].astype(bool)
    non_self = ~is_self

    nbr_Ca = raw["nbr_Ca"][non_self]
    dist = np.linalg.norm(nbr_Ca, axis=-1)
    d_hat = nbr_Ca / np.clip(dist, 1e-8, None)[:, None]
    theta = np.arccos(np.clip(d_hat[:, 0], -1.0, 1.0))
    phi = np.arctan2(d_hat[:, 1], d_hat[:, 2])

    stats = dict(
        seq_dist=raw["seq_dist"][non_self].astype(np.float32),
        dist_3d=dist.astype(np.float32),
        theta=theta.astype(np.float32),
        phi=phi.astype(np.float32),
        res_type_i=raw["res_type_i"][non_self].astype(np.int64),
        res_type_j=raw["res_type_j"][non_self].astype(np.int64),
        protein_idx=raw["protein_idx"][non_self].astype(np.int64),
        res_idx=raw["res_idx"][non_self].astype(np.int64),
        nbr_res_idx=raw["nbr_res_idx"][non_self].astype(np.int64),
        score=raw[f"scored_attn_{layer}"][non_self].astype(np.float64),
    )
    return stats, L


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stats_npz", help=".npz statistics file produced by stats.py")
    p.add_argument("--h5_file", default="data/pdb_2021aug02.h5",
                   help="HDF5 structure dataset (must be the one stats.npz was built from)")
    p.add_argument("--ids_file", default="data/test_monomers.txt",
                   help="Structure ID list (must be the EXACT list stats.npz was built from)")
    p.add_argument("--max_protein_length", type=int, default=DATA_MAX_PROTEIN_LENGTH,
                   help="Must match the --max_protein_length stats.py was run with")
    p.add_argument("--out_dir", default="edge_examples", help="Output directory")
    p.add_argument("--layer", type=int, default=1, help="1-indexed encoder layer")
    p.add_argument("--n_examples", type=int, default=3,
                   help="Number of examples per pair type — the N highest-scoring "
                        "A->B edges within the chosen hotspot bin (fewer if the bin "
                        "doesn't have that many A->B edges)")

    pr = p.add_argument_group("Residue-pair types")
    pr.add_argument("--pairs", nargs="+", metavar="SRC:NBR",
                    default=["AILV:AILV", "KR:DE", "DE:KR", "C:C", "ST:FWY"],
                    help="Pair types to extract an example for, as 'SRC:NBR'")

    f = p.add_argument_group("Edge filters (match the vis edge-type defaults)")
    f.add_argument("--edge_seq_dist_min", type=int, default=6)
    f.add_argument("--dist_min", type=float, default=4.0)
    f.add_argument("--dist_max", type=float, default=12.0)

    b = p.add_argument_group("Hotspot bin")
    b.add_argument("--n_dist", type=int, default=8)
    b.add_argument("--n_theta", type=int, default=6)
    b.add_argument("--n_phi", type=int, default=8)
    b.add_argument("--min_count", type=int, default=5,
                   help="Minimum edges per group (A->B, A->notB, notA->B, notA->notB) "
                        "for a bin to be a candidate hotspot")
    b.add_argument("--background", choices=["interaction", "pooled"], default="interaction",
                   help="interaction: A->B - A->notB - notA->B + notA->notB.  "
                        "pooled: A->B - mean(A->notB U notA->B)")

    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the chain-order geometric cross-check sampling")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_examples < 1:
        raise ValueError(f"--n_examples must be >= 1, got {args.n_examples}")
    os.makedirs(args.out_dir, exist_ok=True)
    layer = args.layer - 1

    print(f"[extract_edge_example] loading {args.stats_npz!r} (layer {args.layer}) ...")
    stats, L = load_stats_for_layer(args.stats_npz, layer)
    print(f"  {L} encoder layer(s)  {stats['dist_3d'].size:,} non-self edges")

    pairs_meta, pairs_info = _parse_pairs(args.pairs)

    print(f"[extract_edge_example] finding top {args.n_examples} edge(s) per pair ...")
    hits = {}
    for info, (A, B) in zip(pairs_info, pairs_meta):
        edges = find_top_edges(
            stats, A, B, args.edge_seq_dist_min, args.dist_min, args.dist_max,
            args.n_dist, args.n_theta, args.n_phi, args.min_count, args.background,
            args.n_examples,
        )
        if not edges:
            print(f"  {info['lab_a']}→{info['lab_b']}: no edges in window — skipping")
            continue
        if len(edges) < args.n_examples:
            print(f"  {info['lab_a']}→{info['lab_b']}: only {len(edges)}/{args.n_examples} "
                 f"A->B edge(s) in the hotspot bin")
        hits[info["tag"]] = (info, edges)

    if not hits:
        print("[extract_edge_example] nothing to write — no pair had any matching edges")
        return

    needed = {hit["protein_idx"] for _, edges in hits.values() for hit in edges}
    print(f"[extract_edge_example] resolving {len(needed)} protein_idx -> structure "
         f"against {args.ids_file!r} ...")
    mapping = resolve_protein_chains(args.ids_file, args.h5_file, args.max_protein_length, needed)

    print("[extract_edge_example] building structures ...")
    rng = np.random.default_rng(args.seed)
    n_written = 0
    for tag, (info, edges) in hits.items():
        label_base = f"{info['aa_a']}_to_{info['aa_b']}"  # bracket-free, safe as a CIF block name
        for hit in edges:
            pdb_chain = mapping[hit["protein_idx"]]
            out_path = os.path.join(
                args.out_dir, f"example_{tag}_top{hit['rank']}_layer{args.layer}.cif")
            label = f"{label_base}_top{hit['rank']}"
            build_example(stats, hit, pdb_chain, args.h5_file, args.max_protein_length,
                          rng, out_path, label, args.edge_seq_dist_min)
            n_written += 1

    print(f"[extract_edge_example] Done — {n_written} structure(s) written to {args.out_dir!r}")


if __name__ == "__main__":
    main()
