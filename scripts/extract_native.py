"""
extract_native.py
==================
Extract the native sequences and Cα coordinates for the structures used to
evaluate Experiment 2 (finetuning), so they can be compared against the
designed sequences / predicted structures under
models/exp2_finetuning/<model_name>/eval/.

For every structure ID in the IDs file (default: data/test_monomers.txt, the
same list consumed by scripts/exp2_finetuning_02_gen_monomers.sbatch), this loads the
structure from the training/eval HDF5 dataset the same way gen.__main__ does
in its HDF5 mode (pmpnn.data._loader_h5 + _to_pdb_dict) and writes, per
structure, the native sequence and Cα coordinates of the masked (redesigned)
chain(s) — the same chain(s) pmpnn-gen writes designed sequences for:

  <out_dir>/eval/seqs/<id>.fa     — native sequence(s), chains joined by
                                    "/", matching the FASTA convention
                                    written by pmpnn-gen.
  <out_dir>/eval/structs/<id>.cif — native Cα-only structure, one mmCIF
                                    chain per masked chain, written with
                                    gemmi.

Residues with unresolved (non-finite) Cα coordinates are dropped from
*both* the sequence and the structure, and the remaining residues are
renumbered contiguously from 1 — exactly mirroring
gen.__main__._seq_from_sample, which only ever emits designed residues at
positions with a finite backbone coordinate (``mask_row > 0``) and whose
resulting (gapless) sequence is what Boltz actually folds and numbers
1..L. Preserving the original (gapped) residue numbering instead would
silently misalign any downstream ``USalign ... -TMscore 1`` comparison
(which matches residues by identical seqid) for every structure with an
unresolved residue before the end of the chain — i.e. most real structures.

The layout mirrors models/exp2_finetuning/<model_name>/eval/, so --out_dir
(default: models/exp2_finetuning/native) can be treated like any other
model/seed directory, e.g. by scripts/exp2_finetuning_03_boltz.sbatch.

Usage
-----
    python scripts/extract_native.py
    python scripts/extract_native.py --h5_file data/pdb_2021aug02.h5 \\
        --ids_file data/test_monomers.txt --out_dir models/exp2_finetuning/native
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from _cif_utils import write_ca_only_cif
from pmpnn.constants import AA_3TO1, DATA_MAX_PROTEIN_LENGTH
from pmpnn.data import _loader_h5, _to_pdb_dict

_AA_1TO3 = {one: three for three, one in AA_3TO1.items() if three not in ("MSE", "SEC")}
_AA_1TO3["X"] = "UNK"


def _read_ids(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _native_chains(pdb_dict: dict) -> list[str]:
    """Chains to extract: the masked (redesigned) ones, or all chains if none are masked."""
    return pdb_dict["masked_list"] or (pdb_dict["masked_list"] + pdb_dict["visible_list"])


def extract_one(pdb_id: str, h5_file: str, max_length: int):
    """
    Return (name, sequence, chains) for one structure ID, or None if
    unavailable.  `sequence` is the "/"-joined resolved-only sequence (for
    the FASTA file); `chains` is a list of (letter, seq, ca_coords[L,3])
    tuples, one per masked chain (for the CIF file) — already filtered to
    finite Cα positions and renumbered contiguously, so both outputs match
    what a Boltz prediction from this same sequence would be numbered.
    """
    pdb_dict = _to_pdb_dict(_loader_h5([pdb_id], h5_file), max_length)
    if pdb_dict is None:
        return None

    chains = []
    for letter in _native_chains(pdb_dict):
        full_seq = pdb_dict[f"seq_chain_{letter}"]
        full_ca = np.asarray(
            pdb_dict[f"coords_chain_{letter}"][f"CA_chain_{letter}"], dtype=np.float32
        )
        resolved = np.isfinite(full_ca).all(axis=1)
        resolved_seq = "".join(aa for aa, keep in zip(full_seq, resolved) if keep)
        chains.append((letter, resolved_seq, full_ca[resolved]))

    seq = "/".join(c[1] for c in chains)
    return pdb_dict["name"], seq, chains


def _write_cif(name: str, chains: list, out_path: str) -> None:
    """Write a Cα-only mmCIF structure, one chain per (letter, seq, ca) tuple."""
    cif_chains = [
        (
            letter,
            [
                (seqid, _AA_1TO3.get(aa, "UNK"), tuple(xyz))
                for seqid, (aa, xyz) in enumerate(zip(seq, ca), start=1)
            ],
        )
        for letter, seq, ca in chains
    ]
    write_ca_only_cif(name, cif_chains, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--h5_file", default="data/pdb_2021aug02.h5",
                    help="HDF5 structure dataset")
    p.add_argument("--ids_file", default="data/test_monomers.txt",
                    help="Text file of structure IDs, one per line")
    p.add_argument("--out_dir", default="models/exp2_finetuning/native",
                    help="Output directory; writes <out_dir>/eval/seqs and <out_dir>/eval/structs")
    p.add_argument("--max_length", type=int, default=DATA_MAX_PROTEIN_LENGTH)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seqs_dir = os.path.join(args.out_dir, "eval", "seqs")
    structs_dir = os.path.join(args.out_dir, "eval", "structs")
    os.makedirs(seqs_dir, exist_ok=True)
    os.makedirs(structs_dir, exist_ok=True)

    ids = _read_ids(args.ids_file)
    n_written = 0
    for pdb_id in ids:
        result = extract_one(pdb_id, args.h5_file, args.max_length)
        if result is None:
            print(f"  skipping {pdb_id} (not found in HDF5 or filtered by max_length)")
            continue
        name, seq, chains = result

        with open(os.path.join(seqs_dir, f"{name}.fa"), "w") as f:
            f.write(f">{name}\n{seq}\n")
        _write_cif(name, chains, os.path.join(structs_dir, f"{name}.cif"))
        n_written += 1

    print(f"Done. {n_written}/{len(ids)} structure(s) written to {args.out_dir}")


if __name__ == "__main__":
    main()
