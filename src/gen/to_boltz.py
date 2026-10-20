"""
to_boltz.py
===========
Convert pmpnn-gen FASTA output into per-sequence Boltz YAML input files.

Each FASTA file under the input directory (one per structure, as produced by
pmpnn-gen) contains one record per temperature/sample combination, with
headers like:

    >1ami_A, score=0.8393, recovery=0.5246, blosum=2.695, T=0.10, sample=1

Every record is converted into its own Boltz input YAML (no MSA, matching
prot_no_msa.yaml), with one `protein` entry per chain (chains in the FASTA
sequence are separated by "/", as written by pmpnn-gen for multi-chain
designs). Output files are named "<stem>_T<temp>_sample<sample>.yaml" and
written to a "seqs_boltz" directory alongside the input directory.

Usage
-----
    python -m gen.to_boltz <seqs_dir> [--out_dir <dir>]
"""
from __future__ import annotations

import argparse
import os
import re
import string

import yaml

_TAG_RE = re.compile(r"T=([0-9.]+).*?sample=(\d+)")


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Return a list of (header, sequence) pairs parsed from *path*."""
    entries: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
    if header is not None:
        entries.append((header, "".join(seq_parts)))
    return entries


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def record_tag(header: str, index: int) -> str:
    """Derive a short, unique tag from a pmpnn-gen FASTA header."""
    m = _TAG_RE.search(header)
    if m:
        temp, sample = m.groups()
        return f"T{temp}_sample{sample}"
    return f"seq{index}"


def to_boltz_yaml(sequence: str) -> dict:
    """Build a Boltz input dict for *sequence* (chains separated by '/')."""
    chains = sequence.split("/")
    return {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": string.ascii_uppercase[i],
                    "sequence": chain,
                    "msa": "empty",
                }
            }
            for i, chain in enumerate(chains)
        ],
    }


def convert_file(fasta_path: str, out_dir: str) -> int:
    stem = os.path.splitext(os.path.basename(fasta_path))[0]
    n_written = 0
    for index, (header, sequence) in enumerate(parse_fasta(fasta_path), start=1):
        if not sequence:
            continue
        tag = record_tag(header, index)
        out_path = os.path.join(out_dir, f"{stem}_{tag}.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(to_boltz_yaml(sequence), f, default_flow_style=False, sort_keys=False)
        n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert pmpnn-gen FASTA output into per-sequence Boltz YAML input files."
    )
    p.add_argument("seqs_dir", help="Directory of pmpnn-gen .fa files")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: '<seqs_dir>_boltz', alongside seqs_dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seqs_dir = os.path.normpath(args.seqs_dir)
    if not os.path.isdir(seqs_dir):
        raise ValueError(f"seqs_dir does not exist or is not a directory: {seqs_dir!r}")

    out_dir = args.out_dir or f"{seqs_dir}_boltz"
    os.makedirs(out_dir, exist_ok=True)

    fasta_files = sorted(
        os.path.join(seqs_dir, f) for f in os.listdir(seqs_dir) if f.lower().endswith(".fa")
    )

    total = 0
    for fasta_path in fasta_files:
        n_written = convert_file(fasta_path, out_dir)
        total += n_written
        print(f"{os.path.basename(fasta_path)}: {n_written} sequence(s) -> {out_dir}")

    print(f"Done. {total} Boltz YAML file(s) written to {out_dir}")


if __name__ == "__main__":
    main()
