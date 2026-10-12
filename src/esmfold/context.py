"""
context.py
==========
CLI definition and validated configuration dataclass for the ESMFold application.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from .constants import DEFAULT_CHUNK_SIZE


@dataclass
class ESMFoldContext:
    # ── I/O ───────────────────────────────────────────────────────────────────
    fasta_files: list[Path]
    out_dir: Path

    # ── Output options ────────────────────────────────────────────────────────
    save_pdb: bool

    # ── Inference ─────────────────────────────────────────────────────────────
    chunk_size: int
    cpu: bool

    # ── Runtime state (populated during setup, not from CLI) ──────────────────
    device: object = field(default=None, repr=False)
    model:  object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # ── fasta_files ───────────────────────────────────────────────────────
        if not self.fasta_files:
            raise ValueError("At least one FASTA file must be provided.")
        for p in self.fasta_files:
            if not p.is_file():
                raise ValueError(f"fasta_file does not exist: {p!r}")

        # ── out_dir ───────────────────────────────────────────────────────────
        out_parent = self.out_dir.parent.resolve()
        if not out_parent.is_dir():
            raise ValueError(
                f"Parent directory of out_dir does not exist: {out_parent!r}"
            )

        # ── chunk_size ────────────────────────────────────────────────────────
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")

    @classmethod
    def from_args(cls) -> "ESMFoldContext":
        p = argparse.ArgumentParser(
            description=(
                "Run ESMFold on one or more FASTA files and annotate sequence "
                "headers with pLDDT / pTM / PAE scores."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        p.add_argument(
            "fasta_files",
            nargs="+",
            type=Path,
            help="Input FASTA file(s). Sequences may contain '/' to delimit chains.",
        )

        io = p.add_argument_group("I/O")
        io.add_argument(
            "--out_dir",
            type=Path,
            default=Path("esm_out"),
            help="Directory where annotated FASTA files (and optional PDB files) are written.",
        )
        io.add_argument(
            "--save_pdb",
            action="store_true",
            help="Also write predicted structure as PDB files under <out_dir>/pdb/.",
        )

        inf = p.add_argument_group("Inference")
        inf.add_argument(
            "--chunk_size",
            type=int,
            default=DEFAULT_CHUNK_SIZE,
            help="Trunk chunk size passed to ESMFold. Decrease to reduce peak VRAM usage.",
        )
        inf.add_argument(
            "--cpu",
            action="store_true",
            help="Force inference on CPU (slow; useful when no GPU is available).",
        )

        args = p.parse_args()
        return cls(
            fasta_files=args.fasta_files,
            out_dir=args.out_dir,
            save_pdb=args.save_pdb,
            chunk_size=args.chunk_size,
            cpu=args.cpu,
        )
