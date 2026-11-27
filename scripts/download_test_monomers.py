"""
download_test_monomers.py
==========================
Download the test_monomers / test_heteromers / test_homomers structures from
RCSB, keep only the ones with an SSBOND (disulfide bond) annotation for
their target chain, and write each surviving entry as a single-chain .pdb
file ready to be used directly as --pdb_path for pmpnn-gen (or any other
script that takes a directory of .pdb files), e.g.

    pmpnn-gen data/test_monomers_pdb <out_folder> <checkpoint_path>

For every "PDBID_CHAIN" entry in each IDs file (default: the three files
under index/ — test_monomers.txt, test_heteromers.txt, test_homomers.txt),
this:
  1. Downloads the full PDBID.pdb from RCSB (cached under --cache_dir,
     shared across all IDs files, so re-runs and overlapping PDB IDs
     across splits don't re-download).
  2. Parses the file's SSBOND records and keeps the entry only if CHAIN
     has at least one *intra-chain* disulfide bond (both partners on
     CHAIN) — an inter-chain-only bond isn't observable once the other
     chain is dropped, so it doesn't count.
  3. Writes a single-chain PDB file <out_root>/<stem>_pdb/<PDBID>_<CHAIN>.pdb
     (stem = the IDs file's basename, e.g. "test_monomers") containing that
     chain's matching SSBOND header record(s) followed by its ATOM/HETATM/TER
     records, so it parses with exactly one chain
     (src.gen.__main__.parse_PDB then masks/redesigns all of it by default,
     matching how this entry is meant to be used) while still carrying its
     own disulfide-bond ground truth for downstream analysis (e.g.
     vis.ss_bond_analysis).

Usage
-----
    python scripts/download_test_monomers.py
    python scripts/download_test_monomers.py --ids_files index/test_monomers.txt \\
        --out_root data --cache_dir data/test_pdb_raw
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request

from joblib import Parallel, delayed

_RCSB_URL = "https://files.rcsb.org/download/{pdbid}.pdb"

_DEFAULT_IDS_FILES = [
    "index/test_monomers.txt",
    "index/test_heteromers.txt",
    "index/test_homomers.txt",
]


def _read_ids(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _download_pdb(pdbid: str, cache_dir: str, n_retries: int = 3, timeout: float = 30.0) -> str | None:
    """Download PDBID.pdb to cache_dir (skipped if already cached). Return the local path, or None on failure."""
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{pdbid.upper()}.pdb")
    if os.path.isfile(out_path):
        return out_path

    url = _RCSB_URL.format(pdbid=pdbid.upper())
    for attempt in range(1, n_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            tmp_path = out_path + f".part{os.getpid()}"
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.replace(tmp_path, out_path)
            return out_path
        except urllib.error.URLError as exc:
            if attempt == n_retries:
                print(f"  download failed for {pdbid}: {exc}")
                return None
            time.sleep(2 ** attempt)
    return None


def _intra_chain_ssbonds(pdb_path: str, chain: str) -> list[str]:
    """Return the raw SSBOND lines whose both partners are on *chain*."""
    lines = []
    with open(pdb_path, errors="replace") as f:
        for line in f:
            if not line.startswith("SSBOND"):
                continue
            # PDB format spec: chainID1 at column 16, chainID2 at column 30 (1-indexed).
            if line[15].strip() == chain and line[29].strip() == chain:
                lines.append(line if line.endswith("\n") else line + "\n")
    return lines


def _extract_chain(pdb_path: str, chain: str, ssbond_lines: list[str], out_path: str) -> bool:
    """
    Write a single-chain PDB file: ssbond_lines verbatim, followed by chain's
    ATOM/HETATM/TER records. Return False if the chain has no atoms.
    """
    lines = list(ssbond_lines)
    with open(pdb_path, errors="replace") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM", "TER") and len(line) > 21 and line[21] == chain:
                lines.append(line if line.endswith("\n") else line + "\n")
    if not any(line[:6].strip() in ("ATOM", "HETATM") for line in lines):
        return False
    lines.append("END\n")
    with open(out_path, "w") as f:
        f.writelines(lines)
    return True


def process_one(entry: str, cache_dir: str, out_dir: str) -> tuple[str, str]:
    """Return (entry, status); status is "ok", "download_failed", "no_ssbond", or "no_chain_atoms"."""
    pdbid, chain = entry.split("_")
    raw_path = _download_pdb(pdbid, cache_dir)
    if raw_path is None:
        return entry, "download_failed"

    ssbond_lines = _intra_chain_ssbonds(raw_path, chain)
    if not ssbond_lines:
        return entry, "no_ssbond"

    out_path = os.path.join(out_dir, f"{entry}.pdb")
    if not _extract_chain(raw_path, chain, ssbond_lines, out_path):
        return entry, "no_chain_atoms"
    return entry, "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ids_files", nargs="+", default=_DEFAULT_IDS_FILES,
                    help="Text file(s) of PDBID_CHAIN entries, one per line. Each "
                         "file's filtered structures are written to its own "
                         "<out_root>/<stem>_pdb/ directory")
    p.add_argument("--out_root", default="data",
                    help="Parent directory for the per-IDs-file output directories")
    p.add_argument("--cache_dir", default="data/test_pdb_raw",
                    help="Where to cache the full downloaded .pdb files (shared across --ids_files)")
    p.add_argument("--n_jobs", type=int, default=4,
                    help="Parallel downloads (kept modest to stay polite to RCSB)")
    return p.parse_args()


def process_ids_file(ids_file: str, cache_dir: str, out_dir: str, n_jobs: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    entries = _read_ids(ids_file)
    print(f"[download_test_monomers] {len(entries)} entries in {ids_file}")

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_one)(entry, cache_dir, out_dir) for entry in entries
    )

    counts: dict[str, int] = {}
    for _, status in results:
        counts[status] = counts.get(status, 0) + 1

    print(f"[download_test_monomers] {os.path.basename(ids_file)} done:")
    for status in ("ok", "no_ssbond", "no_chain_atoms", "download_failed"):
        if status in counts:
            print(f"  {status:16s} {counts[status]}")
    print(f"  -> {counts.get('ok', 0)}/{len(entries)} structures written to {out_dir}")


def main() -> None:
    args = parse_args()

    for ids_file in args.ids_files:
        stem = os.path.splitext(os.path.basename(ids_file))[0]
        out_dir = os.path.join(args.out_root, f"{stem}_pdb")
        process_ids_file(ids_file, args.cache_dir, out_dir, args.n_jobs)


if __name__ == "__main__":
    main()
