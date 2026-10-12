"""
pt_to_h5.py
===============
Mirror the raw ProteinMPNN .pt file collection into a single HDF5 file,
preserving every field from both PDBID.pt (metadata) and PDBID_CHAINID.pt
(chain data) without applying any assembly transforms or pre-packing.

HDF5 layout
-----------
    /<PDBID>/
        meta/
            attrs : method (str), date (str), resolution (float)
            chains        [C]         variable-length strings
            tm            [C, C, 3]   float32
            asmb_ids      [A]         variable-length strings
            asmb_details  [A]         variable-length strings
            asmb_method   [A]         variable-length strings
            asmb_chains   [A]         variable-length strings
            asmb_xform{k} [n, 4, 4]  float32  (one dataset per assembly block)
        chains/
            <CHAINID>/
                seq   scalar string dataset
                xyz   [L, 14, 3]  float32
                mask  [L, 14]     bool
                bfac  [L, 14]     float32
                occ   [L, 14]     float32

Usage
-----
    python pt_to_h5.py \\
        --dir   /data/pdb \\
        --out   /data/pmpnn.h5 \\
        --gzip  4 \\
        --n_jobs -1
"""
from __future__ import annotations

import argparse
import glob
import logging
import logging.handlers
import os
import time

import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: str | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt_console = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                    datefmt="%H:%M:%S")
    fmt_file    = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    root.addHandler(ch)
    if log_file:
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)
        root.addHandler(fh)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_pdbids(pdb_dir: str) -> list[str]:
    """
    Collect all PDBID strings that have a corresponding PDBID.pt metadata file
    under pdb_dir/pdb/XX/PDBID.pt (the standard ProteinMPNN directory tree).
    """
    pattern = os.path.join(pdb_dir, "pdb", "??", "????.pt")
    meta_files = glob.glob(pattern)
    # Also handle flat layouts (all .pt files directly under pdb_dir/pdb/)
    if not meta_files:
        pattern = os.path.join(pdb_dir, "pdb", "????.pt")
        meta_files = glob.glob(pattern)
    pdbids = []
    for path in meta_files:
        base = os.path.basename(path)
        # PDBID.pt has exactly 4 + 3 = 7 chars (e.g. 1abc.pt)
        # chain files are PDBID_CHAINID.pt — filter them out
        if "_" not in base:
            pdbids.append(base[:-3])   # strip ".pt"
    log.info("Found %d PDBID metadata files under %s", len(pdbids), pdb_dir)
    return sorted(pdbids)


def _prefix(pdb_dir: str, pdbid: str) -> str:
    return os.path.join(pdb_dir, "pdb", pdbid[1:3], pdbid)


# ---------------------------------------------------------------------------
# Per-PDBID worker  (pure numpy / torch, no HDF5)
# ---------------------------------------------------------------------------

_OK_TAG   = "ok"
_SKIP_TAG = "skip"


def load_pdbid(pdbid: str, pdb_dir: str) -> tuple:
    """
    Load PDBID.pt and all PDBID_CHAINID.pt files.

    Returns
    -------
    (_OK_TAG,   pdbid, meta_dict, chain_dicts)
    (_SKIP_TAG, pdbid, reason)

    meta_dict keys  (mirrors PDBID.pt, values converted to plain numpy/str)
        method, date, resolution, chains,
        tm, asmb_ids, asmb_details, asmb_method, asmb_chains,
        asmb_xforms  : dict  {k: np.ndarray [n,4,4]}

    chain_dicts keys  (mirrors PDBID_CHAINID.pt)
        {chid: {seq, xyz, mask, bfac, occ}}
    """
    prefix = _prefix(pdb_dir, pdbid)
    meta_path = prefix + ".pt"

    if not os.path.isfile(meta_path):
        return _SKIP_TAG, pdbid, "missing_meta"

    try:
        raw = torch.load(meta_path, weights_only=False)
    except Exception as exc:
        log.debug("Cannot load %s: %s", meta_path, exc)
        return _SKIP_TAG, pdbid, "corrupt_meta"

    # Safe metadata 
    res_val    = raw.get("resolution", 0.0)
    resolution = float(res_val) if res_val is not None else 0.0

    chains_val = raw.get("chains",       [])
    chains     = [str(c) for c in chains_val] if chains_val is not None else []

    asmb_ids_val = raw.get("asmb_ids",       [])
    asmb_ids     = [str(c) for c in asmb_ids_val] if asmb_ids_val is not None else []

    asmb_details_val = raw.get("asmb_details",       [])
    asmb_details     = [str(c) for c in asmb_details_val] if asmb_details_val is not None else []

    asmb_method_val = raw.get("asmb_method",       [])
    asmb_method     = [str(c) for c in asmb_method_val] if asmb_method_val is not None else []

    asmb_chains_val = raw.get("asmb_chains",       [])
    asmb_chains     = [str(c) for c in asmb_chains_val] if asmb_chains_val is not None else []

    # -- Normalise metadata ---------------------------------------------------
    meta: dict = {
        "method":      str(raw.get("method",     "")),
        "date":        str(raw.get("date",        "")),
        "resolution":  resolution,
        "chains":      chains,
        "asmb_ids":    asmb_ids,
        "asmb_details":asmb_details,
        "asmb_method": asmb_method,
        "asmb_chains": asmb_chains,
    }

    # tm: can be Tensor or ndarray
    tm_raw = raw.get("tm", None)
    if tm_raw is not None:
        meta["tm"] = (tm_raw.numpy() if hasattr(tm_raw, "numpy")
                      else np.asarray(tm_raw)).astype(np.float32)
    else:
        C = len(meta["chains"])
        meta["tm"] = np.zeros((C, C, 3), dtype=np.float32)

    # asmb_xform{k}: one per assembly block
    xforms: dict[int, np.ndarray] = {}
    k = 0
    while True:
        key = f"asmb_xform{k}"
        if key not in raw:
            break
        xf = raw[key]
        xforms[k] = (xf.numpy() if hasattr(xf, "numpy")
                     else np.asarray(xf)).astype(np.float32)
        k += 1
    meta["asmb_xforms"] = xforms

    # -- Load each chain file -------------------------------------------------
    chain_dicts: dict[str, dict] = {}
    for chid in meta["chains"]:
        chain_path = f"{prefix}_{chid}.pt"
        if not os.path.isfile(chain_path):
            log.debug("%s chain %s: file not found", pdbid, chid)
            continue
        try:
            ch = torch.load(chain_path, weights_only=False)
        except Exception as exc:
            log.debug("%s chain %s: load error — %s", pdbid, chid, exc)
            continue

        def _to_f32(x):
            return (x.numpy() if hasattr(x, "numpy") else np.asarray(x)).astype(np.float32)

        def _to_bool(x):
            arr = (x.numpy() if hasattr(x, "numpy") else np.asarray(x))
            return arr.astype(bool)

        chain_dicts[chid] = {
            "seq":  str(ch["seq"]),
            "xyz":  _to_f32(ch["xyz"]),     # [L, 14, 3]
            "mask": _to_bool(ch["mask"]),   # [L, 14]
            "bfac": _to_f32(ch["bfac"]),    # [L, 14]
            "occ":  _to_f32(ch["occ"]),     # [L, 14]
        }

    if not chain_dicts:
        return _SKIP_TAG, pdbid, "no_chains_loaded"

    return _OK_TAG, pdbid, meta, chain_dicts


# ---------------------------------------------------------------------------
# HDF5 writer  (serial — HDF5 is not thread-safe for parallel writes)
# ---------------------------------------------------------------------------

_STR_DT = h5py.string_dtype()   # variable-length UTF-8


def _write_strlist(group: h5py.Group, name: str, strings: list[str]) -> None:
    arr = np.array(strings, dtype=object)
    group.create_dataset(name, data=arr, dtype=_STR_DT)


def write_pdbid(
    hf:          h5py.File,
    pdbid:       str,
    meta:        dict,
    chain_dicts: dict[str, dict],
    gzip_level:  int,
) -> None:
    pg = hf.require_group(pdbid)

    # -- meta group -----------------------------------------------------------
    mg = pg.require_group("meta")
    mg.attrs["method"]     = meta["method"]
    mg.attrs["date"]       = meta["date"]
    mg.attrs["resolution"] = meta["resolution"]

    opts = dict(compression="gzip", compression_opts=gzip_level, shuffle=True)

    _write_strlist(mg, "chains",       meta["chains"])
    _write_strlist(mg, "asmb_ids",     meta["asmb_ids"])
    _write_strlist(mg, "asmb_details", meta["asmb_details"])
    _write_strlist(mg, "asmb_method",  meta["asmb_method"])
    _write_strlist(mg, "asmb_chains",  meta["asmb_chains"])

    mg.create_dataset("tm", data=meta["tm"], **opts)

    for k, xf in meta["asmb_xforms"].items():
        mg.create_dataset(f"asmb_xform{k}", data=xf, **opts)

    # -- chains group ---------------------------------------------------------
    cg = pg.require_group("chains")
    for chid, ch in chain_dicts.items():
        chg = cg.require_group(chid)
        chg.create_dataset("seq",  data=ch["seq"],  dtype=_STR_DT)
        chg.create_dataset("xyz",  data=ch["xyz"],  **opts)       # [L,14,3]
        chg.create_dataset("mask", data=ch["mask"],
                           compression="gzip", compression_opts=gzip_level)
        chg.create_dataset("bfac", data=ch["bfac"], **opts)       # [L,14]
        chg.create_dataset("occ",  data=ch["occ"],  **opts)       # [L,14]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mirror raw ProteinMPNN .pt files into a single HDF5 file."
    )
    p.add_argument("--dir",      required=True, help="Root data directory (contains pdb/)")
    p.add_argument("--out",      required=True, help="Output .h5 file path")
    p.add_argument("--gzip",     default=4,     type=int, help="GZIP compression level (1-9)")
    p.add_argument("--n_jobs",   default=-1,    type=int, help="joblib workers (-1 = all CPUs)")
    p.add_argument("--log_file", default=None,            help="Optional DEBUG log file path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_file)
    t0 = time.perf_counter()

    log.info("=" * 60)
    log.info("pt_to_h5_raw  —  starting")
    log.info("  --dir    %s", args.dir)
    log.info("  --out    %s", args.out)
    log.info("  --gzip   %d", args.gzip)
    log.info("  --n_jobs %d", args.n_jobs)
    log.info("=" * 60)

    pdbids = discover_pdbids(args.dir)
    if not pdbids:
        log.error("No PDBID.pt files found under %s — check --dir.", args.dir)
        return

    log.info("Launching parallel load for %d PDBIDs with %d workers ...",
             len(pdbids), args.n_jobs)

    results = Parallel(
        n_jobs    = args.n_jobs,
        backend   = "loky",
        return_as = "generator",
    )(
        delayed(load_pdbid)(pdbid, args.dir)
        for pdbid in pdbids
    )

    n_written = n_skipped = 0
    skip_reasons: dict[str, int] = {}

    with h5py.File(args.out, "w") as hf:
        with tqdm(total=len(pdbids), desc="writing", unit="pdb",
                  dynamic_ncols=True) as pbar:
            for result in results:
                tag = result[0] # type: ignore
                if tag == _SKIP_TAG:
                    _, pdbid, reason = result # type: ignore
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    n_skipped += 1
                    log.debug("SKIP  %s  (%s)", pdbid, reason)
                else:
                    _, pdbid, meta, chain_dicts = result # type: ignore
                    try:
                        write_pdbid(hf, pdbid, meta, chain_dicts, args.gzip)
                        n_written += 1
                    except Exception as exc:
                        log.debug("Write error for %s: %s", pdbid, exc)
                        skip_reasons["write_error"] = skip_reasons.get("write_error", 0) + 1
                        n_skipped += 1

                pbar.set_postfix(written=n_written, skipped=n_skipped, refresh=False)
                pbar.update(1)

    elapsed = time.perf_counter() - t0
    try:
        size_mb = os.path.getsize(args.out) / 1024 ** 2
    except OSError:
        size_mb = 0.0

    log.info("=" * 60)
    log.info("Written  : %d PDBIDs", n_written)
    log.info("Skipped  : %d", n_skipped)
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        log.info("  %-24s  %d", reason, count)
    log.info("Output   : %s  (%.1f MB)", args.out, size_mb)
    log.info("Wall time: %.1f s", elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()