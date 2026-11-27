"""
sanitize_boltz.py
==================
Reduce each Boltz-predicted structure to a Cα-only mmCIF, matching the
format written by scripts/extract_native.py for native structures, so the
two can be compared directly (e.g. by scripts/tmscore.py).

For every model/seed directory under models/exp2_finetuning/, this reads
every full-atom structure under

  eval/structs_boltz/boltz_results_seqs_boltz/predictions/<stem_tag>/
  <stem_tag>_model_0.cif

and writes a Cα-only version to

  eval/structs_boltz_ca/<stem_tag>.cif

A structure is skipped (and counted) if its file is missing, unreadable,
or has no chain with a Cα atom.

Usage
-----
    python scripts/sanitize_boltz.py
    python scripts/sanitize_boltz.py models/exp2_finetuning/oa1_seed0
"""
from __future__ import annotations

import argparse
import os

import gemmi

from _boltz_utils import discover_stem_tags, predictions_dir, raw_predicted_cif_path
from _cif_utils import write_ca_only_cif


def _extract_ca_chains(path: str) -> list:
    """Return a list of (chain_letter, residues) for scripts/_cif_utils.write_ca_only_cif."""
    structure = gemmi.read_structure(path)
    if len(structure) == 0:
        return []

    chains = []
    for chain in structure[0]:
        residues = []
        for residue in chain:
            ca = residue.find_atom("CA", "*")
            if ca is None:
                continue
            residues.append((residue.seqid.num, residue.name, (ca.pos.x, ca.pos.y, ca.pos.z)))
        if residues:
            chains.append((chain.name, residues))
    return chains


def sanitize_one(in_path: str, out_path: str) -> bool:
    """Write a Cα-only mmCIF for *in_path* to *out_path*. Return False if skipped."""
    try:
        chains = _extract_ca_chains(in_path)
    except Exception as e:
        print(f"  skipping {in_path} ({e})")
        return False

    if not chains:
        print(f"  skipping {in_path} (no Cα atoms found)")
        return False

    name = os.path.splitext(os.path.basename(out_path))[0]
    write_ca_only_cif(name, chains, out_path)
    return True


def process_model_dir(model_dir: str) -> None:
    stem_tags = discover_stem_tags(model_dir)
    if not stem_tags:
        print(f"  {model_dir}: no predictions found, skipping")
        return

    out_dir = os.path.join(model_dir, "eval", "structs_boltz_ca")
    os.makedirs(out_dir, exist_ok=True)

    n_ok, n_skipped = 0, 0
    for stem_tag in stem_tags:
        in_path = raw_predicted_cif_path(model_dir, stem_tag)
        if not os.path.isfile(in_path):
            n_skipped += 1
            continue
        out_path = os.path.join(out_dir, f"{stem_tag}.cif")
        if sanitize_one(in_path, out_path):
            n_ok += 1
        else:
            n_skipped += 1

    print(f"  {model_dir}: {n_ok} sanitized, {n_skipped} skipped -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "model_dirs", nargs="*",
        help="Model/seed directories to process, e.g. 'models/exp2_finetuning/oa1_seed0'. "
             "Defaults to every subdirectory of --root with a Boltz predictions folder.",
    )
    p.add_argument(
        "--root", default="models/exp2_finetuning",
        help="Experiment root directory, used to auto-discover model_dirs when none are given.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_dirs:
        model_dirs = list(args.model_dirs)
    else:
        model_dirs = sorted(
            os.path.join(args.root, name)
            for name in os.listdir(args.root)
            if os.path.isdir(predictions_dir(os.path.join(args.root, name)))
        )
    if not model_dirs:
        raise ValueError(f"No model/seed directories with Boltz predictions found under {args.root!r}")

    print(f"[sanitize_boltz] Processing {len(model_dirs)} model/seed director(y/ies)")
    for model_dir in model_dirs:
        process_model_dir(model_dir)


if __name__ == "__main__":
    main()
