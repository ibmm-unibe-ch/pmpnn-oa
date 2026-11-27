"""
tmscore.py
==========
Compute the TM-score between each Boltz-predicted structure and its native
structure for the models produced by Experiment 2 (finetuning).

For every model/seed directory under models/exp2_finetuning/ (e.g.
``oa1_seed0``), this looks at every sanitized, Cα-only predicted structure
under

  eval/structs_boltz_ca/<stem_tag>.cif

(``stem_tag`` = ``"<pdbid_chain>_T<temp>_sample<n>"`` for FASTA headers with
``T=.../sample=...`` fields, as written by pmpnn-gen, or
``"<pdbid_chain>_seq<n>"`` for headers without them (e.g. the plain
``>pdbid_chain`` headers written by scripts/extract_native.py itself, when
running this same pipeline on the native sequences as a self-consistency
baseline) — both are ``gen.to_boltz.record_tag``'s conventions, written by
scripts/sanitize_boltz.py from the raw ``gen.to_boltz`` + ``boltz predict``
output — run that first), finds the matching native structure

  models/exp2_finetuning/native/eval/structs/<pdbid_chain>.cif

(written by scripts/extract_native.py) and runs:

  USalign <predicted.cif> <native.cif> -TMscore 1

This does not require every structure to be present on either side: a
prediction with no matching native file (or vice versa) is simply skipped,
so a few missing structures (e.g. from a native-extraction or Boltz
failure) don't block computing statistics over everything else. A
structure pair is also skipped if USalign's output can't be parsed, or if
the two structures don't have the same number of residues (the pair is
superposed by ``-TMscore 1``, i.e. assuming matching residue indices, so a
length mismatch means the comparison isn't meaningful).

Results are written to eval/structs_boltz/TMscore, a tab-separated file
with one row per successfully-compared structure:

  stem_tag  tm_score  rmsd  aligned_length  seq_id  length

Usage
-----
    python scripts/tmscore.py  [model_dir ...]  [--root models/exp2_finetuning]
        [--native_dir models/exp2_finetuning/native]  [--usalign_bin USalign]

With no model_dir arguments, every subdirectory of --root containing an
eval/structs_boltz/boltz_results_seqs_boltz/predictions folder is used.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess

from _boltz_utils import discover_stem_tags, predictions_dir

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_SUFFIX_RE = re.compile(r"_(?:T[0-9.]+_sample\d+|seq\d+)$")


def _predicted_cif_path(model_dir: str, stem_tag: str) -> str:
    return os.path.join(model_dir, "eval", "structs_boltz_ca", f"{stem_tag}.cif")


def _native_cif_path(native_dir: str, stem_tag: str) -> str:
    native_id = _SUFFIX_RE.sub("", stem_tag)
    return os.path.join(native_dir, "eval", "structs", f"{native_id}.cif")


# ---------------------------------------------------------------------------
# USalign invocation + output parsing
# ---------------------------------------------------------------------------

_LEN_RE = re.compile(r"Length of Structure_(\d):\s*(\d+) residues")
_ALIGNED_RE = re.compile(
    r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+),\s*Seq_ID=n_identical/n_aligned=\s*([\d.]+)"
)
_TM_RE = re.compile(r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_1")


def run_usalign(usalign_bin: str, predicted_cif: str, native_cif: str) -> str:
    result = subprocess.run(
        [usalign_bin, predicted_cif, native_cif, "-TMscore", "1"],
        capture_output=True, text=True,
    )
    return result.stdout


def parse_usalign_output(stdout: str):
    """
    Return (tm_score, rmsd, aligned_length, seq_id, length) parsed from
    USalign's stdout, or None if the two structures don't have the same
    number of residues or any expected field is missing.
    """
    lengths = dict(
        (int(idx), int(length)) for idx, length in _LEN_RE.findall(stdout)
    )
    if lengths.get(1) is None or lengths.get(2) is None or lengths[1] != lengths[2]:
        return None

    aligned_match = _ALIGNED_RE.search(stdout)
    tm_match = _TM_RE.search(stdout)
    if aligned_match is None or tm_match is None:
        return None

    aligned_length, rmsd, seq_id = aligned_match.groups()
    tm_score = tm_match.group(1)
    return float(tm_score), float(rmsd), int(aligned_length), float(seq_id), lengths[1]


# ---------------------------------------------------------------------------
# Per-model/seed processing
# ---------------------------------------------------------------------------

def process_model_dir(model_dir: str, native_dir: str, usalign_bin: str) -> None:
    stem_tags = discover_stem_tags(model_dir)
    if not stem_tags:
        print(f"  {model_dir}: no predictions found, skipping")
        return

    out_path = os.path.join(model_dir, "eval", "structs_boltz", "TMscore")
    n_ok, n_skipped = 0, 0

    with open(out_path, "w") as f:
        f.write("stem_tag\ttm_score\trmsd\taligned_length\tseq_id\tlength\n")
        for stem_tag in stem_tags:
            predicted_cif = _predicted_cif_path(model_dir, stem_tag)
            native_cif = _native_cif_path(native_dir, stem_tag)

            if not os.path.isfile(predicted_cif) or not os.path.isfile(native_cif):
                n_skipped += 1
                continue

            stdout = run_usalign(usalign_bin, predicted_cif, native_cif)
            parsed = parse_usalign_output(stdout)
            if parsed is None:
                n_skipped += 1
                continue

            tm_score, rmsd, aligned_length, seq_id, length = parsed
            f.write(f"{stem_tag}\t{tm_score:.5f}\t{rmsd:.3f}\t{aligned_length}\t{seq_id:.3f}\t{length}\n")
            n_ok += 1

    print(f"  {model_dir}: {n_ok} compared, {n_skipped} skipped (mismatch/missing) -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    p.add_argument(
        "--native_dir", default="models/exp2_finetuning/native",
        help="Directory produced by scripts/extract_native.py (contains an 'eval/structs' subfolder).",
    )
    p.add_argument(
        "--usalign_bin", default="USalign",
        help="Path to the USalign executable (e.g. after 'module load usalign').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    usalign_bin = shutil.which(args.usalign_bin) or args.usalign_bin
    if shutil.which(usalign_bin) is None and not os.path.isfile(usalign_bin):
        raise ValueError(
            f"USalign executable not found: {args.usalign_bin!r}. "
            f"Is it on PATH (e.g. 'module load usalign')? Use --usalign_bin to override."
        )

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

    print(f"[tmscore] Processing {len(model_dirs)} model/seed director(y/ies)")
    for model_dir in model_dirs:
        process_model_dir(model_dir, args.native_dir, usalign_bin)


if __name__ == "__main__":
    main()
