"""
_boltz_utils.py
================
Shared helpers for locating Boltz prediction output, used by
sanitize_boltz.py and tmscore.py.
"""
from __future__ import annotations

import os


def predictions_dir(model_dir: str) -> str:
    return os.path.join(
        model_dir, "eval", "structs_boltz", "boltz_results_seqs_boltz", "predictions"
    )


def raw_predicted_cif_path(model_dir: str, stem_tag: str) -> str:
    return os.path.join(predictions_dir(model_dir), stem_tag, f"{stem_tag}_model_0.cif")


def discover_stem_tags(model_dir: str) -> list:
    directory = predictions_dir(model_dir)
    if not os.path.isdir(directory):
        return []
    return sorted(
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    )
