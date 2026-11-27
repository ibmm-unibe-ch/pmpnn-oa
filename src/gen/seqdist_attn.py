"""
Sequence-distance-blind ablation of generation.

Registers a forward-pre-hook on ProteinFeatures.embeddings (the
PositionalEncodings submodule, see pmpnn/model.py) that overwrites its
``offset`` input — the true sequence distance (residue_idx[i] -
residue_idx[j]) between each residue and its K spatial neighbours — with
a single fixed constant for every edge, before PositionalEncodings does
its usual clamping/one-hot/linear embedding. The ``mask`` input (same-
chain indicator) is left untouched, so the only signal destroyed is
*how far apart in sequence* two residues are; everything else
PositionalEncodings would normally compute from that offset (including
the "different chain" sentinel bucket) still flows through unchanged
for whatever the fixed offset value resolves to.

Unlike gen.isotropic_attn / gen.uniform_attn / gen.permute_attn, this
hook has nothing to do with per-layer attention: PositionalEncodings is
called once per forward pass, inside ProteinFeatures, before the
encoder stack even starts, and feeds into every mode's edge features
(not just --mode oa). So there is no "--layers" option here, and the
ablation applies identically across --mode oa/gat/vanilla.

Comparing recovery / score / BLOSUM against unablated generation
(gen.__main__) probes whether the model is actually using the relative
sequence-position information PositionalEncodings computes, as opposed
to only the spatial (distance/orientation) information carried by the
other edge features.

Usage
-----
python -m gen.seqdist_attn <pdb_path> <out_folder> <checkpoint_path> [options]
    [--seqdist_offset 6]

Accepts exactly the same CLI options as gen.__main__ (see
GenerationContext), plus --seqdist_offset: the fixed sequence-distance
value (default 6) substituted for every edge's true offset.
"""
from __future__ import annotations

import torch

from ._ablation_common import parse_int_arg, run_generation_with_hook


# ---------------------------------------------------------------------------
# Sequence-distance-fixing ablation
# ---------------------------------------------------------------------------

def _fix_offset(offset_value: int):
    """Forward-pre-hook on PositionalEncodings: replace the real offset
    tensor with one filled with `offset_value`, leaving mask untouched."""
    def _hook(module, args):
        offset, mask = args
        return torch.full_like(offset, offset_value), mask
    return _hook


def register_seqdist_hook(model, offset_value: int):
    """
    Attach the fixed-offset ablation to the model's
    ProteinFeatures.embeddings (PositionalEncodings) submodule. Returns
    the hook handle (call .remove() to restore normal, true-offset
    behaviour).
    """
    return model.features.embeddings.register_forward_pre_hook(_fix_offset(offset_value))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _install(ctx, offset_value: int) -> None:
    register_seqdist_hook(ctx.model, offset_value)
    print(f"[seqdist_attn] Installed fixed-offset ({offset_value}) sequence-"
          f"distance hook on PositionalEncodings (mode={ctx.mode}).")


def main():
    offset_value = parse_int_arg("seqdist_offset", 6)
    run_generation_with_hook(lambda ctx: _install(ctx, offset_value))


if __name__ == "__main__":
    main()
