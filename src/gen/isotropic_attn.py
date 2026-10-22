"""
Isotropic (angular-blind) ablation of generation.

Registers a forward hook on every encoder layer's OAAttention.degree_gate
submodule that zeroes its output for every spherical-harmonic degree
l >= 1, keeping degree l=0 untouched. degree_gate already outputs one
value per degree (shape [..., l_max + 1], index l <-> degree l — see
OAAttention in pmpnn/model.py), so this needs no masking/self-position
bookkeeping like gen.uniform_attn / gen.permute_attn.

degree_gate's output gates OAAttention's radial*angular weight tensor W
per spherical-harmonic degree before it's contracted with the geometry
to form attention scores. Since the l=0 spherical harmonic is a constant
(direction-independent), zeroing every l >= 1 gate value collapses that
contraction down to its l=0 term only — so the resulting attention scores
become a function of the radial (Ca-Ca distance) profile between
neighbours alone, with no directional/angular dependence whatsoever.
Comparing recovery / score / BLOSUM against unablated generation
(gen.__main__) probes whether the model is actually using the angular
(l >= 1) geometric information OAAttention computes, as opposed to only
the isotropic, distance-only part of it.

Usage
-----
python -m gen.isotropic_attn <pdb_path> <out_folder> <checkpoint_path> [options]
    [--isotropic_attn_layers "0 2"]

Accepts exactly the same CLI options as gen.__main__ (see GenerationContext),
plus --isotropic_attn_layers: a space/comma-separated list of 0-indexed
encoder layers to ablate (default "all" — every encoder layer).

Only --mode oa has a degree_gate to ablate (OAAttention); --mode vanilla
and --mode gat have no such module and run unmodified (a warning is
printed).
"""
from __future__ import annotations

import torch

from ._ablation_common import run_ablation


# ---------------------------------------------------------------------------
# Degree-gate ablation
# ---------------------------------------------------------------------------

def _zero_high_degrees(module, args, output: torch.Tensor) -> torch.Tensor:
    """Forward hook on OAAttention.degree_gate: zero every degree l >= 1,
    leave l=0 (index 0) untouched."""
    output = output.clone()
    output[..., 1:] = 0.0
    return output


def register_isotropic_hooks(model, layers: list[int] | None = None) -> list:
    """
    Attach the l>=1 degree-zeroing ablation to the given 0-indexed
    encoder layers' OAAttention.degree_gate submodule (default: every
    encoder layer). Layers without an OAAttention — i.e. running in
    gat/vanilla mode — have nothing to attach to and are silently
    skipped. Returns the list of hook handles (call .remove() on each to
    restore normal, direction-dependent behaviour).
    """
    n_layers = len(model.encoder_layers)
    target_layers = set(range(n_layers)) if layers is None else set(layers)
    invalid = target_layers - set(range(n_layers))
    if invalid:
        raise ValueError(
            f"isotropic_attn_layers {sorted(invalid)} out of range [0, {n_layers})")

    handles = []
    for idx, layer in enumerate(model.encoder_layers):
        if idx not in target_layers:
            continue
        if not getattr(layer, "use_orbital_attn", False):
            continue
        handles.append(layer.orbital_attn.degree_gate.register_forward_hook(_zero_high_degrees))

    return handles


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _install(ctx, layers) -> None:
    if ctx.mode != "oa":
        print(f"[isotropic_attn] WARNING — mode={ctx.mode} has no degree_gate "
              f"to ablate (OAAttention only); generation will be unmodified.")
        return
    register_isotropic_hooks(ctx.model, layers=layers)
    layers_desc = "all" if layers is None else sorted(layers)
    print(f"[isotropic_attn] Installed l>=1 degree-zeroing hooks on encoder "
          f"layer(s) {layers_desc} (of {ctx.num_encoder_layers}, "
          f"mode={ctx.mode}).")


def main():
    run_ablation("isotropic_attn_layers", _install)


if __name__ == "__main__":
    main()
