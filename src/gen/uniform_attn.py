"""
Self-attention-preserving, neighbour-uniformizing ablation of generation.

Rewrites every node's attention output so that:
  - the attention weight a node assigns to itself is left exactly as the
    model computed it;
  - the remaining probability mass (1 - self-weight) is spread uniformly
    over that node's other valid neighbours, discarding whatever edge- or
    geometry-driven pattern the model learned for them.

Comparing recovery / score / BLOSUM against unablated generation
(gen.__main__) probes whether the learned neighbour attention pattern is
actually load-bearing, or whether the model only needs to know which
neighbour is itself. See gen.permute_attn for a complementary ablation
that preserves the magnitude of each weight but randomizes which
neighbour it's assigned to.

Usage
-----
python -m gen.uniform_attn <pdb_path> <out_folder> <checkpoint_path> [options]
    [--uniform_attn_layers "0 2"]

Accepts exactly the same CLI options as gen.__main__ (see GenerationContext),
plus --uniform_attn_layers: a space/comma-separated list of 0-indexed encoder
layers to ablate (default "all" — every encoder layer), so that different
layer combinations can be tested.

Only --mode oa and --mode gat have an attention module to ablate; --mode
vanilla runs unmodified (a warning is printed).
"""
from __future__ import annotations

import torch

from ._ablation_common import register_attention_ablation_hooks, run_ablation


# ---------------------------------------------------------------------------
# Attention ablation
# ---------------------------------------------------------------------------

def _uniformize_non_self(
    alpha:       torch.Tensor,  # [B, N, K]
    mask_attend: torch.Tensor,  # [B, N, K]
    E_idx:       torch.Tensor,  # [B, N, K]
) -> torch.Tensor:
    B, N, K = alpha.shape
    node_idx = torch.arange(N, device=alpha.device).view(1, N, 1)
    valid    = mask_attend > 0
    self_pos = (E_idx == node_idx) & valid

    self_pos_f  = self_pos.to(alpha.dtype)
    self_weight = (alpha * self_pos_f).sum(-1, keepdim=True)

    non_self_f = (valid & ~self_pos).to(alpha.dtype)
    count      = non_self_f.sum(-1, keepdim=True)
    uniform    = (1.0 - self_weight) / count.clamp(min=1.0)

    return self_pos_f * alpha + non_self_f * uniform


def register_uniform_attention_hooks(model, layers: list[int] | None = None) -> list:
    """
    Attach the self-preserving/uniform-elsewhere ablation to the given
    0-indexed encoder layers (default: every encoder layer). Returns the
    list of hook handles (call .remove() on each to restore normal
    attention).
    """
    return register_attention_ablation_hooks(model, _uniformize_non_self, layers=layers)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _install(ctx, layers) -> None:
    if ctx.mode == "vanilla":
        print("[uniform_attn] WARNING — mode=vanilla has no attention module "
              "to ablate; generation will be unmodified.")
        return
    register_uniform_attention_hooks(ctx.model, layers=layers)
    layers_desc = "all" if layers is None else sorted(layers)
    print(f"[uniform_attn] Installed attention-ablation hooks on encoder "
          f"layer(s) {layers_desc} (of {ctx.num_encoder_layers}, "
          f"mode={ctx.mode}).")


def main():
    run_ablation("uniform_attn_layers", _install)


if __name__ == "__main__":
    main()
