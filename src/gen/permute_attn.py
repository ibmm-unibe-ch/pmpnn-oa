"""
Self-attention-preserving, neighbour-permuting ablation of generation.

Rewrites every node's attention output so that:
  - the attention weight a node assigns to itself is left exactly as the
    model computed it;
  - the weights it assigns to its other valid neighbours are randomly
    permuted among those same neighbours — same multiset of values (so
    the row still sums to 1, exactly as before), but reassigned to
    different neighbour identities.

Unlike gen.uniform_attn (which flattens the non-self weights to a single
uniform value, destroying both which neighbour gets attended to AND the
shape of the attention distribution), this ablation keeps the learned
*distribution* of attention mass intact and only scrambles *which*
neighbour each weight lands on. Comparing recovery / score / BLOSUM
against unablated generation (gen.__main__) probes whether the model
needs to know specifically which neighbour each weight belongs to, as
opposed to just needing a non-trivial (e.g. peaked) attention profile
over its neighbourhood.

Usage
-----
python -m gen.permute_attn <pdb_path> <out_folder> <checkpoint_path> [options]
    [--permute_attn_layers "0 2"]

Accepts exactly the same CLI options as gen.__main__ (see GenerationContext),
plus --permute_attn_layers: a space/comma-separated list of 0-indexed
encoder layers to ablate (default "all" — every encoder layer), so that
different layer combinations can be tested.

Only --mode oa and --mode gat have an attention module to ablate; --mode
vanilla runs unmodified (a warning is printed). The permutation is drawn
fresh (from torch's global RNG, seeded via --seed) on every forward pass,
independently per encoder layer.
"""
from __future__ import annotations

import torch

from ._ablation_common import register_attention_ablation_hooks, run_ablation


# ---------------------------------------------------------------------------
# Attention ablation
# ---------------------------------------------------------------------------

def _permute_non_self(
    alpha:       torch.Tensor,  # [B, N, K]
    mask_attend: torch.Tensor,  # [B, N, K]
    E_idx:       torch.Tensor,  # [B, N, K]
) -> torch.Tensor:
    """
    Randomly permute alpha's non-self, valid entries among themselves
    (per (batch, node) row), leaving the self-attention weight and every
    masked-out position untouched.

    Implemented with a double-argsort trick so the permutation is fully
    vectorized despite each row having a different number of valid
    non-self neighbours (no per-row Python loop):

      1. order/rank: a uniformly random bijection from each row's
         non-self-valid positions to {0, ..., count-1} (ranking iid
         random keys, with non-valid positions pushed last).
      2. order2: the deterministic inverse of the *canonical* (increasing
         position-index) enumeration of the same positions.
      3. pi = order2[rank] composes the two into a uniformly random
         permutation of the non-self-valid position set onto itself.
    """
    B, N, K = alpha.shape
    node_idx = torch.arange(N, device=alpha.device).view(1, N, 1)
    valid    = mask_attend > 0
    self_pos = (E_idx == node_idx) & valid
    non_self_valid = valid & ~self_pos

    k_idx = torch.arange(K, device=alpha.device).view(1, 1, K).expand(B, N, K)

    r      = torch.rand(alpha.shape, device=alpha.device)
    keyed  = torch.where(non_self_valid, r, r + 2.0)
    order  = torch.argsort(keyed, dim=-1)
    rank   = torch.argsort(order, dim=-1)

    key2   = torch.where(non_self_valid, k_idx, k_idx + K)
    order2 = torch.argsort(key2, dim=-1)

    pi       = torch.gather(order2, -1, rank)
    permuted = torch.gather(alpha, -1, pi)

    return torch.where(non_self_valid, permuted, alpha)


def register_permute_attention_hooks(model, layers: list[int] | None = None) -> list:
    """
    Attach the self-preserving/neighbour-permuting ablation to the given
    0-indexed encoder layers (default: every encoder layer). Returns the
    list of hook handles (call .remove() on each to restore normal
    attention).
    """
    return register_attention_ablation_hooks(model, _permute_non_self, layers=layers)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _install(ctx, layers) -> None:
    if ctx.mode == "vanilla":
        print("[permute_attn] WARNING — mode=vanilla has no attention module "
              "to ablate; generation will be unmodified.")
        return
    register_permute_attention_hooks(ctx.model, layers=layers)
    layers_desc = "all" if layers is None else sorted(layers)
    print(f"[permute_attn] Installed attention-ablation hooks on encoder "
          f"layer(s) {layers_desc} (of {ctx.num_encoder_layers}, "
          f"mode={ctx.mode}).")


def main():
    run_ablation("permute_attn_layers", _install)


if __name__ == "__main__":
    main()
