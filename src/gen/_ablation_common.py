"""
Shared plumbing for the attention-ablation generation scripts
(gen.uniform_attn, gen.permute_attn): the generic per-encoder-layer hook
scaffolding (capturing E_idx and rewriting whatever the attention
submodule returns), CLI parsing for the --<flag>_layers option, and the
generation-loop driver — otherwise identical to gen.__main__.main(). Only
each script's own alpha-rewriting function differs between them.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from pmpnn.constants import AA_ALPHABET
from pmpnn.model import ProteinMPNN

from .__main__ import (
    _MODE_KWARGS,
    _OA_MODES,
    _load_jsonl,
    _protein_source,
    _run_conditional_probs,
    _run_generation,
    _run_score_only,
    _run_unconditional_probs,
    gen_featurize,
)
from .context import GenerationContext


# ---------------------------------------------------------------------------
# Generic per-layer attention-rewriting hook
# ---------------------------------------------------------------------------

def register_attention_ablation_hooks(model: ProteinMPNN, modify_fn, layers=None) -> list:
    """
    Attach ``modify_fn(alpha, mask_attend, E_idx) -> alpha`` as a forward
    hook on the given 0-indexed encoder layers' attention module
    (OAAttention or GATAttention), default every encoder layer. Returns
    the list of hook handles (call .remove() on each to restore normal
    attention).

    No change to pmpnn/model.py is needed: the hooks intercept the
    attention submodules' own forward() in place, so this applies
    identically whether the encoder is run via ProteinMPNN.sample /
    tied_sample (autoregressive generation) or ProteinMPNN.forward
    (teacher-forced scoring).
    """
    n_layers = len(model.encoder_layers)
    target_layers = set(range(n_layers)) if layers is None else set(layers)
    invalid = target_layers - set(range(n_layers))
    if invalid:
        raise ValueError(
            f"attn_layers {sorted(invalid)} out of range [0, {n_layers})")

    handles = []

    for idx, layer in enumerate(model.encoder_layers):
        if idx not in target_layers:
            continue

        captured_E_idx = {}

        def _capture_E_idx(module, args, kwargs, _store=captured_E_idx):
            # EncLayer.forward(h_V, h_E, E_idx, R_cut, Y, mask_V=None, mask_attend=None)
            _store["E_idx"] = args[2]

        handles.append(layer.register_forward_pre_hook(_capture_E_idx, with_kwargs=True))

        if getattr(layer, "use_orbital_attn", False):
            def _oa_hook(module, args, output, _store=captured_E_idx, _fn=modify_fn):
                # OAAttention.forward(R_cut, Y, edge_features, mask_attend) -> (alpha, spectral)
                _, _, _, mask_attend = args
                alpha, spectral = output
                return _fn(alpha, mask_attend, _store["E_idx"]), spectral

            handles.append(layer.orbital_attn.register_forward_hook(_oa_hook))

        if getattr(layer, "use_gat", False):
            def _gat_hook(module, args, output, _store=captured_E_idx, _fn=modify_fn):
                # GATAttention.forward(h_E, mask_attend) -> alpha
                _, mask_attend = args
                return _fn(output, mask_attend, _store["E_idx"])

            handles.append(layer.gat_attn.register_forward_hook(_gat_hook))

    return handles


# ---------------------------------------------------------------------------
# CLI: pull --<flag> out of argv before GenerationContext.from_args()
# ---------------------------------------------------------------------------

def parse_layers_arg(flag: str) -> list[int] | None:
    """
    Pull --<flag> out of sys.argv (leaving the rest untouched for
    GenerationContext.from_args()) and parse it into a list of 0-indexed
    layers, or None for "all".
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(f"--{flag}", type=str, default="all")
    ns, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    spec = getattr(ns, flag).strip()
    if spec.lower() == "all":
        return None
    return [int(tok) for tok in spec.replace(",", " ").split()]


def parse_int_arg(flag: str, default: int) -> int:
    """
    Pull --<flag> out of sys.argv (leaving the rest untouched for
    GenerationContext.from_args()) and parse it as an int, defaulting to
    `default` if not given.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(f"--{flag}", type=int, default=default)
    ns, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return getattr(ns, flag)


# ---------------------------------------------------------------------------
# Shared generation-loop driver (mirrors gen.__main__.main())
# ---------------------------------------------------------------------------

def run_generation_with_hook(install_fn) -> None:
    """
    Entry point shared by every ablation generation script: build the
    usual GenerationContext + model, call install_fn(ctx) to register
    whatever hooks (or print whatever "not applicable to this mode"
    warning) the caller wants, then run the normal gen.__main__
    generation loop unmodified. install_fn owns the full decision of
    whether/how its ablation applies to ctx.mode (different ablations
    target different submodules that don't all exist in every mode) and
    is responsible for its own log message.
    """
    ctx = GenerationContext.from_args()

    ctx.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(ctx.seed)

    mode_kwargs = _MODE_KWARGS[ctx.mode]
    oa_kwargs = dict(
        num_radial_orb        = ctx.num_radial_orb,
        r_max_orb              = ctx.r_max_orb,
        p_cutoff               = ctx.p_cutoff,
        l_max                  = ctx.l_max,
        weight_mlp_hidden_dim  = ctx.weight_mlp_hidden_dim,
        gate_hidden_dim        = ctx.gate_hidden_dim,
    ) if ctx.mode in _OA_MODES else {}
    ctx.model = ProteinMPNN.from_pretrained(
        ctx.checkpoint_path,
        map_location          = ctx.device,
        node_features         = ctx.hidden_dim,
        edge_features         = ctx.hidden_dim,
        hidden_dim            = ctx.hidden_dim,
        num_encoder_layers    = ctx.num_encoder_layers,
        num_decoder_layers    = ctx.num_decoder_layers,
        k_neighbors           = ctx.num_neighbors,
        dropout               = ctx.dropout,
        augment_eps           = 0.0,
        **oa_kwargs,
        **mode_kwargs,
    )
    ctx.model.to(ctx.device)
    ctx.model.eval()

    install_fn(ctx)

    os.makedirs(ctx.out_folder, exist_ok=True)

    chain_id_dict        = _load_jsonl(ctx.chain_id_jsonl)        if ctx.chain_id_jsonl        else None
    fixed_positions_dict = _load_jsonl(ctx.fixed_positions_jsonl) if ctx.fixed_positions_jsonl else None
    tied_positions_dict  = _load_jsonl(ctx.tied_positions_jsonl)  if ctx.tied_positions_jsonl  else None
    bias_AA_dict         = _load_jsonl(ctx.bias_AA_jsonl)         if ctx.bias_AA_jsonl         else None
    bias_by_res_dict     = _load_jsonl(ctx.bias_by_res_jsonl)     if ctx.bias_by_res_jsonl     else None
    omit_AA_dict         = _load_jsonl(ctx.omit_AA_jsonl)         if ctx.omit_AA_jsonl         else None
    pssm_dict            = _load_jsonl(ctx.pssm_jsonl)            if ctx.pssm_jsonl            else None

    omit_AAs_np = np.array([1.0 if aa in ctx.omit_AAs else 0.0 for aa in AA_ALPHABET], dtype=np.float32)
    bias_AAs_np = np.zeros(21, dtype=np.float32)
    if bias_AA_dict:
        for aa, bias in bias_AA_dict.items():
            if aa in AA_ALPHABET:
                bias_AAs_np[AA_ALPHABET.index(aa)] = float(bias)

    input_chain_list = ctx.pdb_path_chains.split() if ctx.pdb_path_chains else None
    temperatures     = [float(t) for t in ctx.sampling_temp.split()]

    with _protein_source(ctx, input_chain_list) as protein_batches:
        for proteins in protein_batches:
            name     = proteins[0]["name"]
            out_path = os.path.join(ctx.out_folder, f"{name}.fa")
            print(f"Processing {name} ...")

            try:
                (X, S, mask, lengths, chain_M, residue_idx, chain_encoding_all,
                chain_M_pos, omit_AA_mask, pssm_coef, pssm_bias, pssm_log_odds_all,
                bias_by_res_all, chain_list_list, visible_list_list, masked_list_list,
                masked_chain_length_list_list, tied_pos_list_of_lists_list,
                ) = gen_featurize(
                    proteins, ctx.device,
                    chain_id_dict        = chain_id_dict,
                    fixed_positions_dict = fixed_positions_dict,
                    omit_AA_dict         = omit_AA_dict,
                    tied_positions_dict  = tied_positions_dict,
                    pssm_dict            = pssm_dict,
                    bias_by_res_dict     = bias_by_res_dict,
                )

                if ctx.backbone_noise > 0:
                    X = X + ctx.backbone_noise * torch.randn_like(X)

                if ctx.score_only:
                    _run_score_only(
                        ctx, name, X, S, mask, chain_M, chain_M_pos, residue_idx,
                        chain_encoding_all, chain_list_list, masked_list_list, out_path,
                    )
                elif ctx.conditional_probs_only:
                    _run_conditional_probs(
                        ctx, name, X, S, mask, chain_M, chain_M_pos, residue_idx,
                        chain_encoding_all, out_path,
                    )
                elif ctx.unconditional_probs_only:
                    _run_unconditional_probs(
                        ctx, name, X, S, mask, chain_M, chain_M_pos, residue_idx,
                        chain_encoding_all, out_path,
                    )
                else:
                    _run_generation(
                        ctx, name, X, S, mask, chain_M, chain_M_pos, residue_idx,
                        chain_encoding_all, chain_list_list, masked_list_list,
                        tied_pos_list_of_lists_list,
                        omit_AAs_np, bias_AAs_np, omit_AA_mask,
                        pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all,
                        temperatures, ctx.num_seq_per_target, out_path,
                    )

                print(f"  → {out_path}")
            except Exception as e:
                print(f"    ERROR: {e}", file=sys.stderr)
                continue


def run_ablation(layers_flag: str, install_fn) -> None:
    """
    Entry point for per-layer attention-ablation generation scripts:
    parse --<layers_flag> (a space/comma-separated list of 0-indexed
    encoder layers, or "all"), then run_generation_with_hook with
    install_fn(ctx, layers).
    """
    layers = parse_layers_arg(layers_flag)
    run_generation_with_hook(lambda ctx: install_fn(ctx, layers))
