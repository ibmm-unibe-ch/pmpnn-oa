"""
model_size.py
=============
Analyse the parameter count of ProteinMPNN (OA / GAT / vanilla).

Usage
-----
    python model_size.py                  # OA model (default)
    python model_size.py --mode gat       # GAT model
    python model_size.py --mode vanilla   # vanilla model

Optional overrides (same names as ProteinMPNN.__init__):
    python model_size.py --hidden_dim 256 --num_encoder_layers 4
    python model_size.py --mode gat --hidden_dim 128 --l_max 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import from the protein_mpnn_oa package.  If it is not installed, we look
# for the source files one directory up from this script and create a shim
# package on-the-fly so the relative imports inside model.py still resolve.
# ---------------------------------------------------------------------------
try:
    from pmpnn.model import ProteinMPNN, OAAttention, GATAttention
    from pmpnn.constants import (
        PMPNN_HIDDEN_DIM, PMPNN_NUM_ENCODER_LAYERS, PMPNN_NUM_DECODER_LAYERS,
        PMPNN_K_NEIGHBORS, PMPNN_DROPOUT, PMPNN_AUGMENT_EPS,
        OA_NUM_RADIAL, OA_R_MAX, OA_P_CUTOFF, OA_L_MAX,
        OA_WEIGHT_MLP_HIDDEN_DIM, OA_GATE_HIDDEN_DIM,
        NUM_AA,
    )
except ModuleNotFoundError:
    # ── fallback: build a temporary package from the source files ──────────
    import importlib
    import types
    import shutil
    import tempfile

    SRC = Path(__file__).resolve().parent  # directory containing model.py etc.
    if not (SRC / "model.py").exists():
        SRC = SRC.parent                   # one level up as a second guess
    if not (SRC / "model.py").exists():
        raise FileNotFoundError(
            "Cannot locate model.py.  Place model_size.py next to model.py "
            "or install the protein_mpnn_oa package."
        )

    _pkg_dir = Path(tempfile.mkdtemp()) / "protein_mpnn_oa"
    _pkg_dir.mkdir()
    shutil.copy(SRC / "model.py",     _pkg_dir / "model.py")
    shutil.copy(SRC / "constants.py", _pkg_dir / "constants.py")
    (_pkg_dir / "__init__.py").write_text("")
    sys.path.insert(0, str(_pkg_dir.parent))

    from protein_mpnn_oa.model import ProteinMPNN, OAAttention, GATAttention  # noqa
    from protein_mpnn_oa.constants import (                                     # noqa
        PMPNN_HIDDEN_DIM, PMPNN_NUM_ENCODER_LAYERS, PMPNN_NUM_DECODER_LAYERS,
        PMPNN_K_NEIGHBORS, PMPNN_DROPOUT, PMPNN_AUGMENT_EPS,
        OA_NUM_RADIAL, OA_R_MAX, OA_P_CUTOFF, OA_L_MAX,
        OA_WEIGHT_MLP_HIDDEN_DIM, OA_GATE_HIDDEN_DIM,
        NUM_AA,
    )


# ── helpers ─────────────────────────────────────────────────────────────────

def n_params(module: nn.Module, only_trainable: bool = False) -> int:
    """Total number of (optionally trainable) scalar parameters."""
    pred = (lambda p: p.requires_grad) if only_trainable else (lambda p: True)
    return sum(p.numel() for p in module.parameters() if pred(p))


def _linear_info(lin: nn.Linear) -> str:
    bias = f" + bias({lin.out_features})" if lin.bias is not None else ""
    return (f"Linear({lin.in_features}→{lin.out_features}){bias}"
            f"  [{n_params(lin):,}]")


def _embedding_info(emb: nn.Embedding) -> str:
    return f"Embedding({emb.num_embeddings}×{emb.embedding_dim})  [{n_params(emb):,}]"


def _layernorm_info(ln: nn.LayerNorm) -> str:
    return f"LayerNorm({ln.normalized_shape[0]})  [{n_params(ln):,}]"


def separator(char: str = "─", width: int = 72) -> str:
    return char * width


def section(title: str, width: int = 72) -> str:
    pad = width - len(title) - 4
    return f"┌── {title} {'─' * pad}┐"


def human(n: int) -> str:
    """Pretty-print a parameter count as e.g. 1.23 M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


# ── OA / GAT layer detail ───────────────────────────────────────────────────

def print_oa_layer(attn: OAAttention) -> None:
    """Detailed breakdown of a single OAAttention module."""
    n_sh = attn.n_sh

    print()
    print(section("OAAttention — First Encoder Layer"))
    print(f"  num_radial : {attn.num_radial}")
    print(f"  n_sh       : {n_sh}  (= (l_max+1)² spherical-harmonic components)")
    print()

    # weight_mlp
    wm = attn.weight_mlp
    print("  weight_mlp  (edge features → radial × SH filter weights)")
    for i, layer in enumerate(wm):
        if isinstance(layer, nn.Linear):
            print(f"    [{i}] {_linear_info(layer)}")
        else:
            print(f"    [{i}] {layer.__class__.__name__}")
    wm_total = n_params(wm)
    print(f"    ── subtotal  {wm_total:>10,}  ({human(wm_total)})")
    print()

    # degree_gate
    dg = attn.degree_gate
    print("  degree_gate  (edge features → per-ℓ gating scalars)")
    for i, layer in enumerate(dg):
        if isinstance(layer, nn.Linear):
            print(f"    [{i}] {_linear_info(layer)}")
        else:
            print(f"    [{i}] {layer.__class__.__name__}")
    dg_total = n_params(dg)
    print(f"    ── subtotal  {dg_total:>10,}  ({human(dg_total)})")
    print()

    oa_total = n_params(attn)
    print(f"  {'─'*60}")
    print(f"  OAAttention total   {oa_total:>10,}  ({human(oa_total)})")
    print()


def print_gat_layer(attn: GATAttention) -> None:
    """Detailed breakdown of a single GATAttention module."""
    print()
    print(section("GATAttention — First Encoder Layer"))
    print()
    print(f"  W (transform) : {_linear_info(attn.W)}")
    print(f"  a (score)     : {_linear_info(attn.a)}")
    print()
    gat_total = n_params(attn)
    print(f"  GATAttention total  {gat_total:>10,}  ({human(gat_total)})")
    print()


# ── full model breakdown ─────────────────────────────────────────────────────

def print_model_breakdown(model: ProteinMPNN, mode: str) -> None:
    """Top-level parameter breakdown of the full ProteinMPNN model."""

    feat   = model.features
    enc    = model.encoder_layers
    dec    = model.decoder_layers
    W_e    = model.W_e
    W_s    = model.W_s
    W_out  = model.W_out

    print()
    print(separator("═"))
    print(f"  ProteinMPNN ({mode.upper()})  —  Full Parameter Breakdown")
    print(separator("═"))

    # ── ProteinFeatures ─────────────────────────────────────────────────────
    print()
    print("  ProteinFeatures")
    emb = feat.embeddings
    print(f"    embeddings.linear  : {_linear_info(emb.linear)}")
    print(f"    edge_embedding     : {_linear_info(feat.edge_embedding)}")
    print(f"    norm_edges         : {_layernorm_info(feat.norm_edges)}")
    print(f"    bessel             : {n_params(feat.bessel):,}  (non-trainable buffers only)")
    print(f"    cutoff             : {n_params(feat.cutoff):,}  (non-trainable buffers only)")
    feat_total = n_params(feat)
    print(f"    ── subtotal  {feat_total:>10,}  ({human(feat_total)})")

    # ── projection / output heads ────────────────────────────────────────────
    print()
    print("  Projection / output heads")
    print(f"    W_e   : {_linear_info(W_e)}")
    print(f"    W_s   : {_embedding_info(W_s)}")
    print(f"    W_out : {_linear_info(W_out)}")
    heads_total = n_params(W_e) + n_params(W_s) + n_params(W_out)
    print(f"    ── subtotal  {heads_total:>10,}  ({human(heads_total)})")

    # ── encoder layers ────────────────────────────────────────────────────────
    print()
    print("  Encoder layers  (× {})".format(len(enc)))
    for idx, layer in enumerate(enc):
        layer_total = n_params(layer)
        # split out the attention sub-module for clarity
        if hasattr(layer, "orbital_attn"):
            oa_p = n_params(layer.orbital_attn)
            base_p = layer_total - oa_p
            print(f"    enc[{idx}]  base {base_p:>9,}  + OA {oa_p:>7,}  = {layer_total:,}  ({human(layer_total)})")
        elif hasattr(layer, "gat_attn"):
            gat_p = n_params(layer.gat_attn)
            base_p = layer_total - gat_p
            print(f"    enc[{idx}]  base {base_p:>9,}  + GAT {gat_p:>6,}  = {layer_total:,}  ({human(layer_total)})")
        else:
            print(f"    enc[{idx}]  {layer_total:>9,}  ({human(layer_total)})")
    enc_total = n_params(enc)
    print(f"    ── subtotal  {enc_total:>10,}  ({human(enc_total)})")

    # ── encoder layer internal breakdown (one representative layer) ──────────
    print()
    _print_enclayer_internals(enc[0], mode)

    # ── decoder layers ────────────────────────────────────────────────────────
    print()
    print("  Decoder layers  (× {})".format(len(dec)))
    for idx, layer in enumerate(dec):
        layer_total = n_params(layer)
        print(f"    dec[{idx}]  {layer_total:>9,}  ({human(layer_total)})")
    dec_total = n_params(dec)
    print(f"    ── subtotal  {dec_total:>10,}  ({human(dec_total)})")

    # ── grand total ───────────────────────────────────────────────────────────
    grand = n_params(model)
    print()
    print(separator("═"))
    print(f"  Grand total  {grand:>10,}  ({human(grand)})")
    print()

    # component pie in text
    components = {
        "ProteinFeatures" : feat_total,
        "Heads (W_e/s/out)": heads_total,
        "Encoder layers"  : enc_total,
        "Decoder layers"  : dec_total,
    }
    print("  Component shares:")
    bar_width = 30
    for name, cnt in components.items():
        frac = cnt / grand
        bar  = "█" * round(frac * bar_width)
        print(f"    {name:<22} {cnt:>9,}  {frac:5.1%}  {bar}")
    print(separator("═"))


def _print_enclayer_internals(layer, mode: str) -> None:
    """Print the sub-layer breakdown of a single EncLayer."""
    print("  EncLayer internal breakdown (one layer)")
    rows = [
        ("norm1",  layer.norm1),
        ("norm2",  layer.norm2),
        ("norm3",  layer.norm3),
        ("W1",     layer.W1),
        ("W2",     layer.W2),
        ("W3",     layer.W3),
        ("W11",    layer.W11),
        ("W12",    layer.W12),
        ("W13",    layer.W13),
        ("dense",  layer.dense),
    ]
    if hasattr(layer, "orbital_attn"):
        rows.append(("orbital_attn", layer.orbital_attn))
    if hasattr(layer, "gat_attn"):
        rows.append(("gat_attn", layer.gat_attn))

    for name, mod in rows:
        p = n_params(mod)
        print(f"    {name:<16} {p:>9,}  ({human(p)})")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["oa", "gat", "vanilla"], default="oa",
                   help="Model variant to analyse (default: oa)")
    # Architecture overrides — same names as ProteinMPNN.__init__
    p.add_argument("--hidden_dim",          type=int,   default=PMPNN_HIDDEN_DIM)
    p.add_argument("--num_encoder_layers",  type=int,   default=PMPNN_NUM_ENCODER_LAYERS)
    p.add_argument("--num_decoder_layers",  type=int,   default=PMPNN_NUM_DECODER_LAYERS)
    p.add_argument("--k_neighbors",         type=int,   default=PMPNN_K_NEIGHBORS)
    p.add_argument("--dropout",             type=float, default=PMPNN_DROPOUT)
    p.add_argument("--num_radial_orb",      type=int,   default=OA_NUM_RADIAL)
    p.add_argument("--r_max_orb",           type=float, default=OA_R_MAX)
    p.add_argument("--p_cutoff",            type=int,   default=OA_P_CUTOFF)
    p.add_argument("--l_max",               type=int,   default=OA_L_MAX)
    p.add_argument("--weight_mlp_hidden_dim", type=int, default=OA_WEIGHT_MLP_HIDDEN_DIM)
    p.add_argument("--gate_hidden_dim",     type=int,   default=OA_GATE_HIDDEN_DIM)
    return p.parse_args()


def build_model(args: argparse.Namespace) -> ProteinMPNN:
    use_oa      = (args.mode == "oa")
    use_gat     = (args.mode == "gat")
    use_vanilla = (args.mode == "vanilla")
    return ProteinMPNN(
        hidden_dim            = args.hidden_dim,
        node_features         = args.hidden_dim,
        edge_features         = args.hidden_dim,
        num_encoder_layers    = args.num_encoder_layers,
        num_decoder_layers    = args.num_decoder_layers,
        k_neighbors           = args.k_neighbors,
        dropout               = args.dropout,
        num_radial_orb        = args.num_radial_orb,
        r_max_orb             = args.r_max_orb,
        p_cutoff              = args.p_cutoff,
        l_max                 = args.l_max,
        use_orbital_attn      = use_oa,
        weight_mlp_hidden_dim = args.weight_mlp_hidden_dim,
        gate_hidden_dim       = args.gate_hidden_dim,
        use_vanilla           = use_vanilla,
        use_gat               = use_gat,
    )


def main() -> None:
    args = parse_args()

    print()
    print(separator("═"))
    print(f"  ProteinMPNN Model Size Analysis  —  mode={args.mode.upper()}")
    print(separator("─"))
    print(f"  hidden_dim            : {args.hidden_dim}")
    print(f"  num_encoder_layers    : {args.num_encoder_layers}")
    print(f"  num_decoder_layers    : {args.num_decoder_layers}")
    print(f"  k_neighbors           : {args.k_neighbors}")
    if args.mode != "vanilla":
        print(f"  l_max                 : {args.l_max}  "
              f"→ n_sh = {(args.l_max + 1) ** 2}")
    if args.mode == "oa":
        print(f"  num_radial_orb        : {args.num_radial_orb}")
        print(f"  r_max_orb             : {args.r_max_orb} Å")
        print(f"  weight_mlp_hidden_dim : {args.weight_mlp_hidden_dim}")
        print(f"  gate_hidden_dim       : {args.gate_hidden_dim}")
    print(separator("═"))

    model = build_model(args)
    model.eval()

    # ── attention layer detail ────────────────────────────────────────────────
    first_enc = model.encoder_layers[0]
    if args.mode == "oa" and hasattr(first_enc, "orbital_attn"):
        print_oa_layer(first_enc.orbital_attn)
    elif args.mode == "gat" and hasattr(first_enc, "gat_attn"):
        print_gat_layer(first_enc.gat_attn)
    else:
        print("\n  (vanilla mode — no attention module; aggregation uses plain sum/scale)\n")

    # ── full model breakdown ──────────────────────────────────────────────────
    print_model_breakdown(model, args.mode)


if __name__ == "__main__":
    main()