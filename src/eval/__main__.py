"""
Teacher-forced evaluation of ProteinMPNN-OA on the held-out test split.

Usage
-----
python -m pmpnn_eval \\
    <h5_file> <index> <val_clu_index> <test_clu_index> <from_pretrained> \\
    [options]

Runs a single teacher-forced pass over the test set and reports per-token NLL
loss, perplexity, sequence recovery, and per-residue-type BLOSUM62 scores.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from tqdm import tqdm

from pmpnn.constants import (
    AA_ALPHABET,
    BLOSUM62,
    DATALOADER_BATCH_SIZE,
    DATALOADER_NUM_WORKERS,
)
from pmpnn.data import (
    ProteinH5Dataset,
    StreamingStructureLoader,
    _collate_skip_none,
    build_training_clusters,
    worker_init_fn,
)
from pmpnn.loss import loss_nll
from pmpnn.model import ProteinMPNN
from pmpnn.preprocess import featurize

from .context import EvalContext


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Residue-type metrics helpers
# ---------------------------------------------------------------------------

def _per_type_metrics(S_true_flat, S_pred_flat):
    recover, false_pred = {}, {}
    for t in range(len(AA_ALPHABET)):
        sel = S_true_flat == t
        if sel.sum() == 0:
            recover[t] = false_pred[t] = float("nan")
        else:
            rec = (S_pred_flat[sel] == t).mean()
            recover[t]    = float(rec)
            false_pred[t] = float(1.0 - rec)
    return recover, false_pred


def _blosum_score(S_true_flat, S_pred_flat):
    scores     = BLOSUM62[S_true_flat, S_pred_flat].astype(np.float32)
    mean_score = float(scores.mean())
    per_type   = {
        t: float(scores[S_true_flat == t].mean()) if (S_true_flat == t).sum() > 0 else float("nan")
        for t in range(len(AA_ALPHABET))
    }
    return mean_score, per_type


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_MODE_KWARGS = {
    "vanilla": dict(use_vanilla=True,  use_orbital_attn=False, use_gat=False),
    "oa":      dict(use_vanilla=False, use_orbital_attn=True,  use_gat=False),
    "gat":     dict(use_vanilla=False, use_orbital_attn=False, use_gat=True),
}


def load_test_loader(ctx):
    _, _, test_clusters = build_training_clusters(
        ctx.index, ctx.val_clu_index, ctx.test_clu_index, ctx.rescut, ctx.debug
    )

    g = torch.Generator()
    g.manual_seed(ctx.seed)

    test_set = ProteinH5Dataset(
        list(test_clusters.keys()), test_clusters, ctx.h5_file, ctx.max_protein_length
    )
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size     = DATALOADER_BATCH_SIZE,
        shuffle        = False,
        pin_memory     = False,
        num_workers    = DATALOADER_NUM_WORKERS,
        collate_fn     = _collate_skip_none,
        generator      = g,
        worker_init_fn = worker_init_fn,
    )
    return StreamingStructureLoader(loader, ctx.mini_batch_size, num_units=ctx.num_examples_per_epoch)


def load_model(ctx):
    mode_kwargs = _MODE_KWARGS[ctx.mode]
    oa_kwargs = dict(
        num_radial_orb        = ctx.num_radial_orb,
        r_max_orb             = ctx.r_max_orb,
        p_cutoff              = ctx.p_cutoff,
        l_max                 = ctx.l_max,
        weight_mlp_hidden_dim = ctx.weight_mlp_hidden_dim,
        gate_hidden_dim       = ctx.gate_hidden_dim,
    ) if ctx.mode == "oa" else {}

    model = ProteinMPNN.from_pretrained(
        ctx.from_pretrained,
        map_location       = ctx.device,
        node_features      = ctx.hidden_dim,
        edge_features      = ctx.hidden_dim,
        hidden_dim         = ctx.hidden_dim,
        num_encoder_layers = ctx.num_encoder_layers,
        num_decoder_layers = ctx.num_decoder_layers,
        k_neighbors        = ctx.num_neighbors,
        dropout            = ctx.dropout,
        augment_eps        = ctx.backbone_noise,
        **oa_kwargs,
        **mode_kwargs,
    )
    model.to(ctx.device)
    return model


# ---------------------------------------------------------------------------
# Batch function
# ---------------------------------------------------------------------------

def eval_batch_fn(ctx, batch):
    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, ctx.device)
    mask_for_loss = mask * chain_M
    randn = torch.randn_like(chain_M)
    out = ctx.model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
    loss, _, true_false = loss_nll(S, out.log_probs, mask_for_loss)
    S_pred    = torch.argmax(out.log_probs, -1)
    flat_mask = mask_for_loss == 1
    return (
        torch.sum(loss * mask_for_loss).item(),
        torch.sum(true_false * mask_for_loss).item(),
        torch.sum(mask_for_loss).item(),
        S[flat_mask].cpu().numpy().astype(np.int32),
        S_pred[flat_mask].cpu().numpy().astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def test_epoch_fn(ctx):
    ctx.model.eval()
    total_sum = total_acc = total_weights = 0.0
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in tqdm(ctx.test_loader, desc="test", disable=not ctx.use_tqdm):
            s, a, w, s_true, s_pred = eval_batch_fn(ctx, batch)
            total_sum     += s
            total_acc     += a
            total_weights += w
            all_true.append(s_true)
            all_pred.append(s_pred)

    loss = total_sum / total_weights
    acc  = total_acc / total_weights

    S_true_flat = np.concatenate(all_true)
    S_pred_flat = np.concatenate(all_pred)
    recover, false_pred      = _per_type_metrics(S_true_flat, S_pred_flat)
    mean_blosum, blosum_type = _blosum_score(S_true_flat, S_pred_flat)

    return dict(
        loss        = loss,
        perplexity  = float(np.exp(loss)),
        accuracy    = acc,
        blosum      = mean_blosum,
        recover     = recover,
        false_pred  = false_pred,
        blosum_type = blosum_type,
    )


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def print_results(results):
    fmt = lambda x: np.format_float_positional(np.float32(x), unique=False, precision=4)
    print(f"loss:       {fmt(results['loss'])}")
    print(f"perplexity: {fmt(results['perplexity'])}")
    print(f"accuracy:   {fmt(results['accuracy'])}")
    print(f"blosum62:   {fmt(results['blosum'])}")
    print()
    print(f"{'AA':<4} {'recovery':>10} {'false_pred':>12} {'blosum':>8}")
    print("-" * 38)
    for t, aa in enumerate(AA_ALPHABET):
        rec  = results['recover'][t]
        fp   = results['false_pred'][t]
        bl   = results['blosum_type'][t]
        r_s  = fmt(rec) if not np.isnan(rec) else "   n/a"
        fp_s = fmt(fp)  if not np.isnan(fp)  else "   n/a"
        bl_s = fmt(bl)  if not np.isnan(bl)  else "   n/a"
        print(f"{aa:<4} {r_s:>10} {fp_s:>12} {bl_s:>8}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ctx = EvalContext.from_args()

    if ctx.debug:
        ctx.num_examples_per_epoch = 50
        ctx.max_protein_length     = 1000
        ctx.mini_batch_size        = 1000

    ctx.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(ctx.seed)

    ctx.test_loader = load_test_loader(ctx)
    ctx.model       = load_model(ctx)

    results = test_epoch_fn(ctx)
    print_results(results)


if __name__ == "__main__":
    main()
