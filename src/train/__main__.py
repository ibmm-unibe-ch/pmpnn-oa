import math
import os
import random
import time

import numpy as np
from tqdm import tqdm
import torch
import wandb

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
from pmpnn.loss import loss_nll, loss_smoothed
from pmpnn.model import ProteinMPNN
from pmpnn.optimizer import _make_optimizer
from pmpnn.preprocess import featurize

from .context import Context


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
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------

def _subset_smallest_clusters(train_clusters, fraction):
    """Keep the `fraction` of training clusters with the fewest member entries.

    Clusters are sorted ascending by (n_entries, cluster_id) so the selection is
    deterministic on ties, then the smallest ceil(fraction * N) are retained.
    """
    if fraction >= 1.0:
        return train_clusters
    n_total = len(train_clusters)
    n_keep  = max(1, math.ceil(fraction * n_total))
    ordered = sorted(train_clusters.items(), key=lambda kv: (len(kv[1]), kv[0]))
    kept    = dict(ordered[:n_keep])
    print(f"train_cluster_fraction={fraction}: keeping {n_keep}/{n_total} smallest "
          f"training clusters ({sum(len(v) for v in kept.values())}/"
          f"{sum(len(v) for v in train_clusters.values())} entries)")
    return kept


def load_dataloaders(ctx):
    train_clusters, valid_clusters, _ = build_training_clusters(
        ctx.index, ctx.val_clu_index, ctx.test_clu_index, ctx.rescut, ctx.debug
    )
    train_clusters = _subset_smallest_clusters(train_clusters, ctx.train_cluster_fraction)

    g = torch.Generator()
    g.manual_seed(ctx.seed)

    loader_kwargs = dict(
        batch_size  = DATALOADER_BATCH_SIZE,
        shuffle     = True,
        pin_memory  = False,
        num_workers = DATALOADER_NUM_WORKERS,
        collate_fn  = _collate_skip_none,
        generator   = g,
    )
    train_set = ProteinH5Dataset(list(train_clusters.keys()), train_clusters, ctx.h5_file, ctx.max_protein_length)
    valid_set = ProteinH5Dataset(list(valid_clusters.keys()), valid_clusters, ctx.h5_file, ctx.max_protein_length)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **loader_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **loader_kwargs)

    train_loader = StreamingStructureLoader(train_loader, ctx.mini_batch_size, num_units=ctx.num_examples_per_epoch)
    valid_loader = StreamingStructureLoader(valid_loader, ctx.mini_batch_size, num_units=ctx.num_examples_per_epoch)
    return train_loader, valid_loader


_MODE_KWARGS = {
    "vanilla": dict(use_vanilla=True,  use_orbital_attn=False, use_gat=False),
    "oa":      dict(use_vanilla=False, use_orbital_attn=True,  use_gat=False),
    "gat":     dict(use_vanilla=False, use_orbital_attn=False, use_gat=True),
}


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

    model_kwargs = dict(
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

    if ctx.from_pretrained:
        model = ProteinMPNN.from_pretrained(
            ctx.from_pretrained,
            map_location=ctx.device,
            **model_kwargs,
        )
    else:
        model = ProteinMPNN(**model_kwargs)

    model.to(ctx.device)
    return model


def apply_freeze(ctx):
    if ctx.freeze_mode == "freeze_encoder":
        ctx.model.freeze_encoder()
    elif ctx.freeze_mode == "freeze_decoder":
        ctx.model.freeze_decoder()
    elif ctx.freeze_mode == "freeze":
        ctx.model.freeze_encoder()
        ctx.model.freeze_decoder()


def load_optimizer(ctx, model):
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    return _make_optimizer(trainable, ctx.hidden_dim, ctx.total_step)


# ---------------------------------------------------------------------------
# Batch functions
# ---------------------------------------------------------------------------

def train_batch_fn(ctx, batch, num_accum):
    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, ctx.device)
    mask_for_loss = mask * chain_M
    randn = torch.randn_like(chain_M)

    if ctx.mixed_precision:
        with torch.cuda.amp.autocast():
            out = ctx.model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
            _, loss_av_smoothed = loss_smoothed(S, out.log_probs, mask_for_loss)
        ctx.scaler.scale(loss_av_smoothed / num_accum).backward()
    else:
        out = ctx.model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
        _, loss_av_smoothed = loss_smoothed(S, out.log_probs, mask_for_loss)
        (loss_av_smoothed / num_accum).backward()

    loss, _, true_false = loss_nll(S, out.log_probs, mask_for_loss)
    return (
        torch.sum(loss * mask_for_loss).item(),
        torch.sum(true_false * mask_for_loss).item(),
        torch.sum(mask_for_loss).item(),
    )


def val_batch_fn(ctx, batch, generator):
    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, ctx.device)
    mask_for_loss = mask * chain_M
    randn = torch.randn(chain_M.shape, generator=generator, dtype=chain_M.dtype, device=chain_M.device)
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
# Epoch functions
# ---------------------------------------------------------------------------

def train_epoch_fn(ctx):
    ctx.model.train()
    total_sum = total_acc = total_weights = 0.0
    num_accum = ctx.batch_size // ctx.mini_batch_size
    accum_count = 0
    ctx.optimizer.zero_grad()

    for batch in ctx.train_loader:
        s, a, w = train_batch_fn(ctx, batch, num_accum)
        total_sum     += s
        total_acc     += a
        total_weights += w
        accum_count   += 1

        if accum_count == num_accum:
            if ctx.mixed_precision:
                if ctx.gradient_norm > 0:
                    ctx.scaler.unscale_(ctx.optimizer.optimizer)
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.gradient_norm)
                ctx.scaler.step(ctx.optimizer)
                ctx.scaler.update()
            else:
                if ctx.gradient_norm > 0:
                    torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), ctx.gradient_norm)
                ctx.optimizer.step()
            ctx.optimizer.zero_grad()
            ctx.total_step += 1
            accum_count = 0

        yield total_sum / total_weights, total_acc / total_weights


def val_epoch_fn(ctx):
    ctx.model.eval()
    total_sum = total_acc = total_weights = 0.0
    all_true, all_pred = [], []
    # Re-seeded every epoch, independent of ctx.seed.
    generator = torch.Generator(device=ctx.device)
    generator.manual_seed(ctx.val_seed)
    with torch.no_grad():
        for batch in ctx.valid_loader:
            s, a, w, s_true, s_pred = val_batch_fn(ctx, batch, generator)
            total_sum     += s
            total_acc     += a
            total_weights += w
            all_true.append(s_true)
            all_pred.append(s_pred)
            yield total_sum / total_weights, total_acc / total_weights

    S_true_flat = np.concatenate(all_true)
    S_pred_flat = np.concatenate(all_pred)
    val_recover, val_false_pred = _per_type_metrics(S_true_flat, S_pred_flat)
    mean_blosum, per_type_blosum = _blosum_score(S_true_flat, S_pred_flat)
    ctx._last_val_extras = dict(
        blosum      = mean_blosum,
        recover     = val_recover,
        false_pred  = val_false_pred,
        blosum_type = per_type_blosum,
        S_true      = S_true_flat,
        S_pred      = S_pred_flat,
    )


def _consume_epoch(generator, desc, use_tqdm=True):
    bar = tqdm(generator, desc=desc, leave=False, disable=not use_tqdm)
    loss = acc = None
    for loss, acc in bar:
        bar.set_postfix(ppl=f"{np.exp(loss):.3f}", acc=f"{acc:.3f}")
    return loss, acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fn(ctx):
    best_val_loss     = 1e9
    epochs_no_improve = 0

    for epoch_offset in range(ctx.num_epochs):
        e  = ctx.start_epoch + epoch_offset
        t0 = time.time()

        train_loss, train_acc = _consume_epoch(train_epoch_fn(ctx), f"epoch {e+1} train", ctx.use_tqdm)
        val_loss,   val_acc   = _consume_epoch(val_epoch_fn(ctx),   f"epoch {e+1} val  ", ctx.use_tqdm)

        dt = time.time() - t0

        fmt = lambda x: np.format_float_positional(np.float32(x), unique=False, precision=3)
        line = (
            f"epoch: {e+1}, step: {ctx.total_step}, "
            f"time: {dt:.1f}, "
            f"train: {fmt(np.exp(train_loss))}, valid: {fmt(np.exp(val_loss))}, "
            f"train_acc: {fmt(train_acc)}, valid_acc: {fmt(val_acc)}"
        )
        print(line)
        with open(ctx.logfile, "a") as f:
            f.write(line + "\n")

        ckpt_data = {
            "epoch":                e + 1,
            "step":                 ctx.total_step,
            "num_edges":            ctx.num_neighbors,
            "noise_level":          ctx.backbone_noise,
            "model_state_dict":     ctx.model.state_dict(),
            "optimizer_state_dict": ctx.optimizer.optimizer.state_dict(),
        }
        torch.save(ckpt_data, ctx.base_folder + "model_weights/epoch_last.pt")

        if (e + 1) % ctx.save_model_every_n_epochs == 0:
            torch.save(ckpt_data,
                       ctx.base_folder + f"model_weights/epoch{e+1}_step{ctx.total_step}.pt")

        extras = ctx._last_val_extras
        log_dict = {
            "epoch":            e + 1,
            "time":             dt,
            "train/loss":       train_loss,
            "train/perplexity": np.exp(train_loss),
            "train/accuracy":   train_acc,
            "val/loss":         val_loss,
            "val/perplexity":   np.exp(val_loss),
            "val/accuracy":     val_acc,
            "val/blosum":       extras["blosum"],
        }
        for t, aa in enumerate(AA_ALPHABET):
            log_dict[f"val/type/{aa}/recover"]         = extras["recover"][t]
            log_dict[f"val/type/{aa}/false_pred_rate"] = extras["false_pred"][t]
            log_dict[f"val/type/{aa}/blosum"]          = extras["blosum_type"][t]
        log_dict["val/confusion_matrix"] = wandb.plot.confusion_matrix(
            y_true      = extras["S_true"].tolist(),
            preds       = extras["S_pred"].tolist(),
            class_names = list(AA_ALPHABET),
        )
        if (val_loss - best_val_loss) / best_val_loss < -ctx.min_delta:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(
                {k: v for k, v in ckpt_data.items() if k != "optimizer_state_dict"},
                ctx.base_folder + "model_weights/best.pt",
            )
        else:
            epochs_no_improve += 1

        log_dict["early_stopping/epochs_no_improve"] = epochs_no_improve
        log_dict["early_stopping/best_val_loss"]     = best_val_loss
        wandb.log(log_dict, step=e + 1)

        if epochs_no_improve >= ctx.patience:
            print(f"Early stopping at epoch {e+1} (no improvement for {ctx.patience} epochs).")
            return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ctx = Context.from_args()

    if ctx.debug:
        ctx.num_examples_per_epoch = 50
        ctx.max_protein_length     = 1000
        ctx.batch_size             = 1000
        ctx.mini_batch_size        = 1000

    ctx.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(ctx.seed)
    ctx.scaler = torch.cuda.amp.GradScaler()

    ctx.base_folder = time.strftime(ctx.out_dir, time.localtime())
    if not ctx.base_folder.endswith("/"):
        ctx.base_folder += "/"
    ctx.model_name = os.path.basename(ctx.base_folder.rstrip("/"))
    os.makedirs(ctx.base_folder + "model_weights", exist_ok=True)

    ctx.logfile = ctx.base_folder + "log.txt"
    with open(ctx.logfile, "w") as f:
        f.write("Epoch\tTrain\tValidation\n")

    ctx.train_loader, ctx.valid_loader = load_dataloaders(ctx)
    ctx.model     = load_model(ctx)
    apply_freeze(ctx)
    ctx.optimizer = load_optimizer(ctx, ctx.model)

    wandb.init(
        project = ctx.wandb_project,
        name    = ctx.model_name,
        config  = {k: v for k, v in vars(ctx).items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool))},
        dir     = ctx.base_folder,
    )

    train_fn(ctx)
    wandb.finish()


if __name__ == "__main__":
    main()
