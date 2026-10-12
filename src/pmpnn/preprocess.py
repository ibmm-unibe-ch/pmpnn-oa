from __future__ import annotations

import random

import numpy as np
import torch

from .constants import AA_ALPHABET, CHAIN_ALPHABET


def featurize(batch, device):
    B      = len(batch)
    lengths = np.array([len(b["seq"]) for b in batch], dtype=np.int32)
    L_max  = int(lengths.max())

    X                = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    residue_idx      = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M          = np.zeros([B, L_max], dtype=np.float32)
    mask_self        = np.ones([B, L_max, L_max], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S                = np.zeros([B, L_max], dtype=np.int32)

    for i, b in enumerate(batch):
        masked_chains  = list(b["masked_list"])
        visible_chains = list(b["visible_list"])
        all_chains     = masked_chains + visible_chains

        # Promote visible chains that are identical to a masked chain
        visible_temp = {lv: b[f"seq_chain_{lv}"] for lv in visible_chains}
        masked_temp  = {lm: b[f"seq_chain_{lm}"] for lm in masked_chains}
        for km, vm in masked_temp.items():
            for kv, vv in visible_temp.items():
                if vm == vv and kv not in masked_chains:
                    masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)

        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)

        x_chain_list      = []
        chain_mask_list   = []
        chain_seq_list    = []
        chain_encoding_list = []
        c  = 1
        l0 = 0
        l1 = 0

        for letter in all_chains:
            chain_seq    = b[f"seq_chain_{letter}"]
            chain_length = len(chain_seq)
            coords       = b[f"coords_chain_{letter}"]
            x_chain = np.stack([
                coords[f"N_chain_{letter}"],
                coords[f"CA_chain_{letter}"],
                coords[f"C_chain_{letter}"],
                coords[f"O_chain_{letter}"],
            ], axis=1)   # [L, 4, 3]

            chain_mask = np.ones(chain_length) if letter in masked_chains else np.zeros(chain_length)
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c * np.ones(chain_length))

            l1 += chain_length
            mask_self[i, l0:l1, l0:l1] = 0.0
            residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
            l0 += chain_length
            c  += 1

        x            = np.concatenate(x_chain_list, axis=0)
        all_sequence = "".join(chain_seq_list)
        m            = np.concatenate(chain_mask_list, axis=0)
        chain_enc    = np.concatenate(chain_encoding_list, axis=0)

        l = len(all_sequence)
        X[i, :l]                  = x
        X[i, l:]                  = np.nan
        chain_M[i, :l]            = m
        chain_encoding_all[i, :l] = chain_enc

        indices = np.array([AA_ALPHABET.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask  = np.isfinite(X.sum((2, 3))).astype(np.float32)
    X[isnan] = 0.0

    return (
        torch.from_numpy(X).to(dtype=torch.float32, device=device),
        torch.from_numpy(S).to(dtype=torch.long, device=device),
        torch.from_numpy(mask).to(dtype=torch.float32, device=device),
        lengths,
        torch.from_numpy(chain_M).to(dtype=torch.float32, device=device),
        torch.from_numpy(residue_idx).to(dtype=torch.long, device=device),
        torch.from_numpy(mask_self).to(dtype=torch.float32, device=device),
        torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device),
    )
