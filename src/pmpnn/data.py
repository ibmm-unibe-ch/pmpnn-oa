from __future__ import annotations

import csv
import random
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from dateutil import parser

from .constants import (
    AA_ALPHABET,
    CHAIN_ALPHABET,
    DATA_DATCUT,
    DATA_HOMO_THRESHOLD,
    PMPNN_MAX_CHAINS,
)

# ---------------------------------------------------------------------------
# Cluster / split building (unchanged logic, same CSV/cluster file format)
# ---------------------------------------------------------------------------

def build_training_clusters(
    index: str,
    val_clu_index: str,
    test_clu_index: str,
    rescut: float,
    debug: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    val_ids  = set(int(l) for l in open(val_clu_index))
    test_ids = set(int(l) for l in open(test_clu_index))
    datcut   = parser.parse(DATA_DATCUT)

    if debug:
        val_ids = set()
        test_ids = set()

    with open(index) as f:
        reader = csv.reader(f)
        next(reader)
        rows = [
            [r[0], r[3], int(r[4])]
            for r in reader
            if float(r[2]) <= rescut and parser.parse(r[1]) <= datcut
        ]

    if debug:
        rows = rows[:20]

    train, valid, test = {}, {}, {}
    for r in rows:
        cluster_id = r[2]
        entry = r[:2]
        if cluster_id in val_ids:
            valid.setdefault(cluster_id, []).append(entry)
        elif cluster_id in test_ids:
            test.setdefault(cluster_id, []).append(entry)
        else:
            train.setdefault(cluster_id, []).append(entry)

    if debug:
        valid = train
    return train, valid, test


# ---------------------------------------------------------------------------
# HDF5-backed loader — replaces loader_pdb + the .pt chain files
# ---------------------------------------------------------------------------

_h5_handles: Dict[Tuple[str, bool], h5py.File] = {}

def _get_h5(h5_file: str, in_memory: bool = False) -> h5py.File:
    key = (h5_file, in_memory)
    if key not in _h5_handles:
        if in_memory:
            _h5_handles[key] = h5py.File(h5_file, "r", driver="core", backing_store=False, libver="latest")
        else:
            _h5_handles[key] = h5py.File(h5_file, "r", swmr=True)
    return _h5_handles[key]


def _loader_h5(
    item: List[str],
    h5_file: str,
    homo_threshold: float = DATA_HOMO_THRESHOLD,
    in_memory: bool = False,
) -> dict:
    """
    Load one protein entry from the HDF5 file.

    Mirrors the logic of the original loader_pdb:
      1. Open PDBID group in the HDF5.
      2. Find assemblies that contain the requested chain (chid).
      3. Pick one assembly at random.
      4. Apply the stored rotation/translation transforms to xyz.
      5. Return {seq, xyz, idx, masked, label} in the same format.

    xyz is trimmed to the first 4 backbone atoms (N, Cα, C, O) so that
    featurize() receives the same [L, 4, 3] layout as before.
    """
    pdbid, chid = item[0].split("_")

    hf = _get_h5(h5_file, in_memory)
    if pdbid not in hf:
        return {"seq": np.zeros(5)}

    pg   = hf[pdbid]
    mg   = pg["meta"]
    cg   = pg["chains"]

    chains     = [c.decode() if isinstance(c, bytes) else c for c in mg["chains"][:]]
    asmb_ids   = [c.decode() if isinstance(c, bytes) else c for c in mg["asmb_ids"][:]]
    asmb_chains_list = [c.decode() if isinstance(c, bytes) else c for c in mg["asmb_chains"][:]]
    tm         = mg["tm"][:]   # [C, C, 3]
    chids      = np.array(chains)

    # Find assemblies containing chid (metadata only, no chain I/O yet)
    asmb_candidates = [
        a for a, ac in zip(asmb_ids, asmb_chains_list)
        if chid in ac.split(",")
    ]

    if not asmb_candidates:
        # No assembly — fall back to single chain
        if chid not in cg:
            return {"seq": np.zeros(5)}
        seq_val = cg[chid]["seq"][()]
        seq_str = seq_val.decode() if isinstance(seq_val, bytes) else seq_val
        xyz     = cg[chid]["xyz"][:, :4, :].astype(np.float32)
        L       = len(seq_str)
        return {
            "seq":    seq_str,
            "xyz":    torch.from_numpy(xyz),
            "idx":    torch.zeros(L, dtype=torch.int32),
            "masked": torch.tensor([0], dtype=torch.int32),
            "label":  item[0],
        }

    asmb_i = random.choice(asmb_candidates)
    idx    = [k for k, a in enumerate(asmb_ids) if a == asmb_i]

    # Determine which chains are actually needed for this assembly
    needed = {chid}
    for k in idx:
        needed |= set(asmb_chains_list[k].split(",")) & set(chains)

    # Load only the needed chains from HDF5
    chain_data: Dict[str, dict] = {}
    for c in needed:
        if c not in cg:
            continue
        seq_val = cg[c]["seq"][()]
        chain_data[c] = {
            "seq": seq_val.decode() if isinstance(seq_val, bytes) else seq_val,
            "xyz": cg[c]["xyz"][:, :4, :].astype(np.float32),  # [L, 4, 3]
        }

    # homo-chain detection
    if chid in chids:
        chid_pos = np.where(chids == chid)[0][0]
        seqid    = tm[chid_pos, :, 1]
    else:
        seqid = np.zeros(len(chids))
    homo = {ch_j for seqid_j, ch_j in zip(seqid, chids) if seqid_j > homo_threshold}

    # Apply assembly transforms
    asmb: Dict[tuple, np.ndarray] = {}
    for k in idx:
        key = f"asmb_xform{k}"
        if key not in mg:
            continue
        xform = mg[key][:]        # [n, 4, 4]
        u = xform[:, :3, :3]      # [n, 3, 3]
        r = xform[:, :3, 3]       # [n, 3]
        ac_chains = set(asmb_chains_list[k].split(",")) & set(chains)
        for c in ac_chains:
            if c not in chain_data:
                continue
            xyz   = torch.from_numpy(chain_data[c]["xyz"])   # [L, 4, 3]
            u_t   = torch.from_numpy(u).float()
            r_t   = torch.from_numpy(r).float()
            # [n, L, 4, 3]
            xyz_ru = torch.einsum("bij,raj->brai", u_t, xyz) + r_t[:, None, None, :]
            for i, xyz_i in enumerate(xyz_ru):
                asmb[(c, k, i)] = xyz_i.numpy()

    if not asmb:
        return {"seq": np.zeros(5)}

    seq, xyz_list, idx_list, masked = "", [], [], []
    for counter, ((c, k, _), xyz_i) in enumerate(asmb.items()):
        seq += chain_data[c]["seq"]
        xyz_list.append(torch.from_numpy(xyz_i))
        idx_list.append(torch.full((xyz_i.shape[0],), counter, dtype=torch.int32))
        if c in homo:
            masked.append(counter)

    return {
        "seq":    seq,
        "xyz":    torch.cat(xyz_list, dim=0),
        "idx":    torch.cat(idx_list, dim=0),
        "masked": torch.tensor(masked, dtype=torch.int32),
        "label":  item[0],
    }


# ---------------------------------------------------------------------------
# Dataset that maps cluster entries → loaded protein dicts
# ---------------------------------------------------------------------------

class ProteinH5Dataset(torch.utils.data.Dataset):
    def __init__(self, ids: List, cluster_dict: Dict, h5_file: str, max_length: int,
                 in_memory: bool = False):
        self.ids          = ids
        self.cluster_dict = cluster_dict
        self.h5_file      = h5_file
        self.max_length   = max_length
        self.in_memory    = in_memory

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        cluster_id = self.ids[index]
        entry = random.choice(self.cluster_dict[cluster_id])
        return _to_pdb_dict(_loader_h5(entry, self.h5_file, in_memory=self.in_memory), self.max_length)


def _collate_skip_none(batch: list) -> list:
    return [x for x in batch if x is not None]


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2**32) + worker_id)


# ---------------------------------------------------------------------------
# Streaming pipeline: _to_pdb_dict + StreamingStructureLoader
# ---------------------------------------------------------------------------

_ALPHABET_SET = set(AA_ALPHABET)


def _to_pdb_dict(t: dict, max_length: int):
    """Convert raw _loader_h5 output to a pdb_dict ready for featurize(). Returns None if filtered."""
    if "label" not in t:
        return None

    my_dict    = {}
    mask_list  = []
    visible_list = []
    concat_seq = ""

    unique_idx = list(np.unique(t["idx"]))
    if len(unique_idx) >= PMPNN_MAX_CHAINS:
        return None

    seq_arr = np.array(list(t["seq"]))

    for idx in unique_idx:
        letter = CHAIN_ALPHABET[idx]
        res    = np.argwhere(t["idx"] == idx)
        initial_sequence = "".join(seq_arr[res][0])

        # Trim His-tags at boundaries
        for trim_start, trim_end in [
            (0,   6),  (1,   7),  (2,   8),  (3,   9),  (4,  10),
            (-6, None), (-7, -1), (-8, -2), (-9, -3), (-10, -4),
        ]:
            sl = initial_sequence[trim_start:trim_end]
            if sl == "HHHHHH":
                if trim_start >= 0 and trim_end is None:
                    res = res[:, trim_start:]
                elif trim_end is None:
                    res = res[:, :-6]
                else:
                    end_col = trim_end if trim_end < 0 else None
                    res = res[:, :end_col] if trim_start == 0 else res[:, trim_start + 6:]

        if res.shape[1] < 4:
            continue

        chain_seq = "".join(seq_arr[res][0])
        my_dict[f"seq_chain_{letter}"] = chain_seq
        concat_seq += chain_seq

        if idx in t["masked"]:
            mask_list.append(letter)
        else:
            visible_list.append(letter)

        all_atoms = np.array(t["xyz"][res])[0]   # [L, 4, 3]
        my_dict[f"coords_chain_{letter}"] = {
            f"N_chain_{letter}":  all_atoms[:, 0, :].tolist(),
            f"CA_chain_{letter}": all_atoms[:, 1, :].tolist(),
            f"C_chain_{letter}":  all_atoms[:, 2, :].tolist(),
            f"O_chain_{letter}":  all_atoms[:, 3, :].tolist(),
        }

    if not concat_seq or len(concat_seq) > max_length:
        return None
    if set(concat_seq) - _ALPHABET_SET:
        return None

    my_dict["name"]          = t["label"]
    my_dict["masked_list"]   = mask_list
    my_dict["visible_list"]  = visible_list
    my_dict["num_of_chains"] = len(mask_list) + len(visible_list)
    my_dict["seq"]           = concat_seq
    return my_dict


# ---------------------------------------------------------------------------
# StreamingStructureLoader
# ---------------------------------------------------------------------------


class StreamingStructureLoader:
    """
    Wraps a DataLoader that yields lists of pdb_dicts (via _collate_skip_none).
    Accumulates items into a rolling buffer, sorts each buffer chunk by sequence
    length, and emits token-budget batches — same packing logic as the old
    StructureLoader, but without materialising the full epoch upfront.

    Each call to __iter__ starts a fresh pass over the DataLoader, so the
    training loop can simply do `for batch in loader_train:` every epoch.
    """

    def __init__(self, data_loader, batch_size: int, buffer_size: int = 2_000,
                 num_units: int = 1_000_000):
        self.data_loader = data_loader
        self.batch_size  = batch_size
        self.buffer_size = buffer_size
        self.num_units   = num_units

    def __iter__(self):
        buffer  = []
        n_seen  = 0
        for items in self.data_loader:          # items is a list[dict]
            buffer.extend(items)
            n_seen += len(items)
            if len(buffer) >= self.buffer_size:
                yield from self._emit(buffer)
                buffer = []
            if n_seen >= self.num_units:
                break
        if buffer:
            yield from self._emit(buffer)

    def _emit(self, buffer: list):
        lengths   = [len(item["seq"]) for item in buffer]
        sorted_ix = np.argsort(lengths)
        batch: list = []
        for ix in sorted_ix:
            size = lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(buffer[ix])
            else:
                if batch:
                    yield batch
                batch = [buffer[ix]]
        if batch:
            yield batch
