"""
Sequence generation with ProteinMPNN-OA.

Usage
-----
python generate.py <pdb_path> <out_folder> <checkpoint_path> [options]

Equivalent to protein_mpnn_run.py from the original ProteinMPNN repository,
adapted to use the ProteinMPNN implementation in pmpnn/model.py.
"""
from __future__ import annotations

import sys
import contextlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile

import numpy as np
import torch

from pmpnn.constants import AA_ALPHABET, AA_3TO1, BLOSUM62
from pmpnn.model import ProteinMPNN
from .context import GenerationContext

_MODE_KWARGS = {
    "vanilla": dict(use_vanilla=True,  use_orbital_attn=False, use_gat=False),
    "oa":      dict(use_vanilla=False, use_orbital_attn=True,  use_gat=False),
    "gat":     dict(use_vanilla=False, use_orbital_attn=False, use_gat=True),
}

_OA_MODES = {"oa"}


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def parse_PDB(pdb_file: str, input_chain_list: list | None = None) -> list:
    """
    Parse a single PDB file into a list containing one protein dict.

    Dict keys:
        name, seq, num_of_chains, masked_list, visible_list,
        seq_chain_X, coords_chain_X  (for each chain X)

    input_chain_list: chains to redesign (masked); None = redesign all.
    """
    name = os.path.splitext(os.path.basename(pdb_file))[0]

    chain_ids: list[str] = []
    res_order: dict[str, list] = {}
    res_atoms: dict[str, dict] = {}
    res_seq:   dict[str, dict] = {}

    with open(pdb_file) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C", "O"):
                continue
            res_name = line[17:20].strip()
            chain_id = line[21]
            res_id   = line[22:27].strip()
            x, y, z  = float(line[30:38]), float(line[38:46]), float(line[46:54])
            aa = AA_3TO1.get(res_name, "X")

            if chain_id not in chain_ids:
                chain_ids.append(chain_id)
                res_order[chain_id] = []
                res_atoms[chain_id] = {}
                res_seq[chain_id]   = {}

            if res_id not in res_atoms[chain_id]:
                res_atoms[chain_id][res_id] = {}
                res_order[chain_id].append(res_id)
                res_seq[chain_id][res_id] = aa

            res_atoms[chain_id][res_id][atom_name] = [x, y, z]

    my_dict: dict = {"name": name}

    for chain in chain_ids:
        N_list, CA_list, C_list, O_list, seq = [], [], [], [], ""
        for res_id in res_order[chain]:
            atoms = res_atoms[chain][res_id]
            if "CA" not in atoms:
                continue
            N_list.append(atoms.get("N",  [0., 0., 0.]))
            CA_list.append(atoms.get("CA", [0., 0., 0.]))
            C_list.append(atoms.get("C",  [0., 0., 0.]))
            O_list.append(atoms.get("O",  [0., 0., 0.]))
            seq += res_seq[chain][res_id]

        my_dict[f"seq_chain_{chain}"] = seq
        my_dict[f"coords_chain_{chain}"] = {
            f"N_chain_{chain}":  np.array(N_list,  dtype=np.float32),
            f"CA_chain_{chain}": np.array(CA_list, dtype=np.float32),
            f"C_chain_{chain}":  np.array(C_list,  dtype=np.float32),
            f"O_chain_{chain}":  np.array(O_list,  dtype=np.float32),
        }

    my_dict["num_of_chains"] = len(chain_ids)
    my_dict["seq"] = "".join(my_dict.get(f"seq_chain_{c}", "") for c in chain_ids)

    if input_chain_list is not None:
        masked_list  = [c for c in input_chain_list if c in chain_ids]
        visible_list = [c for c in chain_ids if c not in masked_list]
    else:
        masked_list  = list(chain_ids)
        visible_list = []

    my_dict["masked_list"]  = masked_list
    my_dict["visible_list"] = visible_list
    return [my_dict]


# ---------------------------------------------------------------------------
# Generation featurize
# ---------------------------------------------------------------------------

def gen_featurize(
    proteins: list,
    device,
    chain_id_dict:        dict | None = None,
    fixed_positions_dict: dict | None = None,
    omit_AA_dict:         dict | None = None,
    tied_positions_dict:  dict | None = None,
    pssm_dict:            dict | None = None,
    bias_by_res_dict:     dict | None = None,
):
    """
    Build all tensors needed for model.sample() / model.tied_sample().

    fixed_positions_dict:  {name: {chain: [1-indexed positions to fix]}}
    tied_positions_dict:   {name: [{chain: [1-indexed pos], ...}, ...]}
      Each dict in the list is a group of positions (across chains) that are
      constrained to share the same amino acid in tied_sample.
    pssm_dict:             {name: {chain: {pssm_coef, pssm_bias, pssm_log_odds}}}
    bias_by_res_dict:      {name: {chain: np.ndarray [L, 21]}}
    omit_AA_dict:          {name: {chain: {"1-indexed pos": ["AA", ...]}}}

    Returns
    -------
    X, S, mask, lengths, chain_M, residue_idx, chain_encoding_all,
    chain_M_pos, omit_AA_mask, pssm_coef, pssm_bias, pssm_log_odds_all,
    bias_by_res_all,
    chain_list_list, visible_list_list, masked_list_list,
    masked_chain_length_list_list, tied_pos_list_of_lists_list
    """
    B = len(proteins)
    lengths = np.array([len(p["seq"]) for p in proteins], dtype=np.int32)
    L_max   = int(lengths.max())

    X                 = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    residue_idx       = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M           = np.zeros([B, L_max], dtype=np.float32)
    chain_M_pos       = np.ones([B, L_max],  dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S                 = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask      = np.zeros([B, L_max, 21], dtype=np.float32)
    pssm_coef_all     = np.zeros([B, L_max], dtype=np.float32)
    pssm_bias_all     = np.zeros([B, L_max, 21], dtype=np.float32)
    pssm_log_odds_all = -1e8 * np.ones([B, L_max, 21], dtype=np.float32)
    bias_by_res_all   = np.zeros([B, L_max, 21], dtype=np.float32)

    chain_list_list                = []
    visible_list_list              = []
    masked_list_list               = []
    masked_chain_length_list_list  = []
    tied_pos_list_of_lists_list    = []

    for i, prot in enumerate(proteins):
        name = prot["name"]

        if chain_id_dict and name in chain_id_dict:
            masked_list, visible_list = chain_id_dict[name]
        else:
            masked_list  = list(prot["masked_list"])
            visible_list = list(prot["visible_list"])

        all_chains = masked_list + visible_list

        l0 = 0
        c  = 1
        masked_chain_length_list = []
        chain_start: dict[str, int] = {}

        for letter in all_chains:
            chain_seq    = prot[f"seq_chain_{letter}"]
            chain_length = len(chain_seq)
            coords       = prot[f"coords_chain_{letter}"]
            chain_start[letter] = l0

            x_chain = np.stack([
                coords[f"N_chain_{letter}"],
                coords[f"CA_chain_{letter}"],
                coords[f"C_chain_{letter}"],
                coords[f"O_chain_{letter}"],
            ], axis=1)
            X[i, l0:l0 + chain_length] = x_chain

            chain_M[i, l0:l0 + chain_length] = 1.0 if letter in masked_list else 0.0
            if letter in masked_list:
                masked_chain_length_list.append(chain_length)

            if fixed_positions_dict and name in fixed_positions_dict:
                for pos in fixed_positions_dict[name].get(letter, []):
                    idx = int(pos) - 1
                    if 0 <= idx < chain_length:
                        chain_M_pos[i, l0 + idx] = 0.0

            for j, aa in enumerate(chain_seq):
                S[i, l0 + j] = AA_ALPHABET.index(aa) if aa in AA_ALPHABET else AA_ALPHABET.index("X")

            if omit_AA_dict and name in omit_AA_dict and letter in omit_AA_dict[name]:
                for pos_str, aas in omit_AA_dict[name][letter].items():
                    idx = int(pos_str) - 1
                    if 0 <= idx < chain_length:
                        for aa in aas:
                            if aa in AA_ALPHABET:
                                omit_AA_mask[i, l0 + idx, AA_ALPHABET.index(aa)] = 1.0

            if pssm_dict and name in pssm_dict and letter in pssm_dict[name]:
                pd = pssm_dict[name][letter]
                pssm_coef_all[i, l0:l0 + chain_length]     = pd.get("pssm_coef", 0.0)
                pssm_bias_all[i, l0:l0 + chain_length]      = pd.get("pssm_bias", 0.0)
                pssm_log_odds_all[i, l0:l0 + chain_length] = pd.get("pssm_log_odds", -1e8)

            if bias_by_res_dict and name in bias_by_res_dict and letter in bias_by_res_dict[name]:
                bias_by_res_all[i, l0:l0 + chain_length] = bias_by_res_dict[name][letter]

            residue_idx[i, l0:l0 + chain_length]       = 100 * (c - 1) + np.arange(chain_length)
            chain_encoding_all[i, l0:l0 + chain_length] = c
            l0 += chain_length
            c  += 1

        chain_list_list.append(all_chains)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

        if tied_positions_dict and name in tied_positions_dict:
            tied_global: list[list[int]] = []
            for group in tied_positions_dict[name]:
                global_group: list[int] = []
                for letter, positions in group.items():
                    if letter in chain_start:
                        off = chain_start[letter]
                        global_group += [off + int(p) - 1 for p in positions]
                if global_group:
                    tied_global.append(global_group)
            tied_pos_list_of_lists_list.append(tied_global)
        else:
            tied_pos_list_of_lists_list.append([])

    isnan = np.isnan(X)
    mask  = np.isfinite(X.sum((2, 3))).astype(np.float32)
    X[isnan] = 0.0

    def _t(arr, dtype=torch.float32):
        return torch.from_numpy(arr).to(dtype=dtype, device=device)

    return (
        _t(X),
        _t(S, torch.long),
        _t(mask),
        lengths,
        _t(chain_M),
        _t(residue_idx, torch.long),
        _t(chain_encoding_all, torch.long),
        _t(chain_M_pos),
        _t(omit_AA_mask),
        _t(pssm_coef_all),
        _t(pssm_bias_all),
        _t(pssm_log_odds_all),
        _t(bias_by_res_all),
        chain_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        tied_pos_list_of_lists_list,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scores(S, log_probs, mask):
    loss = torch.nn.NLLLoss(reduction="none")(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1),
    ).view(S.size(0), S.size(1))
    return torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1).clamp(min=1)


def _seq_from_sample(S_sample_row, chain_encoding_row, mask_row, all_chains, masked_list):
    """Reconstruct per-chain sequences for masked chains, joined by '/'."""
    enc = chain_encoding_row.cpu().tolist()
    s   = S_sample_row.cpu().numpy()
    m   = mask_row.cpu().numpy()
    aa  = np.array(list(AA_ALPHABET))
    seqs = []
    for c_idx, letter in enumerate(all_chains):
        if letter not in masked_list:
            continue
        c_val = c_idx + 1
        positions = [j for j, (e, mv) in enumerate(zip(enc, m)) if e == c_val and mv > 0]
        seqs.append("".join(aa[s[positions]]))
    return "/".join(seqs)


def _blosum_score(S_true, S_sample, mask):
    """Mean BLOSUM62 score over masked positions, normalized by their count."""
    m       = mask.cpu().numpy().astype(bool)
    true_np = S_true.cpu().numpy()[m].astype(np.int32)
    pred_np = S_sample.cpu().numpy()[m].astype(np.int32)
    return float(BLOSUM62[true_np, pred_np].mean()) if m.any() else 0.0


def _load_jsonl(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            out.update(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Mode-specific run functions
# ---------------------------------------------------------------------------

def _run_score_only(ctx, name, X, S, mask, chain_M, chain_M_pos, residue_idx,
                    chain_encoding_all, chain_list_list, masked_list_list, out_path):
    mask_for_loss = mask * chain_M * chain_M_pos
    randn = torch.randn(chain_M.shape, device=X.device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=ctx.mixed_precision):
        log_probs = ctx.model(
            X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn,
            use_input_decoding_order=True,
            decoding_order=torch.argsort(randn),
        ).log_probs
    scores = _scores(S, log_probs, mask_for_loss)

    with open(out_path, "w") as f:
        for b_ix in range(X.shape[0]):
            seq_str = _seq_from_sample(
                S[b_ix], chain_encoding_all[b_ix], mask[b_ix],
                chain_list_list[b_ix], masked_list_list[b_ix],
            )
            f.write(f">{name}, score={scores[b_ix].item():.4f}\n{seq_str}\n")


def _run_conditional_probs(ctx, name, X, S, mask, chain_M, chain_M_pos, residue_idx,
                           chain_encoding_all, out_path):
    if ctx.conditional_probs_only_backbone_noise > 0:
        X = X + ctx.conditional_probs_only_backbone_noise * torch.randn_like(X)
    randn = torch.randn(chain_M.shape, device=X.device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=ctx.mixed_precision):
        log_probs = ctx.model(
            X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn,
        ).log_probs
    np.save(
        os.path.join(os.path.dirname(out_path), f"{name}_conditional_probs.npy"),
        torch.exp(log_probs).cpu().numpy(),
    )
    print(f"  saved conditional probs for {name}")


def _run_unconditional_probs(ctx, name, X, S, mask, chain_M, chain_M_pos, residue_idx,
                              chain_encoding_all, out_path):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=ctx.mixed_precision):
        log_probs = ctx.model(
            X, torch.zeros_like(S), mask, torch.zeros_like(chain_M),
            residue_idx, chain_encoding_all, torch.zeros_like(chain_M),
        ).log_probs
    np.save(
        os.path.join(os.path.dirname(out_path), f"{name}_unconditional_probs.npy"),
        torch.exp(log_probs).cpu().numpy(),
    )
    print(f"  saved unconditional probs for {name}")


def _run_generation(ctx, name, X, S, mask, chain_M, chain_M_pos, residue_idx,
                    chain_encoding_all, chain_list_list, masked_list_list,
                    tied_pos_list_of_lists_list, omit_AAs_np, bias_AAs_np,
                    omit_AA_mask, pssm_coef, pssm_bias, pssm_log_odds_all,
                    bias_by_res_all, temperatures, num_seq_per_target, out_path):

    pssm_log_odds_mask = (pssm_log_odds_all > ctx.pssm_threshold).float()
    use_omit_mask = omit_AA_mask.sum() > 0

    with open(out_path, "w") as f:
        for temp in temperatures:
            for ix in range(num_seq_per_target):
                randn = torch.randn(chain_M.shape, device=X.device)
                use_tied = len(tied_pos_list_of_lists_list[0]) > 0

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=ctx.mixed_precision):
                    if use_tied:
                        tied_pos  = tied_pos_list_of_lists_list[0]
                        tied_beta = torch.ones(X.size(1), device=X.device)
                        sample_out = ctx.model.tied_sample(
                            X, randn, S, chain_M, chain_encoding_all, residue_idx,
                            mask=mask, temperature=temp,
                            omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                            chain_M_pos=chain_M_pos,
                            omit_AA_mask=omit_AA_mask if use_omit_mask else None,
                            pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=ctx.pssm_multi,
                            pssm_log_odds_flag=ctx.pssm_log_odds_flag,
                            pssm_log_odds_mask=pssm_log_odds_mask,
                            pssm_bias_flag=ctx.pssm_bias_flag,
                            tied_pos=tied_pos, tied_beta=tied_beta,
                            bias_by_res=bias_by_res_all,
                        )
                    else:
                        sample_out = ctx.model.sample(
                            X, randn, S, chain_M, chain_encoding_all, residue_idx,
                            mask=mask, temperature=temp,
                            omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                            chain_M_pos=chain_M_pos,
                            omit_AA_mask=omit_AA_mask if use_omit_mask else None,
                            pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=ctx.pssm_multi,
                            pssm_log_odds_flag=ctx.pssm_log_odds_flag,
                            pssm_log_odds_mask=pssm_log_odds_mask,
                            pssm_bias_flag=ctx.pssm_bias_flag,
                            bias_by_res=bias_by_res_all,
                        )

                S_sample = sample_out["S"]

                mask_for_loss = mask * chain_M * chain_M_pos
                randn2 = torch.randn(chain_M.shape, device=X.device)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=ctx.mixed_precision):
                    log_probs = ctx.model(
                        X, S_sample, mask, chain_M * chain_M_pos,
                        residue_idx, chain_encoding_all, randn2,
                    ).log_probs
                score = _scores(S_sample, log_probs, mask_for_loss)[0].item()

                native_mask = mask_for_loss[0]
                recovery = (
                    ((S_sample[0] == S[0]).float() * native_mask).sum()
                    / native_mask.sum().clamp(min=1)
                ).item()

                blosum = _blosum_score(S[0], S_sample[0], native_mask)

                seq_str = _seq_from_sample(
                    S_sample[0], chain_encoding_all[0], mask[0],
                    chain_list_list[0], masked_list_list[0],
                )
                f.write(
                    f">{name}, score={score:.4f}, "
                    f"recovery={recovery:.4f}, blosum={blosum:.3f}, "
                    f"T={temp:.2f}, sample={ix+1}\n"
                    f"{seq_str}\n"
                )


# ---------------------------------------------------------------------------
# PDB file collection
# ---------------------------------------------------------------------------

_TAR_EXTS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


@contextlib.contextmanager
def _collect_pdb_files(pdb_path: str):
    """
    Yield a sorted list of PDB file paths from any of:
      - a directory of .pdb files
      - a .zip archive containing .pdb files
      - a .tar / .tar.gz / .tgz / .tar.bz2 / .tar.xz archive
      - a text file listing one PDB path per line (any other extension)
      - a single .pdb file

    Archives are extracted to a temporary directory that is removed on exit.
    """
    tmpdir = None
    try:
        if os.path.isdir(pdb_path):
            files = sorted(
                os.path.join(pdb_path, f)
                for f in os.listdir(pdb_path)
                if f.lower().endswith(".pdb")
            )
        elif pdb_path.lower().endswith(".zip"):
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(pdb_path) as zf:
                zf.extractall(tmpdir)
            files = sorted(
                os.path.join(root, f)
                for root, _, fs in os.walk(tmpdir)
                for f in fs
                if f.lower().endswith(".pdb")
            )
        elif any(pdb_path.lower().endswith(ext) for ext in _TAR_EXTS):
            tmpdir = tempfile.mkdtemp()
            with tarfile.open(pdb_path) as tf:
                tf.extractall(tmpdir)
            files = sorted(
                os.path.join(root, f)
                for root, _, fs in os.walk(tmpdir)
                for f in fs
                if f.lower().endswith(".pdb")
            )
        elif pdb_path.lower().endswith(".pdb"):
            files = [pdb_path]
        else:
            # Text file: one PDB path per line, # lines are comments
            with open(pdb_path) as fh:
                files = [
                    line.strip() for line in fh
                    if line.strip() and not line.startswith("#")
                ]
        yield files
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _read_pdb_ids(pdb_path: str) -> list:
    """Return a list of PDB IDs from a bare ID string or a text file of IDs."""
    if os.path.isfile(pdb_path):
        with open(pdb_path) as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return [pdb_path]


@contextlib.contextmanager
def _protein_source(ctx, input_chain_list):
    """
    Yield protein-dict batches ([dict]) from either PDB files or an HDF5 dataset.

    PDB mode  (ctx.h5_file is empty):
        pdb_path is resolved by _collect_pdb_files; each file is parsed with
        parse_PDB and filtered to max_length.

    HDF5 mode (ctx.h5_file is set):
        pdb_path is a single PDB ID (e.g. "1ABC_A") or a text file of such IDs.
        Each ID is loaded via _loader_h5 + _to_pdb_dict from pmpnn.data,
        which replicates the training-time assembly selection logic.
        input_chain_list / --pdb_path_chains is ignored in this mode; masked vs
        visible chains are determined by homo-chain detection in the HDF5 loader
        and can be overridden per-structure via --chain_id_jsonl.
    """
    if ctx.h5_file:
        from pmpnn.data import _loader_h5, _to_pdb_dict
        pdb_ids = _read_pdb_ids(ctx.pdb_path)

        def _gen():
            for pdb_id in pdb_ids:
                prot = _to_pdb_dict(_loader_h5([pdb_id], ctx.h5_file), ctx.max_length)
                if prot is None:
                    print(f"  skipping {pdb_id} (not found in HDF5 or filtered by max_length)")
                    continue
                yield [prot]

        yield _gen()
    else:
        with _collect_pdb_files(ctx.pdb_path) as pdb_files:
            def _gen():
                for pdb_file in pdb_files:
                    proteins = [
                        p for p in parse_PDB(pdb_file, input_chain_list)
                        if len(p["seq"]) <= ctx.max_length
                    ]
                    if not proteins:
                        print(f"  skipping {pdb_file} (exceeds max_length={ctx.max_length})")
                        continue
                    yield proteins

            yield _gen()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ctx = GenerationContext.from_args()

    ctx.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(ctx.seed)

    mode_kwargs = _MODE_KWARGS[ctx.mode]
    oa_kwargs = dict(
        num_radial_orb        = ctx.num_radial_orb,
        r_max_orb             = ctx.r_max_orb,
        p_cutoff              = ctx.p_cutoff,
        l_max                 = ctx.l_max,
        weight_mlp_hidden_dim = ctx.weight_mlp_hidden_dim,
        gate_hidden_dim       = ctx.gate_hidden_dim,
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

if __name__ == "__main__":
    main()