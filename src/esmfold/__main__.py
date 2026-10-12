"""
__main__.py
===========
Entry point and application logic for the ESMFold inference application.

Responsibilities
----------------
- Load the ESMFold model and tokeniser from HuggingFace.
- Iterate over one or more input FASTA files.
- For each sequence, prepare multi-chain inputs (glycine linker + custom
  position IDs), run inference, extract per-residue and global quality scores,
  and annotate the FASTA header.
- Optionally write predicted structures as PDB files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

from .constants import CHAIN_POSITION_GAP, ESMFOLD_MODEL_ID, LINKER
from .context import ESMFoldContext

torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return a list of (header, sequence) pairs parsed from *path*."""
    entries: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(seq_parts)))
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if header is not None:
        entries.append((header, "".join(seq_parts)))
    return entries


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_inputs(
    tokenizer,
    sequence: str,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    """Tokenise *sequence* and build a boolean residue mask.

    For multi-chain sequences (chains separated by ``/``), chains are joined
    with a glycine linker and custom ``position_ids`` are assigned so that each
    chain occupies a distinct, non-overlapping position range.  The residue
    mask is ``True`` only for real (non-linker) residues, so that quality
    scores are computed exclusively over the biological sequence.

    Parameters
    ----------
    tokenizer:
        ESMFold tokeniser loaded from HuggingFace.
    sequence:
        Amino-acid sequence, optionally with ``/`` separating chains.
    device:
        Target device for all tensors.

    Returns
    -------
    inputs:
        Dict of tensors ready for ``model(**inputs)``.
    residue_mask:
        Boolean tensor of shape ``(1, L)`` indicating real residues.
    """
    chains = sequence.split("/")

    if len(chains) == 1:
        inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        residue_mask = torch.ones(
            1, inputs["input_ids"].shape[1], dtype=torch.bool, device=device
        )
        return inputs, residue_mask

    joined = LINKER.join(chains)
    inputs = tokenizer([joined], return_tensors="pt", add_special_tokens=False)
    total_len = inputs["input_ids"].shape[1]

    position_ids = torch.zeros(1, total_len, dtype=torch.long)
    residue_mask = torch.zeros(1, total_len, dtype=torch.bool)

    token_idx = 0
    pos = 0
    for i, chain in enumerate(chains):
        if i > 0:
            # Advance over the linker tokens, keeping positions contiguous.
            for _ in range(len(LINKER)):
                position_ids[0, token_idx] = pos
                pos += 1
                token_idx += 1
            # Then skip a gap so the next chain starts far away in position space.
            pos += CHAIN_POSITION_GAP
        for _ in range(len(chain)):
            position_ids[0, token_idx] = pos
            residue_mask[0, token_idx] = True
            pos += 1
            token_idx += 1

    inputs["position_ids"] = position_ids
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, residue_mask.to(device)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    tokenizer,
    sequence: str,
    device: torch.device,
) -> tuple:
    """Run a single forward pass and return ``(output, residue_mask)``."""
    inputs, residue_mask = prepare_inputs(tokenizer, sequence, device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        output = model(**inputs)
    return output, residue_mask


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def extract_scores(
    output,
    residue_mask: torch.Tensor,
) -> tuple[float, float, float]:
    """Extract mean pLDDT, pTM, and mean PAE from model output.

    Scores are masked to real (non-linker) residues via *residue_mask*.

    Returns
    -------
    plddt : float
        Mean predicted local-distance difference test score (0–100).
    ptm : float
        Predicted TM-score (0–1).
    pae : float
        Mean predicted aligned error in Ångströms.
    """
    mask = residue_mask[0]
    plddt = output.plddt[0][mask].mean().item()
    ptm = output.ptm.reshape(-1)[0].item()
    pae_matrix = output.predicted_aligned_error[0]
    pae = pae_matrix[mask][:, mask].mean().item()
    return plddt, ptm, pae


# ---------------------------------------------------------------------------
# PDB conversion
# ---------------------------------------------------------------------------

def convert_to_pdb(output) -> str:
    """Convert ESMFold model output to a PDB-format string."""
    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    cpu_output = {
        k: v.to("cpu").numpy()
        for k, v in output.items()
        if hasattr(v, "numpy")
    }
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = cpu_output["atom37_atom_exists"]
    protein = OFProtein(
        aatype=cpu_output["aatype"][0],
        atom_positions=final_atom_positions[0],
        atom_mask=final_atom_mask[0],
        residue_index=cpu_output["residue_index"][0] + 1,
        b_factors=cpu_output["plddt"][0],
        chain_index=cpu_output["chain_index"][0] if "chain_index" in cpu_output else None,
    )
    return to_pdb(protein)


# ---------------------------------------------------------------------------
# Header annotation
# ---------------------------------------------------------------------------

def annotate_header(header: str, plddt: float, ptm: float, pae: float) -> str:
    """Append pLDDT / pTM / PAE scores to a FASTA header string."""
    return f"{header}, plddt={plddt:.4f}, ptm={ptm:.4f}, pae={pae:.4f}"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_fasta(
    fasta_path: Path,
    ctx: ESMFoldContext,
) -> None:
    """Run ESMFold on every sequence in *fasta_path* and write results.

    Annotated sequences are written to ``<ctx.out_dir>/<fasta_path.name>``.
    If ``ctx.save_pdb`` is ``True``, PDB files are written to
    ``<ctx.out_dir>/pdb/<stem>_seq<N>.pdb``.
    """
    entries = parse_fasta(fasta_path)
    annotated: list[tuple[str, str]] = []

    pdb_dir = ctx.out_dir / "pdb" if ctx.save_pdb else None
    if pdb_dir is not None:
        pdb_dir.mkdir(parents=True, exist_ok=True)

    for i, (header, sequence) in enumerate(entries):
        print(f"  [{i + 1}/{len(entries)}] {header[:80]}", flush=True)
        try:
            output, residue_mask = run_inference(
                ctx.model, ctx.tokenizer, sequence, ctx.device
            )
            plddt, ptm, pae = extract_scores(output, residue_mask)
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            annotated.append((header, sequence))
            continue

        new_header = annotate_header(header, plddt, ptm, pae)
        annotated.append((new_header, sequence))
        print(f"    plddt={plddt:.4f}  ptm={ptm:.4f}  pae={pae:.4f}", flush=True)

        if ctx.save_pdb:
            stem = f"{fasta_path.stem}_seq{i + 1}"
            pdb_text = convert_to_pdb(output)
            (pdb_dir / f"{stem}.pdb").write_text(pdb_text)

    out_fasta = ctx.out_dir / fasta_path.name
    with out_fasta.open("w") as f:
        for header, sequence in annotated:
            f.write(f">{header}\n{sequence}\n")

    print(f"  Written: {out_fasta}", flush=True)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(ctx: ESMFoldContext) -> None:
    """Load the ESMFold tokeniser and model into *ctx* in-place."""
    print("Loading ESMFold...", flush=True)
    ctx.tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_MODEL_ID)
    ctx.model = EsmForProteinFolding.from_pretrained(
        ESMFOLD_MODEL_ID, low_cpu_mem_usage=True
    )
    # Cast the language-model trunk to fp16 to reduce VRAM.
    ctx.model.esm = ctx.model.esm.half()
    if not ctx.cpu:
        ctx.model = ctx.model.to(ctx.device)
    ctx.model.trunk.set_chunk_size(ctx.chunk_size)
    ctx.model.eval()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ctx = ESMFoldContext.from_args()

    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    ctx.device = torch.device("cpu" if ctx.cpu else "cuda")

    load_model_and_tokenizer(ctx)

    for fasta_path in ctx.fasta_files:
        print(f"\nProcessing: {fasta_path}", flush=True)
        process_fasta(fasta_path, ctx)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
