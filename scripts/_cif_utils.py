"""
_cif_utils.py
=============
Shared helpers for writing mmCIF structures, used by extract_native.py
(native structures, built from HDF5 coordinates), sanitize_boltz.py
(Boltz-predicted structures, built from an existing full-atom mmCIF), and
extract_edge_example.py (backbone + B-factor-coded attention examples).
"""
from __future__ import annotations

import gemmi

_ELEMENT_OF_ATOM = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C"}


def write_ca_only_cif(name: str, chains: list, out_path: str) -> None:
    """
    Write a Cα-only mmCIF structure.

    chains: list of (chain_letter, residues) tuples, where residues is a
    list of (seqid, three_letter_resname, (x, y, z)) tuples, already
    filtered to resolved (finite) Cα positions.
    """
    structure = gemmi.Structure()
    structure.name = name
    model = gemmi.Model("1")

    for letter, residues in chains:
        chain = gemmi.Chain(letter)
        for seqid, resname, xyz in residues:
            residue = gemmi.Residue()
            residue.name = resname
            residue.seqid = gemmi.SeqId(int(seqid), " ")
            residue.het_flag = "A"
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)

    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(out_path)


def write_backbone_cif(name: str, chains: list, out_path: str, comment: str | None = None) -> None:
    """
    Write a backbone (+ virtual Cβ) mmCIF structure with a per-residue
    B-factor, e.g. for coloring a local attention environment by score in
    ChimeraX / PyMOL ("color bfactor").

    chains: list of (chain_letter, residues) tuples, where residues is a
    list of (seqid, three_letter_resname, atoms, b_iso) tuples; atoms is a
    dict mapping atom name ("N", "CA", "C", "O", "CB") to an (x, y, z)
    tuple — all five are optional per residue, only the ones present are
    written.

    comment: optional free text written as a block of "# "-prefixed mmCIF
    comment lines above the data_ block (e.g. to record which residue is
    the attending one, since B-factor alone isn't self-explanatory when
    just skimming the file). Each line of `comment` becomes its own
    comment line; a plain top-level comment is the standard CIF way to
    attach free text that isn't tied to a specific category/loop.
    """
    structure = gemmi.Structure()
    structure.name = name
    model = gemmi.Model("1")

    for letter, residues in chains:
        chain = gemmi.Chain(letter)
        for seqid, resname, atoms, b_iso in residues:
            residue = gemmi.Residue()
            residue.name = resname
            residue.seqid = gemmi.SeqId(int(seqid), " ")
            residue.het_flag = "A"
            for atom_name, xyz in atoms.items():
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(_ELEMENT_OF_ATOM.get(atom_name, "C"))
                atom.pos = gemmi.Position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
                atom.b_iso = float(b_iso)
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)

    structure.add_model(model)
    structure.setup_entities()
    text = structure.make_mmcif_document().as_string()
    if comment:
        header = "".join(f"# {line}\n" for line in comment.splitlines())
        text = header + text
    with open(out_path, "w") as f:
        f.write(text)
