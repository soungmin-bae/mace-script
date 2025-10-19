#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute forces from POSCAR using MACE and write phonopy-compatible vasprun.xml
"""

from ase.io import read
from mace.calculators import MACECalculator
import xml.etree.ElementTree as ET
import argparse


def get_mace_calculator(model_path="/Users/bama/Desktop/data/code/MACE/models/2023-12-03-mace-128-L1_epoch-199.model", device="cpu"):
    """Load MACE model."""
    return MACECalculator(model_path=model_path, device=device, default_dtype="float32")


def atoms_to_vasprun_xml(atoms, energy, forces, output_file="vasprun.xml"):
    """Write minimal phonopy-compatible vasprun.xml (with generator)."""
    root = ET.Element("modeling")

    # generator block
    gen = ET.SubElement(root, "generator")
    ET.SubElement(gen, "i", name="program", type="string").text = "vasp"
    ET.SubElement(gen, "i", name="version", type="string").text = "6.5.0  "

    # atominfo block
    atominfo = ET.SubElement(root, "atominfo")
    ET.SubElement(atominfo, "atoms").text = str(len(atoms))

    # empty structure
    ET.SubElement(root, "structure")

    # calculation block
    calc = ET.SubElement(root, "calculation")

    # forces
    varr = ET.SubElement(calc, "varray", name="forces")
    for f in forces:
        ET.SubElement(varr, "v").text = f"{f[0]: .10f}  {f[1]: .10f}  {f[2]: .10f}"

    # energy (optional but safe)
    energy_tag = ET.SubElement(calc, "energy")
    ET.SubElement(energy_tag, "i", name="e_fr_energy").text = f"{energy:.10f}"

    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)

    with open(output_file, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8")

    print(f"✅ Wrote {output_file} with <generator> tag (phonopy-compatible)")


def main():
    parser = argparse.ArgumentParser(description="Compute forces and write phonopy-compatible vasprun.xml")
    parser.add_argument("--input", "-i", type=str, default="POSCAR")
    parser.add_argument("--model", "-m", type=str, default="/Users/bama/Desktop/data/code/MACE/models/2023-12-03-mace-128-L1_epoch-199.model")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    atoms = read(args.input)
    calc = get_mace_calculator(args.model, device=args.device)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"Energy = {energy:.6f} eV")
    print("Forces (first atom):", forces[0])

    atoms_to_vasprun_xml(atoms, energy, forces)


if __name__ == "__main__":
    main()

