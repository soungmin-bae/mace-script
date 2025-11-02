#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Relax atomic structures using a local MACE model (ASE + FIRE optimizer).

Features:
  - Fully compatible with ASE >=3.20, Python >=3.7
  - Supports VASP-like ISIF modes (0–7)
  - Fixed-axis relaxation via --fix-axis
  - Batch mode: process multiple POSCAR-* files automatically
  - Outputs: CONTCAR-*, OUTCAR-*, vasprun-*.xml, relax-*_log.txt, relax-*_log.pdf
  - Customizable output names via --contcar, --outcar, and --vasprun
  - Logs total energy, max force, and stress per step
  - Generates fully phonopy-compatible vasprun.xml
"""
from __future__ import annotations
import argparse, sys, os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.optimize import FIRE
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
from ase.filters import UnitCellFilter, StrainFilter, ExpCellFilter
import xml.etree.ElementTree as ET

# === PyDefect Imports ===
from pymatgen.io.ase import AseAtomsAdaptor
from monty.json import MontyEncoder
import json


# ============================================================
# 1) MACE calculator
# ============================================================
def get_mace_calculator(device="cpu"):
    from mace.calculators import MACECalculator
    return MACECalculator(
        model_path="/Users/bama/package/MACE/models/mace-omat-0-small-fp32.model",
        device=device,
        default_dtype="float32",
    )


# ============================================================
# 2) Logger (with context manager)
# ============================================================
class Logger:
    def __init__(self, logfile="relax_log.txt"):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.flush()
        finally:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ============================================================
# 3) ISIF modes and axis fix
# ============================================================
def build_axis_mask(fix_axis: list[str]):
    mask = np.ones((3, 3), dtype=bool)
    for ax in fix_axis:
        if ax.lower() == "a": mask[0, :] = False
        elif ax.lower() == "b": mask[1, :] = False
        elif ax.lower() == "c": mask[2, :] = False
    return mask


def get_relax_target(atoms, isif: int, fix_axis: list[str]):
    if isif == 0:
        print(" ISIF=0 → Single-point calculation.")
        return atoms
    elif isif == 1:
        print(" ISIF=1 → Stress evaluation only.")
        atoms.set_constraint(FixAtoms(range(len(atoms))))
        return atoms
    elif isif == 2:
        print(" ISIF=2 → Relax atomic positions only.")
        return atoms
    elif isif == 3:
        print("️ ISIF=3 → Relax atoms + full cell (volume & shape).")
        mask = build_axis_mask(fix_axis)
        print(f" Fixed axes: {', '.join(fix_axis).upper() or '(none)'}")
        return UnitCellFilter(atoms, mask=mask)
    elif isif == 4:
        print("️ ISIF=4 → Relax cell only.")
        atoms.set_constraint(FixAtoms(range(len(atoms))))
        return UnitCellFilter(atoms, mask=build_axis_mask(fix_axis))
    elif isif == 5:
        print("️ ISIF=5 → Relax atoms + shape (volume fixed).")
        return ExpCellFilter(atoms, mask=build_axis_mask(fix_axis))
    elif isif == 6:
        print("️ ISIF=6 → Relax atoms + volume (shape fixed).")
        return StrainFilter(atoms, mask=build_axis_mask(fix_axis))
    elif isif == 7:
        print("️ ISIF=7 → Relax atoms + anisotropic shape.")
        return ExpCellFilter(atoms, mask=build_axis_mask(fix_axis))
    else:
        raise ValueError("Unsupported ISIF. Choose 0–7.")


# ============================================================
# 4) Output writers
# ============================================================
def write_outcar(atoms, energy, outcar_name="OUTCAR"):
    """Write a pymatgen-parsable OUTCAR with more realistic dummy data."""
    forces = atoms.get_forces()
    cart_positions = atoms.get_positions()
    drift = np.mean(forces, axis=0)
    n_atoms = len(atoms)
    volume = atoms.get_volume()

    direct_cell = atoms.get_cell()
    reciprocal_cell = np.linalg.inv(direct_cell).T
    direct_lengths = np.linalg.norm(direct_cell, axis=1)
    reciprocal_lengths = np.linalg.norm(reciprocal_cell, axis=1)
    scaled_positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    unique_symbols = sorted(list(set(symbols)))

    with open(outcar_name, "w") as f:
        # --- Dummy Header & Parameters for pymatgen parsing ---
        f.write(" vasp.6.5.0 24Aug23 (build 2023-08-24 10:00:00) complex\n")
        f.write("\n")
        for sym in unique_symbols:
            f.write(f" POTCAR:    PAW_PBE {sym.ljust(2)} 01Jan2000 (PAW_PBE {sym.ljust(2)} 01Jan2000)\n")
        f.write("\n")

        # --- Dummy INCAR section ---
        f.write(" INCAR:\n")
        f.write("   ENCUT  =      520.000\n")
        f.write("   ISMEAR =          0\n")
        f.write("   SIGMA  =        0.05\n")
        f.write("   ISIF   =          3\n")
        f.write("   IBRION =          2\n")
        f.write("\n")

        # --- Dummy Parameters section ---
        f.write(" Parameters (and plain-wave basis):\n")
        f.write(" total plane-waves  NPLWV =      10000\n")
        # --- Dummy table for per-kpoint plane waves ---
        f.write("\n\n\n" + "-"*104 + "\n\n\n")
        f.write(" k-point   1 :       0.0000    0.0000    0.0000\n")
        f.write("  number of plane waves:    10000\n\n")
        f.write(" maximum and minimum number of plane-waves:    10000   10000\n")
        f.write(f"  NELECT =    {float(n_atoms * 6):.4f}\n")
        f.write(f"  NBANDS =        {n_atoms * 4}\n")
        f.write("\n")

        # --- Lattice and Geometry ---
        f.write(f" volume of cell : {volume:12.4f}\n\n")
        f.write("  direct lattice vectors                    reciprocal lattice vectors\n")
        for i in range(3):
            d = direct_cell[i]
            r = reciprocal_cell[i]
            f.write(f"    {d[0]:12.9f} {d[1]:12.9f} {d[2]:12.9f}    {r[0]:12.9f} {r[1]:12.9f} {r[2]:12.9f}\n")
        f.write("\n")
        f.write("  length of vectors\n")
        f.write(f"    {direct_lengths[0]:12.9f} {direct_lengths[1]:12.9f} {direct_lengths[2]:12.9f}    {reciprocal_lengths[0]:12.9f} {reciprocal_lengths[1]:12.9f} {reciprocal_lengths[2]:12.9f}\n")
        f.write("\n")

        # --- Dummy Electronic Structure ---
        f.write(" E-fermi :   0.0000     alpha+bet :       0.0000     alpha-bet :       0.0000\n\n")

        # --- Positions and Forces ---
        f.write("  position of ions in fractional coordinates (direct lattice)\n")
        for pos in scaled_positions:
            f.write(f"     {pos[0]:11.9f} {pos[1]:11.9f} {pos[2]:11.9f}\n")
        f.write("\n")
        f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
        f.write(" -----------------------------------------------------------------------------------\n")
        for i in range(n_atoms):
            x, y, z = cart_positions[i]
            fx, fy, fz = forces[i]
            f.write(f" {x:12.5f} {y:12.5f} {z:12.5f}   {fx:12.6f} {fy:12.6f} {fz:12.6f}\n")
        f.write(" -----------------------------------------------------------------------------------\n")
        f.write(f"  total drift:                         {drift[0]:12.6f} {drift[1]:12.6f} {drift[2]:12.6f}\n\n")

        # --- Stress Tensor (converted to kB) ---
        f.write("  TOTAL-FORCE (eV/Angst)  ... external pressure =      0.00 kB  Pullay stress =      0.00 kB\n")
        f.write("  in kB         XX          YY          ZZ          XY          YZ          ZX\n")
        try:
            stress_kBar = atoms.get_stress(voigt=True) * 160.21766208 * 10
            s_vasp = [stress_kBar[0], stress_kBar[1], stress_kBar[2], stress_kBar[5], stress_kBar[3], stress_kBar[4]]
            f.write(f"  Total    {s_vasp[0]:11.4f} {s_vasp[1]:11.4f} {s_vasp[2]:11.4f} {s_vasp[3]:11.4f} {s_vasp[4]:11.4f} {s_vasp[5]:11.4f}\n")
        except Exception:
            f.write("  Total         0.0000      0.0000      0.0000      0.0000      0.0000      0.0000\n")
        f.write("\n")

        # --- Final Energy ---
        f.write(" FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
        f.write(" ---------------------------------------------------\n")
        f.write(f"  free  energy   TOTEN  =  {energy:20.8f} eV\n")
        f.write(f"  energy  without entropy=  {energy:20.8f}  energy(sigma->0) =  {energy:20.8f}\n")
    print(f" Wrote {outcar_name} (with dummy data for pymatgen)")


def write_vasprun_xml(atoms, energy, xml_name="vasprun-mace.xml"):
    """Write vasprun.xml in phonopy-compatible minimal format."""
    forces = atoms.get_forces()
    root = ET.Element("modeling")

    gen = ET.SubElement(root, "generator")
    ET.SubElement(gen, "i", name="program", type="string").text = "vasp"
    ET.SubElement(gen, "i", name="version", type="string").text = "6.5.0  "

    atominfo = ET.SubElement(root, "atominfo")
    ET.SubElement(atominfo, "atoms").text = str(len(atoms))

    struct = ET.SubElement(root, "structure", name="finalpos")
    crystal = ET.SubElement(struct, "crystal")
    basis = ET.SubElement(crystal, "varray", name="basis")
    for vec in atoms.get_cell():
        ET.SubElement(basis, "v").text = f" {vec[0]:16.8f} {vec[1]:16.8f} {vec[2]:16.8f} "
    ET.SubElement(crystal, "i", name="volume").text = f" {atoms.get_volume():.8f} "
    positions = ET.SubElement(struct, "varray", name="positions")
    for pos in atoms.get_scaled_positions():
        ET.SubElement(positions, "v").text = f" {pos[0]:16.8f} {pos[1]:16.8f} {pos[2]:16.8f} "

    calc = ET.SubElement(root, "calculation")
    varr = ET.SubElement(calc, "varray", name="forces")
    for f in forces:
        ET.SubElement(varr, "v").text = f"{f[0]: .10f}  {f[1]: .10f}  {f[2]: .10f}"

    energy_tag = ET.SubElement(calc, "energy")
    ET.SubElement(energy_tag, "i", name="e_fr_energy").text = f"{energy:.10f}"

    ET.indent(root, space="  ", level=0)
    tree = ET.ElementTree(root)
    with open(xml_name, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8")

    print(f" Wrote {xml_name} (phonopy-compatible XML)")


def write_calc_results_json(atoms, energy, filename="calc_results.json"):
    """Write PyDefect-compatible calc_results.json file."""
    pmg_struct = AseAtomsAdaptor.get_structure(atoms)
    data = {
        "@module": "pydefect.analyzer.calc_results",
        "@class": "CalcResults",
        "@version": "0.9.4",
        "structure": pmg_struct.as_dict(),
        "energy": float(energy),
        "magnetization": 0.0,
        "potentials": [0.0 for _ in range(len(atoms))],
        "electronic_conv": True,
        "ionic_conv": True,
    }
    with open(filename, "w") as f:
        json.dump(data, f, cls=MontyEncoder, indent=2)
    print(f"🧩 Wrote {filename} (PyDefect-compatible)")


def write_pydefect_dummy_files():
    """Write dummy files for pydefect in the current directory."""
    json_content = '{"@module":"pydefect.analyzer.band_edge_states","@class":"PerfectBandEdgeState","@version":"0.9.7","vbm_info":{"@module":"pydefect.analyzer.band_edge_states","@class":"EdgeInfo","@version":"0.9.7","band_idx":0,"kpt_coord":[0.0,0.0,0.0],"orbital_info":{"@module":"pydefect.analyzer.band_edge_states","@class":"OrbitalInfo","@version":"0.9.7","energy":0.0,"orbitals":{},"occupation":1.0,"participation_ratio":null}},"cbm_info":{"@module":"pydefect.analyzer.band_edge_states","@class":"EdgeInfo","@version":"0.9.7","band_idx":0,"kpt_coord":[0.0,0.0,0.0],"orbital_info":{"@module":"pydefect.analyzer.band_edge_states","@class":"OrbitalInfo","@version":"0.9.7","energy":5.0,"orbitals":{},"occupation":0.0,"participation_ratio":null}}}'
    yaml_content = """system: ZnO
vbm: 0.0
cbm: 5.0
ele_dielectric_const:
- - 1.0
  - 0.0
  - 0.0
- - 0.0
  - 1.0
  - 0.0
- - 0.0
  - 0.0
  - 1.0
ion_dielectric_const:
- - 1.0
  - 0.0
  - 0.0
- - 0.0
  - 1.0
  - 0.0
- - 0.0
  - 0.0
  - 1.0
"""
    with open("perfect_band_edge_state.json", "w") as f:
        f.write(json_content)
    with open("unitcell.yaml", "w") as f:
        f.write(yaml_content)


# ============================================================
# 5) Relaxation routine
# ============================================================
def relax_structure(input_file="POSCAR", fmax=0.01, smax=0.001,
                    device="cpu", isif=2, fix_axis=None,
                    quiet=False, contcar_name="CONTCAR",
                    outcar_name="OUTCAR", xml_name="vasprun-mace.xml",
                    make_pdf=True, write_json=False):
    atoms = read(input_file)
    prefix = os.path.basename(input_file)
    output_dir = os.path.dirname(input_file)
    if not quiet:
        print(f" Loaded structure from {input_file} ({len(atoms)} atoms)")
    calc = get_mace_calculator(device)
    atoms.calc = calc
    target = get_relax_target(atoms, isif, fix_axis or [])

    energies, steps, forces_hist, stress_hist = [], [], [], []

    if isif in (0, 1):
        e = atoms.get_potential_energy()
        write_outcar(atoms, e, outcar_name)
        write_vasprun_xml(atoms, e, xml_name)
        if write_json:
            write_calc_results_json(atoms, e, filename=os.path.join(output_dir, "calc_results.json"))
        write(contcar_name, atoms, format="vasp")
    else:
        if not quiet:
            print(f"⚙️  Starting FIRE relaxation (fmax={fmax:.4f} eV/Å, smax={smax:.4f} eV/Å³, ISIF={isif})")

        with Trajectory(os.path.join(output_dir, f"relax-{prefix}.traj"), "w", target) as traj:
            opt = FIRE(target, maxstep=0.1, dt=0.1, trajectory=traj)

            def log_callback():
                e = atoms.get_potential_energy()
                fmax_cur = np.abs(atoms.get_forces()).max()
                stress_cur = np.abs(atoms.get_stress()).max() if isif >= 3 else 0.0
                steps.append(len(steps))
                energies.append(e)
                forces_hist.append(fmax_cur)
                stress_hist.append(stress_cur)
                print(f" Step {len(steps):4d} | E = {e:.6f} eV | Fmax={fmax_cur:.5f} | σmax={stress_cur:.5f}")
                if fmax_cur < fmax and (isif < 3 or stress_cur < smax):
                    print("✅ Converged: force & stress thresholds satisfied.")
                    if hasattr(opt, "stop"):
                        opt.stop()
                    else:
                        raise SystemExit

            opt.attach(log_callback, interval=1)
            try:
                opt.run(fmax=fmax)
            except SystemExit:
                pass

        e = atoms.get_potential_energy()
        write(contcar_name, atoms, format="vasp")
        write_outcar(atoms, e, outcar_name)
        write_vasprun_xml(atoms, e, xml_name)
        if write_json:
            write_calc_results_json(atoms, e, filename=os.path.join(output_dir, "calc_results.json"))

    # ============================================================
    # 6) Optional Log plot
    # ============================================================
    if make_pdf:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        if energies:
            ax1.plot(steps, energies, color="tab:blue", marker="o", lw=1.0, label="Total Energy (eV)")
        else:
            e_final = atoms.get_potential_energy()
            ax1.scatter([0], [e_final], color="tab:blue", label="Single-point Energy (eV)")
        ax1.set_xlabel("Optimization step" if energies else "Single point")
        ax1.set_ylabel("Energy (eV)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.3)

        if energies and forces_hist:
            ax2 = ax1.twinx()
            ax2.plot(steps, forces_hist, color="tab:red", marker="s", lw=1.0, label="Fmax (eV/Å)")
            if isif >= 3:
                ax2.plot(steps, stress_hist, color="tab:green", marker="^", lw=1.0, label="σmax (eV/Å³)")
            ax2.set_ylabel("Force / Stress", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax1.legend(loc="best")

        plt.title(f"Relaxation progress ({prefix})")
        plt.tight_layout()
        pdf_name = os.path.join(output_dir, f"relax-{prefix}_log.pdf")
        plt.savefig(pdf_name)
        plt.close(fig)
        print(f" Saved detailed log plot → {pdf_name}")


# ============================================================
# 7) CLI with batch support
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relax atomic structures using MACE with VASP-like ISIF modes. Supports multiple input files (POSCAR-*).",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input", "-i", type=str, nargs="+", default=["POSCAR"],
                        help="Input file(s) or pattern(s) (e.g. POSCAR-*).")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence threshold (eV/Å).")
    parser.add_argument("--smax", type=float, default=0.001, help="Stress convergence threshold (eV/Å³).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--isif", type=int, default=2, choices=list(range(8)))
    parser.add_argument("--fix-axis", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-pdf", action="store_true", help="Disable log PDF output")
    parser.add_argument("--pydefect", action="store_true", help="Write PyDefect-compatible files (calc_results.json, unitcell.yaml, perfect_band_edge_state.json).")
    parser.add_argument("--contcar", type=str, default=None, help="Output CONTCAR file name.")
    parser.add_argument("--outcar", type=str, default=None, help="Output OUTCAR file name.")
    parser.add_argument("--vasprun", type=str, default=None, help="Output vasprun.xml file name.")

    args = parser.parse_args()
    fix_axis = [ax.strip().lower() for ax in args.fix_axis.split(",") if ax.strip()]

    if args.pydefect:
        write_pydefect_dummy_files()

    input_patterns = args.input
    input_files = []
    for pat in input_patterns:
        input_files.extend(glob.glob(pat))
    input_files = sorted(set(input_files))

    if not input_files:
        print(f"❌ No files match pattern(s): {input_patterns}")
        sys.exit(1)

    if (args.contcar or args.outcar or args.vasprun) and len(input_files) > 1:
        print("⚠️ WARNING: Custom output names (--contcar, --outcar, --vasprun) are used with multiple input files.")
        print("Output files may be overwritten. Consider running files one by one.")

    orig_stdout = sys.stdout

    for infile in input_files:
        prefix = os.path.basename(infile)
        output_dir = os.path.dirname(infile)
        log_name = os.path.join(output_dir, f"relax-{prefix}_log.txt")

        contcar_name = os.path.join(output_dir, args.contcar or f"CONTCAR-{prefix}")
        outcar_name = os.path.join(output_dir, args.outcar or f"OUTCAR-{prefix}")
        xml_name = os.path.join(output_dir, args.vasprun or f"vasprun-{prefix}.xml")

        try:
            with Logger(log_name) as lg:
                sys.stdout = lg
                if args.pydefect:
                    print("NOTE: perfect_band_edge_state.json and unitcell.yaml were written as dummy files for pydefect dei and pydefect des.")
                print(f"▶ Using MACE on '{infile}' | ISIF={args.isif} | fmax={args.fmax} | smax={args.smax} | device={args.device}")
                relax_structure(
                    input_file=infile,
                    fmax=args.fmax,
                    smax=args.smax,
                    device=args.device,
                    isif=args.isif,
                    fix_axis=fix_axis,
                    quiet=args.quiet,
                    contcar_name=contcar_name,
                    outcar_name=outcar_name,
                    xml_name=xml_name,
                    make_pdf=not args.no_pdf,
                    write_json=args.pydefect,
                )
                results_path_info = f"in '{output_dir}'" if output_dir else "in the current directory"
                print(f"✅ Finished {infile} → Results saved {results_path_info}")
                print("-" * 80)
        except Exception as e:
            sys.stdout = orig_stdout
            print(f"[SKIP] {infile}: {e}")
            continue
        finally:
            sys.stdout = orig_stdout
