#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Relax atomic structures using a local MACE model (ASE + FIRE optimizer).

Patched version (2025-10-26):
  ✅ Fixed file descriptor leaks (Logger + Trajectory context manager)
  ✅ Restores sys.stdout properly after each file
  ✅ Added matplotlib 'Agg' backend for headless batch mode
  ✅ Added --no-pdf option to disable log PDF output
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


# ============================================================
# 1) MACE calculator
# ============================================================
def get_mace_calculator(device="cpu"):
    from mace.calculators import MACECalculator
    # Dockerfile에 설정된 환경 변수에서 모델 경로를 가져옵니다.
    model_path = os.getenv("MACE_MODEL_PATH", "/Users/bama/package/MACE/models/mace-omat-0-small-fp32.model")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("MACE model not found at path specified by MACE_MODEL_PATH env var.")

    return MACECalculator(
        model_path=model_path,
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
        print("🧊 ISIF=0 → Single-point calculation.")
        return atoms
    elif isif == 1:
        print("📊 ISIF=1 → Stress evaluation only.")
        atoms.set_constraint(FixAtoms(range(len(atoms))))
        return atoms
    elif isif == 2:
        print("🔧 ISIF=2 → Relax atomic positions only.")
        return atoms
    elif isif == 3:
        print("🏗️ ISIF=3 → Relax atoms + full cell (volume & shape).")
        mask = build_axis_mask(fix_axis)
        print(f"📏 Fixed axes: {', '.join(fix_axis).upper() or '(none)'}")
        return UnitCellFilter(atoms, mask=mask)
    elif isif == 4:
        print("🏗️ ISIF=4 → Relax cell only.")
        atoms.set_constraint(FixAtoms(range(len(atoms))))
        return UnitCellFilter(atoms, mask=build_axis_mask(fix_axis))
    elif isif == 5:
        print("🏗️ ISIF=5 → Relax atoms + shape (volume fixed).")
        return ExpCellFilter(atoms, mask=build_axis_mask(fix_axis))
    elif isif == 6:
        print("🏗️ ISIF=6 → Relax atoms + volume (shape fixed).")
        return StrainFilter(atoms, mask=build_axis_mask(fix_axis))
    elif isif == 7:
        print("🏗️ ISIF=7 → Relax atoms + anisotropic shape.")
        return ExpCellFilter(atoms, mask=build_axis_mask(fix_axis))
    else:
        raise ValueError("Unsupported ISIF. Choose 0–7.")


# ============================================================
# 4) Output writers
# ============================================================
def write_outcar(atoms, energy, outcar_name="OUTCAR"):
    forces = atoms.get_forces()
    positions = atoms.get_positions()
    drift = np.mean(forces, axis=0)
    n_atoms = len(atoms)
    with open(outcar_name, "w") as f:
        f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
        f.write(" -----------------------------------------------------------------------------------\n")
        for i in range(n_atoms):
            x, y, z = positions[i]
            fx, fy, fz = forces[i]
            f.write(f" {x:12.5f} {y:12.5f} {z:12.5f}   {fx:12.6f} {fy:12.6f} {fz:12.6f}\n")
        f.write(" -----------------------------------------------------------------------------------\n")
        f.write(f"  total drift:                         {drift[0]:12.6f} {drift[1]:12.6f} {drift[2]:12.6f}\n\n")
        f.write(" FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
        f.write(" ---------------------------------------------------\n")
        f.write(f"  free  energy   TOTEN  =  {energy:20.8f} eV\n")
        f.write(f"  energy  without entropy=  {energy:20.8f}  energy(sigma->0) =  {energy:20.8f}\n")
    print(f"📝 Wrote {outcar_name}")


def write_vasprun_xml(atoms, energy, xml_name="vasprun-mace.xml"):
    """Write vasprun.xml in phonopy-compatible minimal format."""
    forces = atoms.get_forces()
    root = ET.Element("modeling")

    gen = ET.SubElement(root, "generator")
    ET.SubElement(gen, "i", name="program", type="string").text = "vasp"
    ET.SubElement(gen, "i", name="version", type="string").text = "6.5.0  "

    atominfo = ET.SubElement(root, "atominfo")
    ET.SubElement(atominfo, "atoms").text = str(len(atoms))

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

    print(f"📝 Wrote {xml_name} (phonopy-compatible XML)")


# ============================================================
# 5) Relaxation routine
# ============================================================
def relax_structure(input_file="POSCAR", fmax=0.01, smax=0.001,
                    device="cpu", isif=2, fix_axis=None,
                    quiet=False, contcar_name="CONTCAR",
                    outcar_name="OUTCAR", xml_name="vasprun-mace.xml",
                    make_pdf=True):
    atoms = read(input_file)
    prefix = os.path.basename(input_file)
    if not quiet:
        print(f"📥 Loaded structure from {input_file} ({len(atoms)} atoms)")
    calc = get_mace_calculator(device)
    atoms.calc = calc
    target = get_relax_target(atoms, isif, fix_axis or [])

    energies, steps, forces_hist, stress_hist = [], [], [], []

    if isif in (0, 1):
        e = atoms.get_potential_energy()
        write_outcar(atoms, e, outcar_name)
        write_vasprun_xml(atoms, e, xml_name)
        print(f"DEBUG: Attempting to write CONTCAR to {contcar_name}") # ADD THIS
        write(contcar_name, atoms, format="vasp")
    else:
        if not quiet:
            print(f"⚙️  Starting FIRE relaxation (fmax={fmax:.4f} eV/Å, smax={smax:.4f} eV/Å³, ISIF={isif})")

        with Trajectory(f"relax-{prefix}.traj", "w", target) as traj:
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
        print(f"DEBUG: Attempting to write CONTCAR to {contcar_name}") # ADD THIS
        write(contcar_name, atoms, format="vasp")
        write_outcar(atoms, e, outcar_name)
        write_vasprun_xml(atoms, e, xml_name)

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
        pdf_name = f"data/relax-{prefix}_log.pdf" # PDF 파일도 data/ 폴더에 저장되도록 수정
        plt.savefig(pdf_name)
        plt.close(fig)
        print(f"📈 Saved detailed log plot → {pdf_name}")


# ============================================================
# 7) CLI with batch support
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relax atomic structures using MACE with VASP-like ISIF modes. Supports multiple input files (POSCAR-*).",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input", "-i", type=str, nargs='+', default=["POSCAR"],
                        help="Input file(s) or pattern(s) (e.g. POSCAR-*).")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--smax", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--isif", type=int, default=2, choices=list(range(8)))
    parser.add_argument("--fix-axis", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-pdf", action="store_true", help="Disable log PDF output")  # <-- added

    args = parser.parse_args()
    fix_axis = [ax.strip().lower() for ax in args.fix_axis.split(",") if ax.strip()]

    input_patterns = args.input
    input_files = []
    for pat in input_patterns:
        input_files.extend(glob.glob(pat))
    input_files = sorted(set(input_files))

    if not input_files:
        print(f"❌ No files match pattern(s): {input_patterns}")
        sys.exit(1)

    orig_stdout = sys.stdout

    for infile in input_files:
        prefix = os.path.basename(infile)
        log_name = f"data/relax-{prefix}_log.txt"
        try:
            with Logger(log_name) as lg:
                sys.stdout = lg
                print(f"▶ Using MACE on '{infile}' | ISIF={args.isif} | fmax={args.fmax} | smax={args.smax} | device={args.device}")
                relax_structure(
                    input_file=infile,
                    fmax=args.fmax,
                    smax=args.smax,
                    device=args.device,
                    isif=args.isif,
                    fix_axis=fix_axis,
                    quiet=args.quiet,
                    contcar_name=f"data/CONTCAR-{prefix}",
                    outcar_name=f"data/OUTCAR-{prefix}",
                    xml_name=f"data/vasprun-{prefix}.xml",
                    make_pdf=not args.no_pdf,  # <-- added
                )
                print(f"✅ Finished {infile} → Results saved as data/CONTCAR-{prefix}, data/OUTCAR-{prefix}, data/vasprun-{prefix}.xml, data/relax-{prefix}_log.txt")
                if not args.no_pdf:
                    print(f"and data/relax-{prefix}_log.pdf")
                print("-" * 80)
        except Exception as e:
            sys.stdout = orig_stdout
            print(f"[SKIP] {infile}: {e}")
            continue
        finally:
            sys.stdout = orig_stdout
