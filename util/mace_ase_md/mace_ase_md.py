#!/usr/bin/env python
# -*- coding: utf-8 -*-
# POSCAR --(MACE + ASE / NPT or NVT(NTE))--> (md.traj, md.log, XDATCAR, md.csv)

from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.npt import NPT
# NVT: prefer Nose–Hoover chain; fallback to Berendsen if unavailable.
try:
    from ase.md.nose_hoover_chain import NoseHooverChainNVT as NVT_NHC
except Exception:
    NVT_NHC = None
try:
    from ase.md.nvtberendsen import NVTBerendsen as NVT_Ber
except Exception:
    NVT_Ber = None

from ase.md.logger import MDLogger
from ase.geometry import cellpar_to_cell
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
import ase.units as u
import numpy as np
import argparse, csv

# --- Defaults -----------------------------------------------------------------
MODEL_PATH_DEF     = "/Users/bama/package/MACE/models/mace-omat-0-small.model"
DEVICE_DEF         = "cpu"         # use "cuda" to enable GPU
TEMP_K_DEF         = 300.0         # target temperature [K]
PRESS_GPa_DEF      = 0.0           # target pressure [GPa] (NPT only)
TSTEP_fs_DEF       = 2.0           # MD time step [fs]
NSTEPS_DEF         = 20000         # total number of MD steps
THERMO_Tau_fs_DEF  = 100.0         # thermostat time constant [fs]
BARO_Tau_fs_DEF    = 1000.0        # barostat time constant [fs] (NPT only)
SAVE_EVERY_DEF     = 100           # save interval for traj/log
XDAT_EVERY_DEF     = 1             # XDATCAR write interval
PRINT_EVERY_DEF    = 1             # console print interval
RNG_SEED_DEF       = None          # set an int (e.g., 42) for reproducibility
CSV_PATH_DEF       = "md.csv"      # CSV log file
XDATCAR_DEF        = "XDATCAR"     # XDATCAR file
TRAJ_PATH_DEF      = "md.traj"     # ASE trajectory file
LOG_PATH_DEF       = "md.log"      # MD text log
ENSEMBLE_DEF       = "npt"         # "npt" (default) or "nte" (=NVT)
# -----------------------------------------------------------------------------

# Unit conversion: 1 (eV/Å^3) = 160.21766208 GPa
EV_A3_TO_GPa = 160.21766208

def get_mace_calculator(model_path, device):
    """Construct MACE calculator (float64 recommended to match model dtype)."""
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=[model_path], device=device, default_dtype="float64")

def parse_poscar_header_for_xdatcar(poscar_path="POSCAR"):
    """Read species and counts from POSCAR header for XDATCAR blocks."""
    with open(poscar_path, "r") as f:
        lines = [next(f) for _ in range(7)]
    species = lines[5].split()
    counts  = [int(x) for x in lines[6].split()]
    return species, counts

def build_argparser():
    """Build CLI with defaults shown and examples in the epilog."""
    class _Fmt(argparse.ArgumentDefaultsHelpFormatter,
               argparse.RawTextHelpFormatter):
        pass

    p = argparse.ArgumentParser(
        description="Minimal NpT or NVT (NTE) MD with MACE + ASE (inputs: POSCAR; outputs: md.traj/md.log/XDATCAR/md.csv)",
        formatter_class=_Fmt,
        epilog=(
            "Examples:\n"
            "  # NPT (Nose–Hoover barostat) — 600 K, 1 GPa, GPU, save every 100 steps\n"
            "  python md.py --ensemble npt --temp 600 --press 1.0 --ttau 100 --ptau 1000 \\\n"
            "               --device cuda --nsteps 20000 --save-every 100\n"
            "\n"
            "  # NVT (NTE; prefers Nose–Hoover chain, falls back to Berendsen) — 600 K, 5,000 steps\n"
            "  python md.py --ensemble nte --temp 600 --ttau 100 --nsteps 5000\n"
            "\n"
            "  # Reproducible run (fixed seed) + adjusted print/save intervals\n"
            "  python md.py --ensemble npt --temp 300 --press 0.0 --ttau 100 --ptau 1000 \\\n"
            "               --seed 42 --print-every 10 --save-every 100\n"
        )
    )
    p.add_argument("--model", default=MODEL_PATH_DEF, help="MACE model path")
    p.add_argument("--device", choices=["cpu","cuda"], default=DEVICE_DEF, help="compute device")
    p.add_argument("--ensemble", choices=["npt","nte"], default=ENSEMBLE_DEF,
                   help="MD ensemble: npt (Nose–Hoover barostat) or nte (=NVT; Nose–Hoover chain preferred, else Berendsen)")
    p.add_argument("--temp", type=float, default=TEMP_K_DEF, help="target temperature [K]")
    p.add_argument("--press", type=float, default=PRESS_GPa_DEF, help="target pressure [GPa] (NPT only)")
    p.add_argument("--tstep", type=float, default=TSTEP_fs_DEF, help="MD time step [fs]")
    p.add_argument("--nsteps", type=int, default=NSTEPS_DEF, help="number of MD steps")
    p.add_argument("--ttau", type=float, default=THERMO_Tau_fs_DEF, help="thermostat time constant [fs]")
    p.add_argument("--ptau", type=float, default=BARO_Tau_fs_DEF, help="barostat time constant [fs] (NPT only)")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY_DEF, help="traj/log save interval")
    p.add_argument("--xdat-every", type=int, default=XDAT_EVERY_DEF, help="XDATCAR write interval")
    p.add_argument("--print-every", type=int, default=PRINT_EVERY_DEF, help="stdout print interval")
    p.add_argument("--seed", type=int, default=RNG_SEED_DEF, help="random seed (None for random)")
    p.add_argument("--csv", default=CSV_PATH_DEF, help="CSV log path for MD outputs")
    p.add_argument("--xdatcar", default=XDATCAR_DEF, help="XDATCAR path")
    p.add_argument("--traj", default=TRAJ_PATH_DEF, help="ASE trajectory path")
    p.add_argument("--log", default=LOG_PATH_DEF, help="MD text log path")
    return p

def main():
    args = build_argparser().parse_args()

    # 0) Read input structure.
    atoms = read("POSCAR")

    # Upper-triangular cell is recommended for NPT (harmless for NVT; keeps cell normalized).
    tri_cell = cellpar_to_cell(atoms.cell.cellpar())
    atoms.set_cell(tri_cell, scale_atoms=True)
    atoms.pbc = True

    # Calculator.
    atoms.calc = get_mace_calculator(args.model, args.device)

    # Initialize velocities; remove net translation and rotation.
    rng = (np.random.default_rng(args.seed) if args.seed is not None else None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, force_temp=True, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    # 1) MD integrator setup.
    timestep  = args.tstep * u.fs
    ttime     = args.ttau  * u.fs

    if args.ensemble == "npt":
        # NPT with Nose–Hoover barostat (ASE NPT).
        extstress = args.press * u.GPa
        pfact     = (args.ptau * u.fs) ** 2 * u.GPa
        dyn = NPT(
            atoms,
            timestep=timestep,
            temperature_K=args.temp,
            externalstress=extstress,
            ttime=ttime,
            pfactor=pfact,
        )
    else:
        # NVT (NTE): prefer Nose–Hoover chain; fallback to Berendsen.
        if NVT_NHC is not None:
            dyn = NVT_NHC(
                atoms,
                timestep=timestep,
                temperature_K=args.temp,
                tdamp=ttime,   # thermostat damping time constant
            )
        elif NVT_Ber is not None:
            dyn = NVT_Ber(
                atoms,
                timestep=timestep,
                temperature_K=args.temp,
                taut=ttime,    # Berendsen thermostat time constant
            )
        else:
            raise ImportError(
                "NVT integrator not found in this ASE installation. "
                "Please install/update ASE with NoseHooverChainNVT or NVTBerendsen."
            )

    # 2) Logging: trajectory + text logger.
    traj = Trajectory(args.traj, "w", atoms)
    dyn.attach(traj.write, interval=args.save_every)
    logfile = open(args.log, "w")
    dyn.attach(MDLogger(dyn, atoms, logfile, header=True, stress=True, peratom=False),
               interval=args.save_every)

    # 3) XDATCAR setup.
    species, counts = parse_poscar_header_for_xdatcar("POSCAR")
    xdat_handle = open(args.xdatcar, "w")

    # 4) CSV (custom observables) setup.
    csv_handle = open(args.csv, "w", newline="")
    csv_writer = csv.writer(csv_handle)
    csv_writer.writerow(["step","time_fs","Epot_eV","Ekin_eV","Etot_eV","T_K","Vol_A3","P_GPa","H_eV"])

    # State & utilities.
    config_idx = 0
    step_counter = 0

    def write_xdatcar_block():
        """Append one XDATCAR configuration block from current Atoms state."""
        nonlocal config_idx
        config_idx += 1
        xdat_handle.write(" ".join(species) + "\n")
        xdat_handle.write("    1.000000\n")
        cell = atoms.cell.array
        for vec in cell:
            xdat_handle.write(f" {vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")
        xdat_handle.write(" " + "              ".join(species) + "\n")
        xdat_handle.write("".join([f"{c:17d}" for c in counts]) + "\n")
        xdat_handle.write(f"Direct configuration= {config_idx:5d}\n")
        for s in atoms.get_scaled_positions(wrap=True):
            xdat_handle.write(f"   {s[0]:.8f}   {s[1]:.8f}   {s[2]:.8f}\n")

    def collect_observables():
        """Compute a set of common MD observables from the current state."""
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        etot = epot + ekin
        temp = atoms.get_temperature()
        vol  = atoms.get_volume()
        sigma = atoms.get_stress(voigt=False)  # stress tensor in eV/Å^3
        p_eVa3 = -np.trace(sigma) / 3.0
        p_GPa  = p_eVa3 * EV_A3_TO_GPa
        H = etot + p_eVa3 * vol            # enthalpy-like quantity (E + pV) in eV
        t_fs = step_counter * args.tstep
        return epot, ekin, etot, temp, vol, p_GPa, H, t_fs

    def print_status_line(epot, ekin, etot, temp, vol, p_GPa, H, t_fs):
        """Pretty single-line status for stdout."""
        print(
            f"Step{step_counter:7d} | t={t_fs:7.2f} fs | "
            f"Epot={epot: .6f} eV | Ekin={ekin: .6f} eV | Etot={etot: .6f} eV | "
            f"T={temp:7.2f} K | Vol={vol:8.3f} Å^3 | P={p_GPa: 7.4f} GPa | H={H: .6f} eV"
        )

    def write_csv_line(epot, ekin, etot, temp, vol, p_GPa, H, t_fs):
        """Append one row of observables to the CSV log."""
        csv_writer.writerow([step_counter, t_fs, epot, ekin, etot, temp, vol, p_GPa, H])

    # ▶ Initial (step 0) record: console + XDATCAR + CSV.
    epot, ekin, etot, temp, vol, p_GPa, H, t_fs = collect_observables()
    print_status_line(epot, ekin, etot, temp, vol, p_GPa, H, t_fs)
    write_xdatcar_block()
    write_csv_line(epot, ekin, etot, temp, vol, p_GPa, H, t_fs)
    step_counter += 1  # subsequent integration starts at step 1

    # Per-step callback.
    def on_step():
        nonlocal step_counter
        epot, ekin, etot, temp, vol, p_GPa, H, t_fs = collect_observables()
        if (step_counter % args.print_every) == 0:
            print_status_line(epot, ekin, etot, temp, vol, p_GPa, H, t_fs)
        if (step_counter % args.xdat_every) == 0:
            write_xdatcar_block()
        write_csv_line(epot, ekin, etot, temp, vol, p_GPa, H, t_fs)
        step_counter += 1

    dyn.attach(on_step, interval=1)

    # 5) Run MD.
    dyn.run(args.nsteps)

    # 6) Finalize.
    xdat_handle.close()
    csv_handle.close()
    print(f"Done ({args.ensemble.upper()} MD): outputs → {args.traj} / {args.log} / {args.xdatcar} / {args.csv}")

if __name__ == "__main__":
    main()
