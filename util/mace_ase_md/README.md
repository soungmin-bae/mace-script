# MD with MACE + ASE (NPT / NVT)

> Script name: **`mace_ase_md.py`**

Run short molecular dynamics using a MACE potential with ASE.  
Inputs: `POSCAR`  
Outputs: `md.traj`, `md.log`, `XDATCAR`, `md.csv`

- **Ensembles**:  
  - `npt` (default): Nose–Hoover barostat (ASE `NPT`)  
  - `nte` (=NVT): prefers Nose–Hoover chain (ASE `NoseHooverChainNVT`), falls back to Berendsen (`NVTBerendsen`)
- **Devices**: CPU by default; set `--device cuda` to use GPU (depends on your MACE install).

---

## 1) Requirements
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [MACE](https://github.com/ACEsuit/mace)

---

## 2) Quick Start

Place your `POSCAR` and a MACE model file (e.g., `mace-omat-0-small.model`) in the working directory.

### NPT (default)
```bash
python mace_ase_md.py --ensemble npt --temp 600 --press 1.0 --ttau 100 --ptau 1000                       --device cuda --nsteps 20000 --save-every 100
```

### NVT (NTE)
```bash
python mace_ase_md.py --ensemble nte --temp 600 --ttau 100 --nsteps 5000
```

### Reproducible run
```bash
python mace_ase_md.py --ensemble npt --temp 300 --press 0.0 --ttau 100 --ptau 1000                       --seed 42 --print-every 10 --save-every 100
```

---

## 3) CLI

```
python mace_ase_md.py -h
```

Key options (defaults shown by `-h`):

- `--model`: Path to MACE model (`.model`)
- `--device {cpu,cuda}`: Compute device
- `--ensemble {npt, nte}`: NPT or NVT
- `--temp`: Target temperature (K)
- `--press`: Target pressure (GPa, NPT only; isotropic)
- `--tstep`: Time step (fs)
- `--nsteps`: Number of MD steps
- `--ttau`: Thermostat time constant (fs)
- `--ptau`: Barostat time constant (fs, NPT only)
- `--save-every`: Interval for writing `md.traj` and logger
- `--xdat-every`: Interval for writing `XDATCAR` frames
- `--print-every`: Console print interval
- `--seed`: RNG seed (fixed if you want reproducibility)
- `--csv`, `--xdatcar`, `--traj`, `--log`: Output paths

---

## 4) Outputs

- **`md.traj`** — ASE trajectory (use `ase gui md.traj` or read via `ase.io.read`)
- **`md.log`** — Text log from `MDLogger` (energies, T, stress, …)
- **`XDATCAR`** — VASP-style trajectory snapshots (scaled positions + cell per frame)
- **`md.csv`** — Tabular observables per step:

| column       | description                                 |
|--------------|---------------------------------------------|
| `step`       | MD step index (starts at 0 record)          |
| `time_fs`    | Time (fs)                                   |
| `Epot_eV`    | Potential energy (eV)                        |
| `Ekin_eV`    | Kinetic energy (eV)                          |
| `Etot_eV`    | Total energy (eV)                            |
| `T_K`        | Instantaneous temperature (K)                |
| `Vol_A3`     | Cell volume (Å³)                             |
| `P_GPa`      | Hydrostatic pressure estimate (GPa)          |
| `H_eV`       | `E + pV` (enthalpy-like, eV)                 |

> Pressure is computed from the virial: \( p = -\mathrm{tr}(\sigma)/3 \) converted from eV/Å³ to GPa.
