# mace_ase_relax.py

**Automated structure relaxation and single-point calculations using MACE + ASE**

This script provides a fully automated workflow for structure relaxation using MACE (Machine-Learned Atomistic Energy) models with ASE’s FIRE optimizer.  
It supports VASP-like `ISIF` modes (0–7), Phonopy-compatible `vasprun.xml` output, and batch processing of multiple `POSCAR-*` files.

---

## Features

- Uses **MACE** as the interatomic potential calculator
- ASE **FIRE optimizer** for geometry relaxation
- **VASP ISIF-compatible** relaxation modes (0–7)
- **Phonopy-compatible** `vasprun.xml` generation
- **Batch processing** for multiple structures (e.g., `POSCAR-*`)
- Automatic generation of **log files** and **PDF energy/force plots**
- Supports **fixed-axis relaxation** via `--fix-axis a,b,c`
- Compatible with ASE ≥ 3.20, Python ≥ 3.8

---

## 📂 Output Files

For each input file (`POSCAR-001`, `POSCAR-002`, ...), the following files are produced:

```
CONTCAR-POSCAR-001
OUTCAR-POSCAR-001
vasprun-POSCAR-001.xml
relax-POSCAR-001_log.txt
relax-POSCAR-001_log.pdf
```

---

## Usage

```bash
# Standard atomic relaxation (ISIF=2: atomic positions only)
python mace_ase_relax.py -i POSCAR

# Batch relaxation for multiple structures
python mace_ase_relax.py -i POSCAR-* --isif 2

# Full cell relaxation (atoms + lattice)
python mace_ase_relax.py -i POSCAR --isif 3

# Relaxation with fixed a-axis
python mace_ase_relax.py -i POSCAR --isif 3 --fix-axis a

# Single-point calculation
python mace_ase_relax.py -i POSCAR --isif 0
```

---

## Command Line Options

| Option | Description | Default |
|--------|--------------|----------|
| `-i`, `--input` | Input file(s) or pattern (e.g., `POSCAR-*`) | `POSCAR` |
| `--isif` | VASP ISIF mode (0–7) | 2 |
| `--fmax` | Force convergence threshold (eV/Å) | 0.01 |
| `--smax` | Stress convergence threshold (eV/Å³) | 0.001 |
| `--device` | Calculation device (`cpu` or `cuda`) | `cpu` |
| `--fix-axis` | Fix lattice axes during relaxation (comma-separated) | None |
| `--quiet` | Suppress detailed output | False |

---

## Dependencies

- Python ≥ 3.8  
- ASE ≥ 3.20  
- matplotlib  
- numpy  
- MACE (`pip install git+https://github.com/ACEsuit/mace.git`)

---

## Model Configuration

Edit the MACE model path in the script:

```python
def get_mace_calculator(device="cpu"):
    from mace.calculators import MACECalculator
    return MACECalculator(
        model_path="/path/to/your/mace-model.model",
        device=device,
        default_dtype="float32",
    )
```

---

## Log and Visualization

- `relax-*_log.txt` → text log (energy, forces, stress per step)  
- `relax-*_log.pdf` → plotted energy/force/stress convergence  
- `vasprun-*.xml` → phonopy-compatible XML output  

---
