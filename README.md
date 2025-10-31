# mace_ase_relax

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
- Automatic generation of **log files** and **PDF energy/force plots** (can be disabled with `--no-logs`)
- Supports **fixed-axis relaxation** via `--fix-axis a,b,c`
- Compatible with ASE ≥ 3.20, Python ≥ 3.8

---

## Docker Usage

The way to run this script is via the pre-built Docker image.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) must be installed and running.

### 1. Pull the Image
Download the official images from Docker Hub. We provide images pre-configured with specific MACE models:

```bash
# Pull the image with the r2scan model
docker pull soungminbae/mace-relax-cpu:r2scan

# Pull the image with the pbe-u model (default small fp32 model)
docker pull soungminbae/mace-relax-cpu:pbe-u

```
> **Note:** If you wish to build an image with a different MACE model, you can use the `--build-arg BUILD_MODEL_NAME=<your_model_file.model>` option during `docker build`.

### 2. Run Calculations
Use the `docker run` command to perform calculations. The key is to mount your current working directory (containing your structure files) into the container's `/app/data` directory using the `-v $(pwd):/app/data` flag.

**Command Template:**
```bash
docker run --rm -v $(pwd):/app/data soungminbae/mace-relax-cpu:<image_tag> --input data/<your_file> [options]
```

**Examples:**

```bash
# Run relaxation using the r2scan model (ISIF=3)
docker run --rm -v $(pwd):/app/data soungminbae/mace-relax-cpu:r2scan --input data/POSCAR 

# Run relaxation using the pbe-u model (ISIF=2)
docker run --rm -v $(pwd):/app/data soungminbae/mace-relax-cpu:pbe-u --input data/POSCAR 

```
> **Note:** All output files (`CONTCAR-*`, `OUTCAR-*`, etc.) will be created in your current directory on your host machine.
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

> Use `--no-logs` to suppress creation of `relax-*_log.txt` and `relax-*_log.pdf`.

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

# Disable TXT/PDF logs
python mace_ase_relax.py -i POSCAR --no-logs
```

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i`, `--input` | Input file(s) or pattern (e.g., `POSCAR-*`) | `POSCAR` |
| `--isif` | VASP ISIF mode (0–7) | 2 |
| `--fmax` | Force convergence threshold (eV/Å) | 0.01 |
| `--smax` | Stress convergence threshold (eV/Å³) | 0.001 |
| `--device` | Calculation device (`cpu` or `cuda`) | `cpu` |
| `--fix-axis` | Fix lattice axes during relaxation (comma-separated, e.g., `a` or `a,c`) | None |
| `--quiet` | Suppress detailed output | False |
| `--no-logs` | Do not write `relax-*_log.txt` and `relax-*_log.pdf` | False |

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

### 🔗 Pretrained models

You can find pretrained **MACE foundations/models** and checkpoints in the **ACEsuit/mace-foundations** repository:  
<https://github.com/ACEsuit/mace-foundations>

For example, models like `2023-12-03-mace-128-L1_epoch-199.model` can be sourced from releases or training artifacts referenced there.  
(Adjust the `model_path` accordingly to your local copy.)

---

## Log and Visualization

- `relax-*_log.txt` → text log (energy, forces, stress per step)  
- `relax-*_log.pdf` → plotted energy/force/stress convergence  
- `vasprun-*.xml` → phonopy-compatible XML output  
- Add `--no-logs` to disable the TXT/PDF outputs and keep stdout-only logging.

---

## Training data

If you are using models trained on the **MPtrj** dataset, please cite the following paper:

```
@article{deng2023chgnet,
      title={CHGNet: Pretrained universal neural network potential for charge-informed atomistic modeling},
      author={Bowen Deng and Peichen Zhong and KyuJung Jun and Janosh Riebesell and Kevin Han and Christopher J. Bartel and Gerbrand Ceder},
      year={2023},
      eprint={2302.14231},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

The MPtrj dataset and related resources are linked from the CHGNet project pages.

> If you redistribute or publish results produced using this script and a pretrained MACE model, please make sure to cite the **MACE** method and any **datasets/models** you rely on (e.g., CHGNet/MPtrj, mace-foundations).

---

## Notes

- `vasprun-*.xml` is a **minimal** VASP-like XML that Phonopy can parse for forces/energies.  
- `--fix-axis` accepts any subset of `{a,b,c}` (e.g., `a`, `b,c`) when using cell filters (`ISIF≥3`).  
- For batch runs, the script writes **per-input** outputs with the input file name appended.

---
