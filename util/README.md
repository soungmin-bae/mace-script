# seekpath2bandconf.py

Generate a **phonopy `band.conf`** directly from a **POSCAR** using **SeeK‑path** (no VASPKIT needed).  
It also reads `phonopy_disp.yaml` to set the `DIM = a b c`, formats the **BAND** line as a single line of q‑points (junctions deduped), and prints a concise summary (POSCAR used, space group, and the q‑path).

---

## Features

- **POSCAR** parser (VASP4/5, `Selective dynamics`, `Direct/Cartesian`)
- Calls **SeeK‑path** to standardize the cell and generate a **continuous** k‑path
- Reads **`phonopy_disp.yaml`** to determine `DIM`
- **BAND** is written on **one line** (one k‑point per node, no duplicates)
- **BAND_LABELS** cleaned (`GAMMA → GM` by default; underscores removed)
- Optional **ATOM_NAME** override (`--atom-names` or `--rename`)
- Terminal **summary**: POSCAR path, space group (symbol & number), Bravais, q‑path & coordinates
- Configurable **symmetry tolerance** via `--symprec` (default: `1e-5`)

---

## Requirements
- Packages:
  - `seekpath`
  - `spglib`

##  Usage

```
python seekpath2bandconf.py --poscar POSCAR --yaml phonopy_disp.yaml --out band.conf
```

This creates `band.conf` (example):
```ini
ATOM_NAME = K Zr P O       ; or blank if POSCAR (VASP4) has no element line
DIM = 3 3 3
BAND = 0.000 0.000 0.000    0.500 0.500 0.500    ...    0.000 0.000 0.000
BAND_LABELS = GM T H2 H0 L GM S0 S2 F GM
FORCE_SETS = READ
FC_SYMMETRY = .TRUE.
EIGENVECTORS = .TRUE.
```

Then plot with phonopy:
```bash
phonopy -p band.conf
```
---

## Commandline options

```
--poscar POSCAR           Path to POSCAR (default: POSCAR)
--yaml phonopy_disp.yaml  Path to phonopy_disp.yaml (default: phonopy_disp.yaml)
--out band.conf           Output band.conf (default: band.conf)

--gamma GM                Label used for Γ (default: "GM", e.g., use "Γ" for the symbol)
--symprec 1e-5            Symmetry tolerance passed to SeeK-path (default: 1e-5)

--atom-names "K Zr P O"   Override ATOM_NAME line explicitly
--rename "Na=K,Zr=Zr"     Rename mapping applied to symbols read from POSCAR
--no-defaults             Do not append FORCE_SETS/FC_SYMMETRY/EIGENVECTORS lines
```

**Examples**
```bash
# Default usage
python seekpath2bandconf.py

# Use Γ symbol and tighter symmetry tolerance
python seekpath2bandconf.py --gamma "Γ" --symprec 1e-6

# Override ATOM_NAME explicitly
python seekpath2bandconf.py --atom-names "K Zr P O"

# Rename only Na → K using names present in POSCAR
python seekpath2bandconf.py --rename "Na=K"
```

---


## Sample Output (terminal)

```
[OK] Wrote band.conf
------------------------------------------------------------
[Seekpath] POSCAR: /path/to/POSCAR
[Seekpath] Space group: R-3c (No.167), Bravais: rhombohedral
[Seekpath] Q-path (labels): GM - T - H2 - H0 - L - GM - S0 - S2 - F - GM
[Seekpath] Q-points (reciprocal crystal units):
   GM : 0.000  0.000  0.000
    T : 0.500  0.500  0.500
   H2 : 0.759  0.241  0.500
   H0 : 0.500 -0.241  0.241
    L : 0.500  0.000  0.000
   GM : 0.000  0.000  0.000
   S0 : 0.370 -0.370  0.000
   S2 : 0.630  0.000  0.370
    F : 0.500  0.000  0.500
   GM : 0.000  0.000  0.000
[Seekpath] Total q-points: 10
------------------------------------------------------------
```

---

