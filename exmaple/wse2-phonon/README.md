# Example
### 1. Genereate finte displacements using phonopy
```
phonopy -d --dim="3 3 2" -c POSCAR-primitive
```
### 2. Run MACE with isif=0 (single point) for POSCAR-001, POSCAR-002, and POSCAR-003

```
python mace_ase_relax.py -i POSCAR-0* --isif 0
```
### 3. Generate FORCE_SETS
```
phonopy -f vasprun-POSCAR-00*
```

### 4. Calculate phonon dispersion
* band.conf 
```
DIM = 3 3 2

# Band structure calculation
BAND = 0.0000 0.0000 0.0000  0.0000 0.5000 0.0000 0.5000 0.5000 0.0000 0.0000 0.0000 0.0000

BAND_POINTS = 51
BAND_LABELS = Γ X M Γ
BAND_CONNECTION = TRUE
EIGENVECTORS=.true.
```
* The dispersion saved as a pdf file.
```
phonopy -p band.conf -s
```
