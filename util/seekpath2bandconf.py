#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seekpath2bandconf.py
Make phonopy band.conf directly from POSCAR using SeeK-path (no VASPKIT needed).

- Robust POSCAR parser (VASP4/5, Selective dynamics, Direct/Cartesian)
- POSCAR -> SeeK-path standard k-path (continuous chain)
- DIM: from --dim "a b c" (highest priority) or phonopy_disp.yaml if present; otherwise blank
- BAND: single line, one k-point per node (junctions deduped)
- BAND_LABELS: cleaned (GAMMA->GM by default, underscores removed)
- ATOM_NAME override: --atom-names "K Zr P O" or --rename "Na=K"
- Summary print: POSCAR used, space group, q-path & coords, DIM source

Usage
  pip install seekpath spglib
  python seekpath2bandconf.py --poscar POSCAR --yaml phonopy_disp.yaml --out band.conf
  # optional:
  #   --atom-names "K Zr P O"
  #   --rename "Na=K"
  #   --gamma "Γ"
  #   --symprec 1e-5
  #   --dim "3 3 3"
  #   --no-defaults
"""

import argparse
import re
from pathlib import Path

# (optional) silence DeprecationWarnings from spglib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ------------------------- POSCAR I/O -------------------------

def read_poscar(poscar_path: Path):
    with poscar_path.open("r", encoding="utf-8") as f:
        raw = [ln.rstrip("\n") for ln in f]

    def next_nonempty(i):
        while i < len(raw) and raw[i].strip() == "":
            i += 1
        return i

    i = next_nonempty(0)
    if i >= len(raw):
        raise ValueError("POSCAR is empty")

    comment = raw[i]; i = next_nonempty(i+1)
    if i >= len(raw):
        raise ValueError("POSCAR: missing scale line")
    scale = float(raw[i].split()[0]); i = next_nonempty(i+1)

    # lattice vectors
    lat = []
    for _ in range(3):
        if i >= len(raw):
            raise ValueError("POSCAR: missing lattice vectors")
        parts = raw[i].split()
        if len(parts) < 3:
            raise ValueError("POSCAR: lattice vector line has < 3 numbers")
        vec = [float(x) for x in parts[:3]]
        lat.append([scale * v for v in vec])
        i = next_nonempty(i+1)

    if i >= len(raw):
        raise ValueError("POSCAR: missing symbols/counts line")

    # Detect VASP5 (symbols line) vs VASP4
    tokens = raw[i].split()

    def is_number(x):
        try:
            float(x); return True
        except Exception:
            return False

    vasp5 = (tokens and any(not is_number(t) for t in tokens))
    if vasp5:
        symbols = tokens[:]                   # e.g. Na Zr P O
        i = next_nonempty(i+1)
        if i >= len(raw):
            raise ValueError("POSCAR: missing counts line after symbols")
        counts = [int(x) for x in raw[i].split()]
        i = next_nonempty(i+1)
        if len(symbols) != len(counts):
            if len(symbols) > len(counts):
                symbols = symbols[:len(counts)]
            else:
                symbols = symbols + [f"E{j+1}" for j in range(len(counts)-len(symbols))]
    else:
        # VASP4: counts here, no symbols line
        counts = [int(x) for x in tokens]
        symbols = []  # no symbols; ATOM_NAME will be blank unless overridden
        i = next_nonempty(i+1)

    # Optional "Selective dynamics"
    if i < len(raw) and raw[i].strip().lower().startswith("selective"):
        i = next_nonempty(i+1)

    # Coordinate type
    if i >= len(raw):
        raise ValueError("POSCAR: missing coordinate type line")
    ctok = raw[i].strip().lower()
    direct = ctok.startswith("d")
    cart = ctok.startswith("c")
    if not (direct or cart):
        raise ValueError(f"POSCAR: unknown coordinate type line: {raw[i]}")
    i = next_nonempty(i+1)

    # Atom coordinates
    nat = sum(counts)
    coord_lines, read = [], 0
    while i < len(raw) and read < nat:
        if raw[i].strip() != "":
            coord_lines.append(raw[i])
            read += 1
        i += 1
    if read < nat:
        raise ValueError("POSCAR: not enough atomic coordinate lines")

    if direct:
        frac = []
        for ln in coord_lines:
            parts = ln.split()
            x, y, z = [float(u) for u in parts[:3]]
            frac.append([x, y, z])
    else:
        import numpy as np
        A = np.array(lat).T  # columns a,b,c
        Ainv = np.linalg.inv(A)
        frac = []
        for ln in coord_lines:
            parts = ln.split()
            cx, cy, cz = [float(u) for u in parts[:3]]
            f = Ainv @ (scale * np.array([cx, cy, cz]))
            frac.append(f.tolist())

    # kinds: build from counts
    kinds = []
    nsp = len(counts)
    for si in range(nsp):
        kinds.extend([si + 1] * counts[si])

    return {
        "comment": comment,
        "lattice": lat,
        "frac": frac,
        "kinds": kinds,
        "symbols": symbols,  # may be []
        "counts": counts,
    }


# ------------------------- phonopy_disp.yaml -> DIM -------------------------

def read_dim_from_yaml(yaml_path: Path):
    """Return [a,b,c] or None if not found or file missing."""
    try:
        return _read_dim_yaml_inner(yaml_path)
    except FileNotFoundError:
        return None

def _read_dim_yaml_inner(yaml_path: Path):
    dim = None
    pat_dim = re.compile(r'^\s*dim:\s*"([^"]+)"\s*$', re.IGNORECASE)
    pat_row = re.compile(r'^\s*-\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*$')

    with yaml_path.open("r", encoding="utf-8") as f:
        for ln in f:
            m = pat_dim.match(ln)
            if m:
                parts = m.group(1).split()
                if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
                    dim = [int(p) for p in parts]
                break
    if dim is not None:
        return dim

    rows, in_super = [], False
    with yaml_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if "supercell_matrix" in ln:
                in_super = True
                continue
            if in_super:
                if ln.strip().startswith("- ["):
                    m = pat_row.match(ln)
                    if m:
                        rows.append([int(m.group(i)) for i in range(1, 4)])
                else:
                    break

    if rows:
        if all(rows[i][j] == (rows[i][i] if i == j else 0) for i in range(3) for j in range(3)):
            dim = [abs(rows[i][i]) for i in range(3)]
        else:
            import math
            dim = [max(1, int(round(math.sqrt(sum(c*c for c in r))))) for r in rows[:3]]
            print("[WARN] Non-diagonal supercell_matrix; approximated DIM =", dim)

    return dim  # may be None


# ------------------------- helpers -------------------------

def _fmt(x):
    v = float(x)
    if abs(v) < 1e-12:
        v = 0.0
    s = f"{v:.3f}"
    if s == "-0.000":
        s = "0.000"
    return s

def _clean_label(lbl: str, gamma="GM"):
    if lbl.upper() == "GAMMA":
        return gamma
    return lbl.replace("_", "")


# ------------------------- label chain & band -------------------------

def build_label_chain(path_segments):
    if not path_segments:
        return []
    chain = []
    s0, e0 = path_segments[0]
    chain.append(s0); chain.append(e0)
    for (s, e) in path_segments[1:]:
        if chain[-1] != s:
            chain.append(s)
        chain.append(e)
    dedup = [chain[0]]
    for lab in chain[1:]:
        if lab != dedup[-1]:
            dedup.append(lab)
    return dedup

def band_points_one_line_from_seekpath(path_data, gamma_label="GM"):
    pc = path_data["point_coords"]
    segs = path_data["path"]
    chain = build_label_chain(segs)
    labels = [_clean_label(x, gamma_label) for x in chain]
    pts = [f"{_fmt(pc[lab][0])} {_fmt(pc[lab][1])} {_fmt(pc[lab][2])}" for lab in chain]
    band_line = "BAND = " + "    ".join(pts)
    return band_line, labels, chain  # chain: raw labels for summary


# ------------------------- atom-name override -------------------------

def parse_atom_override(args, symbols_from_poscar):
    if args.atom_names:
        return args.atom_names.split()
    if args.rename:
        ren = {}
        for pair in args.rename.split(","):
            old, new = pair.split("=")
            ren[old.strip()] = new.strip()
        if symbols_from_poscar:
            return [ren.get(s, s) for s in symbols_from_poscar]
        else:
            return []
    return symbols_from_poscar


# ------------------------- pretty summary -------------------------

def print_summary(poscar_path: Path, path_data, chain_labels, gamma_cleaned_labels, dim, dim_source):
    try:
        poscar_disp = str(poscar_path.resolve())
    except Exception:
        poscar_disp = str(poscar_path)

    sg_int = path_data.get("spacegroup_international", "?")
    sg_no  = path_data.get("spacegroup_number", "?")
    bravais = path_data.get("bravais_lattice", "?")

    print("------------------------------------------------------------")
    print(f"[Seekpath] POSCAR: {poscar_disp}")
    print(f"[Seekpath] Space group: {sg_int} (No.{sg_no}), Bravais: {bravais}")
    print(f"[Seekpath] Q-path (labels): {' - '.join(gamma_cleaned_labels)}")
    print("[Seekpath] Q-points (reciprocal crystal units):")
    pc = path_data["point_coords"]
    for raw_lab, clean_lab in zip(chain_labels, gamma_cleaned_labels):
        k = pc[raw_lab]
        print(f"  {clean_lab:>4s} : {_fmt(k[0])}  {_fmt(k[1])}  {_fmt(k[2])}")
    print(f"[Seekpath] Total q-points: {len(chain_labels)}")

    if dim is None:
        print("[Info] DIM not set (no phonopy_disp.yaml). The output band.conf will contain 'DIM =' (blank).")
        print("       Provide --yaml phonopy_disp.yaml or use --dim \"a b c\" to set it explicitly.")
    else:
        print(f"[Seekpath] DIM = {dim[0]} {dim[1]} {dim[2]}  (source: {dim_source})")
    print("------------------------------------------------------------")


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Make phonopy band.conf via SeeK-path (one-line BAND)")
    ap.add_argument("--poscar", default="POSCAR", type=Path)
    ap.add_argument("--yaml", default="phonopy_disp.yaml", type=Path)
    ap.add_argument("--out", default="band.conf", type=Path)
    ap.add_argument("--gamma", default="GM", help="Gamma label for BAND_LABELS (e.g., GM or Γ)")
    ap.add_argument("--symprec", type=float, default=1e-5, help="Symmetry tolerance passed to SeeK-path (default: 1e-5)")
    ap.add_argument("--dim", default=None, help='Override DIM as a string "a b c" (e.g., "3 3 3")')
    ap.add_argument("--no-defaults", action="store_true")
    ap.add_argument("--atom-names", default=None, help='Override ATOM_NAME, e.g. "K Zr P O"')
    ap.add_argument("--rename", default=None, help='Rename mapping, e.g. "Na=K,Zr=Zr"')
    args = ap.parse_args()

    # POSCAR -> cell
    pos = read_poscar(args.poscar)
    atom_names = parse_atom_override(args, pos["symbols"])
    lattice, positions, numbers = pos["lattice"], pos["frac"], pos["kinds"]
    cell = (lattice, positions, numbers)

    # SeeK-path (with symprec)
    import seekpath
    path_data = seekpath.get_path(cell, symprec=args.symprec)

    band_line, labels, chain_raw = band_points_one_line_from_seekpath(path_data, gamma_label=args.gamma)

    # DIM priority: --dim > --yaml (if exists) > None (blank)
    dim = None
    dim_source = None
    if args.dim:
        parts = args.dim.split()
        if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
            dim = [int(p) for p in parts]
            dim_source = "--dim"
        else:
            print("[WARN] --dim must be like: --dim \"3 3 3\"; ignoring it.")
    if dim is None:
        if args.yaml and Path(args.yaml).exists():
            dim = read_dim_from_yaml(Path(args.yaml))
            dim_source = "phonopy_disp.yaml" if dim is not None else None
        else:
            dim = None

    # Write band.conf
    out_lines = []
    if atom_names:
        out_lines.append(f"ATOM_NAME = {' '.join(atom_names)}")
    else:
        out_lines.append("ATOM_NAME =")
    if dim is None:
        out_lines.append("DIM =")  # leave blank as requested
    else:
        out_lines.append(f"DIM = {dim[0]} {dim[1]} {dim[2]}")
    out_lines.append(band_line)
    out_lines.append(f"BAND_LABELS = {' '.join(labels)}")
    if not args.no_defaults:
        out_lines.append("FORCE_SETS = READ")
        out_lines.append("FC_SYMMETRY = .TRUE.")
        out_lines.append("EIGENVECTORS = .TRUE.")

    Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {args.out}")

    # Summary
    print_summary(args.poscar, path_data, chain_raw, labels, dim, dim_source)


if __name__ == "__main__":
    main()
