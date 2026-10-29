"""Cell library utilities for NeuroGuard-X.

Maps cell names to canonical descriptions:
    (function_family, n_inputs, drive_strength, is_sequential, output_ports)
Supports both TRIT vendor standard cells and positional primitives (ISCAS).
"""

import os
import re
import json
import glob
from collections import namedtuple
from typing import Set, Dict, Tuple, List

CellInfo = namedtuple("CellInfo", ["function_family", "n_inputs", "drive_strength", "is_sequential", "output_ports"])

# 12 canonical function families:
CANONICAL_FAMILIES = [
    "AND", "AOI", "BUF", "DFF", "INV", "MUX", "NAND", "NOR", "OAI", "OR", "XNOR", "XOR"
]

POS_PRIMITIVES = {
    "and": ("AND", 2, 1, False, ("Q",)),
    "nand": ("NAND", 2, 1, False, ("Q",)),
    "or": ("OR", 2, 1, False, ("Q",)),
    "nor": ("NOR", 2, 1, False, ("Q",)),
    "xor": ("XOR", 2, 1, False, ("Q",)),
    "xnor": ("XNOR", 2, 1, False, ("Q",)),
    "not": ("INV", 1, 1, False, ("Q",)),
    "buf": ("BUF", 1, 1, False, ("Q",)),
    "dff": ("DFF", 1, 1, True, ("Q", "QN")),
}

def classify_cell(name: str) -> CellInfo:
    """Map a cell name to its CellInfo.
    Raises ValueError on any unmapped cell.
    """
    lower_name = name.lower()
    if lower_name in POS_PRIMITIVES:
        f, n, d, seq, out = POS_PRIMITIVES[lower_name]
        return CellInfo(f, n, d, seq, out)

    # DFF cells: sdffs1, dffs2, dffcs2, dffles2
    m = re.match(r"^(s?dff[a-z]*)s(\d+)$", name)
    if m:
        drive = int(m.group(2))
        return CellInfo("DFF", 1, drive, True, ("Q", "QN"))

    # Inverters: i1s*, hi1s*, ib1s*
    m = re.match(r"^(h?i1|ib1)s(\d+)$", name)
    if m:
        drive = int(m.group(2))
        return CellInfo("INV", 1, drive, False, ("Q",))

    # Buffers: nb1s*, bufs*
    m = re.match(r"^(nb1|buf)s(\d+)$", name)
    if m:
        drive = int(m.group(2))
        return CellInfo("BUF", 1, drive, False, ("Q",))

    # NAND: nnd<N>s<D>
    m = re.match(r"^nnd(\d+)s(\d+)$", name)
    if m:
        n_in = int(m.group(1))
        drive = int(m.group(2))
        return CellInfo("NAND", n_in, drive, False, ("Q",))

    # NOR: nor<N>s<D>
    m = re.match(r"^nor(\d+)s(\d+)$", name)
    if m:
        n_in = int(m.group(1))
        drive = int(m.group(2))
        return CellInfo("NOR", n_in, drive, False, ("Q",))

    # AND: and<N>s<D>
    m = re.match(r"^and(\d+)s(\d+)$", name)
    if m:
        n_in = int(m.group(1))
        drive = int(m.group(2))
        return CellInfo("AND", n_in, drive, False, ("Q",))

    # OR: or<N>s<D>
    m = re.match(r"^or(\d+)s(\d+)$", name)
    if m:
        n_in = int(m.group(1))
        drive = int(m.group(2))
        return CellInfo("OR", n_in, drive, False, ("Q",))

    # XOR: xor<N>s<D>
    m = re.match(r"^xor(\d+)s(\d+)$", name)
    if m:
        n_in = int(m.group(1))
        drive = int(m.group(2))
        return CellInfo("XOR", n_in, drive, False, ("Q",))

    # XNOR: xnr<N>s<D>
    m = re.match(r"^xnr(\d+)s(\d+)$", name)
    if m:
        n_in = int(m.group(1))
        drive = int(m.group(2))
        return CellInfo("XNOR", n_in, drive, False, ("Q",))

    # MUX: mx<N>s<D>, mxi<N>s<D>, dsmxc<N>s<D>
    m = re.match(r"^(mxi?|dsmxc)(\d+)s(\d+)$", name)
    if m:
        code = m.group(2)
        drive = int(m.group(3))
        n_in = sum(int(c) for c in code)
        return CellInfo("MUX", n_in, drive, False, ("Q",))

    # AOI: aoi<digits>s<D>, aoai<digits>s<D>
    m = re.match(r"^(aoi|aoai)(\d+)s(\d+)$", name)
    if m:
        digits = m.group(2)
        n_in = sum(int(d) for d in digits)
        drive = int(m.group(3))
        return CellInfo("AOI", n_in, drive, False, ("Q",))

    # OAI: oai<digits>s<D>
    m = re.match(r"^oai(\d+)s(\d+)$", name)
    if m:
        digits = m.group(1)
        n_in = sum(int(d) for d in digits)
        drive = int(m.group(2))
        return CellInfo("OAI", n_in, drive, False, ("Q",))

    raise ValueError(f"Unmapped cell name: {name}")


def find_dataset_netlists(data_root: str) -> List[str]:
    """Collect all gate-level netlists in TRIT-TC, TRIT-TS, and clean ISCAS in TjFree."""
    netlists = []
    # TRIT-TC
    netlists.extend(glob.glob(os.path.join(data_root, "TRIT-TC", "*", "*.v")))
    # TRIT-TS
    netlists.extend(glob.glob(os.path.join(data_root, "TRIT-TS", "*", "*.v")))
    # TRIT-TS original_designs
    netlists.extend(glob.glob(os.path.join(data_root, "TRIT-TS", "original_designs", "*.v")))
    # TjFree gate-level ISCAS files
    iscas_names = [
        "c17", "c432", "c499", "c880", "c1355", "c1908", "c2670",
        "c3540", "c5315", "c6288", "c7552", "s13207", "s1423", "s15850", "s35932"
    ]
    for iscas in iscas_names:
        p = os.path.join(data_root, "TjFree", iscas, f"{iscas}.v")
        if os.path.exists(p):
            netlists.append(p)
    return sorted(list(set(netlists)))


def list_cells(data_root: str) -> Set[str]:
    """Scan all target gate-level netlists and collect all unique cell types."""
    netlists = find_dataset_netlists(data_root)
    # Match pattern: cell_name instance_name (
    pat = re.compile(r"^\s*([a-zA-Z0-9_]+)\s+(\\\S+|\S+)\s*\(")
    keywords = {"module", "input", "output", "wire", "reg", "assign", "endmodule", "always", "initial"}

    cells: Set[str] = set()
    for path in netlists:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = pat.match(line)
                if m:
                    cname = m.group(1)
                    if cname not in keywords:
                        cells.add(cname)
    return cells


def build_library(data_root: str, out_path: str) -> Dict[str, dict]:
    """Scan netlists, classify all cells, write results/cell_inventory.json, and return dict."""
    cells = list_cells(data_root)
    library: Dict[str, dict] = {}
    unmapped = []
    for c in sorted(cells):
        try:
            info = classify_cell(c)
            library[c] = info._asdict()
        except ValueError:
            unmapped.append(c)

    if unmapped:
        raise RuntimeError(f"Unmapped cells encountered ({len(unmapped)}): {unmapped}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(library, fp, indent=2, sort_keys=True)
    return library


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate cell inventory JSON.")
    parser.add_argument("--data-root", default="data", help="Root folder of the netlists.")
    parser.add_argument("--output", default="results/cell_inventory.json", help="Output JSON path.")
    args = parser.parse_args()
    lib = build_library(args.data_root, args.output)
    print(f"Successfully mapped {len(lib)} distinct cells into {args.output}")
