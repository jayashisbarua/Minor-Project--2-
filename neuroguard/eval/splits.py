"""Evaluation splits for NeuroGuard-X.

Guarantees:
1. LOFO (Leave-One-Family-Out): 8 outer folds. All netlists of a base design (across TRIT-TC,
   TRIT-TS, and clean baselines) remain strictly in the test fold.
2. Validation split is carved strictly from training families (never touching test families).
3. Leave-one-Trojan-type-out: Train on combinational, test on sequential FSM/counter.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np

BASE_DESIGNS = ["c2670", "c3540", "c5315", "c6288", "s13207", "s1423", "s15850", "s35932"]


def get_all_cached_netlists(cache_dir: str = "cache") -> List[Path]:
    """Return all cached PyG Data filepaths."""
    return sorted(list(Path(cache_dir).glob("*/*.pt")))


def get_base_design(path: Path | str) -> str:
    """Extract canonical base design name."""
    name = Path(path).stem
    for bd in BASE_DESIGNS:
        if name.startswith(bd):
            return bd
    return "other"


def create_lofo_splits(
    netlist_paths: Optional[List[Path]] = None,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, List[Path]]]:
    """Generate 8 Leave-One-Family-Out folds.

    Returns
    -------
    Dict[fold_name, {'train': [...], 'val': [...], 'test': [...]}]
    """
    if netlist_paths is None:
        netlist_paths = get_all_cached_netlists()

    # Group netlists by base design
    by_family: Dict[str, List[Path]] = {bd: [] for bd in BASE_DESIGNS}
    for p in netlist_paths:
        bd = get_base_design(p)
        if bd in by_family:
            by_family[bd].append(p)

    rng = np.random.default_rng(seed)
    folds: Dict[str, Dict[str, List[Path]]] = {}

    for test_family in BASE_DESIGNS:
        test_netlists = list(by_family[test_family])
        train_families = [f for f in BASE_DESIGNS if f != test_family]

        train_pool: List[Path] = []
        for tf in train_families:
            train_pool.extend(by_family[tf])

        # Hold out one entire training family as validation to prevent near-duplicate leakage
        val_family_idx = (BASE_DESIGNS.index(test_family) + 1) % len(train_families)
        val_family = train_families[val_family_idx]
        val_netlists = list(by_family[val_family])

        train_netlists = [p for p in train_pool if get_base_design(p) != val_family]

        # Deterministic shuffle
        rng.shuffle(train_netlists)
        rng.shuffle(val_netlists)
        rng.shuffle(test_netlists)

        folds[test_family] = {
            "train": train_netlists,
            "val": val_netlists,
            "test": test_netlists,
            "test_family": test_family,
            "val_family": val_family,
        }

    return folds


def create_trojan_type_splits(
    netlist_paths: Optional[List[Path]] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, List[Path]]]:
    """Create leave-one-Trojan-type splits:
    - Fold 'test_sequential': Train on combinational (TRIT-TC), Test on sequential (TRIT-TS)
    - Fold 'test_combinational': Train on sequential (TRIT-TS), Test on combinational (TRIT-TC)
    """
    if netlist_paths is None:
        netlist_paths = get_all_cached_netlists()

    combinational: List[Path] = []
    sequential: List[Path] = []

    for p in netlist_paths:
        bd = get_base_design(p)
        # Combinational families: c2670, c3540, c5315, c6288
        if bd.startswith("c"):
            combinational.append(p)
        elif bd.startswith("s"):
            sequential.append(p)

    rng = np.random.default_rng(seed)
    rng.shuffle(combinational)
    rng.shuffle(sequential)

    # Carve validation
    val_c_count = max(1, int(len(combinational) * 0.15))
    val_s_count = max(1, int(len(sequential) * 0.15))

    return {
        "test_sequential": {
            "train": combinational[val_c_count:],
            "val": combinational[:val_c_count],
            "test": sequential,
        },
        "test_combinational": {
            "train": sequential[val_s_count:],
            "val": sequential[:val_s_count],
            "test": combinational,
        },
    }

