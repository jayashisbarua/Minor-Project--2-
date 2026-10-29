#!/usr/bin/env python3
"""
Phase 2 Experiment: Multi-View Graph Construction & Caching.
Converts all 914 infected netlists and clean designs into PyG Data objects,
validates node alignment across structural and functional views,
and caches them to cache/<family>/<name>.pt.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import glob
import json
from pathlib import Path
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from neuroguard.graphs.build_views import build_multiview_graph, extract_base_design
from neuroguard.parsing.netlist_parser import NetlistParser

def main():
    parser = NetlistParser()
    tc_dirs = glob.glob("data/TRIT-TC/*")
    ts_dirs = [d for d in glob.glob("data/TRIT-TS/*") if Path(d).name != "original_designs"]
    all_dirs = tc_dirs + ts_dirs

    # Add clean original designs (sequential gate-level ISCAS-89 from TRIT-TS)
    clean_designs = glob.glob("data/TRIT-TS/original_designs/*.v")
    # Add clean TjFree gate-level ISCAS-85 files
    iscas_85 = [
        "c17", "c432", "c499", "c880", "c1355", "c1908", "c2670",
        "c3540", "c5315", "c6288", "c7552"
    ]
    for iscas in iscas_85:
        p = f"data/TjFree/{iscas}/{iscas}.v"
        if os.path.exists(p):
            clean_designs.append(p)

    total_converted = 0
    total_errors = 0
    stats_by_family = {}

    print(f"Building and caching multi-view graphs for {len(all_dirs)} infected netlists...")
    for d in tqdm(all_dirs, desc="Infected netlists"):
        path = Path(d)
        name = path.name
        v_file = path / f"{name}.v"
        log_file = path / "log.txt"
        if not v_file.exists():
            continue

        try:
            base = extract_base_design(name)
            cache_dir = Path("cache") / base
            cache_path = cache_dir / f"{name}.pt"
            if cache_path.exists() and "--overwrite" not in sys.argv:
                data = torch.load(cache_path, weights_only=False)
            else:
                data = build_multiview_graph(v_file, log_file, parser=parser)
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(data, cache_path)

            family = data.family

            if family not in stats_by_family:
                stats_by_family[family] = {"count": 0, "total_nodes": 0, "total_struct_edges": 0, "total_func_edges": 0}
            stats_by_family[family]["count"] += 1
            stats_by_family[family]["total_nodes"] += data.num_nodes
            stats_by_family[family]["total_struct_edges"] += data.edge_index_struct.shape[1]
            stats_by_family[family]["total_func_edges"] += data.edge_index_func.shape[1]
            total_converted += 1
        except Exception as e:
            print(f"Error converting {name}: {e}")
            total_errors += 1

    print(f"Caching clean baseline designs ({len(clean_designs)} files)...")
    for v_str in tqdm(clean_designs, desc="Clean designs"):
        v_path = Path(v_str)
        name = v_path.stem
        try:
            data = build_multiview_graph(v_path, parser=parser)
            family = data.family
            cache_dir = Path("cache") / family
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{name}.pt"
            torch.save(data, cache_path)
            total_converted += 1
        except Exception as e:
            print(f"Error converting clean {name}: {e}")
            total_errors += 1

    summary = {
        "total_converted": total_converted,
        "total_errors": total_errors,
        "families": stats_by_family,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/p2_graph_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\nPhase 2 Complete: Successfully converted and validated {total_converted} graphs (errors: {total_errors}).")
    assert total_errors == 0, f"Encountered {total_errors} errors during graph conversion"

if __name__ == "__main__":
    main()
