#!/usr/bin/env python3
"""
Phase 1 Experiment: Dataset Statistics & Audit.
Parses representative and aggregate netlists across all 8 base designs,
computes nodes, edges, gates, and Trojan gate counts, and writes
results/p1_dataset_stats.json.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import glob
import json
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from collections import defaultdict
from neuroguard.parsing.netlist_parser import NetlistParser

BASE_DESIGNS = ["c2670", "c3540", "c5315", "c6288", "s13207", "s1423", "s15850", "s35932"]

def extract_base_design(name: str) -> str:
    for bd in BASE_DESIGNS:
        if name.startswith(bd):
            return bd
    return "other"

def main():
    parser = NetlistParser()
    tc_dirs = glob.glob("data/TRIT-TC/*")
    ts_dirs = [d for d in glob.glob("data/TRIT-TS/*") if Path(d).name != "original_designs"]
    all_infected = tc_dirs + ts_dirs

    stats_by_family = defaultdict(lambda: {
        "netlist_count": 0,
        "total_nodes": 0,
        "total_edges": 0,
        "total_gates": 0,
        "total_trojan_gates": 0,
        "trojan_types": set(),
    })

    print(f"Scanning {len(all_infected)} netlists...")
    for d in all_infected:
        path = Path(d)
        name = path.name
        base = extract_base_design(name)
        v_file = path / f"{name}.v"
        log_file = path / "log.txt"

        if not v_file.exists():
            continue

        graph = parser.parse(v_file, log_file)
        family_stat = stats_by_family[base]
        family_stat["netlist_count"] += 1
        family_stat["total_nodes"] += graph.number_of_nodes()
        family_stat["total_edges"] += graph.number_of_edges()
        family_stat["total_gates"] += graph.graph.get("num_gates", 0)
        family_stat["total_trojan_gates"] += graph.graph.get("trojan_gate_count", 0)
        
        # Trojan type from log
        if log_file.exists():
            with open(log_file, "r", errors="ignore") as f:
                for line in f:
                    if "Type:" in line:
                        family_stat["trojan_types"].add(line.split("Type:")[1].strip())

    # Format output
    summary = {}
    for base in sorted(stats_by_family.keys()):
        s = stats_by_family[base]
        n = max(1, s["netlist_count"])
        summary[base] = {
            "netlists": s["netlist_count"],
            "avg_nodes": round(s["total_nodes"] / n, 1),
            "avg_edges": round(s["total_edges"] / n, 1),
            "avg_gates": round(s["total_gates"] / n, 1),
            "total_trojan_gates": s["total_trojan_gates"],
            "avg_trojan_gates": round(s["total_trojan_gates"] / n, 2),
            "trojan_types": sorted(list(s["trojan_types"])),
        }

    os.makedirs("results", exist_ok=True)
    with open("results/p1_dataset_stats.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\nDataset Statistics Summary:")
    print(f"{'Base Design':<12} | {'Netlists':<8} | {'Avg Gates':<10} | {'Avg Edges':<10} | {'Troj Gates':<10} | {'Trojan Types'}")
    print("-" * 80)
    for base, data in summary.items():
        print(f"{base:<12} | {data['netlists']:<8} | {data['avg_gates']:<10} | {data['avg_edges']:<10} | {data['total_trojan_gates']:<10} | {', '.join(data['trojan_types'])}")

if __name__ == "__main__":
    main()
