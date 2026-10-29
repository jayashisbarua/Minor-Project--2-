#!/usr/bin/env python3
"""
Phase 3 Experiment: Logic Simulation for Temporal Features.
Evaluates gate-level bit-parallel logic simulation across designs,
computes runtime budgets, measures rare_score separation for Trojan triggers,
and caches temporal results in results/temporal_cache/<family>/<name>.json.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import json
import random
import glob
from pathlib import Path
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from neuroguard.parsing.netlist_parser import NetlistParser
from neuroguard.graphs.simulate import BitParallelSimulator
from neuroguard.graphs.build_views import extract_base_design

def main():
    parser = NetlistParser()
    sim = BitParallelSimulator(num_cycles=500, seed=42) # 32,000 vectors

    # 1. Runtime benchmark on clean s13207scan
    clean_path = Path("data/TRIT-TS/original_designs/s13207scan.v")
    assert clean_path.exists(), "Clean design s13207scan.v not found"
    G_clean = parser.parse(clean_path)
    t0 = time.time()
    clean_stats = sim.simulate(G_clean)
    clean_runtime = time.time() - t0

    # 2. Measure rare_score and p1 separation across a representative sample of infected netlists
    tc_dirs = glob.glob("data/TRIT-TC/*")
    ts_dirs = [d for d in glob.glob("data/TRIT-TS/*") if Path(d).name != "original_designs"]
    all_dirs = tc_dirs + ts_dirs

    random.seed(42)
    sample_dirs = random.sample(all_dirs, min(40, len(all_dirs)))

    all_trigger_rares = []
    all_other_rares = []
    all_payload_rares = []
    sample_runtimes = []

    temporal_cache_dir = Path("results/temporal_cache")
    temporal_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {len(sample_dirs)} representative netlists...")
    for d in sample_dirs:
        path = Path(d)
        name = path.name
        v_file = path / f"{name}.v"
        log_file = path / "log.txt"
        if not v_file.exists():
            continue

        base = extract_base_design(name)
        cache_file = temporal_cache_dir / f"{name}_temporal.json"

        G = parser.parse(v_file, log_file)
        t_start = time.time()
        stats = sim.simulate(G)
        t_sim = time.time() - t_start
        sample_runtimes.append(t_sim)

        # Cache simulation results
        with open(cache_file, "w") as fp:
            json.dump(stats, fp)

        for n, data in G.nodes(data=True):
            if data.get("type") == "GATE":
                role = data.get("role", 0)
                rare = stats[n]["rare_score"]
                if role == 1:
                    all_trigger_rares.append(rare)
                elif role == 3:
                    all_payload_rares.append(rare)
                else:
                    all_other_rares.append(rare)

    summary = {
        "clean_benchmark_design": "s13207scan.v",
        "clean_nodes": G_clean.number_of_nodes(),
        "clean_edges": G_clean.number_of_edges(),
        "clean_runtime_seconds": round(clean_runtime, 2),
        "avg_sample_runtime_seconds": round(float(np.mean(sample_runtimes)), 2),
        "sample_netlists_count": len(sample_dirs),
        "trigger_gates_count": len(all_trigger_rares),
        "trigger_rare_score_mean": round(float(np.mean(all_trigger_rares)), 4),
        "trigger_rare_score_median": round(float(np.median(all_trigger_rares)), 4),
        "other_gates_count": len(all_other_rares),
        "other_rare_score_mean": round(float(np.mean(all_other_rares)), 4),
        "other_rare_score_median": round(float(np.median(all_other_rares)), 4),
    }

    with open("results/p3_simulation_results.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\nPhase 3 Simulation Results:")
    print(f"Clean Benchmark ({summary['clean_benchmark_design']}): {summary['clean_runtime_seconds']}s")
    print(f"Trigger Gates (N={summary['trigger_gates_count']}): Mean Rare Score = {summary['trigger_rare_score_mean']}, Median = {summary['trigger_rare_score_median']}")
    print(f"Other Gates (N={summary['other_gates_count']}): Mean Rare Score = {summary['other_rare_score_mean']}, Median = {summary['other_rare_score_median']}")

if __name__ == "__main__":
    main()

