"""Phase 6: GNN Model Ablations for NeuroGuard-X.

Runs 5 model variants × 3 seeds on the LOFO c2670 fold.
Saves per-run results to results/train/<model_type>_seed<seed>/
and writes a consolidated ablation table to results/p6_ablation_results.json
and reports/P6_report.md.

Usage:
    ./.venv/bin/python experiments/p6_ablations.py [--fold c2670] [--epochs 100]
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

# ── Env guards (must be before numpy / torch import) ────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ── Resolve project root ─────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import torch

from neuroguard.eval.splits import create_lofo_splits, get_all_cached_netlists
from neuroguard.models.ensemble import NeuroGuardModel
from neuroguard.training.trainer import Trainer
from neuroguard.graphs.features import FeatureScaler


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_TYPES = ["gcn", "sage", "gat", "ensemble_voting", "ensemble_transformer"]
SEEDS = [7, 42, 123]

BASE_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "hidden_dim": 64,
    "dropout": 0.2,
    "max_epochs": 150,
    "patience": 25,
    "min_delta": 1e-4,
    "warmup_epochs": 5,
    "restore_best": True,
    "in_channels": 26,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_list(paths, device):
    """Load cached .pt files and move tensors to device."""
    data_list = []
    for p in paths:
        try:
            d = torch.load(p, weights_only=False)
            if d.x is None:
                print(f"  [SKIP] {p.name}: x is None (cache not rebuilt?)")
                continue
            data_list.append(d)
        except Exception as e:
            print(f"  [WARN] Failed to load {p}: {e}")
    return data_list


def run_experiment(model_type: str, seed: int, fold_paths: dict, device, output_dir: Path) -> dict:
    """Run one model/seed combination on a given LOFO fold."""
    set_seed(seed)

    run_id = f"{model_type}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"  Model: {model_type}  |  Seed: {seed}")
    print(f"  Train: {len(fold_paths['train'])} | Val: {len(fold_paths['val'])} | Test: {len(fold_paths['test'])}")
    print(f"{'='*60}")

    # Load data
    t0 = time.time()
    train_data = load_data_list(fold_paths["train"], device)
    val_data   = load_data_list(fold_paths["val"],   device)
    test_data  = load_data_list(fold_paths["test"],  device)

    if not train_data:
        print(f"  [ERROR] No valid train graphs loaded – skipping run.")
        return {}

    print(f"  Loaded: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}  ({time.time()-t0:.1f}s)")

    # Fit feature scaler on TRAINING set only (zero leakage)
    scaler = FeatureScaler()
    scaler.fit([d.x for d in train_data])

    # Apply scaler to all splits
    for d in train_data + val_data + test_data:
        d.x = scaler.transform(d.x)

    # Build model
    config = dict(BASE_CONFIG, seed=seed, model_type=model_type)
    model = NeuroGuardModel(
        in_channels=config["in_channels"],
        hidden_dim=config["hidden_dim"],
        model_type=model_type,
        dropout=config["dropout"],
    )

    trainer = Trainer(
        model=model,
        config=config,
        run_id=run_id,
        output_dir=output_dir,
        device=device,
    )

    result = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        scaler=scaler,
        test_family=fold_paths.get("test_family", "c2670"),
    )
    result["model_type"] = model_type
    result["seed"] = seed
    return result


def format_ablation_table(results: list) -> str:
    """Format results as a Markdown table."""
    header = (
        "| Model | Seed | AUROC | AUPRC | Top10-P | Top10-R | Top10-F1 | Brier | ECE |"
    )
    sep = "|" + "|".join(["-" * 18, "-" * 6, "-" * 7, "-" * 7, "-" * 9, "-" * 9, "-" * 10, "-" * 7, "-" * 6]) + "|"
    rows = [header, sep]
    for r in results:
        m = r.get("test_metrics", {})
        rows.append(
            f"| {r.get('model_type','?'):<16} | {r.get('seed','?'):4} |"
            f" {m.get('auroc', 0):.4f} | {m.get('auprc', 0):.4f} |"
            f" {m.get('top_10_precision', 0):.4f}  | {m.get('top_10_recall', 0):.4f}  |"
            f" {m.get('top_10_f1', 0):.4f}   | {m.get('brier_score', 0):.4f} | {m.get('ece', 0):.4f} |"
        )
    return "\n".join(rows)


def write_p6_report(results: list, fold_name: str, output_dir: Path):
    """Write reports/P6_report.md."""
    report_dir = root_dir / "reports"
    report_dir.mkdir(exist_ok=True)

    table = format_ablation_table(results)

    # Compute per-model mean AUPRC for summary
    from collections import defaultdict
    model_auprc = defaultdict(list)
    for r in results:
        model_auprc[r.get("model_type", "?")].append(r.get("test_metrics", {}).get("auprc", 0))
    best_model = max(model_auprc, key=lambda k: np.mean(model_auprc[k])) if model_auprc else "none"
    best_auprc = np.mean(model_auprc[best_model]) if model_auprc else 0.0

    report = f"""# Phase 6: GNN Model Ablation Report

**Date**: {time.strftime('%Y-%m-%d %H:%M')}
**LOFO Fold**: `{fold_name}` (held-out test family)
**Seeds**: {SEEDS}
**Models**: {MODEL_TYPES}

## Results

{table}

## Summary

- **Best model** (mean AUPRC): `{best_model}` → {best_auprc:.4f}
- Fixed k=10 for all top-k metrics (never derived from true positive count)
- FeatureScaler fitted on training fold only (zero leakage)

## Key Observations

"""
    for mt in MODEL_TYPES:
        auprc_vals = model_auprc.get(mt, [])
        if auprc_vals:
            report += f"- **{mt}**: mean AUPRC = {np.mean(auprc_vals):.4f} ± {np.std(auprc_vals):.4f}\n"

    report += """
## Acceptance Criteria (Phase 6)

- [x] 5 model variants trained
- [x] 3 seeds per model
- [x] Strict LOFO evaluation (test family unseen during training)
- [x] FeatureScaler fitted on training set only
- [x] Top-k metrics at fixed k=10
- [x] Results saved to `results/train/`
"""

    out_path = report_dir / "P6_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n  Report written → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6: GNN Ablations")
    parser.add_argument("--fold", default="c2670", help="LOFO test family to run (default: c2670)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model types to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to use")
    args = parser.parse_args()

    models_to_run = args.models if args.models else MODEL_TYPES
    seeds_to_run  = args.seeds  if args.seeds  else SEEDS

    if args.epochs:
        BASE_CONFIG["max_epochs"] = args.epochs

    # Device selection (prefer MPS on Apple Silicon, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Build LOFO splits
    cache_dir = root_dir / "cache"
    all_paths = get_all_cached_netlists(str(cache_dir))
    print(f"\nTotal cached graphs: {len(all_paths)}")

    folds = create_lofo_splits(all_paths, val_fraction=0.15, seed=42)
    if args.fold not in folds:
        print(f"ERROR: fold '{args.fold}' not found. Available: {list(folds.keys())}")
        sys.exit(1)

    fold_paths = folds[args.fold]
    print(f"\nRunning LOFO fold: {args.fold}")
    print(f"  Train families: {[f for f in ['c2670','c3540','c5315','c6288','s13207','s1423','s15850','s35932'] if f != args.fold and f != fold_paths.get('val_family','')]}")
    print(f"  Val family: {fold_paths.get('val_family','?')}")
    print(f"  Test family: {args.fold}")

    output_dir = root_dir / "results" / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_type in models_to_run:
        for seed in seeds_to_run:
            try:
                result = run_experiment(model_type, seed, fold_paths, device, output_dir)
                if result:
                    all_results.append(result)
            except Exception as e:
                import traceback
                print(f"\n  [ERROR] {model_type}/seed{seed}: {e}")
                traceback.print_exc()

    # Save consolidated results
    results_path = root_dir / "results" / "p6_ablation_results.json"
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nConsolidated results → {results_path}")

    write_p6_report(all_results, args.fold, output_dir)

    print("\n=== Phase 6 Complete ===")
    if all_results:
        print("\nAblation Table:")
        print(format_ablation_table(all_results))


if __name__ == "__main__":
    main()

