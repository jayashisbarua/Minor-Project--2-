"""Evaluation, splits, and metrics module."""
from neuroguard.eval.splits import create_lofo_splits, create_trojan_type_splits
from neuroguard.eval.metrics import compute_node_metrics, compute_graph_metrics
from neuroguard.eval.stats import mcnemar_test, bootstrap_ci

