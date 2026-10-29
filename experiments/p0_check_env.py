#! /usr/bin/env python3
"""
Phase 0: Environment sanity check.
Ensures Python, PyTorch, PyTorch Geometric are importable and functional.
Creates a tiny torch_geometric.data.Data object and runs a single SAGEConv forward pass.
"""
import sys
import subprocess
import importlib

def print_version(pkg_name, module_name=None):
    try:
        mod = importlib.import_module(module_name or pkg_name)
        print(f"{pkg_name} version:", getattr(mod, "__version__", "unknown"))
    except Exception as e:
        print(f"Failed to import {pkg_name}: {e}")
        sys.exit(1)

def main():
    print("Python version:", sys.version)
    print_version('torch')
    print_version('torch_geometric')
    # Simple sanity test
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
        x = torch.randn((5, 3))  # 5 nodes, 3 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        conv = SAGEConv(in_channels=3, out_channels=2)
        out = conv(data.x, data.edge_index)
        print("SAGEConv forward pass succeeded, output shape:", out.shape)
    except Exception as e:
        print("Sanity test failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()