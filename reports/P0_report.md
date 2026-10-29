# Phase 0 Report: Environment Setup & Sanity Check

## Summary
- **Python Version**: 3.10.22 (arm64, Apple Silicon)
- **PyTorch Version**: 2.14.1 (MPS acceleration available)
- **PyG (torch_geometric) Version**: 2.8.1
- **Key Scientific Libraries**: NumPy 2.2.5, SciPy 1.15.3, Scikit-learn 1.7.2, NetworkX 3.4.2, Pandas 2.3.3, PyYAML 6.0.3, pytest 9.2.0
- **Virtual Environment**: `.venv` in workspace root.
- **Environment Freeze**: saved to `results/environment.txt`.

## Data Verification
Verified the `data/` folder directory counts:
- `data/TRIT-TC`: 581 entries (580 Trojan netlists + README/structure)
- `data/TRIT-TS`: 335 entries (334 Trojan netlists + `original_designs/`)
- `data/TjFree`: 339 entries

## Sanity Test
`experiments/p0_check_env.py` executed successfully:
- Initialized `Data(x, edge_index)`
- Executed `SAGEConv` forward pass without errors, output shape `(5, 2)`
- `pytest tests/test_parser.py` ran and passed.

## Notes & Observations
- On macOS Apple Silicon with Accelerate/OpenBLAS, multi-threaded BLAS worker contention was avoided by setting `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.
- Phase 0 acceptance criteria are fully met.

