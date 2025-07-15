# CUDA Math Labs

High-performance CUDA math library research and contributions for NVIDIA CUDA Math Libraries.

## Structure
- `libs/` - Forked repositories and upstream PR mirrors
- `bench/` - Reproducible benchmarks with JSON + plots  
- `docs/` - MkDocs site with theory write-ups
- `notebooks/` - Jupyter notebooks with roofline plots
- `examples/` - Demo implementations and samples
- `tests/` - Unit tests and accuracy validation

## Quick Start
```bash
# Setup development environment
./scripts/setup_dev.sh

# Run benchmarks
./scripts/run_benchmarks.sh

# Build documentation
mkdocs serve
```
