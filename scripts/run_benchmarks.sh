#!/bin/bash
set -e

echo "Running CUDA Math Labs benchmarks..."

# Run roofline analysis
cd bench/roofline
python generate_roofline.py

# Run accuracy tests
cd ../accuracy  
python accuracy_analysis.py

echo "Benchmarks complete. Results in bench/results/"
