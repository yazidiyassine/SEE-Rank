#!/usr/bin/env bash
# Run spectral diagnostic experiments for Phi-2
python run_experiment.py --model microsoft/phi-2 --out-dir ./results/phi "$@"
