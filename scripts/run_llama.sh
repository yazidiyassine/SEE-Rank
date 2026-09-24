#!/usr/bin/env bash
# Run spectral diagnostic experiments for LLaMA-3.2-1B
python run_experiment.py --model meta-llama/Llama-3.2-1B --out-dir ./results/llama "$@"
