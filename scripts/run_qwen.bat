@echo off
REM Run spectral diagnostic experiments for Qwen2.5-1.5B
python run_experiment.py --model Qwen/Qwen2.5-1.5B --out-dir ./results/qwen %*
