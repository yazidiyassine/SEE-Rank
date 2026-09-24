# Spectral Diagnostics of LoRA Rank Utilization & SEE-Rank

Official reproduction repository for the empirical spectral diagnostic framework of Low-Rank Adaptation (LoRA) and the **Spectral Early-Exit Rank Selection (SEE-Rank)** protocol.

---

## Overview

This repository provides an automated, reproducible pipeline for:
1. **Spectral Measurement**: Tracking exact singular value distributions, stable rank $\operatorname{sr}(\Delta W)$, effective rank $\operatorname{erank}(\Delta W)$, and rank utilization $\rho(\Delta W, r)$ across training checkpoints.
2. **Scaling Law Characterization**: Modeling sub-linear capacity scaling $\operatorname{sr}(r) \propto r^\beta$ against logarithmic and saturating alternatives via Akaike Information Criterion (AIC).
3. **Invariance Testing**: Evaluating adapter spectral collapse under weight decay, high-rate dropout, and Weight-Decomposed Low-Rank Adaptation (DoRA).
4. **Construct Validity**: In-place SVD truncation tests ($k \in \{1, r\}$) verifying whether leading singular directions carry downstream task utility.
5. **SEE-Rank Protocol**: An early-probing decision rule evaluating marginal spectral gains $g_k = \frac{\operatorname{sr}(r_k)}{\operatorname{sr}(r_{k-1})} - 1$ at early checkpoints to stop rank escalation before completing full training sweeps.

---

## Repository Structure

```
code/
├── src/                          # Modular Python package
│   ├── __init__.py               # Top-level API exports
│   ├── config.py                 # Default settings and command-line argument parsing
│   ├── data.py                   # Dataset loading, deduplication, and split isolation
│   ├── spectral.py               # QR-based exact SVD, stable rank, and layer snapshots
│   ├── evaluation.py             # Perplexity, TruthfulQA MC1, ARC-Easy, and SVD truncation test
│   ├── trainer.py                # Training loop with non-uniform checkpoint logging
│   ├── see_rank.py               # SEE-Rank decision algorithm, tau sensitivity grid, and baselines
│   └── analysis.py               # Functional-form fitting (AIC), TOST, and seed correlations
├── notebooks/                    # Interactive Jupyter notebooks for experimentation
│   ├── run_qwen_spectral.ipynb   # Experiment pipeline for Qwen2.5-1.5B
│   ├── run_llama_spectral.ipynb  # Experiment pipeline for LLaMA-3.2-1B
│   └── run_phi_spectral.ipynb    # Experiment pipeline for Phi-2
├── scripts/                      # Shell and batch execution scripts
│   ├── run_qwen.sh / .bat
│   ├── run_llama.sh / .bat
│   └── run_phi.sh / .bat
├── archive/                      # Historical raw logs and original notebook runs
├── run_experiment.py             # Main production CLI entry point
├── requirements.txt              # Pinned software dependencies
├── .gitignore                    # Environment and artifact ignore rules
└── README.md
```

---

## Installation

### Prerequisites
* Python $\ge 3.10$
* PyTorch $\ge 2.1.0$ with CUDA support
* NVIDIA GPU with $\ge 12$ GB VRAM (e.g., T4, V100, A100, RTX 3090/4090)

```bash
git clone https://github.com/yazidiyassine/PhD-Project.git
cd "PhD-Project/Articles/Article 10/rev 21-09-2026/code"
pip install -r requirements.txt
```

---

## Usage

### 1. Command-Line Interface (`run_experiment.py`)

Run diagnostic sweeps across model families:

```bash
# Qwen2.5-1.5B (Default: ranks 4, 8, 16, 32, 64; 2 seeds)
python run_experiment.py --model Qwen/Qwen2.5-1.5B --out-dir ./results/qwen

# LLaMA-3.2-1B
python run_experiment.py --model meta-llama/Llama-3.2-1B --out-dir ./results/llama

# Phi-2 (2.7B)
python run_experiment.py --model microsoft/phi-2 --out-dir ./results/phi
```

### CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--model` | Hugging Face model identifier | `Qwen/Qwen2.5-1.5B` |
| `--ranks` | Space-separated list of candidate ranks | `4 8 16 32 64` |
| `--seeds` | Space-separated list of random seeds | `42 123` |
| `--alpha-schedule` | Scaling convention (`fixed`, `2r`, `sqrt`) | `2r` |
| `--target-scope` | Adaptation scope (`attn` or `attn_mlp`) | `attn` |
| `--train-steps` | Total optimization budget per condition | `40` |
| `--probe-step` | Early probe step for SEE-Rank evaluation | `10` |
| `--no-4bit` | Train with 16-bit precision instead of 4-bit NF4 | `False` |
| `--out-dir` | Directory where output CSV/JSON logs are saved | `./results` |
| `--gpu` | Physical GPU index | `0` |
| `--fast` | Fast pilot schedule (40 steps, 2 seeds) | Preset |
| `--full` | Extended schedule (200 steps, 3 seeds) | Preset |

---

### 2. Convenience Shell / Batch Scripts

Execute reproduction runs directly:

```bash
# Linux / macOS:
bash scripts/run_qwen.sh
bash scripts/run_llama.sh
bash scripts/run_phi.sh

# Windows:
scripts\run_qwen.bat
scripts\run_llama.bat
scripts\run_phi.bat
```

---

### 3. Interactive Notebooks

Launch Jupyter Notebook and open any notebook in `notebooks/`:
```bash
jupyter notebook notebooks/run_qwen_spectral.ipynb
```

---

## Spectral Metrics & Diagnostic Definitions

* **Stable Rank**:
  $$\operatorname{sr}(\Delta W) = \frac{\|\Delta W\|_F^2}{\|\Delta W\|_2^2} = \frac{\sum_{i=1}^r \sigma_i^2}{\sigma_1^2} \in [1, r]$$
  Measures the distribution of spectral energy across singular directions.

* **Rank Utilization**:
  $$\rho(\Delta W, r) = \frac{\operatorname{sr}(\Delta W)}{r} \times 100\%$$
  Rescales stable rank to a nominal-rank-invariant percentage.

* **Top-1 Energy Concentration**:
  $$\epsilon_1(\Delta W) = \frac{\sigma_1^2}{\sum_{i=1}^r \sigma_i^2} = \frac{1}{\operatorname{sr}(\Delta W)}$$
  Captures the proportion of variance aligned with the dominant singular direction.

* **SEE-Rank Decision Rule**:
  $$g_k = \frac{\widehat{\operatorname{sr}}(r_k)}{\widehat{\operatorname{sr}}(r_{k-1})} - 1 < \tau$$
  Evaluated at early step $t_\mathrm{probe}$ across candidate ranks to terminate rank allocation before incurring full training costs.

---

## Output Artifacts

For each model, the execution produces the following files:
* `spectral_{model}_results.csv`: Run-level metrics, perplexity, downstream accuracies, and adapter norms.
* `spectral_{model}_trajectory.csv`: Checkpoint-level spectral measurements across all layers.
* `spectral_{model}_svd.csv`: Exact singular values logged at boundary steps.
* `spectral_{model}_downstream_items.csv`: Item-by-item question correctness for bootstrap confidence intervals.
* `spectral_{model}_truncation.csv`: Validation perplexity under top-$k$ SVD truncation ($k \in \{1, r\}$).
* `spectral_{model}_see_rank.csv`: SEE-Rank selected ranks, baselines, and end-to-end compute cost.
* `spectral_{model}_see_rank_trace.csv`: Marginal gain traces at each rank step.
* `spectral_{model}_functional_forms.json`: Parameter fits and AIC scores for power-law, logarithmic, and saturating models.
* `spectral_{model}_metadata.json`: Exact seeds, hardware details, timing, and library versions.

---

## License

This project is licensed under the MIT License.
