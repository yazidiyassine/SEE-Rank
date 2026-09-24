#!/usr/bin/env python3
"""
Main experiment driver for LoRA adapter spectral diagnostics and SEE-Rank evaluation.
"""

import os
import sys
import time
import json
import platform
import importlib

from src.config import get_config, parse_args


def main():
    args = parse_args()
    config = get_config(args)

    # Validate environment and import machine learning dependencies
    try:
        import torch
        import pandas as pd
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from tqdm.auto import tqdm
    except ImportError as e:
        print(f"Error: Missing required dependency ({e}).")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

    from src.data import load_data
    from src.spectral import resolve_target_modules, count_adapted_leaves
    from src.trainer import single_run
    from src.see_rank import run_see_rank_analysis
    from src.analysis import (
        build_methods,
        fit_functional_forms,
        seed_level_correlation,
    )

    t_start = time.time()
    model_id = config["MODEL"]
    config["_SHORT"] = model_id.split("/")[-1]
    os.makedirs(config["OUT_DIR"], exist_ok=True)

    device = f"cuda:{config['GPU_INDEX']}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(config["GPU_INDEX"])

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(hf_token)
        except Exception:
            pass

    print(f"Device : {device}")
    if device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(config["GPU_INDEX"])
        vram = torch.cuda.get_device_properties(config["GPU_INDEX"]).total_memory / (1024 ** 3)
        print(f"GPU    : {gpu_name} ({vram:.1f} GB VRAM)")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading base model {model_id} (4bit={config['USE_4BIT']})...")
    single_gpu_map = {"": config["GPU_INDEX"]} if device.startswith("cuda") else None
    load_kwargs = dict(device_map=single_gpu_map, trust_remote_code=True)

    if config["USE_4BIT"]:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    base_model.config.use_cache = False
    print("✓ Base model loaded")

    config["_TARGET_MODULES"] = resolve_target_modules(base_model, config["TARGET_SCOPE"])
    n_adapted = count_adapted_leaves(base_model, config["_TARGET_MODULES"])
    model_type = getattr(base_model.config, "model_type", None)
    print(f"✓ Target modules (model_type={model_type}, scope={config['TARGET_SCOPE']}): {config['_TARGET_MODULES']}")
    print(f"✓ Adapted linear leaves per run: {n_adapted}")

    train_ds, val_ds, data_manifest = load_data(tokenizer, config)

    methods = build_methods(config["RANKS"], config)
    total_runs = len(methods) * len(config["SEEDS"])
    print(f"\nExperimental grid: {len(methods)} configurations × {len(config['SEEDS'])} seeds = {total_runs} runs")

    all_results, all_spectral, all_svd, all_mc1, all_arc, all_trunc = [], [], [], [], [], []

    f_res = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_results.csv")
    f_traj = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_trajectory.csv")
    f_svd = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_svd.csv")
    f_items = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_downstream_items.csv")
    f_trunc = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_truncation.csv")
    f_seerank = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_see_rank.csv")
    f_seerank_trace = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_see_rank_trace.csv")
    f_forms = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_functional_forms.json")
    f_meta = os.path.join(config["OUT_DIR"], f"spectral_{config['_SHORT']}_metadata.json")

    run_bar = tqdm(total=total_runs, desc="Progress", unit="run")
    for method_label, rank, alpha, dropout, wd, use_dora in methods:
        for seed in config["SEEDS"]:
            run_bar.set_postfix_str(f"{method_label} s={seed}")
            try:
                res, spec, svd, mc1_items, arc_items, trunc_rows = single_run(
                    base_model=base_model,
                    tokenizer=tokenizer,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    method_label=method_label,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    wd=wd,
                    use_dora=use_dora,
                    seed=seed,
                    config=config,
                    device=device,
                )
                all_results.append(res)
                for r in spec:
                    r.update({"model": config["_SHORT"], "method": method_label, "seed": seed, "rank": rank})
                all_spectral.extend(spec)
                for r in svd:
                    r.update({"model": config["_SHORT"], "method": method_label, "seed": seed, "rank": rank})
                all_svd.extend(svd)
                all_mc1.extend(mc1_items)
                all_arc.extend(arc_items)
                all_trunc.extend(trunc_rows)

                pd.DataFrame(all_results).to_csv(f_res, index=False)
                pd.DataFrame(all_spectral).to_csv(f_traj, index=False)
            except Exception as e:
                print(f"  [Error] Run failed for {method_label} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
            run_bar.update(1)
    run_bar.close()

    df_res = pd.DataFrame(all_results)
    df_spec = pd.DataFrame(all_spectral)
    df_svd = pd.DataFrame(all_svd)
    df_items = pd.DataFrame(all_mc1 + all_arc)
    df_trunc = pd.DataFrame(all_trunc)

    df_res.to_csv(f_res, index=False)
    df_spec.to_csv(f_traj, index=False)
    if len(df_svd):
        df_svd.to_csv(f_svd, index=False)
    if len(df_items):
        df_items.to_csv(f_items, index=False)
    if len(df_trunc):
        df_trunc.to_csv(f_trunc, index=False)

    print(f"\n✓ Saved results: {f_res} ({len(df_res)} rows)")
    print(f"✓ Saved trajectories: {f_traj} ({len(df_spec)} rows)")
    if len(df_svd):
        print(f"✓ Saved SVD spectra: {f_svd} ({len(df_svd)} rows)")
    if len(df_items):
        print(f"✓ Saved item-level downstream results: {f_items} ({len(df_items)} rows)")
    if len(df_trunc):
        print(f"✓ Saved truncation validation results: {f_trunc} ({len(df_trunc)} rows)")

    # SEE-Rank decision analysis
    if len(df_res) and len(df_spec):
        try:
            df_see, df_trace = run_see_rank_analysis(df_res, df_spec, config)
            if len(df_see):
                df_see.to_csv(f_seerank, index=False)
                df_trace.to_csv(f_seerank_trace, index=False)
                print(f"✓ Saved SEE-Rank summary: {f_seerank} ({len(df_see)} rows)")
        except Exception as e:
            print(f"  [Warning] SEE-Rank analysis failed: {e}")

    # Functional-form fitting (AIC comparison)
    forms_out = {}
    if len(df_spec):
        final_ba = df_spec[
            (df_spec["step"] == config["TRAIN_STEPS"]) &
            (df_spec["delta_type"] == "BA") &
            (~df_spec["degenerate"])
        ]
        rank_sr = final_ba.groupby("rank")["stable_rank"].mean()
        if len(rank_sr) >= 3:
            forms_out = fit_functional_forms(rank_sr.index.tolist(), rank_sr.values.tolist())
            print("\nFunctional-form fit (lower AIC indicates preferred model):")
            for name, fit_params in forms_out.items():
                print(f"  {name}: {fit_params}")
    with open(f_forms, "w", encoding="utf-8") as fh:
        json.dump(forms_out, fh, indent=2)

    # Seed-level correlations
    corr_summary = []
    if len(df_res) >= 4:
        for metric in ["final_ppl", "truthfulqa_mc1", "arc_easy"]:
            c = seed_level_correlation(df_res, "mean_sr_final", metric)
            if c:
                corr_summary.append(c)
        print("\nSeed-level Pearson correlations with final mean stable rank:")
        for c in corr_summary:
            print(f"  {c['y']}: r={c['r']:.4f} (p={c['p']:.6f}, n={c['n']})")

    # Run metadata & environment reproducibility report
    meta = {
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "resolved_target_modules": config["_TARGET_MODULES"],
        "n_adapted_leaves": n_adapted,
        "model_config_summary": {
            k: getattr(base_model.config, k, None)
            for k in [
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
            ]
        },
        "data_manifest": data_manifest,
        "seed_level_correlations": corr_summary,
        "library_versions": {
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", None),
        },
        "device": device,
        "total_wallclock_min": round((time.time() - t_start) / 60, 2),
    }

    for lib in ["transformers", "peft", "datasets", "bitsandbytes", "accelerate", "scipy"]:
        try:
            meta["library_versions"][lib] = importlib.import_module(lib).__version__
        except Exception:
            meta["library_versions"][lib] = "not available"

    with open(f_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"✓ Saved experiment metadata: {f_meta}")

    total_time = (time.time() - t_start) / 60
    print(f"\nExecution finished in {total_time:.1f} minutes.")


if __name__ == "__main__":
    main()
