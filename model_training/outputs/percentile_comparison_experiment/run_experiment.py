"""Reproducible ZeroShield six-model percentile comparison experiment.

Uses the project's existing model implementations in the repository-root
models.py. Outputs are isolated in this experiment directory.
"""
from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MODELS_DIR = OUT / "models"
TRAIN = ROOT / "train_benign_only.csv"
VALID = ROOT / "validation_set.csv"
ZERO = ROOT / "zero_day_test_scaled_v2.csv"
FEATURE_FILE = ROOT / "model_training" / "outputs" / "zero_shield_feature_columns.json"
PERCENTILES = [90, 93, 95, 97, 99]
SEED = 42
POOL_SIZE = 600_000
TRAIN_ROWS = 500_000
REFERENCE_ROWS = 100_000
TRAIN_CHUNK = 100_000
EVAL_CHUNK = 50_000
BATCH = 2048
MODELS = ["AE", "VAE", "DAE", "CAE", "Deep SVDD", "OCSVM"]
MODEL_LABELS = {
    "AE": "Autoencoder", "VAE": "Variational Autoencoder",
    "DAE": "Denoising Autoencoder", "CAE": "Contractive Autoencoder",
    "Deep SVDD": "Deep SVDD", "OCSVM": "One-Class SVM",
}

start = datetime.now(timezone.utc)
log_lines: list[str] = [f"Start time (UTC): {start.isoformat()}"]
warnings: list[str] = []
training_records: list[dict] = []


def log(message: str) -> None:
    print(message, flush=True)
    log_lines.append(message)


def load_existing_impl():
    spec = importlib.util.spec_from_file_location("zeroshield_existing_models", ROOT / "models.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load existing root models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def common_benign_pool(features: list[str]) -> tuple[np.ndarray, int]:
    """Uniform fixed-size row sample using independent random priorities."""
    rng = np.random.default_rng(SEED)
    best_keys = np.empty(0, dtype=np.float64)
    best_rows = np.empty((0, len(features)), dtype=np.float32)
    seen = 0
    for chunk in pd.read_csv(TRAIN, usecols=features, chunksize=TRAIN_CHUNK, low_memory=True):
        arr = chunk.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32, copy=False)
        keys = rng.random(len(arr))
        seen += len(arr)
        if len(best_keys) == 0:
            merged_keys, merged_rows = keys, arr.copy()
        else:
            merged_keys = np.concatenate((best_keys, keys))
            merged_rows = np.concatenate((best_rows, arr), axis=0)
        keep = min(POOL_SIZE, len(merged_keys))
        if len(merged_keys) > keep:
            idx = np.argpartition(merged_keys, keep - 1)[:keep]
            best_keys, best_rows = merged_keys[idx], merged_rows[idx]
        else:
            best_keys, best_rows = merged_keys, merged_rows
        log(f"Training-source scan: {seen:,} benign rows read")
        del arr, keys, merged_keys, merged_rows
    if seen < POOL_SIZE:
        raise ValueError(f"Need {POOL_SIZE:,} disjoint training/reference rows; found {seen:,}.")
    # Stable order and split are fully determined by the random keys and seed.
    order = np.argsort(best_keys, kind="stable")
    pool = best_rows[order]
    del best_rows, best_keys
    split_rng = np.random.default_rng(SEED + 1)
    order = split_rng.permutation(len(pool))
    train = np.array(pool[order[:TRAIN_ROWS]], dtype=np.float32, order="C", copy=True)
    reference = np.array(pool[order[TRAIN_ROWS:TRAIN_ROWS + REFERENCE_ROWS]], dtype=np.float32, order="C", copy=True)
    np.save(OUT / "training_sample_500k.npy", train)
    np.save(OUT / "benign_reference_100k.npy", reference)
    return train, reference, seen


def score_torch(fn, x: np.ndarray) -> np.ndarray:
    return np.asarray(fn(x), dtype=np.float64).reshape(-1)


def get_labels(chunk: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(chunk["binary_label"], errors="coerce")
    missing = y.isna()
    if missing.any():
        fallback = chunk.loc[missing, "label"].astype(str).str.strip().ne("BenignTraffic").astype(np.int8)
        y.loc[missing] = fallback.to_numpy()
    if y.isna().any():
        raise ValueError("Could not resolve every binary_label from the project's label rule.")
    vals = y.astype(np.int8).to_numpy()
    bad = np.setdiff1d(np.unique(vals), np.array([0, 1], dtype=np.int8))
    if len(bad):
        raise ValueError(f"Unexpected binary labels: {bad.tolist()}")
    return vals


def metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + tn + fp + fn
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fpr, "specificity": specificity, "tpr": recall,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "total": int(total),
    }


def evaluate(path: Path, features: list[str], scorer, thresholds: list[float], model_name: str,
             category: bool = False) -> tuple[list[dict], dict, int]:
    counts = [{k: 0 for k in ("tp", "tn", "fp", "fn")} for _ in thresholds]
    category_counts: dict[str, list[dict]] = defaultdict(
        lambda: [{k: 0 for k in ("tp", "fn")} for _ in thresholds]
    )
    rows_seen = 0
    cols = features + ["binary_label", "label"]
    for chunk_num, chunk in enumerate(pd.read_csv(path, usecols=cols, chunksize=EVAL_CHUNK, low_memory=True), 1):
        xdf = chunk[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = get_labels(chunk)
        scores = np.asarray(scorer(xdf, x), dtype=np.float64).reshape(-1)
        if len(scores) != len(y) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid anomaly scores for {model_name}: chunk {chunk_num}")
        preds = scores[:, None] >= np.asarray(thresholds)[None, :]
        for j in range(len(thresholds)):
            p = preds[:, j]
            counts[j]["tp"] += int(np.count_nonzero(p & (y == 1)))
            counts[j]["tn"] += int(np.count_nonzero((~p) & (y == 0)))
            counts[j]["fp"] += int(np.count_nonzero(p & (y == 0)))
            counts[j]["fn"] += int(np.count_nonzero((~p) & (y == 1)))
        if category:
            labels = chunk["label"].astype(str).to_numpy()
            for cat in np.unique(labels):
                m = labels == cat
                # The zero-day file is expected to contain attacks only. Retain
                # category totals explicitly and calculate detected/missed per cut.
                if np.any(y[m] == 0):
                    warnings.append(f"Benign-labelled rows found in zero-day category {cat!r}.")
                for j in range(len(thresholds)):
                    c = category_counts[cat][j]
                    c["samples"] = c.get("samples", 0) + int(np.count_nonzero(m & (y == 1)))
                    c["detected"] = c.get("detected", 0) + int(np.count_nonzero(m & (y == 1) & preds[:, j]))
                    c["missed"] = c.get("missed", 0) + int(np.count_nonzero(m & (y == 1) & (~preds[:, j])))
        rows_seen += len(chunk)
        if chunk_num == 1 or chunk_num % 20 == 0:
            log(f"{model_name}: streamed {path.name}: {rows_seen:,} rows")
        del chunk, xdf, x, y, scores, preds
    result = [metrics(**v) for v in counts]
    return result, dict(category_counts), rows_seen


def model_specifications(impl) -> dict:
    return {
        "AE": {"class": impl.AE, "train": lambda m, x: impl.train_ae(m, x, False), "score": lambda m: lambda x: impl.ae_scores(m, x), "file": "AE.pt"},
        "VAE": {"class": impl.VAE, "train": impl.train_vae, "score": lambda m: lambda x: impl.vae_scores(m, x), "file": "VAE.pt"},
        "DAE": {"class": impl.DenoisingAutoencoder, "train": lambda m, x: impl.train_ae(m, x, True), "score": lambda m: lambda x: impl.ae_scores(m, x), "file": "DAE.pt"},
        "CAE": {"class": impl.CAE, "train": impl.train_cae, "score": lambda m: lambda x: impl.cae_scores(m, x), "file": "CAE.pt"},
        "Deep SVDD": {"class": impl.DeepSVDD, "train": impl.train_svdd, "file": "Deep_SVDD.pt"},
    }


def make_figures(master: pd.DataFrame, thresholds: pd.DataFrame, cats: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                        "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300})
    metric_names = [
        ("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall / TPR"),
        ("F1", "F1 score"), ("FPR", "False positive rate"), ("Specificity", "Specificity"),
        ("Zero_Day_Detection_Rate", "Zero-day detection rate"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 17), sharex=True)
    colors = dict(zip(MODELS, plt.cm.tab10.colors[:len(MODELS)]))
    for ax, (metric, title) in zip(axes.flat, metric_names):
        for model in MODELS:
            d = master[master.Model == model].sort_values("Percentile")
            ax.plot(d.Percentile, 100*d[metric], marker="o", linewidth=1.8, color=colors[model], label=model)
        ax.set_title(title)
        ax.set_ylabel("Percent (%)")
        ax.set_xticks(PERCENTILES, [f"{p}%" for p in PERCENTILES])
        ax.grid(True, alpha=.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Six-model performance across benign-calibrated percentiles", fontsize=15, y=.995)
    fig.tight_layout(rect=(0, .03, 1, .98))
    fig.savefig(OUT / "all_models_percentile_comparison.png", bbox_inches="tight")
    plt.close(fig)

    chart_specs = [
        ("F1", "F1 score", "f1_vs_percentile.png"), ("Recall", "Recall / TPR", "recall_vs_percentile.png"),
        ("FPR", "False positive rate", "fpr_vs_percentile.png"),
        ("Zero_Day_Detection_Rate", "Zero-day detection rate", "zero_day_vs_percentile.png"),
        ("Precision", "Precision", "precision_vs_percentile.png"),
        ("Accuracy", "Accuracy", "accuracy_vs_percentile.png"),
    ]
    for metric, title, filename in chart_specs:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for model in MODELS:
            d = master[master.Model == model].sort_values("Percentile")
            ax.plot(d.Percentile, 100*d[metric], marker="o", linewidth=2, color=colors[model], label=model)
        ax.set(title=f"{title} by benign reference percentile", xlabel="Threshold percentile", ylabel="Percent (%)")
        ax.set_xticks(PERCENTILES, [f"{p}%" for p in PERCENTILES]); ax.grid(True, alpha=.25)
        ax.legend(ncol=2, frameon=False); fig.tight_layout(); fig.savefig(OUT / filename, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model in MODELS:
        row = thresholds[thresholds.Model == model].iloc[0]
        values = [row[f"Threshold_{p}"] for p in PERCENTILES]
        ax.plot(PERCENTILES, values, marker="o", linewidth=2, color=colors[model], label=model)
    ax.set(title="Anomaly threshold by benign reference percentile", xlabel="Percentile", ylabel="Anomaly score threshold")
    ax.set_xticks(PERCENTILES, [f"{p}%" for p in PERCENTILES]); ax.grid(True, alpha=.25); ax.legend(ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "threshold_vs_percentile.png", bbox_inches="tight"); plt.close(fig)

    at95 = master[master.Percentile == 95].set_index("Model").reindex(MODELS)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for ax, model in zip(axes.flat, MODELS):
        r = at95.loc[model]
        mat = np.array([[r.TN, r.FP], [r.FN, r.TP]], dtype=np.int64)
        im = ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="black", fontsize=9)
        ax.set(title=model, xticks=[0, 1], xticklabels=["Pred benign", "Pred attack"],
               yticks=[0, 1], yticklabels=["Benign", "Attack"])
    fig.suptitle("Validation confusion matrices at the 95th percentile", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_comparison.png", bbox_inches="tight"); plt.close(fig)

    p95 = cats[cats.Percentile == 95].pivot(index="Attack_Category", columns="Model", values="Detection_Rate").reindex(columns=MODELS)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(p95.to_numpy()*100, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=25, ha="right")
    ax.set_yticks(range(len(p95.index)), p95.index)
    for i in range(len(p95.index)):
        for j in range(len(MODELS)):
            v = p95.iloc[i, j]
            if pd.notna(v): ax.text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Detection rate (%)")
    ax.set_title("Zero-day attack-category detection at the 95th percentile")
    fig.tight_layout(); fig.savefig(OUT / "zero_day_heatmap.png", bbox_inches="tight"); plt.close(fig)

    zd = master.pivot(index="Model", columns="Percentile", values="Zero_Day_Detection_Rate").reindex(index=MODELS, columns=PERCENTILES)
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    im = ax.imshow(zd.to_numpy()*100, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(PERCENTILES)), [f"{p}%" for p in PERCENTILES]); ax.set_yticks(range(len(MODELS)), MODELS)
    for i in range(len(MODELS)):
        for j in range(len(PERCENTILES)):
            ax.text(j, i, f"{zd.iloc[i,j]*100:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Detection rate (%)"); ax.set_title("Zero-day detection across models and percentiles")
    fig.tight_layout(); fig.savefig(OUT / "zero_day_percentile_heatmap.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    if not all(p.exists() for p in (TRAIN, VALID, ZERO, FEATURE_FILE)):
        raise FileNotFoundError("One or more inspected input files are missing.")
    features = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
    if len(features) != 43 or len(set(features)) != len(features):
        raise ValueError(f"Expected the project's ordered 43-feature list, got {len(features)} unique={len(set(features))}.")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    impl = load_existing_impl()
    # Keep the existing root script's model implementations and configuration.
    impl.SEED = SEED; impl.EPOCHS = 8; impl.LR = 1e-3; impl.LATENT = 16
    impl.NOISE_STD = 0.05; impl.CAE_LAMBDA = 1e-4; impl.BATCH = BATCH
    impl.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    impl.TRAIN_SAMPLE_SIZE = TRAIN_ROWS; impl.TRAIN_SCAN_CHUNK = TRAIN_CHUNK
    impl.EVAL_CHUNK = EVAL_CHUNK; impl.OUT = OUT
    impl.SAMPLE_FILE = OUT / "training_sample_500k.npy"
    impl.FEATURE_FILE = FEATURE_FILE

    log(f"Project root: {ROOT}")
    log(f"Training source: {TRAIN} ({TRAIN.stat().st_size:,} bytes)")
    log(f"Validation source: {VALID} ({VALID.stat().st_size:,} bytes)")
    log(f"Zero-day source: {ZERO} ({ZERO.stat().st_size:,} bytes)")
    log(f"Features ({len(features)} ordered): {features}")
    log(f"Environment: Python {platform.python_version()}, torch {torch.__version__}, sklearn {sklearn.__version__}, pandas {pd.__version__}, NumPy {np.__version__}")
    log(f"Device: {impl.DEVICE}; CPU: {platform.processor() or 'not reported'}; logical CPUs: {os.cpu_count()}")
    train_x, reference_x, train_source_rows = common_benign_pool(features)
    log(f"Selected disjoint benign rows: {len(train_x):,} training; {len(reference_x):,} threshold-reference")

    model_rows: list[dict] = []
    all_results: list[dict] = []
    category_rows: list[dict] = []
    threshold_rows: list[dict] = []
    train_cfg = {
        "AE": {"architecture": "43-64-32-16-32-64-43", "loss": "mean squared reconstruction error", "optimizer": "Adam"},
        "VAE": {"architecture": "43-64-32-(mu/logvar 16)-32-64-43", "loss": "MSE + 0.01 * KL; score is MSE decoded from mu", "optimizer": "Adam"},
        "DAE": {"architecture": "43-64-32-16-32-64-43", "loss": "MSE on clean target with Gaussian input noise std 0.05", "optimizer": "Adam"},
        "CAE": {"architecture": "43-64(sigmoid)-16-64(ReLU)-43", "loss": "MSE + 1e-4 * contractive penalty", "optimizer": "Adam"},
        "Deep SVDD": {"architecture": "43-64-32-16; bias-free ReLU encoder", "loss": "squared distance to fixed center; weight decay 1e-6", "optimizer": "Adam"},
        "OCSVM": {"architecture": "RBF OneClassSVM on seeded 10,000-row subset", "loss": None, "optimizer": None,
                  "kernel": "rbf", "gamma": "scale", "nu": 0.05, "cache_size_mb": 1024,
                  "tol": 1e-3, "shrinking": True, "random_state": SEED},
    }
    for model_name in MODELS:
        name = MODEL_LABELS[model_name]
        log(f"\n=== Training {name} ===")
        t0 = time.perf_counter()
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
        loss = None
        if model_name == "OCSVM":
            rng = np.random.default_rng(SEED)
            svm_idx = rng.choice(len(train_x), size=10_000, replace=False)
            svm_x = train_x[svm_idx]
            svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale", cache_size=1024,
                              shrinking=True, tol=1e-3)
            svm.fit(svm_x)
            training_seconds = time.perf_counter() - t0
            model = svm
            scorer = lambda xdf, xnp: -model.decision_function(xnp)
            reference_scores = -model.decision_function(reference_x)
            training_count = len(svm_x)
            model_path = MODELS_DIR / "One_Class_SVM.joblib"
            import joblib
            joblib.dump({"model": svm, "features": features, "reference_scores": None}, model_path)
            del svm_x, svm_idx
        elif model_name == "Deep SVDD":
            model, center, loss = impl.train_svdd(impl.DeepSVDD(len(features)), train_x)
            training_seconds = time.perf_counter() - t0
            model = model.to(impl.DEVICE)
            scorer = lambda xdf, xnp: impl.svdd_scores(model, center, xnp)
            reference_scores = score_torch(lambda x: impl.svdd_scores(model, center, x), reference_x)
            training_count = len(train_x)
            model_path = MODELS_DIR / "Deep_SVDD.pt"
            checkpoint = {"state_dict": model.state_dict(), "center": center.detach().cpu(),
                          "input_dim": len(features), "latent_dim": 16, "features": features,
                          "seed": SEED, "training_rows": training_count, "reference_rows": len(reference_x)}
            torch.save(checkpoint, model_path)
        else:
            specs = model_specifications(impl)
            spec = specs[model_name]
            model = spec["class"](len(features))
            model = model.to(impl.DEVICE)
            model, loss = spec["train"](model, train_x)
            training_seconds = time.perf_counter() - t0
            scorer_only = spec["score"](model)
            scorer = lambda xdf, xnp, f=scorer_only: f(xnp)
            reference_scores = score_torch(scorer_only, reference_x)
            training_count = len(train_x)
            model_path = MODELS_DIR / spec["file"]
            checkpoint = {"state_dict": model.state_dict(), "input_dim": len(features), "latent_dim": 16,
                          "features": features, "seed": SEED, "training_rows": training_count,
                          "training_source": str(TRAIN), "reference_rows": len(reference_x)}
            if model_name == "DAE": checkpoint["noise_std"] = 0.05
            if model_name == "CAE": checkpoint["contractive_lambda"] = 1e-4
            torch.save(checkpoint, model_path)

        reference_scores = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
        if len(reference_scores) != REFERENCE_ROWS or not np.isfinite(reference_scores).all():
            raise ValueError(f"Invalid benign reference scores for {name}.")
        cuts = [float(np.percentile(reference_scores, p)) for p in PERCENTILES]
        model_elapsed = training_seconds
        training_records.append({"model": model_name, "label": name, "training_time_seconds": model_elapsed,
                                 "training_loss": None if loss is None else float(loss),
                                 "training_rows": int(training_count), "reference_rows": len(reference_x),
                                 "checkpoint": str(model_path), "thresholds": dict(zip(map(str, PERCENTILES), cuts))})
        threshold_rows.append({"Model": model_name, **{f"Threshold_{p}": cut for p, cut in zip(PERCENTILES, cuts)}})

        val_metrics, _, val_rows = evaluate(VALID, features, scorer, cuts, name, category=False)
        zero_metrics, per_categories, zero_rows = evaluate(ZERO, features, scorer, cuts, name, category=True)
        for p, t, vm, zm in zip(PERCENTILES, cuts, val_metrics, zero_metrics):
            zd_rate = zm["recall"]
            all_results.append({"Model": model_name, "Percentile": p, "Threshold": t,
                                "Accuracy": vm["accuracy"], "Precision": vm["precision"], "Recall": vm["recall"],
                                "F1": vm["f1"], "FPR": vm["fpr"], "Specificity": vm["specificity"], "TPR": vm["tpr"],
                                "Zero_Day_Detection_Rate": zd_rate, "TP": vm["tp"], "TN": vm["tn"],
                                "FP": vm["fp"], "FN": vm["fn"], "Evaluation_Samples": vm["total"],
                                "Evaluation_Positive_Samples": vm["tp"] + vm["fn"],
                                "Evaluation_Negative_Samples": vm["tn"] + vm["fp"],
                                "Zero_Day_Samples": zm["tp"] + zm["fn"], "Zero_Day_Detected": zm["tp"],
                                "Zero_Day_Missed": zm["fn"], "Validation_Rows": val_rows, "Zero_Day_Rows": zero_rows})
            for cat, counts_by_p in per_categories.items():
                cc = counts_by_p[PERCENTILES.index(p)]
                samples = int(cc.get("samples", 0)); detected = int(cc.get("detected", 0)); missed = int(cc.get("missed", 0))
                category_rows.append({"Model": model_name, "Percentile": p, "Attack_Category": cat,
                                      "Samples": samples, "Detected": detected, "Missed": missed,
                                      "Detection_Rate": detected / samples if samples else 0.0})
        model_rows.append({"Model": model_name, "Architecture": train_cfg[model_name]["architecture"],
                           "Training_Samples": int(training_count), "Reference_Samples": len(reference_x),
                           "Epochs": 8 if model_name != "OCSVM" else 0, "Batch_Size": BATCH if model_name != "OCSVM" else None,
                           "Learning_Rate": 1e-3 if model_name != "OCSVM" else None,
                           "Optimizer": train_cfg[model_name]["optimizer"], "Training_Loss": train_cfg[model_name]["loss"],
                           "Final_Training_Loss": None if loss is None else float(loss),
                           "Training_Time_Seconds": model_elapsed, "Device": str(impl.DEVICE if model_name != "OCSVM" else "CPU"),
                           "Checkpoint": str(model_path), **{k: v for k, v in train_cfg[model_name].items()
                                                              if k not in ("architecture", "loss", "optimizer")}})
        log(f"Completed {name}; elapsed {model_elapsed:.2f}s; thresholds {cuts}")
        if model_name not in ("OCSVM", "Deep SVDD"):
            del model, scorer_only
        elif model_name == "Deep SVDD":
            del model, center, scorer
        else:
            del model, scorer
        del reference_scores, val_metrics, zero_metrics, per_categories
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    master = pd.DataFrame(all_results).sort_values(["Model", "Percentile"])
    thresholds_df = pd.DataFrame(threshold_rows)
    cats_df = pd.DataFrame(category_rows).sort_values(["Attack_Category", "Model", "Percentile"])
    master.to_csv(OUT / "master_results.csv", index=False)
    thresholds_df.to_csv(OUT / "threshold_results.csv", index=False)
    cats_df.to_csv(OUT / "zero_day_category_results.csv", index=False)
    # Descriptive best points are post-evaluation summaries. They never affect
    # model fitting or benign-only calibration thresholds.
    summary = []
    for model_name in MODELS:
        d = master[master.Model == model_name]
        best_f1 = d.loc[d.F1.idxmax()]
        best_zero = d.loc[d.Zero_Day_Detection_Rate.idxmax()]
        best_fpr = d.loc[d.FPR.idxmin()]
        balanced = d.assign(Balanced_Descriptive_Score=(d.F1 * d.Zero_Day_Detection_Rate * d.Specificity).clip(lower=0) ** (1/3))
        best_bal = balanced.loc[balanced.Balanced_Descriptive_Score.idxmax()]
        summary.append({"Model": model_name, "Best_F1_Percentile": int(best_f1.Percentile), "Best_F1": best_f1.F1,
                        "Best_Zero_Day_Percentile": int(best_zero.Percentile), "Best_Zero_Day_Rate": best_zero.Zero_Day_Detection_Rate,
                        "Lowest_FPR_Percentile": int(best_fpr.Percentile), "Lowest_FPR": best_fpr.FPR,
                        "Best_Balanced_Descriptive_Percentile": int(best_bal.Percentile),
                        "Best_Balanced_Descriptive_Score": best_bal.Balanced_Descriptive_Score})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "best_operating_points.csv", index=False)
    pd.DataFrame(model_rows).merge(summary_df, on="Model", how="left").to_csv(OUT / "model_summary.csv", index=False)

    sanity = []
    for row in master.to_dict(orient="records"):
        ok_pos = row["TP"] + row["FN"] == row["Evaluation_Positive_Samples"]
        ok_neg = row["TN"] + row["FP"] == row["Evaluation_Negative_Samples"]
        ok_total = row["TP"] + row["TN"] + row["FP"] + row["FN"] == row["Evaluation_Samples"]
        calc = metrics(row["TP"], row["TN"], row["FP"], row["FN"])
        for m in ("Accuracy", "Precision", "Recall", "F1", "FPR", "Specificity"):
            key = m.lower()
            if key == "fpr" or key == "f1": pass
            if not math.isclose(float(row[m]), float(calc[key]), rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(f"Metric mismatch {row['Model']} {row['Percentile']} {m}")
        if not (ok_pos and ok_neg and ok_total):
            raise AssertionError(f"Confusion-count invariant failed for {row['Model']} {row['Percentile']}")
        if row["Zero_Day_Detected"] + row["Zero_Day_Missed"] != row["Zero_Day_Samples"]:
            raise AssertionError("Zero-day total invariant failed.")
        sanity.append({"Model": row["Model"], "Percentile": row["Percentile"], "positive_sum_ok": ok_pos,
                       "negative_sum_ok": ok_neg, "total_sum_ok": ok_total, "metrics_recomputed_ok": True,
                       "zero_day_sum_ok": True})
    if len(master) != 30 or master[["Threshold", "Accuracy", "Precision", "Recall", "F1", "FPR", "Specificity", "TPR", "Zero_Day_Detection_Rate"]].isna().any().any():
        raise AssertionError("Master result count or required numeric field validation failed.")
    if len(cats_df) == 0 or (cats_df.Samples <= 0).any() or (cats_df.Detected + cats_df.Missed != cats_df.Samples).any():
        raise AssertionError("Zero-day category count validation failed.")
    pd.DataFrame(sanity).to_csv(OUT / "sanity_checks.csv", index=False)
    make_figures(master, thresholds_df, cats_df)

    config = {
        "experiment": "ZeroShield AI six-model percentile comparison",
        "start_time_utc": start.isoformat(), "project_root": str(ROOT), "experiment_directory": str(OUT),
        "datasets": {"benign_training_source": str(TRAIN), "validation_evaluation": str(VALID),
                     "zero_day_evaluation": str(ZERO), "training_source_rows_scanned": train_source_rows,
                     "training_rows": TRAIN_ROWS, "benign_reference_rows": REFERENCE_ROWS,
                     "validation_rows": int(master.Validation_Rows.iloc[0]), "zero_day_rows": int(master.Zero_Day_Rows.iloc[0])},
        "features": features, "raw_feature_count": None,
        "preprocessing": {"feature_representation": "Existing 43-column prepared/scaled CSV values; no scaler was fitted or refitted.",
                          "scaling_documentation": "Project documentation identifies StandardScaler-prepared data; no fitted scaler artifact or raw transformation script exists in this repository, so fit-source details are unavailable.",
                          "cleaning": "Replace +/- infinity with NaN, fill NaN with 0, cast features to float32.",
                          "label_rule": "Use numeric binary_label; where missing, label == BenignTraffic maps to 0 and every other label maps to 1."},
        "seed": SEED, "random_sample_method": "Uniform fixed-size sample by independent seeded random priorities; disjoint random permutation split.",
        "percentiles": PERCENTILES,
        "threshold_methodology": "For each fitted model, score 100,000 disjoint benign reference rows not used for fitting; use NumPy percentile (linear default) at each requested percentile; freeze cuts and apply unchanged to validation and zero-day files. No labels or attack scores determine thresholds.",
        "evaluation_methodology": "Stream validation_set.csv and zero_day_test_scaled_v2.csv in chunks of 50,000. Apply all five frozen thresholds per score pass. Positive means anomaly score >= threshold.",
        "sample_sizes": {"common benign sampled pool": POOL_SIZE, "neural and Deep SVDD training": TRAIN_ROWS,
                         "threshold reference": REFERENCE_ROWS, "OCSVM fit subset": 10_000,
                         "validation": int(master.Validation_Rows.iloc[0]), "zero_day": int(master.Zero_Day_Rows.iloc[0])},
        "model_configurations": {**train_cfg, "AE": {**train_cfg["AE"], "hidden_widths": [64, 32], "latent_dim": 16,
                                                        "activation": "ReLU", "epochs": 8, "batch_size": BATCH, "learning_rate": 1e-3,
                                                        "early_stopping": False, "weight_decay": 0},
                                  "VAE": {**train_cfg["VAE"], "hidden_widths": [64, 32], "latent_dim": 16,
                                          "activation": "ReLU", "epochs": 8, "batch_size": BATCH, "learning_rate": 1e-3,
                                          "kl_weight": 0.01, "early_stopping": False},
                                  "DAE": {**train_cfg["DAE"], "hidden_widths": [64, 32], "latent_dim": 16,
                                          "activation": "ReLU", "noise_distribution": "Gaussian", "noise_std": 0.05,
                                          "epochs": 8, "batch_size": BATCH, "learning_rate": 1e-3},
                                  "CAE": {**train_cfg["CAE"], "latent_dim": 16, "contractive_lambda": 1e-4,
                                          "epochs": 8, "batch_size": BATCH, "learning_rate": 1e-3},
                                  "Deep SVDD": {**train_cfg["Deep SVDD"], "latent_dim": 16, "epochs": 8,
                                                "batch_size": BATCH, "learning_rate": 1e-3, "weight_decay": 1e-6,
                                                "center_rule": "mean encoder output on training benign subset; abs values below 1e-3 clamped with sign"}},
        "thresholds": threshold_rows, "training_records": training_records,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__, "device": str(impl.DEVICE),
                        "cuda_available": torch.cuda.is_available(), "cpu": platform.processor(), "logical_cpus": os.cpu_count()},
        "sanity_checks": {"model_percentile_rows": len(master), "category_rows": len(cats_df), "all_checks_passed": True},
        "warnings": warnings,
    }
    (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    env = [f"Python: {platform.python_version()}", f"PyTorch: {torch.__version__}", f"scikit-learn: {sklearn.__version__}",
           f"pandas: {pd.__version__}", f"NumPy: {np.__version__}", f"Matplotlib: {plt.matplotlib.__version__}",
           f"Device: {impl.DEVICE}", f"CPU: {platform.processor() or 'not reported'}", f"Logical CPUs: {os.cpu_count()}",
           f"CUDA available: {torch.cuda.is_available()}"]
    (OUT / "environment_info.txt").write_text("\n".join(env) + "\n", encoding="utf-8")
    log("Numerical checks passed for all 30 model-percentile combinations.")
    log(f"Category rows: {len(cats_df):,}; categories: {sorted(cats_df.Attack_Category.unique().tolist())}")
    log(f"Finished UTC: {datetime.now(timezone.utc).isoformat()}")
    (OUT / "experiment_log.txt").write_text("\n".join(log_lines + ["\nModel training records:", json.dumps(training_records, indent=2),
                                                       "\nWarnings:", json.dumps(warnings, indent=2)]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("EXPERIMENT FAILED:\n" + traceback.format_exc())
        log_lines.append(f"End time (UTC): {datetime.now(timezone.utc).isoformat()}")
        (OUT / "experiment_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
        raise
