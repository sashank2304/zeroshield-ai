from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# ZeroShield AI - Tuned Deep SVDD
# ============================================================
# Valid design:
#   train_benign_only.csv -> training only
#   validation_set.csv    -> hyperparameter + threshold tuning
#   zero_day_test_....csv -> final held-out test ONLY
#
# Selection target:
#   highest validation attack recall subject to benign FPR <= 5%
#   (F1, then precision, break ties)
#
# The zero-day set is NEVER used to choose architecture,
# hyperparameters, or threshold.
# ============================================================

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train_benign_only.csv"
VALID = ROOT / "validation_set.csv"
ZERO = ROOT / "zero_day_test_scaled_v2.csv"
OUT = ROOT / "model_training" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory-safe settings
TRAIN_CHUNK = 100_000
EVAL_CHUNK = 50_000
TUNE_TRAIN_ROWS = 250_000
TUNE_VALID_BENIGN = 100_000
TUNE_VALID_ATTACK = 100_000

# Search space: small enough for a student laptop, but meaningful.
CANDIDATES = [
    {"name": "SVDD_small", "hidden": [64, 32], "latent": 8, "lr_pre": 1e-3, "lr_svdd": 1e-3, "weight_decay": 1e-6},
    {"name": "SVDD_medium", "hidden": [128, 64], "latent": 16, "lr_pre": 5e-4, "lr_svdd": 5e-4, "weight_decay": 1e-6},
    {"name": "SVDD_wide", "hidden": [128, 64], "latent": 32, "lr_pre": 5e-4, "lr_svdd": 3e-4, "weight_decay": 1e-6},
]

THRESHOLD_PERCENTILES = [90, 92, 94, 95, 96, 97, 98, 99]
MAX_FPR = 0.05
TUNE_BATCH = 4096
FINAL_BATCH = 4096
TUNE_PRE_EPOCHS = 3
TUNE_SVDD_EPOCHS = 6
FINAL_PRE_EPOCHS = 5
FINAL_SVDD_EPOCHS = 12

SAMPLE_FILE = OUT / "deep_svdd_tuning_benign_250k.npy"
VALID_TUNE_FILE = OUT / "deep_svdd_validation_tuning.npz"
FEATURE_FILE = OUT / "zero_shield_feature_columns.json"
TUNING_FILE = OUT / "Deep_SVDD_tuning_candidates.csv"
THRESHOLD_FILE = OUT / "Deep_SVDD_selected_thresholds.csv"
MODEL_FILE = OUT / "Deep_SVDD_Tuned.pt"
RESULT_FILE = OUT / "Deep_SVDD_Tuned_results.json"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def section(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def get_features():
    if FEATURE_FILE.exists():
        return json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
    h = pd.read_csv(TRAIN, nrows=0)
    features = [c for c in h.columns if c not in {"label", "binary_label"}]
    FEATURE_FILE.write_text(json.dumps(features, indent=2), encoding="utf-8")
    return features


def clean_x(df, features):
    return (df[features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype(np.float32))


def clean_y(chunk):
    y = pd.to_numeric(chunk["binary_label"], errors="coerce")
    missing = y.isna()
    if missing.any():
        if "label" not in chunk.columns:
            raise ValueError("binary_label contains missing values and label is unavailable.")
        y.loc[missing] = (
            chunk.loc[missing, "label"].astype(str).str.strip().ne("BenignTraffic").astype(np.int8).to_numpy()
        )
    if y.isna().any():
        raise ValueError("Could not reconstruct all missing binary labels.")
    y = y.astype(np.int8).to_numpy()
    bad = np.setdiff1d(np.unique(y), np.array([0, 1], dtype=np.int8))
    if len(bad):
        raise ValueError(f"Unexpected binary label values: {bad.tolist()}")
    return y


def build_training_sample(features):
    if SAMPLE_FILE.exists():
        x = np.load(SAMPLE_FILE).astype(np.float32, copy=False)
        print(f"Using cached benign tuning sample: {len(x):,} rows")
        return x

    print(f"Creating benign tuning sample ({TUNE_TRAIN_ROWS:,} rows)...")
    rng = np.random.default_rng(SEED)
    reservoir = np.empty((TUNE_TRAIN_ROWS, len(features)), dtype=np.float32)
    seen = 0
    for chunk in pd.read_csv(TRAIN, usecols=features, chunksize=TRAIN_CHUNK, low_memory=True):
        arr = (chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
               .to_numpy(dtype=np.float32, copy=False))
        for row in arr:
            seen += 1
            if seen <= TUNE_TRAIN_ROWS:
                reservoir[seen - 1] = row
            else:
                j = rng.integers(0, seen)
                if j < TUNE_TRAIN_ROWS:
                    reservoir[j] = row
        if seen >= TUNE_TRAIN_ROWS:
            # Reservoir is full; continue scanning is unnecessary because the first
            # 250k rows have now already been randomized by reservoir sampling only
            # if we continue. So keep scanning to make it genuinely representative.
            pass
        if seen % 1_000_000 == 0:
            print(f"  scanned {seen:,} training rows...")
    np.save(SAMPLE_FILE, reservoir[:min(seen, TUNE_TRAIN_ROWS)])
    print(f"Saved {SAMPLE_FILE}")
    return reservoir[:min(seen, TUNE_TRAIN_ROWS)]


def build_validation_tuning_subset(features):
    if VALID_TUNE_FILE.exists():
        z = np.load(VALID_TUNE_FILE)
        print("Using cached balanced validation tuning subset.")
        return z["xb"], z["xa"]

    print("Scanning validation file to build a balanced tuning subset...")
    rng = np.random.default_rng(SEED + 1)
    benign_parts = []
    attack_parts = []
    benign_left = TUNE_VALID_BENIGN
    attack_left = TUNE_VALID_ATTACK
    processed = 0

    usecols = features + ["binary_label", "label"]
    for chunk in pd.read_csv(VALID, usecols=usecols, chunksize=EVAL_CHUNK, low_memory=True):
        y = clean_y(chunk)
        x = clean_x(chunk, features).to_numpy(dtype=np.float32, copy=False)

        for cls, left, parts in [(0, benign_left, benign_parts), (1, attack_left, attack_parts)]:
            if left <= 0:
                continue
            idx = np.flatnonzero(y == cls)
            if len(idx):
                take = min(left, len(idx))
                chosen = rng.choice(idx, size=take, replace=False)
                parts.append(x[chosen])
                if cls == 0:
                    benign_left -= take
                else:
                    attack_left -= take

        processed += len(chunk)
        if processed % 1_000_000 == 0:
            print(f"  scanned {processed:,} validation rows...")
        if benign_left <= 0 and attack_left <= 0:
            break

    xb = np.concatenate(benign_parts, axis=0)
    xa = np.concatenate(attack_parts, axis=0)
    np.savez(VALID_TUNE_FILE, xb=xb, xa=xa)
    print(f"Saved balanced tuning subset: {len(xb):,} benign + {len(xa):,} attack")
    return xb, xa


class PretrainAE(nn.Module):
    def __init__(self, d, hidden, latent):
        super().__init__()
        enc = []
        p = d
        for h in hidden:
            enc += [nn.Linear(p, h), nn.ReLU()]
            p = h
        enc += [nn.Linear(p, latent)]
        self.encoder = nn.Sequential(*enc)

        dec = []
        p = latent
        for h in reversed(hidden):
            dec += [nn.Linear(p, h), nn.ReLU()]
            p = h
        dec += [nn.Linear(p, d)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DeepSVDD(nn.Module):
    def __init__(self, d, hidden, latent):
        super().__init__()
        layers = []
        p = d
        for h in hidden:
            layers += [nn.Linear(p, h, bias=False), nn.ReLU()]
            p = h
        layers += [nn.Linear(p, latent, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def loader(x, batch):
    return DataLoader(
        TensorDataset(torch.from_numpy(x.astype(np.float32, copy=False))),
        batch_size=batch, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


def pretrain(cfg, x, epochs, batch):
    model = PretrainAE(x.shape[1], cfg["hidden"], cfg["latent"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr_pre"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    model.train()
    final_loss = 0.0
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for (b,) in loader(x, batch):
            b = b.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(b), b)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        final_loss = total / max(1, n)
        print(f"  Pretrain epoch {ep:02d}/{epochs} | loss={final_loss:.6f}")
    return model, final_loss


@torch.no_grad()
def center_from_encoder(encoder, x, batch):
    encoder.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32, copy=False))),
                    batch_size=batch, shuffle=False, num_workers=0)
    total = None
    count = 0
    for (b,) in dl:
        z = encoder(b.to(DEVICE, non_blocking=True))
        s = z.sum(0)
        total = s if total is None else total + s
        count += len(b)
    c = (total / count).detach()
    eps = 1e-4
    c = torch.where(c.abs() < eps, torch.where(c >= 0, torch.full_like(c, eps), torch.full_like(c, -eps)), c)
    return c


def init_svdd_from_ae(ae, cfg):
    svdd = DeepSVDD(ae.encoder[0].in_features, cfg["hidden"], cfg["latent"]).to(DEVICE)
    ae_linear = [m for m in ae.encoder if isinstance(m, nn.Linear)]
    svdd_linear = [m for m in svdd.net if isinstance(m, nn.Linear)]
    if len(ae_linear) != len(svdd_linear):
        raise RuntimeError("Pretraining encoder and SVDD network do not match.")
    with torch.no_grad():
        for a, s in zip(ae_linear, svdd_linear):
            s.weight.copy_(a.weight)
    return svdd


def train_svdd(model, center, x, cfg, epochs, batch):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr_svdd"], weight_decay=cfg["weight_decay"])
    model.train()
    final_loss = 0.0
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for (b,) in loader(x, batch):
            b = b.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            z = model(b)
            loss = ((z - center) ** 2).sum(1).mean()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        final_loss = total / max(1, n)
        print(f"  SVDD epoch {ep:02d}/{epochs} | loss={final_loss:.8f}")
    return final_loss


@torch.no_grad()
def score_array(model, center, x, batch):
    model.eval()
    out = []
    for start in range(0, len(x), batch):
        stop = min(start + batch, len(x))
        b = torch.from_numpy(x[start:stop]).to(DEVICE, non_blocking=True)
        z = model(b)
        out.append(((z - center) ** 2).sum(1).cpu().numpy())
    return np.concatenate(out)


def metric_counts(tp, tn, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    fpr = fp / (fp + tn) if fp + tn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "fpr": fpr, "specificity": specificity, "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn)}


def select_threshold(train_scores, benign_scores, attack_scores):
    rows = []
    for p in THRESHOLD_PERCENTILES:
        t = float(np.percentile(train_scores, p))
        bp = (benign_scores >= t).astype(np.int8)
        ap = (attack_scores >= t).astype(np.int8)
        m = metric_counts(int(ap.sum()), int(len(bp) - bp.sum()), int(bp.sum()), int(len(ap) - ap.sum()))
        rows.append({"percentile": p, "threshold": t, **m})

    feasible = [r for r in rows if r["fpr"] is not None and r["fpr"] <= MAX_FPR]
    pool = feasible if feasible else rows
    selected = max(pool, key=lambda r: (r["recall"], r["f1"], r["precision"], -r["fpr"]))
    return selected, rows


def eval_large(path, features, model, center, threshold, category=False):
    tp = tn = fp = fn = 0
    cats = {}
    processed = 0
    usecols = features + ["binary_label"] + (["label"] if category else [])
    for chunk_no, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=EVAL_CHUNK, low_memory=True), 1):
        x_df = clean_x(chunk, features)
        x = x_df.to_numpy(dtype=np.float32, copy=False)
        y = clean_y(chunk)
        scores = score_array(model, center, x, FINAL_BATCH)
        pred = (scores >= threshold).astype(np.int8)

        tp += int(((pred == 1) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())

        if category:
            labels = chunk["label"].astype(str).to_numpy()
            for c in np.unique(labels):
                mask = labels == c
                cy = y[mask]
                cp = pred[mask]
                if c not in cats:
                    cats[c] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
                cats[c]["tp"] += int(((cp == 1) & (cy == 1)).sum())
                cats[c]["tn"] += int(((cp == 0) & (cy == 0)).sum())
                cats[c]["fp"] += int(((cp == 1) & (cy == 0)).sum())
                cats[c]["fn"] += int(((cp == 0) & (cy == 1)).sum())

        processed += len(chunk)
        if chunk_no == 1 or chunk_no % 20 == 0:
            print(f"  processed {processed:,} rows...")

    overall = metric_counts(tp, tn, fp, fn)
    cat_metrics = {c: metric_counts(**v) for c, v in cats.items()}
    return {"rows": processed, "overall": overall, "categories": cat_metrics}


def main():
    section("ZEROSHIELD AI - TUNED DEEP SVDD")
    print(f"Device: {DEVICE}")
    print("Training: benign only")
    print("Selection: validation recall maximized subject to FPR <= 5%")
    print("Zero-day: held out until the final evaluation")

    for p in [TRAIN, VALID, ZERO]:
        if not p.exists():
            raise FileNotFoundError(p)

    features = get_features()
    print(f"Features: {len(features)}")

    x_tune = build_training_sample(features)
    xb, xa = build_validation_tuning_subset(features)

    tuning_rows = []
    best_cfg = None
    best_key = None
    best_threshold_table = None

    for i, cfg in enumerate(CANDIDATES, 1):
        section(f"TUNING CANDIDATE {i}/{len(CANDIDATES)} - {cfg['name']}")
        print(f"hidden={cfg['hidden']}, latent={cfg['latent']}, pre_lr={cfg['lr_pre']}, svdd_lr={cfg['lr_svdd']}")

        ae, pre_loss = pretrain(cfg, x_tune, TUNE_PRE_EPOCHS, TUNE_BATCH)
        svdd = init_svdd_from_ae(ae, cfg)
        center = center_from_encoder(ae.encoder, x_tune, TUNE_BATCH)
        svdd_loss = train_svdd(svdd, center, x_tune, cfg, TUNE_SVDD_EPOCHS, TUNE_BATCH)

        train_scores = score_array(svdd, center, x_tune, TUNE_BATCH)
        benign_scores = score_array(svdd, center, xb, TUNE_BATCH)
        attack_scores = score_array(svdd, center, xa, TUNE_BATCH)
        selected, threshold_rows = select_threshold(train_scores, benign_scores, attack_scores)

        print(f"\nSelected tuning threshold: {selected['percentile']}th percentile")
        print(f"Threshold: {selected['threshold']:.8f}")
        print(f"Recall: {selected['recall'] * 100:.2f}%")
        print(f"F1: {selected['f1'] * 100:.2f}%")
        print(f"FPR: {selected['fpr'] * 100:.2f}%")

        tuning_rows.append({
            "candidate": cfg["name"],
            "hidden": str(cfg["hidden"]),
            "latent": cfg["latent"],
            "pre_lr": cfg["lr_pre"],
            "svdd_lr": cfg["lr_svdd"],
            "weight_decay": cfg["weight_decay"],
            "pretrain_loss": pre_loss,
            "svdd_loss": svdd_loss,
            "selected_percentile": selected["percentile"],
            "threshold": selected["threshold"],
            "validation_precision": selected["precision"],
            "validation_recall": selected["recall"],
            "validation_f1": selected["f1"],
            "validation_fpr": selected["fpr"],
            "validation_specificity": selected["specificity"]
        })

        key = (selected["recall"], selected["f1"], selected["precision"], -selected["fpr"])
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = dict(cfg)
            best_cfg["selected_percentile"] = selected["percentile"]
            best_cfg["tuning_threshold"] = selected["threshold"]
            best_threshold_table = threshold_rows

        del ae, svdd, center, train_scores, benign_scores, attack_scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(tuning_rows).to_csv(TUNING_FILE, index=False)
    pd.DataFrame(best_threshold_table).to_csv(THRESHOLD_FILE, index=False)

    section("FINAL RETRAINING - SELECTED DEEP SVDD")
    print(f"Selected architecture: {best_cfg['name']}")
    print(f"Hidden: {best_cfg['hidden']}")
    print(f"Latent: {best_cfg['latent']}")
    print(f"Chosen threshold percentile: {best_cfg['selected_percentile']}%")

    print("\nLoading all benign training rows...")
    train_df = pd.read_csv(TRAIN, usecols=features, low_memory=True)
    x_final = (train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
               .to_numpy(dtype=np.float32, copy=False))
    del train_df
    gc.collect()
    print(f"Final training matrix: {len(x_final):,} × {len(features)}")

    ae, pre_loss = pretrain(best_cfg, x_final, FINAL_PRE_EPOCHS, FINAL_BATCH)
    final_svdd = init_svdd_from_ae(ae, best_cfg)
    final_center = center_from_encoder(ae.encoder, x_final, FINAL_BATCH)
    svdd_loss = train_svdd(final_svdd, final_center, x_final, best_cfg, FINAL_SVDD_EPOCHS, FINAL_BATCH)

    train_scores = score_array(final_svdd, final_center, x_final, FINAL_BATCH)
    final_threshold = float(np.percentile(train_scores, best_cfg["selected_percentile"]))

    torch.save({
        "state_dict": final_svdd.state_dict(),
        "center": final_center.cpu(),
        "features": features,
        "input_dim": len(features),
        "hidden": best_cfg["hidden"],
        "latent_dim": best_cfg["latent"],
        "threshold": final_threshold,
        "threshold_percentile": best_cfg["selected_percentile"],
        "pretrain_loss": pre_loss,
        "svdd_loss": svdd_loss,
        "seed": SEED
    }, MODEL_FILE)

    print(f"\nFinal model saved to:\n{MODEL_FILE}")
    print(f"Final threshold: {final_threshold:.8f}")

    section("FULL VALIDATION EVALUATION")
    validation = eval_large(VALID, features, final_svdd, final_center, final_threshold, category=False)
    vm = validation["overall"]
    print("\nFINAL VALIDATION RESULTS")
    print(f"Accuracy    : {vm['accuracy'] * 100:.2f}%")
    print(f"Precision   : {vm['precision'] * 100:.2f}%")
    print(f"Recall      : {vm['recall'] * 100:.2f}%")
    print(f"F1-score    : {vm['f1'] * 100:.2f}%")
    print(f"FPR         : {vm['fpr'] * 100:.2f}%")
    print(f"Specificity : {vm['specificity'] * 100:.2f}%")
    print(f"TP={vm['tp']:,} | TN={vm['tn']:,} | FP={vm['fp']:,} | FN={vm['fn']:,}")

    section("FINAL ZERO-DAY EVALUATION")
    zero = eval_large(ZERO, features, final_svdd, final_center, final_threshold, category=True)
    zm = zero["overall"]
    print("\nFINAL ZERO-DAY RESULTS")
    total_attacks = zm["tp"] + zm["fn"]
    print(f"Total attacks : {total_attacks:,}")
    print(f"Detected      : {zm['tp']:,}")
    print(f"Missed        : {zm['fn']:,}")
    print(f"Detection rate: {zm['recall'] * 100:.2f}%")
    print("FPR/Specificity: N/A (zero-day file contains attacks only)")
    print("\nZero-day category detection:")
    for c, m in sorted(zero["categories"].items()):
        total = m["tp"] + m["fn"]
        rate = m["tp"] / total if total else 0.0
        print(f"  {c:25s}: {rate * 100:7.2f}% ({m['tp']:,}/{total:,})")

    payload = {
        "model": "Deep SVDD",
        "selection_rule": "maximize validation recall subject to validation FPR <= 5%; tie-break F1 then precision",
        "selected_configuration": best_cfg,
        "final_threshold": final_threshold,
        "validation": validation,
        "zero_day": zero,
        "model_file": str(MODEL_FILE),
        "tuning_file": str(TUNING_FILE)
    }
    RESULT_FILE.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    print(f"\nResults saved to:\n{RESULT_FILE}")
    print(f"\nTuning table saved to:\n{TUNING_FILE}")
    print(f"Threshold table saved to:\n{THRESHOLD_FILE}")

    del ae, final_svdd, final_center, x_final, train_scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    section("DONE")


if __name__ == "__main__":
    main()
