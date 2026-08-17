"""User-driven fine-tuning job (learning mode).

run_training_job(store, artifacts_dir, anchor_dir, versions_dir, progress_cb):

  1. Pull the user's uploaded samples from the SampleStore (normal -> direct
     class, relay -> relay class), split them deterministically 70/30 into
     train / holdout (with <10 samples everything trains and metrics are
     flagged in-sample).
  2. Fine-tune the shipped CNN (artifacts/cnn.pt) on anchor-train + user-train
     windows (2 s / 1 s hop) at LR 1e-4 for up to 3 epochs with BCE
     pos_weight; early-stop on anchor-dev sample-level AUC. Rehearsal on the
     anchor subset guards against catastrophic forgetting.
  3. Retrain LightGBM fresh on handcrafted features of the same windows.
  4. Report before-vs-after metrics: anchor-dev AUC + hit@2%FPR and user
     holdout per-label accuracy (mean-of-windows fused model score, the same
     eval style as eval/evaluate.py).
  5. Save versions/v{n}/ (cnn.pt, lgbm.txt, feature_scaler.joblib,
     training_report.json; fuser/calibrators/thresholds copied forward
     unchanged) and update versions/current.json.

Designed to finish well under 8 minutes on 2 CPUs: the anchor rehearsal set is
small (~350 files) and the window budget is capped (deterministic subsample).
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from relayguard.common import TARGET_SR, iter_windows, load_audio, load_audio_bytes

USER_SPLIT_SEED = 12345
WINDOW_CAP = 25000
CNN_LR = 1e-4
CNN_EPOCHS = 3
CNN_BATCH = 16
# fusion fallback weights for the two window models (configs/default.yaml),
# renormalized to a combined "window-model" score used for before/after eval.
CNN_W, GBM_W = 0.35, 0.25

# artifacts that must exist in a version dir for it to be activatable
VERSION_MODEL_FILES = ("cnn.pt", "lgbm.txt", "feature_scaler.joblib")
# artifacts carried forward unchanged from the base artifacts dir
FORWARD_FILES = ("fuser.joblib", "calibrators.joblib", "fusion_thresholds.json")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cb(progress_cb, stage: str, frac: float) -> None:
    try:
        progress_cb(stage, max(0.0, min(1.0, frac)))
    except Exception:
        pass


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _windows_of(audio: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(w, dtype=np.float32) for w in iter_windows(audio)]


def _user_split(samples: list[dict]) -> tuple[list[dict], list[dict], bool]:
    """Deterministic 70/30 train/holdout split. With <10 user samples every
    sample is used for training and the same set doubles as the (in-sample)
    evaluation set."""
    samples = sorted(samples, key=lambda s: s.get("id", ""))
    if len(samples) < 10:
        return samples, samples, True
    rng = np.random.default_rng(USER_SPLIT_SEED)
    order = rng.permutation(len(samples))
    n_train = max(1, int(round(0.7 * len(samples))))
    train = [samples[i] for i in sorted(order[:n_train])]
    holdout = [samples[i] for i in sorted(order[n_train:])]
    if not holdout:  # keep at least one holdout sample when possible
        train, holdout = samples[:-1], samples[-1:]
    return train, holdout, False


# ---------------------------------------------------------------------------
# scoring helpers (before/after eval)
# ---------------------------------------------------------------------------

def _cnn_probs(model, windows: list[np.ndarray], batch: int = 64) -> np.ndarray:
    import torch

    from relayguard.models.cnn import logmel

    if not windows:
        return np.zeros(0)
    model.eval()
    out = np.zeros(len(windows), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            wavs = torch.from_numpy(np.stack(windows[i : i + batch]))
            out[i : i + len(wavs)] = torch.sigmoid(model(logmel(wavs))).numpy()
    return out


def _gbm_probs(booster, scaler, windows: list[np.ndarray]) -> np.ndarray:
    from relayguard.features import extract_batch

    if not windows:
        return np.zeros(0)
    feats = extract_batch(windows)
    if scaler is not None:
        feats = scaler.transform(feats)
    return np.asarray(
        booster.predict(feats, num_iteration=booster.best_iteration or None),
        dtype=np.float64,
    )


def _fused_scores(cnn_p: np.ndarray, gbm_p: np.ndarray) -> np.ndarray:
    return (CNN_W * cnn_p + GBM_W * gbm_p) / (CNN_W + GBM_W)


def _sample_scores(model, booster, scaler, entries: list[dict]) -> np.ndarray:
    """Mean-of-windows fused model score per entry (eval/evaluate.py style)."""
    per_entry: list[float] = []
    for e in entries:
        wins = e["windows"]
        if not wins:
            per_entry.append(0.0)
            continue
        cnn_p = _cnn_probs(model, wins) if model is not None else np.zeros(len(wins))
        gbm_p = _gbm_probs(booster, scaler, wins) if booster is not None else np.zeros(len(wins))
        per_entry.append(float(_fused_scores(cnn_p, gbm_p).mean()))
    return np.asarray(per_entry)


def _cls_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    from relayguard.eval.evaluate import compute_metrics

    m = compute_metrics(np.asarray(labels), np.asarray(scores, dtype=np.float64))
    return {"auc": m["auc"], "hit_at_fpr_2": m["hit_at_fpr_2"], "eer": m["eer"]}


def _per_label_accuracy(labels: list[int], scores: np.ndarray) -> dict:
    """Accuracy at the 0.5 fused-score midpoint, broken out per class."""
    out: dict = {}
    names = {0: "normal", 1: "relay"}
    for lab in (0, 1):
        idx = [i for i, y in enumerate(labels) if y == lab]
        if not idx:
            continue
        correct = sum(1 for i in idx if (scores[i] >= 0.5) == bool(lab))
        out[names[lab]] = {
            "n": len(idx),
            "accuracy": round(correct / len(idx), 4),
        }
    all_correct = sum(1 for i, y in enumerate(labels) if (scores[i] >= 0.5) == bool(y))
    out["overall"] = {
        "n": len(labels),
        "accuracy": round(all_correct / max(len(labels), 1), 4),
    }
    return out


# ---------------------------------------------------------------------------
# version bookkeeping
# ---------------------------------------------------------------------------

def next_version(versions_dir: str | Path) -> int:
    vdir = Path(versions_dir)
    n = 0
    for p in vdir.glob("v*"):
        if p.is_dir() and p.name[1:].isdigit():
            n = max(n, int(p.name[1:]))
    return n + 1


def read_current(versions_dir: str | Path) -> int:
    """Active version number; 0 = the base artifacts shipped with the app."""
    try:
        cur = json.loads((Path(versions_dir) / "current.json").read_text())
        return int(cur.get("active_version", 0))
    except Exception:
        return 0


def write_current(versions_dir: str | Path, version: int) -> None:
    vdir = Path(versions_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "current.json").write_text(
        json.dumps({"active_version": int(version), "updated_at": _iso_now()}, indent=2)
    )


def version_dir_for(versions_dir: str | Path, artifacts_dir: str | Path, version: int) -> Path:
    """Directory an active version's artifacts live in (0 -> base artifacts)."""
    if int(version) == 0:
        return Path(artifacts_dir)
    return Path(versions_dir) / f"v{int(version)}"


def list_versions(versions_dir: str | Path) -> list[dict]:
    vdir = Path(versions_dir)
    out = []
    for p in sorted(vdir.glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1):
        if not (p.is_dir() and p.name[1:].isdigit()):
            continue
        entry: dict = {"version": int(p.name[1:]), "path": str(p)}
        rep = p / "training_report.json"
        if rep.exists():
            try:
                r = json.loads(rep.read_text())
                entry["created_at"] = r.get("created_at")
                entry["user_samples"] = r.get("user_samples")
            except Exception:
                pass
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------

def run_training_job(store, artifacts_dir, anchor_dir, versions_dir, progress_cb) -> dict:
    """Execute one fine-tuning job. Returns the training report dict; raises
    on failure (the caller records the error in the job state)."""
    import joblib
    import lightgbm as lgb
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    from relayguard.features import FEATURE_NAMES, extract_batch
    from relayguard.models.cnn import RelayCNN, logmel

    torch.set_num_threads(2)
    artifacts_dir = Path(artifacts_dir)
    anchor_dir = Path(anchor_dir)
    versions_dir = Path(versions_dir)
    t_start = time.time()
    warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. user samples
    # ------------------------------------------------------------------ #
    _cb(progress_cb, "loading user samples", 0.02)
    rows = store.list_samples()
    user_entries: list[dict] = []
    for row in rows:
        try:
            audio = load_audio_bytes(store.read_audio(row), fmt="wav")
        except Exception as exc:
            warnings.append(f"sample {row.get('id', '?')[:8]} unreadable: {exc}")
            continue
        user_entries.append(
            {
                "id": row.get("id"),
                "label": 1 if row.get("label") == "relay" else 0,
                "windows": _windows_of(audio),
                "duration_s": row.get("duration_s"),
            }
        )
    n_user = len(user_entries)
    n_user_relay = sum(e["label"] for e in user_entries)
    if n_user < 4:
        warnings.append(
            f"only {n_user} usable user samples (<4): fine-tuning mostly on the "
            "anchor rehearsal set; upload more of your own audio for a stronger effect"
        )

    user_train, user_hold, in_sample = _user_split(user_entries)

    # ------------------------------------------------------------------ #
    # 2. anchor rehearsal windows
    # ------------------------------------------------------------------ #
    _cb(progress_cb, "loading anchor rehearsal data", 0.08)
    anchor_train_meta = _load_jsonl(anchor_dir / "anchor_metadata.jsonl")
    anchor_dev_meta = _load_jsonl(anchor_dir / "dev_metadata.jsonl")
    if not anchor_train_meta:
        raise RuntimeError(f"anchor data missing at {anchor_dir}")

    anchor_train: list[dict] = []
    for rec in anchor_train_meta:
        try:
            audio = load_audio(anchor_dir / rec["file"])
        except Exception as exc:
            warnings.append(f"anchor file {rec.get('file')} unreadable: {exc}")
            continue
        anchor_train.append(
            {"label": 1 if rec["label"] == "relay" else 0, "windows": _windows_of(audio)}
        )
    anchor_dev: list[dict] = []
    for rec in anchor_dev_meta:
        try:
            audio = load_audio(anchor_dir / rec["file"])
        except Exception:
            continue
        anchor_dev.append(
            {"label": 1 if rec["label"] == "relay" else 0, "windows": _windows_of(audio)}
        )

    # flat window pools -------------------------------------------------- #
    train_windows: list[np.ndarray] = []
    train_labels: list[int] = []
    train_is_user: list[bool] = []
    for e in anchor_train:
        train_windows.extend(e["windows"])
        train_labels.extend([e["label"]] * len(e["windows"]))
        train_is_user.extend([False] * len(e["windows"]))
    for e in user_train:
        train_windows.extend(e["windows"])
        train_labels.extend([e["label"]] * len(e["windows"]))
        train_is_user.extend([True] * len(e["windows"]))

    n_total = len(train_windows)
    if n_total > WINDOW_CAP:
        # deterministic cap: keep every user window, subsample anchor windows
        rng = np.random.default_rng(USER_SPLIT_SEED)
        anchor_idx = [i for i, u in enumerate(train_is_user) if not u]
        user_idx = [i for i, u in enumerate(train_is_user) if u]
        keep_anchor = sorted(rng.choice(anchor_idx, WINDOW_CAP - len(user_idx), replace=False).tolist())
        keep = sorted(user_idx + keep_anchor)
        train_windows = [train_windows[i] for i in keep]
        train_labels = [train_labels[i] for i in keep]
        warnings.append(f"window cap applied: {n_total} -> {WINDOW_CAP} windows")

    y_train = np.asarray(train_labels, dtype=np.int64)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos

    # ------------------------------------------------------------------ #
    # 3. CNN fine-tune (early stop on anchor-dev AUC)
    # ------------------------------------------------------------------ #
    base_cnn = RelayCNN()
    base_cnn.load_state_dict(torch.load(artifacts_dir / "cnn.pt", map_location="cpu", weights_only=True))
    base_cnn.eval()

    base_gbm = lgb.Booster(model_file=str(artifacts_dir / "lgbm.txt"))
    base_scaler = joblib.load(artifacts_dir / "feature_scaler.joblib")

    dev_windows = [w for e in anchor_dev for w in e["windows"]]
    dev_win_labels = np.asarray([e["label"] for e in anchor_dev for _ in e["windows"]])
    dev_entry_bounds = np.cumsum([len(e["windows"]) for e in anchor_dev])

    def dev_auc(model) -> float | None:
        from sklearn.metrics import roc_auc_score

        if not dev_windows or len(np.unique(dev_win_labels)) < 2:
            return None
        probs = _cnn_probs(model, dev_windows)
        # sample-level aggregation (mean of windows), like models/train.py
        sample_scores, prev = [], 0
        for b in dev_entry_bounds:
            sample_scores.append(float(probs[prev:b].mean()))
            prev = b
        y = [e["label"] for e in anchor_dev]
        return float(roc_auc_score(y, sample_scores)) if len(set(y)) > 1 else None

    model = RelayCNN()
    model.load_state_dict(base_cnn.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 1e-3)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(USER_SPLIT_SEED)

    best_auc, best_state = -1.0, None
    n_items = len(train_windows)
    train_stack = np.stack(train_windows)  # (N, 32000) float32, built once
    for epoch in range(CNN_EPOCHS):
        model.train()
        order = rng.permutation(n_items)
        for i in range(0, n_items, CNN_BATCH):
            idx = order[i : i + CNN_BATCH]
            xb = torch.from_numpy(train_stack[idx])
            yb = torch.from_numpy(y_train[idx].astype(np.float32))
            opt.zero_grad()
            loss = loss_fn(model(logmel(xb)), yb)
            loss.backward()
            opt.step()
        auc = dev_auc(model)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        _cb(progress_cb, f"fine-tuning CNN (epoch {epoch + 1}/{CNN_EPOCHS}"
                         + (f", anchor-dev AUC {auc:.4f}" if auc is not None else ")"),
            0.15 + 0.35 * (epoch + 1) / CNN_EPOCHS)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ------------------------------------------------------------------ #
    # 4. LightGBM retrain
    # ------------------------------------------------------------------ #
    _cb(progress_cb, "extracting features for GBM", 0.55)
    X_train = extract_batch(train_windows)
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)

    wtr = np.where(y_train == 1, len(y_train) / (2.0 * max(n_pos, 1)),
                   len(y_train) / (2.0 * max(n_neg, 1)))
    dtrain = lgb.Dataset(Xtr_s, label=y_train, weight=wtr,
                         feature_name=list(FEATURE_NAMES))
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
              "num_leaves": 31, "min_data_in_leaf": 10, "feature_fraction": 0.9,
              "bagging_fraction": 0.9, "bagging_freq": 1, "seed": USER_SPLIT_SEED,
              "num_threads": 2, "verbose": -1}
    callbacks = [lgb.log_evaluation(period=0)]
    valid = None
    num_rounds = 300
    _cb(progress_cb, "training GBM", 0.70)
    if dev_windows and len(np.unique(dev_win_labels)) > 1:
        X_dev = scaler.transform(extract_batch(dev_windows))
        wdv = np.where(dev_win_labels == 1,
                       len(dev_win_labels) / (2.0 * max(int(dev_win_labels.sum()), 1)),
                       len(dev_win_labels) / (2.0 * max(len(dev_win_labels) - int(dev_win_labels.sum()), 1)))
        valid = lgb.Dataset(X_dev, label=dev_win_labels, weight=wdv, reference=dtrain)
        callbacks.append(lgb.early_stopping(50, verbose=False))
        num_rounds = 500
    gbm = lgb.train(params, dtrain, num_boost_round=num_rounds,
                    valid_sets=[valid] if valid else None, callbacks=callbacks)

    # ------------------------------------------------------------------ #
    # 5. before / after metrics
    # ------------------------------------------------------------------ #
    _cb(progress_cb, "evaluating before/after", 0.85)
    anchor_labels = [e["label"] for e in anchor_dev]
    before_anchor = _sample_scores(base_cnn, base_gbm, base_scaler, anchor_dev)
    after_anchor = _sample_scores(model, gbm, scaler, anchor_dev)

    hold_labels = [e["label"] for e in user_hold]
    before_user = _sample_scores(base_cnn, base_gbm, base_scaler, user_hold)
    after_user = _sample_scores(model, gbm, scaler, user_hold)

    report = {
        "created_at": _iso_now(),
        "duration_s": None,  # filled below
        "user_samples": {
            "total": n_user,
            "relay": n_user_relay,
            "normal": n_user - n_user_relay,
            "train": len(user_train),
            "holdout": len(user_hold),
            "in_sample_metrics": bool(in_sample),
        },
        "anchor": {"train_files": len(anchor_train), "dev_files": len(anchor_dev)},
        "train_windows": int(len(train_windows)),
        "cnn": {"lr": CNN_LR, "epochs": CNN_EPOCHS, "batch": CNN_BATCH,
                "best_anchor_dev_auc": best_auc if best_auc >= 0 else None},
        "gbm": {"trees": gbm.best_iteration or gbm.num_trees()},
        "metrics": {
            "anchor_dev": {
                "n": len(anchor_labels),
                "before": _cls_metrics(anchor_labels, before_anchor),
                "after": _cls_metrics(anchor_labels, after_anchor),
            },
            "user_holdout": {
                "n": len(hold_labels),
                "in_sample": bool(in_sample),
                "before": _per_label_accuracy(hold_labels, before_user),
                "after": _per_label_accuracy(hold_labels, after_user),
            },
        },
        "warnings": warnings,
    }

    # ------------------------------------------------------------------ #
    # 6. save version
    # ------------------------------------------------------------------ #
    _cb(progress_cb, "saving model version", 0.95)
    versions_dir.mkdir(parents=True, exist_ok=True)
    n = next_version(versions_dir)
    vdir = versions_dir / f"v{n}"
    vdir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), vdir / "cnn.pt")
    # Keep the torch-free runtime able to serve this version: export the
    # wav->logit ONNX next to cnn.pt (best effort; the torch fallback in
    # CNNDetector covers environments where the export itself fails).
    try:
        from relayguard.models.export_onnx import export_onnx

        export_onnx(vdir / "cnn.pt", vdir / "cnn.onnx")
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"onnx export skipped: {exc}")
    gbm.save_model(str(vdir / "lgbm.txt"))
    joblib.dump(scaler, vdir / "feature_scaler.joblib")
    for fname in FORWARD_FILES:
        src = artifacts_dir / fname
        if src.exists():
            shutil.copy2(src, vdir / fname)
    report["duration_s"] = round(time.time() - t_start, 1)
    report["version"] = n
    (vdir / "training_report.json").write_text(json.dumps(report, indent=2))
    write_current(versions_dir, n)
    _cb(progress_cb, "done", 1.0)
    return report
