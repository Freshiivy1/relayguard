"""Training CLI for RelayCNN + LightGBM (SPEC section 4 C2).

Usage:
    python -m relayguard.models.train --data-dir DATA --out-dir artifacts/ \
        --epochs 15 --batch 64 --seed 0
    python -m relayguard.models.train --self-test --out-dir /tmp/rg_artifacts \
        --epochs 2

Dataset layout (SPEC 3.2): WAVs (16kHz mono, 4s) + metadata.jsonl with
file/label/split/... fields. Binary target = relay vs everything-else
(class-weighted BCE / sample-weighted LightGBM); per-class scores reported.

Artifacts written to --out-dir:
    cnn.pt                 RelayCNN state_dict (best dev AUC)
    cnn.meta.json          model meta (params, best dev AUC, epoch, seed)
    lgbm.txt               LightGBM booster
    feature_scaler.joblib  StandardScaler fitted on train features
    lgbm.meta.json         feature names + best iteration + dev AUC

--self-test generates a tiny synthetic fixture dataset (24 WAVs + metadata)
proving the full pipeline end-to-end without real data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from relayguard.common import TARGET_SR, iter_windows, load_audio, save_wav
from relayguard.features import FEATURE_NAMES, extract_batch
from relayguard.models.cnn import RelayCNN, count_parameters, logmel

WINDOWS_PER_SAMPLE = 3  # 4s WAV, 2s window, 1s hop


# ---------------------------------------------------------------------------
# Synthetic fixture dataset (self-test / tests; NOT the real datagen)
# ---------------------------------------------------------------------------

def synth_clean(rng: np.random.Generator, sr: int = TARGET_SR,
                dur: float = 4.0) -> np.ndarray:
    """Harmonic speech-like signal: f0 90-250Hz, ~4Hz syllabic AM, formant-ish
    resonances, full 0-8kHz bandwidth, light noise."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    f0 = rng.uniform(90.0, 250.0)
    x = np.zeros(n)
    n_harm = int(7500 // f0)
    # one smooth formant envelope per signal (formants above 1kHz so the
    # natural 2f0/3f0 harmonics keep their smooth decay and loudspeaker THD
    # shows up as excess energy at 2f0/3f0)
    formants = rng.choice([1000.0, 2000.0, 3000.0], size=3)
    for k in range(1, n_harm + 1):
        formant = max(1.0 / (1.0 + ((k * f0 - F) / 900.0) ** 2)
                      for F in formants)
        x += (0.6 ** k) * formant * np.sin(
            2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    # syllabic amplitude modulation + true pauses (smoothed on/off gate)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(3.0, 5.5) * t
                            + rng.uniform(0, 2 * np.pi))
    gate = (np.sin(2 * np.pi * rng.uniform(0.8, 1.5) * t
                   + rng.uniform(0, 2 * np.pi)) > -0.2).astype(float)
    gate = np.convolve(gate, np.ones(int(0.02 * sr)) / int(0.02 * sr),
                       mode="same")
    x = x * am * gate
    # fricative-like bursts: broadband noise 1.5-7.5kHz, so clean speech has
    # real energy above 3.4kHz (this is what the relay LPF removes)
    from scipy.signal import butter, sosfiltfilt
    bursts = (np.sin(2 * np.pi * rng.uniform(1.5, 3.0) * t
                     + rng.uniform(0, 2 * np.pi)) > 0.5).astype(float)
    noise = sosfiltfilt(butter(2, 1500.0, "highpass", fs=sr, output="sos"),
                        rng.standard_normal(n))
    x += rng.uniform(0.05, 0.15) * bursts * noise
    x += rng.uniform(0.001, 0.01) * rng.standard_normal(n)
    return (x / (np.abs(x).max() + 1e-9) * rng.uniform(0.3, 0.7)).astype(np.float32)


def _reverb_ir(x: np.ndarray, rng: np.random.Generator, sr: int,
               rt60: float | None = None) -> np.ndarray:
    """Cheap room reverb: convolve with an exponentially decaying noise IR
    (fixture approximation of the pyroomacoustics room in datagen chain.py)."""
    from scipy.signal import fftconvolve
    rt60 = rt60 if rt60 is not None else rng.uniform(0.15, 0.6)
    n_ir = max(int(rt60 * sr), int(0.03 * sr))
    ir = rng.standard_normal(n_ir) * np.exp(-6.91 * np.arange(n_ir) / n_ir)
    n_ramp = int(0.002 * sr)
    ir[:n_ramp] *= np.linspace(0.0, 1.0, n_ramp)
    ir[0] += 1.0  # direct path
    y = fftconvolve(x, ir)[: len(x)]
    return y.astype(np.float64)


def simulate_fixture_relay(clean: np.ndarray, rng: np.random.Generator,
                           sr: int = TARGET_SR) -> np.ndarray:
    """Simulated speakerphone relay: HPF ~300Hz + LPF ~3.4kHz + tanh soft-clip
    + echo comb + limiter (fixture approximation of datagen chain.py)."""
    from scipy.signal import butter, sosfiltfilt
    hp = sosfiltfilt(butter(2, rng.uniform(200, 400), "highpass", fs=sr,
                            output="sos"), clean)
    drive = rng.uniform(2.0, 6.0)
    dist = np.tanh(drive * hp) / np.tanh(drive)
    # band-limit AFTER the nonlinearity (loudspeaker response removes the
    # HF distortion products, as in a real transducer)
    lp = sosfiltfilt(butter(3, rng.uniform(3200, 3800), "lowpass", fs=sr,
                            output="sos"), dist)
    rev = _reverb_ir(lp, rng, sr, rt60=rng.uniform(0.15, 0.6))
    # smart-amp limiter: peak normalize + soft-knee compression above 0.8
    y = rev / (np.abs(rev).max() + 1e-9) * rng.uniform(0.7, 0.95)
    out = np.where(np.abs(y) > 0.8,
                   np.sign(y) * (0.8 + 0.2 * np.tanh((np.abs(y) - 0.8) / 0.2)),
                   y)
    return out.astype(np.float32)


def simulate_fixture_hardneg(clean: np.ndarray, kind: str,
                             rng: np.random.Generator,
                             sr: int = TARGET_SR) -> np.ndarray:
    """Fixture hard negatives: reverb-only / background-TV-ish / band-limited."""
    from scipy.signal import butter, sosfiltfilt
    if kind == "hardneg_reverb":
        y = _reverb_ir(clean, rng, sr, rt60=rng.uniform(0.3, 0.8))
    elif kind == "hardneg_tv":
        noise = synth_clean(rng, sr=sr, dur=len(clean) / sr)
        y = clean + rng.uniform(0.2, 0.5) * noise
    elif kind == "hardneg_headset":
        y = sosfiltfilt(butter(2, [200, 4000], "bandpass", fs=sr,
                               output="sos"), clean)
    else:
        y = clean
    return (y / (np.abs(y).max() + 1e-9) * 0.6).astype(np.float32)


def make_fixture_dataset(out_dir: str | Path, n_per_class: int = 6,
                         seed: int = 0) -> Path:
    """Generate a tiny synthetic dataset (WAVs + metadata.jsonl) per SPEC 3.2."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    labels = ["relay", "direct", "hardneg_tv", "hardneg_reverb"]
    records = []
    idx = 0
    for label in labels:
        for i in range(n_per_class):
            clean = synth_clean(rng)
            if label == "relay":
                audio = simulate_fixture_relay(clean, rng)
            elif label == "direct":
                audio = clean
            else:
                audio = simulate_fixture_hardneg(clean, label, rng)
            split = "train" if i < n_per_class - 2 else ("dev" if i == n_per_class - 2 else "test")
            rel = f"{label}/{idx:06d}.wav"
            save_wav(out_dir / rel, audio, TARGET_SR)
            records.append({
                "file": rel, "label": label, "split": split,
                "speaker_id": f"spk_{label}_{i}",
                "codec1": "none", "codec2": "none",
                "rt60": 0.4 if label == "relay" else 0.0,
                "distance_m": 1.0 if label == "relay" else 0.0,
                "device": "fixture", "snr_db": 30.0,
            })
            idx += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out_dir


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def load_metadata(data_dir: str | Path) -> list[dict]:
    """Read metadata.jsonl -> list of sample dicts (SPEC 3.2)."""
    path = Path(data_dir) / "metadata.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {data_dir}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_windows(data_dir: str | Path, rec: dict) -> np.ndarray:
    """All 2s windows of one sample -> (n_windows, 32000) float32."""
    audio = load_audio(Path(data_dir) / rec["file"])
    return np.stack(list(iter_windows(audio)))


def window_index(samples: list[dict], split: str) -> list[dict]:
    """Expand sample metadata into per-window records for a split."""
    out = []
    for rec in samples:
        if rec.get("split", "train") != split:
            continue
        for w in range(WINDOWS_PER_SAMPLE):
            out.append({"file": rec["file"], "win": w,
                        "label": 1 if rec["label"] == "relay" else 0,
                        "meta": rec})
    return out


def _load_windows_batch(data_dir: str | Path, batch: list[dict],
                        cache: dict) -> tuple[np.ndarray, np.ndarray]:
    wavs, labels = [], []
    for item in batch:
        key = item["file"]
        if key not in cache:
            if len(cache) > 64:          # tiny LRU-ish bound
                cache.pop(next(iter(cache)))
            cache[key] = sample_windows(data_dir, item["meta"])
        wavs.append(cache[key][item["win"]])
        labels.append(item["label"])
    return np.stack(wavs), np.asarray(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# CNN training
# ---------------------------------------------------------------------------

def _predict_cnn(model: RelayCNN, data_dir, items, batch: int,
                 device: torch.device) -> np.ndarray:
    model.eval()
    probs = np.zeros(len(items), dtype=np.float64)
    cache: dict = {}
    with torch.no_grad():
        for i in range(0, len(items), batch):
            wavs, _ = _load_windows_batch(data_dir, items[i:i + batch], cache)
            logits = model(logmel(torch.from_numpy(wavs).to(device)))
            probs[i:i + len(wavs)] = torch.sigmoid(logits).cpu().numpy()
    return probs


def _sample_level_auc(items: list[dict], probs: np.ndarray) -> float | None:
    """AUC after mean-aggregating window scores per sample file."""
    by_file: dict[str, list] = {}
    lab: dict[str, int] = {}
    for item, p in zip(items, probs):
        by_file.setdefault(item["file"], []).append(p)
        lab[item["file"]] = item["label"]
    y = np.array([lab[f] for f in by_file])
    s = np.array([np.mean(v) for v in by_file.values()])
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def train_cnn(samples: list[dict], data_dir, out_dir: Path, epochs: int,
              batch: int, seed: int, patience: int = 4,
              boost: dict[str, float] | None = None) -> dict:
    """Train RelayCNN; early stop on dev sample-level AUC; save cnn.pt+meta.

    ``boost`` maps metadata label -> oversampling factor for the train split
    (e.g. {"hardneg_reverb": 2} duplicates those window items), a simple
    class-weight tuning knob for hard-negative slices."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    train_items = window_index(samples, "train")
    if boost:
        extra = []
        for item in train_items:
            mult = boost.get(item["meta"].get("label"), 1)
            for _ in range(int(round(mult)) - 1):
                extra.append(item)
        if extra:
            train_items = train_items + extra
            print(f"[cnn] boost oversampling added {len(extra)} window items",
                  flush=True)
    dev_items = window_index(samples, "dev") or window_index(samples, "test")
    if not train_items:
        raise ValueError("no training windows found (check splits)")

    n_pos = sum(i["label"] for i in train_items)
    n_neg = len(train_items) - n_pos
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 1e-3)])

    model = RelayCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    rng = np.random.default_rng(seed)
    best_auc, best_state, best_epoch, bad = -1.0, None, -1, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_items))
        t0 = time.time()
        cache: dict = {}
        total_loss, nb = 0.0, 0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            wavs, labels = _load_windows_batch(
                data_dir, [train_items[j] for j in idx], cache)
            xb = torch.from_numpy(wavs).to(device)
            yb = torch.from_numpy(labels).to(device)
            opt.zero_grad()
            loss = loss_fn(model(logmel(xb)), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            nb += 1
        dev_probs = _predict_cnn(model, data_dir, dev_items, batch, device) \
            if dev_items else np.zeros(0)
        dev_auc = _sample_level_auc(dev_items, dev_probs) if dev_items else None
        print(f"[cnn] epoch {epoch + 1}/{epochs} loss={total_loss / max(nb, 1):.4f} "
              f"dev_auc={dev_auc if dev_auc is not None else float('nan'):.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        score = dev_auc if dev_auc is not None else -total_loss
        if score > best_auc:
            best_auc, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[cnn] early stop at epoch {epoch + 1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "cnn.pt")
    meta = {"arch": "RelayCNN", "n_mels": 64, "n_fft": 512, "hop": 160,
            "n_frames": 200, "params": count_parameters(model),
            "best_dev_auc": best_auc, "best_epoch": best_epoch + 1,
            "seed": seed, "task": "relay_vs_all"}
    (out_dir / "cnn.meta.json").write_text(json.dumps(meta, indent=2))
    return {"model": model, "meta": meta}


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def _features_for_split(samples, data_dir, split):
    """Features for every window of a split, extracted file-by-file so peak
    memory stays O(one file) instead of O(split) (matters at 5k+ samples)."""
    items = window_index(samples, split)
    if not items:
        return None, None, None
    n_win = {rec["file"]: 0 for rec in samples
             if rec.get("split", "train") == split}
    for item in items:
        n_win[item["file"]] += 1
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for rec in samples:
        if rec.get("split", "train") != split:
            continue
        wins = sample_windows(data_dir, rec)
        k = n_win.get(rec["file"], 0)
        if not k:
            continue
        F = extract_batch([wins[w] for w in range(min(k, len(wins)))])
        feats.extend(F)
        labels.extend([1 if rec["label"] == "relay" else 0] * len(F))
    return np.stack(feats), np.asarray(labels), items


def train_gbm(samples: list[dict], data_dir, out_dir: Path,
              seed: int, boost: dict[str, float] | None = None) -> dict:
    """Train LightGBM on handcrafted features (sample-weighted); save booster
    + StandardScaler + meta. ``boost`` multiplies the sample weight of the
    named metadata labels (hard-negative class-weight tuning)."""
    Xtr, ytr, tr_items = _features_for_split(samples, data_dir, "train")
    if Xtr is None:
        raise ValueError("no training windows found (check splits)")
    Xdv, ydv, dev_items = _features_for_split(samples, data_dir, "dev")
    if Xdv is None:
        Xdv, ydv, dev_items = _features_for_split(samples, data_dir, "test")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)

    n_pos = max(int(ytr.sum()), 1)
    n_neg = len(ytr) - n_pos
    wtr = np.where(ytr == 1, len(ytr) / (2.0 * n_pos), len(ytr) / (2.0 * max(n_neg, 1)))
    if boost:
        mult = np.array([boost.get(it["meta"].get("label"), 1.0)
                         for it in tr_items])
        wtr = wtr * mult
        print(f"[gbm] boost weights applied to "
              f"{int((mult != 1.0).sum())} windows", flush=True)

    dtrain = lgb.Dataset(Xtr_s, label=ytr, weight=wtr,
                         feature_name=list(FEATURE_NAMES))
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
              "num_leaves": 31, "min_data_in_leaf": 10, "feature_fraction": 0.9,
              "bagging_fraction": 0.9, "bagging_freq": 1, "seed": seed,
              "num_threads": 2, "verbose": -1}
    callbacks = [lgb.log_evaluation(period=0)]
    valid = None
    num_rounds = 500
    if Xdv is not None and len(np.unique(ydv)) > 1 and len(ytr) >= 1000:
        wdv = np.where(ydv == 1, len(ydv) / (2.0 * max(int(ydv.sum()), 1)),
                       len(ydv) / (2.0 * max(len(ydv) - int(ydv.sum()), 1)))
        valid = lgb.Dataset(scaler.transform(Xdv), label=ydv, weight=wdv,
                            reference=dtrain)
        callbacks.append(lgb.early_stopping(50, verbose=False))
    elif len(ytr) < 1000:
        # tiny datasets: dev-AUC early stopping freezes the model after a
        # handful of trees with probabilities compressed at the prior;
        # a fixed moderate round count is more reliable
        num_rounds = 200
    booster = lgb.train(params, dtrain, num_boost_round=num_rounds,
                        valid_sets=[valid] if valid else None,
                        callbacks=callbacks)

    booster.save_model(str(out_dir / "lgbm.txt"))
    joblib.dump(scaler, out_dir / "feature_scaler.joblib")

    dev_auc = None
    if Xdv is not None and len(np.unique(ydv)) > 1:
        probs = booster.predict(scaler.transform(Xdv),
                                num_iteration=booster.best_iteration or None)
        dev_auc = _sample_level_auc(dev_items, probs)
    meta = {"feature_names": list(FEATURE_NAMES),
            "best_iteration": booster.best_iteration or booster.num_trees(),
            "dev_auc": dev_auc, "seed": seed, "task": "relay_vs_all"}
    (out_dir / "lgbm.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[gbm] trees={meta['best_iteration']} dev_auc={dev_auc}", flush=True)
    return {"booster": booster, "scaler": scaler, "meta": meta}


# ---------------------------------------------------------------------------
# Per-class report
# ---------------------------------------------------------------------------

def per_class_report(samples, data_dir, cnn_model=None, gbm=None, scaler=None,
                     split: str = "test") -> dict:
    """Mean sample-level relay score per metadata label (and per-label AUC vs
    relay where possible)."""
    items = window_index(samples, split) or window_index(samples, "dev")
    if not items:
        return {}
    scores: dict[str, np.ndarray] = {}
    if cnn_model is not None:
        scores["cnn"] = _predict_cnn(cnn_model, data_dir, items, 64,
                                     torch.device("cpu"))
    if gbm is not None and scaler is not None:
        # features aligned with items (same order as window_index)
        wavs, cache = [], {}
        for item in items:
            if item["file"] not in cache:
                cache[item["file"]] = sample_windows(data_dir, item["meta"])
            wavs.append(cache[item["file"]][item["win"]])
        scores["gbm"] = gbm.predict(scaler.transform(extract_batch(wavs)),
                                    num_iteration=gbm.best_iteration or None)
    report: dict = {}
    for name, probs in scores.items():
        by_label: dict[str, list] = {}
        relay_scores: dict[str, list] = {}
        for item, p in zip(items, probs):
            by_label.setdefault(item["meta"]["label"], []).append(float(p))
            if item["label"] == 1:
                relay_scores.setdefault(item["meta"]["label"], []).append(float(p))
        entry = {}
        for lab, vals in sorted(by_label.items()):
            row = {"n_windows": len(vals), "mean_score": float(np.mean(vals))}
            if lab != "relay" and relay_scores:
                y = [1] * len(relay_scores["relay"]) + [0] * len(vals)
                s = relay_scores["relay"] + vals
                if len(np.unique(y)) > 1:
                    row["auc_vs_relay"] = float(roc_auc_score(y, s))
            entry[lab] = row
        report[name] = entry
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="dataset dir with WAVs + metadata.jsonl (SPEC 3.2)")
    ap.add_argument("--out-dir", default="artifacts/")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true",
                    help="generate a tiny synthetic fixture dataset into "
                         "<out-dir>/selftest_data and train on it")
    ap.add_argument("--boost", default=None,
                    help="comma list label:factor, e.g. "
                         "'hardneg_reverb:2,hardneg_car:2' — CNN oversampling "
                         "+ GBM sample-weight tuning for hard-negative slices")
    args = ap.parse_args(argv)

    torch.set_num_threads(2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        data_dir = make_fixture_dataset(out_dir / "selftest_data", seed=args.seed)
        print(f"[self-test] fixture dataset at {data_dir}", flush=True)
    elif args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        ap.error("--data-dir required (or use --self-test)")

    samples = load_metadata(data_dir)
    print(f"[data] {len(samples)} samples from {data_dir}", flush=True)

    boost = None
    if args.boost:
        boost = {}
        for part in args.boost.split(","):
            lab, _, fac = part.partition(":")
            boost[lab.strip()] = float(fac)

    cnn_res = train_cnn(samples, data_dir, out_dir, args.epochs, args.batch,
                        args.seed, boost=boost)
    gbm_res = train_gbm(samples, data_dir, out_dir, args.seed, boost=boost)

    report = per_class_report(samples, data_dir, cnn_res["model"],
                              gbm_res["booster"], gbm_res["scaler"])
    summary = {"cnn": cnn_res["meta"], "gbm": gbm_res["meta"],
               "per_class": report}
    (out_dir / "train_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
