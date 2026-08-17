"""Evaluation harness CLI (SPEC section 4 C2).

    python -m relayguard.eval.evaluate --data-dir DATA --artifacts-dir artifacts/ \
        --out artifacts/eval_report.md

Scores every sample of the dataset (test split if present, else all) with the
detectors whose artifacts exist in --artifacts-dir (GBM and/or CNN) plus the
always-available rule detectors, aggregates window scores per sample (mean),
and computes:
  - ROC / AUC, EER
  - hit rate (recall on relay) at FPR 1% / 2% / 5% operating points
  - per-slice tables: by metadata label (each hardneg_* class FPR
    individually), by codec2, by device preset (FPR/recall at the global
    2%-FPR operating threshold + slice AUC vs relay where defined)

Writes a markdown report to --out and a JSON twin next to it
(eval_report.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from relayguard.common import iter_windows, load_audio
from relayguard.models.detectors import (
    BandwidthForensics,
    CNNDetector,
    DistortionDetector,
    GBMDetector,
    ReverbDetector,
)
from relayguard.models.train import load_metadata

FPR_POINTS = (0.01, 0.02, 0.05)

# Labels counting as relay-positive (real_relay = real call voice through the
# simulated relay chain, from datagen/augment_real / real_augmented).
POSITIVE_LABELS = frozenset({"relay", "real_relay"})


# ---------------------------------------------------------------------------
# Metric primitives (importable, unit-tested)
# ---------------------------------------------------------------------------

def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate from ROC curve (interpolated FPR == FNR point)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = np.argmin(np.abs(fpr - fnr))
    # linear interpolation between the two bracketing points
    if i > 0 and (fpr[i] - fnr[i]) * (fpr[i - 1] - fnr[i - 1]) < 0:
        x0, x1 = fpr[i - 1], fpr[i]
        y0, y1 = (fpr - fnr)[i - 1], (fpr - fnr)[i]
        return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return float((fpr[i] + fnr[i]) / 2.0)


def hit_rate_at_fpr(labels: np.ndarray, scores: np.ndarray,
                    target_fpr: float) -> float:
    """Recall on the relay class at the operating point whose FPR is the
    largest value <= target_fpr (Neyman-Pearson style)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = np.where(fpr <= target_fpr + 1e-12)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def threshold_at_fpr(labels: np.ndarray, scores: np.ndarray,
                     target_fpr: float) -> float:
    """Score threshold achieving the largest FPR <= target_fpr."""
    fpr, _, thr = roc_curve(labels, scores)
    ok = np.where(fpr <= target_fpr + 1e-12)[0]
    if not len(ok):
        return float("inf")
    return float(thr[ok[-1]])


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """AUC, EER, hit rates at FPR 1%/2%/5%."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auc": None, "eer": None,
                **{f"hit_at_fpr_{int(p * 100)}": None for p in FPR_POINTS}}
    out = {"auc": float(roc_auc_score(labels, scores)),
           "eer": eer(labels, scores)}
    for p in FPR_POINTS:
        out[f"hit_at_fpr_{int(p * 100)}"] = hit_rate_at_fpr(labels, scores, p)
    return out


def per_slice_table(labels: np.ndarray, scores: np.ndarray,
                    slices: np.ndarray, threshold: float) -> list[dict]:
    """Per-slice FPR (non-relay slices) / recall (relay slice) at a fixed
    threshold, plus slice AUC vs relay where both classes exist."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    relay_mask = labels == 1
    rows = []
    for val in sorted(set(slices)):
        m = np.asarray(slices) == val
        row = {"slice": str(val), "n": int(m.sum()),
               "n_relay": int((m & relay_mask).sum())}
        if (m & relay_mask).any():
            row["recall"] = float((scores[m & relay_mask] >= threshold).mean())
        if (m & ~relay_mask).any():
            row["fpr"] = float((scores[m & ~relay_mask] >= threshold).mean())
        # slice AUC: relay samples vs this slice's negatives
        neg = m & ~relay_mask
        if neg.any() and relay_mask.any():
            y = np.concatenate([np.ones(relay_mask.sum()), np.zeros(neg.sum())])
            s = np.concatenate([scores[relay_mask], scores[neg]])
            row["auc_vs_relay"] = float(roc_auc_score(y, s))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Dataset scoring
# ---------------------------------------------------------------------------

def available_detectors(artifacts_dir: str | Path) -> list:
    """GBM/CNN detectors if their artifacts exist, plus rule detectors."""
    artifacts_dir = Path(artifacts_dir)
    dets = []
    if (artifacts_dir / "lgbm.txt").exists():
        dets.append(GBMDetector(artifacts_dir))
    if (artifacts_dir / "cnn.pt").exists():
        dets.append(CNNDetector(artifacts_dir))
    dets += [BandwidthForensics(), ReverbDetector(), DistortionDetector()]
    return dets


def score_dataset(data_dir: str | Path, artifacts_dir: str | Path,
                  split: str | None = None) -> dict:
    """Score every sample -> {detector_name: (labels, scores, meta_records)}.

    Window scores are mean-aggregated per sample. Uses the requested split;
    defaults to 'test' when present else all samples."""
    samples = load_metadata(data_dir)
    splits = {s.get("split", "train") for s in samples}
    if split is None:
        split = "test" if "test" in splits else None
    if split is not None:
        samples = [s for s in samples if s.get("split", "train") == split]

    dets = available_detectors(artifacts_dir)
    labels = np.array([1 if s["label"] in POSITIVE_LABELS else 0
                       for s in samples])
    scores = {d.name: np.zeros(len(samples)) for d in dets}
    for i, rec in enumerate(samples):
        audio = load_audio(Path(data_dir) / rec["file"])
        windows = list(iter_windows(audio))
        for d in dets:
            scores[d.name][i] = float(np.mean(
                [d.detect(w).score for w in windows]))
    # Score ensemble: plain average of the two model detectors (C2 fallback).
    if "cnn" in scores and "gbm" in scores:
        scores["cnn_gbm_avg"] = 0.5 * (scores["cnn"] + scores["gbm"])
    return {"labels": labels, "scores": scores, "samples": samples,
            "split": split, "detectors": list(scores)}


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def _md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "(no rows)\n"
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            _fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_report(result: dict) -> dict:
    """Metrics + slice tables for every scored detector."""
    labels = result["labels"]
    samples = result["samples"]
    report: dict = {"split": result["split"], "n_samples": len(samples),
                    "detectors": {}}
    for name, scores in result["scores"].items():
        entry = {"overall": compute_metrics(labels, scores)}
        thr = threshold_at_fpr(labels, scores, 0.02)
        entry["threshold_at_fpr_2"] = thr
        for key in ("label", "codec2", "device"):
            slices = np.array([str(s.get(key, "?")) for s in samples])
            entry[f"by_{key}"] = per_slice_table(labels, scores, slices, thr)
        pair = np.array([f"{s.get('codec1', '?')}->{s.get('codec2', '?')}"
                         for s in samples])
        entry["by_codec_pair"] = per_slice_table(labels, scores, pair, thr)
        report["detectors"][name] = entry
    return report


def render_markdown(report: dict) -> str:
    lines = ["# RelayGuard evaluation report", "",
             f"- split: `{report['split']}`",
             f"- samples: {report['n_samples']}", ""]
    for name, entry in report["detectors"].items():
        lines.append(f"## Detector: `{name}`")
        lines.append("")
        lines.append(_md_table([entry["overall"]],
                               ["auc", "eer", "hit_at_fpr_1",
                                "hit_at_fpr_2", "hit_at_fpr_5"]))
        lines.append(f"Operating threshold @ FPR 2%: "
                     f"`{_fmt(entry['threshold_at_fpr_2'])}`")
        lines.append("")
        for key, title in (("label", "By metadata label"),
                           ("codec2", "By codec2"),
                           ("codec_pair", "By codec pair (codec1->codec2)"),
                           ("device", "By device preset")):
            lines.append(f"### {title}")
            lines.append(_md_table(entry[f"by_{key}"],
                                   ["slice", "n", "n_relay", "recall", "fpr",
                                    "auc_vs_relay"]))
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--artifacts-dir", default="artifacts/")
    ap.add_argument("--out", default="artifacts/eval_report.md")
    ap.add_argument("--split", default=None,
                    help="split to evaluate (default: test if present else all)")
    args = ap.parse_args(argv)

    result = score_dataset(args.data_dir, args.artifacts_dir, args.split)
    report = build_report(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report))
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote {out_path} and {json_path}")
    for name, entry in report["detectors"].items():
        print(f"[eval] {name}: {entry['overall']}")
    return report


if __name__ == "__main__":
    main()
