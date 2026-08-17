"""Mid-call acoustic channel-switch detection via delta-BIC (info.md 2.5).

Rationale: innocent odd channels (cheap headset, bathroom, car kit) are weird
from t=0 and STAY constant; a relay being added or speakerphone toggled makes
the channel "become weird at t=127 s". Acoustic Change Detection (delta-BIC on
rolling feature vectors, cf. Zhong et al. Eurospeech 2003) is the classic tool.

Method:
- Feature stream: per-100 ms mean log-mel vectors (24 mels). If
  relayguard.features.extract (SPEC 3.4) is importable, its handcrafted vector
  (computed on 1 s hops, z-scored, held over the hop) is appended as extra
  feature columns; the module degrades gracefully to mel-only otherwise.
- Slide two adjacent 4 s windows with a 2 s hop. For each split, compare a
  pooled diagonal-Gaussian model with a split two-Gaussian model:
      delta_BIC = N*sum(log var_pooled) - Na*sum(log var_A) - Nb*sum(log var_B)
  (penalty terms cancel for equal model dimensionality; diagonal covariance
  keeps the estimate well-conditioned on 40-frame windows).
- Peaks above an adaptive threshold (median + k*MAD, floored) are change
  points; a 4 s refractory keeps the strongest peak per neighbourhood.

Score contract:
- no switch + stable channel          -> 0.1  (benign constant device)
- >= 2 switches / unstable meander    -> 0.4
- exactly 1 strong mid-call switch    -> 0.85+ (relay added / speakerphone on)
"""
from __future__ import annotations

import numpy as np

from relayguard.common import DetectorScore
from relayguard.context._mel import log_mel

WINDOW_S = 4.0          # each side of the candidate split
HOP_S = 2.0             # hop between candidate splits
FRAME_S = 0.100         # feature frame period
N_MELS = 24
REFRACTORY_S = 4.0
MAD_K = 6.0             # adaptive threshold = median + MAD_K * 1.4826 * MAD
MIN_THRESH = 25.0       # absolute floor for a peak to count as a switch

try:  # optional enrichment (SPEC 3.4); absent -> mel-only fallback
    from relayguard.features.extract import extract_features as _extract_features
except Exception:  # pragma: no cover - depends on other agents' work
    _extract_features = None


def _feature_stream(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Per-100 ms feature matrix (n_frames, D) and frame period in seconds."""
    lm = log_mel(audio, sr, n_mels=N_MELS, frame_ms=100.0, hop_ms=100.0)
    feats = lm
    if _extract_features is not None:  # pragma: no cover - integration path
        win = int(1.0 * sr)
        extra = []
        pos = 0
        while pos + win <= len(audio):
            extra.append(_extract_features(audio[pos:pos + win], sr))
            pos += win
        if extra:
            extra = np.asarray(extra, dtype=np.float64)
            extra = (extra - extra.mean(axis=0)) / (extra.std(axis=0) + 1e-9)
            # hold each 1 s vector over its ten 100 ms frames
            rep = np.repeat(extra, int(1.0 / FRAME_S), axis=0)
            n = min(len(rep), len(feats))
            feats = np.concatenate([feats[:n], rep[:n]], axis=1)
    return np.asarray(feats, dtype=np.float64), FRAME_S


def _delta_bic_curve(feats: np.ndarray, frame_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(positions_s, delta_bic) at each candidate split."""
    win = int(round(WINDOW_S / frame_s))
    hop = int(round(HOP_S / frame_s))
    n = len(feats)
    eps = 1e-8
    pos_list, val_list = [], []
    for c in range(win, n - win + 1, hop):
        a = feats[c - win:c]
        b = feats[c:c + win]
        p = feats[c - win:c + win]
        va = a.var(axis=0) + eps
        vb = b.var(axis=0) + eps
        vp = p.var(axis=0) + eps
        val = len(p) * np.log(vp).sum() - len(a) * np.log(va).sum() - len(b) * np.log(vb).sum()
        pos_list.append(c * frame_s)
        val_list.append(float(val))
    return np.asarray(pos_list), np.asarray(val_list)


def _adaptive_threshold(vals: np.ndarray, n_frames: int = 0,
                        n_dims: int = N_MELS) -> float:
    """Scale-aware adaptive threshold for the delta-BIC curve.

    delta-BIC magnitudes scale with (window frames x feature dims), and the
    enrichment path (50 z-scored features appended to 24 mels) inflates them
    ~10x; a plain median + k*MAD threshold then lets a true switch inflate
    its own threshold. Instead:
    - scale floor: ``n_frames * n_dims * 0.8`` (expected order of magnitude
      of a real change in a pooled diagonal-Gaussian comparison), and
    - noise baseline: median of the LOWER HALF of the curve + 3*MAD
      (lower half excludes candidate peaks, so a true switch cannot raise
      its own bar).
    """
    if len(vals) == 0:
        return MIN_THRESH
    scale_floor = 0.8 * max(n_frames, 1) * max(n_dims, 1)
    lower = np.sort(vals)[: max(1, len(vals) // 2)]
    base = float(np.median(lower))
    mad = float(np.median(np.abs(lower - base))) * 1.4826
    return max(MIN_THRESH, scale_floor, base + 3.0 * mad)


def _pick_peaks(pos: np.ndarray, vals: np.ndarray,
                thresh: float) -> list[tuple[float, float]]:
    """Local maxima above threshold with a refractory period (strongest first)."""
    if len(vals) == 0:
        return []
    cand = []
    for i in range(len(vals)):
        left = vals[i - 1] if i > 0 else -np.inf
        right = vals[i + 1] if i < len(vals) - 1 else -np.inf
        if vals[i] > thresh and vals[i] >= left and vals[i] >= right:
            cand.append((float(vals[i]), float(pos[i])))
    cand.sort(reverse=True)  # strongest first
    accepted: list[tuple[float, float]] = []
    for strength, t in cand:
        if all(abs(t - t0) >= REFRACTORY_S for _, t0 in accepted):
            accepted.append((t, strength))
    accepted.sort()
    return accepted


def _threshold_for(feats: np.ndarray, frame_s: float,
                   vals: np.ndarray) -> float:
    """Threshold for a curve computed from ``feats`` (pooled window frames)."""
    win = int(round(WINDOW_S / frame_s))
    return _adaptive_threshold(vals, n_frames=2 * win, n_dims=feats.shape[1])


def detect_channel_switches(audio: np.ndarray, sr: int = 16000) -> list[float]:
    """Return timestamps (seconds) of detected mid-call channel switches."""
    audio = np.asarray(audio, dtype=np.float32)
    feats, frame_s = _feature_stream(audio, sr)
    pos, vals = _delta_bic_curve(feats, frame_s)
    thresh = _threshold_for(feats, frame_s, vals)
    return [t for t, _ in _pick_peaks(pos, vals, thresh)]


def analyze(audio: np.ndarray, sr: int = 16000) -> DetectorScore:
    """Call-level channel-switch analysis -> DetectorScore(name='channel_switch')."""
    audio = np.asarray(audio, dtype=np.float32)
    feats, frame_s = _feature_stream(audio, sr)
    pos, vals = _delta_bic_curve(feats, frame_s)
    thresh = _threshold_for(feats, frame_s, vals)
    peaks = _pick_peaks(pos, vals, thresh)

    # global stability: how far the typical (median) divergence sits below the
    # switch threshold. 1.0 = rock-steady channel; ~0 = constant meandering.
    if len(vals):
        med = float(np.median(vals))
        stability = float(np.clip(1.0 - med / max(thresh, 1e-9), 0.0, 1.0))
    else:
        stability = 1.0

    n_sw = len(peaks)
    if n_sw == 0:
        score = 0.10 if stability >= 0.5 else 0.40
    elif n_sw == 1:
        strength = peaks[0][1] / max(thresh, 1e-9)
        score = float(np.clip(0.85 + 0.05 * min(strength - 1.0, 2.0), 0.85, 0.95))
    else:
        score = 0.40  # multiple switches -> unstable channel, ambiguous

    details = {
        "timestamps_s": [t for t, _ in peaks],
        "switch_strengths": [s for _, s in peaks],
        "n_switches": n_sw,
        "bic_threshold": float(thresh),
        "global_stability": stability,
        "curve": {"positions_s": [float(p) for p in pos],
                  "delta_bic": [float(v) for v in vals]},
        "features_backend": "mel+extract" if _extract_features is not None else "mel",
        "scoring": {
            "no_switch_stable": 0.10, "no_switch_unstable": 0.40,
            "single_strong_switch": "0.85 + 0.05*min(strength-1, 2) (cap 0.95)",
            "multiple_switches": 0.40,
        },
    }
    return DetectorScore(name="channel_switch", score=score, details=details)
