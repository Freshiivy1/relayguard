"""Window-level detectors (SPEC 3.3 / section 4 C2).

Model detectors (lazy artifact loading):
    CNNDetector(artifacts_dir)   - RelayCNN wav->logit; ONNX (cnn.onnx) first,
                                   torch (cnn.pt) fallback
    GBMDetector(artifacts_dir)   - LightGBM on handcrafted features, needs
                                   lgbm.txt + feature_scaler.joblib

Rule detectors (feature -> [0,1] via documented logistic mappings; thresholds
justified from info.md section 2 ranked detection angles):
    BandwidthForensics - subband voids / band-edge drops (angle #2)
    ReverbDetector     - SRMR-style ratio + envelope decay (angle #3)
    DistortionDetector - THD + crest + limiter plateau (angle #4)

Every detector: detect(window, sr=16000) -> common.DetectorScore.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from relayguard.common import DetectorScore
from relayguard.features import FEATURE_INDEX, extract_features


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _f(feats: np.ndarray, name: str) -> float:
    return float(feats[FEATURE_INDEX[name]])


# ---------------------------------------------------------------------------
# Model-backed detectors
# ---------------------------------------------------------------------------

class CNNDetector:
    """RelayCNN window detector. Lazily loads the CNN from artifacts_dir.

    Backend selection (first wins):
      1. ONNX: ``cnn.onnx`` via onnxruntime — the only backend available in
         the deployed (torch-free) container. The exported graph covers the
         full waveform -> log-mel -> logit path (see models/export_onnx.py),
         so it takes the same raw 2 s window the torch path did.
      2. torch fallback: ``cnn.pt`` via RelayCNN + torch logmel, used when
         cnn.onnx or onnxruntime is missing (e.g. a dev/training checkout).
    """

    name = "cnn"

    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self._backend: str | None = None   # "onnx" | "torch"
        self._session = None               # onnxruntime.InferenceSession
        self._model = None                 # torch RelayCNN

    def _load(self) -> str:
        if self._backend is not None:
            return self._backend
        onnx_path = self.artifacts_dir / "cnn.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
            except ImportError:
                pass  # onnxruntime absent -> torch fallback below
            else:
                self._session = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"])
                self._backend = "onnx"
                return self._backend
        path = self.artifacts_dir / "cnn.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"CNN artifact not found: {onnx_path} or {path}")
        import torch
        from relayguard.models.cnn import RelayCNN
        model = RelayCNN()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self._model = model
        self._backend = "torch"
        return self._backend

    @staticmethod
    def _wav32000(window: np.ndarray) -> np.ndarray:
        wav = np.asarray(window, dtype=np.float32).ravel()[:32000]
        if wav.size < 32000:
            wav = np.pad(wav, (0, 32000 - wav.size))
        return wav

    def detect(self, window: np.ndarray, sr: int = 16000) -> DetectorScore:
        backend = self._load()
        wav = self._wav32000(window)
        if backend == "onnx":
            logit = float(self._session.run(
                None, {"waveform": wav[None, :]})[0][0])
        else:
            import torch
            from relayguard.models.cnn import logmel
            with torch.no_grad():
                logit = float(self._model(logmel(torch.from_numpy(wav).unsqueeze(0)))[0])
        return DetectorScore(name=self.name, score=_sigmoid(logit),
                             details={"logit": round(logit, 4),
                                      "backend": backend})


class GBMDetector:
    """LightGBM window detector on handcrafted features. Lazily loads
    lgbm.txt + feature_scaler.joblib from artifacts_dir."""

    name = "gbm"

    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self._booster = None
        self._scaler = None

    def _load(self):
        if self._booster is not None:
            return self._booster, self._scaler
        import joblib
        import lightgbm as lgb
        model_path = self.artifacts_dir / "lgbm.txt"
        scaler_path = self.artifacts_dir / "feature_scaler.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"GBM artifact not found: {model_path}")
        self._booster = lgb.Booster(model_file=str(model_path))
        self._scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        return self._booster, self._scaler

    def detect(self, window: np.ndarray, sr: int = 16000) -> DetectorScore:
        booster, scaler = self._load()
        feats = extract_features(window, sr=sr).reshape(1, -1)
        if scaler is not None:
            feats = scaler.transform(feats)
        score = float(booster.predict(feats)[0])
        return DetectorScore(name=self.name, score=_clip01(score),
                             details={"n_trees": booster.num_trees()})


def _clip01(v: float) -> float:
    return float(min(max(v, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Rule-based detectors
# ---------------------------------------------------------------------------

class _RuleDetector:
    """Base: extract features once, map to [0,1] via a logistic on a weighted
    evidence sum. Subclasses define weights + midpoint and document the
    physics (info.md section 2)."""

    name = "rule"
    # logistic sharpness: z = SHARPNESS * (evidence - MIDPOINT)
    sharpness = 10.0
    midpoint = 0.5

    def evidence(self, feats: np.ndarray) -> tuple[float, dict]:
        raise NotImplementedError

    def detect(self, window: np.ndarray, sr: int = 16000) -> DetectorScore:
        feats = extract_features(window, sr=sr)
        z, details = self.evidence(feats)
        score = _sigmoid(self.sharpness * (z - self.midpoint))
        details["evidence"] = round(z, 4)
        return DetectorScore(name=self.name, score=score, details=details)


class BandwidthForensics(_RuleDetector):
    """Double-bandlimit / spectral-void forensics (info.md sec.2 angle #2).

    Physics: the relay loudspeaker path band-limits speech to ~300Hz-3.4kHz
    (HPF 120-500Hz + LPF 3.4-10kHz per SPEC 4/C1); a direct call keeps either
    fullband energy (wideband codecs) or a single clean codec band-edge.
    Evidence terms (all in [0,1], weighted):
      - bandedge_drop_300  (>0 when energy below 300Hz is missing: HPF)
      - bandedge_drop_3400 (>0 when energy above 3.4kHz collapses: LPF)
      - hf_void_frac       (fraction of 3.4-8kHz bins that are voids)
      - 1 - hbcorr_*       (double codec pass decorrelates the high-band
                            envelope from the 1-3.4kHz reference; info.md #5)
    Both band edges must fire together for a strong score: a single edge is
    consistent with one codec (direct-call hard negative).
    """
    name = "bandwidth_forensics"

    def evidence(self, feats: np.ndarray):
        d300 = max(_f(feats, "bandedge_drop_300"), 0.0)
        d3400 = max(_f(feats, "bandedge_drop_3400"), 0.0)
        void = _f(feats, "hf_void_frac")
        decorr = 1.0 - 0.5 * (_f(feats, "hbcorr_3k4_4k") + _f(feats, "hbcorr_4k_8k"))
        edges = min(d300, d3400)          # conjunctive: both edges required
        z = 0.40 * edges + 0.20 * (d300 + d3400) / 2.0 + 0.20 * void + 0.20 * decorr
        return z, {"drop_300": round(d300, 3), "drop_3400": round(d3400, 3),
                   "hf_void_frac": round(void, 3), "hb_decorr": round(decorr, 3)}


class ReverbDetector(_RuleDetector):
    """Room-reverb detector (info.md sec.2 angle #3).

    Physics: the relay adds a room convolution (RT60 0.15-0.9s, low DRR);
    direct calls are anechoic-ish (hard-negative direct+reverb is the FP risk,
    so this detector alone must never be decisive - fusion handles that).
    Evidence:
      - 1 - srmr_mean  (reverb smears envelope modulation energy from the
                        3-20Hz syllabic band into 20-160Hz -> SRMR drops)
      - 1 - env_decay_slope (shallow post-peak envelope decay = reverb tail)
      - env_decay_t12  (longer time to decay 12dB after peaks)
    gap_flatness is intentionally NOT used here: dry-speech gaps are
    noise-floor (flat) while reverberant gaps initially hold a structured
    tonal tail, so the sign of that cue is chain-dependent.
    """
    name = "reverb"

    def evidence(self, feats: np.ndarray):
        srmr = _f(feats, "srmr_mean")
        decay = _f(feats, "env_decay_slope")
        t12 = _f(feats, "env_decay_t12")
        z = (0.35 * (1.0 - srmr) + 0.35 * (1.0 - decay) + 0.30 * t12)
        return z, {"srmr_mean": round(srmr, 3), "env_decay_slope": round(decay, 3),
                   "env_decay_t12": round(t12, 3)}


class DistortionDetector(_RuleDetector):
    """Loudspeaker nonlinearity detector (info.md sec.2 angle #4; Ren et al.
    2019 report TPR 97.8% in-domain for THD/dynamic-range cues).

    Physics: small loudspeakers add harmonic distortion (energy at 2f0/3f0 on
    voiced frames) and smart-amp limiting (envelope plateaus, compressed
    dynamic range, lowered crest factor).
    Evidence:
      - thd_proxy        ((E@2f0+E@3f0)/E@f0, normalized /4)
      - plateau_frac     (fraction of frames pinned within 1.5dB of p95 RMS)
      - 1 - crest_factor_mean (limiting/compression lowers peak-to-RMS)
      - 1 - dyn_range_db (limiter compresses short-term dynamic range)
    """
    name = "distortion"

    def evidence(self, feats: np.ndarray):
        thd = _f(feats, "thd_proxy")
        plateau = _f(feats, "plateau_frac")
        crest = _f(feats, "crest_factor_mean")
        dyn = _f(feats, "dyn_range_db")
        z = (0.40 * thd + 0.25 * plateau
             + 0.20 * (1.0 - crest) + 0.15 * (1.0 - dyn))
        return z, {"thd_proxy": round(thd, 3), "plateau_frac": round(plateau, 3),
                   "crest_factor_mean": round(crest, 3),
                   "dyn_range_db": round(dyn, 3)}
