"""ECAPA-TDNN speaker verification wrapper (SpeechBrain, lazy-loaded).

Design notes (info.md section 4 / SPEC 4-C4):
- Model: ``speechbrain/spkrec-ecapa-voxceleb`` (Apache-2.0, CPU-OK).
  Cached under ``~/.cache/relayguard/ecapa``; downloaded ONCE, lazily, the
  first time verification is actually invoked. Importing this module never
  touches the network or speechbrain.
- Expected accuracy: single-digit EER at 8 kHz telephony, ~8-10% EER on
  relayed genuine speech (loudspeaker + room + double codec degrades the
  channel). Therefore operating thresholds MUST be calibrated per channel
  class; the constants below are DOCUMENTED PLACEHOLDERS, not production
  values. Enroll during a verified call on the same channel class,
  3-5 phrases, >= 15 s net speech.
- Audio contract: internal 16 kHz mono float32; 8 kHz input is resampled
  to 16 kHz first (pass ``sr=8000``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from relayguard.common import to_mono_16k, TARGET_SR

EMB_DIM = 192  # ECAPA-TDNN embedding size
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SAVEDIR = Path.home() / ".cache" / "relayguard" / "ecapa"

# Minimum net speech required for a usable verification sample (seconds).
MIN_NET_SPEECH_S = 3.0

# --- Threshold guide (PLACEHOLDERS — calibrate on channel-matched data) -----
# Raw cosine decision thresholds per channel class. Relayed genuine speech
# sits at ~8-10% EER (vs single-digit on handset), so the relay channel
# threshold is set LOWER to keep false-reject of genuine callers acceptable.
IDENTITY_THRESHOLDS = {
    "handset": 0.30,
    "speakerphone_relay": 0.20,
}
# Logistic calibration parameters p = sigmoid(a * cosine + b), per channel.
# Placeholders until fitted with Platt scaling on enrolled-vs-impostor scores
# collected per channel class (see relayguard.fusion.calibrate).
_LOGISTIC_PARAMS = {
    "handset": (8.0, -2.4),          # midpoint at cosine 0.30
    "speakerphone_relay": (6.0, -1.2),  # midpoint at cosine 0.20
}


class BiometricsUnavailable(RuntimeError):
    """Raised when speaker biometrics cannot run (speechbrain/model missing)."""


def cosine_to_prob(score: float, channel: str = "handset") -> float:
    """Map a raw cosine score to P(same speaker) via per-channel logistic.

    ``channel`` is "handset" or "speakerphone_relay". Parameters are
    documented placeholders — calibrate per channel before production use.
    """
    if channel not in _LOGISTIC_PARAMS:
        raise ValueError(f"unknown channel {channel!r}; "
                         f"expected one of {sorted(_LOGISTIC_PARAMS)}")
    a, b = _LOGISTIC_PARAMS[channel]
    z = np.clip(a * float(score) + b, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-z)))


# --- Net-speech gate ---------------------------------------------------------
# Absolute RMS floor (dBFS) for the fallback energy VAD: anything below is
# treated as silence/noise regardless of the adaptive floor.
ABS_FLOOR_DB = -45.0


def _fallback_speech_frames(audio: np.ndarray, sr: int, frame_ms: float = 30.0
                            ) -> np.ndarray:
    """Minimal energy VAD used when relayguard.context.vad finds no speech.

    A frame counts as speech when its RMS dBFS is above BOTH:
    - an absolute floor (-45 dBFS), and
    - an adaptive floor ``max(-45, p20 + 6 dB)`` (p20 = 20th percentile of
      frame RMS dB, a robust noise-floor estimate).
    For constant-energy material (p95 - p20 < 6 dB, e.g. a sustained tone at
    ~100% duty cycle) the adaptive floor would sit above every frame, so the
    absolute floor alone decides. Pure numpy; deterministic. Digital silence
    is still rejected (RMS far below -45 dBFS).
    """
    frame = max(1, int(sr * frame_ms / 1000.0))
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    abs_ok = rms_db > ABS_FLOOR_DB
    p20 = float(np.percentile(rms_db, 20.0))  # robust noise-floor estimate
    p95 = float(np.percentile(rms_db, 95.0))
    if (p95 - p20) < 6.0:
        # constant-energy sustained material: absolute floor alone decides
        return abs_ok
    adaptive_floor = max(ABS_FLOOR_DB, p20 + 6.0)
    return abs_ok & (rms_db > adaptive_floor)


def _speech_frames(audio: np.ndarray, sr: int, frame_ms: float = 30.0
                   ) -> np.ndarray:
    """Prefer relayguard.context.vad when it finds ANY speech; else fall back.

    The context VAD (p20 + 10 dB floor plus flatness/ZCR gates) can net 0.0 s
    on ~90%-duty synthetic fixtures where the adaptive floor sits above every
    frame — in that case the local energy VAD is used instead.
    """
    try:
        from relayguard.context.vad import get_speech_frames
        mask = np.asarray(get_speech_frames(audio, sr, frame_ms=frame_ms),
                          dtype=bool)
        if mask.any():
            return mask
    except Exception:
        pass
    return _fallback_speech_frames(audio, sr, frame_ms=frame_ms)


def net_speech_seconds(audio: np.ndarray, sr: int = TARGET_SR,
                       frame_ms: float = 30.0) -> float:
    """Net voiced seconds in ``audio`` (16 kHz; resampled if sr differs)."""
    audio16 = to_mono_16k(audio, sr)
    mask = _speech_frames(audio16, TARGET_SR, frame_ms=frame_ms)
    return float(mask.sum() * (frame_ms / 1000.0))


def _import_speechbrain():
    """Import the SpeechBrain speaker-recognition interface (lazy).

    Isolated in its own function so tests can monkeypatch the import.
    """
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
        return SpeakerRecognition
    except ImportError:
        try:  # older speechbrain API
            from speechbrain.pretrained import SpeakerRecognition
            return SpeakerRecognition
        except ImportError as exc:
            raise BiometricsUnavailable(
                "speechbrain is not installed; speaker verification is "
                "unavailable. Install speechbrain to enable ECAPA-TDNN "
                "biometrics (see requirements/docs)."
            ) from exc


class SpeakerVerifier:
    """Lazy SpeechBrain ECAPA-TDNN wrapper.

    The model is loaded on first use of :meth:`embedding`; construction and
    the net-speech gate never require speechbrain.
    """

    def __init__(self, savedir: str | Path | None = None):
        self.savedir = Path(savedir) if savedir else DEFAULT_SAVEDIR
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        cls = _import_speechbrain()
        try:
            self._model = cls.from_hparams(
                source=MODEL_SOURCE,
                savedir=str(self.savedir),
                run_opts={"device": "cpu"},
            )
        except BiometricsUnavailable:
            raise
        except Exception as exc:  # download failure, corrupt cache, OOM, ...
            raise BiometricsUnavailable(
                f"failed to load ECAPA-TDNN model {MODEL_SOURCE!r} into "
                f"{self.savedir}: {exc}"
            ) from exc
        return self._model

    # -- audio prep -----------------------------------------------------------
    @staticmethod
    def _prep(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        return to_mono_16k(audio, sr)

    # -- net-speech gate ------------------------------------------------------
    @staticmethod
    def net_speech_gate(audio: np.ndarray, min_s: float = MIN_NET_SPEECH_S,
                        sr: int = TARGET_SR) -> tuple[bool, float]:
        """Return (ok, net_seconds). ok = net voiced speech >= min_s."""
        net = net_speech_seconds(audio, sr)
        return (net >= min_s, net)

    # -- embeddings -----------------------------------------------------------
    def embedding(self, audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        """L2-normalized 192-d embedding for one utterance."""
        import torch
        model = self._ensure_model()
        audio16 = self._prep(audio, sr)
        wav = torch.from_numpy(np.ascontiguousarray(audio16, dtype=np.float32))
        with torch.no_grad():
            emb = model.encode_batch(wav.unsqueeze(0)).squeeze().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float64).ravel()
        norm = np.linalg.norm(emb)
        if norm <= 0:
            raise BiometricsUnavailable("model returned a zero embedding")
        return emb / norm

    def enroll(self, audios: list[np.ndarray], sr: int = TARGET_SR) -> np.ndarray:
        """Enroll from several utterances -> 192-d centroid.

        Per-utterance embeddings are L2-normalized, averaged, then the mean
        is re-normalized (spherical centroid).
        """
        if not audios:
            raise ValueError("enroll() requires at least one utterance")
        embs = np.stack([self.embedding(a, sr) for a in audios])
        centroid = embs.mean(axis=0)
        return centroid / np.linalg.norm(centroid)

    def verify(self, audio: np.ndarray, enrolled_emb: np.ndarray,
               sr: int = TARGET_SR) -> float:
        """Cosine similarity in [-1, 1] of ``audio`` vs enrolled centroid."""
        emb = self.embedding(audio, sr)
        centroid = np.asarray(enrolled_emb, dtype=np.float64).ravel()
        n = np.linalg.norm(centroid)
        if n <= 0:
            raise ValueError("enrolled embedding has zero norm")
        return float(np.clip(np.dot(emb, centroid / n), -1.0, 1.0))


# --- Voiceprint persistence ---------------------------------------------------
def save_voiceprint(path: str | Path, emb: np.ndarray, meta: dict | None = None
                    ) -> None:
    """Persist an enrolled centroid + JSON metadata to ``path`` (.npz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = np.asarray(emb, dtype=np.float32).ravel()
    if emb.shape != (EMB_DIM,):
        raise ValueError(f"voiceprint must be {EMB_DIM}-d, got {emb.shape}")
    np.savez(str(path), embedding=emb,
             meta=json.dumps(meta or {}, sort_keys=True))


def load_voiceprint(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a voiceprint saved by :func:`save_voiceprint`.

    Returns (embedding float32 192-d, meta dict).
    """
    with np.load(str(path), allow_pickle=False) as z:
        emb = z["embedding"].astype(np.float32)
        meta = json.loads(str(z["meta"]))
    return emb, meta
