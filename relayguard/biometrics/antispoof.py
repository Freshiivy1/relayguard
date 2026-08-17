"""Signal-level liveness / replay cues — no external weights (SPEC 4-C4).

Three lightweight numpy/scipy cues:
  1. band-edge energy       — replay through a loudspeaker (+ possible
                              telephony band-limit) starves the >3.4 kHz
                              band and the <120 Hz band relative to the
                              speech band.
  2. reverb-tail estimate   — replay radiates into a room: speech offsets
                              decay with a measurable RT60-ish tail instead
                              of dropping cleanly.
  3. quantization-noise     — codec/replay chains leave an unnaturally flat,
                              stationary noise floor in low-energy frames.
    flatness

Output: common.DetectorScore(name="antispoof_replay_cues", score in [0,1]).

IMPORTANT (info.md section 4): genuine callers heard through their OWN
speakerphone relay ELEVATE this score BY DESIGN. This detector feeds the
fusion engine as evidence; it must NEVER auto-reject a call. The optional
AASISTHook is interface-ready for a learned anti-spoofing model when
codec+replay-augmented weights become available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from relayguard.common import DetectorScore, to_mono_16k, TARGET_SR
from .verifier import BiometricsUnavailable

_EPS = 1e-12


def _frame_rms(audio: np.ndarray, sr: int, frame_ms: float = 30.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000.0))
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    return np.sqrt(np.mean(frames ** 2, axis=1) + _EPS)


def _band_edge_score(audio: np.ndarray, sr: int) -> tuple[float, dict]:
    """High if the spectrum is suspiciously band-limited (relay band-drop)."""
    spec = np.abs(np.fft.rfft(audio.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total = spec.sum() + _EPS
    low = spec[freqs < 120.0].sum() / total
    high = spec[(freqs >= 3400.0) & (freqs <= min(8000.0, sr / 2))].sum() / total
    # Clean direct 16 kHz speech typically has >1% energy above 3.4 kHz and a
    # little low-end rumble; loudspeaker/telephony replay crushes both edges.
    high_score = float(np.clip(1.0 - high / 0.01, 0.0, 1.0))
    low_score = float(np.clip(1.0 - low / 0.005, 0.0, 1.0))
    score = 0.7 * high_score + 0.3 * low_score
    return score, {"low_band_ratio": float(low), "high_band_ratio": float(high)}


def _reverb_tail_score(audio: np.ndarray, sr: int) -> tuple[float, dict]:
    """High if speech offsets decay slowly (room reverb tail from replay)."""
    rms = _frame_rms(audio, sr)
    if len(rms) < 10 or rms.max() < 1e-4:
        return 0.0, {"tail_frames_median": 0.0}
    db = 20.0 * np.log10(rms + _EPS)
    peak = np.max(db)
    active = db > (peak - 40.0)  # ignore deep silence
    tails = []
    i = 0
    while i < len(db) - 1:
        # offset: active frame followed by drop
        if active[i] and db[i + 1] < db[i] - 3.0:
            j = i + 1
            while j < len(db) and db[j] > db[i] - 25.0:
                j += 1
            tails.append(float(j - i))
            i = j
        else:
            i += 1
    if not tails:
        return 0.0, {"tail_frames_median": 0.0}
    med = float(np.median(tails))
    # ~3 frames (90 ms) = clean offset; >=10 frames (300 ms+) = audible tail.
    score = float(np.clip((med - 3.0) / 7.0, 0.0, 1.0))
    return score, {"tail_frames_median": med}


def _qnoise_flatness_score(audio: np.ndarray, sr: int) -> tuple[float, dict]:
    """High if the noise floor is flat/stationary (codec quantization)."""
    rms = _frame_rms(audio, sr)
    if len(rms) < 10:
        return 0.0, {"noise_flatness": 0.0, "noise_stationarity": 0.0}
    floor_thresh = np.percentile(rms, 25.0)
    quiet = audio[: len(rms) * int(sr * 0.03)].reshape(len(rms), -1)[
        rms <= floor_thresh]
    if len(quiet) < 3:
        return 0.0, {"noise_flatness": 0.0, "noise_stationarity": 0.0}
    specs = np.abs(np.fft.rfft(quiet.astype(np.float64), axis=1)) + _EPS
    flat = float(np.mean(np.exp(np.mean(np.log(specs), axis=1))
                         / np.mean(specs, axis=1)))
    stationarity = float(1.0 - np.clip(
        np.std(np.mean(specs, axis=1)) / (np.mean(np.mean(specs, axis=1)) + _EPS),
        0.0, 1.0))
    # Natural room/line noise is colored (flatness <~0.3); flat + stationary
    # floors point at quantization / comfort noise from a replay chain.
    score = float(np.clip((flat - 0.25) / 0.45, 0.0, 1.0) * (0.5 + 0.5 * stationarity))
    return score, {"noise_flatness": flat, "noise_stationarity": stationarity}


def analyze(audio: np.ndarray, sr: int = TARGET_SR) -> DetectorScore:
    """Replay/liveness cue score for a call-level audio segment.

    score ~ P(replay-like signal chain). Feeds fusion only — never a verdict.
    """
    audio16 = to_mono_16k(audio, sr)
    band_s, band_d = _band_edge_score(audio16, TARGET_SR)
    rev_s, rev_d = _reverb_tail_score(audio16, TARGET_SR)
    qn_s, qn_d = _qnoise_flatness_score(audio16, TARGET_SR)
    score = float(np.clip(0.4 * band_s + 0.35 * rev_s + 0.25 * qn_s, 0.0, 1.0))
    return DetectorScore(
        name="antispoof_replay_cues",
        score=score,
        details={
            "band_edge": band_d,
            "reverb_tail": rev_d,
            "quantization_noise": qn_d,
            "note": ("genuine speakerphone-relayed callers elevate this score "
                     "by design; fusion evidence only, never auto-reject"),
        },
    )


class AASISTHook:
    """Interface-ready stub for a learned anti-spoofing model (AASIST).

    No weights ship with RelayGuard. ``load()`` raises BiometricsUnavailable
    until a weights path with a real checkpoint is provided, at which point
    the loading code must be implemented (torch.jit / state_dict).
    """

    def __init__(self, weights_path: str | Path | None = None):
        self.weights_path = Path(weights_path) if weights_path else None
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        raise BiometricsUnavailable(
            "AASIST anti-spoofing weights are not available "
            f"(weights_path={self.weights_path}); hook is interface-ready "
            "but load() is unimplemented until weights are provided."
        )

    def analyze(self, audio: np.ndarray, sr: int = TARGET_SR) -> DetectorScore:
        self.load()  # raises until weights exist
        raise BiometricsUnavailable("AASIST inference not implemented")
