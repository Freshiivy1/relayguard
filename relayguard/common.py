"""RelayGuard shared contracts — audio I/O, windowing, schemas.

Owner: orchestrator (contract-level). Do not modify without updating SPEC.md.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

TARGET_SR = 16000


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convert arbitrary audio array to mono float32 16 kHz in [-1, 1]."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # (channels, samples) or (samples, channels)
        axis = 0 if audio.shape[0] <= audio.shape[1] and audio.shape[0] <= 8 else 1
        audio = audio.mean(axis=axis)
    audio = np.nan_to_num(audio)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    if sr != TARGET_SR:
        # rational resample
        g = np.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)
    return audio


def load_audio(path: str | Path) -> np.ndarray:
    """Load any audio file -> mono float32 16 kHz."""
    data, sr = sf.read(str(path), always_2d=True)
    return to_mono_16k(data.T, sr)


def load_audio_bytes(raw: bytes, sr: int | None = None, fmt: str = "wav") -> np.ndarray:
    """Load audio from bytes. fmt='wav' parses a WAV container; fmt='pcm16' treats
    raw as little-endian int16 samples at the given sr."""
    if fmt == "pcm16":
        if sr is None:
            raise ValueError("sr required for pcm16")
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return to_mono_16k(data, sr)
    data, file_sr = sf.read(io.BytesIO(raw), always_2d=True)
    return to_mono_16k(data.T, file_sr)


def save_wav(path: str | Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    """Save float32 [-1,1] audio as 16-bit PCM WAV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def iter_windows(audio: np.ndarray, win_s: float = 2.0, hop_s: float = 1.0,
                 sr: int = TARGET_SR) -> Iterator[np.ndarray]:
    """Yield overlapping windows of 16 kHz mono audio. Last partial window is
    zero-padded to full length."""
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    if len(audio) <= win:
        out = np.zeros(win, dtype=np.float32)
        out[: len(audio)] = audio
        yield out
        return
    pos = 0
    while pos + win <= len(audio):
        yield audio[pos : pos + win]
        pos += hop
    tail = audio[pos:]
    if len(tail) >= hop // 2:
        out = np.zeros(win, dtype=np.float32)
        out[: len(tail)] = tail
        yield out


@dataclass
class DetectorScore:
    name: str            # detector id
    score: float         # P(relay) in [0,1]
    details: dict = field(default_factory=dict)


@dataclass
class Verdict:
    state: str           # "GREEN" | "CHALLENGE" | "RED"
    confidence: float
    fused_score: float
    detector_scores: list = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "confidence": round(float(self.confidence), 4),
            "fused_score": round(float(self.fused_score), 4),
            "reason": self.reason,
            "detectors": [
                {"name": d.name, "score": round(float(d.score), 4), "details": d.details}
                for d in self.detector_scores
            ],
        }


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config; defaults to configs/default.yaml next to repo root."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def pcm16_bytes(audio: np.ndarray) -> bytes:
    """float32 [-1,1] -> little-endian PCM16 bytes (for API responses/tests)."""
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
