"""Energy + spectral-flatness VAD and turn segmentation. No external models.

Design (per SPEC 4 / info.md 2.1):
- Frame the signal (default 30 ms frames, hop = frame length).
- Primary decision: frame RMS dB above an adaptive noise floor defined as
  the 20th percentile of frame RMS dB + 10 dB.
- Refinement: reject noise-like high-energy frames by requiring spectral
  flatness < 0.5 (speech is harmonic/comb-like) and a plausible zero-crossing
  rate (speech at 16 kHz sits roughly in 0.5%..45% crossings/sample; steady
  tones fall below, broadband hiss above).
- Hangover: a speech decision is held for 5 frames after the last speech
  frame so intra-word gaps do not fragment turns.
- Segments shorter than 150 ms are dropped (clicks/bumps).
- segment_turns merges segments separated by gaps < 0.35 s into one turn.

All operations are pure numpy and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relayguard.context._mel import frame_audio

HANGOVER_FRAMES = 5
MIN_SEGMENT_S = 0.150
MERGE_GAP_S = 0.35
FLATNESS_MAX = 0.5
ZCR_MIN, ZCR_MAX = 0.005, 0.45
FLOOR_PERCENTILE = 20.0
FLOOR_OFFSET_DB = 10.0


@dataclass
class Turn:
    """A contiguous speaking turn."""
    start_s: float
    end_s: float
    rms_db: float
    mean_f0: float | None = None

    @property
    def dur_s(self) -> float:
        return self.end_s - self.start_s


def _frame_features(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rms_db, spectral_flatness, zcr) per frame."""
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    win = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    flat = np.exp(np.mean(np.log(spec + 1e-12), axis=1)) / (np.mean(spec, axis=1) + 1e-12)
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1) if frames.shape[1] > 1 \
        else np.zeros(len(frames))
    return rms_db, flat, zcr


def _apply_hangover(mask: np.ndarray, hangover: int) -> np.ndarray:
    """Hangover as gap-bridging: non-speech runs of <= `hangover` frames that
    sit BETWEEN two speech runs are filled in (intra-utterance pauses survive).
    Leading/trailing silence is never marked speech, so inter-turn gaps are
    preserved for turn segmentation."""
    out = mask.copy()
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= hangover:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _mask_to_segments(mask: np.ndarray, frame_s: float) -> list[tuple[float, float]]:
    segs: list[tuple[float, float]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append((i * frame_s, j * frame_s))
            i = j
        else:
            i += 1
    return segs


def get_speech_frames(audio: np.ndarray, sr: int = 16000,
                      frame_ms: float = 30.0) -> np.ndarray:
    """Return a boolean array, one entry per frame_ms frame: True = speech."""
    audio = np.asarray(audio, dtype=np.float64)
    frame_len = max(8, int(sr * frame_ms / 1000.0))
    n_frames = max(1, int(np.ceil(len(audio) / frame_len))) if len(audio) else 0
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    frames = frame_audio(audio, frame_len, frame_len)
    rms_db, flat, zcr = _frame_features(frames)

    floor_db = np.percentile(rms_db, FLOOR_PERCENTILE) + FLOOR_OFFSET_DB
    mask = (rms_db > floor_db) & (flat < FLATNESS_MAX) & (zcr >= ZCR_MIN) & (zcr <= ZCR_MAX)

    mask = _apply_hangover(mask, HANGOVER_FRAMES)

    # drop segments shorter than the minimum
    frame_s = frame_len / sr
    min_frames = max(1, int(round(MIN_SEGMENT_S / frame_s)))
    keep = np.zeros_like(mask)
    for start_s, end_s in _mask_to_segments(mask, frame_s):
        a = int(round(start_s / frame_s))
        b = int(round(end_s / frame_s))
        if (b - a) >= min_frames:
            keep[a:b] = True
    return keep[:n_frames]


def _estimate_f0(seg: np.ndarray, sr: int,
                 fmin: float = 60.0, fmax: float = 400.0) -> float | None:
    """Median autocorrelation f0 over voiced sub-frames, or None."""
    if len(seg) < int(0.08 * sr):
        return None
    frame_len = int(0.040 * sr)
    hop = int(0.020 * sr)
    frames = frame_audio(seg, frame_len, hop)
    lo, hi = int(sr / fmax), min(int(sr / fmin), frame_len - 1)
    if hi <= lo:
        return None
    f0s: list[float] = []
    for f in frames:
        f = f - f.mean()
        energy = float(np.mean(f ** 2))
        if energy < 1e-6:
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        ac = ac / (ac[0] + 1e-12)
        peak = int(np.argmax(ac[lo:hi])) + lo
        if ac[peak] > 0.3:
            f0s.append(sr / peak)
    if not f0s:
        return None
    return float(np.median(f0s))


def segment_turns(audio: np.ndarray, sr: int = 16000,
                  frame_ms: float = 30.0,
                  merge_gap_s: float = MERGE_GAP_S) -> list[Turn]:
    """Segment audio into speaking turns.

    Gaps shorter than merge_gap_s (default 0.35 s) between speech segments
    are bridged into a single turn.
    """
    audio = np.asarray(audio, dtype=np.float64)
    mask = get_speech_frames(audio, sr=sr, frame_ms=frame_ms)
    frame_len = max(8, int(sr * frame_ms / 1000.0))
    frame_s = frame_len / sr
    segs = _mask_to_segments(mask, frame_s)
    if not segs:
        return []
    merged: list[list[float]] = [list(segs[0])]
    for start_s, end_s in segs[1:]:
        if start_s - merged[-1][1] < merge_gap_s:
            merged[-1][1] = end_s
        else:
            merged.append([start_s, end_s])
    turns: list[Turn] = []
    for start_s, end_s in merged:
        seg = audio[int(start_s * sr): min(len(audio), int(end_s * sr) + 1)]
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12)) if len(seg) else 0.0
        turns.append(Turn(
            start_s=float(start_s),
            end_s=float(end_s),
            rms_db=float(20.0 * np.log10(rms + 1e-12)),
            mean_f0=_estimate_f0(seg, sr),
        ))
    return turns
