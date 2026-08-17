"""Internal numpy-only log-mel / framing helpers shared by context detectors.

Kept private (underscore) so the public contract stays vad/conversation/
change_detect/scene only. All functions are deterministic.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-10


def frame_audio(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Slice 1-D audio into (n_frames, frame_len) non/overlapping frames.

    If audio is shorter than frame_len it is zero-padded to a single frame.
    """
    audio = np.asarray(audio, dtype=np.float64)
    frame_len = int(max(1, frame_len))
    hop = int(max(1, hop))
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    n = 1 + (len(audio) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return audio[idx]


def hz_to_mel(f) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int,
                   fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lo, ce, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (fft_freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - ce, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def log_mel(audio: np.ndarray, sr: int = 16000, n_mels: int = 32,
            frame_ms: float = 25.0, hop_ms: float = 10.0,
            n_fft: int | None = None) -> np.ndarray:
    """Log-mel spectrogram, shape (n_frames, n_mels). Pure numpy."""
    frame_len = max(16, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if n_fft is None:
        n_fft = 1
        while n_fft < frame_len:
            n_fft *= 2
    frames = frame_audio(audio, frame_len, hop)
    win = np.hanning(frame_len)
    spec = np.abs(np.fft.rfft(frames * win, n=n_fft, axis=1)) ** 2
    fb = mel_filterbank(sr, n_fft, n_mels)
    return np.log(spec @ fb.T + _EPS)


def rms_db_track(audio: np.ndarray, sr: int = 16000,
                 frame_ms: float = 50.0, hop_ms: float | None = None) -> np.ndarray:
    """Frame-wise RMS in dB."""
    if hop_ms is None:
        hop_ms = frame_ms
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    frames = frame_audio(audio, frame_len, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def rms_db_to_linear(db) -> np.ndarray:
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)
