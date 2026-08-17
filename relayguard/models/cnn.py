"""Compact CNN on log-mel spectrograms (SPEC 3.4 / section 4 C2).

RelayCNN: <=400K params. Input: log-mel of a 2s/16kHz window
(n_mels=64, n_fft=512, hop=160 -> 200 frames). Architecture:
Conv2d stack (1->32->64->128, 3x3, BatchNorm, ReLU, MaxPool2d(2))
-> adaptive avg pool over freq-time -> MLP(128->64->1) -> logit.

`logmel` is a torch-only (no torchaudio) log-mel front end.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
N_FFT = 512
HOP = 160
N_MELS = 64
N_FRAMES = 200          # 2s @ 16kHz with hop 160
FMIN = 30.0
FMAX = 8000.0


def _hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sr: int,
                    fmin: float, fmax: float) -> torch.Tensor:
    """Triangular mel filterbank, HTK mel scale -> (n_mels, n_fft//2+1)."""
    n_bins = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sr / 2.0, n_bins)
    mels = torch.linspace(_hz_to_mel(torch.tensor(fmin)),
                          _hz_to_mel(torch.tensor(fmax)), n_mels + 2)
    freqs = _mel_to_hz(mels)
    fb = torch.zeros(n_mels, n_bins)
    for i in range(n_mels):
        lo, c, hi = freqs[i], freqs[i + 1], freqs[i + 2]
        up = (fft_freqs - lo) / (c - lo)
        down = (hi - fft_freqs) / (hi - c)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    # slaney-style area normalization (keeps magnitude scale sane)
    enorm = 2.0 / (freqs[2:] - freqs[:-2])
    fb = fb * enorm.unsqueeze(1)
    return fb


class _MelBank(nn.Module):
    """Cached mel filterbank as a buffer so logmel is torch-only and
    follows the model/device dtype."""

    def __init__(self, n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR):
        super().__init__()
        self.register_buffer("fb", _mel_filterbank(n_mels, n_fft, sr, FMIN, FMAX))
        self.n_fft = n_fft
        self.hop = HOP
        self.register_buffer("window", torch.hann_window(n_fft))


_MEL_BANK: _MelBank | None = None


def _get_bank(device, dtype) -> _MelBank:
    global _MEL_BANK
    if _MEL_BANK is None:
        _MEL_BANK = _MelBank()
    return _MEL_BANK.to(device=device, dtype=dtype)


def logmel(wav_batch: torch.Tensor, n_mels: int = N_MELS) -> torch.Tensor:
    """Log-mel spectrogram of a batch of waveforms (torch-only).

    Args:
        wav_batch: (B, T) float tensor, T nominally 32000 (2s @ 16kHz).
        n_mels: number of mel bins (default 64).

    Returns:
        (B, n_mels, 200) log-mel tensor; time is center-cropped/zero-padded
        to exactly N_FRAMES=200 frames.
    """
    if wav_batch.dim() == 1:
        wav_batch = wav_batch.unsqueeze(0)
    bank = _get_bank(wav_batch.device, wav_batch.dtype)
    spec = torch.stft(
        wav_batch, n_fft=bank.n_fft, hop_length=bank.hop,
        win_length=bank.n_fft, window=bank.window,
        center=True, pad_mode="constant", return_complex=True,
    )
    power = spec.real.pow(2) + spec.imag.pow(2)      # (B, n_fft//2+1, T')
    mel = torch.matmul(bank.fb, power)               # (B, n_mels, T')
    out = torch.log(mel + 1e-6)
    # crop/pad time to exactly N_FRAMES
    t = out.shape[-1]
    if t >= N_FRAMES:
        out = out[..., :N_FRAMES]
    else:
        out = F.pad(out, (0, N_FRAMES - t), value=float(torch.log(torch.tensor(1e-6))))
    return out


class RelayCNN(nn.Module):
    """Compact binary relay classifier on log-mel (SPEC: <=400K params)."""

    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.n_mels = n_mels

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, 200) or (B, 1, n_mels, 200) -> logits (B,)."""
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        h = self.features(mel)
        h = self.pool(h).flatten(1)
        return self.head(h).squeeze(-1)

    def forward_wave(self, wav_batch: torch.Tensor) -> torch.Tensor:
        """Convenience: waveform batch (B, T) -> logits (B,)."""
        return self.forward(logmel(wav_batch, self.n_mels))


def count_parameters(model: nn.Module | None = None) -> int:
    """Total trainable parameter count (defaults to a fresh RelayCNN)."""
    if model is None:
        model = RelayCNN()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
