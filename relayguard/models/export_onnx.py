"""Export RelayCNN (full waveform -> logit path) to ONNX (deployment).

The runtime container ships ``artifacts/cnn.onnx`` + onnxruntime only; torch
is *not* installed in the deployed image (training-only dependency). This
script re-exports the ONNX artifact reproducibly from a ``cnn.pt`` checkpoint.

The exported graph covers the WHOLE path CNNDetector feeds at inference:
raw waveform samples (B, 32000) -> log-mel -> CNN -> logit (B,). The log-mel
front end is re-expressed with ONNX-safe ops (the torch.stft version in
cnn.py uses complex tensors, which ONNX/opset-17 cannot represent):

  * STFT framing + Hann window + rfft  ==  a single conv1d (kernel = n_fft,
    stride = hop, pad = n_fft//2) whose 2*(n_fft//2+1) filters are the
    windowed cos / -sin DFT basis vectors. Numerically identical to
    torch.stft(center=True, pad_mode="constant", return_complex=True).
  * mel projection, log(mel + 1e-6), crop to N_FRAMES frames: matmul + log
    + slice.

Usage:
    python -m relayguard.models.export_onnx                       # artifacts/cnn.pt -> artifacts/cnn.onnx
    python -m relayguard.models.export_onnx --ckpt versions/v1/cnn.pt --out versions/v1/cnn.onnx
    python -m relayguard.models.export_onnx --validate            # export + parity report
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from relayguard.models.cnn import (
    FMAX,
    FMIN,
    HOP,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SR,
    RelayCNN,
    _mel_filterbank,
)

N_BINS = N_FFT // 2 + 1          # 257 rfft bins
WINDOW_SAMPLES = 2 * SR          # 32000 = 2 s @ 16 kHz


class WavToLogit(nn.Module):
    """ONNX-exportable clone of ``RelayCNN.forward_wave`` (wav -> logit).

    Input:  ``waveform`` float32 (B, 32000) raw samples.
    Output: ``logit`` float32 (B,).
    """

    def __init__(self, model: RelayCNN):
        super().__init__()
        self.model = model
        window = torch.hann_window(N_FFT)
        n = torch.arange(N_FFT, dtype=torch.float32)
        k = torch.arange(N_BINS, dtype=torch.float32).unsqueeze(1)
        ang = 2.0 * math.pi * k * n / N_FFT                    # (N_BINS, N_FFT)
        w_cos = torch.cos(ang) * window
        w_sin = -torch.sin(ang) * window
        weight = torch.cat([w_cos, w_sin], dim=0).unsqueeze(1)  # (2*N_BINS, 1, N_FFT)
        self.register_buffer("stft_weight", weight)
        self.register_buffer("fb", _mel_filterbank(N_MELS, N_FFT, SR, FMIN, FMAX))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = F.conv1d(wav.unsqueeze(1), self.stft_weight,
                        stride=HOP, padding=N_FFT // 2)        # (B, 2*N_BINS, T')
        re, im = spec[:, :N_BINS], spec[:, N_BINS:]
        power = re * re + im * im                              # (B, N_BINS, T')
        mel = torch.matmul(self.fb, power)                     # (B, N_MELS, T')
        logmel = torch.log(mel + 1e-6)[..., :N_FRAMES]         # (B, N_MELS, 200)
        return self.model(logmel)


def load_wrapper(ckpt_path: str | Path) -> WavToLogit:
    """Load a RelayCNN state_dict checkpoint into the export wrapper."""
    model = RelayCNN()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    wrapper = WavToLogit(model)
    wrapper.eval()
    return wrapper


def export_onnx(ckpt_path: str | Path, out_path: str | Path,
                opset: int = 17) -> Path:
    """Export the wav->logit wrapper to ONNX (dynamic batch dim)."""
    wrapper = load_wrapper(ckpt_path)
    dummy = torch.zeros(1, WINDOW_SAMPLES)
    out_path = Path(out_path)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["waveform"],
        output_names=["logit"],
        dynamic_axes={"waveform": {0: "batch"}, "logit": {0: "batch"}},
        dynamo=False,
    )
    return out_path


def _torch_logits(wrapper: WavToLogit, wavs: np.ndarray) -> np.ndarray:
    """Reference logits via the ORIGINAL torch path (cnn.logmel + RelayCNN)."""
    from relayguard.models.cnn import logmel

    with torch.no_grad():
        mel = logmel(torch.from_numpy(np.ascontiguousarray(wavs, dtype=np.float32)))
        return wrapper.model(mel).numpy()


def _onnx_logits(session, wavs: np.ndarray) -> np.ndarray:
    outs = []
    for i in range(0, len(wavs), 16):
        batch = np.ascontiguousarray(wavs[i : i + 16], dtype=np.float32)
        outs.append(session.run(["logit"], {"waveform": batch})[0])
    return np.concatenate(outs)


def _pad32000(window: np.ndarray) -> np.ndarray:
    wav = np.asarray(window, dtype=np.float32).ravel()[:WINDOW_SAMPLES]
    if wav.size < WINDOW_SAMPLES:
        wav = np.pad(wav, (0, WINDOW_SAMPLES - wav.size))
    return wav


def validate(onnx_path: str | Path, ckpt_path: str | Path,
             samples_dir: str | Path | None = None) -> float:
    """Parity check: ONNX (onnxruntime) vs original torch path.

    Runs 20 random inputs plus every 2 s window of the bundled sample WAVs
    (when samples_dir is given). Returns the max abs logit difference."""
    import onnxruntime as ort

    wrapper = load_wrapper(ckpt_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    batches: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    batches.append(rng.uniform(-1.0, 1.0, (20, WINDOW_SAMPLES)).astype(np.float32))

    if samples_dir is not None:
        from relayguard.common import iter_windows, load_audio

        for wav_path in sorted(Path(samples_dir).glob("*.wav")):
            audio = load_audio(wav_path)
            wins = [_pad32000(w) for w in iter_windows(audio)]
            batches.append(np.stack(wins))

    max_diff = 0.0
    for wavs in batches:
        ref = _torch_logits(wrapper, wavs)
        got = _onnx_logits(session, wavs)
        diff = float(np.max(np.abs(ref - got))) if len(wavs) else 0.0
        max_diff = max(max_diff, diff)
        print(f"  batch n={len(wavs):3d}  max|dlogit| = {diff:.3e}")
    print(f"PARITY: max abs logit difference = {max_diff:.3e} "
          f"({'PASS' if max_diff < 1e-3 else 'FAIL'} vs 1e-3 tolerance)")
    return max_diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/cnn.pt")
    ap.add_argument("--out", default=None,
                    help="ONNX output path (default: <ckpt stem>.onnx)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--validate", action="store_true",
                    help="run ONNX-vs-torch parity after export")
    ap.add_argument("--samples-dir", default="static/samples")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(args.ckpt).with_suffix(".onnx")
    export_onnx(args.ckpt, out, opset=args.opset)
    print(f"exported {args.ckpt} -> {out} (opset {args.opset})")
    if args.validate:
        validate(out, args.ckpt, args.samples_dir)


if __name__ == "__main__":
    main()
