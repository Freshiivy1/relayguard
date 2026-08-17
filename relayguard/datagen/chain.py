"""relayguard.datagen.chain — relay / direct / hard-negative simulation chains.

Implements SPEC.md section 4-C1. All simulators take
``(clean: np.ndarray, rng: np.random.Generator, profile: dict | None)``
and return ``(audio_4s_float32, meta_dict)`` where meta carries the fields
of SPEC 3.2 (``label``, ``codec1``, ``codec2``, ``rt60``, ``distance_m``,
``device``, ``snr_db``) plus a ``details`` sub-dict of chain parameters.

Conventions:
- Internal audio: mono float32, 16 kHz, exactly 4.0 s (64000 samples).
- ``snr_db == -1.0`` means "no noise applied" (sentinel; schema wants float).
- BOTH classes get codec roundtrips; relay = codec1 + speaker/room + codec2.
"""
from __future__ import annotations

import subprocess
from typing import Any

import numpy as np
import pyroomacoustics as pra
from scipy.signal import butter, fftconvolve, iirpeak, istft, lfilter, stft

SR = 16000
DURATION = 4.0
N_SAMPLES = int(SR * DURATION)

CODECS = ("gsm", "opus", "mulaw")
NO_NOISE_SNR = -1.0  # sentinel for meta["snr_db"] when no noise was mixed

# ---------------------------------------------------------------------------
# Device presets (SPEC 4-C1: >= 8 named loudspeaker presets, randomized params)
# Each preset biases: HPF fc (Hz), LPF fc (Hz), #peaking bands, peak gain (dB),
# soft-clip drive. Peaking freqs are always drawn in 700-2500 Hz.
# ---------------------------------------------------------------------------
DEVICE_PRESETS: dict[str, dict[str, tuple[float, float] | tuple[int, int]]] = {
    "pixel_speaker":   {"hpf": (180, 320), "lpf": (5500, 7500), "n_peaks": (1, 2), "peak_gain": (3, 6), "drive": (1.2, 2.2)},
    "iphone_earpiece": {"hpf": (300, 500), "lpf": (3400, 4800), "n_peaks": (1, 3), "peak_gain": (4, 8), "drive": (1.5, 3.0)},
    "budget_android":  {"hpf": (250, 500), "lpf": (3400, 5500), "n_peaks": (2, 3), "peak_gain": (5, 9), "drive": (2.0, 4.0)},
    "laptop_speaker":  {"hpf": (120, 250), "lpf": (6000, 7500), "n_peaks": (1, 2), "peak_gain": (3, 6), "drive": (1.0, 2.0)},
    "tablet":          {"hpf": (150, 300), "lpf": (5000, 7000), "n_peaks": (1, 2), "peak_gain": (3, 7), "drive": (1.2, 2.5)},
    "watch_speaker":   {"hpf": (380, 500), "lpf": (3400, 4500), "n_peaks": (1, 3), "peak_gain": (6, 9), "drive": (2.5, 4.0)},
    "car_speaker":     {"hpf": (120, 200), "lpf": (6000, 7500), "n_peaks": (1, 2), "peak_gain": (3, 5), "drive": (1.0, 1.8)},
    "bluetooth_mini":  {"hpf": (200, 400), "lpf": (4500, 6500), "n_peaks": (1, 3), "peak_gain": (4, 8), "drive": (1.5, 3.5)},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fit_length(x: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Trim or zero-pad to exactly ``n`` samples (float32)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) >= n:
        return x[:n].copy()
    out = np.zeros(n, dtype=np.float32)
    out[: len(x)] = x
    return out


def loop_to(x: np.ndarray, n: int) -> np.ndarray:
    """Tile ``x`` to length ``n`` (for background material)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def _match_rms(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (y * (_rms(ref) / (_rms(y) + 1e-12))).astype(np.float32)


def _finalize(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Peak-normalize to a random target level in [0.5, 0.95]."""
    peak = float(np.max(np.abs(y))) + 1e-12
    tgt = float(rng.uniform(0.5, 0.95))
    return (y * (tgt / peak)).astype(np.float32)


def _bandpass(x: np.ndarray, lo: float, hi: float, sr: int = SR, order: int = 4) -> np.ndarray:
    hi = min(hi, sr / 2 - 500)
    b, a = butter(order, [lo, hi], "band", fs=sr)
    return lfilter(b, a, x).astype(np.float32)


# ---------------------------------------------------------------------------
# Codec roundtrips (ffmpeg subprocess pipes; SPEC 4-C1)
# ---------------------------------------------------------------------------
def _ffmpeg_pipe(pcm: bytes, enc_args: list[str], dec_args: list[str]) -> np.ndarray:
    enc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         *enc_args, "pipe:1"],
        input=pcm, capture_output=True)
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {enc.stderr[:400]!r}")
    dec = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         *dec_args, "-i", "pipe:0",
         "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=enc.stdout, capture_output=True)
    if dec.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {dec.stderr[:400]!r}")
    return np.frombuffer(dec.stdout, dtype="<i2").astype(np.float32) / 32768.0


def codec_roundtrip(audio: np.ndarray, codec: str, rng: np.random.Generator) -> np.ndarray:
    """Encode -> decode -> resample to 16 kHz via ffmpeg pipes.

    gsm: forced 8 kHz band (libgsm, raw gsm container);
    mulaw: G.711 at 8 kHz (wav container);
    opus: random bitrate 12-64 kbps (ogg container).
    Returns float32 audio, same length as input.
    """
    codec = str(codec)
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype("<i2").tobytes()
    if codec == "gsm":
        out = _ffmpeg_pipe(pcm,
                           ["-ar", "8000", "-ac", "1", "-c:a", "libgsm", "-f", "gsm"],
                           ["-f", "gsm", "-ar", "8000"])
    elif codec == "mulaw":
        out = _ffmpeg_pipe(pcm,
                           ["-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", "-f", "wav"],
                           ["-f", "wav"])
    elif codec == "opus":
        bitrate = int(rng.integers(12, 65))  # 12-64 kbps
        out = _ffmpeg_pipe(pcm,
                           ["-c:a", "libopus", "-b:a", f"{bitrate}k", "-f", "ogg"],
                           ["-f", "ogg"])
    else:
        raise ValueError(f"unknown codec {codec!r}; expected one of {CODECS}")
    return fit_length(out, len(audio))


def _draw_codec(rng: np.random.Generator) -> str:
    return str(rng.choice(CODECS))


# ---------------------------------------------------------------------------
# Loudspeaker model (SPEC 4-C1)
# ---------------------------------------------------------------------------
def _limiter(x: np.ndarray, threshold: float = 0.7, ratio: float = 4.0) -> np.ndarray:
    """Static limiter: samples above ``threshold`` compressed at ``ratio``:1."""
    a = np.abs(x)
    over = a > threshold
    if not np.any(over):
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (threshold + (a[over] - threshold) / ratio)
    return y


def apply_loudspeaker(audio: np.ndarray, rng: np.random.Generator,
                      device: str | None = None, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Phone loudspeaker: band-limit + peaking resonances + tanh soft-clip + limiter."""
    name = str(device) if device else str(rng.choice(list(DEVICE_PRESETS)))
    p = DEVICE_PRESETS[name]
    hpf = float(np.clip(rng.uniform(*p["hpf"]), 120.0, 500.0))
    lpf_cap = min(10000.0, sr / 2 - 500.0)
    lpf = float(np.clip(rng.uniform(*p["lpf"]), 3400.0, lpf_cap))
    if lpf <= hpf + 200:
        lpf = min(lpf_cap, hpf + 2000.0)

    y = lfilter(*butter(4, hpf, "high", fs=sr), audio)
    y = lfilter(*butter(4, lpf, "low", fs=sr), y)

    n_peaks = int(rng.integers(int(p["n_peaks"][0]), int(p["n_peaks"][1]) + 1))
    peaks = []
    for _ in range(n_peaks):
        f0 = float(rng.uniform(700.0, 2500.0))
        q = float(rng.uniform(2.0, 8.0))
        gain_db = float(rng.uniform(*p["peak_gain"]))
        b, a = iirpeak(f0 / (sr / 2), q)
        y = y + (10.0 ** (gain_db / 20.0) - 1.0) * lfilter(b, a, y)
        peaks.append({"f0": round(f0, 1), "q": round(q, 2), "gain_db": round(gain_db, 2)})

    drive = float(rng.uniform(*p["drive"]))
    y = np.tanh(drive * y) / np.tanh(drive)
    y = _limiter(y, threshold=0.7, ratio=4.0)

    info = {"device": name, "hpf_hz": round(hpf, 1), "lpf_hz": round(lpf, 1),
            "drive": round(drive, 2), "peaks": peaks}
    return y.astype(np.float32), info


# ---------------------------------------------------------------------------
# Room simulation (pyroomacoustics ShoeBox)
# ---------------------------------------------------------------------------
def apply_room(audio: np.ndarray, rng: np.random.Generator, sr: int = SR,
               rt60_range: tuple[float, float] = (0.15, 0.9),
               dist_range: tuple[float, float] = (0.3, 3.0),
               dim_range: tuple[float, float] = (2.5, 8.0),
               ) -> tuple[np.ndarray, float, float]:
    """Convolve with a random ShoeBox RIR. Returns (audio, rt60, distance_m).

    RIR generation is fast on this box (<0.1 s at max_order<=40), so no pool
    cache is needed (SPEC's >1 s/RIR condition does not trigger).
    """
    rt60 = float(rng.uniform(*rt60_range))
    dims = [float(rng.uniform(*dim_range)) for _ in range(3)]
    max_dim = max(dims)
    max_order = int(np.clip(np.ceil(rt60 * 343.0 / max_dim * 1.05), 3, 40))
    try:
        e_abs, _ = pra.inverse_sabine(rt60, dims)
    except ValueError:
        # Sabine needs absorption > 1 (very short rt60 in a tiny room, e.g.
        # car cabin). Clamp to near-max absorption; the tail dies fast, which
        # approximates the requested short rt60 well.
        e_abs = 0.95
    room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(float(e_abs)),
                       max_order=max_order)

    margin = min(0.5, min(dims) / 4.0)
    mic = np.array([rng.uniform(margin, d - margin) for d in dims])
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction) + 1e-12
    dist = float(rng.uniform(*dist_range))
    src = np.clip(mic + direction * dist, margin, np.array(dims) - margin)
    dist = float(np.linalg.norm(src - mic))  # effective distance after clamping

    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    rir = room.rir[0][0]
    y = fftconvolve(audio, rir)[: len(audio)]
    y = _match_rms(y, audio)
    return y.astype(np.float32), round(rt60, 3), round(dist, 2)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def make_noise(rng: np.random.Generator, n: int, kind: str | None = None,
               sr: int = SR, lowpass_hz: float | None = None) -> np.ndarray:
    """Gaussian or filtered brown noise, unit-ish RMS."""
    kind = kind or str(rng.choice(["gaussian", "brown"]))
    if kind == "brown":
        x = np.cumsum(rng.standard_normal(n))
        x = x - np.mean(x)
        x = x / (np.max(np.abs(x)) + 1e-12)
        b, a = butter(2, 40.0, "high", fs=sr)  # tame DC/rumble buildup
        x = lfilter(b, a, x)
    else:
        x = rng.standard_normal(n)
    if lowpass_hz is not None:
        b, a = butter(4, min(lowpass_hz, sr / 2 - 500), "low", fs=sr)
        x = lfilter(b, a, x)
    x = x / (_rms(x) + 1e-12)
    return x.astype(np.float32)


def add_noise(audio: np.ndarray, rng: np.random.Generator, snr_db: float,
              **noise_kw: Any) -> np.ndarray:
    """Mix noise at ``snr_db`` relative to the speech RMS."""
    n = make_noise(rng, len(audio), **noise_kw)
    scale = _rms(audio) / (10.0 ** (snr_db / 20.0))
    return (audio + scale * n).astype(np.float32)


# ---------------------------------------------------------------------------
# Small channel blocks
# ---------------------------------------------------------------------------
def apply_mic_eq(audio: np.ndarray, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Mild close-talk mic EQ: gentle bass proximity boost in 100-300 Hz."""
    fc = float(rng.uniform(100.0, 300.0))
    boost_db = float(rng.uniform(2.0, 6.0))
    b, a = butter(2, fc, "low", fs=sr)
    y = audio + (10.0 ** (boost_db / 20.0) - 1.0) * lfilter(b, a, audio)
    return y.astype(np.float32)


def _fixed_eq(audio: np.ndarray, rng: np.random.Generator, hpf: float, lpf: float,
              n_peaks: int = 0, peak_gain: tuple[float, float] = (2.0, 6.0),
              sr: int = SR) -> tuple[np.ndarray, dict]:
    """Constant (time-invariant) non-flat EQ for headset/car hard negatives."""
    lpf = min(lpf, sr / 2 - 500)
    y = lfilter(*butter(4, hpf, "high", fs=sr), audio)
    y = lfilter(*butter(4, lpf, "low", fs=sr), y)
    peaks = []
    for _ in range(n_peaks):
        f0 = float(rng.uniform(700.0, 2500.0))
        gain_db = float(rng.uniform(*peak_gain))
        b, a = iirpeak(f0 / (sr / 2), float(rng.uniform(3.0, 8.0)))
        y = y + (10.0 ** (gain_db / 20.0) - 1.0) * lfilter(b, a, y)
        peaks.append({"f0": round(f0, 1), "gain_db": round(gain_db, 2)})
    return y.astype(np.float32), {"hpf_hz": hpf, "lpf_hz": lpf, "peaks": peaks}


# ---------------------------------------------------------------------------
# Noise-suppression simulation (hardneg_ns): spectral subtraction with a hard
# attenuation floor -> musical-noise artifacts.
# ---------------------------------------------------------------------------
def apply_noise_suppression(audio: np.ndarray, rng: np.random.Generator,
                            sr: int = SR) -> tuple[np.ndarray, dict]:
    in_snr = float(rng.uniform(8.0, 25.0))
    noisy = add_noise(audio, rng, in_snr)
    n_fft, hop = 512, 256
    _, _, Z = stft(noisy, fs=sr, window="hann", nperseg=n_fft,
                   noverlap=n_fft - hop, boundary="zeros", padded=True)
    mag = np.abs(Z)
    noise_floor = np.percentile(mag, 20, axis=1, keepdims=True)  # min-statistics-ish
    atten_db = float(rng.uniform(20.0, 40.0))
    floor_gain = 10.0 ** (-atten_db / 20.0)
    gain = np.clip((mag - 1.5 * noise_floor) / (mag + 1e-12), floor_gain, 1.0)
    _, y = istft(gain * Z, fs=sr, window="hann", nperseg=n_fft,
                 noverlap=n_fft - hop)
    y = _match_rms(fit_length(y, len(audio)), audio)
    return y.astype(np.float32), {"ns_input_snr_db": round(in_snr, 1),
                                  "ns_atten_db": round(atten_db, 1)}


# ---------------------------------------------------------------------------
# TV-like background (hardneg_tv): different-speaker speech if provided, else
# amplitude-modulated band-limited noise with a speech-like 4-8 Hz envelope.
# ---------------------------------------------------------------------------
def synth_tv_bg(rng: np.random.Generator, n: int, sr: int = SR) -> np.ndarray:
    base = make_noise(rng, n, kind="brown")
    base = _bandpass(base, 300.0, 3400.0, sr=sr, order=4)
    t = np.arange(n) / sr
    f_am = float(rng.uniform(4.0, 8.0))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * f_am * t + rng.uniform(0, 2 * np.pi))
    # syllabic on/off gating (~3 syllables/sec), smoothed
    gate_raw = (rng.standard_normal(int(np.ceil(n / 2400)) + 2) > 0.1).astype(np.float32)
    gate = np.repeat(gate_raw, 2400)[:n]
    b, a = butter(2, 6.0, "low", fs=sr)
    gate = np.clip(lfilter(b, a, gate), 0.0, 1.0)
    bg = base * env * (0.25 + 0.75 * gate)
    return (bg / (_rms(bg) + 1e-12)).astype(np.float32)


def _tv_background(rng: np.random.Generator, n: int, profile: dict,
                   sr: int = SR) -> np.ndarray:
    bg_src = profile.get("bg_speech") if profile else None
    if bg_src is not None:
        bg = loop_to(np.asarray(bg_src, dtype=np.float32), n)
        bg = _bandpass(bg, 150.0, 6000.0, sr=sr, order=2)  # light EQ (TV speaker-ish)
    else:
        bg = synth_tv_bg(rng, n, sr=sr)
    return (bg / (_rms(bg) + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# Meta helper
# ---------------------------------------------------------------------------
def _meta(label: str, codec1: str, codec2: str, rt60: float, distance_m: float,
          device: str, snr_db: float, details: dict | None = None) -> dict:
    return {
        "label": label,
        "codec1": str(codec1),
        "codec2": str(codec2),
        "rt60": round(float(rt60), 3),
        "distance_m": round(float(distance_m), 2),
        "device": str(device),
        "snr_db": round(float(snr_db), 1),
        "details": details or {},
    }


# ---------------------------------------------------------------------------
# Simulators (SPEC 4-C1)
# ---------------------------------------------------------------------------
def simulate_direct(clean: np.ndarray, rng: np.random.Generator,
                    profile: dict | None = None) -> tuple[np.ndarray, dict]:
    """Direct call: talker -> phone mic -> ONE codec. Never a speaker/room chain."""
    profile = profile or {}
    x = fit_length(clean)
    codec1 = str(profile.get("codec1") or _draw_codec(rng))
    details: dict[str, Any] = {}

    if rng.random() < 0.5:
        x = apply_mic_eq(x, rng)
        details["mic_eq"] = True
    rt60, dist = 0.0, round(float(rng.uniform(0.05, 0.3)), 2)
    if rng.random() < 0.4:  # light room reverb so classes aren't trivially separable
        x, rt60, dist = apply_room(x, rng, rt60_range=(0.05, 0.3),
                                   dist_range=(0.1, 0.6))
        details["light_reverb"] = True
    snr = NO_NOISE_SNR
    if rng.random() < 0.5:
        snr = float(rng.uniform(15.0, 35.0))
        x = add_noise(x, rng, snr)
    x = codec_roundtrip(x, codec1, rng)
    x = _finalize(x, rng)
    return x, _meta("direct", codec1, "none", rt60, dist, "none", snr, details)


def simulate_relay(clean: np.ndarray, rng: np.random.Generator,
                   profile: dict | None = None) -> tuple[np.ndarray, dict]:
    """Speakerphone relay: codec1 -> loudspeaker -> room+noise -> codec2."""
    profile = profile or {}
    x = fit_length(clean)
    codec1 = str(profile.get("codec1") or _draw_codec(rng))
    codec2 = str(profile.get("codec2") or _draw_codec(rng))

    x = codec_roundtrip(x, codec1, rng)
    x, spk = apply_loudspeaker(x, rng, device=profile.get("device"))
    x, rt60, dist = apply_room(x, rng, rt60_range=(0.15, 0.9), dist_range=(0.3, 3.0))
    snr = float(rng.uniform(5.0, 30.0))
    x = add_noise(x, rng, snr)
    x = codec_roundtrip(x, codec2, rng)
    x = _finalize(x, rng)
    return x, _meta("relay", codec1, codec2, rt60, dist, spk["device"], snr,
                    {"speaker": spk})


def simulate_hardneg_tv(clean: np.ndarray, rng: np.random.Generator,
                        profile: dict | None = None) -> tuple[np.ndarray, dict]:
    """Direct call with TV-like background; the primary talker stays CLEAN direct."""
    profile = profile or {}
    x = fit_length(clean)
    codec1 = str(profile.get("codec1") or _draw_codec(rng))

    bg = _tv_background(rng, N_SAMPLES, profile)
    level_db = float(rng.uniform(-20.0, -8.0))  # bg level relative to speech RMS
    y = x + (10.0 ** (level_db / 20.0)) * _rms(x) * bg
    y = y.astype(np.float32)
    if rng.random() < 0.5:
        y = apply_mic_eq(y, rng)
    y = codec_roundtrip(y, codec1, rng)
    y = _finalize(y, rng)
    details = {"bg_level_db": round(level_db, 1),
               "bg_source": "other_speaker" if profile.get("bg_speech") is not None else "synth_am_noise"}
    return y, _meta("hardneg_tv", codec1, "none", 0.0, 0.1, "none",
                    round(-level_db, 1), details)


def simulate_hardneg_reverb(clean: np.ndarray, rng: np.random.Generator,
                            profile: dict | None = None) -> tuple[np.ndarray, dict]:
    """Direct with STRONG room reverb (rt60 0.4-0.9 s); no speaker EQ, single codec."""
    profile = profile or {}
    x = fit_length(clean)
    codec1 = str(profile.get("codec1") or _draw_codec(rng))
    x, rt60, dist = apply_room(x, rng, rt60_range=(0.4, 0.9), dist_range=(0.5, 2.5))
    x = codec_roundtrip(x, codec1, rng)
    x = _finalize(x, rng)
    return x, _meta("hardneg_reverb", codec1, "none", rt60, dist, "none",
                    NO_NOISE_SNR)


def simulate_hardneg_ns(clean: np.ndarray, rng: np.random.Generator,
                        profile: dict | None = None) -> tuple[np.ndarray, dict]:
    """Direct through aggressive noise-gate simulation (musical-noise artifacts)."""
    profile = profile or {}
    x = fit_length(clean)
    codec1 = str(profile.get("codec1") or _draw_codec(rng))
    x, ns = apply_noise_suppression(x, rng)
    x = codec_roundtrip(x, codec1, rng)
    x = _finalize(x, rng)
    return x, _meta("hardneg_ns", codec1, "none", 0.0, 0.1, "none",
                    ns["ns_input_snr_db"], ns)


def simulate_hardneg_headset(clean: np.ndarray, rng: np.random.Generator,
                             profile: dict | None = None) -> tuple[np.ndarray, dict]:
    """Direct + FIXED non-flat EQ (cheap headset); NO reverb, NO distortion."""
    profile = profile or {}
    x = fit_length(clean)
    codec1 = str(profile.get("codec1") or _draw_codec(rng))
    hpf = float(rng.uniform(180.0, 220.0))   # ~200 Hz HPF
    lpf = float(rng.uniform(5500.0, 6500.0))  # ~6 kHz LPF
    x, eq = _fixed_eq(x, rng, hpf, lpf, n_peaks=int(rng.integers(1, 3)),
                      peak_gain=(2.0, 6.0))
    x = codec_roundtrip(x, codec1, rng)
    x = _finalize(x, rng)
    return x, _meta("hardneg_headset", codec1, "none", 0.0, 0.05, "headset",
                    NO_NOISE_SNR, {"eq": eq})


def simulate_hardneg_car(clean: np.ndarray, rng: np.random.Generator,
                         profile: dict | None = None) -> tuple[np.ndarray, dict]:
    """Car hands-free: very short reverb + mild speaker-ish EQ + road noise.

    Single codec, NO double codec, NO loudspeaker distortion/limiter.
    """
    profile = profile or {}
    x = fit_length(clean)
    codec1 = str(profile.get("codec1") or _draw_codec(rng))
    x, eq = _fixed_eq(x, rng, 150.0, 7000.0, n_peaks=1, peak_gain=(1.0, 4.0))
    x, rt60, dist = apply_room(x, rng, rt60_range=(0.05, 0.15),
                               dist_range=(0.3, 1.0), dim_range=(1.6, 2.6))
    snr = float(rng.uniform(8.0, 20.0))
    x = add_noise(x, rng, snr, kind="brown", lowpass_hz=float(rng.uniform(500.0, 1500.0)))
    x = codec_roundtrip(x, codec1, rng)
    x = _finalize(x, rng)
    return x, _meta("hardneg_car", codec1, "none", rt60, dist, "car_kit", snr,
                    {"eq": eq, "road_noise": True})


SIMULATORS = {
    "relay": simulate_relay,
    "direct": simulate_direct,
    "hardneg_tv": simulate_hardneg_tv,
    "hardneg_reverb": simulate_hardneg_reverb,
    "hardneg_ns": simulate_hardneg_ns,
    "hardneg_headset": simulate_hardneg_headset,
    "hardneg_car": simulate_hardneg_car,
}
