"""Handcrafted acoustic features for speakerphone-relay detection (SPEC 3.4).

Pure numpy/scipy, deterministic, target <50ms per 2s/16kHz window on 2 CPUs.

Feature groups (aligned with info.md section 2 ranked detection angles):
  - Subband energy ratios + band-edge drops (loudspeaker band-limit ~300Hz-3.4kHz
    and double-bandlimit forensics, angle #2).
  - Spectral shape stats (centroid/spread/flatness/rolloff/crest/tilt/moments).
  - Reverb/modulation: SRMR-style modulation energy ratio (angle #3), envelope
    decay slope / T12 after energy peaks (reverb tail), gap spectral flatness.
  - Loudspeaker nonlinearity (angle #4): f0 autocorrelation track on voiced
    frames, THD proxy (energy at 2f0/3f0 relative to f0), waveform crest-factor
    stats, short-term dynamic range, envelope plateau fraction (smart-amp limiter).
  - Codec artifacts (angle #5): high-band envelope decorrelation vs the 1-3.4kHz
    band (double codec pass decorrelates), quantization-noise-floor flatness.
  - NS confounder: noise-floor stationarity + dead-floor fraction (NS gates drive
    the floor to ~0; used to *discount* relay evidence, info.md section 5).
  - Comfort-noise/gating (angle #6): abrupt energy-drop rate at speech offsets,
    stationary synthetic-noise score in gaps.

All features are float32, deterministic, and roughly normalized to [0,1] (a few
signed quantities are clipped to [-1,1]); exact normalization documented per
feature below.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Fixed feature contract
# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    # --- subband energy ratios (fraction of total power) ---
    "subband_ratio_0_300",      # E(0-300Hz)/total
    "subband_ratio_300_1k",     # E(300Hz-1k)/total
    "subband_ratio_1k_3k4",     # E(1k-3.4k)/total
    "subband_ratio_3k4_5k",     # E(3.4k-5k)/total
    "subband_ratio_5k_8k",      # E(5k-8k)/total
    # --- band-edge drops (spectral void detection), dB/40 clipped [-1,1] ---
    "bandedge_drop_300",        # mean logP(300-700Hz) - mean logP(80-300Hz); >0 = HPF
    "bandedge_drop_3400",       # mean logP(2.8-3.4k) - mean logP(3.4-4.6k); >0 = LPF
    "hf_void_frac",             # fraction of 3.4-8k bins with power <1e-3*ref band
    "hf_slope",                 # linreg slope of mean logP over 3-8kHz, *8000/80, [-1,1]
    # --- spectral shape stats (per-frame means unless noted) ---
    "spec_centroid",            # /8000
    "spec_spread",              # /8000
    "spec_flatness",            # Wiener entropy [0,1]
    "spec_rolloff85",           # /8000
    "spec_rolloff95",           # /8000
    "spec_crest",               # max(meanP)/mean(meanP), /50 clipped
    "spec_tilt",                # linreg slope of log-spectrum vs bin, *257/60, [-1,1]
    "spec_skew",                # skewness of mean log-spectrum, /3 clipped [-1,1]
    "spec_kurtosis",            # excess kurtosis of mean log-spectrum, /6 clipped [-1,1]
    # --- reverb / modulation (SRMR-style) ---
    "srmr_mean",                # mean over 4 octave bands of E_mod(3-20Hz)/E_mod(20-160Hz), /6 clipped
    "srmr_max",                 # max over bands, /6 clipped
    "srmr_band1",               # 125-250Hz band ratio, /6 clipped
    "srmr_band4",               # 1-2kHz band ratio, /6 clipped
    "env_decay_slope",          # dB drop from peak to mean level 80-250ms after, /30; 0=reverb tail
    "env_decay_t12",            # median frames(10ms) to drop 12dB after peaks, /30 clipped
    "gap_flatness",             # mean spectral flatness of bottom-30% energy frames
    "mod_syllable_ratio",       # envelope modulation energy 3-6Hz / 0.5-50Hz, *2 clipped
    # --- loudspeaker nonlinearity / dynamics ---
    "f0_mean",                  # mean f0 of voiced frames, /400 clipped
    "f0_std",                   # std of f0 track, /100 clipped
    "voiced_frac",              # fraction of frames with a stable f0 in 70-400Hz
    "thd_proxy",                # log10(1 + (E@2f0+E@3f0)/E@f0) on voiced frames, /1.5 clipped
    "thd2_ratio",               # log10(1 + E@2f0/E@f0), /1.2 clipped
    "thd3_ratio",               # log10(1 + E@3f0/E@f0), /1.2 clipped
    "crest_factor_mean",        # mean per-frame peak/RMS, /8 clipped
    "crest_factor_std",         # std of per-frame crest factor, /4 clipped
    "dyn_range_db",             # p95-p5 of frame RMS dB, /60 clipped
    "plateau_frac",             # fraction of frames within 1.5dB of p95 RMS (limiter signature)
    # --- codec artifacts ---
    "hbcorr_3k4_4k",            # env corr(3.4-4k, 1-3.4k), mapped (c+1)/2
    "hbcorr_4k_8k",             # env corr(4-8k, 1-3.4k), mapped (c+1)/2
    "qnoise_flatness",          # spectral flatness of bottom-5% energy frames
    # --- NS confounder ---
    "noisefloor_var",           # mean variance of silent-frame log-spectra, /20 clipped
    "dead_floor_frac",          # fraction of silent T-F cells with power <1e-6*max
    # --- comfort noise / gating ---
    "gating_drop_rate",         # abrupt (>8dB) energy drops per speech offset, /2 clipped
    "comfort_noise_score",      # mean consecutive-frame spectral corr in gaps, (c+1)/2
    # --- generic signal stats ---
    "mean_energy_db",           # (mean frame RMS dB + 60)/60 clipped
    "zcr_mean",                 # mean zero-crossing rate, /0.5 clipped
    "zcr_std",                  # std of ZCR, /0.5 clipped
    "resonance_peak_db",        # 700-1500Hz peak prominence over flanks, dB/20 clipped
    "flux_mean",                # mean RMS frame-to-frame log-spectral change, /5 clipped
    "flux_std",                 # std of spectral flux, /3 clipped
    "silence_frac",             # fraction of frames 40dB below the loudest frame
]

FEATURE_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}

_EPS = 1e-12

# Main analysis STFT: 32ms frames, 10ms hop (matches the CNN log-mel params).
_N_FFT = 512
_HOP = 160
_WIN = np.hanning(_N_FFT).astype(np.float64)

# Envelope STFT for SRMR-style modulation analysis: 16ms frames, 2ms hop
# (500Hz envelope sampling -> modulation Nyquist 250Hz, covers the 20-160Hz band).
_ENV_N_FFT = 256
_ENV_HOP = 32
_ENV_WIN = np.hanning(_ENV_N_FFT).astype(np.float64)

_FREQS = np.fft.rfftfreq(_N_FFT, 1.0 / 16000.0)
_ENV_FREQS = np.fft.rfftfreq(_ENV_N_FFT, 1.0 / 16000.0)


def _frames(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    """Slice 1-D signal into (n_frames, n) frames via stride tricks."""
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    n_fr = 1 + (len(x) - n) // hop
    shape = (n_fr, n)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _stft_power(x: np.ndarray, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    """|rfft|^2 of windowed frames -> (n_bins, n_frames)."""
    fr = _frames(x, n_fft, hop) * win
    spec = np.fft.rfft(fr, axis=1)
    return (spec.real**2 + spec.imag**2).T


def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs < hi)


def _lin_slope(y: np.ndarray) -> float:
    """Least-squares slope of y vs its index."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    denom = (x * x).sum()
    if denom <= 0:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0.0 when either side is (near-)constant."""
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.sqrt((a * a).sum()))
    nb = float(np.sqrt((b * b).sum()))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float((a * b).sum() / (na * nb))


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(max(v, lo), hi))


# ---------------------------------------------------------------------------
# Group extractors
# ---------------------------------------------------------------------------

def _subband_features(P: np.ndarray, mean_logP: np.ndarray) -> dict[str, float]:
    total = P.sum() + _EPS
    f = _FREQS
    bands = [(0.0, 300.0), (300.0, 1000.0), (1000.0, 3400.0),
             (3400.0, 5000.0), (5000.0, 8000.0)]
    out = {}
    for name, (lo, hi) in zip(
        ["subband_ratio_0_300", "subband_ratio_300_1k", "subband_ratio_1k_3k4",
         "subband_ratio_3k4_5k", "subband_ratio_5k_8k"], bands):
        out[name] = _clip(P[_band_mask(f, lo, hi)].sum() / total)

    def mlog(lo, hi):
        m = _band_mask(f, lo, hi)
        return float(mean_logP[m].mean()) if m.any() else 0.0

    # wide flanking windows: a speaker HPF/LPF tilts a broad region, while a
    # single codec edge is sharp; 40dB across the window maps to 1.0
    out["bandedge_drop_300"] = float(np.clip(
        (mlog(300.0, 700.0) - mlog(80.0, 300.0)) / 40.0, -1.0, 1.0))
    out["bandedge_drop_3400"] = float(np.clip(
        (mlog(2800.0, 3400.0) - mlog(3400.0, 4600.0)) / 40.0, -1.0, 1.0))

    meanP = P.mean(axis=1)
    hf = _band_mask(f, 3400.0, 8000.0)
    ref = _band_mask(f, 300.0, 3400.0)
    ref_p = float(np.median(meanP[ref])) if ref.any() else 0.0
    out["hf_void_frac"] = _clip(
        (meanP[hf] < ref_p * 1e-3).mean() if hf.any() and ref_p > 0 else 0.0)

    slope_band = _band_mask(f, 3000.0, 8000.0)
    # slope is per bin (bin width 31.25Hz): -> per Hz (*32), then normalized
    # so that an 80dB drop across the full 8kHz span maps to 1.0
    out["hf_slope"] = float(np.clip(
        _lin_slope(mean_logP[slope_band]) * 32.0 * 8000.0 / 80.0, -1.0, 1.0))
    return out


def _spectral_stats(P: np.ndarray, mean_logP: np.ndarray) -> dict[str, float]:
    f = _FREQS
    psum = P.sum(axis=0) + _EPS
    centroid = (P * f[:, None]).sum(axis=0) / psum
    spread = np.sqrt((((f[:, None] - centroid[None, :]) ** 2) * P).sum(axis=0) / psum)
    logP = np.log(P + _EPS)
    flatness = np.exp(logP.mean(axis=0)) / (P.mean(axis=0) + _EPS)

    cs = np.cumsum(P, axis=0)
    total = cs[-1:, :] + _EPS
    idx85 = (cs < 0.85 * total).sum(axis=0)
    idx95 = (cs < 0.95 * total).sum(axis=0)
    step = f[1] - f[0]
    roll85 = np.minimum(idx85, len(f) - 1) * step
    roll95 = np.minimum(idx95, len(f) - 1) * step

    meanP = P.mean(axis=1)
    mu = float(mean_logP.mean())
    sd = float(mean_logP.std()) + _EPS
    z = (mean_logP - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean() - 3.0)

    return {
        "spec_centroid": _clip(float(centroid.mean()) / 8000.0),
        "spec_spread": _clip(float(spread.mean()) / 8000.0),
        "spec_flatness": _clip(float(flatness.mean())),
        "spec_rolloff85": _clip(float(roll85.mean()) / 8000.0),
        "spec_rolloff95": _clip(float(roll95.mean()) / 8000.0),
        "spec_crest": _clip(float(meanP.max() / (meanP.mean() + _EPS)) / 50.0),
        "spec_tilt": float(np.clip(_lin_slope(mean_logP) * 257.0 / 60.0, -1.0, 1.0)),
        "spec_skew": float(np.clip(skew / 3.0, -1.0, 1.0)),
        "spec_kurtosis": float(np.clip(kurt / 6.0, -1.0, 1.0)),
    }


def _modulation_spectrum(env: np.ndarray, env_sr: float) -> tuple[np.ndarray, np.ndarray]:
    """FFT magnitude-squared of a zero-mean envelope -> (mod_power, mod_freqs)."""
    env = env - env.mean()
    mod = np.fft.rfft(env)
    mp = mod.real**2 + mod.imag**2
    return mp, np.fft.rfftfreq(len(env), 1.0 / env_sr)


def _srmr_features(x: np.ndarray, sr: int) -> dict[str, float]:
    """SRMR-style: per-octave-band envelope modulation ratio
    E(3-20Hz)/E(20-160Hz). Dry speech peaks ~4Hz (high ratio); reverb smears
    energy into high modulation rates (low ratio)."""
    mag = np.sqrt(_stft_power(x, _ENV_N_FFT, _ENV_HOP, _ENV_WIN) + _EPS)
    env_sr = sr / _ENV_HOP
    bands = [(125.0, 250.0), (250.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0)]
    ratios = []
    for lo, hi in bands:
        m = _band_mask(_ENV_FREQS, lo, hi)
        env = mag[m].sum(axis=0) if m.any() else np.zeros(mag.shape[1])
        mp, mf = _modulation_spectrum(env, env_sr)
        low = mp[_band_mask(mf, 3.0, 20.0)].sum()
        high = mp[_band_mask(mf, 20.0, 160.0)].sum()
        ratios.append(float(low / (high + _EPS)))
    ratios = np.asarray(ratios)

    # broadband syllable-rate modulation strength (3-6Hz band)
    m = _band_mask(_ENV_FREQS, 200.0, 3400.0)
    env = mag[m].sum(axis=0) if m.any() else np.zeros(mag.shape[1])
    mp, mf = _modulation_spectrum(env, env_sr)
    syll = mp[_band_mask(mf, 3.0, 6.0)].sum()
    tot = mp[_band_mask(mf, 0.5, 50.0)].sum()

    return {
        "srmr_mean": _clip(float(ratios.mean()) / 6.0),
        "srmr_max": _clip(float(ratios.max()) / 6.0),
        "srmr_band1": _clip(ratios[0] / 6.0),
        "srmr_band4": _clip(ratios[3] / 6.0),
        "mod_syllable_ratio": _clip(2.0 * float(syll / (tot + _EPS))),
    }


def _decay_features(P: np.ndarray, flatness: np.ndarray) -> dict[str, float]:
    """Reverb-tail features from the 10ms energy envelope."""
    logE = 10.0 * np.log10(P.sum(axis=0) + _EPS)
    peaks, _ = _find_peaks(logE)
    tails, t12s = [], []
    T = len(logE)
    for p in peaks:
        # tail decay: dB drop from the peak to the mean level 80-250ms after
        hi = min(p + 26, T)
        if hi - p >= 8:
            tails.append(float(logE[p] - logE[p + 8:hi].mean()))
        j = p + 1
        while j < min(p + 31, T) and logE[p] - logE[j] < 12.0:
            j += 1
        t12s.append(j - p)
    if tails:
        decay = _clip(float(np.mean(tails)) / 30.0)     # 0 = no decay (reverb)
        t12 = _clip(float(np.median(t12s)) / 30.0)
    else:
        decay, t12 = 0.5, 0.5

    thr = np.percentile(P.sum(axis=0), 30.0)
    gaps = P.sum(axis=0) <= thr
    if gaps.sum() >= 2:
        gap_flat = float(flatness[gaps].mean())
    else:
        gap_flat = float(flatness[np.argmin(P.sum(axis=0))])
    return {
        "env_decay_slope": decay,
        "env_decay_t12": t12,
        "gap_flatness": _clip(gap_flat),
    }


def _find_peaks(x: np.ndarray, prominence: float = 6.0, distance: int = 10):
    """Minimal local-maxima peak picker with prominence + min-distance filtering
    (avoids scipy.signal.find_peaks overhead; deterministic)."""
    idx = [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1]]
    peaks = []
    for i in idx:
        lo = max(0, i - distance * 3)
        hi = min(len(x), i + distance * 3 + 1)
        base = max(x[lo:i].min(initial=x[i]), x[i + 1:hi].min(initial=x[i]))
        if x[i] - base >= prominence:
            peaks.append(i)
    # enforce min distance keeping the higher peak
    kept: list[int] = []
    for i in sorted(peaks, key=lambda p: -x[p]):
        if all(abs(i - k) >= distance for k in kept):
            kept.append(i)
    return np.array(sorted(kept), dtype=int), None


def _f0_thd_features(fr: np.ndarray, rms: np.ndarray, sr: int) -> dict[str, float]:
    """f0 via autocorrelation on voiced frames (energy > median, 70-400Hz);
    THD proxy from the frame periodogram: (E@2f0 + E@3f0) / E@f0."""
    med = float(np.median(rms))
    voiced = (rms > med) & (rms > 1e-4)
    n_fft = 1024
    min_lag = int(sr / 400.0)   # 400Hz
    max_lag = int(sr / 70.0)    # 70Hz
    f0s, thd2s, thd3s = [], [], []
    bw = 25.0                   # Hz half-bandwidth around harmonic
    faxis = np.fft.rfftfreq(n_fft, 1.0 / sr)
    for frame in fr[voiced]:
        frame = frame - frame.mean()
        spec = np.fft.rfft(frame, n=n_fft)
        ac = np.fft.irfft(spec.real**2 + spec.imag**2, n=n_fft)[: _N_FFT]
        if ac[0] <= _EPS:
            continue
        ac = ac / ac[0]
        seg = ac[min_lag:max_lag + 1]
        if len(seg) == 0:
            continue
        lag = int(np.argmax(seg)) + min_lag
        if ac[lag] < 0.3:       # weak periodicity -> treat as unvoiced
            continue
        f0 = sr / lag
        pw = spec.real**2 + spec.imag**2

        def eat(mult):
            m = _band_mask(faxis, mult * f0 - bw, mult * f0 + bw)
            return float(pw[m].sum()) if m.any() else 0.0

        e1 = eat(1.0)
        if e1 <= _EPS:
            continue
        f0s.append(f0)
        thd2s.append(eat(2.0) / e1)
        thd3s.append(eat(3.0) / e1)

    n_frames = len(fr)
    if f0s:
        f0s = np.asarray(f0s)
        t2 = float(np.mean(thd2s))
        t3 = float(np.mean(thd3s))
        # log-scaled: clean speech harmonic ratios are <~1 (decaying
        # harmonics), loudspeaker THD pushes 2f0/3f0 energy well above that
        return {
            "f0_mean": _clip(float(f0s.mean()) / 400.0),
            "f0_std": _clip(float(f0s.std()) / 100.0),
            "voiced_frac": _clip(len(f0s) / max(n_frames, 1)),
            "thd_proxy": _clip(float(np.log10(1.0 + t2 + t3)) / 1.5),
            "thd2_ratio": _clip(float(np.log10(1.0 + t2)) / 1.2),
            "thd3_ratio": _clip(float(np.log10(1.0 + t3)) / 1.2),
        }
    return {
        "f0_mean": 0.0, "f0_std": 0.0, "voiced_frac": 0.0,
        "thd_proxy": 0.0, "thd2_ratio": 0.0, "thd3_ratio": 0.0,
    }


def _dynamics_features(fr: np.ndarray, rms: np.ndarray,
                       rms_db: np.ndarray) -> dict[str, float]:
    peak = np.abs(fr).max(axis=1)
    crest = peak / (rms + _EPS)
    p95 = float(np.percentile(rms_db, 95.0))
    p5 = float(np.percentile(rms_db, 5.0))
    active = rms_db > -50.0
    plateau = float(((rms_db >= p95 - 1.5) & active).sum() / max(active.sum(), 1))
    zcr = (np.abs(np.diff(np.signbit(fr).astype(np.int8), axis=1)).mean(axis=1)) / 2.0
    return {
        "crest_factor_mean": _clip(float(crest.mean()) / 8.0),
        "crest_factor_std": _clip(float(crest.std()) / 4.0),
        "dyn_range_db": _clip((p95 - p5) / 60.0),
        "plateau_frac": _clip(plateau),
        "mean_energy_db": _clip((float(rms_db.mean()) + 60.0) / 60.0),
        "zcr_mean": _clip(float(zcr.mean()) / 0.5),
        "zcr_std": _clip(float(zcr.std()) / 0.5),
    }


def _codec_ns_gating_features(P: np.ndarray, logP: np.ndarray,
                              flatness: np.ndarray,
                              rms_db: np.ndarray) -> dict[str, float]:
    f = _FREQS
    energy = P.sum(axis=0)

    def env(lo, hi):
        m = _band_mask(f, lo, hi)
        return np.log(P[m].sum(axis=0) + _EPS) if m.any() else np.zeros(P.shape[1])

    e_ref = env(1000.0, 3400.0)
    c1 = _corr(env(3400.0, 4000.0), e_ref)
    c2 = env(4000.0, 8000.0)
    c2c = _corr(c2, e_ref)

    order = np.argsort(energy)
    n_sil = max(2, int(0.2 * len(energy)))
    sil = order[:n_sil]
    q_sil = order[: max(2, int(0.05 * len(energy)))]
    logP_sil = logP[:, sil]
    floor_var = float(logP_sil.var(axis=1).mean())
    dead = float((P[:, sil] < P.max() * 1e-6).mean())

    # consecutive-frame spectral correlation inside gaps (comfort noise =
    # stationary, synthetic-sounding noise -> very high correlation)
    if n_sil >= 3:
        cs = [_corr(logP_sil[:, i], logP_sil[:, i + 1])
              for i in range(n_sil - 1)]
        comfort = float(np.mean(cs))
    else:
        comfort = 0.0

    # abrupt drops aligned with speech offsets (NLP gating / level pumping)
    thr = np.median(energy)
    voiced = energy > thr
    offsets = int(np.sum(voiced[:-1] & ~voiced[1:]))
    drops = int(np.sum(np.diff(rms_db) < -8.0))

    return {
        "hbcorr_3k4_4k": _clip((c1 + 1.0) / 2.0),
        "hbcorr_4k_8k": _clip((c2c + 1.0) / 2.0),
        "qnoise_flatness": _clip(float(flatness[q_sil].mean())),
        "noisefloor_var": _clip(floor_var / 20.0),
        "dead_floor_frac": _clip(dead),
        "comfort_noise_score": _clip((comfort + 1.0) / 2.0),
        "gating_drop_rate": _clip(drops / (offsets + 1.0) / 2.0),
        "silence_frac": _clip(float((rms_db < rms_db.max() - 40.0).mean())),
    }


def _resonance_feature(mean_logP: np.ndarray) -> float:
    """Loudspeaker resonance: peak prominence in 700-1500Hz over flanking
    400-700Hz / 1500-2000Hz bands (info.md sec.1: resonance ~1kHz)."""
    f = _FREQS
    band = _band_mask(f, 700.0, 1500.0)
    flank = _band_mask(f, 400.0, 700.0) | _band_mask(f, 1500.0, 2000.0)
    if not band.any() or not flank.any():
        return 0.0
    prom = float(mean_logP[band].max() - mean_logP[flank].mean())
    return _clip(prom / 20.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(window: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract the fixed handcrafted feature vector from one audio window.

    Args:
        window: mono float array, nominal 2s at `sr` (any length >= 512 ok).
        sr: sample rate (features are tuned for 16000).

    Returns:
        np.float32 array of shape (len(FEATURE_NAMES),) aligned to FEATURE_NAMES.
    """
    x = np.asarray(window, dtype=np.float64).ravel()
    if sr != 16000:
        # features are calibrated for 16kHz; resample rationally if needed
        from scipy.signal import resample_poly
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g)
        sr = 16000
    if len(x) < _N_FFT:
        x = np.pad(x, (0, _N_FFT - len(x)))

    P = _stft_power(x, _N_FFT, _HOP, _WIN)              # (257, T)
    logP = np.log(P + _EPS)
    mean_logP = logP.mean(axis=1)
    flatness = np.exp(logP.mean(axis=0)) / (P.mean(axis=0) + _EPS)

    fr = _frames(x, _N_FFT, _HOP)                        # (T, 512) waveform
    rms = np.sqrt((fr ** 2).mean(axis=1) + _EPS)
    rms_db = 20.0 * np.log10(rms + _EPS)

    feats: dict[str, float] = {}
    feats.update(_subband_features(P, mean_logP))
    feats.update(_spectral_stats(P, mean_logP))
    feats.update(_srmr_features(x, sr))
    feats.update(_decay_features(P, flatness))
    feats.update(_f0_thd_features(fr, rms, sr))
    feats.update(_dynamics_features(fr, rms, rms_db))
    feats.update(_codec_ns_gating_features(P, logP, flatness, rms_db))
    feats["resonance_peak_db"] = _resonance_feature(mean_logP)

    dflux = np.diff(logP, axis=1)
    flux = np.sqrt((dflux ** 2).mean(axis=0)) if dflux.size else np.zeros(1)
    feats["flux_mean"] = _clip(float(flux.mean()) / 5.0)
    feats["flux_std"] = _clip(float(flux.std()) / 3.0)

    out = np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def extract_batch(windows, sr: int = 16000) -> np.ndarray:
    """Extract features for a list/iterable of windows -> (N, n_features) float32."""
    return np.stack([extract_features(w, sr=sr) for w in windows]).astype(np.float32)
