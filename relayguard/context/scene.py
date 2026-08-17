"""Lightweight background-scene tagging: TV / music / crowd / noise heuristics.

SPEC 4 C3 + info.md 5: scene tags exist primarily as false-positive defense
(fusion guardrail: scene high + coupling low -> void secondary-voice relay
evidence). No pretrained models; heuristic cues on BACKGROUND segments only
(non-primary-speech frames from the VAD):

- tv     : speech-like envelope modulation (4-8 Hz syllabic band) + stable
           program-material loudness + broad (not purely tonal) spectrum.
- music  : strong harmonic periodicity + tempo peaks in envelope autocorrelation
           (0.5-4 Hz beat band) + tonal spectrum stability (low flatness).
- crowd  : incoherent multi-pitch babble - many weak/diverse f0 candidates,
           high spectral flux, diffuse (unpeaked) modulation spectrum.
- noise  : stationary broadband - high spectral flatness, very stable loudness,
           low spectral flux.

Each tag is a heuristic score in [0,1]; the formulas are documented inline and
mirrored in details["logic"]. DetectorScore.score is a mild relay-suspicion
prior: clearly recognized benign program material LOWERS suspicion; ambiguous
or plain-noise backgrounds stay neutral (~0.45). Fusion consumes the tags.
"""
from __future__ import annotations

import numpy as np

from relayguard.common import DetectorScore
from relayguard.context._mel import frame_audio, rms_db_track
from relayguard.context.vad import get_speech_frames

FRAME_MS = 25.0
HOP_MS = 10.0


def _background_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, str]:
    """Select background-only samples. Returns (audio, selection_mode)."""
    mask = get_speech_frames(audio, sr, frame_ms=30.0)
    frame_len = int(0.030 * sr)
    n_frames = int(np.ceil(len(audio) / frame_len)) if len(audio) else 0
    if n_frames == 0:
        return audio, "whole_audio_empty"
    speech_samples = np.zeros(len(audio), dtype=bool)
    for i in np.where(mask)[0]:
        speech_samples[i * frame_len: min(len(audio), (i + 1) * frame_len)] = True
    bg = audio[~speech_samples]
    if len(bg) >= int(1.0 * sr):
        return bg, "vad_gaps"
    # fallback: low-energy frames (quietest third)
    db = rms_db_track(audio, sr, frame_ms=50.0)
    if len(db) >= 6:
        cut = np.percentile(db, 33.0)
        quiet = audio[np.repeat(db <= cut, int(0.050 * sr))[:len(audio)]]
        if len(quiet) >= int(0.5 * sr):
            return quiet, "quiet_frames_fallback"
    return audio, "whole_audio_fallback"


def _modulation_spectrum(env_db: np.ndarray, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """FFT of the loudness envelope -> (freqs_hz, normalized magnitude)."""
    env = env_db - env_db.mean()
    if len(env) < 16 or np.allclose(env, 0.0):
        return np.array([0.0]), np.array([1.0])
    mag = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), d=hop_s)
    total = mag.sum()
    return freqs, mag / (total + 1e-12)


def _band_energy(freqs: np.ndarray, mag: np.ndarray, lo: float, hi: float) -> float:
    sel = (freqs >= lo) & (freqs <= hi)
    return float(mag[sel].sum())


def _peakiness(freqs: np.ndarray, mag: np.ndarray, lo: float, hi: float) -> float:
    """max/mean within a band -> 1 means a single dominant periodicity."""
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        return 0.0
    band = mag[sel]
    return float(band.max() / (band.mean() + 1e-12) / len(band)) if len(band) else 0.0


def _frame_spectra(bg: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (log power spectra, spectral flatness, autocorr harmonicity)."""
    frame_len = int(sr * FRAME_MS / 1000.0)
    hop = int(sr * HOP_MS / 1000.0)
    frames = frame_audio(bg, frame_len, hop)
    win = np.hanning(frame_len)
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    log_spec = np.log(spec + 1e-12)
    flat = np.exp(np.mean(np.log(spec + 1e-12), axis=1)) / (np.mean(spec, axis=1) + 1e-12)

    harm = np.zeros(len(frames))
    lo_lag, hi_lag = int(sr / 400.0), int(sr / 60.0)
    for i, f in enumerate(frames):
        f = f - f.mean()
        if np.mean(f ** 2) < 1e-8:
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        ac = ac / (ac[0] + 1e-12)
        hi = min(hi_lag, len(ac) - 1)
        if hi > lo_lag:
            harm[i] = ac[lo_lag:hi].max()
    return log_spec, flat, harm


def _spectral_flux(log_spec: np.ndarray) -> float:
    if len(log_spec) < 2:
        return 0.0
    diff = np.linalg.norm(np.diff(log_spec, axis=0), axis=1)
    base = np.linalg.norm(log_spec, axis=1).mean()
    return float(diff.mean() / (base + 1e-12))


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def analyze(audio: np.ndarray, sr: int = 16000) -> DetectorScore:
    """Tag the background scene -> DetectorScore(name='scene_context')."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < int(0.5 * sr):
        return DetectorScore(name="scene_context", score=0.45, details={
            "tv": 0.0, "music": 0.0, "crowd": 0.0, "noise": 0.0,
            "error": "audio too short"})

    bg, mode = _background_audio(audio, sr)
    hop_s = HOP_MS / 1000.0
    env_db = rms_db_track(bg, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS)
    log_spec, flat, harm = _frame_spectra(bg, sr)
    freqs, mod = _modulation_spectrum(env_db, hop_s)

    # ---- raw cues ----
    mod_total = _band_energy(freqs, mod, 1.0, 16.0) + 1e-12
    mod48 = _band_energy(freqs, mod, 4.0, 8.0) / mod_total          # syllabic band share
    tempo_peak = _peakiness(freqs, mod, 0.5, 4.0)                   # beat periodicity
    mod_diffuse = 1.0 - _clip01(_peakiness(freqs, mod, 1.0, 16.0) * 3.0)  # spread modulation
    flat_mean = float(flat.mean())
    harm_mean = float(harm.mean())
    flux = _spectral_flux(log_spec)
    loud_std = float(env_db.std()) if len(env_db) else 0.0
    loud_stab = 1.0 - _clip01(loud_std / 12.0)                      # program-material stability
    noise_stab = 1.0 - _clip01(loud_std / 6.0)                      # stricter, for noise
    # multi-pitch babble proxy: fraction of frames with WEAK periodicity
    # (harmonicity in 0.2..0.6) - many competing weak f0 candidates, unlike
    # music (strong single periodicity) or noise (no periodicity at all).
    weak_voiced_frac = float(np.mean((harm > 0.2) & (harm < 0.6)))
    babble = _clip01(weak_voiced_frac * 1.5) if harm_mean > 0.05 else 0.0

    # ---- heuristic tag scores (documented) ----
    # flat modulation spectrum baseline: an unmodulated envelope puts ~0.27 of
    # the 1-16 Hz band energy in 4-8 Hz, so only excess above 0.30 counts.
    mod48_score = _clip01((mod48 - 0.30) / 0.20)
    flux_score = _clip01(flux / 0.15)
    flatness_noise = _clip01((flat_mean - 0.40) / 0.40)   # 1 = broadband noise-like
    tv = _clip01(0.50 * mod48_score
                 + 0.30 * loud_stab
                 + 0.20 * _clip01(1.0 - abs(flat_mean - 0.30) / 0.30))
    music = _clip01(0.50 * _clip01((harm_mean - 0.25) / 0.45)
                    + 0.30 * _clip01(tempo_peak * 4.0)
                    + 0.20 * (1.0 - _clip01(flat_mean / 0.40)))
    crowd = _clip01((0.40 * flux_score + 0.35 * babble + 0.25 * mod_diffuse)
                    * (1.0 - flatness_noise))  # crowd is comb-like, not flat
    noise = _clip01(0.45 * flatness_noise
                    + 0.35 * noise_stab
                    + 0.20 * (1.0 - flux_score))

    # mild relay-suspicion prior: clear benign program material lowers suspicion
    benign = max(tv, music, crowd)
    score = _clip01(0.45 - 0.35 * benign)

    details = {
        "tv": tv, "music": music, "crowd": crowd, "noise": noise,
        "background_mode": mode,
        "background_s": float(len(bg) / sr),
        "cues": {
            "mod_4_8hz_share": float(mod48),
            "tempo_peakiness_0.5_4hz": float(tempo_peak),
            "modulation_diffuseness": float(mod_diffuse),
            "spectral_flatness_mean": flat_mean,
            "harmonicity_mean": harm_mean,
            "spectral_flux": float(flux),
            "loudness_std_db": loud_std,
            "loudness_stability": float(loud_stab),
            "babble_multipitch_proxy": float(babble),
        },
        "logic": {
            "tv": "0.50*(mod48-0.30)/0.20 + 0.30*loud_stab + 0.20*broad-spectrum",
            "music": "0.50*harmonicity + 0.30*tempo_peak + 0.20*(1-flatness/0.4)",
            "crowd": "(0.40*flux/0.15 + 0.35*weak-f0-babble + 0.25*mod_diffuse) * (1-noise_flatness)",
            "noise": "0.45*(flatness-0.4)/0.4 + 0.35*stationarity + 0.20*(1-flux/0.15)",
            "score": "0.45 - 0.35*max(tv,music,crowd); tags are the primary output",
        },
    }
    return DetectorScore(name="scene_context", score=score, details=details)
