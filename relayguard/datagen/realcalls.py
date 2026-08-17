"""Harvest REAL phone-call audio for sim-to-real augmentation (SPEC: real data).

Pipeline: download publicly posted call recordings (YouTube via yt-dlp, or
direct audio URLs as a fallback), segment speech with the repo VAD, classify
telephony channel bandwidth, and emit a 4 s-segment dataset + JSONL metadata.

PRIVACY / ETHICS BOUNDARY (hard):
- Audio is used ONLY as acoustic training/eval material (telephony channel
  characteristics). No voice biometrics, no identity processing, no cloning.
- Metadata is anonymous: we store only an opaque video_id, never titles,
  names, or any personal info.

Only publicly posted media is fetched; yt-dlp is run with default behavior
(no age-gate/login/paywall bypass). Failures are skipped, never worked around.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from relayguard.common import TARGET_SR, load_audio, save_wav
from relayguard.context.vad import get_speech_frames

log = logging.getLogger(__name__)

# Segmentation / classification constants
FRAME_MS = 30.0
MERGE_GAP_S = 0.35
PEAK_TARGET = 0.7
RMS_MIN_DB = -40.0
SEG_LEN_S = 4.0
BAND_EDGE_HZ = 3400.0
NARROWBAND_MAX_HF_RATIO = 0.05

_AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".amr", ".aac")


def _video_id(url: str) -> str:
    """Anonymous source id. YouTube ids when parseable, else a short url hash.
    Never a title or any personal info."""
    m = re.search(r"(?:youtube\.com/watch\?.*v=|youtu\.be/|shorts/)([\w-]{6,})", url)
    if m:
        return m.group(1)[:16]
    return "src_" + hashlib.sha1(url.encode()).hexdigest()[:12]


def download_audio(url: str, out_dir: str | Path, timeout_s: int = 60) -> Path:
    """Download publicly posted audio -> 16 kHz mono WAV. Returns wav path.

    YouTube (and other extractor-supported) URLs go through yt-dlp
    (bestaudio, default behavior only). Direct media-file URLs are fetched
    over plain HTTP. Conversion to 16 kHz mono WAV uses ffmpeg.
    Raises RuntimeError on any failure — callers skip the source.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = _video_id(url)
    wav_out = out_dir / f"{vid}.wav"
    tmp = Path(tempfile.mkdtemp(prefix="rg_dl_", dir=str(out_dir)))
    try:
        is_direct = url.split("?")[0].lower().endswith(_AUDIO_EXTS)
        if is_direct:
            src = tmp / ("src" + Path(url.split("?")[0]).suffix)
            with urllib.request.urlopen(url, timeout=timeout_s) as r, open(src, "wb") as f:
                shutil.copyfileobj(r, f)
        else:
            tmpl = str(tmp / "src.%(ext)s")
            subprocess.run(
                ["yt-dlp", "-f", "bestaudio/best", "--no-playlist",
                 "--socket-timeout", "20", "--retries", "2",
                 "-o", tmpl, url],
                check=True, timeout=timeout_s,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            found = [p for p in tmp.iterdir() if p.name.startswith("src.")]
            if not found:
                raise RuntimeError(f"yt-dlp produced no file for {url}")
            src = found[0]
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-ac", "1", "-ar", str(TARGET_SR), str(wav_out)],
            check=True, timeout=timeout_s,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_out
    except Exception as e:  # skip-fast policy: any failure -> RuntimeError
        raise RuntimeError(f"download_audio failed for {url}: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _runs_from_mask(mask: np.ndarray, frame_s: float) -> list[tuple[float, float]]:
    runs: list[tuple[float, float]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i * frame_s, j * frame_s))
            i = j
        else:
            i += 1
    return runs


def _merge_runs(runs: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    if not runs:
        return []
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] < gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _cut_runs(runs: list[tuple[float, float]], min_s: float,
              max_s: float) -> list[tuple[float, float]]:
    """Split merged speech runs into [min_s, max_s] windows."""
    out: list[tuple[float, float]] = []
    for s, e in runs:
        dur = e - s
        if dur < min_s:
            continue
        if dur <= max_s:
            out.append((s, e))
            continue
        pos = s
        while e - pos >= max_s:
            out.append((pos, pos + max_s))
            pos += max_s
        if e - pos >= min_s:  # short tail: keep only if it stands alone
            out.append((pos, e))
    return out


def segment_speech(wav_path: str | Path, min_s: float = 3.0,
                   max_s: float = 8.0) -> list[Path]:
    """VAD-segment a 16 kHz wav into [min_s, max_s] speech clips.

    Uses relayguard.context.vad.get_speech_frames, merges runs separated by
    <0.35 s, cuts long runs, drops raw segments with RMS < -40 dBFS, then
    peak-normalizes survivors to 0.7. Returns written segment paths
    (in ``<stem>_segments/`` next to the wav).
    """
    wav_path = Path(wav_path)
    audio = load_audio(wav_path)
    mask = get_speech_frames(audio, sr=TARGET_SR, frame_ms=FRAME_MS)
    frame_s = max(8, int(TARGET_SR * FRAME_MS / 1000.0)) / TARGET_SR
    runs = _cut_runs(_merge_runs(_runs_from_mask(mask, frame_s), MERGE_GAP_S),
                     min_s, max_s)
    seg_dir = wav_path.parent / f"{wav_path.stem}_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (s, e) in enumerate(runs):
        seg = audio[int(s * TARGET_SR): min(len(audio), int(e * TARGET_SR))]
        if len(seg) < int(min_s * TARGET_SR):
            continue
        rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-12)
        if rms_db < RMS_MIN_DB:  # near-silence / dead air: skip
            continue
        peak = float(np.max(np.abs(seg)))
        if peak > 0:
            seg = seg * (PEAK_TARGET / peak)
        p = seg_dir / f"{wav_path.stem}_seg{i:03d}.wav"
        save_wav(p, seg.astype(np.float32))
        paths.append(p)
    return paths


def band_energy_ratio(audio: np.ndarray, sr: int = TARGET_SR,
                      edge_hz: float = BAND_EDGE_HZ) -> float:
    """Fraction of total spectral energy above ``edge_hz``."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return 0.0
    win = np.hanning(len(audio))
    power = np.abs(np.fft.rfft(audio * win)) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    hi = float(power[freqs > edge_hz].sum())
    total = float(power.sum()) + 1e-12
    return hi / total


def classify_channel(segment: str | Path | np.ndarray,
                     sr: int = TARGET_SR) -> str:
    """Heuristic channel label: 'narrowband' (telephony-band, <5% of energy
    above 3.4 kHz => classic phone call) vs 'wideband'.

    Narrowband segments are usable as real direct-call examples; wideband
    ones can still feed the relay simulator as source material.
    """
    audio = load_audio(segment) if isinstance(segment, (str, Path)) \
        else np.asarray(segment, dtype=np.float32)
    ratio = band_energy_ratio(audio, sr)
    return "narrowband" if ratio < NARROWBAND_MAX_HF_RATIO else "wideband"


def _fit_4s(audio: np.ndarray) -> np.ndarray:
    n = int(SEG_LEN_S * TARGET_SR)
    if len(audio) >= n:  # center-crop
        start = (len(audio) - n) // 2
        return audio[start:start + n]
    out = np.zeros(n, dtype=np.float32)  # zero-pad tail
    out[:len(audio)] = audio
    return out


def build_real_dataset(source_urls_file: str | Path, out_dir: str | Path,
                       min_s: float = 3.0, max_s: float = 8.0,
                       timeout_s: int = 60) -> dict:
    """End-to-end harvest: urls -> downloads -> segments -> 4 s WAVs + JSONL.

    Appends lines to ``out_dir/realcalls_metadata.jsonl``:
      {"file", "source_url", "video_id", "segment_idx", "channel",
       "duration_s"}
    Failures per source are logged and skipped. Returns a per-source summary
    dict {video_id: {"narrowband": n, "wideband": m, "status": ...}}.
    """
    out_dir = Path(out_dir)
    seg_out = out_dir / "segments"
    seg_out.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "realcalls_metadata.jsonl"
    dl_dir = Path(tempfile.mkdtemp(prefix="rg_sources_", dir=str(out_dir)))

    urls = [ln.strip() for ln in Path(source_urls_file).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    summary: dict[str, dict] = {}
    try:
        with open(meta_path, "a") as meta:
            for url in urls:
                vid = _video_id(url)
                try:
                    wav = download_audio(url, dl_dir, timeout_s=timeout_s)
                    segs = segment_speech(wav, min_s=min_s, max_s=max_s)
                except Exception as e:
                    log.warning("skipping %s: %s", url, e)
                    summary[vid] = {"narrowband": 0, "wideband": 0,
                                    "status": f"failed: {e}"}
                    continue
                counts = {"narrowband": 0, "wideband": 0}
                for idx, seg_path in enumerate(segs):
                    audio = load_audio(seg_path)
                    channel = classify_channel(audio)
                    out_path = seg_out / f"{vid}_{idx:03d}.wav"
                    save_wav(out_path, _fit_4s(audio))
                    counts[channel] += 1
                    meta.write(json.dumps({
                        "file": str(out_path.relative_to(out_dir)),
                        "source_url": url,
                        "video_id": vid,
                        "segment_idx": idx,
                        "channel": channel,
                        "duration_s": round(len(audio) / TARGET_SR, 3),
                    }) + "\n")
                meta.flush()
                summary[vid] = {**counts, "status": "ok"}
    finally:
        shutil.rmtree(dl_dir, ignore_errors=True)
    return summary
