"""relayguard.datagen.build_dataset — CLI to generate the labeled dataset.

Scans a LibriSpeech-style tree for FLACs (speaker id = parent-parent folder),
creates 4.0 s samples for every label class via ``chain.py`` simulators, and
writes WAVs + a ``metadata.jsonl`` exactly per SPEC 3.2.

- Speaker-disjoint 80/10/10 train/dev/test splits (md5 hash of speaker id).
- Leave-condition-out: the test split additionally holds out >= 2 device
  presets and >= 1 codec pair (only test relay samples use them; logged to
  ``holdouts.json`` and stdout).
- Resume-safe: samples whose WAV exists AND already has a metadata line are
  skipped.
- Sharding: ``--shard k --num-shards N`` generates indices i with i % N == k.

Example:
    python -m relayguard.datagen.build_dataset \
        --n-per-class 150 --out-dir /tmp/dataset --source-dir .../dev-clean --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from relayguard import common
from relayguard.datagen import chain

LABELS = [
    "relay",
    "direct",
    "hardneg_tv",
    "hardneg_reverb",
    "hardneg_ns",
    "hardneg_headset",
    "hardneg_car",
]

META_KEYS = ["file", "label", "split", "speaker_id", "codec1", "codec2",
             "rt60", "distance_m", "device", "snr_db"]


# ---------------------------------------------------------------------------
# Scanning / splits
# ---------------------------------------------------------------------------
def speaker_split(speaker_id: str) -> str:
    """Deterministic 80/10/10 split by md5 hash of the speaker id."""
    h = int(hashlib.md5(speaker_id.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if h < 80 else ("dev" if h < 90 else "test")


def scan_sources(source_dir: str | Path) -> dict[str, dict[str, list[Path]]]:
    """Return {split: {speaker_id: [flac paths]}} from a LibriSpeech tree."""
    by_split: dict[str, dict[str, list[Path]]] = {"train": {}, "dev": {}, "test": {}}
    for f in sorted(Path(source_dir).glob("**/*.flac")):
        try:
            speaker_id = f.parent.parent.name  # <reader>/<chapter>/<file>.flac
        except IndexError:  # pragma: no cover - defensive
            continue
        by_split[speaker_split(speaker_id)].setdefault(speaker_id, []).append(f)
    return by_split


def choose_holdouts(seed: int) -> tuple[list[str], tuple[str, str]]:
    """Pick >=2 device presets and >=1 codec pair to hold out for test only."""
    rng = np.random.default_rng(seed ^ 0x5EED)
    devices = sorted(str(d) for d in rng.choice(list(chain.DEVICE_PRESETS), size=2, replace=False))
    pairs = [(a, b) for a in chain.CODECS for b in chain.CODECS]
    codec_pair = pairs[int(rng.integers(len(pairs)))]
    return devices, codec_pair


# ---------------------------------------------------------------------------
# Audio slicing
# ---------------------------------------------------------------------------
def load_slice(path: Path, rng: np.random.Generator, n: int) -> np.ndarray:
    """Load a FLAC and take a random n-sample slice (zero-padded if short)."""
    audio = common.load_audio(path)
    if len(audio) >= n:
        start = int(rng.integers(0, len(audio) - n + 1))
        return audio[start : start + n].astype(np.float32)
    return chain.fit_length(audio, n)


def _pick_speaker_clip(split_map: dict[str, list[Path]], rng: np.random.Generator,
                       exclude: str | None = None) -> tuple[str, Path]:
    speakers = [s for s in sorted(split_map) if s != exclude]
    if not speakers:
        speakers = sorted(split_map)
    spk = str(rng.choice(speakers))
    return spk, split_map[spk][int(rng.integers(len(split_map[spk])))]


# ---------------------------------------------------------------------------
# Sample plan / profiles
# ---------------------------------------------------------------------------
def _draw_pair_excluding(rng: np.random.Generator, excluded: tuple[str, str]) -> tuple[str, str]:
    while True:
        pair = (str(rng.choice(chain.CODECS)), str(rng.choice(chain.CODECS)))
        if pair != excluded:
            return pair


def _relay_profile(split: str, rng: np.random.Generator,
                   held_devices: list[str], held_pair: tuple[str, str]) -> dict:
    """Relay profiles implement leave-condition-out: held-out devices / codec
    pairs appear ONLY in the test split (with ~50% probability each)."""
    profile: dict[str, str] = {}
    free_devices = [d for d in chain.DEVICE_PRESETS if d not in held_devices]
    if split == "test":
        if rng.random() < 0.5:
            profile["device"] = str(rng.choice(held_devices))
        else:
            profile["device"] = str(rng.choice(free_devices))
        if rng.random() < 0.5:
            profile["codec1"], profile["codec2"] = held_pair
        else:
            profile["codec1"], profile["codec2"] = _draw_pair_excluding(rng, held_pair)
    else:
        profile["device"] = str(rng.choice(free_devices))
        profile["codec1"], profile["codec2"] = _draw_pair_excluding(rng, held_pair)
    return profile


def _sample_seed(seed: int, label: str, index: int) -> int:
    return int(hashlib.md5(f"{seed}:{label}:{index}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"
    n_samples = int(round(args.duration * chain.SR))

    by_split = scan_sources(args.source_dir)
    if not any(by_split[s] for s in by_split):
        raise SystemExit(f"no FLAC files found under {args.source_dir}")

    held_devices, held_pair = choose_holdouts(args.seed)
    print(f"[build] holdout devices (test-only): {held_devices}")
    print(f"[build] holdout codec pair (test-only): {held_pair}")
    with open(out_dir / "holdouts.json", "w") as f:
        json.dump({"seed": args.seed, "held_devices": held_devices,
                   "held_codec_pair": list(held_pair)}, f, indent=2)

    # resume: collect files that already have a metadata line
    done: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except (json.JSONDecodeError, KeyError):
                    continue

    all_speakers: dict[str, list[Path]] = {}
    for split_map in by_split.values():
        for spk, files in split_map.items():
            all_speakers.setdefault(spk, []).extend(files)

    counts = {label: 0 for label in LABELS}
    written = skipped = 0
    t0 = time.time()
    meta_f = open(meta_path, "a")

    for label in LABELS:
        sim = chain.SIMULATORS[label]
        for i in range(args.n_per_class):
            if i % args.num_shards != args.shard:
                continue
            rel = f"{label}/{i:06d}.wav"
            wav_path = out_dir / rel
            if rel in done and wav_path.exists():
                skipped += 1
                counts[label] += 1
                continue

            rng = np.random.default_rng(_sample_seed(args.seed, label, i))
            # Stratified ~80/10/10 split per class (deterministic in i) so every
            # class gets dev/test coverage when n-per-class >= 10; restricted to
            # NON-EMPTY speaker pools so a speaker never leaks across splits.
            pos = i % 10
            want = "train" if pos < 8 else ("dev" if pos == 8 else "test")
            if by_split[want]:
                split = want
            else:
                avail = [s for s in ("train", "dev", "test") if by_split[s]]
                weights = np.array([{"train": 0.8, "dev": 0.1, "test": 0.1}[s] for s in avail])
                split = str(rng.choice(avail, p=weights / weights.sum()))
            speaker_id, clip = _pick_speaker_clip(by_split[split], rng)
            clean = load_slice(clip, rng, n_samples)

            profile: dict = {}
            if label == "relay":
                profile = _relay_profile(split, rng, held_devices, held_pair)
            if label == "hardneg_tv":
                bg_spk, bg_clip = _pick_speaker_clip(all_speakers, rng, exclude=speaker_id)
                profile["bg_speech"] = load_slice(bg_clip, rng, n_samples)
                profile["bg_speaker_id"] = bg_spk

            audio, meta = sim(clean, rng, profile)
            audio = chain.fit_length(audio, n_samples)
            common.save_wav(wav_path, audio)

            record = {"file": rel, "label": label, "split": split,
                      "speaker_id": speaker_id}
            record.update({k: meta[k] for k in META_KEYS if k in meta})
            record["details"] = meta.get("details", {})
            meta_f.write(json.dumps(record) + "\n")
            meta_f.flush()
            counts[label] += 1
            written += 1
            if written % 100 == 0:
                rate = written / (time.time() - t0)
                print(f"[build] progress: {written} written, {skipped} skipped "
                      f"({rate:.1f} samples/s)")

    meta_f.close()
    dt = time.time() - t0
    summary = {"written": written, "skipped": skipped, "seconds": round(dt, 1),
               "counts_per_class": counts}
    print(f"[build] done in {dt:.1f}s: {written} written, {skipped} skipped (resume)")
    for label in LABELS:
        print(f"[build]   {label:16s} {counts[label]}")
    return summary


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Build the RelayGuard labeled dataset.")
    ap.add_argument("--n-per-class", type=int, required=True,
                    help="samples per label class (across all shards)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--source-dir", type=str, required=True,
                    help="LibriSpeech root containing **/*.flac")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.num_shards:
        ap.error("--shard must be in [0, --num-shards)")
    return build(args)


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
