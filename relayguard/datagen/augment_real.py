"""relayguard.datagen.augment_real — augment the simulated dataset with REAL
call audio (SPEC: real data / sim-to-real augmentation; metadata per SPEC 3.2).

Input: a realcalls harvest dir produced by ``build_real_dataset``
(``realcalls_metadata.jsonl`` + ``segments/*.wav``).

For every usable real segment this writes into the dataset dir:

1. label "direct": the narrowband segment copied as-is (real direct
   telephony audio), and
2. label "relay": the same real voice passed through
   ``chain.simulate_relay`` (codec1 -> loudspeaker -> room -> codec2),
   i.e. REAL speech through the SIMULATED speakerphone relay.

When the narrowband pool is large enough (>= 2x --target-per-class), the two
classes instead use DISJOINT subsets of segments (half copied direct, the
rest relayed), matching the literal "subset / the rest" protocol; otherwise
each segment yields both variants so the >=250-per-class targets are met.
Wideband segments feed only the relay simulator (they are not real
narrowband telephony, but are valid source voices).

Splits (SPEC: domain-shift slice):
- A held-out 15% of real source ids (md5 of video_id) go ENTIRELY to split
  "test" — the domain-shift test slice.
- The remaining 85% of source ids are assigned to train (90%) / dev (10%).
- speaker_id for every real sample is "src_" + sha1(video_id)[:12], so the
  same source can never leak across splits.

Resume-safe: samples whose WAV exists AND already has a metadata line are
skipped. Appends to ``<out-dir>/metadata.jsonl``.

Example:
    python -m relayguard.datagen.augment_real \
        --realcalls-dir .../data/realcalls \
        --out-dir .../data/dataset_sim --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from relayguard import common
from relayguard.datagen import chain

META_KEYS = ["file", "label", "split", "speaker_id", "codec1", "codec2",
             "rt60", "distance_m", "device", "snr_db"]

TEST_FRACTION = 0.15   # held-out real source ids -> test (domain-shift slice)
DEV_FRACTION = 0.10    # of the remaining source ids -> dev; rest -> train


def _src_hash(video_id: str) -> str:
    return "src_" + hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:12]


def _split_for(video_id: str) -> str:
    h = int(hashlib.md5(("realsplit:" + video_id).encode()).hexdigest(), 16) % 100
    if h < int(TEST_FRACTION * 100):
        return "test"
    h2 = int(hashlib.md5(("realdev:" + video_id).encode()).hexdigest(), 16) % 100
    return "dev" if h2 < int(DEV_FRACTION * 100) else "train"


def _load_rows(realcalls_dir: Path) -> list[dict]:
    meta = realcalls_dir / "realcalls_metadata.jsonl"
    rows: list[dict] = []
    with open(meta) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seg = realcalls_dir / row["file"]
            if seg.exists():
                row["_seg_path"] = seg
                rows.append(row)
    return rows


def _done_files(meta_path: Path) -> set[str]:
    done: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _write_sample(out_dir: Path, meta_f, done: set[str], rel: str, label: str,
                  split: str, speaker_id: str, audio: np.ndarray | None,
                  copy_from: Path | None, meta: dict) -> bool:
    """Write one WAV + one metadata.jsonl line (resume-safe)."""
    wav_path = out_dir / rel
    if rel in done and wav_path.exists():
        return False
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_from is not None:
        shutil.copyfile(copy_from, wav_path)
    else:
        common.save_wav(wav_path, audio)
    record = {"file": rel, "label": label, "split": split,
              "speaker_id": speaker_id}
    record.update({k: meta[k] for k in META_KEYS if k in meta})
    record["details"] = meta.get("details", {})
    meta_f.write(json.dumps(record) + "\n")
    meta_f.flush()
    return True


def run(args: argparse.Namespace) -> dict:
    realcalls_dir = Path(args.realcalls_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"

    rows = _load_rows(realcalls_dir)
    narrow = [r for r in rows if r.get("channel") == "narrowband"]
    wide = [r for r in rows if r.get("channel") != "narrowband"]
    print(f"[augment] {len(rows)} segments: {len(narrow)} narrowband, "
          f"{len(wide)} wideband")

    # Disjoint protocol when the pool allows it; otherwise paired variants.
    disjoint = len(narrow) >= 2 * args.target_per_class
    if disjoint:
        order = sorted(narrow, key=lambda r: (r["video_id"], r["segment_idx"]))
        half = len(order) // 2
        direct_rows, relay_narrow = order[:half], order[half:]
    else:
        direct_rows = narrow
        relay_narrow = narrow
    relay_rows = relay_narrow + ([] if disjoint else wide)
    print(f"[augment] protocol={'disjoint' if disjoint else 'paired'}; "
          f"direct sources={len(direct_rows)}, relay sources={len(relay_rows)}")

    done = _done_files(meta_path)
    counts = {"direct": 0, "relay": 0}
    meta_f = open(meta_path, "a")

    for i, row in enumerate(direct_rows):
        vid = row["video_id"]
        rel = f"real_direct/{i:06d}.wav"
        meta = {
            "codec1": "real", "codec2": "none", "rt60": 0.0,
            "distance_m": -1.0, "device": "real_telephony", "snr_db": -1.0,
            "details": {"source": "ncsu_robocall_audio_dataset",
                        "video_id": vid, "channel": row.get("channel"),
                        "duration_s": row.get("duration_s"), "real": True},
        }
        if _write_sample(out_dir, meta_f, done, rel, "direct", _split_for(vid),
                         _src_hash(vid), None, row["_seg_path"], meta):
            counts["direct"] += 1

    for i, row in enumerate(relay_rows):
        vid = row["video_id"]
        rel = f"real_relay/{i:06d}.wav"
        if rel in done and (out_dir / rel).exists():
            continue
        audio = common.load_audio(row["_seg_path"])
        rng = np.random.default_rng(
            int(hashlib.md5(f"{args.seed}:real_relay:{i}".encode()).hexdigest()[:8], 16))
        sim_audio, meta = chain.simulate_relay(audio, rng)
        meta["details"] = dict(meta.get("details", {}))
        meta["details"].update({
            "source": "ncsu_robocall_audio_dataset", "video_id": vid,
            "channel": row.get("channel"), "real": True,
            "note": "real voice through simulated speakerphone relay"})
        if _write_sample(out_dir, meta_f, done, rel, "relay", _split_for(vid),
                         _src_hash(vid), sim_audio, None, meta):
            counts["relay"] += 1

    meta_f.close()
    print(f"[augment] wrote {counts['direct']} real-direct + "
          f"{counts['relay']} real-relay samples into {out_dir}")
    return counts


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description="Augment the dataset with real call audio.")
    ap.add_argument("--realcalls-dir", type=str, required=True,
                    help="dir with realcalls_metadata.jsonl + segments/")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="dataset dir (appends metadata.jsonl)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target-per-class", type=int, default=250,
                    help="pool threshold for the disjoint subset protocol")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
