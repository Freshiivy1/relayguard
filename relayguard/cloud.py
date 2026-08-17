"""User-sample storage for the learning mode (uploads + fine-tuning).

Two interchangeable backends behind one interface:

  * Supabase  - when SUPABASE_URL + SUPABASE_KEY env vars are set. Uses plain
    HTTPS REST via httpx (no supabase-py dependency): the "relayguard-audio"
    Storage bucket holds the WAVs and the PostgREST "training_samples" table
    holds one metadata row per upload.
  * local     - fallback: <user_data>/audio/*.wav + <user_data>/samples.jsonl.

Interface (identical in both modes):
    save_sample(audio_bytes, label, meta) -> dict   (the stored row)
    list_samples() -> list[dict]
    delete_sample(sample_id) -> bool
    read_audio(sample) -> bytes
    backend_name -> "supabase" | "local"

Also ships qc_audio(): quick upload quality control (duration, net speech
seconds via context.vad, bandwidth class, clipping, RMS level) producing a
machine-readable qc dict + human-readable warnings.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from relayguard.common import TARGET_SR

BUCKET = "relayguard-audio"
TABLE = "training_samples"

MIN_DURATION_S = 2.0
MAX_DURATION_S = 120.0
VALID_LABELS = ("normal", "relay")


# ---------------------------------------------------------------------------
# QC analysis
# ---------------------------------------------------------------------------

def _net_speech_seconds(audio: np.ndarray) -> float:
    """Net voiced speech seconds via context.vad, with an energy-gate fallback."""
    try:
        from relayguard.context.vad import get_speech_frames

        frames = get_speech_frames(audio, TARGET_SR, frame_ms=30)
        return float(np.asarray(frames).sum()) * 0.03
    except Exception:
        frame = int(0.03 * TARGET_SR)
        n = len(audio) // frame
        if n == 0:
            return 0.0
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
        thr = max(0.01, 0.25 * float(rms.max()))
        return float((rms > thr).sum()) * 0.03


def _bandwidth_class(audio: np.ndarray, sr: int = TARGET_SR) -> tuple[str, dict]:
    """Crude bandwidth classification from long-term band energy fractions."""
    if len(audio) < 256:
        return "unknown", {}
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total = float(spec.sum()) + 1e-12

    def frac(lo: float, hi: float) -> float:
        return float(spec[(freqs >= lo) & (freqs < hi)].sum()) / total

    bands = {
        "below_300hz": round(frac(0, 300), 4),
        "300_3400hz": round(frac(300, 3400), 4),
        "3400_8000hz": round(frac(3400, sr / 2), 4),
    }
    if bands["3400_8000hz"] < 0.02 and bands["below_300hz"] < 0.05:
        cls = "telephony"        # ~300Hz-3.4kHz only (narrowband call)
    elif bands["3400_8000hz"] < 0.08:
        cls = "wideband"
    else:
        cls = "fullband"
    return cls, bands


def qc_audio(audio: np.ndarray, sr: int = TARGET_SR) -> tuple[dict, list[str]]:
    """Quality-control an uploaded sample.

    Returns (qc_dict, warnings): qc_dict is JSON-serializable (duration,
    net speech, bandwidth class + band energies, clipping stats, RMS level);
    warnings are human-readable strings for the console."""
    audio = np.asarray(audio, dtype=np.float32).ravel()
    duration_s = len(audio) / float(sr)
    net_speech_s = _net_speech_seconds(audio)

    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    rms_dbfs = 20.0 * float(np.log10(max(rms, 1e-10)))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clip_frac = float(np.mean(np.abs(audio) >= 0.98)) if len(audio) else 0.0

    bw_class, bands = _bandwidth_class(audio, sr)

    qc = {
        "duration_s": round(duration_s, 3),
        "net_speech_s": round(net_speech_s, 3),
        "rms_dbfs": round(rms_dbfs, 1),
        "peak": round(peak, 4),
        "clip_frac": round(clip_frac, 5),
        "clipping": bool(clip_frac > 0.001),
        "bandwidth": bw_class,
        "band_energy": bands,
    }

    warnings: list[str] = []
    if duration_s < MIN_DURATION_S:
        warnings.append(f"too short ({duration_s:.1f}s, need >= {MIN_DURATION_S:.0f}s)")
    if duration_s > MAX_DURATION_S:
        warnings.append(f"too long ({duration_s:.1f}s, max {MAX_DURATION_S:.0f}s)")
    if net_speech_s < 1.0:
        warnings.append(f"only {net_speech_s:.1f}s speech detected")
    if qc["clipping"]:
        warnings.append(f"clipping detected ({clip_frac * 100:.1f}% of samples at full scale)")
    if rms_dbfs < -40.0:
        warnings.append(f"very low level ({rms_dbfs:.0f} dBFS RMS) - speak closer to the mic")
    elif rms_dbfs > -6.0:
        warnings.append(f"very hot signal ({rms_dbfs:.0f} dBFS RMS) - may be distorted")
    if bw_class == "telephony":
        warnings.append("narrowband audio (little energy above 3.4 kHz)")
    return qc, warnings


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SampleStore:
    """Supabase-or-local sample store (see module docstring)."""

    def __init__(
        self,
        local_dir: str | Path,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        bucket: str = BUCKET,
        table: str = TABLE,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.bucket = bucket
        self.table = table
        self._lock = threading.Lock()
        self._url = (supabase_url or "").rstrip("/") or None
        self._key = supabase_key or None
        self._client = None
        if self._url and self._key:
            import httpx

            self._client = httpx.Client(timeout=30.0)

    @classmethod
    def from_env(cls, app_root: str | Path) -> "SampleStore":
        """Build a store from environment configuration.

        SUPABASE_URL + SUPABASE_KEY set -> Supabase backend; otherwise local
        fallback under <app_root>/user_data (RELAYGUARD_USER_DATA overrides).
        """
        local_dir = os_environ("RELAYGUARD_USER_DATA") or str(Path(app_root) / "user_data")
        return cls(
            local_dir=local_dir,
            supabase_url=os_environ("SUPABASE_URL"),
            supabase_key=os_environ("SUPABASE_KEY"),
        )

    # -- introspection ---------------------------------------------------- #
    @property
    def backend_name(self) -> str:
        return "supabase" if self._client is not None else "local"

    @property
    def supabase_configured(self) -> bool:
        return bool(self._url and self._key)

    # -- public interface ------------------------------------------------- #
    def save_sample(self, audio_bytes: bytes, label: str, meta: dict) -> dict:
        """Store one labeled sample; returns the stored row (with id)."""
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}")
        sample_id = str(uuid.uuid4())
        qc = meta.get("qc") or {}
        row = {
            "id": sample_id,
            "file_path": f"{sample_id}.wav",
            "label": label,
            "duration_s": float(qc.get("duration_s") or meta.get("duration_s") or 0.0),
            "net_speech_s": float(qc.get("net_speech_s") or 0.0),
            "qc_json": qc,
            "notes": str(meta.get("notes") or ""),
            "created_at": _now_iso(),
        }
        if self._client is not None:
            self._supabase_save(row, audio_bytes)
        else:
            self._local_save(row, audio_bytes)
        return row

    def list_samples(self) -> list[dict]:
        if self._client is not None:
            return self._supabase_list()
        return self._local_list()

    def delete_sample(self, sample_id: str) -> bool:
        if self._client is not None:
            return self._supabase_delete(sample_id)
        return self._local_delete(sample_id)

    def read_audio(self, sample: dict) -> bytes:
        if self._client is not None:
            return self._supabase_read(sample)
        return self._local_read(sample)

    # -- local backend ---------------------------------------------------- #
    @property
    def _audio_dir(self) -> Path:
        return self.local_dir / "audio"

    @property
    def _jsonl(self) -> Path:
        return self.local_dir / "samples.jsonl"

    def _local_save(self, row: dict, audio_bytes: bytes) -> None:
        with self._lock:
            self._audio_dir.mkdir(parents=True, exist_ok=True)
            (self._audio_dir / row["file_path"]).write_bytes(audio_bytes)
            with open(self._jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")

    def _local_read_rows_unlocked(self) -> list[dict]:
        if not self._jsonl.exists():
            return []
        rows = []
        with open(self._jsonl) as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        return rows

    def _local_list(self) -> list[dict]:
        with self._lock:
            rows = self._local_read_rows_unlocked()
            # most recent first
            rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
            return rows

    def _local_delete(self, sample_id: str) -> bool:
        with self._lock:
            rows = self._local_read_rows_unlocked()
            keep = [r for r in rows if r.get("id") != sample_id]
            if len(keep) == len(rows):
                return False
            gone = next((r for r in rows if r.get("id") == sample_id), None)
            self.local_dir.mkdir(parents=True, exist_ok=True)
            with open(self._jsonl, "w") as f:
                for r in sorted(keep, key=lambda r: r.get("created_at", "")):
                    f.write(json.dumps(r) + "\n")
            if gone:
                try:
                    (self._audio_dir / gone.get("file_path", "")).unlink(missing_ok=True)
                except Exception:
                    pass
            return True

    def _local_read(self, sample: dict) -> bytes:
        return (self._audio_dir / sample["file_path"]).read_bytes()

    # -- Supabase backend (plain REST) ------------------------------------ #
    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {"apikey": self._key, "Authorization": f"Bearer {self._key}"}
        if extra:
            h.update(extra)
        return h

    def _supabase_save(self, row: dict, audio_bytes: bytes) -> None:
        # 1) WAV -> Storage bucket
        obj_url = f"{self._url}/storage/v1/object/{self.bucket}/{row['file_path']}"
        r = self._client.post(
            obj_url,
            content=audio_bytes,
            headers=self._headers({"Content-Type": "audio/wav", "x-upsert": "true"}),
        )
        if r.status_code >= 300:
            raise RuntimeError(f"storage upload failed ({r.status_code}): {r.text[:300]}")
        # 2) metadata row -> PostgREST table
        db_url = f"{self._url}/rest/v1/{self.table}"
        r = self._client.post(
            db_url,
            json=row,
            headers=self._headers(
                {"Content-Type": "application/json", "Prefer": "return=representation"}
            ),
        )
        if r.status_code >= 300:
            # best-effort rollback of the audio object
            try:
                self._client.delete(obj_url, headers=self._headers())
            except Exception:
                pass
            raise RuntimeError(f"metadata insert failed ({r.status_code}): {r.text[:300]}")
        # created_at comes from the DB default; keep the returned value if given
        try:
            body = r.json()
            if isinstance(body, list) and body and body[0].get("created_at"):
                row["created_at"] = body[0]["created_at"]
        except Exception:
            pass

    def _supabase_list(self) -> list[dict]:
        url = f"{self._url}/rest/v1/{self.table}?select=*&order=created_at.desc"
        r = self._client.get(url, headers=self._headers())
        if r.status_code >= 300:
            raise RuntimeError(f"list failed ({r.status_code}): {r.text[:300]}")
        rows = r.json()
        return rows if isinstance(rows, list) else []

    def _supabase_delete(self, sample_id: str) -> bool:
        rows = [r for r in self._supabase_list() if r.get("id") == sample_id]
        if not rows:
            return False
        url = f"{self._url}/rest/v1/{self.table}?id=eq.{sample_id}"
        r = self._client.delete(url, headers=self._headers())
        if r.status_code >= 300:
            raise RuntimeError(f"delete failed ({r.status_code}): {r.text[:300]}")
        try:
            obj_url = f"{self._url}/storage/v1/object/{self.bucket}/{rows[0]['file_path']}"
            self._client.delete(obj_url, headers=self._headers())
        except Exception:
            pass
        return True

    def _supabase_read(self, sample: dict) -> bytes:
        url = f"{self._url}/storage/v1/object/{self.bucket}/{sample['file_path']}"
        r = self._client.get(url, headers=self._headers())
        if r.status_code >= 300:
            raise RuntimeError(f"audio read failed ({r.status_code}): {r.text[:300]}")
        return r.content


def os_environ(key: str) -> Optional[str]:
    """os.environ.get, isolated for testability."""
    import os

    v = os.environ.get(key)
    return v if v else None
