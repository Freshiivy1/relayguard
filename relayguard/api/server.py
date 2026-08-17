"""RelayGuard FastAPI service (C5).

Endpoints:
  GET  /health              -> service + module availability status
  POST /analyze             -> full detection pipeline -> Verdict JSON
  POST /enroll              -> enroll voiceprint from multiple WAVs
  POST /challenge/start     -> start a random-digit challenge session
  POST /challenge/respond   -> answer a challenge with a WAV utterance
  Learning mode:
  POST /training/upload     -> upload a labeled sample (QC + store)
  GET  /training/samples    -> list uploaded samples
  DELETE /training/sample/{id}
  POST /training/start      -> start background fine-tuning job
  GET  /training/status     -> job state/progress
  GET  /training/versions   -> artifact versions + active one
  POST /training/activate   -> hot-swap to a version (rollback)
  GET  /training/backend    -> storage backend info

Graceful degradation: relayguard.models.*, relayguard.context.* and
relayguard.biometrics.* are imported lazily behind try/except. When a module
group is unavailable its detectors are skipped and responses carry
``"degraded": true``.

Run with: uvicorn relayguard.api.server:app
"""
from __future__ import annotations

import io
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from relayguard.common import (
    TARGET_SR,
    DetectorScore,
    iter_windows,
    load_audio,
    load_audio_bytes,
    load_config,
)
from relayguard.fusion import FusionEngine

app = FastAPI(title="RelayGuard", version="0.1.0")

# Demo console: allow cross-origin browser access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------- #
# Config / state
# ----------------------------------------------------------------------- #
# App root = directory containing the ``relayguard`` package (…/relayguard/api/server.py).
_APP_ROOT = Path(__file__).resolve().parents[2]

_CONFIG: dict = load_config(os.environ.get("RELAYGUARD_CONFIG") or None)


def _artifacts_dir() -> Path:
    """Trained-artifact directory. RELAYGUARD_ARTIFACTS wins; otherwise prefer
    ``<app_root>/artifacts`` so the server works from any CWD, falling back to
    the historical ``./artifacts`` default."""
    env = os.environ.get("RELAYGUARD_ARTIFACTS")
    if env:
        return Path(env)
    app_default = _APP_ROOT / "artifacts"
    if app_default.exists():
        return app_default
    return Path("./artifacts")


def _static_dir() -> Path:
    return Path(os.environ.get("RELAYGUARD_STATIC", str(_APP_ROOT / "static")))


def _versions_dir() -> Path:
    return Path(os.environ.get("RELAYGUARD_VERSIONS", str(_APP_ROOT / "versions")))


def _anchor_dir() -> Path:
    return Path(os.environ.get("RELAYGUARD_ANCHOR", str(_APP_ROOT / "anchor_data")))


# ----------------------------------------------------------------------- #
# Detector registry: which artifact directory the analysis pipeline uses.
# Training hot-swaps this after a successful fine-tune / on rollback.
# ----------------------------------------------------------------------- #
class DetectorRegistry:
    """Thread-safe holder of the active model-artifacts directory.

    analyze() resolves window detectors through this registry, so swapping
    the directory (new trained version, or rollback) takes effect live."""

    def __init__(self, base_dir: Path) -> None:
        self._lock = threading.Lock()
        self._dir = base_dir

    def dir(self) -> Path:
        with self._lock:
            return self._dir

    def reload(self, path: str | Path) -> None:
        """Point the pipeline at a different artifacts directory and drop
        cached detector instances so they reload from the new location."""
        p = Path(path)
        with self._lock:
            self._dir = p
            _WINDOW_DETECTOR_CACHE.clear()


_REGISTRY = DetectorRegistry(_artifacts_dir())


def _init_registry_from_versions() -> None:
    """On startup, honor versions/current.json if it points at a valid
    trained version (survives restarts)."""
    try:
        from relayguard.training import VERSION_MODEL_FILES, read_current, version_dir_for

        v = read_current(_versions_dir())
        if v > 0:
            vdir = version_dir_for(_versions_dir(), _artifacts_dir(), v)
            if all((vdir / f).exists() for f in VERSION_MODEL_FILES):
                _REGISTRY.reload(vdir)
    except Exception:
        pass


def _resolve_config_paths(cfg: dict) -> None:
    """Make artifact-relative paths in the config CWD-independent: if
    fusion.calibrators_path does not exist as written, look for the file (by
    basename) inside the artifacts directory."""
    fcfg = cfg.get("fusion", {})
    cal = fcfg.get("calibrators_path")
    if cal and not Path(cal).exists():
        cand = _artifacts_dir() / Path(cal).name
        if cand.exists():
            fcfg["calibrators_path"] = str(cand.resolve())
        else:
            fcfg.pop("calibrators_path", None)


_resolve_config_paths(_CONFIG)


def _vp_dir() -> Path | None:
    d = os.environ.get("RELAYGUARD_VP_DIR")
    return Path(d) if d else None


# In-memory stores (voiceprints + challenge sessions).
_VP_STORE: dict[str, np.ndarray] = {}
_SESSIONS: dict[str, dict] = {}

# Lazy singletons.
_ENGINE: FusionEngine | None = None
_VERIFIER: Any = None
_VERIFIER_TRIED = False
_WINDOW_DETECTOR_CACHE: dict[str, list] = {}

_init_registry_from_versions()


def _engine() -> FusionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FusionEngine(_CONFIG)
        fuser_path = _artifacts_dir() / "fuser.joblib"
        try:
            if fuser_path.exists():
                _ENGINE.load_fuser(fuser_path)
        except Exception:
            pass
    return _ENGINE


# ----------------------------------------------------------------------- #
# Lazy optional-module loading (built in parallel by other agents)
# ----------------------------------------------------------------------- #
def _module_available(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False


def _module_status() -> dict[str, bool]:
    return {
        "models": _module_available("relayguard.models.detectors")
        or _module_available("relayguard.models.train"),
        "context": _module_available("relayguard.context"),
        "biometrics": _module_available("relayguard.biometrics"),
    }


def _get_verifier():
    """Lazy SpeakerVerifier; None when biometrics are unavailable."""
    global _VERIFIER, _VERIFIER_TRIED
    if _VERIFIER_TRIED:
        return _VERIFIER
    _VERIFIER_TRIED = True
    try:
        from relayguard.biometrics import verifier as _v

        cls = getattr(_v, "SpeakerVerifier", None) or getattr(_v, "Verifier", None)
        _VERIFIER = cls() if cls else None
    except Exception:
        _VERIFIER = None
    return _VERIFIER


def _load_window_detectors() -> tuple[list, bool]:
    """Instantiate available window-level detectors (CNN/GBM from artifacts,
    rule detectors always if the models module imports). Missing artifacts or
    import failures are skipped silently (graceful degradation).

    Returns (detectors, has_model_artifacts): has_model_artifacts is True when
    at least one CNN or GBM artifact-backed detector was loaded.
    """
    active_dir = _REGISTRY.dir()
    art = str(active_dir.resolve())
    if art in _WINDOW_DETECTOR_CACHE:
        return _WINDOW_DETECTOR_CACHE[art]
    detectors: list = []
    has_artifacts = False
    mod = None
    for name in ("relayguard.models.detectors", "relayguard.models.train"):
        try:
            mod = __import__(name, fromlist=["*"])
            break
        except Exception:
            continue
    if mod is not None:
        cnn_cls = getattr(mod, "CNNDetector", None)
        if cnn_cls is not None:
            try:
                # CNNDetector expects the artifacts DIRECTORY (it locates
                # cnn.onnx / cnn.pt inside), not the file path itself. ONNX is
                # the runtime artifact in the torch-free container; cnn.pt is
                # the torch fallback (training/dev checkouts).
                if (active_dir / "cnn.onnx").exists() or (active_dir / "cnn.pt").exists():
                    detectors.append(cnn_cls(str(active_dir)))
                    has_artifacts = True
            except Exception:
                pass
        gbm_cls = getattr(mod, "GBMDetector", None)
        if gbm_cls is not None:
            try:
                if (active_dir / "lgbm.txt").exists():
                    detectors.append(gbm_cls(str(active_dir)))
                    has_artifacts = True
            except Exception:
                pass
        for rule_name in ("BandwidthForensics", "ReverbDetector", "DistortionDetector"):
            cls = getattr(mod, rule_name, None)
            if cls is not None:
                try:
                    detectors.append(cls())
                except Exception:
                    pass
    _WINDOW_DETECTOR_CACHE[art] = (detectors, has_artifacts)
    return detectors, has_artifacts


class _FunctionAnalyzer:
    """Adapter wrapping a module-level analyze(audio, sr) function.

    Context/biometrics modules ship plain functions, not classes; this gives
    them the same call surface as class-based analyzers.
    """

    def __init__(self, fn):
        self._fn = fn

    def analyze(self, audio, sr):
        return self._fn(audio, sr)


def _instantiate_analyzers(module_name: str, candidates: tuple[str, ...]) -> list:
    """Best-effort construction of a call-level analyzer from a module."""
    try:
        mod = __import__(module_name, fromlist=["*"])
    except Exception:
        return []
    for attr in candidates:
        cls = getattr(mod, attr, None)
        if cls is None:
            continue
        try:
            return [cls()]
        except Exception:
            continue
    # Preferred fallback: a module-level analyze(audio, sr) function. This is
    # checked BEFORE the class scan so e.g. biometrics.antispoof.analyze is
    # not shadowed by the AASISTHook class in the same module.
    fn = getattr(mod, "analyze", None)
    if callable(fn):
        return [_FunctionAnalyzer(fn)]
    # Fallback: any class in the module exposing an analyze() method.
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and hasattr(obj, "analyze") and obj.__module__ == mod.__name__:
            try:
                return [obj()]
            except Exception:
                continue
    return []


def _load_context_detectors() -> list:
    detectors: list = []
    detectors += _instantiate_analyzers(
        "relayguard.context.conversation",
        ("ConversationAnalyzer", "ConversationContext", "ConversationDetector"),
    )
    detectors += _instantiate_analyzers(
        "relayguard.context.change_detect",
        ("ChannelSwitchDetector", "ChannelSwitchAnalyzer", "ChannelChangeDetector"),
    )
    detectors += _instantiate_analyzers(
        "relayguard.context.scene",
        ("SceneTagger", "SceneAnalyzer", "SceneDetector", "SceneContext"),
    )
    # Biometric call-level cue: quantization-noise / replay forensics.
    detectors += _instantiate_analyzers(
        "relayguard.biometrics.antispoof",
        ("AntispoofAnalyzer", "AntispoofDetector", "ReplayCueDetector"),
    )
    return detectors


# ----------------------------------------------------------------------- #
# Pydantic models
# ----------------------------------------------------------------------- #
class ChallengeStartRequest(BaseModel):
    voiceprint_id: Optional[str] = None


class ChallengeStartResponse(BaseModel):
    session_id: str
    digit_string: str
    expires_at: float


class ChallengeRespondResult(BaseModel):
    session_id: str
    content_match: Optional[bool] = None
    net_speech_s: float
    identity_score: Optional[float] = None
    verdict: str
    relay_context: Optional[dict] = None


class EnrollResponse(BaseModel):
    voiceprint_id: str
    n_files: int


# ----------------------------------------------------------------------- #
# Pipeline helpers
# ----------------------------------------------------------------------- #
def _run_pipeline(audio: np.ndarray) -> tuple[dict, bool]:
    """Run the full detection pipeline; returns (verdict_dict, degraded)."""
    engine = _engine()
    window_detectors, has_model_artifacts = _load_window_detectors()
    context_detectors = _load_context_detectors()

    window_scores: list[list[DetectorScore]] = []
    for win in iter_windows(audio):
        scores: list[DetectorScore] = []
        for det in window_detectors:
            try:
                s = det.detect(win, TARGET_SR)
                if isinstance(s, DetectorScore):
                    scores.append(s)
            except Exception:
                continue
        window_scores.append(scores)

    context_scores: list[DetectorScore] = []
    for det in context_detectors:
        try:
            s = det.analyze(audio, TARGET_SR)
            if isinstance(s, DetectorScore):
                context_scores.append(s)
        except Exception:
            continue

    verdict = engine.fuse(window_scores, context_scores)
    # Degraded when no CNN/GBM artifacts are loaded OR no context detectors.
    degraded = (not has_model_artifacts) or (len(context_detectors) == 0)
    out = verdict.to_dict()
    # Per-window timeline for the console chart (diagnostics attached by
    # FusionEngine.fuse; additive only, verdict semantics unchanged).
    win_fused = getattr(verdict, "window_fused", []) or []
    win_smoothed = getattr(verdict, "window_smoothed", []) or []
    hop_s = float((_CONFIG.get("audio") or {}).get("hop_s", 1.0))
    out["timeline"] = {
        "window_start_s": [round(i * hop_s, 3) for i in range(len(win_fused))],
        "fused": [round(float(s), 4) for s in win_fused],
        "smoothed": [round(float(s), 4) for s in win_smoothed],
        "theta_green": engine.theta_green,
        "theta_red": engine.theta_red,
    }
    return out, degraded


async def _read_upload_audio(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    try:
        return load_audio_bytes(raw, fmt="wav")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}")


def _net_speech_seconds(audio: np.ndarray) -> float:
    """Net voiced speech seconds via context.vad when available, else a simple
    energy-gate fallback."""
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


def _sweep_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in _SESSIONS.items() if s["expires_at"] <= now]
    for sid in expired:
        _SESSIONS.pop(sid, None)


def _load_voiceprint(voiceprint_id: str) -> np.ndarray | None:
    if voiceprint_id in _VP_STORE:
        return _VP_STORE[voiceprint_id]
    d = _vp_dir()
    if d is not None:
        p = d / f"{voiceprint_id}.npy"
        if p.exists():
            try:
                emb = np.load(str(p))
                _VP_STORE[voiceprint_id] = emb
                return emb
            except Exception:
                return None
    return None


# ----------------------------------------------------------------------- #
# Endpoints
# ----------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modules": _module_status()}


@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile | None = File(None),
    x_audio_format: str | None = Header(None),
    x_sample_rate: int | None = Header(None),
) -> dict:
    """Analyze a call recording. Accepts multipart WAV upload (field ``file``)
    or a raw PCM16 body with X-Audio-Format: pcm16 + X-Sample-Rate headers."""
    if file is not None:
        audio = await _read_upload_audio(file)
    else:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="no audio provided")
        if (x_audio_format or "").lower() != "pcm16":
            raise HTTPException(
                status_code=400,
                detail="raw body requires X-Audio-Format: pcm16 header",
            )
        try:
            audio = load_audio_bytes(body, sr=x_sample_rate or TARGET_SR, fmt="pcm16")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}")

    verdict, degraded = _run_pipeline(audio)
    return _finalize_verdict(verdict, degraded, len(audio))


def _finalize_verdict(verdict: dict, degraded: bool, n_samples: int) -> dict:
    verdict["degraded"] = bool(degraded)
    verdict["audio_s"] = round(n_samples / TARGET_SR, 3)
    return verdict


# ----------------------------------------------------------------------- #
# Test console: static UI + bundled demo samples
# ----------------------------------------------------------------------- #
def _samples_manifest() -> list[dict]:
    path = _static_dir() / "samples" / "samples.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    out = []
    for entry in data if isinstance(data, list) else []:
        f = entry.get("file")
        if not f:
            continue
        if (_static_dir() / "samples" / f).exists():
            out.append(entry)
    return out


@app.get("/samples")
def list_samples() -> dict:
    """List bundled demo samples from static/samples/samples.json."""
    return {"samples": _samples_manifest()}


@app.get("/analyze_sample")
def analyze_sample(name: str) -> dict:
    """Run the same pipeline as POST /analyze on a bundled sample WAV."""
    manifest = _samples_manifest()
    by_stem = {Path(e["file"]).stem: e for e in manifest}
    entry = by_stem.get(Path(name).stem)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown sample: {name}")
    path = (_static_dir() / "samples" / entry["file"]).resolve()
    samples_root = (_static_dir() / "samples").resolve()
    if samples_root not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="sample file not found")
    try:
        audio = load_audio(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not decode sample: {exc}")
    verdict, degraded = _run_pipeline(audio)
    out = _finalize_verdict(verdict, degraded, len(audio))
    out["sample"] = entry
    return out


@app.get("/", include_in_schema=False)
def console_index() -> FileResponse:
    """Serve the full test console at the root URL (no redirect). Prefer the
    standalone root index.html (kept identical to static/index.html) so any
    static file server also renders the console; fall back to static/."""
    for candidate in (_APP_ROOT / "index.html", _static_dir() / "index.html"):
        if candidate.exists():
            return FileResponse(str(candidate))
    raise HTTPException(status_code=404, detail="console not installed")


_static = _static_dir()
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(files: list[UploadFile] = File(...)) -> EnrollResponse:
    """Enroll a voiceprint from one or more WAV files."""
    verifier = _get_verifier()
    if verifier is None:
        raise HTTPException(status_code=503, detail="biometrics unavailable")
    audios = [await _read_upload_audio(f) for f in files]
    if not audios:
        raise HTTPException(status_code=400, detail="no audio files provided")
    try:
        emb = np.asarray(verifier.enroll(audios), dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"enrollment failed: {exc}")
    voiceprint_id = uuid.uuid4().hex
    _VP_STORE[voiceprint_id] = emb
    d = _vp_dir()
    if d is not None:
        try:
            d.mkdir(parents=True, exist_ok=True)
            np.save(str(d / f"{voiceprint_id}.npy"), emb)
        except Exception:
            pass
    return EnrollResponse(voiceprint_id=voiceprint_id, n_files=len(audios))


@app.post("/challenge/start", response_model=ChallengeStartResponse)
def challenge_start(body: ChallengeStartRequest) -> ChallengeStartResponse:
    """Start a random-digit challenge session (CSPRNG digits, short TTL)."""
    _sweep_sessions()
    if body.voiceprint_id is not None and _load_voiceprint(body.voiceprint_id) is None:
        raise HTTPException(status_code=404, detail="unknown voiceprint_id")
    bcfg = _CONFIG.get("biometrics", {})
    n_digits = int(bcfg.get("challenge_digits", 5))
    ttl = float(bcfg.get("challenge_ttl_s", 30))
    digits = "".join(secrets.choice("0123456789") for _ in range(n_digits))
    session_id = uuid.uuid4().hex
    expires_at = time.time() + ttl
    _SESSIONS[session_id] = {
        "digit_string": digits,
        "expires_at": expires_at,
        "voiceprint_id": body.voiceprint_id,
        "retries": 0,
    }
    return ChallengeStartResponse(
        session_id=session_id, digit_string=digits, expires_at=expires_at
    )


@app.post("/challenge/respond", response_model=ChallengeRespondResult)
async def challenge_respond(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> ChallengeRespondResult:
    """Submit the challenge utterance (WAV). Unknown session -> 404, expired
    session -> 410."""
    # Look up before sweeping so an expired-but-present session yields 410
    # rather than 404.
    sess = _SESSIONS.get(session_id)
    if sess is None:
        _sweep_sessions()
        raise HTTPException(status_code=404, detail="unknown session_id")
    if sess["expires_at"] <= time.time():
        _SESSIONS.pop(session_id, None)
        raise HTTPException(status_code=410, detail="session expired")
    _sweep_sessions()

    audio = await _read_upload_audio(file)
    bcfg = _CONFIG.get("biometrics", {})
    max_retries = int(bcfg.get("max_retries", 2))
    min_speech = float(bcfg.get("min_net_speech_s", 3.0))

    net_speech_s = _net_speech_seconds(audio)

    # Content check: pluggable via biometrics.challenge when available;
    # default None (system relies on identity score).
    content_match: bool | None = None
    try:
        from relayguard.biometrics import challenge as _ch

        checker = getattr(_ch, "default_content_checker", None)
        if callable(checker):
            content_match = checker(audio, sess["digit_string"])
    except Exception:
        content_match = None

    identity_score: float | None = None
    relay_context: dict | None = None
    vp_id = sess.get("voiceprint_id")
    if vp_id is not None:
        emb = _load_voiceprint(vp_id)
        verifier = _get_verifier()
        if emb is not None and verifier is not None:
            try:
                identity_score = float(verifier.verify(audio, emb))
            except Exception:
                identity_score = None
        # Fused relay context for known voiceprints.
        try:
            verdict, degraded = _run_pipeline(audio)
            verdict["degraded"] = degraded
            relay_context = verdict
        except Exception:
            relay_context = None

    sess["retries"] += 1
    speech_ok = net_speech_s >= min_speech
    identity_ok = identity_score is None or identity_score >= 0.5
    content_ok = content_match is None or content_match
    if speech_ok and identity_ok and content_ok:
        verdict_str = "accept"
        _SESSIONS.pop(session_id, None)
    elif sess["retries"] <= max_retries:
        verdict_str = "retry"
    else:
        verdict_str = "reject"
        _SESSIONS.pop(session_id, None)

    return ChallengeRespondResult(
        session_id=session_id,
        content_match=content_match,
        net_speech_s=round(net_speech_s, 3),
        identity_score=identity_score,
        verdict=verdict_str,
        relay_context=relay_context,
    )


# ----------------------------------------------------------------------- #
# Learning mode: uploads, background training, versions, hot-swap
# ----------------------------------------------------------------------- #
_STORE: Any = None
_STORE_TRIED = False


def _sample_store():
    """Lazy SampleStore singleton (Supabase when env-configured, else local)."""
    global _STORE, _STORE_TRIED
    if not _STORE_TRIED:
        _STORE_TRIED = True
        try:
            from relayguard.cloud import SampleStore

            _STORE = SampleStore.from_env(_APP_ROOT)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=503, detail=f"sample store unavailable: {exc}")
    return _STORE


_JOB_LOCK = threading.Lock()
_JOB: dict = {"state": "idle", "progress": 0.0, "stage": "idle",
              "job_id": None, "result": None, "error": None}


def _job_state() -> dict:
    with _JOB_LOCK:
        return dict(_JOB)


def _set_job(**fields) -> None:
    with _JOB_LOCK:
        _JOB.update(fields)


def _run_training_thread(job_id: str) -> None:
    from relayguard.training import (
        run_training_job,
        read_current,
        version_dir_for,
    )

    def progress_cb(stage: str, frac: float) -> None:
        _set_job(stage=stage, progress=round(float(frac), 4))

    try:
        report = run_training_job(
            _sample_store(), _artifacts_dir(), _anchor_dir(), _versions_dir(), progress_cb
        )
    except Exception as exc:
        _set_job(state="error", stage="error", error=str(exc), progress=1.0)
        return
    # hot-swap: point the live pipeline at the freshly trained version
    try:
        active = read_current(_versions_dir())
        vdir = version_dir_for(_versions_dir(), _artifacts_dir(), active)
        _REGISTRY.reload(vdir)
    except Exception:
        pass
    _set_job(state="done", stage="done", progress=1.0, result=report)


def _encode_wav_bytes(audio: np.ndarray) -> bytes:
    """float32 16kHz mono -> PCM16 WAV bytes (uniform stored format)."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), TARGET_SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


@app.post("/training/upload")
async def training_upload(
    file: UploadFile = File(...),
    label: str = Form(...),
    notes: str = Form(""),
) -> dict:
    """Upload one labeled training sample (normal | relay). Runs QC and stores
    the audio + metadata in the active backend."""
    from relayguard.cloud import MAX_DURATION_S, MIN_DURATION_S, VALID_LABELS, qc_audio

    if label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail=f"label must be one of {list(VALID_LABELS)}")
    audio = await _read_upload_audio(file)
    duration_s = len(audio) / TARGET_SR
    if duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
        raise HTTPException(
            status_code=400,
            detail=f"duration {duration_s:.1f}s outside allowed range "
                   f"[{MIN_DURATION_S:.0f}s, {MAX_DURATION_S:.0f}s]",
        )
    qc, warnings = qc_audio(audio)
    store = _sample_store()
    try:
        row = store.save_sample(
            _encode_wav_bytes(audio), label,
            {"qc": qc, "notes": notes, "duration_s": qc["duration_s"]},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"could not store sample: {exc}")
    return {"sample_id": row["id"], "qc": qc, "warnings": warnings,
            "backend": store.backend_name}


@app.get("/training/samples")
def training_samples() -> dict:
    store = _sample_store()
    try:
        rows = store.list_samples()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"could not list samples: {exc}")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("label", "?")] = counts.get(r.get("label", "?"), 0) + 1
    return {"samples": rows, "backend": store.backend_name, "counts_by_label": counts}


@app.delete("/training/sample/{sample_id}")
def training_delete_sample(sample_id: str) -> dict:
    store = _sample_store()
    try:
        ok = store.delete_sample(sample_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"could not delete sample: {exc}")
    if not ok:
        raise HTTPException(status_code=404, detail=f"unknown sample: {sample_id}")
    return {"deleted": sample_id, "backend": store.backend_name}


def _torch_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


@app.post("/training/start")
def training_start() -> dict:
    """Kick off a background fine-tuning job (409 if one is already running,
    503 in a torch-free deployment — training is the only torch user)."""
    if not _torch_available():
        raise HTTPException(
            status_code=503,
            detail="training requires torch (not installed in this deployment)",
        )
    with _JOB_LOCK:
        if _JOB.get("state") == "running":
            raise HTTPException(status_code=409, detail="a training job is already running")
        job_id = uuid.uuid4().hex
        _JOB.update({"state": "running", "progress": 0.0, "stage": "starting",
                     "job_id": job_id, "result": None, "error": None})
    thread = threading.Thread(target=_run_training_thread, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/training/status")
def training_status() -> dict:
    return _job_state()


@app.get("/training/versions")
def training_versions() -> dict:
    from relayguard.training import list_versions, read_current

    return {"versions": list_versions(_versions_dir()),
            "active": read_current(_versions_dir()),
            "base_version": 0}


@app.post("/training/activate")
def training_activate(version: int = 0) -> dict:
    """Hot-swap the live pipeline to a previous artifact version (0 = the
    base model shipped with the app)."""
    from relayguard.training import (
        VERSION_MODEL_FILES,
        read_current,
        version_dir_for,
        write_current,
    )

    vdir = version_dir_for(_versions_dir(), _artifacts_dir(), version)
    if version < 0 or not vdir.exists() or not all((vdir / f).exists() for f in VERSION_MODEL_FILES):
        raise HTTPException(status_code=404, detail=f"unknown or incomplete version: {version}")
    previous = read_current(_versions_dir())
    write_current(_versions_dir(), version)
    _REGISTRY.reload(vdir)
    return {"active_version": version, "previous": previous, "active_dir": str(vdir)}


@app.get("/training/backend")
def training_backend() -> dict:
    store = _sample_store()
    return {"backend": store.backend_name,
            "supabase_configured": store.supabase_configured}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("relayguard.api.server:app", host="0.0.0.0", port=8000)
