# SPEC.md — RelayGuard: Speakerphone-Relay Call Detection System
Single source of truth. All modules implement these contracts EXACTLY. Python 3.12, CPU-only target. No librosa/numba dependencies — use numpy/scipy/soundfile/torch only.

## 0. Environment
- Installed: numpy 2.2, scipy 1.16, torch 2.8 (CPU), scikit-learn 1.7, lightgbm 4.6, soundfile, pyroomacoustics, fastapi, uvicorn. ffmpeg 5.1 with libgsm, libopus, pcm_mulaw (G.711).
- speechbrain (for biometrics) — installing separately; biometrics module must degrade gracefully if unavailable.
- Repo: /mnt/agents/output/project (git). Each agent works in a git worktree.

## 1. System overview
```
call audio (WAV/PCM, any sr -> internal 16kHz mono float32)
  -> WindowAnalyzer (2s windows, 1s hop)
      -> AcousticDetectors (CNN model + handcrafted-feature GBM + rule detectors)
      -> ContextAnalyzer (VAD/turn stats, channel-change detection, scene tags)
  -> TemporalSmoother (HMM over per-window scores)
  -> FusionEngine (calibrated score fusion + guardrail rules) -> Verdict: GREEN | CHALLENGE | RED
  -> if CHALLENGE/RED: BiometricChallenge (random digits -> verify -> final verdict)
```

## 2. Repository layout
```
project/
  relayguard/
    __init__.py
    common.py            # audio I/O, resampling, windowing, schemas (OWNER: C1)
    datagen/
      __init__.py
      chain.py           # relay/direct/hard-negative simulation chains (C1)
      build_dataset.py   # CLI: generate labeled dataset + metadata JSONL (C1)
    features/
      __init__.py
      extract.py         # handcrafted feature vector, FEATURE_NAMES contract (C2)
    models/
      __init__.py
      cnn.py             # compact CNN on log-mel (C2)
      train.py           # training CLI for CNN + GBM (C2)
    context/
      __init__.py
      vad.py             # neural-free VAD (energy+spectral) (C3)
      conversation.py    # turn-taking stats, latency, coupling (C3)
      change_detect.py   # mid-call acoustic channel switch detection (C3)
      scene.py           # lightweight scene tagging (TV/music/crowd heuristics) (C3)
    biometrics/
      __init__.py
      verifier.py        # ECAPA-TDNN speaker verification wrapper (C4)
      challenge.py       # random-digit challenge flow state machine (C4)
      antispoof.py       # replay/liveness heuristics + optional AASIST hook (C4)
    fusion/
      __init__.py
      calibrate.py       # Platt scaling per detector (C5)
      engine.py          # fusion + guardrails + 3-state verdict + HMM smoothing (C5)
    api/
      __init__.py
      server.py          # FastAPI: /analyze, /enroll, /challenge, /verify, /health (C5)
    eval/
      __init__.py
      evaluate.py        # metrics harness: hit rate, per-slice FPR, ROC/EER (C2)
  tests/                 # per-module pytest tests (each agent)
  configs/default.yaml
  requirements.txt
  README.md
```

## 3. Data contracts (SACRED — do not change)
### 3.1 Audio
- Internal format: mono float32 numpy array, 16000 Hz, range [-1, 1].
- `relayguard.common.load_audio(path) -> np.ndarray` (any sr -> 16k).
- `relayguard.common.iter_windows(audio, win_s=2.0, hop_s=1.0) -> Iterator[np.ndarray]`.

### 3.2 Dataset sample (datagen output)
- WAV 16kHz mono 16-bit PCM, 4.0 seconds.
- `metadata.jsonl`, one JSON per sample:
```json
{"file": "relay/000123.wav", "label": "relay|direct|hardneg_tv|hardneg_reverb|hardneg_ns|hardneg_headset|hardneg_car",
 "split": "train|dev|test", "speaker_id": "...", "codec1": "gsm|opus|mulaw|none", "codec2": "...",
 "rt60": 0.0, "distance_m": 0.0, "device": "preset_name", "snr_db": 0.0}
```
- Splits: speaker-disjoint; test also holds out >=1 codec pair and >=2 device presets (leave-condition-out).

### 3.3 Detector output
```python
@dataclass
class DetectorScore:      # relayguard.common.DetectorScore
    name: str             # detector id
    score: float          # P(relay) in [0,1], uncalibrated ok
    details: dict         # free-form diagnostics
```
Every window-level detector: `detect(window: np.ndarray, sr: int = 16000) -> DetectorScore`.
Call-level context detectors: `analyze(audio: np.ndarray, sr: int = 16000) -> DetectorScore`.

### 3.4 Handcrafted feature vector (features/extract.py)
- `FEATURE_NAMES: list[str]` — fixed ordered list, ~40-80 dims covering: subband energy ratios (0-300/300-1k/1k-3.4k/3.4k-8k), spectral centroid/flatness/rolloff, SRMR-style modulation-energy ratio, envelope decay stats (reverb tail), THD proxy (2f0/3f0 ratios on voiced frames), crest factor + dynamic range stats, spectral void detection (band-edge drops), noise-floor stationarity (NS confounder), codec-artifact stats (high-band decorrelation), comfort-noise/gating stats.
- `extract_features(window, sr=16000) -> np.ndarray` aligned to FEATURE_NAMES. Must be pure numpy/scipy, <50ms/window on this CPU.

### 3.5 Verdict
```python
@dataclass
class Verdict:            # relayguard.common.Verdict
    state: str            # "GREEN" | "CHALLENGE" | "RED"
    confidence: float     # calibrated [0,1]
    fused_score: float
    detector_scores: list # DetectorScore list
    reason: str           # human-readable explanation
```

## 4. Module responsibilities
### C1 — common.py + datagen
- common.py: load_audio, iter_windows, DetectorScore, Verdict, simple YAML config loader.
- chain.py: `simulate_direct(clean, rng, profile) -> (audio, meta)`, `simulate_relay(clean, rng, profile) -> (audio, meta)`, `simulate_hardneg_<kind>(...)` for tv|reverb|ns|headset|car.
  - Codecs via ffmpeg subprocess pipes: gsm (8k), opus (12-64kbps), pcm_mulaw. Roundtrip encode->decode->resample to 16k. BOTH classes get codec roundtrips; relay gets codec1+codec2 (independent draws).
  - Loudspeaker model: HPF 120-500Hz + LPF 3.4-10kHz (randomized) + 1-3 parametric resonance peaks + tanh soft-clip + dynamic limiter. Device presets >=8 named curves, randomized params.
  - Room: pyroomacoustics shoebox, RT60 0.15-0.9s, speaker-mic distance 0.3-3m.
  - Hard negatives per info.md section 3/5.
- build_dataset.py CLI: `--n-per-class --out-dir --source-dir --seed`; parallelizable via `--shard/--num-shards`; writes WAVs + metadata.jsonl; deterministic per seed. Must resume safely (skip existing files).

### C2 — features + models + eval
- features/extract.py per 3.4.
- models/cnn.py: compact CNN (<=400K params) on 64-bin log-mel (n_fft=512, hop=160), input 2s window; `RelayCNN(n_mels=64)`; `forward(mel) -> logit`.
- train.py CLI: trains (a) RelayCNN (Adam, BCE, 2s windows from dataset WAVs, on-the-fly log-mel via torch) and (b) LightGBM on extracted features; saves `artifacts/cnn.pt` and `artifacts/lgbm.txt` + `artifacts/feature_scaler.joblib`. Handle class balance via weights. CPU-efficient: batch 64, default epochs 15, early stop on dev AUC.
- Also exposes window detectors wrapping trained models: `CNNDetector(model_path)` and `GBMDetector(model_dir)` implementing detect() contract, plus 3 rule-based detectors: `BandwidthForensics`, `ReverbDetector`, `DistortionDetector` (thresholded feature logic from info.md, each outputs DetectorScore).
- eval/evaluate.py CLI: given dataset dir + model artifacts -> overall hit rate @ fixed FPR, ROC/AUC/EER, per-slice (per label + per codec-pair + per device) FPR/recall table, JSON + markdown report to `artifacts/eval_report.md`.

### C3 — context
- vad.py: energy+spectral-flatness VAD, `get_speech_frames(audio, sr, frame_ms=30) -> np.ndarray[bool]`, `segment_turns(...) -> list[Turn(start,end)]`. No external model downloads.
- conversation.py: turn stats (turn count, mean turn len, gap distribution, overlap proxy), response-latency percentiles (needs two-channel? NO — single channel: agent/customer unknown, so compute inter-turn gap stats + speech-rate anomalies), background-coupling proxy (cross-correlation of background segments' envelope with primary speech onsets — "does background react"), voice-count heuristic (spectral-clustering of window embeddings = mean log-mel vectors; report 1 vs >=2 stable clusters + cluster stability over time).
- change_detect.py: delta-BIC-style change-point detection on rolling feature vectors (use features/extract.py vectors); `detect_channel_switches(audio) -> list[float]` timestamps + `ChannelSwitchScore` DetectorScore (no switch + stable channel -> benign).
- scene.py: lightweight tags without big models: TV/music/crowd probability heuristics (spectral stationarity, harmonic music structure, multi-pitch babble estimate, program-material loudness stability). Output DetectorScore name="scene_context" with details={"tv":p,"music":p,"crowd":p}.
- All call-level detectors implement analyze() contract.

### C4 — biometrics
- verifier.py: SpeechBrain ECAPA-TDNN wrapper (`speechbrain/spkrec-ecapa-voxceleb`, local_files_only-friendly, lazy load). `enroll(audios: list[np.ndarray]) -> np.ndarray` (mean L2-normed embedding), `verify(audio, enrolled_emb) -> float` cosine similarity. Graceful ImportError -> raises BiometricsUnavailable. 8kHz input -> resample 16k.
- challenge.py: `ChallengeSession` state machine: generate 5-digit CSPRNG string (nonce, 30s expiry) -> `validate_response(audio) -> ChallengeResult(content_match: bool, net_speech_s: float, identity_score: float|None, verdict: str)`; content check via simple digit-template DTW? NO — expose `content_checker` protocol with a pluggable ASR function; default implementation = always-None (system logs and relies on identity score) with interface ready for Whisper later. Max 2 retries. Enforce >=3s net speech via context.vad.
- antispoof.py: signal-level liveness heuristics (reverb/bandwidth replay cues reusing features) + optional AASIST hook interface (not required to ship weights). Output DetectorScore-style dict.

### C5 — fusion + api
- calibrate.py: Platt scaling (logistic on score) per detector; `fit(scores, labels)`, `Calibrator.save/load` (joblib).
- engine.py: `FusionEngine(config)`:
  - Inputs: list[DetectorScore] per window + context DetectorScores; learned fuser = logistic regression over calibrated detector scores + confounder interactions (trainable via `fit_fuser(dataset_scores, labels)`; until trained, fallback = documented weighted average from config).
  - Guardrail rules (from info.md 5): car-kit whitelist (single voice + stable channel + short-RT60 profile -> cap at CHALLENGE); TV/crowd (scene high + coupling low -> void secondary-voice relay evidence); NS confounder discount.
  - HMM/EMA temporal smoothing over window fused scores (alpha configurable) so single windows can't flip verdict.
  - `verdict(...) -> Verdict` with reason strings.
- api/server.py FastAPI:
  - `POST /analyze` (WAV upload or raw PCM16 16k) -> Verdict JSON (runs full pipeline).
  - `POST /enroll` (multiple WAVs) -> voiceprint id; `POST /challenge/start` -> {session_id, digit_string}; `POST /challenge/respond` (session_id + WAV) -> ChallengeResult; `GET /health`.
  - In-memory session store w/ TTL; clean pydantic models; uvicorn entrypoint `relayguard.api.server:app`.

## 5. Quality gates (every agent)
- pytest tests for own module(s) must pass (`python -m pytest tests/test_<module>.py`).
- No new heavy deps without justification; requirements.txt updated.
- Deterministic seeds everywhere; no network calls at runtime except speechbrain model cache (lazy, documented).
- Commit to own branch in own worktree; do NOT touch other modules' files.
