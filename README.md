# RelayGuard — deployable app + browser test console

RelayGuard detects **speakerphone-relay calls** (a caller piping the call through a nearby
loudspeaker — another room, a bot, a TTS rig) from the audio channel alone. It fuses five
window-level detectors (CNN on log-mel, LightGBM on telephony features, bandwidth / reverb /
distortion forensics) with call-level context detectors (channel-switch, conversation coupling,
scene tagging, replay cues) into a calibrated three-state verdict: **GREEN / CHALLENGE / RED**.

This package is the fully-trained system plus a single-page browser console for trying it.

## Run locally

```bash
cd relayguard_app
pip install -r requirements.txt        # pure PyPI, torch-free runtime (CNN runs on ONNX)
uvicorn relayguard.api.server:app --port 8000
# open http://localhost:8000
```

The runtime is **torch-free**: the CNN ships as `artifacts/cnn.onnx` (full
waveform → log-mel → logit graph, opset 17) and is served by `onnxruntime`.
`torch` is only needed for the Train tab's fine-tune job and the offline
training/eval CLIs — install it separately if you want those:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.
Without torch, `POST /training/start` returns a clean
`503 "training requires torch (not installed in this deployment)"`; every
other endpoint works normally. To regenerate the ONNX artifact from a
checkpoint: `python -m relayguard.models.export_onnx --validate` (needs
torch + onnxruntime).

Artifacts and config resolve automatically from the app root. You can override with
environment variables:

| variable | default | meaning |
|---|---|---|
| `RELAYGUARD_ARTIFACTS` | `./artifacts` (app root) | trained models directory |
| `RELAYGUARD_CONFIG` | `configs/default.yaml` (app root) | calibrated thresholds config |
| `RELAYGUARD_STATIC` | `static/` (app root) | console + demo samples |
| `SUPABASE_URL` / `SUPABASE_KEY` | unset | learning-mode cloud storage (see below) |
| `RELAYGUARD_USER_DATA` | `user_data/` (app root) | local upload store (no Supabase) |
| `RELAYGUARD_VERSIONS` | `versions/` (app root) | trained artifact versions |
| `RELAYGUARD_ANCHOR` | `anchor_data/` (app root) | anchor rehearsal dataset |

## Run with Docker

```bash
docker build -t relayguard .
docker run -p 8000:8000 relayguard
# open http://localhost:8000
```

The image installs **pure PyPI only** (no PyTorch index, no torch wheel), so
it builds to roughly **~300MB** (down from ~1GB+ with the CPU torch wheel).
It copies everything the app needs — `relayguard/`, `artifacts/` (including
`cnn.onnx`), `configs/`, `static/`, `index.html`, `anchor_data/` — exposes
port 8000 and carries a `HEALTHCHECK` against `GET /health`.

## The test console (http://localhost:8000)

- **Analyze audio** — three ways to test:
  1. click a bundled sample (held-out test audio with known ground truth),
  2. drag-drop / upload any audio file,
  3. record from your microphone (max 15 s).
  Shows the verdict badge, fused-score gauge, detector breakdown and a per-window
  timeline with the calibrated decision thresholds — mid-call speakerphone switches
  are visible as the smoothed curve crossing theta_red.
- **Biometric challenge** — random 5-digit challenge with TTL; record your spoken
  response; shows accept/retry/reject, net speech seconds, content match, identity
  score and relay context.
- **Train** — user-driven learning (see next section).
- **How it works** — short explainer of the verdict logic and guardrails.

## Learning mode (user-driven fine-tuning)

The **Train** tab lets the system learn from *your* audio:

1. **Upload labeled samples** — file picker or microphone recording, labeled
   *Normal call* or *Speakerphone relay* (2–120 s). Each upload is quality-checked
   on the spot: duration, net speech seconds (VAD), bandwidth class, clipping and
   RMS level, with human-readable warnings (e.g. "clipping detected",
   "only 1.2 s speech").
2. **Press "Train model"** — *(requires torch; see "Run locally" — in a
   torch-free deployment this endpoint answers 503)* a background job fine-tunes the shipped CNN
   (LR 1e-4, up to 3 epochs, early-stopped on anchor-dev AUC) and retrains the
   LightGBM on your samples **plus a built-in anchor rehearsal set**
   (`anchor_data/`, 350 train + 70 dev files subsampled from the original
   simulated dataset). The rehearsal set prevents catastrophic forgetting: your
   handful of samples adapts the model without erasing its original training.
   The run reports before/after metrics — anchor-dev AUC and hit@2%FPR, and
   per-label accuracy on a held-out 30% of your uploads (flagged *in-sample*
   when you have <10 uploads and everything is used for training).
3. **Versioning + hot-swap** — every run saves `versions/v{n}/`
   (cnn.pt + cnn.onnx, lgbm.txt, feature_scaler.joblib + training_report.json;
   the fuser, calibrators and thresholds are carried forward unchanged) and the
   live pipeline swaps to it immediately. The versions dropdown + *Activate* button
   roll back to any previous version (v0 = the base model) with zero downtime.

### Where uploads are stored

- **Local mode (default)** — uploads land in `user_data/audio/` with metadata in
  `user_data/samples.jsonl`. No setup needed; great for trying it.
- **Supabase mode** — set `SUPABASE_URL` + `SUPABASE_KEY` and uploads go to a
  `relayguard-audio` Storage bucket (WAVs) + a `training_samples` table
  (metadata), via plain REST (httpx, no supabase-py dependency).

**Supabase setup (exact steps):**

1. Create a project at [supabase.com](https://supabase.com) (free tier is fine).
2. In the project dashboard open **SQL Editor** and run `supabase_setup.sql`
   (creates the `training_samples` table, the `relayguard-audio` bucket, and
   demo RLS policies — the file carries a warning to tighten them for production).
   If you prefer, create the bucket manually: **Storage → New bucket →
   `relayguard-audio`** (public off).
3. Copy your project URL (**Settings → API → Project URL**) and an API key
   (anon or service-role).
4. Start the server with:
   ```bash
   SUPABASE_URL=https://<project>.supabase.co SUPABASE_KEY=<key> \
     uvicorn relayguard.api.server:app --port 8000
   ```
5. The Train tab badge switches to **Storage: Supabase**; `GET /training/backend`
   reports the same.

### Bundled samples

`static/samples/` ships 7 held-out test WAVs (`samples.json` manifest):
2 simulated relays, 1 simulated direct, 2 real relays (real phone audio through a
simulated speakerphone chain), 2 real direct calls. `GET /samples` lists them;
`GET /analyze_sample?name=sim_relay_1` runs the identical pipeline as `POST /analyze`.

## Full voice biometrics (optional)

Voice enrollment (`POST /enroll`) and identity scores in challenges use
[speechbrain](https://speechbrain.github.io/), which is deliberately **not** in
`requirements.txt` (heavy dependency chain). Without it, challenges still run —
identity reports `unavailable (install speechbrain)` while digit-content, net-speech
and retry logic work. To enable:

```bash
pip install speechbrain
```

Offline dataset generation (`relayguard.datagen.*`) additionally needs
`pyroomacoustics`; it is not used at runtime.

## API summary

| endpoint | description |
|---|---|
| `GET /health` | module availability |
| `POST /analyze` | multipart WAV (`file`) or raw PCM16 body → verdict JSON |
| `GET /samples` | bundled demo sample manifest |
| `GET /analyze_sample?name=…` | run a bundled sample through the pipeline |
| `POST /enroll` | enroll a voiceprint (needs speechbrain) |
| `POST /challenge/start` / `POST /challenge/respond` | digit challenge session |
| `POST /training/upload` | labeled sample upload (multipart: file, label, notes) → QC |
| `GET /training/samples` / `DELETE /training/sample/{id}` | manage uploads |
| `POST /training/start` / `GET /training/status` | background fine-tune job |
| `GET /training/versions` / `POST /training/activate?version=n` | versions + rollback |
| `GET /training/backend` | storage backend (supabase / local) |

## Evaluation reports

Trained-model evaluation lives in `artifacts/`: `eval_report.md` / `eval_report.json`
(simulated test set), `eval_real.md` / `eval_real.json` (real-call test slice),
`fusion_thresholds.json` (calibrated operating points, Neyman–Pearson on dev),
`train_report.json`, `RESULTS.md`.
