# RelayGuard — Delivery Document
Speakerphone-Relay Call Detection System (anti-fraud). Detects whether far-end call audio is a person speaking directly into their phone (GREEN) or a remote caller relayed through a second phone's loudspeaker (RED), with a voice-biometric identity challenge instead of call termination.

## Headline results (held-out test: unseen speakers, devices, codec pairs)
| Metric | Target | Achieved |
|---|---|---|
| Relay hit rate @ ≤2% false-positive rate | ≥90% | **97.5%** (FPR 0.3%) |
| CNN AUC / EER (sim test) | — | 0.9977 / 1.39% |
| REAL phone-network slice (565 real call segments → simulated relay) | — | AUC 1.000, hit rate 100% @1% FPR |
| TV-background FP rate | ≤2% | 0.0% |
| Reverb-room FP rate | ≤2% | 1.25% |
| Noise-suppression / headset FP rate | ≤2% | 0.0% / 0.0% |
| Car hands-free FP rate | ≤3% | 0.0% (dev slice) |

## Deliverables
- `project/` — full source (git, master `cd9159f`): 28 modules, 106/106 tests passing
- `relayguard_build/artifacts/` — trained models: cnn.pt (101K-param CNN), lgbm.txt, feature_scaler, fuser.joblib, calibrators.joblib, fusion_thresholds.json, default_calibrated.yaml, eval reports, RESULTS.md
- `relayguard_build/data/` — datasets: dataset_sim (5,262 samples), realcalls (565 real telephony segments), real_augmented (1,130 real direct/relay pairs)
- `SPEC.md`, `info.md` — system spec + consolidated research brief (4-agent research swarm)

## Architecture (multi-angle ensemble, per research)
1. **Window detectors (2s)**: RelayCNN on log-mel (primary), LightGBM on 50 handcrafted acoustic features, plus rule detectors: bandwidth forensics, reverb (SRMR-style), loudspeaker distortion.
2. **Context detectors (call-level)**: VAD/turn-taking + voice-count/identity-drift + background-coupling (TV killer), mid-call channel-switch detection (delta-BIC), acoustic scene tags, anti-spoof replay cues.
3. **Fusion**: per-detector Platt calibration → logistic fuser → guardrail rules (car-kit whitelist, TV/crowd void, NS confounder) → EMA temporal smoothing → 3-state verdict GREEN / CHALLENGE / RED.
4. **Biometrics** (challenge, not termination): ECAPA-TDNN (SpeechBrain, lazy/optional) + random 5-digit CSPRNG challenge, ≥3s net-speech gate, retry logic; only relay-acoustics + biometric-fail together confirm a third party.

## Run it
```bash
pip install -r project/requirements.txt   # + ffmpeg with gsm/opus
cd project && python -m pytest tests/ -q  # 106 passing
RELAYGUARD_ARTIFACTS=/mnt/agents/output/relayguard_build/artifacts \
  uvicorn relayguard.api.server:app --port 8000
# POST /analyze (WAV or PCM16) -> verdict JSON; /enroll, /challenge/start, /challenge/respond
```
Note: use `relayguard_build/artifacts/default_calibrated.yaml` as the config for calibrated thresholds.

## Honest limitations (from RESULTS.md)
- Real-world relay recordings don't exist in training (no public corpus) — the REAL slice uses real telephony voices through our simulated relay chain; validate on live traffic before production thresholds.
- Dev→test threshold skew observed; production calibration on your own call audio is mandatory (biometric thresholds especially: expect single-digit EER at 8kHz, higher on relayed genuine speech).
- Guardrails R1/R2 can misfire on real telephony — they fail SAFE (CHALLENGE, not RED), but tune with live data.
- Voice biometrics run in degraded mode until `speechbrain` is installed (pip install speechbrain; model downloads on first use).
- YouTube was unreachable from the build sandbox; real-call data came from a public research corpus of real phone-network calls instead.

## Next steps for production
1. Collect real speakerphone-relay calls (consent-based) → add as training/eval slice.
2. Calibrate fusion + biometric thresholds on your live call mix; verify per-scenario FPR.
3. Install speechbrain for the biometric layer; enroll customers during verified calls.
4. GPU retrain of a larger model (AASIST/wav2vec2 front-end) if you need >97.5%.
