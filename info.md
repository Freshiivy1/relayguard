# RelayGuard — Consolidated Research Brief (info.md)
Synthesis of 4 parallel research agents (acoustics science / datasets & simulation / voice biometrics / false-positive defense). Full briefs in /mnt/agents/output/19feefaf-*/ and /mnt/agents/output/speakerphone_relay_fp_defense_brief.md.

## 1. Problem physics
Speakerphone-relay chain: `codec1 → DAC → phone-B loudspeaker → room RIR → phone-A mic/AFE → codec2`.
Direct call chain: `talker → phone-A mic → codec`. The relay adds: loudspeaker transduction (band-limit ~300Hz–4kHz, resonance ~1kHz, THD/clipping, smart-amp limiting), a room convolution (RT60 0.2–0.8s, low DRR), and a second codec pass. This is exactly the ASVspoof replay-attack (physical access) signature — published detectors reach EER 0.5–2% on comparable tasks; cross-device generalization is the known weak point (mitigate with aggressive augmentation + leave-device-out eval).

## 2. Ranked detection angles (from acoustics agent)
1. Learned classifier on log-mel/CQCC spectrograms trained on simulated relay chains (highest power; LCNN/ResNet/AASIST family).
2. Bandwidth & double-bandlimit forensics: subband energy ratios, spectral voids, codec band-edge mismatches.
3. Room acoustics: SRMR (Speech-to-Reverb Modulation Energy Ratio), blind DRR/RT60, spectral flatness of inter-phoneme gaps.
4. Loudspeaker nonlinearity: THD on voiced frames (energy at 2f0/3f0), crest-factor/dynamic-range compression (Ren et al. 2019: TPR 97.8% in-domain).
5. Double-compression codec artifacts: quantization-noise statistics, phase features (forensics literature: 93–99% single-vs-double AMR accuracy).
6. AEC/speakerphone-DSP artifacts: NLP gating, comfort noise in gaps, level pumping.
7. High-frequency anti-aliasing signatures (only if capture >8kHz).

## 3. Training data recipe (from data agent)
- Source speech: LibriSpeech (CC BY 4.0, openslr.org). Environment-constrained: use test-clean/dev-clean subset.
- Relay simulation: clean → codec1 (ffmpeg: AMR-NB/AMR-WB/Opus/G.711/GSM, random bitrate) → loudspeaker EQ (HPF 120–500Hz + LPF 3.4–10kHz + parametric resonance peaks + tanh/clip distortion + dynamic limiting) → room (pyroomacoustics shoebox, RT60 0.15–0.9s, speaker-mic distance 0.3–3m, SNR 5–30dB noise) → optional capture-chain NS/AGC → codec2 (independent draw).
- Direct controls: clean → single codec (+ mild mic EQ, optional light reverb RT60<0.3s to avoid trivial separation, optional NS).
- CRITICAL: codec roundtrips on BOTH classes so the model learns loudspeaker/room, not codec presence.
- Hard-negative FP classes (GREEN): (a) direct + TV/music/babble background mixed (MUSAN-style), (b) direct + strong room reverb, (c) direct + aggressive noise-suppression simulation, (d) direct + band-limited EQ (cheap headset), (e) car-cabin simulation (short RT60 50–150ms + mild speaker EQ, single codec).
- Metadata JSONL per sample: label, chain params (room, RT60, distance, codecs, device preset).
- Splits: speaker-disjoint AND condition-disjoint (hold out codec pairs / room sizes / device presets).
- External validation: real replay corpora exist (ASVspoof 2021 PA Zenodo 4834716, ReMASC IEEE DataPort) — optional later download.

## 4. Voice biometrics (from biometrics agent)
- SV model: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0, CPU-OK); production alternative WeSpeaker ResNet34 ONNX.
- Expect single-digit EER at 8kHz telephony; ~8–10% EER on relayed genuine speech → thresholds must be channel-aware; enroll during a verified call (same channel class), 3–5 phrases, >=15s net speech.
- Challenge flow: CSPRNG random 5–6 digit string (one-time, 30s expiry) → TTS prompt → capture → VAD quality gate (>=3s net speech, retry max 2) → ASR digit content check → resample 8→16kHz → ECAPA embedding cosine vs enrolled centroid → calibrated LLR identity score.
- Anti-spoofing: AASIST (MIT, clovaai) fine-tuned with codec+replay augmentation; genuine-relayed callers trigger replay scores BY DESIGN → feed into fusion, never auto-reject.
- Fusion rule table (q_relay, s_id, s_spoof, content): only relay-acoustics + biometric-fail together confirm third-party; relay + biometric-pass = caller on own speakerphone (step-up monitoring).

## 5. False-positive defense (from FP agent)
- No single feature is FP-safe. Ensemble of detectors + per-detector Platt/isotonic calibration + learned fuser (logistic regression) + hard guardrail rules + 3-state verdict.
- FP matrix (scenario → discriminator): TV background → zero conversational coupling + primary talker clean; car hands-free → single party + short cabin RT60 + normal latency → whitelist to max CHALLENGE; bathroom echo → reverb uniform from t=0, no transducer/codec artifacts; cheap headset → time-invariant odd EQ (baseline first 2–5s, only *changes* suspicious); aggressive NS → dead-flat noise floor + T-F localized artifacts (confounder feature discounts relay evidence); crowd → unstable diarization clusters; VoIP loss → burst-correlated impairment.
- Conversational intelligence: Silero VAD turn-taking stats; speaker counting / embedding stability; response-latency percentiles; background-response contingency test (does background answer the agent? — best TV killer); mid-call channel-switch detection (delta-BIC / embedding change-point) — "weird from t=0 and constant" = benign device, "became weird at t=127s" = relay added. Highest-value RED feature.
- Scene classification: PANNs (panns-inference, CNN14) / YAMNet classes Television/Radio/Music/Crowd.
- Verdict: 3-state with temporal smoothing (HMM). theta_high set by Neyman-Pearson (recall>=0.90 on confirmed relays), theta_low by FPR<=1–2%, FPR verified PER SCENARIO SLICE (car-kit<=3%, TV<=2% targets).

## 6. Environment constraints
Build host: CPU-only, 2 cores, 4GB RAM → compact CNN (~300–500K params) on log-mel, LightGBM/sklearn GBM on handcrafted features, pretrained SV/anti-spoof models for inference only. Design must scale to GPU unchanged.
