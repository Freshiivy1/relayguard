"""Conversational-intelligence detector (info.md 2.1-2.3, SPEC 4 C3).

Single-channel call audio -> turn-taking statistics, a voice-count heuristic
(mean log-mel turn embeddings + agglomerative clustering), identity drift, and
a background-coupling test ("does the background source react to the primary
conversation?" — the strongest TV/crowd false-positive killer).

Score semantics: P(relay-ish conversational context) in [0,1].
- >= 2 stable voice clusters     -> a second coherent party exists
- high background coupling       -> background RESPONDS to the conversation
                                    (a relayed room reacts; a TV does not)
- high identity drift            -> voice/channel of a party changes mid-call

All raw statistics are exposed in DetectorScore.details for the fusion layer.
Pure numpy/scipy, deterministic, no model downloads.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks

from relayguard.common import DetectorScore
from relayguard.context._mel import log_mel, rms_db_track
from relayguard.context.vad import Turn, get_speech_frames, segment_turns

# --- tuning constants (documented heuristics) ---
CLUSTER_DIST_THRESH = 0.08   # cosine distance threshold for turn-embedding clustering
STABLE_MIN_TURNS = 2         # a "stable" cluster owns at least this many turns
STABLE_MIN_FRAC = 0.20       # ... or this fraction of all turns, whichever larger
COUPLING_GRID_MS = 50.0      # grid for the coupling cross-correlation
COUPLING_MAX_LAG_S = 1.5     # reaction window after a primary-speech onset
LONG_PAUSE_S = 2.5           # gap longer than this counts toward pause anomaly


def _turn_embeddings(audio: np.ndarray, sr: int, turns: list[Turn],
                     n_mels: int = 32) -> np.ndarray:
    """One L2-normalized mean log-mel vector per turn."""
    embs = []
    for t in turns:
        seg = audio[int(t.start_s * sr): int(t.end_s * sr)]
        if len(seg) < int(0.05 * sr):
            seg = audio[max(0, int(t.start_s * sr) - int(0.05 * sr)):
                        int(t.end_s * sr) + int(0.05 * sr)]
        if len(seg) == 0:
            embs.append(np.zeros(n_mels))
            continue
        lm = log_mel(seg, sr, n_mels=n_mels)
        e = lm.mean(axis=0)
        embs.append(e / (np.linalg.norm(e) + 1e-12))
    return np.asarray(embs)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) /
                 ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _voice_count(embs: np.ndarray) -> dict:
    """Cluster turn embeddings; report cluster count/stability and drift."""
    n = len(embs)
    if n == 0:
        return {"n_turns_embedded": 0, "n_clusters_raw": 0, "n_clusters": 0,
                "cluster_stability": 0.0, "identity_drift": 0.0, "labels": []}
    if n == 1:
        return {"n_turns_embedded": 1, "n_clusters_raw": 1, "n_clusters": 1,
                "cluster_stability": 1.0, "identity_drift": 0.0, "labels": [0]}
    z = linkage(embs, method="average", metric="cosine")
    labels = fcluster(z, t=CLUSTER_DIST_THRESH, criterion="distance") - 1
    uniq, counts = np.unique(labels, return_counts=True)
    min_members = max(STABLE_MIN_TURNS, int(np.ceil(STABLE_MIN_FRAC * n)))
    stable = [c for c, k in zip(uniq, counts) if k >= min_members]

    stabilities = []
    for c in stable:
        members = embs[labels == c]
        centroid = members.mean(axis=0)
        sims = [1.0 - _cos_dist(m, centroid) for m in members]
        stabilities.append(float(np.mean(sims)))
    cluster_stability = float(np.mean(stabilities)) if stabilities else 0.0

    drift = 0.0
    for i in range(1, n):
        if labels[i] == labels[i - 1] and labels[i] in stable:
            drift = max(drift, _cos_dist(embs[i - 1], embs[i]))

    return {"n_turns_embedded": n, "n_clusters_raw": int(len(uniq)),
            "n_clusters": int(len(stable)),
            "cluster_stability": cluster_stability,
            "identity_drift": float(drift),
            "labels": [int(x) for x in labels]}


def _background_coupling(audio: np.ndarray, sr: int, turns: list[Turn],
                         total_s: float) -> float:
    """Peak normalized cross-correlation (lag 0..1.5 s) between primary-speech
    onsets and the background-only RMS envelope.

    Background samples = frames whose RMS lies between p20 and p50 measured
    during gaps between primary turns. High coupling => the background source
    reacts to the conversation (suspicious relay room / gated mic); near-zero
    => independent program material (TV-like, benign).
    """
    if len(turns) < 2 or total_s < 4.0:
        return 0.0
    grid_ms = COUPLING_GRID_MS
    env_db = rms_db_track(audio, sr, frame_ms=grid_ms)
    n_grid = len(env_db)
    t_grid = (np.arange(n_grid) + 0.5) * grid_ms / 1000.0

    speech_mask = np.zeros(n_grid, dtype=bool)
    for t in turns:
        speech_mask |= (t_grid >= t.start_s) & (t_grid <= t.end_s)
    gap_idx = np.where(~speech_mask)[0]
    if len(gap_idx) < int(1.0 / (grid_ms / 1000.0)):
        return 0.0
    gap_db = env_db[gap_idx]
    p20, p50 = np.percentile(gap_db, [20.0, 50.0])
    # background-only: quiet-to-moderate gap frames (exclude bleed of loud speech)
    bg_idx = gap_idx[(gap_db >= p20 - 3.0) & (gap_db <= p50 + 6.0)]
    if len(bg_idx) < int(1.0 / (grid_ms / 1000.0)):
        return 0.0

    # envelope series valid only at background samples; interpolate across the rest
    bg_env = np.full(n_grid, np.nan)
    bg_env[bg_idx] = env_db[bg_idx]
    valid = ~np.isnan(bg_env)
    if valid.sum() < 0.05 * n_grid:
        return 0.0
    bg_env = np.interp(np.arange(n_grid), np.where(valid)[0], bg_env[valid])

    # onset impulse train at turn starts
    onsets = np.zeros(n_grid)
    for t in turns[1:]:  # first turn has no preceding context to react to
        k = int(t.start_s * 1000.0 / grid_ms)
        if 0 <= k < n_grid:
            onsets[k] = 1.0
    if onsets.sum() < 2:
        return 0.0

    a = onsets - onsets.mean()
    b = bg_env - bg_env.mean()
    denom = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-12)
    max_lag = int(COUPLING_MAX_LAG_S * 1000.0 / grid_ms)
    best = 0.0
    for lag in range(0, max_lag + 1):
        if lag == 0:
            num = float(np.sum(a * b))
        else:
            num = float(np.sum(a[:-lag] * b[lag:]))
        best = max(best, num / denom)
    return float(np.clip(best, 0.0, 1.0))


def _speaking_rate_stability(audio: np.ndarray, sr: int,
                             turns: list[Turn]) -> float:
    """1 - coefficient-of-variation of per-turn envelope-peak rate (clipped).

    Rate proxy = energy-envelope peaks per second (syllable-rate surrogate).
    Stable speaking rate (real person) -> near 1; erratic (gated relay audio,
    TV cuts) -> lower.
    """
    rates = []
    for t in turns:
        dur = t.end_s - t.start_s
        if dur < 0.25:
            continue
        seg = audio[int(t.start_s * sr): int(t.end_s * sr)]
        env = rms_db_track(seg, sr, frame_ms=20.0, hop_ms=10.0)
        if len(env) < 8:
            continue
        peaks, _ = find_peaks(env, distance=6, prominence=1.5)
        rates.append(len(peaks) / dur)
    if len(rates) < 2:
        return 1.0
    rates = np.asarray(rates)
    mean = rates.mean()
    if mean <= 1e-6:
        return 1.0
    cv = float(rates.std() / mean)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def analyze(audio: np.ndarray, sr: int = 16000) -> DetectorScore:
    """Call-level conversational-context analysis -> DetectorScore."""
    audio = np.asarray(audio, dtype=np.float32)
    total_s = len(audio) / float(sr) if sr else 0.0
    if total_s < 0.5:
        return DetectorScore(name="conversation_context", score=0.1, details={
            "error": "audio too short", "n_turns": 0, "n_clusters": 0,
            "cluster_stability": 0.0, "identity_drift": 0.0,
            "background_coupling": 0.0})

    speech_frames = get_speech_frames(audio, sr)
    turns = segment_turns(audio, sr)
    durs = np.array([t.dur_s for t in turns])
    gaps = np.array([turns[i + 1].start_s - turns[i].end_s
                     for i in range(len(turns) - 1)])
    speech_ratio = float(np.mean(speech_frames)) if len(speech_frames) else 0.0

    gap_p50 = float(np.percentile(gaps, 50)) if len(gaps) else 0.0
    gap_p95 = float(np.percentile(gaps, 95)) if len(gaps) else 0.0
    long_pause_index = float(np.mean(gaps > LONG_PAUSE_S)) if len(gaps) else 0.0

    embs = _turn_embeddings(audio, sr, turns)
    vc = _voice_count(embs)
    coupling = _background_coupling(audio, sr, turns, total_s)
    rate_stab = _speaking_rate_stability(audio, sr, turns)

    # --- suspicion fusion (documented heuristic, bounded [0,1]) ---
    score = 0.10
    if vc["n_clusters"] >= 2:
        # a second coherent, stable voice: scaled by how compact the clusters are
        score += 0.35 * float(np.clip(vc["cluster_stability"], 0.0, 1.0))
    score += 0.30 * float(np.clip(coupling / 0.5, 0.0, 1.0))      # coupling ~0.5+ = strong
    score += 0.25 * float(np.clip(vc["identity_drift"] / 0.25, 0.0, 1.0))
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        "total_s": float(total_s),
        "n_turns": int(len(turns)),
        "turn_dur_mean_s": float(durs.mean()) if len(durs) else 0.0,
        "turn_dur_std_s": float(durs.std()) if len(durs) else 0.0,
        "gap_p50_s": gap_p50,
        "gap_p95_s": gap_p95,
        "speech_ratio": speech_ratio,
        "long_pause_anomaly_index": long_pause_index,
        "speaking_rate_stability": rate_stab,
        "n_clusters": vc["n_clusters"],
        "n_clusters_raw": vc["n_clusters_raw"],
        "cluster_stability": vc["cluster_stability"],
        "identity_drift": vc["identity_drift"],
        "background_coupling": coupling,
        "turns": [{"start_s": t.start_s, "end_s": t.end_s,
                   "rms_db": t.rms_db, "mean_f0": t.mean_f0} for t in turns],
        "cluster_labels": vc["labels"],
        "scoring": {
            "base": 0.10,
            "multi_voice": ">=2 stable clusters -> +0.35*cluster_stability",
            "coupling": "+0.30*clip(coupling/0.5,0,1)",
            "drift": "+0.25*clip(identity_drift/0.25,0,1)",
        },
    }
    return DetectorScore(name="conversation_context", score=score, details=details)
