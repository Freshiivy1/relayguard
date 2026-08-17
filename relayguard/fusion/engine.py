"""Fusion engine: calibrated score fusion + guardrails + temporal smoothing
+ 3-state verdict (C5).

Pipeline per call:
  1. Per window: calibrate each detector score (identity if no calibrator),
     combine via the learned logistic fuser if fitted, else a documented
     fallback weighted average from config["fusion"]["fallback_weights"]
     (missing detectors -> remaining weights renormalized).
  2. Guardrail rules adjust the fused score / cap the verdict:
       R1 constant benign channel (car-kit/headset): stable channel +
          single stable voice cluster + short-reverb profile -> cap at
          CHALLENGE (never RED).
       R2 background media/crowd void: scene tv/crowd high + no
          conversational coupling -> subtract 0.25 of relay evidence.
       R3 aggressive noise-suppression confounder: dead-flat noise floor ->
          shrink fused score 15% toward 0.5.
  3. EMA temporal smoothing over per-window fused scores (alpha from
     config["fusion"]["smoothing_alpha"]) so single windows cannot flip the
     verdict.
  4. 3-state verdict: smoothed < theta_green -> GREEN, >= theta_red -> RED,
     else CHALLENGE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from relayguard.common import DetectorScore, Verdict

from .calibrate import CalibratorBank

# Guardrail reason strings (stable substrings; tests and API clients rely on them).
R1_REASON = "constant benign channel (car-kit/headset profile)"
R2_REASON = "background media/crowd without conversational coupling"
R3_REASON = "aggressive noise-suppression confounder"

# Detail keys probed for guardrail evidence (context modules are built in
# parallel; we accept several plausible key names defensively).
_CLUSTER_KEYS = ("n_voice_clusters", "voice_clusters", "num_clusters", "n_speakers", "n_clusters")
_STABILITY_KEYS = ("cluster_stability", "voice_stability", "cluster_persistence")
_RT60_KEYS = ("rt60", "rt60_s", "rt60_est")
_COUPLING_KEYS = ("background_coupling", "coupling", "bg_coupling", "background_response_coupling")
_NS_FLAG_KEYS = ("ns_confounder", "dead_floor", "noise_floor_stationary")
_NOISE_KEYS = ("noise", "noise_floor_stationarity", "noise_stationarity")
# Nested quantization-noise cue probed from biometrics.antispoof:
# details["quantization_noise"]["noise_flatness" / "noise_stationarity"].


def _first_key(details: dict, keys: Iterable[str]):
    for k in keys:
        if k in details:
            try:
                return float(details[k])
            except (TypeError, ValueError):
                if isinstance(details[k], bool):
                    return float(details[k])
    return None


def _find(scores: Iterable[DetectorScore], *names: str) -> DetectorScore | None:
    wanted = set(names)
    for s in scores or []:
        if s.name in wanted:
            return s
    return None


class FusionEngine:
    """Calibrated fusion + guardrails + EMA smoothing -> 3-state Verdict."""

    def __init__(self, config: dict, calibrators: CalibratorBank | None = None) -> None:
        self.config = config or {}
        fcfg = self.config.get("fusion", {})
        self.smoothing_alpha = float(fcfg.get("smoothing_alpha", 0.4))
        self.theta_green = float(fcfg.get("theta_green", 0.35))
        self.theta_red = float(fcfg.get("theta_red", 0.80))
        # Documented fallback: weighted average of calibrated detector scores;
        # when a detector is missing, remaining weights are renormalized to
        # sum to 1. With no scores at all the fused score is the neutral 0.5.
        self.fallback_weights: dict[str, float] = dict(fcfg.get("fallback_weights", {}))
        self.calibrators = calibrators or CalibratorBank()
        # Optional pre-fitted calibrator bank on disk (lazy, best-effort).
        cal_path = fcfg.get("calibrators_path")
        if cal_path and not self.calibrators.calibrators:
            try:
                if Path(cal_path).exists():
                    self.calibrators = CalibratorBank.load(cal_path)
            except Exception:
                pass
        self.fuser = None  # sklearn LogisticRegression or None
        self.fuser_feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    # Score combination
    # ------------------------------------------------------------------ #
    def _calibrated_map(self, scores: Iterable[DetectorScore]) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in scores or []:
            out[s.name] = self.calibrators.calibrate(s.name, float(s.score))
        return out

    def _combine_window(self, score_map: dict[str, float],
                        context_map: dict[str, float]) -> float:
        """Combine one window's calibrated scores (+ call-level context scores)."""
        merged = dict(score_map)
        for k, v in context_map.items():  # context scores join the pool
            merged.setdefault(k, v)
        if self.fuser is not None:
            vec = [merged.get(name, 0.5) for name in self.fuser_feature_names]
            proba = self.fuser.predict_proba([vec])[0]
            classes = list(self.fuser.classes_)
            p = proba[classes.index(1)] if 1 in classes else float(proba[-1])
            return float(np.clip(p, 0.0, 1.0))
        # Fallback: renormalized weighted average over present detectors.
        num, den = 0.0, 0.0
        for name, w in self.fallback_weights.items():
            if name in merged:
                num += w * merged[name]
                den += w
        if den > 0:
            return float(np.clip(num / den, 0.0, 1.0))
        if merged:  # detectors present but no configured weights: plain mean
            return float(np.clip(np.mean(list(merged.values())), 0.0, 1.0))
        return 0.5  # no evidence at all -> neutral

    # ------------------------------------------------------------------ #
    # Guardrails
    # ------------------------------------------------------------------ #
    def _guardrail_r1(self, context_scores: list[DetectorScore],
                      window_scores: list[list[DetectorScore]]) -> bool:
        """Constant benign channel (car-kit/headset) whitelist check."""
        ch = _find(context_scores, "channel_switch", "channel_switch_score")
        if ch is None or float(ch.score) >= 0.3:
            return False
        conv = _find(context_scores, "conversation_context", "conversation")
        if conv is None:
            return False
        clusters = _first_key(conv.details, _CLUSTER_KEYS)
        if clusters is None or clusters != 1:
            return False
        stability = _first_key(conv.details, _STABILITY_KEYS)
        if stability is not None and stability < 0.7:
            return False
        # Short-reverb profile: explicit rt60 detail or a low reverb score.
        flat = [s for win in window_scores for s in win]
        rt60 = None
        for cand in list(context_scores) + flat:
            rt60 = _first_key(cand.details, _RT60_KEYS)
            if rt60 is not None:
                break
        if rt60 is not None:
            return rt60 < 0.3
        rev = _find(flat, "reverb", "reverb_detector")
        if rev is not None:
            return float(rev.score) < 0.4
        return False

    def _guardrail_r2(self, context_scores: list[DetectorScore]) -> bool:
        """Background TV/crowd without conversational coupling -> void evidence."""
        scene = _find(context_scores, "scene_context", "scene")
        conv = _find(context_scores, "conversation_context", "conversation")
        if scene is None or conv is None:
            return False
        tv = float(scene.details.get("tv", 0.0) or 0.0)
        crowd = float(scene.details.get("crowd", 0.0) or 0.0)
        coupling = _first_key(conv.details, _COUPLING_KEYS)
        if coupling is None:
            coupling = 1.0  # unknown coupling -> do not void
        return (tv > 0.6 or crowd > 0.6) and coupling < 0.2

    def _guardrail_r3(self, context_scores: list[DetectorScore],
                      window_scores: list[list[DetectorScore]]) -> bool:
        """Aggressive noise-suppression confounder (dead-flat noise floor)."""
        all_scores = list(context_scores) + [s for win in window_scores for s in win]
        for s in all_scores:
            for k in _NS_FLAG_KEYS:
                if s.details.get(k):
                    return True
            if s.name in ("scene_context", "scene"):
                noise = _first_key(s.details, _NOISE_KEYS)
                if noise is not None and noise > 0.7:
                    return True
            # Nested quantization-noise cue (biometrics.antispoof): a flat AND
            # stationary noise floor points at codec/comfort-noise suppression.
            qn = s.details.get("quantization_noise")
            if isinstance(qn, dict):
                flat = _first_key(qn, ("noise_flatness", "flatness"))
                stationary = _first_key(qn, ("noise_stationarity", "stationarity"))
                if (flat is not None and stationary is not None
                        and flat > 0.5 and stationary > 0.7):
                    return True
        return False

    # ------------------------------------------------------------------ #
    # Smoothing + verdict
    # ------------------------------------------------------------------ #
    def _smooth(self, scores: list[float]) -> float:
        """EMA over per-window fused scores; returns the final smoothed value."""
        if not scores:
            return 0.5
        ema = float(scores[0])
        a = self.smoothing_alpha
        for s in scores[1:]:
            ema = a * float(s) + (1.0 - a) * ema
        return ema

    def _state(self, smoothed: float) -> str:
        eps = 1e-9  # absorb float noise from weighted-average renormalization
        if smoothed >= self.theta_red - eps:
            return "RED"
        if smoothed < self.theta_green - eps:
            return "GREEN"
        return "CHALLENGE"

    def _confidence(self, state: str, smoothed: float) -> float:
        if state == "RED":
            return float(np.clip(smoothed, 0.0, 1.0))
        if state == "GREEN":
            return float(np.clip(1.0 - smoothed, 0.0, 1.0))
        span = max(self.theta_red - self.theta_green, 1e-6)
        margin = min(smoothed - self.theta_green, self.theta_red - smoothed)
        return float(np.clip(0.5 + margin / span, 0.0, 1.0))

    def _reason(self, window_scores: list[list[DetectorScore]],
                context_scores: list[DetectorScore],
                guardrails: list[str]) -> str:
        # Rank detectors by fallback weight * deviation from neutral.
        agg: dict[str, list[float]] = {}
        for s in [d for win in window_scores for d in win] + list(context_scores):
            agg.setdefault(s.name, []).append(self.calibrators.calibrate(s.name, float(s.score)))
        ranked = sorted(
            agg.items(),
            key=lambda kv: self.fallback_weights.get(kv[0], 0.0) * abs(np.mean(kv[1]) - 0.5),
            reverse=True,
        )
        top = [f"{name}={np.mean(vals):.2f}" for name, vals in ranked[:3]]
        parts = []
        parts.append("top detectors: " + (", ".join(top) if top else "none"))
        if guardrails:
            parts.append("guardrails: " + "; ".join(guardrails))
        return "; ".join(parts)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fuse(self, window_scores: list[list[DetectorScore]],
             context_scores: list[DetectorScore]) -> Verdict:
        """Fuse per-window detector scores + call-level context -> Verdict."""
        if not window_scores:
            raise ValueError("FusionEngine.fuse: window_scores must be non-empty")
        context_scores = list(context_scores or [])
        context_map = self._calibrated_map(context_scores)

        guardrails: list[str] = []
        r2 = self._guardrail_r2(context_scores)
        r3 = self._guardrail_r3(context_scores, window_scores)
        r1 = self._guardrail_r1(context_scores, window_scores)
        if r2:
            guardrails.append(R2_REASON)
        if r3:
            guardrails.append(R3_REASON)
        if r1:
            guardrails.append(R1_REASON)

        fused_per_window: list[float] = []
        for win in window_scores:
            s = self._combine_window(self._calibrated_map(win), context_map)
            if r2:  # void uncoupled background-media relay evidence
                s = max(0.0, s - 0.25)
            if r3:  # NS confounder: shrink 15% toward 0.5
                s = 0.5 + (s - 0.5) * 0.85
            fused_per_window.append(s)

        smoothed = self._smooth(fused_per_window)
        state = self._state(smoothed)
        if r1 and state == "RED":  # whitelist: never RED on constant benign channel
            state = "CHALLENGE"

        verdict = Verdict(
            state=state,
            confidence=self._confidence(state, smoothed),
            fused_score=float(smoothed),
            detector_scores=[d for win in window_scores for d in win] + context_scores,
            reason=self._reason(window_scores, context_scores, guardrails),
        )
        # Diagnostic trace (additive; does not affect the verdict). The API
        # layer surfaces these for the per-window timeline chart.
        verdict.window_fused = [float(s) for s in fused_per_window]
        ema_trace: list[float] = []
        ema = float(fused_per_window[0])
        ema_trace.append(ema)
        for s in fused_per_window[1:]:
            ema = self.smoothing_alpha * float(s) + (1.0 - self.smoothing_alpha) * ema
            ema_trace.append(float(ema))
        verdict.window_smoothed = ema_trace
        return verdict

    # SPEC 4-C5 names the entry point verdict(); keep both.
    verdict = fuse

    # ------------------------------------------------------------------ #
    # Learned fuser
    # ------------------------------------------------------------------ #
    def fit_fuser(self, rows: list[dict], labels: Iterable[int]) -> "FusionEngine":
        """Train the learned logistic fuser over (raw) detector-score rows.

        rows: list of dicts detector-name -> score (may be missing detectors;
        missing entries are filled with the neutral 0.5). Uses balanced class
        weights.
        """
        from sklearn.linear_model import LogisticRegression

        labels = np.asarray(list(labels), dtype=np.int64)
        if len(rows) != len(labels):
            raise ValueError("rows and labels length mismatch")
        names: set[str] = set()
        for row in rows:
            names.update(row.keys())
        self.fuser_feature_names = sorted(names)
        X = np.array([[row.get(n, 0.5) for n in self.fuser_feature_names] for row in rows],
                     dtype=np.float64)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(X, labels)
        self.fuser = lr
        return self

    def save_fuser(self, path: str | Path) -> None:
        if self.fuser is None:
            raise ValueError("no fitted fuser to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.fuser, "feature_names": self.fuser_feature_names}, str(path))

    def load_fuser(self, path: str | Path) -> "FusionEngine":
        obj = joblib.load(str(path))
        self.fuser = obj["model"]
        self.fuser_feature_names = list(obj["feature_names"])
        return self
