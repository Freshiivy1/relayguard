"""Random-digit challenge flow (SPEC 4-C4).

Flow: CSPRNG 5-digit one-time nonce (30 s TTL) -> TTS prompt -> capture ->
net-speech quality gate (>= 3 s, max 2 retries) -> optional ASR digit
content check (pluggable ``content_checker`` protocol; default None = system
logs and relies on the identity score, interface ready for Whisper later) ->
ECAPA cosine vs enrolled centroid -> verdict.

Verdicts never auto-terminate a call: FAIL means "escalate to a human agent
/ second factor". Genuine users on their own speakerphone are expected to
pass biometrics even when the relay-acoustic detectors fire (fusion rule:
relay-acoustics + biometric-fail together confirm third-party relay).
"""
from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from relayguard.common import TARGET_SR
from .verifier import (
    BiometricsUnavailable,
    IDENTITY_THRESHOLDS,
    MIN_NET_SPEECH_S,
    SpeakerVerifier,
    net_speech_seconds,
)

DIGIT_STRING_LEN = 5
SESSION_TTL_S = 30.0
MAX_RETRIES = 2

# content_checker protocol: fn(audio_16k: np.ndarray, expected_digits: str)
#     -> True (digits spoken match), False (mismatch), or None (undecidable).
ContentChecker = Callable[[np.ndarray, str], Optional[bool]]


@dataclass
class ChallengeSession:
    """One-time challenge state. ``retries`` counts consumed retries."""

    session_id: str
    digit_string: str
    created_at: float
    ttl: float = SESSION_TTL_S
    retries: int = 0
    max_retries: int = MAX_RETRIES
    channel: str = "speakerphone_relay"

    @classmethod
    def generate(cls, ttl: float = SESSION_TTL_S,
                 channel: str = "speakerphone_relay",
                 store: "SessionStore | None" = None) -> dict:
        """Create a fresh session (CSPRNG 5-digit nonce) and return its dict."""
        session = cls(
            session_id=uuid.uuid4().hex,
            digit_string="".join(secrets.choice("0123456789")
                                 for _ in range(DIGIT_STRING_LEN)),
            created_at=time.time(),
            ttl=ttl,
            channel=channel,
        )
        if store is not None:
            store.put(session)
        return {
            "session_id": session.session_id,
            "digit_string": session.digit_string,
            "expires_at": session.created_at + session.ttl,
        }

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) > self.expires_at


@dataclass
class ChallengeResult:
    content_match: Optional[bool]      # None = no content checker provided
    net_speech_s: float
    identity_score: Optional[float]    # None = biometrics unavailable/unset
    verdict: str                       # PASS | FAIL | RETRY | EXPIRED
    reason: str

    def to_dict(self) -> dict:
        return {
            "content_match": self.content_match,
            "net_speech_s": round(float(self.net_speech_s), 3),
            "identity_score": (None if self.identity_score is None
                               else round(float(self.identity_score), 4)),
            "verdict": self.verdict,
            "reason": self.reason,
        }


class SessionStore:
    """In-memory session store with TTL sweep."""

    def __init__(self):
        self._sessions: dict[str, ChallengeSession] = {}

    def put(self, session: ChallengeSession) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> ChallengeSession | None:
        return self._sessions.get(session_id)

    def sweep(self, now: float | None = None) -> int:
        """Drop expired sessions; returns number removed."""
        now = now if now is not None else time.time()
        dead = [sid for sid, s in self._sessions.items() if s.expired(now)]
        for sid in dead:
            del self._sessions[sid]
        return len(dead)

    def __len__(self) -> int:
        return len(self._sessions)


def validate_response(session: ChallengeSession, audio: np.ndarray,
                      content_checker: ContentChecker | None = None,
                      verifier: SpeakerVerifier | None = None,
                      enrolled_emb: np.ndarray | None = None,
                      sr: int = TARGET_SR,
                      now: float | None = None) -> ChallengeResult:
    """Validate a captured challenge response against ``session``.

    Order of checks: TTL -> net-speech gate -> content -> identity.
    """
    now = now if now is not None else time.time()

    # 1) TTL ------------------------------------------------------------
    if session.expired(now):
        return ChallengeResult(None, 0.0, None, "EXPIRED",
                               "challenge session expired "
                               f"(ttl={session.ttl:.0f}s)")

    # 2) Net-speech quality gate ----------------------------------------
    net = net_speech_seconds(audio, sr)
    if net < MIN_NET_SPEECH_S:
        if session.retries < session.max_retries:
            session.retries += 1
            return ChallengeResult(
                None, net, None, "RETRY",
                f"insufficient net speech ({net:.2f}s < {MIN_NET_SPEECH_S}s); "
                f"retry {session.retries}/{session.max_retries}")
        return ChallengeResult(
            None, net, None, "FAIL",
            f"insufficient net speech ({net:.2f}s) after "
            f"{session.max_retries} retries; escalate to human/second factor")

    # 3) Digit content check (pluggable; None = not performed) ----------
    content_match: Optional[bool] = None
    if content_checker is not None:
        content_match = content_checker(audio, session.digit_string)

    # 4) Identity verification (optional, graceful) ---------------------
    identity_score: Optional[float] = None
    identity_note = "identity not checked"
    if enrolled_emb is not None:
        verifier = verifier or SpeakerVerifier()
        try:
            identity_score = verifier.verify(audio, enrolled_emb, sr=sr)
            identity_note = f"cosine={identity_score:.3f}"
        except BiometricsUnavailable as exc:
            identity_note = f"biometrics unavailable ({exc})"

    # 5) Verdict ---------------------------------------------------------
    threshold = IDENTITY_THRESHOLDS.get(
        session.channel, IDENTITY_THRESHOLDS["speakerphone_relay"])
    reasons = [f"net_speech={net:.2f}s", identity_note]
    fail = False
    if identity_score is not None:
        ok = identity_score >= threshold
        fail = fail or not ok
        reasons.append(f"identity {'>=' if ok else '<'} threshold {threshold}")
    if content_match is not None:
        fail = fail or not content_match
        reasons.append("digit content " +
                       ("matched" if content_match else "MISMATCH"))
    verdict = "FAIL" if fail else "PASS"
    if verdict == "FAIL":
        reasons.append("escalate to human/second factor (never auto-terminate)")
    return ChallengeResult(content_match, net, identity_score, verdict,
                           "; ".join(reasons))
