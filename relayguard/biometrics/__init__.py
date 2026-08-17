"""RelayGuard voice-biometrics identity layer (C4).

Enrollment / verification (ECAPA-TDNN via SpeechBrain, lazy-loaded),
random-digit challenge flow, and signal-level anti-spoofing cues.

SpeechBrain is an OPTIONAL dependency: every module here imports and every
test passes without it. The model is only loaded when verification is
actually invoked; failures raise ``BiometricsUnavailable``.
"""
from .verifier import (
    BiometricsUnavailable,
    SpeakerVerifier,
    cosine_to_prob,
    save_voiceprint,
    load_voiceprint,
    EMB_DIM,
)
from .challenge import (
    ChallengeSession,
    ChallengeResult,
    SessionStore,
    validate_response,
)
from .antispoof import analyze as analyze_replay_cues, AASISTHook

__all__ = [
    "BiometricsUnavailable",
    "SpeakerVerifier",
    "cosine_to_prob",
    "save_voiceprint",
    "load_voiceprint",
    "EMB_DIM",
    "ChallengeSession",
    "ChallengeResult",
    "SessionStore",
    "validate_response",
    "analyze_replay_cues",
    "AASISTHook",
]
