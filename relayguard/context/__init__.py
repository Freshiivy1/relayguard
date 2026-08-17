"""RelayGuard conversational/context intelligence layer (C3).

Pure numpy/scipy, deterministic, no model downloads. Detectors:
- vad: energy + spectral-flatness VAD and turn segmentation.
- conversation: turn-taking stats, voice-count heuristic, background coupling.
- change_detect: delta-BIC mid-call channel-switch detection.
- scene: lightweight TV/music/crowd/noise background tagging.
"""
from relayguard.context.vad import Turn, get_speech_frames, segment_turns
from relayguard.context import conversation, change_detect, scene

__all__ = [
    "Turn",
    "get_speech_frames",
    "segment_turns",
    "conversation",
    "change_detect",
    "scene",
]
