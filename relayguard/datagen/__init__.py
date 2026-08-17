"""relayguard.datagen — labeled data generation for relay/direct detection."""
from relayguard.datagen.chain import (
    CODECS,
    DEVICE_PRESETS,
    N_SAMPLES,
    SIMULATORS,
    SR,
    codec_roundtrip,
    fit_length,
    simulate_direct,
    simulate_hardneg_car,
    simulate_hardneg_headset,
    simulate_hardneg_ns,
    simulate_hardneg_reverb,
    simulate_hardneg_tv,
    simulate_relay,
)

__all__ = [
    "CODECS",
    "DEVICE_PRESETS",
    "N_SAMPLES",
    "SIMULATORS",
    "SR",
    "codec_roundtrip",
    "fit_length",
    "simulate_direct",
    "simulate_hardneg_car",
    "simulate_hardneg_headset",
    "simulate_hardneg_ns",
    "simulate_hardneg_reverb",
    "simulate_hardneg_tv",
    "simulate_relay",
]
