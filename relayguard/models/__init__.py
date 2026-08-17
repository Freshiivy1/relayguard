"""Model modules: compact CNN, training CLI, window-level detectors.

The torch-dependent symbols (RelayCNN, logmel, count_parameters) are imported
lazily via PEP 562 so that ``import relayguard.models`` (and, crucially,
``relayguard.models.detectors``) never requires torch. The deployed runtime
container is torch-free; torch is a training-only dependency.
"""

__all__ = ["RelayCNN", "logmel", "count_parameters"]


def __getattr__(name: str):
    if name in __all__:
        from relayguard.models.cnn import RelayCNN, count_parameters, logmel

        return {"RelayCNN": RelayCNN, "logmel": logmel,
                "count_parameters": count_parameters}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
