"""Evaluation harness (SPEC section 4 C2)."""

from relayguard.eval.evaluate import (
    compute_metrics,
    eer,
    hit_rate_at_fpr,
    per_slice_table,
)

__all__ = ["compute_metrics", "eer", "hit_rate_at_fpr", "per_slice_table"]
