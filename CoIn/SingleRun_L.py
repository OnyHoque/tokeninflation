"""Run one batch through the learned verifier.

This file is intentionally small: the shared scoring and validation logic lives in
``library_v1.py``. Keeping this wrapper thin avoids duplicated code and makes the
review artifact easier to audit.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from library_v1 import (
    DEFAULT_MODEL_A_DIR,
    DEFAULT_MODEL_B_DIR,
    DEFAULT_RNN_VERIFIER_PATH,
    CoInVerifier,
)


class SingleRunClassifier:
    """Convenience wrapper for the RNN-based learned verifier."""

    def __init__(
        self,
        model_a_dir: str = DEFAULT_MODEL_A_DIR,
        model_b_dir: str = DEFAULT_MODEL_B_DIR,
        token_ratio: float = 0.5,
        verifier_model_path: str = DEFAULT_RNN_VERIFIER_PATH,
    ) -> None:
        self.verifier = CoInVerifier(
            model_a_dir=model_a_dir,
            model_b_dir=model_b_dir,
            verifier_model_type="RNN",
            verifier_hidden_dim=256,
            verifier_model_path=verifier_model_path,
            token_ratio=token_ratio,
        )
        self.infer_a = self.verifier.infer_a
        self.infer_b = self.verifier.infer_b
        self.token_ratio = token_ratio

    def predict(self, sample: Iterable[Dict[str, Any]], v_thresh: float, p_ratio: float) -> List[int]:
        """Return learned-verifier decisions for ``sample``."""
        return self.verifier.verify_samples(
            sample,
            probing_ratio=p_ratio,
            verification_threshold=v_thresh,
            token_ratio=self.token_ratio,
        )


def singleRun(sample: Iterable[Dict[str, Any]], v_thresh: float = 0.6, p_ratio: float = 0.75) -> List[int]:
    """Backward-compatible function for older experiment scripts."""
    runner = SingleRunClassifier()
    return runner.predict(sample, v_thresh=v_thresh, p_ratio=p_ratio)


# Backward-compatible class name used by older scripts.
SingleRun_class = SingleRunClassifier
