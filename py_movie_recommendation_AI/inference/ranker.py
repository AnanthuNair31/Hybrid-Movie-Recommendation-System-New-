"""
xgb_ranker.py  –  Optimized for deployment
===========================================
Changes vs original
-------------------
1.  Lazy model loading: ranker is None at import time; loaded on first
    rank_candidates() call and cached in _ranker. Zero blocking I/O on import.
2.  Model load wrapped in try/except with a clear FileNotFoundError and a
    generic fallback — avoids a silent None-ranker or cryptic XGBoost error.
3.  Dead import (numpy) removed.
4.  pdf is never mutated in-place: result is written to an explicit .copy()
    so the caller's DataFrame is always preserved.
5.  Required feature columns validated before .values access — raises
    ValueError with the exact missing column names instead of a KeyError.
6.  intent_weight clamped to (0.0, ∞) with a minimum of _MIN_WEIGHT so
    negative / zero values cannot invert or zero-out scores.
7.  NaN / inf guard on XGBoost output: corrupt scores are replaced with 0.0
    rather than poisoning the entire ranking silently.
8.  Full type hints and docstring throughout.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_PATH: str = "model/xgb_ranker.json"

_FEATURE_COLS: tuple[str, ...] = ("ALS_score", "avg_rating", "rating_count")

# Hard floor for intent_weight — prevents zero / negative score inversion
_MIN_WEIGHT: float = 1e-6


# ---------------------------------------------------------------------------
# Lazy model accessor
# ---------------------------------------------------------------------------
_ranker: Optional[xgb.XGBRanker] = None


def _get_ranker() -> xgb.XGBRanker:
    """
    Return the loaded XGBRanker, loading from disk on first call.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at ``_MODEL_PATH``.
    RuntimeError
        If XGBoost raises any other error during load.
    """
    global _ranker
    if _ranker is None:
        model = xgb.XGBRanker()
        try:
            model.load_model(_MODEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"XGBRanker model not found at '{_MODEL_PATH}'. "
                "Ensure the model artefact is present before inference."
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load XGBRanker from '{_MODEL_PATH}': {exc}"
            ) from exc
        _ranker = model
    return _ranker


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_candidates(pdf: pd.DataFrame, intent_weight: float = 1.0) -> pd.DataFrame:
    """
    Score and attach a ``score`` column to a copy of *pdf*.

    The raw XGBoost ranking score is multiplied by *intent_weight* to let
    the caller boost explicit genre-query results relative to implicit ones.

    Parameters
    ----------
    pdf:
        Pandas DataFrame that must contain the columns
        ``ALS_score``, ``avg_rating``, and ``rating_count``.
    intent_weight:
        Multiplicative boost applied to raw XGBoost scores.
        Clamped to a minimum of ``_MIN_WEIGHT`` (≈ 1e-6) so that negative
        or zero weights cannot invert or zero-out the ranking.

    Returns
    -------
    A **copy** of *pdf* with an additional ``score`` column (float).
    NaN / inf scores produced by XGBoost are replaced with 0.0.

    Raises
    ------
    TypeError
        If *pdf* is not a pandas DataFrame.
    ValueError
        If *pdf* is empty or is missing any required feature column.
    FileNotFoundError / RuntimeError
        If the model cannot be loaded (propagated from ``_get_ranker``).
    """
    # ── Input validation ─────────────────────────────────────────────────
    if not isinstance(pdf, pd.DataFrame):
        raise TypeError(f"pdf must be a pandas DataFrame, got {type(pdf).__name__!r}")

    if pdf.empty:
        raise ValueError("pdf must not be empty.")

    missing = [c for c in _FEATURE_COLS if c not in pdf.columns]
    if missing:
        raise ValueError(
            f"pdf is missing required feature column(s): {missing}. "
            f"Expected: {list(_FEATURE_COLS)}"
        )

    # ── Clamp intent_weight ───────────────────────────────────────────────
    weight = max(_MIN_WEIGHT, float(intent_weight))

    # ── Score on a copy — never mutate caller's frame ────────────────────
    result = pdf.copy()

    X = result[list(_FEATURE_COLS)].values
    raw_scores = _get_ranker().predict(X)

    # Guard against NaN / inf from malformed rows
    import numpy as np
    safe_scores = np.where(np.isfinite(raw_scores), raw_scores, 0.0)

    result["score"] = safe_scores * weight
    return result