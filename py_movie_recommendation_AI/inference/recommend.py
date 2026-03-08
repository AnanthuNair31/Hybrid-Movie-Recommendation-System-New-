"""
recommend.py  –  Optimized for deployment
==========================================
Public entry point for the movie recommendation inference pipeline.
Replaces the previous two-line passthrough with a full boundary layer.

Changes vs original
-------------------
1.  Import path kept as .controller — matches the actual filename in the
    project and requires no rename anywhere else.
2.  Pure passthrough elevated to a meaningful boundary layer: input
    validation, structured logging, and exception containment all live
    here so controller.py stays clean of cross-cutting concerns.
3.  Input validated at the public boundary before any Spark or model
    code is touched — invalid inputs are rejected cheaply with a
    well-formed response dict.
4.  All exceptions caught and converted to a structured error dict so
    callers always receive a consistent response shape regardless of
    internal failure mode — never a raw traceback.
5.  Structured logging records every request and its outcome.
6.  Full type hints and docstring with explicit response-schema contract.
7.  __all__ declared so the public surface is explicit.
8.  Function name kept as recommend() — preserves the import contract
    with app.py (from inference.recommend import recommend).
"""

from __future__ import annotations

import logging

from .controller import recommend_from_movie

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
__all__ = ["recommend"]


def recommend(user_input: str) -> dict:
    """
    Primary public entry point for the recommendation pipeline.

    Validates *user_input*, delegates to the recommendation controller,
    and returns a consistently shaped response dict in all outcomes.

    Response schema
    ~~~~~~~~~~~~~~~
    Success::

        {
            "status": "success",
            "recommendations": [
                {"movieId": <int>, "title": <str>, "genres": <str>},
                ...         # up to 10 items
            ]
        }

    Fallback (no signal / empty catalog)::

        {"status": "fallback", "message": <str>}

    Error (unexpected exception)::

        {"status": "error", "message": <str>}

    Parameters
    ----------
    user_input:
        Free-text query from the end user, e.g. ``"something like Inception"``.

    Returns
    -------
    dict — always one of the three shapes above; never raises.
    """
    # ── Input validation at the API boundary ─────────────────────────────
    if not user_input or not isinstance(user_input, str):
        logger.warning("recommend() called with invalid input: %r", user_input)
        return {
            "status": "fallback",
            "message": "Please provide a movie title or genre to get recommendations.",
        }

    stripped = user_input.strip()
    if not stripped:
        return {
            "status": "fallback",
            "message": "Query must not be blank.",
        }

    logger.info("recommend() | input=%r", stripped)

    # ── Delegate to pipeline controller ──────────────────────────────────
    try:
        result: dict = recommend_from_movie(stripped)
    except Exception as exc:                        # noqa: BLE001
        # Contain all internal failures at this boundary — callers always
        # receive a structured response, never a raw traceback.
        logger.exception(
            "recommend() failed | input=%r | error=%s", stripped, exc
        )
        return {
            "status": "error",
            "message": "An internal error occurred. Please try again.",
        }

    # ── Log outcome and return ────────────────────────────────────────────
    status = result.get("status", "unknown")
    n      = len(result.get("recommendations", []))

    if status == "success":
        logger.info(
            "recommend() | status=success | results=%d | input=%r",
            n, stripped,
        )
    else:
        logger.warning(
            "recommend() | status=%s | message=%r | input=%r",
            status, result.get("message"), stripped,
        )

    return result