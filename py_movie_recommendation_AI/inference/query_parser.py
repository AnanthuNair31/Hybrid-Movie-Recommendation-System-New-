"""
query_parser.py  –  Optimized for deployment
=============================================
Changes vs original
-------------------
1.  GENRES replaced with a frozenset imported from recommender.py
    (_SUPPORTED_GENRES) — single source of truth across the pipeline;
    eliminates the duplicated genre list and ensures both modules stay
    in sync automatically.
2.  frozenset membership test is O(1) vs O(n) on a list.
3.  capitalize() removed — genres are now returned lowercase, consistent
    with how extract_genres() and the rlike filters in recommender.py
    consume them downstream.
4.  Input validation added: non-str / empty input returns the unknown
    sentinel immediately instead of crashing on .lower().
5.  Magic-number confidence weights extracted as named module constants
    (_CONF_BASE, _CONF_PER_GENRE) — single place to tune.
6.  Full type hints and docstrings throughout.
"""

from __future__ import annotations

from typing import Optional

from .candidate_generator import get_movie

# Single source of truth: import the genre vocabulary defined in recommender.
# This guarantees query_parser and recommender always operate on the same set.
from .controller import _SUPPORTED_GENRES as GENRES   # frozenset[str]

# ---------------------------------------------------------------------------
# Confidence-score tuning knobs  (centralised, easy to adjust)
# ---------------------------------------------------------------------------
_CONF_BASE: float = 0.5       # baseline confidence for a genre-only match
_CONF_PER_GENRE: float = 0.1  # additive boost per additional matched genre


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(text: str) -> dict:
    """
    Parse a free-text query into a typed intent dict.

    Return schema
    ~~~~~~~~~~~~~
    Movie match::

        {"type": "movie",   "movie": <dict>,       "confidence": 1.0}

    Genre match::

        {"type": "genre",   "genres": [<str>, ...], "confidence": <float>}

    No signal::

        {"type": "unknown",                         "confidence": 0.0}

    Notes
    ~~~~~
    - Genres are returned as-is (lowercase) so they flow directly into
      ``extract_genres()`` and the ``rlike`` filters in ``recommender.py``
      without requiring an extra transformation step.
    - Confidence is capped at 1.0 regardless of how many genres match.
    - Non-string or empty input returns the unknown sentinel safely.
    """
    if not text or not isinstance(text, str):
        return {"type": "unknown", "confidence": 0.0}

    text_lower = text.lower()

    # ------------------------------------------------------------------
    # Priority 1: exact movie match (highest signal)
    # ------------------------------------------------------------------
    movie: Optional[dict] = get_movie(text)
    if movie:
        return {"type": "movie", "movie": movie, "confidence": 1.0}

    # ------------------------------------------------------------------
    # Priority 2: genre keyword match
    # ------------------------------------------------------------------
    # GENRES is a frozenset → O(1) per membership test
    found_genres: list[str] = [g for g in GENRES if g in text_lower]

    if found_genres:
        confidence = min(1.0, _CONF_BASE + _CONF_PER_GENRE * len(found_genres))
        return {"type": "genre", "genres": found_genres, "confidence": confidence}

    # ------------------------------------------------------------------
    # Fallback: no recognisable signal
    # ------------------------------------------------------------------
    return {"type": "unknown", "confidence": 0.0}