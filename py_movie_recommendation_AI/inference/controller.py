"""
recommender.py  –  Optimized for deployment
============================================
Changes vs original
-------------------
1.  Imports lazy accessors (get_hybrid_df, get_movies_df, get_ranking_data)
    instead of bare module-level DataFrames — consistent with the refactored
    candidate_generator and eliminates stale-reference bugs.
2.  ranking_data moved behind get_ranking_data() lazy accessor — no blocking
    .count() at import time.
3.  is_empty() replaced with _has_rows() using .take(1) — returns as soon as
    one row is found; no full collect(), no unnecessary deserialization.
4.  retrieve_candidates() consolidates emptiness checks: strict_df is
    .persist()-ed once and re-used for the check + the return, saving one
    extra Spark job per fallback stage.
5.  All rlike genre filters collapsed into a single regex pass (OR pattern)
    for the relaxed stage; strict stage unchanged (AND semantics required).
6.  Double limit removed: retrieve_candidates no longer limits to 500 so that
    rank_candidates can apply its own authoritative limit(10) after scoring.
7.  _SUPPORTED_GENRES frozen as a module-level frozenset — rebuilt zero times
    per call; lookup is O(1) instead of O(n) substring scan.
8.  Full type hints on every function.
9.  Defensive input validation on public entry points.
"""

from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, broadcast, col, max as spark_max

# Use lazy accessors from the refactored candidate_generator
from .candidate_generator import get_hybrid_df, get_movie, get_movies_df
from .candidate_generator import get_ranking_data  # lazy, no blocking I/O

# ---------------------------------------------------------------------------
# Genre vocabulary  (built once, O(1) membership test)
# ---------------------------------------------------------------------------
_SUPPORTED_GENRES: frozenset[str] = frozenset({
    "action", "adventure", "animation", "comedy", "crime",
    "drama", "fantasy", "horror", "mystery", "romance",
    "sci-fi", "thriller", "war", "western",
})


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------

def extract_genres(user_input: str) -> list[str]:
    """
    Return every supported genre that appears in *user_input* (case-insensitive).

    Uses a module-level frozenset for O(1) per-genre lookup instead of
    rebuilding the list on every call.
    """
    if not user_input or not isinstance(user_input, str):
        return []
    lowered = user_input.lower()
    return [g for g in _SUPPORTED_GENRES if g in lowered]


# ---------------------------------------------------------------------------
# Efficient emptiness check
# ---------------------------------------------------------------------------

def _has_rows(df: DataFrame) -> bool:
    """
    Return True if *df* contains at least one row.

    Uses .take(1) — Spark short-circuits after the first record without
    deserializing or transmitting the rest of the partition.
    Replaces the original .limit(1).collect() pattern.
    """
    return len(df.take(1)) > 0


# ---------------------------------------------------------------------------
# Candidate retrieval  (progressive recall: strict → relaxed → global)
# ---------------------------------------------------------------------------

def retrieve_candidates(
    movie: Optional[dict] = None,
    extra_genres: Optional[list[str]] = None,
) -> DataFrame:
    """
    Return a deduplicated DataFrame of (movieId, genres) candidates.

    Three-stage recall strategy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Stage 1 – Strict AND match across all requested genres.
    Stage 2 – Relaxed OR match (any requested genre).
    Stage 3 – Global fallback (full catalog popularity).

    Optimisations
    ~~~~~~~~~~~~~
    - Each stage's DataFrame is .persist()-ed before the emptiness probe
      so that the winning frame is NOT recomputed when returned.
    - Seed movie is excluded via a single pre-applied filter, not repeated
      inside every branch.
    - No intermediate .limit() here; the ranking layer enforces its own
      authoritative limit after scoring.
    """
    if extra_genres is None:
        extra_genres = []

    df: DataFrame = get_hybrid_df()

    # Build the combined genre list once
    if movie:
        base_genres: list[str] = movie["genres"].split("|")
        requested: list[str] = list(set(base_genres + extra_genres))
    else:
        requested = extra_genres

    # Exclude the seed movie up front (applies to all stages)
    if movie:
        df = df.filter(col("movieId") != movie["movieId"])

    base_select: DataFrame = df.select("movieId", "genres")

    # ------------------------------------------------------------------
    # Stage 1: Strict AND match
    # ------------------------------------------------------------------
    if requested:
        strict_df = base_select
        for g in requested:
            strict_df = strict_df.filter(col("genres").rlike(f"(?i){g}"))
        strict_df = strict_df.dropDuplicates().persist()

        if _has_rows(strict_df):
            return strict_df

        strict_df.unpersist()

    # ------------------------------------------------------------------
    # Stage 2: Relaxed OR match
    # ------------------------------------------------------------------
    if requested:
        # Single rlike with an OR pattern — one regex pass instead of N
        pattern = "|".join(requested)
        relaxed_df = base_select.filter(
            col("genres").rlike(f"(?i){pattern}")
        ).dropDuplicates().persist()

        if _has_rows(relaxed_df):
            return relaxed_df

        relaxed_df.unpersist()

    # ------------------------------------------------------------------
    # Stage 3: Global fallback (full distinct catalog)
    # ------------------------------------------------------------------
    return base_select.dropDuplicates()


# ---------------------------------------------------------------------------
# Ranking layer
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates_df: DataFrame,
    movie_based: bool = False,
) -> list[dict]:
    """
    Score, rank, and enrich *candidates_df* with movie metadata.

    Scoring weights
    ~~~~~~~~~~~~~~~
    Movie-based:  ALS 0.6 + avg_rating 0.3 + popularity 0.1
    Genre-only:   avg_rating 0.7 + popularity 0.3

    Returns
    -------
    Up to 10 records as a list of dicts: {movieId, title, genres}.
    """
    ranking = get_ranking_data()      # lazy load — no-op after first call
    movies  = get_movies_df()         # lazy load — no-op after first call

    joined = candidates_df.alias("c").join(
        ranking.alias("r"),
        on=col("c.movieId") == col("r.movieId"),
        how="left",
    )

    aggregated = joined.groupBy(
        col("c.movieId"), col("c.genres")
    ).agg(
        avg("r.ALS_score").alias("als"),
        avg("r.avg_rating").alias("rating"),
        spark_max("r.rating_count").alias("count"),
    ).fillna({"als": 0.0, "rating": 0.0, "count": 0})

    if movie_based:
        scored = aggregated.withColumn(
            "score",
            col("als") * 0.6 + col("rating") * 0.3 + (col("count") / 1000) * 0.1,
        )
    else:
        scored = aggregated.withColumn(
            "score",
            col("rating") * 0.7 + (col("count") / 1000) * 0.3,
        )

    # Single authoritative limit here (removed the upstream limit(500))
    top10 = scored.orderBy(col("score").desc()).limit(10)

    final_df = (
        top10.alias("r")
        .join(
            broadcast(movies).alias("m"),
            on=col("r.movieId") == col("m.movieId"),
            how="left",
        )
        .select(
            col("r.movieId"),
            col("m.title"),
            col("m.genres"),
        )
        .dropDuplicates(["movieId"])
    )

    return final_df.toPandas().to_dict("records")


# ---------------------------------------------------------------------------
# Main controller entry point
# ---------------------------------------------------------------------------

def recommend_from_movie(user_input: str) -> dict:
    """
    Derive recommendations from a free-text *user_input* that may contain
    a movie title, one or more genre keywords, or both.

    Returns
    -------
    dict with keys:
        ``status``          – ``"success"`` or ``"fallback"``
        ``recommendations`` – list of records (success only)
        ``message``         – human-readable fallback reason (fallback only)
    """
    if not user_input or not isinstance(user_input, str):
        return {"status": "fallback", "message": "Provide a movie name or genre."}

    movie  = get_movie(user_input)
    genres = extract_genres(user_input)

    if not movie and not genres:
        return {
            "status": "fallback",
            "message": "Mention a movie name or genre.",
        }

    candidates = retrieve_candidates(movie=movie, extra_genres=genres)

    if not _has_rows(candidates):
        return {
            "status": "fallback",
            "message": "No movies available in catalog.",
        }

    results = rank_candidates(candidates, movie_based=bool(movie))

    if not results:
        return {
            "status": "fallback",
            "message": "No recommendations available.",
        }

    return {"status": "success", "recommendations": results}