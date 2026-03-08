"""
spark_session.py  –  Optimized for deployment
==============================================
Changes vs original
-------------------
1.  All tunable parameters (master, memory, partitions, app name) read from
    environment variables with sensible defaults — zero code changes needed
    across local / staging / production environments.
2.  .master() only applied when SPARK_MASTER is explicitly set; omitted
    otherwise so that cluster-mode submission (spark-submit --master ...)
    takes full control without being silently overridden.
3.  Kryo serializer enabled — significantly faster than the default Java
    serializer for Spark shuffle and broadcast workloads.
4.  spark.sql.adaptive.enabled and spark.sql.adaptive.coalescePartitions
    both set — AQE automatically right-sizes shuffle partitions at runtime,
    making the static shuffle.partitions value a ceiling, not a fixed cost.
5.  Broadcast join threshold made configurable via env var.
6.  spark.sql.shuffle.partitions falls back to 2× the available CPU cores
    when the env var is absent — better default than the arbitrary 200.
7.  Structured logging (stdlib logging) records whether the session was
    freshly created or reused from an existing context.
8.  Full type annotation on the exported `spark` symbol.
9.  All constants named at the top of the module — single place to audit
    every tunable knob.
"""

from __future__ import annotations

import logging
import multiprocessing
import os

from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables — all overridable via environment variables
# ---------------------------------------------------------------------------
_APP_NAME: str = os.getenv("SPARK_APP_NAME", "MovieRecommendationInference")

# When SPARK_MASTER is unset we do NOT call .master() so that
# spark-submit / cluster managers can supply the master themselves.
_MASTER: str | None = os.getenv("SPARK_MASTER")          # e.g. "local[*]", "yarn"

_DRIVER_MEMORY: str = os.getenv("SPARK_DRIVER_MEMORY", "8g")
_EXECUTOR_MEMORY: str = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
_EXECUTOR_CORES: str = os.getenv("SPARK_EXECUTOR_CORES", "2")

# Default shuffle partitions to 2× logical CPUs; overridable via env var.
_DEFAULT_SHUFFLE_PARTITIONS: str = str(
    int(os.getenv("SPARK_SHUFFLE_PARTITIONS", str(multiprocessing.cpu_count() * 2)))
)

# Broadcast threshold: DataFrames smaller than this (bytes) are auto-broadcast.
_BROADCAST_THRESHOLD: str = os.getenv("SPARK_BROADCAST_THRESHOLD", str(20 * 1024 * 1024))  # 20 MB


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _build_session() -> SparkSession:
    """
    Build or retrieve the SparkSession.

    Configuration strategy
    ~~~~~~~~~~~~~~~~~~~~~~
    - All settings read from env vars so the same codebase runs unmodified
      in local dev, CI, and production cluster environments.
    - .master() is only set when ``SPARK_MASTER`` is explicitly provided;
      this prevents the hardcoded value from shadowing the ``--master`` flag
      passed to ``spark-submit`` in cluster deployments.
    - Kryo serialisation is enabled for faster shuffle / broadcast I/O.
    - Adaptive Query Execution (AQE) + partition coalescing are both
      enabled so Spark right-sizes shuffle partitions at runtime rather
      than using a fixed static value.

    Returns
    -------
    SparkSession (newly created or reused from existing context).
    """
    builder = (
        SparkSession.builder
        .appName(_APP_NAME)
        # ── Memory & parallelism ──────────────────────────────────────────
        .config("spark.driver.memory",              _DRIVER_MEMORY)
        .config("spark.executor.memory",            _EXECUTOR_MEMORY)
        .config("spark.executor.cores",             _EXECUTOR_CORES)
        # ── Serialization ─────────────────────────────────────────────────
        .config("spark.serializer",                 "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryo.unsafe",                "true")
        # ── Adaptive Query Execution ──────────────────────────────────────
        .config("spark.sql.adaptive.enabled",               "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions",             _DEFAULT_SHUFFLE_PARTITIONS)
        # ── Join optimisation ─────────────────────────────────────────────
        .config("spark.sql.autoBroadcastJoinThreshold",     _BROADCAST_THRESHOLD)
    )

    # Only set master when explicitly provided — avoids shadowing spark-submit
    if _MASTER:
        builder = builder.master(_MASTER)

    # getOrCreate() returns an existing session if one already exists.
    existing = SparkSession.getActiveSession()
    session  = builder.getOrCreate()

    if existing is None:
        logger.info(
            "SparkSession created | app=%s | driver_mem=%s | shuffle_partitions=%s",
            _APP_NAME, _DRIVER_MEMORY, _DEFAULT_SHUFFLE_PARTITIONS,
        )
    else:
        logger.debug("SparkSession reused (already active) | app=%s", _APP_NAME)

    return session


# ---------------------------------------------------------------------------
# Module-level export
# ---------------------------------------------------------------------------
spark: SparkSession = _build_session()