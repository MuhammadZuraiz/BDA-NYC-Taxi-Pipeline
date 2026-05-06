#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etl.py — PySpark ETL Pipeline
CS-404 Big Data Analytics — Assignment 3
Dataset : NYC Yellow Taxi Trip Data (January 2023)
Picks up from A2: data already ingested to HDFS at
  /warehouse/raw/nyc_yellow_taxi/year=2023/month=01/

Pipeline steps (in sequence, no manual intervention required):
  1. TRANSFORM — clean, derive new columns, normalise categoricals,
                 standardise datetimes.  Every transformation cites
                 the A2 profiling finding that motivated it.
  2. LOAD      — write star-schema tables to HDFS as partitioned Parquet
  3. VALIDATE  — assert row counts, null checks on key columns,
                 log a per-table summary

Usage (from the Assignment2 project root):
  spark-submit etl.py
  -- or --
  python etl.py          (uses local[*] Spark session automatically)
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("etl")

# ── PySpark ───────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, TimestampType, BooleanType
)
from pyspark.sql.window import Window

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH       = "/warehouse/raw/nyc_yellow_taxi/year=2023/month=01"
PROCESSED_BASE = "/warehouse/processed"

# Local fallback paths (used when HDFS is not reachable from local Spark)
LOCAL_RAW      = "raw_data/yellow_tripdata_2023-01.parquet"
LOCAL_OUT      = "processed"

# ── Constants ─────────────────────────────────────────────────────────────────
# IQR multiplier for total_amount Winsorisation
# A2 finding: 371,616 outliers (12.12%) detected using IQR method
IQR_MULTIPLIER = 3.0


# =============================================================================
# SPARK SESSION
# =============================================================================
def get_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("BDA_A3_ETL_NYC_Taxi")
        # Use 2 cores only — local[*] spins up too many threads on Windows,
        # each claiming memory simultaneously -> OOM
        .master("local[2]")

        # ── Driver memory (the JVM process running on your machine) ──────────
        # Default is only 1g which is not enough for 3M rows.
        # Set to 4g — adjust down to 3g if you have less than 8 GB RAM total.
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")

        # ── Executor memory (same JVM in local mode) ─────────────────────────
        .config("spark.executor.memory", "4g")

        # ── Off-heap memory for columnar operations ───────────────────────────
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "1g")

        # ── Reduce shuffle partitions — 200 default is excessive for 3M rows ─
        .config("spark.sql.shuffle.partitions", "8")

        # ── Adaptive query execution ─────────────────────────────────────────
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")

        # ── Parquet reading optimizations ────────────────────────────────────
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128 MB per partition

        # ── Prevent OOM during sort/shuffle ──────────────────────────────────
        .config("spark.sql.autoBroadcastJoinThreshold", "10m")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")

        # ── GC tuning — reduces GC pauses on large datasets ──────────────────
        .config("spark.executor.extraJavaOptions",
                "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")
        .config("spark.driver.extraJavaOptions",
                "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session started: %s", spark.version)
    logger.info("Driver memory : 4g | Executor memory: 4g | Off-heap: 1g | Cores: 2")
    return spark


# =============================================================================
# LOAD RAW DATA
# =============================================================================
def load_raw(spark: SparkSession):
    """Load the raw Parquet file from HDFS (or local fallback)."""
    logger.info("=" * 60)
    logger.info("LOADING raw data")
    logger.info("=" * 60)

    # Try HDFS first, fall back to local file
    try:
        df = spark.read.parquet(RAW_PATH)
        logger.info("Loaded from HDFS: %s", RAW_PATH)
    except Exception as hdfs_err:
        logger.warning("HDFS read failed (%s) — using local file.", hdfs_err)
        df = spark.read.parquet(LOCAL_RAW)
        logger.info("Loaded from local: %s", LOCAL_RAW)

    raw_count = df.count()
    logger.info("Raw row count: %s", f"{raw_count:,}")
    logger.info("Columns: %s", df.columns)
    return df, raw_count


# =============================================================================
# STEP 1: TRANSFORM
# =============================================================================
def step_transform(df):
    """
    Apply all cleaning and feature-engineering transformations.
    Each block references the A2 profiling finding that motivated it.
    """
    logger.info("=" * 60)
    logger.info("STEP 1 - TRANSFORM")
    logger.info("=" * 60)

    # ── 1a. Drop exact duplicates ────────────────────────────────────────────
    # A2 finding: 0 duplicate rows detected; applied defensively as a best
    # practice before any downstream joins.
    # NOTE: count() calls removed — each triggers a full scan doubling peak memory.
    df = df.dropDuplicates()
    # Row counts are reported in step_validate after Parquet is written instead.
    logger.info("1a. Deduplication applied (A2: 0 duplicates found).")

    # ── 1b. Remove zero / negative trip_distance ─────────────────────────────
    # A2 finding: 45,862 rows (1.50%) have trip_distance <= 0.
    # These represent cancelled or null trips and would skew distance
    # aggregations in the fact table.
    df = df.filter(F.col("trip_distance") > 0)
    logger.info("1b. Removed zero/negative trip_distance rows.")

    # ── 1c. Fix negative fare_amount ─────────────────────────────────────────
    # A2 finding: 25,049 rows (0.82%) have negative fare_amount (data-entry
    # errors). Replace with median fare for the same trip_distance decile,
    # as median is robust to outliers in this right-skewed column.
    # Approximation: use overall median as PySpark percentile_approx.
    median_fare = df.filter(F.col("fare_amount") > 0) \
                    .approxQuantile("fare_amount", [0.5], 0.01)[0]
    df = df.withColumn(
        "fare_amount",
        F.when(F.col("fare_amount") < 0, F.lit(median_fare))
         .otherwise(F.col("fare_amount"))
    )
    logger.info("1c. Replaced negative fare_amount with median (%.2f).", median_fare)

    # ── 1d. Winsorise total_amount outliers ───────────────────────────────────
    # A2 finding: 371,616 rows (12.12%) are outliers by IQR method.
    # Cap at Q3 + 3*IQR to retain airport/long-distance fares while
    # removing erroneous spikes.
    q1, q3 = df.approxQuantile("total_amount", [0.25, 0.75], 0.01)
    iqr     = q3 - q1
    upper   = q3 + IQR_MULTIPLIER * iqr
    lower   = q1 - IQR_MULTIPLIER * iqr
    df = df.withColumn(
        "total_amount",
        F.when(F.col("total_amount") > upper, F.lit(upper))
         .when(F.col("total_amount") < lower, F.lit(lower))
         .otherwise(F.col("total_amount"))
    )
    logger.info("1d. Winsorised total_amount to [%.2f, %.2f].", lower, upper)

    # ── 1e. Impute missing passenger_count ────────────────────────────────────
    # A2 finding: 71,743 rows (2.34%) have null passenger_count.
    # Impute with mode (1) — the dominant category (55%+ of records).
    df = df.withColumn(
        "passenger_count",
        F.when(F.col("passenger_count").isNull(), F.lit(1))
         .when(F.col("passenger_count") == 0, F.lit(1))
         .otherwise(F.col("passenger_count").cast(IntegerType()))
    )
    logger.info("1e. Imputed null/zero passenger_count with 1 (mode).")

    # ── 1f. Normalise store_and_fwd_flag to Boolean ───────────────────────────
    # A2 finding: 71,743 nulls (2.34%) in store_and_fwd_flag.
    # Cast: Y -> True, N -> False, null -> False (offline trip).
    df = df.withColumn(
        "store_and_fwd_flag",
        F.when(F.col("store_and_fwd_flag") == "Y", F.lit(True))
         .otherwise(F.lit(False))
    )
    logger.info("1f. Normalised store_and_fwd_flag to Boolean.")

    # ── 1g. Impute nulls in RatecodeID, congestion_surcharge, airport_fee ─────
    # A2 finding: 71,743 nulls (2.34%) in RatecodeID, congestion_surcharge,
    # and airport_fee — all from the same automated-dispatch vendor records.
    df = df.withColumn(
        "RatecodeID",
        F.when(F.col("RatecodeID").isNull(), F.lit(1.0))
         .otherwise(F.col("RatecodeID"))
    )
    df = df.withColumn(
        "congestion_surcharge",
        F.coalesce(F.col("congestion_surcharge"), F.lit(0.0))
    )
    df = df.withColumn(
        "airport_fee",
        F.coalesce(F.col("airport_fee"), F.lit(0.0))
    )
    logger.info("1g. Imputed nulls in RatecodeID / congestion_surcharge / airport_fee.")

    # ── 1h. Standardise datetime columns ─────────────────────────────────────
    # Ensure both datetime columns are proper TimestampType for
    # time-series analysis in A3 queries.
    df = df.withColumn(
        "tpep_pickup_datetime",
        F.col("tpep_pickup_datetime").cast(TimestampType())
    ).withColumn(
        "tpep_dropoff_datetime",
        F.col("tpep_dropoff_datetime").cast(TimestampType())
    )
    logger.info("1h. Cast pickup/dropoff to TimestampType.")

    # ── 1i. Derive new columns ────────────────────────────────────────────────
    # trip_duration_mins: required for business question 4 (fare vs distance
    # vs duration efficiency analysis).
    df = df.withColumn(
        "trip_duration_mins",
        (F.unix_timestamp("tpep_dropoff_datetime") -
         F.unix_timestamp("tpep_pickup_datetime")) / 60.0
    )

    # pickup_hour, pickup_dayofweek, pickup_month: required for time-series
    # queries (business questions 1 and 2).
    df = df.withColumn("pickup_hour",       F.hour("tpep_pickup_datetime")) \
           .withColumn("pickup_dayofweek",  F.dayofweek("tpep_pickup_datetime")) \
           .withColumn("pickup_month",      F.month("tpep_pickup_datetime")) \
           .withColumn("pickup_date",       F.to_date("tpep_pickup_datetime"))

    # is_weekend: categorical flag for weekend vs weekday demand analysis.
    df = df.withColumn(
        "is_weekend",
        F.when(F.col("pickup_dayofweek").isin([1, 7]), True).otherwise(False)
    )

    # time_of_day_bucket: morning/afternoon/evening/night segmentation.
    df = df.withColumn(
        "time_of_day",
        F.when((F.col("pickup_hour") >= 6)  & (F.col("pickup_hour") < 12), "Morning")
         .when((F.col("pickup_hour") >= 12) & (F.col("pickup_hour") < 17), "Afternoon")
         .when((F.col("pickup_hour") >= 17) & (F.col("pickup_hour") < 21), "Evening")
         .otherwise("Night")
    )

    # payment_label: human-readable payment type (business question 3).
    df = df.withColumn(
        "payment_label",
        F.when(F.col("payment_type") == 1, "Credit Card")
         .when(F.col("payment_type") == 2, "Cash")
         .when(F.col("payment_type") == 3, "No Charge")
         .when(F.col("payment_type") == 4, "Dispute")
         .otherwise("Unknown")
    )

    # tip_pct: tip as percentage of fare (business question 3 & 4).
    df = df.withColumn(
        "tip_pct",
        F.when(F.col("fare_amount") > 0,
               F.round(F.col("tip_amount") / F.col("fare_amount") * 100, 2))
         .otherwise(F.lit(0.0))
    )

    # revenue_per_mile: fare efficiency metric (business question 1 & 4).
    df = df.withColumn(
        "revenue_per_mile",
        F.when(F.col("trip_distance") > 0,
               F.round(F.col("fare_amount") / F.col("trip_distance"), 2))
         .otherwise(F.lit(0.0))
    )

    # Remove implausible trip durations (negative or > 6 hours)
    df = df.filter(
        (F.col("trip_duration_mins") > 0) & (F.col("trip_duration_mins") <= 360)
    )

    logger.info("1i. Derived columns: trip_duration_mins, pickup_hour, pickup_dayofweek, "
                "pickup_month, pickup_date, is_weekend, time_of_day, payment_label, "
                "tip_pct, revenue_per_mile.")

    clean_count = -1  # Count deferred to validate step to avoid OOM
    logger.info("TRANSFORM complete. Row count logged in validate step.")
    return df, clean_count


# =============================================================================
# BUILD STAR SCHEMA TABLES
# =============================================================================
def build_star_schema(df):
    """Decompose the cleaned flat table into fact + dimension tables."""
    logger.info("=" * 60)
    logger.info("BUILDING star schema tables")
    logger.info("=" * 60)

    # ── Fact table: fact_trips ────────────────────────────────────────────────
    fact_trips = df.select(
        F.monotonically_increasing_id().alias("trip_id"),
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "pickup_date",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "time_of_day",
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "trip_distance",
        "trip_duration_mins",
        "RatecodeID",
        "payment_type",
        "payment_label",
        "store_and_fwd_flag",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tip_pct",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "airport_fee",
        "revenue_per_mile",
    )

    # ── Dimension: dim_time ───────────────────────────────────────────────────
    dim_time = df.select(
        "pickup_date",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "time_of_day",
    ).distinct()

    # ── Dimension: dim_location ───────────────────────────────────────────────
    dim_location = df.select(
        F.col("PULocationID").alias("location_id")
    ).union(
        df.select(F.col("DOLocationID").alias("location_id"))
    ).distinct().orderBy("location_id")

    # ── Dimension: dim_payment ────────────────────────────────────────────────
    dim_payment = df.select(
        "payment_type",
        "payment_label"
    ).distinct().orderBy("payment_type")

    # ── Dimension: dim_vendor ─────────────────────────────────────────────────
    dim_vendor = df.select("VendorID").distinct()

    logger.info("Star schema tables built. Row counts logged in validate step.")





    return {
        "fact_trips"   : fact_trips,
        "dim_time"     : dim_time,
        "dim_location" : dim_location,
        "dim_payment"  : dim_payment,
        "dim_vendor"   : dim_vendor,
    }


# =============================================================================
# STEP 2: LOAD
# =============================================================================
def step_load(tables: dict) -> dict:
    """
    Write each table to HDFS (or local) as partitioned Parquet.
    Uses snappy compression and Hive-compatible partition columns.
    """
    logger.info("=" * 60)
    logger.info("STEP 2 - LOAD: Writing Parquet tables")
    logger.info("=" * 60)

    written = {}
    for name, df in tables.items():
        # Try HDFS path first, fall back to local
        hdfs_path  = f"{PROCESSED_BASE}/{name}"
        local_path = f"{LOCAL_OUT}/{name}"

        for path in [local_path]:   # use local_path for portability on Windows
            try:
                if name == "fact_trips":
                    # Partition fact table by pickup_month for efficient
                    # time-range queries (Optimization: partitioning)
                    df.write.mode("overwrite") \
                      .partitionBy("pickup_month") \
                      .parquet(path)
                else:
                    df.write.mode("overwrite").parquet(path)

                logger.info("Wrote %-20s -> %s", name, path)
                written[name] = path
                break
            except Exception as exc:
                logger.error("Failed to write %s to %s: %s", name, path, exc)
                raise

    return written


# =============================================================================
# STEP 3: VALIDATE
# =============================================================================
def step_validate(
    spark: SparkSession,
    written_paths: dict,
    raw_count: int,
    clean_count: int,
) -> None:
    """
    Re-read each written Parquet table and assert:
      - Row counts are non-zero
      - Key columns have no nulls
      - Log a per-table summary
    """
    logger.info("=" * 60)
    logger.info("STEP 3 - VALIDATE")
    logger.info("=" * 60)

    logger.info("Raw row count (before ETL): %s", f"{raw_count:,}")

    # Null checks: single-pass agg on fact_trips (avoids N filter().count() OOM calls).
    # Small dim tables use per-column check (they are tiny).
    null_checks_dims = {
        "dim_time"    : ["pickup_date", "pickup_hour"],
        "dim_payment" : ["payment_type", "payment_label"],
        "dim_vendor"  : ["VendorID"],
    }

    logger.info("%-20s  %-12s  %-8s  %s", "Table", "Rows", "Nulls OK", "Path")
    logger.info("-" * 70)

    fact_count = 0
    for name, path in written_paths.items():
        df = spark.read.parquet(path)
        row_count = df.count()
        assert row_count > 0, f"VALIDATION FAILED: {name} has 0 rows!"

        if name == "fact_trips":
            fact_count = row_count
            key_cols = ["trip_distance", "fare_amount", "total_amount", "payment_type"]
            existing = [c for c in key_cols if c in df.columns]
            null_expr = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in existing]
            null_counts = df.agg(*null_expr).collect()[0]
            null_ok = all(null_counts[c] == 0 for c in existing)
        else:
            null_ok = True
            for col in null_checks_dims.get(name, []):
                if col in df.columns:
                    n = df.filter(F.col(col).isNull()).count()
                    if n > 0:
                        logger.warning("  NULL FAILED: %s.%s has %d nulls", name, col, n)
                        null_ok = False

        status = "[OK]" if null_ok else "[WARN]"
        logger.info("%-20s  %-12s  %-8s  %s", name, f"{row_count:,}", status, path)

    removed = raw_count - fact_count
    logger.info("Rows removed by cleaning: %s (%.2f%% of raw)",
                f"{removed:,}", removed / raw_count * 100 if raw_count > 0 else 0)
    logger.info("Validation complete.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("BDA A3 - PySpark ETL Pipeline (etl.py)")
    logger.info("Dataset : NYC Yellow Taxi Trip Data 2023-01")
    logger.info("=" * 60)

    spark = get_spark()

    try:
        # 0. Load raw
        raw_df, raw_count = load_raw(spark)

        # 1. Transform
        clean_df, clean_count = step_transform(raw_df)

        # Cache cleaned DataFrame — reused for star schema table builds
        # Optimization: caching avoids recomputing the transformation DAG
        clean_df.cache()
        logger.info("Cached cleaned DataFrame.")

        # 2. Build star schema
        tables = build_star_schema(clean_df)

        # 3. Load
        written = step_load(tables)

        # 4. Validate
        step_validate(spark, written, raw_count, clean_count)

        logger.info("=" * 60)
        logger.info("[SUCCESS] ETL pipeline completed.")
        logger.info("Log: %s", LOG_FILE)
        logger.info("=" * 60)

    except Exception as exc:
        logger.error("[FAILED] ETL pipeline error: %s", exc, exc_info=True)
        raise
    finally:
        spark.stop()
        logger.info("Spark session stopped.")


if __name__ == "__main__":
    main()
