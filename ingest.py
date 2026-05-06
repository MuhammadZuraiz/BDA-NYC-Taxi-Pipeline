#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest.py - Fully Automated HDFS Ingestion Pipeline
BDA Assignment 2 | NYC TLC Yellow Taxi Trip Data (2023-01)

Steps executed in sequence (no manual intervention required):
  1. LOAD     - Download raw dataset from NYC TLC open-data CDN
  2. VALIDATE - File integrity (size, format, extension, encoding, row count)
  3. UPLOAD   - Upload validated file to HDFS via WebHDFS API
  4. ORGANIZE - Confirm structured HDFS directory layout
  5. LOG      - Record all successes, warnings, and errors

Usage:
  python ingest.py

Requirements:
  pip install hdfs pyarrow chardet
  Hadoop 3.x must be running (start-dfs.sh / start-all.sh)
  WebHDFS must be enabled (dfs.webhdfs.enabled=true in hdfs-site.xml)
"""

import hashlib
import logging
import os
import subprocess
import sys
import urllib.request
import chardet
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Windows-safe console stream  (fixes UnicodeEncodeError on cp1252 terminals)
# ---------------------------------------------------------------------------
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

_fmt    = "%(asctime)s  %(levelname)-8s  %(message)s"
_datefmt = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_fmt,
    datefmt=_datefmt,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2023-01.parquet"
)

LOCAL_RAW_DIR = Path("raw_data")
LOCAL_RAW_DIR.mkdir(exist_ok=True)
LOCAL_FILE    = LOCAL_RAW_DIR / "yellow_tripdata_2023-01.parquet"

YEAR         = "2023"
MONTH        = "01"
HDFS_BASE    = "/warehouse/raw"
HDFS_DATASET = "nyc_yellow_taxi"
HDFS_TARGET  = f"{HDFS_BASE}/{HDFS_DATASET}/year={YEAR}/month={MONTH}"

WEBHDFS_URL  = "http://localhost:9870"
# On Windows, hadoop shell command is hadoop.cmd
# Also check common install locations if not in PATH
import shutil as _shutil
_hadoop_name = "hadoop.cmd" if sys.platform == "win32" else "hadoop"
_hadoop_path = _shutil.which(_hadoop_name)
if _hadoop_path is None:
    # Try common Windows install paths
    for _candidate in [
        r"C:\hadoop\bin\hadoop.cmd",
        r"C:\Program Files\Hadoop\bin\hadoop.cmd",
    ]:
        if __import__("os").path.exists(_candidate):
            _hadoop_path = _candidate
            break
HADOOP_CMD   = _hadoop_path or _hadoop_name  # fallback to name (will fail with clear error)
HDFS_USER    = os.environ.get("HDFS_USER", "zurai")

MIN_ROWS     = 500_000
EXPECTED_EXT = {".parquet", ".csv", ".json", ".gz"}


# ---------------------------------------------------------------------------
# Helper: run a shell command and log output
# ---------------------------------------------------------------------------
def run_cmd(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    logger.info("CMD: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.stdout.strip():
        logger.info("STDOUT: %s", result.stdout.strip())
    if result.stderr.strip():
        logger.warning("STDERR: %s", result.stderr.strip())
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}): {' '.join(cmd)}"
        )
    return result


# ===========================================================================
# STEP 1: LOAD
# ===========================================================================
def step_load() -> Path:
    logger.info("=" * 60)
    logger.info("STEP 1 - LOAD: Downloading dataset from source")
    logger.info("=" * 60)

    if LOCAL_FILE.exists():
        logger.info("File already present at %s -- skipping download.", LOCAL_FILE)
        return LOCAL_FILE

    logger.info("Source URL  : %s", DATA_URL)
    logger.info("Destination : %s", LOCAL_FILE)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = downloaded / total_size * 100 if total_size > 0 else 0
        if block_num % 500 == 0:
            logger.info(
                "  Downloaded %.1f MB  (%.0f%%)", downloaded / 1e6, min(pct, 100)
            )

    try:
        urllib.request.urlretrieve(DATA_URL, LOCAL_FILE, reporthook=_progress)
        size_mb = LOCAL_FILE.stat().st_size / 1e6
        logger.info("Download complete: %s  (%.1f MB)", LOCAL_FILE, size_mb)
    except Exception as exc:
        logger.error("Download FAILED: %s", exc)
        raise

    return LOCAL_FILE


# ===========================================================================
# STEP 2: VALIDATE
# ===========================================================================
def step_validate(filepath: Path) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 2 - VALIDATE: Pre-upload integrity checks")
    logger.info("=" * 60)

    report = {}

    # 2a. File existence and size
    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        raise FileNotFoundError(filepath)

    size_bytes = filepath.stat().st_size
    report["size_bytes"] = size_bytes
    logger.info("File exists : %s  size=%.2f MB", filepath, size_bytes / 1e6)
    if size_bytes == 0:
        raise ValueError("File is empty -- aborting.")

    # 2b. Extension check
    ext = filepath.suffix.lower()
    if ext not in EXPECTED_EXT:
        logger.warning("Unexpected extension '%s'. Allowed: %s", ext, EXPECTED_EXT)
    else:
        logger.info("Extension OK: %s", ext)
    report["extension"] = ext

    # 2c. MD5 checksum
    md5 = hashlib.md5(filepath.read_bytes()).hexdigest()
    report["md5"] = md5
    logger.info("MD5 checksum: %s", md5)

    # 2d. Encoding detection (Parquet is binary; encoding=None is expected)
    sample    = filepath.read_bytes()[:1_048_576]
    detected  = chardet.detect(sample)
    encoding  = detected.get("encoding", "unknown")
    confidence = detected.get("confidence", 0)
    report["encoding"]            = encoding
    report["encoding_confidence"] = confidence
    logger.info(
        "Encoding detected : %s  (confidence=%.0f%%)", encoding, confidence * 100
    )
    if encoding is None or encoding.upper() in ("", "NONE"):
        logger.info(
            "Encoding is None -- expected for binary Parquet. [OK]"
        )
    elif encoding.upper() not in ("UTF-8", "ASCII", "UTF-8-SIG"):
        logger.warning(
            "Non-UTF-8 encoding (%s) detected. Verify before text-based ETL.", encoding
        )

    # 2e. Row count
    try:
        if ext == ".parquet":
            import pyarrow.parquet as pq
            meta      = pq.read_metadata(filepath)
            row_count = meta.num_rows
        else:
            row_count = sum(1 for _ in open(filepath, "rb")) - 1
        report["row_count"] = row_count
        logger.info("Row count: %s", f"{row_count:,}")
        if row_count < MIN_ROWS:
            logger.warning(
                "Row count %d is below minimum threshold %d!", row_count, MIN_ROWS
            )
        else:
            logger.info("Row count meets the 500,000+ requirement. [OK]")
    except Exception as exc:
        logger.warning("Could not verify row count: %s", exc)
        report["row_count"] = "unknown"

    logger.info("Validation PASSED for %s", filepath.name)
    return report


# ===========================================================================
# STEP 3: UPLOAD
# ===========================================================================
def step_upload(filepath: Path) -> bool:
    logger.info("=" * 60)
    logger.info("STEP 3 - UPLOAD: Pushing file to HDFS")
    logger.info("=" * 60)
    logger.info("HDFS target : %s", HDFS_TARGET)
    logger.info("HDFS user   : %s", HDFS_USER)

    # --- Attempt 1: hdfs Python library (WebHDFS REST API) -----------------
    try:
        from hdfs import InsecureClient
        client = InsecureClient(WEBHDFS_URL, user=HDFS_USER)
        logger.info("Connecting via WebHDFS at %s ...", WEBHDFS_URL)

        client.makedirs(HDFS_TARGET)
        logger.info("HDFS directory ensured: %s", HDFS_TARGET)

        hdfs_path = f"{HDFS_TARGET}/{filepath.name}"
        logger.info("Uploading %s -> %s", filepath.name, hdfs_path)
        with open(filepath, "rb") as fh:
            client.write(hdfs_path, fh, overwrite=True)

        logger.info("Upload SUCCESS via WebHDFS API: %s", hdfs_path)
        return True

    except Exception as api_exc:
        logger.warning("WebHDFS API upload failed: %s", api_exc)
        logger.warning("Falling back to hadoop shell command ...")

    # --- Attempt 2: hadoop shell -------------------------------------------
    try:
        run_cmd([HADOOP_CMD, "fs", "-mkdir", "-p", HDFS_TARGET])
        run_cmd(
            [HADOOP_CMD, "fs", "-put", "-f",
             str(filepath), f"{HDFS_TARGET}/{filepath.name}"]
        )
        logger.info("Upload SUCCESS via hadoop shell.")
        return True

    except FileNotFoundError:
        logger.error(
            "hadoop command not found in PATH.\n"
            "SETUP GUIDE:\n"
            "  Option A (WSL on Windows):\n"
            "    1. Install WSL2: wsl --install\n"
            "    2. Inside WSL, install Hadoop 3.x\n"
            "    3. Run this script from inside the WSL terminal\n"
            "  Option B (Docker):\n"
            "    1. docker pull apache/hadoop:3\n"
            "    2. Mount this directory into the container\n"
            "    3. Run ingest.py from inside the container\n"
            "  Option C (Native Windows Hadoop):\n"
            "    https://hadoop.apache.org/docs/stable/"
            "hadoop-project-dist/hadoop-common/SingleCluster.html"
        )
        raise RuntimeError(
            "Hadoop not found. See log above for setup instructions."
        )
    except Exception as shell_exc:
        logger.error("hadoop shell upload failed: %s", shell_exc)
        logger.error(
            "Troubleshooting checklist:\n"
            "  1. Is HDFS running?       Run: start-dfs.sh\n"
            "  2. Is WebHDFS enabled?    hdfs-site.xml needs:\n"
            "       <name>dfs.webhdfs.enabled</name><value>true</value>\n"
            "  3. Permission denied?     Run: hadoop fs -chmod 777 /\n"
            "  4. Wrong user?            Set env var: HDFS_USER=<your_hadoop_user>\n"
            "  5. NameNode UI:           http://localhost:9870"
        )
        raise RuntimeError(
            "Upload failed. See troubleshooting log above."
        ) from shell_exc


# ===========================================================================
# STEP 4: ORGANIZE
# ===========================================================================
def step_organize() -> None:
    logger.info("=" * 60)
    logger.info("STEP 4 - ORGANIZE: Verifying HDFS directory structure")
    logger.info("=" * 60)

    try:
        result = run_cmd([HADOOP_CMD, "fs", "-ls", "-R", HDFS_BASE], check=False)
        if result.returncode == 0:
            logger.info("HDFS structure under %s:\n%s", HDFS_BASE, result.stdout)
        else:
            logger.warning("Could not list HDFS (hadoop may not be in PATH).")
    except Exception as exc:
        logger.warning("HDFS listing skipped: %s", exc)

    logger.info("Intended HDFS directory layout:")
    logger.info("  %s/", HDFS_BASE)
    logger.info("    %s/", HDFS_DATASET)
    logger.info("      year=%s/", YEAR)
    logger.info("        month=%s/", MONTH)
    logger.info("          %s", LOCAL_FILE.name)


# ===========================================================================
# STEP 5: LOG SUMMARY
# ===========================================================================
def step_log_summary(validation_report: dict) -> None:
    logger.info("=" * 60)
    logger.info("STEP 5 - LOG SUMMARY")
    logger.info("=" * 60)
    logger.info("Pipeline completed at : %s", datetime.now().isoformat())
    logger.info("Local file            : %s", LOCAL_FILE)
    logger.info("HDFS target path      : %s/%s", HDFS_TARGET, LOCAL_FILE.name)
    logger.info(
        "File size             : %.2f MB",
        validation_report.get("size_bytes", 0) / 1e6,
    )
    logger.info(
        "Encoding              : %s (confidence=%.0f%%)",
        validation_report.get("encoding"),
        validation_report.get("encoding_confidence", 0) * 100,
    )
    row_count = validation_report.get("row_count", "N/A")
    logger.info(
        "Row count             : %s",
        f"{row_count:,}" if isinstance(row_count, int) else row_count,
    )
    logger.info("MD5                   : %s", validation_report.get("md5"))
    logger.info("Full log saved to     : %s", LOG_FILE)


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("BDA A2 - HDFS Ingestion Pipeline (ingest.py)")
    logger.info("Dataset : NYC Yellow Taxi Trip Data 2023-01")
    logger.info("=" * 60)

    try:
        raw_file   = step_load()
        val_report = step_validate(raw_file)
        step_upload(raw_file)
        step_organize()
        step_log_summary(val_report)
        logger.info("[SUCCESS] All pipeline steps completed successfully.")
        sys.exit(0)
    except Exception as exc:
        logger.error("[FAILED] Pipeline error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
