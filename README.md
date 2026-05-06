# NYC Yellow Taxi Data Warehouse Pipeline
### BDA Group Project — All Three Milestones

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.3-orange)](https://spark.apache.org)
[![Hadoop](https://img.shields.io/badge/Hadoop-3.x-yellow)](https://hadoop.apache.org)
[![License](https://img.shields.io/badge/License-Academic-green)](.)

> End-to-end big data warehouse system: raw ingestion → HDFS → ETL → dimensional
> schema → Spark SQL analytics → business insights.

---

## Group Members

| Name | Student ID |
|------|-----------|
| Muhammad Zuraiz | 481961 |
| Muhammad Shahzaib | 478038 |

---

## Project Architecture

```
Raw Data (NYC TLC CDN)
        │
        ▼
   ingest.py (A2)
   ├── Download Parquet from TLC CloudFront
   ├── Validate: size, MD5, encoding, row count
   ├── Upload to HDFS /warehouse/raw/
   └── Log all steps
        │
        ▼
HDFS /warehouse/raw/nyc_yellow_taxi/year=2023/month=01/
   yellow_tripdata_2023-01.parquet  (47.6 MB, 3,066,766 rows)
        │
        ▼
   etl.py (A3)
   ├── Transform: 9 cleaning steps (all linked to A2 findings)
   ├── Load: star schema → /warehouse/processed/
   │     ├── fact_trips/       (partitioned by pickup_month)
   │     ├── dim_time/
   │     ├── dim_location/
   │     ├── dim_payment/
   │     └── dim_vendor/
   └── Validate: row counts + null checks
        │
        ▼
   analytics.py (A4)
   ├── 8 Spark SQL queries (window functions, CTEs, subqueries)
   ├── 7 analytical charts
   └── Business insights + optimization documentation
        │
        ▼
   final_report.pdf + presentation slides
```

---

## Dataset

| Property | Detail |
|----------|--------|
| Name     | NYC Yellow Taxi Trip Data — January 2023 |
| Source   | NYC Taxi & Limousine Commission (TLC) |
| URL      | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |
| Format   | Apache Parquet (columnar, Snappy compressed) |
| Size     | 47.6 MB compressed / ~500 MB uncompressed |
| Rows     | 3,066,766 raw → 3,018,034 after ETL |
| Columns  | 19 original + 10 derived = 29 total |

---

## Repository Structure

```
.
├── ingest.py                   # A2 — HDFS ingestion pipeline
├── profiling_report.py         # A2 — data profiling report generator
├── profiling_report.pdf        # A2 — submitted profiling report
├── etl.py                      # A3 — PySpark ETL pipeline
├── analytics.py                # A4 — Spark SQL analytics & visualizations
├── final_report.pdf            # A4 — comprehensive final report (all milestones)
├── presentation_slides.html    # A4 — presentation slides
├── requirements.txt            # all Python dependencies
├── README.md                   # this file
├── raw_data/                   # local download (auto-created by ingest.py)
├── processed/                  # local Spark output (mirrors HDFS)
│   ├── fact_trips/
│   ├── dim_time/
│   ├── dim_location/
│   ├── dim_payment/
│   └── dim_vendor/
├── analytics_output/           # charts, CSV results, query plan
│   ├── chart1_fare_trend.png
│   ├── chart2_tip_by_payment.png
│   ├── chart3_heatmap_hour_day.png
│   ├── chart4_dashboard.png
│   ├── chart5_revenue_trend.png
│   ├── chart6_passenger_analysis.png
│   ├── chart7_vendor_comparison.png
│   ├── query_plan_q8.txt
│   └── q1–q8_results.csv
└── logs/                       # timestamped run logs
```

---

## Setup & Installation

### Prerequisites

| Component | Version  | Purpose |
|-----------|----------|---------|
| Python    | ≥ 3.10   | All scripts |
| Java      | 8 or 17  | Hadoop + Spark JVM |
| Hadoop    | ≥ 3.3    | HDFS storage |
| PySpark   | 3.5.x    | ETL + analytics (not 4.x — requires Java 17) |

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Start Hadoop HDFS

```cmd
# Windows
C:\hadoop\sbin\start-dfs.cmd
hdfs dfsadmin -report

# Linux/Mac
start-dfs.sh
```

---

## Running the Pipeline

### Step 1 — A2: Ingest raw data into HDFS

```bash
python ingest.py
```

Downloads `yellow_tripdata_2023-01.parquet`, validates it, uploads to HDFS,
and logs everything. Takes ~60 seconds on a standard connection.

**Verify:**
```bash
hdfs dfs -ls -R /warehouse/raw/nyc_yellow_taxi
```

### Step 2 — A2: Generate profiling report

```bash
python profiling_report.py
# Output: profiling_report.pdf
```

### Step 3 — A3: Run ETL pipeline

```bash
python etl.py
# Output: processed/ (local) + /warehouse/processed/ (HDFS)
```

Upload local processed tables to HDFS:
```bash
hdfs dfs -mkdir -p /warehouse/processed
hdfs dfs -put processed/fact_trips    /warehouse/processed/fact_trips
hdfs dfs -put processed/dim_time      /warehouse/processed/dim_time
hdfs dfs -put processed/dim_location  /warehouse/processed/dim_location
hdfs dfs -put processed/dim_payment   /warehouse/processed/dim_payment
hdfs dfs -put processed/dim_vendor    /warehouse/processed/dim_vendor
```

### Step 4 — A4: Run analytics

```bash
python analytics.py
# Output: analytics_output/ (charts + CSVs + query plan)
```

---

## ETL Transformation Summary (A3 — linked to A2 findings)

| A2 Finding | Count | ETL Action |
|------------|-------|-----------|
| Negative fare_amount | 25,049 (0.82%) | Replace with median $12.80 |
| Zero trip_distance | 45,862 (1.50%) | Filter out |
| Outliers in total_amount | 371,616 (12.12%) | Winsorise Q3+3×IQR |
| Missing passenger_count | 71,743 (2.34%) | Impute mode=1 |
| store_and_fwd_flag nulls | 71,743 (2.34%) | Cast to Boolean |
| Missing surcharge cols | 71,743 (2.34%) | Fill 0.0 |
| Duplicate rows | 0 (0.00%) | dropDuplicates() |

**Result:** 3,066,766 raw → **3,018,034 clean** (48,732 removed, 1.59%)

---

## Spark SQL Queries (A4 — analytics.py)

| Query | Business Question | Window Function Used |
|-------|-------------------|---------------------|
| Q1 | Revenue by hour & day | `RANK() OVER (PARTITION BY day ORDER BY avg_total DESC)` |
| Q2 | Top zones by time of day | `ROW_NUMBER() OVER (PARTITION BY time_of_day ORDER BY count DESC)` |
| Q3 | Payment vs tip behaviour | `SUM(COUNT(*)) OVER ()` |
| Q4 | Daily fare trend | `LAG(avg_fare) OVER (ORDER BY pickup_date)` |
| Q5 | Fraud/anomaly detection | `RANK() OVER (ORDER BY disputed DESC)` |
| Q6 | Passenger group revenue | `DENSE_RANK() OVER (ORDER BY avg_fare DESC)` + CTE |
| Q7 | Vendor performance | `DENSE_RANK() OVER (PARTITION BY time_of_day ORDER BY avg_fare DESC)` |
| Q8 | Cumulative revenue + 7-day MA | `SUM() OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` + `AVG() OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` |

---

## Optimization Techniques

| Technique | Where | Impact |
|-----------|-------|--------|
| Partitioning | `fact_trips` by `pickup_month` | ~12x I/O reduction for single-month queries |
| Caching | `clean_df.cache()` + query results | 40-60% reduction in job time |
| Broadcast joins | dim tables (<10 KB each) | Eliminates shuffle Exchange nodes |
| Memory tuning | `local[2]`, 4g driver, G1GC | Eliminates OOM on 3M-row dataset |
| Column pruning | FileScan reads 4 of 29 columns | Confirmed in physical plan |
| Adaptive execution | `spark.sql.adaptive.enabled=true` | Auto-coalesces shuffle partitions |

---

## Business Insights Summary

1. **Revenue peaks 18:00-21:00 weekdays** — apply dynamic surge pricing
2. **Midtown Manhattan zones** dominate all time-of-day buckets — pre-position fleet there
3. **Card payment generates 18-20% tip** vs 0% recorded for cash — incentivise card use
4. **City hops (2-8 miles) are most efficient** per revenue/mile — optimise matching
5. **Night+Cash has highest dispute rate (~4-5%)** — flag for audit
6. **Solo riders = 55%+ of revenue** — target subscription pricing for commuters
7. **Vendor 2 outperforms Vendor 1** on dispute rates — share KPIs with dispatch partners
8. **Cumulative revenue grows linearly** with slight upward trend in second half of January