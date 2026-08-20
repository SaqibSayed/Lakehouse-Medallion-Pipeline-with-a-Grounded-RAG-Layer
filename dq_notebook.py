# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Data quality, performance & governance
# MAGIC
# MAGIC **How to run this:** the data-quality section reads the temp views created in the
# MAGIC Part 1 notebook (`bronze_training_raw`, `stg_dedup_exact`, `stg_dedup`) plus the
# MAGIC persisted tables (`silver_training`, `silver_training_rejects`). Temp views are
# MAGIC session-scoped, so either:
# MAGIC
# MAGIC - paste the cells below at the **end of your silver_gold_layer notebook**, or
# MAGIC - run silver_gold_layer first and attach this notebook to the **same cluster/session**, or
# MAGIC - run silver_gold_layer via `%run ./silver_gold_layer` from the cell below.
# MAGIC
# MAGIC The report is deliberately built *inside* the pipeline rather than re-derived
# MAGIC afterwards — a DQ report computed from a separate re-read of the source can drift
# MAGIC from what the pipeline actually did.

# COMMAND ----------

# MAGIC %run ./silver_gold_layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data-quality report
# MAGIC
# MAGIC Rules are lifted directly from the silver_gold_layer transformations, not invented for the
# MAGIC report. Each rule declares three things:
# MAGIC
# MAGIC | Field | Meaning |
# MAGIC |---|---|
# MAGIC | `applicable` | which rows the rule can even be judged on (a reconciliation rule is meaningless where a day count failed to cast) |
# MAGIC | `passes` | the pass condition |
# MAGIC | `severity` | `blocking` = row is quarantined to `silver_training_rejects`; `warn` = row is kept and flagged |
# MAGIC
# MAGIC Separating *applicable* from *passes* matters: without it, rows that fail the cast
# MAGIC rule get double-counted as reconciliation failures too, and the failure totals stop
# MAGIC summing to anything meaningful.

# COMMAND ----------

from pyspark.sql import functions as F, types as T
from datetime import datetime
import uuid

# --- configure to your workspace -----------------------------------------------------
CATALOG = "workspace"
SCHEMA = "default"
DQ_TABLE = f"{CATALOG}.{SCHEMA}.dq_run_log"
DATASET = f"{CATALOG}.{SCHEMA}.silver_training"
# -------------------------------------------------------------------------------------

RUN_ID = str(uuid.uuid4())[:8]
RUN_TS = datetime.utcnow()

# Rule catalogue — mirrors Part 1 Silver steps 1-4
RULES = {
    # Silver Step 1 — TRY_CAST on the three day-count columns
    "days_numeric_cast": {
        "applicable": "1=1",
        "passes": "number_of_days IS NOT NULL AND days_attended IS NOT NULL AND days_absent IS NOT NULL",
        "severity": "warn",
        "note": "TRY_CAST of Number of Days / Attended / Absent to INT",
    },
    # Silver Step 2 — multi-format course date parse
    "course_date_parseable": {
        "applicable": "1=1",
        "passes": "course_date IS NOT NULL",
        "severity": "blocking",
        "note": "Parses ISO / dd-MM-yyyy / yyyy-MM-dd / Excel serial; failures quarantined",
    },
    # Silver Step 3 — working-age range
    "age_in_range_16_100": {
        "applicable": "age_raw IS NOT NULL AND TRIM(age_raw) <> ''",
        "passes": "is_age_valid",
        "severity": "warn",
        "note": "Non-numeric or out-of-range age nulled, row retained",
    },
    # Silver Step 4 — attended + absent = scheduled
    "days_reconcile": {
        "applicable": "number_of_days IS NOT NULL AND days_attended IS NOT NULL AND days_absent IS NOT NULL",
        "passes": "(days_attended + days_absent) = number_of_days",
        "severity": "warn",
        "note": "Day-count reconciliation; flagged via dq_days_mismatch",
    },
}

# COMMAND ----------

# Pipeline-level counts (dedup happens before row-level rules can be evaluated)
rows_in = spark.table("bronze_training_raw").count()
rows_after_exact_dedup = spark.table("stg_dedup_exact").count()
rows_after_fileno_dedup = spark.table("stg_dedup").count()
rows_curated = spark.table("silver_training").count()
rows_quarantined = spark.table("silver_training_rejects").count()

deduped = spark.table("stg_dedup")

# Evaluate every row-level rule in a single pass over the data
agg_exprs = []
for name, r in RULES.items():
    agg_exprs.append(F.sum(F.expr(f"CASE WHEN {r['applicable']} THEN 1 ELSE 0 END")).alias(f"{name}__evaluated"))
    agg_exprs.append(
        F.sum(F.expr(f"CASE WHEN ({r['applicable']}) AND ({r['passes']}) THEN 1 ELSE 0 END")).alias(f"{name}__passed")
    )

all_pass_expr = " AND ".join(f"(NOT ({r['applicable']}) OR ({r['passes']}))" for r in RULES.values())
agg_exprs.append(F.sum(F.expr(f"CASE WHEN {all_pass_expr} THEN 1 ELSE 0 END")).alias("__rows_clean"))

counts = deduped.agg(*agg_exprs).collect()[0].asDict()

# COMMAND ----------

# Assemble the report in long format — one row per rule, ready for monitoring
report_rows = []

# Dedup rules first (evaluated at pipeline level, not row level)
report_rows.append((
    RUN_ID, RUN_TS, DATASET, "exact_duplicate_rows", "warn",
    rows_in, rows_in, rows_after_exact_dedup, rows_in - rows_after_exact_dedup,
    "SELECT DISTINCT collapse of byte-identical rows",
))
report_rows.append((
    RUN_ID, RUN_TS, DATASET, "duplicate_fileno_conflict", "warn",
    rows_in, rows_after_exact_dedup, rows_after_fileno_dedup,
    rows_after_exact_dedup - rows_after_fileno_dedup,
    "Conflicting records per Fileno; latest course_date wins",
))

# Row-level rules
for name, r in RULES.items():
    evaluated = counts[f"{name}__evaluated"] or 0
    passed = counts[f"{name}__passed"] or 0
    report_rows.append((
        RUN_ID, RUN_TS, DATASET, name, r["severity"],
        rows_in, evaluated, passed, evaluated - passed, r["note"],
    ))

schema = T.StructType([
    T.StructField("run_id", T.StringType()),
    T.StructField("run_ts", T.TimestampType()),
    T.StructField("dataset", T.StringType()),
    T.StructField("rule_name", T.StringType()),
    T.StructField("severity", T.StringType()),
    T.StructField("rows_in", T.LongType()),
    T.StructField("rows_evaluated", T.LongType()),
    T.StructField("rows_passed", T.LongType()),
    T.StructField("rows_failed", T.LongType()),
    T.StructField("rule_note", T.StringType()),
])

dq_report = (
    spark.createDataFrame(report_rows, schema)
    .withColumn(
        "pass_rate",
        F.when(F.col("rows_evaluated") > 0,
               F.round(F.col("rows_passed") / F.col("rows_evaluated"), 4)).otherwise(F.lit(None))
    )
)

display(dq_report.orderBy("severity", "rule_name"))

# COMMAND ----------

# Persist — append-only so the table becomes a time series, not a snapshot.
# This is what a Databricks SQL alert or Lakehouse Monitoring dashboard sits on.
(dq_report.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(DQ_TABLE))

print(f"DQ run {RUN_ID} written to {DQ_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitoring query + alert
# MAGIC Point a Databricks SQL alert at the query below on a schedule. It fires when any
# MAGIC blocking rule regresses or a warn rule drops below 90% pass rate.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   run_ts, rule_name, severity, rows_evaluated, rows_failed, pass_rate
# MAGIC FROM workspace.default.dq_run_log
# MAGIC WHERE run_ts >= CURRENT_TIMESTAMP() - INTERVAL 1 DAY
# MAGIC   AND (
# MAGIC        (severity = 'blocking' AND rows_failed > 0)
# MAGIC     OR (severity = 'warn'     AND pass_rate < 0.90)
# MAGIC   )
# MAGIC ORDER BY severity, pass_rate;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Production alternative — DLT expectations
# MAGIC
# MAGIC The same rules expressed as Delta Live Tables expectations push pass/fail counts to
# MAGIC the DLT event log automatically, with no hand-rolled aggregation. Worth stating as
# MAGIC the target-state design:
# MAGIC
# MAGIC ```python
# MAGIC @dlt.table(name="silver_training")
# MAGIC @dlt.expect_or_drop("course_date_parseable", "course_date IS NOT NULL")
# MAGIC @dlt.expect("days_reconcile", "(days_attended + days_absent) = number_of_days")
# MAGIC @dlt.expect("age_in_range_16_100", "is_age_valid")
# MAGIC def silver_training():
# MAGIC     return dlt.read("stg_dedup")
# MAGIC ```
# MAGIC
# MAGIC `expect_or_drop` reproduces the quarantine behaviour; `expect` reproduces the
# MAGIC flag-and-retain behaviour. Metrics land in `event_log(...)` under `flow_progress`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Delta Lake / Spark optimisations
# MAGIC
# MAGIC ### (a) Liquid clustering on `silver_training`
# MAGIC Both Gold aggregates group by `branch` and `course_name`, and any realistic query
# MAGIC filters on `course_date`. Hive partitioning on `course_date` would be the wrong
# MAGIC instinct here — training data is sparse per day, so it would produce one tiny file
# MAGIC per date. Liquid clustering handles the high-cardinality key without that skew, and
# MAGIC the clustering key can be changed later without rewriting the table.
# MAGIC
# MAGIC ### (b) `optimizeWrite` + `autoCompact` + scheduled `OPTIMIZE`
# MAGIC Part 1 uses `CREATE OR REPLACE TABLE` on every run, which rewrites the whole table
# MAGIC and leaves a fresh set of small files each time. At scale that becomes a small-file
# MAGIC problem the Gold aggregation pays for on every scan. Auto-compaction plus a
# MAGIC scheduled `OPTIMIZE` keeps files near the 128MB–1GB target.
# MAGIC
# MAGIC ### (c) AQE shuffle-partition coalescing
# MAGIC The Gold `GROUP BY branch` / `GROUP BY course_name` triggers a shuffle. Without AQE,
# MAGIC Spark plans `spark.sql.shuffle.partitions` (default 200) tasks regardless of data
# MAGIC size — on this dataset that is 200 tasks over a handful of rows, almost entirely
# MAGIC scheduling overhead. AQE coalesces post-shuffle partitions using runtime statistics.
# MAGIC It is on by default in Databricks Runtime; the point for the assessment is knowing
# MAGIC *why* it matters here. AQE also auto-broadcasts the small side of a join — relevant
# MAGIC as soon as Silver is joined to a branch or course dimension.
# MAGIC
# MAGIC ### Also worth changing in Part 1
# MAGIC `inferSchema=True` on the CSV read costs a full extra pass over the file. Declare an
# MAGIC explicit `StructType` instead — it removes the pass and stops schema drift between
# MAGIC runs, which matters more than the speed.

# COMMAND ----------

# Apply (a) and (b)
spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.silver_training CLUSTER BY (branch, course_date)")

spark.sql(f"""
ALTER TABLE {CATALOG}.{SCHEMA}.silver_training SET TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
)
""")

spark.sql(f"OPTIMIZE {CATALOG}.{SCHEMA}.silver_training")

# Confirm AQE is active (c)
print("AQE enabled              :", spark.conf.get("spark.sql.adaptive.enabled"))
print("Coalesce partitions      :", spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled"))
print("Skew join handling       :", spark.conf.get("spark.sql.adaptive.skewJoin.enabled"))
print("Auto-broadcast threshold :", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Unity Catalog governance
# MAGIC
# MAGIC ### Current state → target state
# MAGIC Part 1 creates `silver_training`, `gold_attendance_by_branch` etc. **unqualified**,
# MAGIC so they land in whatever schema the session defaults to. First fix is explicit
# MAGIC three-level naming with a schema per medallion layer. The source CSV is already on a
# MAGIC UC Volume (`/Volumes/workspace/default/saqib/`), so raw file access is governed —
# MAGIC that part is right.
# MAGIC
# MAGIC ### Permissions — grant to groups, never individuals
# MAGIC | Group | Bronze | Silver | Gold | governance |
# MAGIC |---|---|---|---|---|
# MAGIC | `data_engineers` | MODIFY | MODIFY | MODIFY | MODIFY |
# MAGIC | `data_analysts` | — | SELECT | SELECT | SELECT |
# MAGIC | `business_users` | — | — | SELECT | — |
# MAGIC
# MAGIC Business users never touch Silver: it holds `applicant`, which the source header
# MAGIC tagged as PII.
# MAGIC
# MAGIC ### Ownership
# MAGIC Set the owner to a **group**, not the person who ran the notebook. Personal ownership
# MAGIC orphans the objects when someone changes team.
# MAGIC
# MAGIC ### Lineage
# MAGIC Captured automatically for anything read or written through UC — Volume → bronze →
# MAGIC silver → gold → vector index, at table *and* column level, with no instrumentation.
# MAGIC The practical value here: `applicant` is tagged PII once at Silver, and lineage shows
# MAGIC every downstream object that inherits it.
# MAGIC
# MAGIC
# MAGIC ### RAG guardrail
# MAGIC **Refusal on low retrieval confidence.** If the top cosine score falls below
# MAGIC `SIMILARITY_THRESHOLD`, the chain returns an explicit "not supported by the data"
# MAGIC response and never calls the LLM — so weak context cannot become a confident wrong
# MAGIC answer. The threshold is set from the calibration harness (measured separation
# MAGIC between known-answerable and known-unanswerable questions), not guessed.
# MAGIC
# MAGIC Second-order guardrail worth naming: because a Vector Search index is a UC object, it
# MAGIC inherits the source table's grants — a user who cannot `SELECT silver_training`
# MAGIC cannot retrieve its chunks either. Retrieval is permission-aware by construction
# MAGIC rather than by prompt instruction.

# COMMAND ----------

# Governance DDL
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.governance")

# --- Permissions (group names are illustrative — substitute your workspace groups) ---
GOVERNANCE_DDL = f"""
GRANT USE CATALOG ON CATALOG {CATALOG} TO `data_analysts`;
GRANT USE SCHEMA, SELECT ON SCHEMA {CATALOG}.gold   TO `business_users`;
GRANT USE SCHEMA, SELECT ON SCHEMA {CATALOG}.silver TO `data_analysts`;
GRANT USE SCHEMA, MODIFY, SELECT ON SCHEMA {CATALOG}.silver TO `data_engineers`;
GRANT READ VOLUME ON VOLUME {CATALOG}.{SCHEMA}.saqib TO `data_engineers`;

ALTER SCHEMA {CATALOG}.silver OWNER TO `data_engineers`;
ALTER TABLE {CATALOG}.silver.silver_training OWNER TO `data_engineers`;
"""
print(GOVERNANCE_DDL)  # review, then execute against your workspace groups

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Column mask: enforce PII redaction at the table, not in notebook code
# MAGIC CREATE OR REPLACE FUNCTION workspace.governance.mask_applicant(name STRING)
# MAGIC RETURN CASE
# MAGIC   WHEN is_account_group_member('pii_readers') THEN name
# MAGIC   ELSE '***REDACTED***'
# MAGIC END;
# MAGIC
# MAGIC ALTER TABLE workspace.silver.silver_training
# MAGIC   ALTER COLUMN applicant SET MASK workspace.governance.mask_applicant;
# MAGIC
# MAGIC -- Classification tags — drive discovery and downstream policy
# MAGIC ALTER TABLE workspace.silver.silver_training
# MAGIC   ALTER COLUMN applicant SET TAGS ('classification' = 'PII');
# MAGIC ALTER TABLE workspace.silver.silver_training
# MAGIC   SET TAGS ('data_domain' = 'training', 'medallion_layer' = 'silver');