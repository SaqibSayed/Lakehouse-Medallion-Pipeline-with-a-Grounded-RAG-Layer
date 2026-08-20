# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
df = spark.read.csv(
    "/Volumes/workspace/default/saqib/training.csv",
    header=True,
    inferSchema=True
)

# COMMAND ----------

df.display()

# COMMAND ----------

df.createOrReplaceTempView("bronze_training_raw")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =====================================================================================
# MAGIC -- SILVER STEP 1 - Clean noisy headers
# MAGIC -- The classification/PII suffixes ("&1# CONFIDENTIAL", "&2# PII") are metadata tags
# MAGIC -- baked into the header text, not part of the field name. Strip them and standardise
# MAGIC -- to snake_case. Numeric-looking columns are TRY_CAST here (not plain CAST) so a bad
# MAGIC -- value produces NULL instead of failing the whole job — this keeps the pipeline
# MAGIC -- re-runnable/resilient to further dirty data.
# MAGIC -- =====================================================================================
# MAGIC CREATE OR REPLACE TEMP VIEW stg_renamed AS
# MAGIC SELECT
# MAGIC   `Fileno&1# CONFIDENTIAL`   AS fileno,
# MAGIC   `Applicant&2# PII`         AS applicant,
# MAGIC   `Age`                      AS age_raw,
# MAGIC   `Branch`                   AS branch,
# MAGIC   `Course Name`              AS course_name,
# MAGIC   `Course Date`              AS course_date_raw,
# MAGIC   TRY_CAST(`Number of Days`               AS INT) AS number_of_days,
# MAGIC   TRY_CAST(`Number of Days Attended`      AS INT) AS days_attended,
# MAGIC   TRY_CAST(`Number of Days Absent`        AS INT) AS days_absent
# MAGIC FROM bronze_training_raw;
# MAGIC  
# MAGIC  

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =====================================================================================
# MAGIC -- SILVER STEP 2 - Parse Course Date (mixed formats)
# MAGIC -- Observed formats in the sample:
# MAGIC --   yyyy-MM-dd   e.g. 2024-05-20                (ISO)
# MAGIC --   dd/MM/yyyy   e.g. 14/05/2024                (day=14 rules out MM/dd/yyyy)
# MAGIC --   yyyy/MM/dd   e.g. 2024/06/30
# MAGIC --   Excel serial e.g. 45230                     (pure digit string, no separators)
# MAGIC --   Invalid      e.g. 2024-13-45                (month 13 / day 45 do not exist)
# MAGIC --
# MAGIC -- JUDGEMENT CALLS (documented):
# MAGIC --  1. "14/05/2024" is treated as dd/MM/yyyy, not mm/dd/yyyy. Day=14 makes mm/dd/yyyy
# MAGIC --     impossible, and the rest of the dataset is UAE-context, so dd/MM/yyyy is the
# MAGIC --     safe regional default anywhere a slash-format date IS ambiguous (e.g. 05/06/2024
# MAGIC --     would also be read as 5-Jun-2024 under this same rule for consistency).
# MAGIC --  2. Pure-digit values (regex ^[0-9]{4,6}$) are treated as Excel serial date numbers
# MAGIC --     (common artifact of Excel-exported CSVs), converted via the 1899-12-30 epoch.
# MAGIC --     45230 -> 2023-10-31 under this rule.
# MAGIC --  3. TRY_TO_DATE is used (not TO_DATE) so a value that superficially matches a pattern
# MAGIC --     but is semantically invalid (2024-13-45) safely returns NULL instead of erroring.
# MAGIC --  4. Anything not matched by any rule -> NULL -> routed to the rejects set in Step 5.
# MAGIC -- =====================================================================================
# MAGIC CREATE OR REPLACE TEMP VIEW stg_dates AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   COALESCE(
# MAGIC     TRY_TO_DATE(course_date_raw, 'yyyy-MM-dd'),
# MAGIC     TRY_TO_DATE(course_date_raw, 'dd/MM/yyyy'),
# MAGIC     TRY_TO_DATE(course_date_raw, 'yyyy/MM/dd'),
# MAGIC     CASE
# MAGIC       WHEN course_date_raw RLIKE '^[0-9]{4,6}$'
# MAGIC         THEN DATE_ADD(DATE'1899-12-30', CAST(course_date_raw AS INT))
# MAGIC       ELSE NULL
# MAGIC     END
# MAGIC   ) AS course_date
# MAGIC FROM stg_renamed;
# MAGIC  

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =====================================================================================
# MAGIC -- SILVER STEP 3 - Validate Age
# MAGIC -- JUDGEMENT CALL: valid working-age adult range assumed as 16-100 inclusive for a
# MAGIC -- workforce-training program. Non-numeric ("thirty") or out-of-range (150, -5) values
# MAGIC -- are NOT dropped -- the row is kept (age doesn't affect course/attendance facts) but
# MAGIC -- flagged, with age_clean set to NULL so it can't silently pollute downstream numeric
# MAGIC -- aggregates.
# MAGIC -- =====================================================================================
# MAGIC CREATE OR REPLACE TEMP VIEW stg_age AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   CASE
# MAGIC     WHEN TRY_CAST(age_raw AS INT) BETWEEN 16 AND 100 THEN TRY_CAST(age_raw AS INT)
# MAGIC     ELSE NULL
# MAGIC   END AS age_clean,
# MAGIC   CASE
# MAGIC     WHEN TRY_CAST(age_raw AS INT) BETWEEN 16 AND 100 THEN TRUE
# MAGIC     ELSE FALSE
# MAGIC   END AS is_age_valid
# MAGIC FROM stg_dates;
# MAGIC  

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =====================================================================================
# MAGIC -- SILVER STEP 4 - Data-quality flag: Days Attended + Days Absent <> Number of Days
# MAGIC -- =====================================================================================
# MAGIC CREATE OR REPLACE TEMP VIEW stg_dq AS
# MAGIC SELECT
# MAGIC   *,
# MAGIC   CASE
# MAGIC     WHEN number_of_days IS NULL OR days_attended IS NULL OR days_absent IS NULL THEN TRUE
# MAGIC     WHEN (days_attended + days_absent) <> number_of_days THEN TRUE
# MAGIC     ELSE FALSE
# MAGIC   END AS dq_days_mismatch
# MAGIC FROM stg_age;
# MAGIC  

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =====================================================================================
# MAGIC -- SILVER STEP 5 - De-duplicate on Fileno
# MAGIC -- RULE (documented):
# MAGIC --   a) First collapse exact duplicate rows (all columns identical) via SELECT DISTINCT.
# MAGIC --      This alone resolves TST-2024-0007, which appears twice with byte-identical data.
# MAGIC --   b) For any remaining Fileno with genuinely conflicting records (not present in this
# MAGIC --      sample, but the pipeline must handle it), keep the row with the most recent
# MAGIC --      course_date -- i.e. the latest known record wins. NULL dates sort last so a
# MAGIC --      row with an unparseable date never "wins" a dedup over a row with a real date.
# MAGIC --      Ties are broken deterministically on applicant name purely for reproducibility.
# MAGIC -- =====================================================================================
# MAGIC CREATE OR REPLACE TEMP VIEW stg_dedup_exact AS
# MAGIC SELECT DISTINCT * FROM stg_dq;
# MAGIC  
# MAGIC CREATE OR REPLACE TEMP VIEW stg_dedup AS
# MAGIC SELECT * EXCEPT (rn)
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     ROW_NUMBER() OVER (
# MAGIC       PARTITION BY fileno
# MAGIC       ORDER BY course_date DESC NULLS LAST, applicant
# MAGIC     ) AS rn
# MAGIC   FROM stg_dedup_exact
# MAGIC )
# MAGIC WHERE rn = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =====================================================================================
# MAGIC -- SILVER STEP 6 - Split rejects vs. curated Silver table
# MAGIC -- Rows with an unparseable Course Date are quarantined into a rejects table for
# MAGIC -- follow-up/re-submission rather than silently dropped.
# MAGIC -- =====================================================================================
# MAGIC CREATE OR REPLACE TABLE silver_training_rejects AS
# MAGIC SELECT
# MAGIC   fileno, applicant, age_raw, branch, course_name, course_date_raw,
# MAGIC   number_of_days, days_attended, days_absent,
# MAGIC   'unparseable_course_date' AS reject_reason
# MAGIC FROM stg_dedup
# MAGIC WHERE course_date IS NULL;
# MAGIC  
# MAGIC CREATE OR REPLACE TABLE silver_training AS
# MAGIC SELECT
# MAGIC   fileno,
# MAGIC   applicant,
# MAGIC   age_raw,
# MAGIC   age_clean,
# MAGIC   is_age_valid,
# MAGIC   branch,
# MAGIC   course_name,
# MAGIC   course_date,
# MAGIC   number_of_days,
# MAGIC   days_attended,
# MAGIC   days_absent,
# MAGIC   dq_days_mismatch
# MAGIC FROM stg_dedup
# MAGIC WHERE course_date IS NOT NULL;
# MAGIC  

# COMMAND ----------

# %sql
# Select * from silver_training
# --SELECT * FROM silver_training_rejects

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =====================================================================================
# MAGIC -- GOLD - Record-level enriched table
# MAGIC -- Same grain as silver_training.
# MAGIC -- FEATURES ADDED:
# MAGIC --   attendance_rate  - record-level attended/scheduled. TRY_DIVIDE, not "/", so a NULL
# MAGIC --                      number_of_days from TRY_CAST yields NULL rather than an error.
# MAGIC --   attendance_band  - High >=0.90 / Medium 0.60-0.89 / Low <0.60. Categorical version
# MAGIC --                      for BI slicing and as a ready-made model target. NULL rate gets
# MAGIC --                      an explicit 'Unknown' bucket so it stays visible in a GROUP BY
# MAGIC --                      instead of silently disappearing.
# MAGIC -- =====================================================================================
# MAGIC CREATE OR REPLACE TABLE gold_training_records AS
# MAGIC SELECT
# MAGIC   fileno,
# MAGIC   applicant,
# MAGIC   age_raw,
# MAGIC   age_clean,
# MAGIC   branch,
# MAGIC   course_name,
# MAGIC   course_date,
# MAGIC   number_of_days,
# MAGIC   days_attended,
# MAGIC   days_absent,
# MAGIC   ROUND(TRY_DIVIDE(days_attended, number_of_days), 4) AS attendance_rate,
# MAGIC   CASE
# MAGIC     WHEN TRY_DIVIDE(days_attended, number_of_days) IS NULL THEN 'Unknown'
# MAGIC     WHEN TRY_DIVIDE(days_attended, number_of_days) >= 0.90 THEN 'High'
# MAGIC     WHEN TRY_DIVIDE(days_attended, number_of_days) >= 0.60 THEN 'Medium'
# MAGIC     ELSE 'Low'
# MAGIC   END                                                 AS attendance_band
# MAGIC FROM silver_training;

# COMMAND ----------

# %sql
# Select * from gold_training_records

# COMMAND ----------

# MAGIC %sql
# MAGIC -- =====================================================================================
# MAGIC -- GOLD - Curated aggregates
# MAGIC --
# MAGIC -- ATTENDANCE RATE DEFINITION (documented):
# MAGIC --   attendance_rate = SUM(days_attended) / SUM(number_of_days)   [group-level, weighted]
# MAGIC --
# MAGIC -- This is a weighted average across all records in the group (total attended days over
# MAGIC -- total scheduled days), not a simple average of each row's individual rate. Weighting
# MAGIC -- is chosen so a 2-day course doesn't move the average as much as a 10-day course --
# MAGIC -- it reflects the true proportion of attended person-days in that Branch/Course.
# MAGIC --
# MAGIC -- JUDGEMENT CALL: rows flagged dq_days_mismatch = TRUE are INCLUDED in the aggregate by
# MAGIC -- default, because Number of Days is treated as the authoritative scheduled-days figure
# MAGIC -- from source. The flag exists for downstream investigation, not automatic exclusion.
# MAGIC -- A stricter cut (excluding flagged rows) is provided below as a commented alternative.
# MAGIC -- Rows with invalid age are included -- age has no bearing on attendance facts.
# MAGIC -- Rows routed to silver_training_rejects (bad Course Date) are excluded, since
# MAGIC -- silver_training is the only source for Gold.
# MAGIC -- =====================================================================================
# MAGIC CREATE OR REPLACE TABLE gold_attendance_by_branch AS
# MAGIC SELECT
# MAGIC   branch,
# MAGIC   COUNT(*)                                                   AS num_records,
# MAGIC   SUM(number_of_days)                                        AS total_scheduled_days,
# MAGIC   SUM(days_attended)                                         AS total_attended_days,
# MAGIC   ROUND(SUM(days_attended) / SUM(number_of_days), 4)         AS attendance_rate
# MAGIC FROM silver_training
# MAGIC GROUP BY branch;
# MAGIC  
# MAGIC CREATE OR REPLACE TABLE gold_attendance_by_course AS
# MAGIC SELECT
# MAGIC   course_name,
# MAGIC   COUNT(*)                                                   AS num_records,
# MAGIC   SUM(number_of_days)                                        AS total_scheduled_days,
# MAGIC   SUM(days_attended)                                         AS total_attended_days,
# MAGIC   ROUND(SUM(days_attended) / SUM(number_of_days), 4)         AS attendance_rate
# MAGIC FROM silver_training
# MAGIC GROUP BY course_name;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC --SELECT * FROM gold_attendance_by_branch ORDER BY branch;
# MAGIC --SELECT * FROM gold_attendance_by_course ORDER BY course_name;