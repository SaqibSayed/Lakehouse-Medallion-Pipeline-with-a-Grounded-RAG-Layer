# Lakehouse — Medallion Pipeline with a Grounded RAG Layer

An end-to-end Databricks pipeline that ingests a deliberately messy training-attendance
CSV, curates it through Bronze → Silver → Gold, emits a data-quality report suitable for
monitoring, and serves a retrieval-augmented Q&A layer over the Gold table.

---

## Contents

| File | What it does |
|---|---|
| `silver_gold_layer.py` | Bronze ingest → Silver curation → Gold tables (record-level + two aggregates) |
| `dq_notebook.py` | Data-quality report, Delta/Spark optimisations, Unity Catalog governance |
| `gold_rag_layer.py` | Embedding index and grounded Q&A over `gold_training_records` |

Run them in that order. `dq_notebook.py` depends on temp views created by
`silver_gold_layer.py`; `gold_rag_layer.py` depends on the Gold table it writes.

---

## Data model

```
training.csv  (UC Volume)
      │
      ▼
bronze_training_raw            temp view, raw as-read
      │
      ▼
stg_renamed → stg_dates → stg_age → stg_dq → stg_dedup      Silver staging chain
      │
      ├──────────────► silver_training_rejects    unparseable course dates
      │
      ▼
silver_training                curated, one row per Fileno
      │
      ├──────────────► gold_training_records      record grain + engineered features
      ├──────────────► gold_attendance_by_branch  weighted aggregate
      └──────────────► gold_attendance_by_course  weighted aggregate
                              │
                              ▼
                       gold_rag_layer.py          documents → embeddings → Q&A
```

### Source data problems handled

| Problem | Example | Treatment |
|---|---|---|
| Classification tags baked into headers | `Fileno&1# CONFIDENTIAL` | Stripped, renamed to snake_case |
| Mixed date formats | `2024-05-20`, `14/05/2024`, `2024/06/30`, `45230` | Four parse rules incl. Excel serial epoch |
| Invalid dates | `2024-13-45` | `TRY_TO_DATE` → NULL → quarantined |
| Non-numeric / impossible ages | `thirty`, `150`, `-5` | Nulled and flagged, row retained |
| Day counts that don't reconcile | attended + absent ≠ scheduled | Flagged via `dq_days_mismatch` |
| Duplicate Fileno | `TST-2024-0007` twice | Exact-duplicate collapse, then latest-date-wins |
| Non-castable numerics | text in a day-count column | `TRY_CAST` → NULL, job does not fail |

Judgement calls are documented inline in the notebooks rather than here, so they stay
next to the code that implements them.

---

## Setup — Databricks Free Edition

**Use Free Edition, not the legacy Community Edition.** These notebooks rely on Unity
Catalog and UC Volumes. Community Edition has neither. Free Edition ships with Unity
Catalog preconfigured, a default catalog named `workspace`, serverless compute, and DBFS
disabled — which is why the source file lives in a Volume rather than `/FileStore`.

### 1. Create an account

Go to <https://www.databricks.com/learn/free-edition> and sign up with a personal email
address. A work address tied to an existing Databricks account may route you to a trial
of the paid product instead. No credit card is required. Confirm the verification email
and the workspace provisions in a minute or two.

### 2. Create a Volume for the source file

In the left nav: **Catalog** → `workspace` catalog → `default` schema → **Create** →
**Volume**. Name it `saqib` (or anything, as long as you update the path in Step 4).

Then open the volume and **Upload** `training.csv` into it. The resulting path is:

```
/Volumes/workspace/default/saqib/training.csv
```

### 3. Import the notebooks

**Workspace** → your user folder → **⋮** → **Import** → **File**, and select all three
`.py` files. They carry `# Databricks notebook source` headers, so they import as
notebooks rather than plain scripts, with cell boundaries preserved.

Keep all three in the **same folder** — `dq_notebook.py` calls `%run ./silver_gold_layer`,
which resolves relative to its own location.

### 4. Point the notebooks at your paths

| Notebook | Line to check | Default |
|---|---|---|
| `silver_gold_layer.py` | `spark.read.csv(...)` | `/Volumes/workspace/default/saqib/training.csv` |
| `dq_notebook.py` | `CATALOG` / `SCHEMA` | `workspace` / `default` |

If you named the volume something else, change the CSV path. If you are on a paid
workspace with a different catalog, change `CATALOG`.

### 5. Attach compute and run

Attach each notebook to **Serverless** compute (Free Edition default) and use
**Run all**. Expected order and runtime:

1. `silver_gold_layer.py` — under a minute
2. `dq_notebook.py` — under a minute
3. `gold_rag_layer.py` — 2–4 minutes on first run; `%pip install sentence-transformers
   faiss-cpu` and the model download dominate. `dbutils.library.restartPython()` in the
   second cell is required after the pip install, not optional.

---

## What each notebook produces

### `silver_gold_layer.py`

Tables written: `silver_training`, `silver_training_rejects`, `gold_training_records`,
`gold_attendance_by_branch`, `gold_attendance_by_course`.

`gold_training_records` is record grain, same as Silver, with the pipeline-internal
quality flags (`is_age_valid`, `dq_days_mismatch`) dropped and two features added:

- `attendance_rate` — `TRY_DIVIDE(days_attended, number_of_days)`, so a NULL day count
  yields NULL rather than an error
- `attendance_band` — High ≥ 0.90 / Medium 0.60–0.89 / Low < 0.60, with an explicit
  `Unknown` bucket so NULL-rate rows stay visible in a `GROUP BY` instead of vanishing

The aggregate tables use a **weighted** rate — `SUM(attended) / SUM(scheduled)` — not the
mean of per-row rates, so a 2-day course does not swing the average as hard as a 10-day
one. The two definitions differ materially on this dataset; see the SQL answer below.

### `dq_notebook.py`

Emits one row per rule — `rows_in`, `rows_evaluated`, `rows_passed`, `rows_failed`,
`pass_rate` — appended to `workspace.default.dq_run_log` so the table accumulates as a
time series a SQL alert can sit on.

Each rule declares `applicable` separately from `passes`. Without that split, a row whose
day count failed to cast is counted as a reconciliation failure too, and the totals stop
summing to anything explainable.

### `gold_rag_layer.py`

Builds one document per attendance record, embeds with `all-MiniLM-L6-v2`, indexes in
FAISS (normalised vectors + `IndexFlatIP` = cosine similarity), and answers questions
with a deterministic composer that reads figures from structured metadata rather than
re-reading them out of the embedded prose.

The applicant name is deliberately excluded from the document text. `Fileno` is the
record reference, so a name cannot surface through a similarity match or an LLM
completion.

---

## Assessment answers

### Q1. Average attendance rate per branch, lowest to highest

Reading it straight from the pre-aggregated table, which computes the same figure:

```sql
SELECT branch, attendance_rate, num_records
FROM gold_attendance_by_branch
ORDER BY attendance_rate ASC;
```

**Simple mean** — the unweighted average of each record's own rate:

```sql
SELECT
  branch,
  ROUND(AVG(attendance_rate), 4) AS avg_attendance_rate,
  COUNT(*)                       AS num_records
FROM gold_training_records
GROUP BY branch
ORDER BY avg_attendance_rate ASC;
```

The two disagree on this data. Abu Dhabi holds a 2-day course at 0% and a 6-day course at
33%; weighting by course length moves that branch's figure noticeably. Worth one sentence
in the write-up — knowing the two definitions diverge, and saying which you picked and
why, is the actual point of the question.

### Q2. Deployment, scaling, and hallucination control

**Deploy with Databricks Asset Bundles (DAB).** The whole project — notebooks, job
definitions, cluster config, schedules, permissions — is declared in a single
`databricks.yml` and version-controlled alongside the code. `databricks bundle deploy -t
dev` and `-t prod` push the identical asset graph to different targets, differing only in
catalog, schedule, and compute size. This removes hand-clicked jobs entirely: the job that
runs in production is the one defined in the repo, reviewed in a pull request, and
deployed by CI. Sketch:

```yaml
bundle:
  name: training_attendance

resources:
  jobs:
    attendance_pipeline:
      tasks:
        - task_key: silver_gold
          notebook_task: {notebook_path: ./silver_gold_layer.py}
        - task_key: data_quality
          depends_on: [{task_key: silver_gold}]
          notebook_task: {notebook_path: ./dq_notebook.py}
        - task_key: rag_index
          depends_on: [{task_key: data_quality}]
          notebook_task: {notebook_path: ./gold_rag_layer.py}

targets:
  dev:
    variables: {catalog: workspace_dev}
  prod:
    variables: {catalog: workspace_prod}
    resources:
      jobs:
        attendance_pipeline:
          schedule: {quartz_cron_expression: "0 0 2 * * ?"}
```

A GitHub Actions workflow runs `bundle validate` on every pull request and
`bundle deploy -t prod` on merge to main.

**Ingestion.** Today the CSV is read in full on each run. The first change is Auto Loader
(`cloudFiles`) on the Volume, which tracks which files have already been processed and
picks up only new arrivals — incremental with no code restructuring. If attendance starts
arriving continuously from a booking system rather than as a nightly file, the same logic
runs as a Structured Streaming job with `availableNow` triggers for batch economics or
continuous mode for genuine low latency. Lakeflow Declarative Pipelines (formerly DLT) is
the natural home for the whole chain: the DQ rules become `@dlt.expect` decorators,
metrics land in the event log automatically, and the hand-rolled aggregation in
`dq_notebook.py` disappears.

**Containerisation.** Little is needed on-platform: serverless compute and Model Serving
handle packaging. The RAG layer is the exception — it currently pip-installs
`sentence-transformers` at runtime and builds an in-memory FAISS index that dies with the
cluster. Production shape: register the embedding model in MLflow, serve it behind a
Model Serving endpoint (containerised by Databricks), and replace FAISS with a Databricks
Vector Search **Delta Sync index** on `gold_training_records`, which re-embeds
incrementally as the table changes. If the Q&A layer must run outside Databricks, the
retrieval service goes in a Docker image with a pinned model version and reads the index
over the SDK.

**Latency and cost.**

- Precompute embeddings; never embed the corpus per request. Only the query is embedded
  at request time.
- Job clusters, not all-purpose clusters, for scheduled work — they terminate on
  completion.
- Serverless SQL warehouse with a short auto-stop for the BI and alert queries.
- Photon for the aggregation stage; it pays for itself on shuffle-heavy `GROUP BY`.
- Liquid clustering plus `OPTIMIZE` on Silver to avoid small-file scan overhead.
- Cache repeated identical questions in the conversational layer — attendance data
  changes daily at most, so a short TTL cache removes most of the embedding and
  retrieval cost.
- Route by question type: the deterministic composer answers superlatives and lookups
  with no LLM call at all. Only genuinely open-ended questions reach a serving endpoint.

**Reducing hallucination.** No single control is sufficient — this needs layers, and the
measurements from this corpus show why.

1. **Retrieval confidence threshold.** Refuse when the top cosine score falls below
   `SIMILARITY_THRESHOLD`. Necessary but *not* sufficient, and the notebook demonstrates
   the failure: "What exam score did participants achieve on the Export Readiness course?"
   scores 0.642 — higher than two genuinely answerable questions at 0.519 and 0.514.
   Similarity measures topical relatedness, not answerability. The bands overlap, so no
   threshold separates them.
2. **Vocabulary scope gate.** Check whether the query's content words appear anywhere in
   the corpus. This catches exactly the case above: `exam` and `score` exist in no
   document, so the question is refused regardless of how well it scores.
3. **Structured-metadata composition.** Figures are read from metadata, never
   re-transcribed from prose by a model. A number cannot be garbled if no model ever
   reads it.
4. **Constrained generation.** The LLM prompt restricts the model to the retrieved
   context, mandates an exact refusal string when unsupported, and requires a Fileno
   citation for every figure.
5. **Mandatory citations.** Every answer carries the record reference behind it, so a
   reviewer can verify the claim against the table in one query.
6. **Permission-aware retrieval.** A Vector Search index is a Unity Catalog object and
   inherits the source table's grants. A user without `SELECT` on the table cannot
   retrieve its chunks — enforced by the platform, not by a prompt instruction.
7. **Regression evaluation.** Keep a fixed set of answerable and unanswerable questions;
   run it on every deploy with MLflow LLM-as-judge scoring for groundedness. Track
   refusal rate over time — a sudden drop means retrieval has drifted and the guardrail
   has gone quiet.

---
