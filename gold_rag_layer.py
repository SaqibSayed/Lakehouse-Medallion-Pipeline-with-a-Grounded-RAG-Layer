# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "sentence-transformers",
#   "faiss-cpu",
# ]
# ///
# =====================================================================================
# RAG LAYER OVER GOLD ATTENDANCE DATA
# Reads: gold_training_records  (record-level enriched table built in Part 1)
# Pipeline: Gold rows -> text documents -> embeddings -> FAISS index -> grounded Q&A
# =====================================================================================

# COMMAND ----------

# MAGIC %pip install sentence-transformers faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# =====================================================================================
# STEP 1 - Load Gold table
# gold_training_records is the record-level Gold surface: same grain as silver_training,
# with the pipeline-internal quality flags stripped and attendance_rate /
# attendance_band already derived. Rejected rows never reach Gold, so the corpus is
# curated by construction.
# =====================================================================================
gold_df = spark.table("gold_training_records").toPandas()

print(f"Gold records loaded: {len(gold_df)}")
gold_df.head()

# COMMAND ----------

# =====================================================================================
# STEP 2 - Turn Gold rows into text "documents"
# DECISION (documented): one document per attendance record rather than per Gold
# aggregate row. This makes record-level questions answerable ("which records were
# flagged?", "what did the Dubai participant attend?") at the cost of aggregate
# precision -- see the note in Step 6 on superlative questions. Each document carries
# the figures in both natural-language text (for embedding) and structured metadata
# (for exact citation/answer composition, avoiding any risk of an LLM mis-transcribing
# a number when it "reads" it back).
#
# CHANGED vs the Silver version: dq_days_mismatch does not exist in Gold (it is a
# pipeline-internal quality signal that stops at Silver). The sentence it drove is
# replaced by attendance_band, a genuine Gold-layer feature, so the document keeps the
# same shape and the same amount of retrievable detail.
#
# PII: the source header tagged Applicant as PII. The applicant name is deliberately
# NOT written into the embedded text, so it cannot surface through a similarity match
# or an LLM completion. Fileno is used as the record reference instead.
# =====================================================================================
def build_documents(df):
    docs = []
    for _, row in df.iterrows():
        sched = row["number_of_days"]
        att = row["days_attended"]
        rate = float(att) / float(sched) if (pd.notna(sched) and pd.notna(att) and sched) else None
        rate_txt = f"{rate * 100:.2f}%" if rate is not None else "not computable (missing day counts)"
        age_txt = f"aged {int(row['age_clean'])}" if pd.notna(row["age_clean"]) else "with no validated age on record"
        date_txt = str(row["course_date"]) if pd.notna(row["course_date"]) else "an unrecorded date"
        band_txt = (
            f"This record is classified in the {row['attendance_band']} attendance band."
            if pd.notna(row["attendance_band"])
            else "This record has no attendance band classification."
        )

        text = (
            f"Attendance record {row['fileno']}: a participant {age_txt} was enrolled in the "
            f"'{row['course_name']}' course at the {row['branch']} branch, starting {date_txt}. "
            f"The course was scheduled for {row['number_of_days']} days; the participant attended "
            f"{row['days_attended']} days and was absent {row['days_absent']} days, giving a "
            f"record-level attendance rate of {rate_txt}. {band_txt}"
        )

        docs.append({
            "text": text,
            "metadata": {
                "type": "attendance record",
                "name": row["fileno"],
                "branch": row["branch"],
                "course_name": row["course_name"],
                "course_date": date_txt,
                "scheduled": None if pd.isna(sched) else int(sched),
                "attended": None if pd.isna(att) else int(att),
                "absent": None if pd.isna(row["days_absent"]) else int(row["days_absent"]),
                "rate": rate,
                "age": None if pd.isna(row["age_clean"]) else int(row["age_clean"]),
                "attendance_band": None if pd.isna(row["attendance_band"]) else row["attendance_band"],
            },
        })
    return docs

documents = build_documents(gold_df)

for d in documents:
    print(d["text"])

# COMMAND ----------

# =====================================================================================
# STEP 3 - Embed documents and build a vector index (FAISS, in-memory)
# Normalized embeddings + IndexFlatIP => cosine similarity search.
# (Chroma or an in-memory dict of vectors would work identically for a corpus this
# small; FAISS is used here since it ships lightweight and needs no server process.)
# =====================================================================================
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_texts = [d["text"] for d in documents]
doc_embeddings = model.encode(doc_texts, normalize_embeddings=True)

index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings, dtype="float32"))

# COMMAND ----------

# =====================================================================================
# STEP 4 - Retrieval
# =====================================================================================
def retrieve(query, k=3):
    k = min(k, len(documents))
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb, dtype="float32"), k)
    return [(documents[i], float(scores[0][j])) for j, i in enumerate(idxs[0])]

# COMMAND ----------

# =====================================================================================
# STEP 5 - Grounded answer generation, with a refusal path
#
# SIMILARITY_THRESHOLD is the guardrail for requirement #3 ("if the answer is not
# supported by the data, say so"). The corpus only covers branch/course/attendance
# figures per record -- a query about anything outside that (exam scores, cost,
# trainer, gender) should retrieve nothing with meaningful similarity, and gets refused
# before any answer is composed.
#
#
# RETUNE NOTE vs the aggregate-Gold version: these documents all share one sentence
# template and are longer, so baseline cosine similarity sits HIGHER than it did over
# the terse aggregate sentences. 0.35 may rarely fire here and leave the refusal path
# effectively dead. Print top_score against a few known-unanswerable questions and raise
# the threshold to sit between the highest unanswerable score and the lowest answerable
# one -- 0.45 is a realistic starting point for all-MiniLM-L6-v2 on this corpus.
# =====================================================================================

SIMILARITY_THRESHOLD = 0.49


def format_citation(doc):
    m = doc["metadata"]
    rate_txt = "n/a" if m["rate"] is None else f"{m['rate']*100:.2f}%"
    return f"[{m['type']}: {m['name']} — {m['course_name']} @ {m['branch']} — {rate_txt} ({m['attended']}/{m['scheduled']} days)]"

def generate_answer(query, k=3, threshold=SIMILARITY_THRESHOLD, use_llm=False):
    results = retrieve(query, k)
    top_score = results[0][1]

    if top_score < threshold:
        return {
            "question": query,
            "answer": "I don't have enough information in the curated data to answer that question.",
            "grounded": False,
            "top_score": round(top_score, 3),
            "sources": [],
        }

    if use_llm:
        try:
            return _generate_with_llm(query, results)
        except Exception as e:
            print(f"LLM generation unavailable ({e}); falling back to deterministic composer.")

    return _generate_deterministic(query, results)

def _generate_deterministic(query, results):
    """Rule-based composer: picks min/max or a direct lookup from the retrieved docs
    only (never from documents outside the retrieved set) -- keeps the answer strictly
    grounded without depending on an external LLM endpoint being available."""
    docs = [r[0] for r in results]
    q = query.lower()
    rated = [d for d in docs if d["metadata"]["rate"] is not None]

    if rated and any(w in q for w in ["lowest", "worst", "minimum", "least"]):
        target = min(rated, key=lambda d: d["metadata"]["rate"])
        sources = [target]
        body = f"{target['metadata']['name']} has the lowest attendance rate at {target['metadata']['rate']*100:.2f}%."
    elif rated and any(w in q for w in ["highest", "best", "maximum", "most"]):
        target = max(rated, key=lambda d: d["metadata"]["rate"])
        sources = [target]
        body = f"{target['metadata']['name']} has the highest attendance rate at {target['metadata']['rate']*100:.2f}%."
    else:
        target = docs[0]
        m = target["metadata"]
        rate_txt = "not computable" if m["rate"] is None else f"{m['rate']*100:.2f}%"
        sources = [target]
        body = f"{m['name']} — {m['course_name']} at {m['branch']}: attendance rate {rate_txt}."

    citations = " ".join(format_citation(d) for d in sources)
    return {
        "question": query,
        "answer": f"{body} {citations}",
        "grounded": True,
        "top_score": round(results[0][1], 3),
        "sources": [d["metadata"]["name"] for d in sources],
    }

def _generate_with_llm(query, results):
    """Optional: route through a Databricks Foundation Model API serving endpoint for
    natural-language generation, strictly constrained to the retrieved context. Requires
    a workspace with Foundation Model APIs enabled -- wrapped in try/except above so the
    notebook still runs end-to-end without it."""
    context = "\n".join(d["text"] for d, _ in results)
    prompt = f"""You are answering questions ONLY using the context below, drawn from a curated Gold table of training attendance records.
Each context item is a single attendance record. Do not aggregate across records unless every relevant record is present in the context.
If the answer is not directly supported by the context, respond exactly with: "I don't have enough information in the data to answer that."
Never infer or invent a participant's identity - names are withheld as PII.
Always cite the record reference (Fileno) for each figure you used.

Context:
{context}

Question: {query}
Answer:"""
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    response = client.predict(
        endpoint="databricks-meta-llama-3-1-70b-instruct",
        inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 200},
    )
    answer_text = response["choices"][0]["message"]["content"]
    return {
        "question": query,
        "answer": answer_text,
        "grounded": True,
        "top_score": round(results[0][1], 3),
        "sources": [d["metadata"]["name"] for d, _ in results],
    }

# COMMAND ----------

# =====================================================================================
# STEP 6 - Demo: sample questions 
#
# TWO CONSEQUENCES OF RECORD-LEVEL GRAIN -- both handled at the call site, with no
# change to the retrieval or composition logic above:
#
# 1. SUPERLATIVES NEED THE FULL CORPUS. Over the two aggregate Gold tables, k=3 across
#    ~6 rows was enough for min/max to be correct. At record grain the composer only
#    sees the top-3 retrieved RECORDS, so "which record has the lowest rate?" would
#    return the lowest of three arbitrary rows -- a confident but wrong answer.
#    Superlative questions are therefore called with k=len(documents) so the composer
#    sees every record. This is viable because the corpus is small; at scale it must
#    become a computed aggregate rather than a retrieval, since you cannot retrieve the
#    whole table into context.
# =====================================================================================
sample_questions = [
    # --- ABOVE threshold (0.45): answered from the corpus ---
    ("What is the attendance rate for the SME Growth course?", 3),
    ("How many days did the Export Readiness participant attend?", 3),
    ("Which attendance record has the lowest attendance rate?", len(documents)),
    ("Which attendance record has the highest attendance rate?", len(documents)),

    # --- BELOW threshold (0.45): refused on low retrieval confidence ---
    ("What courses are running in sharjah?", 3),
    ("Average age of participants?", 3),
    ("What is the annual leave entitlement?", 3),
]

for q, k in sample_questions:
    result = generate_answer(q, k=k, use_llm=False)  # set use_llm=True if a serving endpoint is configured
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"  (grounded={result['grounded']}, top_score={result['top_score']}, sources={result['sources']})")
    print("-" * 100)