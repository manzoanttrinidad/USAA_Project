import os
import json
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 256
CACHE_PATH = Path("embeddings_cache.sqlite")

# ----------------------------
# OpenAI client (modern SDK)
# ----------------------------
load_dotenv()  # loads from a .env in your project root if present
api_key = os.getenv("OPENAI_API_KEY")   # <-- put your key in the env var OPENAI_API_KEY
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Export it or add it to a .env file.")

client = OpenAI(api_key=api_key)

def _embed_request(inputs, model):
    """Helper that sends a batch of texts to the OpenAI embeddings API."""
    resp = client.embeddings.create(model=model, input=inputs)
    return [d.embedding for d in resp.data]

def get_embedding(text, model="text-embedding-3-small"):
    text = (text or "").replace("\n", " ")
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding



# (B) Newer SDK style:
# from openai import OpenAI
# client = OpenAI()
# def _embed_request(inputs, model):
#     resp = client.embeddings.create(model=model, input=inputs)
#     return [d.embedding for d in resp.data]

# ----------------------------
# Simple SQLite cache
# ----------------------------
def init_cache(db_path=CACHE_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            text TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            vec  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def cache_get_many(conn, texts, model, chunk_size=500):
    rows = {}
    if not texts:
        return rows

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        qmarks = ",".join("?" for _ in chunk)
        sql = f"SELECT text, model, vec FROM cache WHERE text IN ({qmarks}) AND model = ?"
        for t, m, v in conn.execute(sql, (*chunk, model)):
            rows[t] = json.loads(v)
    return rows


def cache_put_many(conn, text_vecs, model):
    conn.executemany(
        "INSERT OR REPLACE INTO cache (text, model, vec) VALUES (?, ?, ?)",
        [(t, model, json.dumps(vec)) for t, vec in text_vecs.items()]
    )
    conn.commit()

# ----------------------------
# Batch embed with retry + cache
# ----------------------------
def embed_texts_batched(texts, model=EMBED_MODEL, batch_size=BATCH_SIZE, sleep_base=1.5):
    conn = init_cache()
    # Clean + dedup while preserving original order
    texts = ["" if t is None else str(t) for t in texts]
    uniq = []
    seen = set()
    for t in texts:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    # pull from cache
    cached = cache_get_many(conn, uniq, model)
    to_fetch = [t for t in uniq if t not in cached and t.strip() != ""]

    # fetch in batches
    fetched = {}
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i+batch_size]
        # robust retry loop
        tries, backoff = 0, sleep_base
        while True:
            try:
                vecs = _embed_request(batch, model=model)
                for t, v in zip(batch, vecs):
                    fetched[t] = v
                break
            except Exception as e:
                tries += 1
                if tries >= 6:
                    raise
                time.sleep(backoff)
                backoff *= 1.8  # exponential backoff

    # persist new cache entries
    if fetched:
        cache_put_many(conn, fetched, model)

    # merge cached + fetched
    store = {**cached, **fetched}

    # map back to original order
    return [store.get(t, store.get("")) for t in texts]

# ----------------------------
# Cosine similarity helpers
# ----------------------------
def to_np(array_of_lists):
    # converts list[list[float]] -> np.ndarray (n, d)
    return np.asarray(array_of_lists, dtype=np.float32)

def cosine_similarity_matrix(A, B):
    # A: (n,d), B: (m,d) -> (n,m)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T

# ----------------------------
# Your specific workflow
# ----------------------------
# 1) Load CSVs
titles_df = pd.read_csv("extracted_urls.csv")
gloss_df  = pd.read_csv("fraud_glossary.csv")

# 2) Normalize strings
titles = titles_df["extracted"].fillna("").astype(str).tolist()
terms  = gloss_df["Term"].fillna("").astype(str).tolist()

# 3) Fast, cached, batched embeddings
title_vecs = embed_texts_batched(titles, model=EMBED_MODEL)
term_vecs  = embed_texts_batched(terms,  model=EMBED_MODEL)

THRESHOLD = 0.60 # adjust

# 0) Make sure your titles CSV has a URL column
#    e.g., columns: ["extracted", "url"] (adjust if your column name differs)
url_col = "url" if "url" in titles_df.columns else "URL"

# 1) Cosine matrix (titles x terms)
A = to_np(title_vecs)
B = to_np(term_vecs)
sim = cosine_similarity_matrix(A, B)  # (n_titles, n_terms)

# 2) Per-title stats
max_sim = sim.max(axis=1)                         # best match per title
argmax_j = sim.argmax(axis=1)                     # index of best term
best_term = [terms[j] for j in argmax_j]

# 3) (optional) all matched terms above threshold (semicolon-separated)
matches = []
for i in range(sim.shape[0]):
    j_idx = np.where(sim[i] >= THRESHOLD)[0]
    matches.append("; ".join(terms[j] for j in j_idx))

# 4) Attach results back to titles_df
titles_df["best_term"] = best_term
titles_df["best_cosine"] = max_sim
titles_df["matched_terms"] = matches

# 5) Filter to “fraud-ish” titles and save URLs for crawling
fraudish = titles_df[titles_df["best_cosine"] >= THRESHOLD].copy()
fraudish[["extracted", url_col, "best_term", "best_cosine", "matched_terms"]].to_csv(
    "fraudish_urls.csv", index=False
)

print(f"Saved {len(fraudish)} rows to fraudish_urls.csv")

