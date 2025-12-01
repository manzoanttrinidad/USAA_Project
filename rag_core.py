# rag_core.py
import os
import pandas as pd
import numpy as np
import io
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client



# --- 1. Load data from Supabase-hosted CSV ---

def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in .env")
    return create_client(url, key)

def load_corpus():
    supabase = get_client()

    resp = (
        supabase
        .table("RAG_Fraud_Terms_Embeddings")
        .select("*")
        .execute()
    )

    rows = resp.data
    if not rows:
        raise RuntimeError("Supabase returned no rows from RAG_Fraud_Terms_Embeddings")

    df = pd.DataFrame(rows)

    # TEMP: inspect what we really have
    print("Supabase columns:", df.columns.tolist())
    # if you're in Streamlit, you can also do:
    # import streamlit as st
    # st.write("Supabase columns:", df.columns.tolist())

    # Find an embedding column by name (case-insensitive)
    candidates = [c for c in df.columns if c.lower() in ("embedding", "embeddings")]
    if not candidates:
        raise RuntimeError(f"Could not find an embedding column in columns: {df.columns.tolist()}")

    emb_col = candidates[0]  # e.g. "Embedding" or "embedding"

    # Convert Postgres float[] -> numpy arrays
    df[emb_col] = df[emb_col].apply(
        lambda arr: np.array(arr, dtype=np.float32) if arr is not None else None
    )

    # Drop any rows with missing embeddings just in case
    df = df[df[emb_col].notnull()].reset_index(drop=True)

    emb_matrix = np.stack(df[emb_col].to_numpy())

    return df, emb_matrix




# --- 2. Embedding helper ---

def get_embedding(text: str, model: str = "text-embedding-3-small"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    text = text.replace("\n", " ")
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


# --- 3. Core RAG pipeline ---

def retrieve_context(question: str, df, emb_matrix, top_k: int = 4):
    q_emb = np.array(get_embedding(question)).reshape(1, -1)
    sims = cosine_similarity(q_emb, emb_matrix).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    top_rows = df.iloc[top_idx]
    return top_rows, sims[top_idx]


def answer_question(question: str, df, emb_matrix):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    top_rows, scores = retrieve_context(question, df, emb_matrix, top_k=4)
    context_snippets = top_rows["text"].tolist()

    system_prompt = (
        "You are a helpful assistant that answers questions using the provided "
        "economics term descriptions as context. If the context doesn't contain "
        "the answer, say you don't know."
    )

    user_prompt = (
        f"Context entries:\n{context_snippets}\n\n"
        f"Question: {question}\n\n"
        "Use only the context above when possible."
    )

    chat = client.chat.completions.create(
        model="gpt-4.1-mini",  # or gpt-4.1 / gpt-3.5-turbo
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = chat.choices[0].message.content
    return answer, top_rows
