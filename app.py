# app.py
import streamlit as st
from rag_core import load_corpus, answer_question

@st.cache_resource
def init_rag():
    df, emb_matrix = load_corpus()
    return df, emb_matrix

st.set_page_config(page_title="Economics RAG Demo", layout="wide")
st.title("📊 Economics Terms Q&A (RAG)")

st.write("Ask a question and I'll answer using the economics_terms.csv corpus hosted on Supabase.")

df, emb_matrix = init_rag()

question = st.text_input("Your question:")
go = st.button("Ask")

if go and question:
    with st.spinner("Thinking..."):
        answer, ctx_rows = answer_question(question, df, emb_matrix)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("See retrieved context rows"):
        st.dataframe(ctx_rows[["text"]])
        st.dataframe(ctx_rows[["url"]])
