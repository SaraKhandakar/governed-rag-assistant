import os
import json
import streamlit as st

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

DB_DIR = "chroma_db"

st.set_page_config(page_title="Governed RAG Assistant", layout="wide")
st.title("📚 Governed RAG Assistant (Local + Cloud modes)")

# ---------- Sidebar: choose mode ----------
st.sidebar.header("Run Mode")

mode = st.sidebar.selectbox(
    "Choose where the LLM runs",
    ["Local (Ollama - free)", "Cloud (Groq - free tier)"]
)

# Local model name (must exist in `ollama list`)
local_model = st.sidebar.text_input("Local model", value="phi3")

# Groq model name (common choices; you can change)
groq_model = st.sidebar.text_input("Groq model", value="llama3-8b-8192")

# ---------- Load vector DB ----------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 4})


# ---------- Choose LLM ----------
def get_llm():
    if mode.startswith("Local"):
        return ChatOllama(model=local_model, temperature=0)

    # Cloud mode (Groq)
    # Streamlit Cloud uses st.secrets; local can use env var
    groq_key = None
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
    else:
        groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        st.error("Cloud mode needs GROQ_API_KEY (add it to Streamlit Secrets or your environment variables).")
        st.stop()

    return ChatGroq(model=groq_model, temperature=0, api_key=groq_key)


llm = get_llm()


# ---------- Helper: format sources ----------
def format_source(doc, idx: int) -> str:
    stype = doc.metadata.get("source_type", "unknown")
    src = doc.metadata.get("source", "unknown")

    if stype == "pdf":
        page = doc.metadata.get("page", None)
        return f"• [{idx}] {src} (page {page})"
    if stype == "excel":
        sheet = doc.metadata.get("sheet", "Sheet1")
        row = doc.metadata.get("row", None)
        return f"• [{idx}] {src} (sheet: {sheet}, row: {row})"
    return f"• [{idx}] {src}"


# ---------- RAG answer ----------
def answer_with_sources(question: str):
    docs = retriever.invoke(question)

    context = ""
    sources = []
    for i, d in enumerate(docs, start=1):
        sources.append(format_source(d, i))
        context += f"\n--- Source {i} ---\n{d.page_content}\n"

    prompt = f"""You are a governed, read-only assistant.
Use ONLY the context below. Do not use outside knowledge.
If the answer is not in the context, say:
"I don't have enough information based on the provided documents."

QUESTION:
{question}

CONTEXT:
{context}

Write a clear answer.
End with citations like [1], [2] referencing the sources.
ANSWER:
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content, sources


# ---------- Chat UI ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

q = st.chat_input("Ask a question about the PDFs/Excels...")
if q:
    st.session_state.history.append(("user", q))
    ans, srcs = answer_with_sources(q)
    st.session_state.history.append(("assistant", ans))
    st.session_state.last_sources = srcs

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

if st.session_state.last_sources:
    st.divider()
    st.subheader("Sources")
    for s in st.session_state.last_sources:
        st.write(s)
        