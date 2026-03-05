import os
import streamlit as st

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq

IS_CLOUD = os.getenv("STREAMLIT_SERVER_RUNNING", "") == "true" or os.path.exists("/mount/src")

DB_DIR = "chroma_db"

st.set_page_config(page_title="Governed RAG Assistant", layout="wide")
st.title("📚 Governed RAG Assistant (Local + Cloud modes)")

# ---------- Sidebar ----------
st.sidebar.header("Run Mode")

if IS_CLOUD:
    st.sidebar.info("Running on Streamlit Cloud → using Cloud (Groq) mode.")
    mode = "Cloud (Groq - free tier)"
else:
    mode = st.sidebar.selectbox(
        "Choose where the LLM runs",
        ["Local (Ollama - free)", "Cloud (Groq - free tier)"]
    )

local_model = st.sidebar.text_input("Local model", value="phi3")
groq_model = st.sidebar.selectbox(
    "Groq model",
    [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "gemma2-9b-it",
    ],
    index=0
)


# ---------- Load vector DB ----------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 2})

# ---------- LLM चयन ----------
def get_llm():
    if mode.startswith("Local"):
        return ChatOllama(model=local_model, temperature=0)

    # Cloud (Groq)
    groq_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.error("Cloud mode needs GROQ_API_KEY (add it to Streamlit Secrets).")
        st.stop()

    return ChatGroq(model=groq_model, temperature=0, api_key=groq_key)

llm = get_llm()

# ---------- Source formatter ----------
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

# ---------- RAG ----------
def answer_with_sources(question: str):
    docs = retriever.invoke(question)

    context_parts = []
    sources = []

    for i, d in enumerate(docs, start=1):
        sources.append(format_source(d, i))

        # limit chunk size to avoid Groq BadRequest (context too long)
        chunk_text = (d.page_content or "")[:1200]
        context_parts.append(f"--- Source {i} ---\n{chunk_text}")

    context = "\n\n".join(context_parts)

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

try:
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content, sources
except Exception as e:
    return f"❌ Cloud LLM error: {e}", sources

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