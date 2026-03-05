import streamlit as st

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

DB_DIR = "chroma_db"

st.set_page_config(page_title="Governed RAG (Free)", layout="wide")
st.title("📚 Governed RAG Assistant (FREE local)")

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 4})

# Local model (must have ollama running + model pulled)
llm = ChatOllama(model="phi3", temperature=0)

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
        