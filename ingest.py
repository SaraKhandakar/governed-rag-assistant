from pathlib import Path
import pandas as pd

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdf"
EXCEL_DIR = DATA_DIR / "excel"
DB_DIR = Path("chroma_db")


def pdf_to_documents(pdf_path: Path) -> list[Document]:
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",
        infer_table_structure=False
    )

    chunks = chunk_by_title(
        elements,
        max_characters=900,
        new_after_n_chars=700,
        combine_text_under_n_chars=200
    )

    docs = []
    for idx, ch in enumerate(chunks):
        text = (getattr(ch, "text", "") or "").strip()
        if not text:
            continue

        page = getattr(getattr(ch, "metadata", None), "page_number", None)

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source_type": "pdf",
                    "source": pdf_path.name,
                    "page": page,
                    "chunk_id": f"{pdf_path.name}::chunk_{idx:05d}",
                }
            )
        )
    return docs


def excel_to_documents(excel_path: Path) -> list[Document]:
    xls = pd.ExcelFile(excel_path)
    docs = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        df = df.dropna(axis=1, how="all")

        for i, row in df.iterrows():
            row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])

            docs.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "source_type": "excel",
                        "source": excel_path.name,
                        "sheet": sheet_name,
                        "row": int(i),
                    }
                )
            )

    return docs


def build_db(all_docs: list[Document]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )
    return db


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    EXCEL_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    excels = sorted(EXCEL_DIR.glob("*.xls")) + sorted(EXCEL_DIR.glob("*.xlsx"))

    if not pdfs and not excels:
        raise FileNotFoundError("No PDF/Excel files found in data/pdf or data/excel")

    all_docs = []

    for pdf in pdfs:
        print(f"📄 Ingesting PDF: {pdf.name}")
        docs = pdf_to_documents(pdf)
        print(f"   ✅ {len(docs)} chunks")
        all_docs.extend(docs)

    for ex in excels:
        print(f"📊 Ingesting Excel: {ex.name}")
        docs = excel_to_documents(ex)
        print(f"   ✅ {len(docs)} rows")
        all_docs.extend(docs)

    print(f"\n🧠 Total docs: {len(all_docs)}")
    build_db(all_docs)
    print(f"✅ Chroma DB saved to: {DB_DIR.resolve()}")


if __name__ == "__main__":
    main()