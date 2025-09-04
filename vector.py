import os
import fitz
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from metrics import timer, log_metric  # NEW

DB_DIR = "./chroma_langchain_db"
os.makedirs(DB_DIR, exist_ok=True)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def extract_text_from_pdf(pdf_path: str) -> str:
    with timer("pdf_extract", {"doc_id": os.path.basename(pdf_path)}):
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    log_metric("pdf_chars", len(text), {"doc_id": os.path.basename(pdf_path)})
    return text

def build_vector_store_from_texts(
    collection_id: str,
    texts_with_meta: List[Tuple[str, Dict]],
):
    collection_name = f"kb_{collection_id}"

    # Tag chunks with stable IDs for eval
    docs: List[Document] = []
    for i, (t, m) in enumerate(texts_with_meta):
        meta = dict(m)
        meta["doc_id"] = f"{collection_id}-{i}"
        docs.append(Document(page_content=t, metadata=meta))

    with timer("embed_ingest", {"doc_id": collection_name}):
        _ = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=DB_DIR,
        )
    log_metric("embed_docs_count", len(docs), {"doc_id": collection_name})

def get_retriever(collection_id: str, k: int = 5):
    collection_name = f"kb_{collection_id}"
    store = Chroma(
        collection_name=collection_name,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    return store.as_retriever(search_kwargs={"k": k})