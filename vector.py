import os
import fitz
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DB_DIR = "./chroma_langchain_db"
os.makedirs(DB_DIR, exist_ok=True)

# Initialize embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def build_vector_store_from_texts(
    collection_id: str,
    texts_with_meta: List[Tuple[str, Dict]],
):
    """
    Create a new Chroma collection for this upload batch and add documents.
    texts_with_meta: list of (text, metadata)
    """
    collection_name = f"kb_{collection_id}"
    docs = [Document(page_content=t, metadata=m) for (t, m) in texts_with_meta]

    # Create a new collection (Chroma wrapper auto-persists with persist_directory)
    _ = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=DB_DIR,
    )


def get_retriever(collection_id: str, k: int = 5):
    """Load retriever for an existing collection."""
    collection_name = f"kb_{collection_id}"
    store = Chroma(
        collection_name=collection_name,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    return store.as_retriever(search_kwargs={"k": k})
