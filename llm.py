from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
from metrics import timer, log_metric  # NEW

# LLM Initialization
model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions based on an uploaded document batch.
Use only the provided context to answer.

Context:
{pdfFile}

Question:
{question}

If the context does not contain the answer, say you don't have enough information.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def ask_llm(question: str, pdf_content: str, doc_id: str | None = None) -> str:
    with timer("llm_infer", {"doc_id": doc_id or "", "user_q": question}):
        result = chain.invoke({"pdfFile": pdf_content, "question": question})
    # Optional: log answer length
    log_metric("llm_answer_chars", len(str(result or "")), {"doc_id": doc_id or "", "user_q": question})
    return str(result)