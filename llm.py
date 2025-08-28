from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time

start_time = time.time()
# Initialize the LLM once (Ollama must be running)
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


def ask_llm(question: str, pdf_content: str) -> str:
    result = chain.invoke({"pdfFile": pdf_content, "question": question})
    print("---Answering took %.3f seconds ---" % (time.time() - start_time))
    return str(result)
