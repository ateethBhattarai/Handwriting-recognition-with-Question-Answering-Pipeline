# 📘 Document Q&A with OCR + RAG (Flask + Chroma + Ollama)

A Flask web app that allows users to upload PDFs or images, automatically extracts text (using **PyMuPDF** for PDFs and **Tesseract OCR** for images), saves all text into a combined PDF, and builds a **vector database** with Chroma.  
Users can then ask questions, and a local **LLM (Ollama with LLaMA3.2)** answers using only the retrieved chunks — a classic **Retrieval-Augmented Generation (RAG)** workflow.

---

## ✨ Features
- Upload multiple **PDF** and **image** files (`.pdf`, `.jpg`, `.jpeg`, `.png`).
- **Images**: preprocessed with OpenCV and read via **Tesseract OCR**.
- **PDFs**: parsed directly with **PyMuPDF** (fast, accurate text extraction).
- All extracted text merged into a new **clean, searchable PDF** (via ReportLab).
- Indexed into a **Chroma vector DB** with **Ollama embeddings** (`mxbai-embed-large`).
- Users can ask questions; top-5 relevant chunks are retrieved and fed to **llama3.2** via Ollama.
- Lightweight **metrics logging**: OCR chars, embedding counts, retrieval IDs, stage timings, and LLM response length.

---

## 🛠️ Requirements

### 1. Python packages
Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Ollama Packages
# Start Ollama server (if not already running)

```bash
ollama serve
```
# Pull the LLM model

```bash
ollama pull llama3.2
```

# Pull the embedding model

```bash
ollama pull mxbai-embed-large
```
