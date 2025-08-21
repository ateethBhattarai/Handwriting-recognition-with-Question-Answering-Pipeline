import os
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename

from ocr_utils import ocr_image_to_text, save_text_as_pdf, is_image_filename
from vector import (
    extract_text_from_pdf,
    build_vector_store_from_texts,
    get_retriever,
)
from llm import ask_llm

pdf_bp = Blueprint("pdf", __name__)

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@pdf_bp.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "files" not in request.files:
            return render_template("upload.html", error="No files provided.")

        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return render_template("upload.html", error="No selected files.")

        # Unique collection id for this upload batch
        collection_id = uuid.uuid4().hex

        all_texts = []  # list of (text, metadata)
        uploaded_names = []

        for f in files:
            if not f or f.filename == "":
                continue
            if not allowed_file(f.filename):
                continue

            safe_name = secure_filename(f.filename)
            uploaded_names.append(safe_name)
            file_path = os.path.join(UPLOAD_DIR, f"{collection_id}_{safe_name}")
            f.save(file_path)

            ext = safe_name.rsplit(".", 1)[1].lower()

            if ext == "pdf":
                # Extract text directly from PDF
                text = extract_text_from_pdf(file_path)
                if text.strip():
                    all_texts.append((text, {"source": safe_name, "type": "pdf"}))
            else:
                # Image -> OCR text
                if is_image_filename(safe_name):
                    text = ocr_image_to_text(file_path)
                    if text.strip():
                        all_texts.append((text, {"source": safe_name, "type": "image"}))

        if not all_texts:
            return render_template(
                "upload.html", error="No readable text found in the uploaded files."
            )

        # Also create one combined PDF with the OCR/text (optional, for user reference)
        combined_pdf_path = os.path.join(PROCESSED_DIR, f"{collection_id}_combined.pdf")
        combined_text = "\n\n".join([t for (t, _) in all_texts])
        save_text_as_pdf(combined_text, combined_pdf_path)

        # Build a fresh vector store for this batch (per collection_id)
        build_vector_store_from_texts(collection_id, all_texts)

        # Redirect to ask page for this specific collection
        return redirect(url_for("pdf.ask_page", collection_id=collection_id))

    return render_template("upload.html")


@pdf_bp.route("/ask/<collection_id>", methods=["GET", "POST"])
def ask_page(collection_id: str):
    # Create a retriever on demand for this collection
    try:
        retriever = get_retriever(collection_id, k=5)
    except Exception:
        abort(404)

    answer = None
    question = None

    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        if question:
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
            answer = ask_llm(question, context)

    return render_template(
        "ask.html", collection_id=collection_id, answer=answer, question=question
    )
