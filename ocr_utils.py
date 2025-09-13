import os
import cv2
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from metrics import timer, log_metric

# Allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = {"png","jpg","jpeg"}

def is_image_filename(filename:str)->bool:
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def preprocess_image(image_path: str):
    # Time this block
    with timer("preprocess", {"doc_id": os.path.basename(image_path)}):
        img = cv2.imread(image_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        bw = cv2.medianBlur(bw, 3)
        return bw

def ocr_image_to_text(image_path: str) -> str:
    processed = preprocess_image(image_path)
    if processed is None:
        return ""

    with timer("ocr", {"doc_id": os.path.basename(image_path)}):
        text = pytesseract.image_to_string(processed) or ""

    try:
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        conf_vals = [int(c) for c in data.get("conf", []) if c not in ("-1", None, "", "-")]
        if conf_vals:
            log_metric("ocr_confidence_mean", sum(conf_vals) / len(conf_vals), {"doc_id": os.path.basename(image_path)})
    except Exception:
        pass
    return text

def save_text_as_pdf(text: str, output_pdf_path: str):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    with timer("pdf_generate", {"doc_id": os.path.basename(output_pdf_path)}):
        c = canvas.Canvas(output_pdf_path, pagesize=letter)
        width, height = letter
        margin = 50
        y = height - margin
        line_height = 14

        for raw_line in text.splitlines():
            line = raw_line.strip().replace("\t", "    ")
            while len(line) > 0:
                chunk = line[:110]
                c.drawString(margin, y, chunk)
                y -= line_height
                line = line[110:]
                if y < margin:
                    c.showPage()
                    y = height - margin
            if y < margin:
                c.showPage()
                y = height - margin
        c.save()
    # simple success flag
    ok = os.path.exists(output_pdf_path) and os.path.getsize(output_pdf_path) > 0
    log_metric("pdf_success", int(ok), {"doc_id": os.path.basename(output_pdf_path)})