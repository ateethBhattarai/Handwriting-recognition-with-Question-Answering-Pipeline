import os
import cv2
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import time


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def is_image_filename(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)

start_time = time.time()

def preprocess_image(image_path: str):
    """Preprocess for OCR: grayscale, binarize, denoise."""
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
    """Run OCR on a single image file after preprocessing."""
    processed = preprocess_image(image_path)
    if processed is None:
        return ""
    text = pytesseract.image_to_string(processed)
    print("The image contained:\n")
    print(text)
    print("---The pdf generation took %.3f seconds ---" % (time.time() - start_time))

    return text or ""


def save_text_as_pdf(text: str, output_pdf_path: str):
    """Create a simple text-only PDF containing the OCR/extracted text."""
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin
    line_height = 14

    for raw_line in text.splitlines():
        # Simple wrapping (naive): split overly long lines
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
