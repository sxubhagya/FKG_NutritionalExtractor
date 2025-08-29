import io
import os
import re
import cv2
import json
import google.generativeai as genai
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

_DOCTR_PREDICTOR = None

def _get_doctr_predictor():
    global _DOCTR_PREDICTOR
    if _DOCTR_PREDICTOR is None:
        try:
            _DOCTR_PREDICTOR = ocr_predictor(pretrained=True)
        except Exception:
            _DOCTR_PREDICTOR = None
    return _DOCTR_PREDICTOR

# ------------------------
# IMAGE PREPROCESSING
# ------------------------
def preprocess_image(image_bytes, max_size=1200):
    from PIL import ImageOps
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    img = np.array(image)[:, :, ::-1]  # RGB → BGR for OpenCV

    h, w = img.shape[:2]
    scale = float(max_size) / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=7)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 9
    )
    return img, th

def detect_label_region(img, gray, min_area=2000):
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    connected = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:
            continue
        candidates.append((area, x, y, w, h))

    if not candidates:
        return img

    candidates.sort(reverse=True, key=lambda t: t[0])
    _, x, y, w, h = candidates[0]
    pad = int(0.02 * max(img.shape[:2]))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1].copy()
    return crop

def correct_perspective(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rect = None
    img_area = crop.shape[0] * crop.shape[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.25 * img_area:
            rect = approx
            break
    if rect is None:
        return crop
    pts = rect.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype="float32")
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxW, maxH = int(max(wA, wB)), int(max(hA, hB))
    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32"
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(crop, M, (maxW, maxH))
    return warped

# ------------------------
# OCR
# ------------------------
def _top_of_bbox(bbox):
    try:
        return bbox[1]
    except:
        return 0

def ocr_ensemble(img_crop, use_doctr=True):
    lines = []

    predictor = _get_doctr_predictor() if use_doctr else None
    if predictor is not None:
        try:
            temp = "._doctr_tmp.jpg"
            cv2.imwrite(temp, img_crop)
            doc = DocumentFile.from_images(temp)
            res = predictor(doc)
            page = res.pages[0]
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join([w.value for w in line.words]).strip()
                    if text:
                        lines.append((text, None, 0.9))
        finally:
            if os.path.exists("._doctr_tmp.jpg"):
                os.remove("._doctr_tmp.jpg")

    rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        if not t:
            continue
        lines.append((t, None, float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0))

    lines_sorted = sorted(lines, key=lambda x: _top_of_bbox(x[1]))
    return lines_sorted

# ------------------------
# GEMINI PARSER
# ------------------------
def gemini_structured_nutrition(raw_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    You are an expert data extraction assistant. Extract all nutrition facts from the provided OCR text.

    Rules:
    - Return ONLY a valid JSON object.
    - Keys: "serving_info" and "nutrition_facts".
    - "serving_info": {{"serving_size": str, "servings_per_container": str or null}}
    - "nutrition_facts": list of {{"nutrient": str, "value": number, "unit": str, "rda_percentage": number or null}}

    Text:
    {raw_text}
    """

    response = model.generate_content(prompt)
    cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
    return json.loads(cleaned_response)

# ------------------------
# MAIN EXECUTION PIPELINE
# ------------------------
def run_pipeline(image_bytes, api_key):
    img, gray = preprocess_image(image_bytes)
    crop = detect_label_region(img, gray)
    crop = correct_perspective(crop)
    lines = ocr_ensemble(crop, use_doctr=True)

    full_text = "\n".join([line[0] for line in lines])
    parsed_data = gemini_structured_nutrition(full_text, api_key)

    # Convert JSON → table rows for Streamlit
    table_rows = []
    for n in parsed_data.get("nutrition_facts", []):
        table_rows.append({
            "Nutrient": n["nutrient"],
            "Quantity": f"{n['value']} {n['unit']}",
            "%RDA": n["rda_percentage"] if n["rda_percentage"] else "-"
        })
    for k, v in parsed_data.get("serving_info", {}).items():
        table_rows.append({"Nutrient": k.replace("_", " ").title(), "Quantity": v, "%RDA": "-"})

    return parsed_data, table_rows
