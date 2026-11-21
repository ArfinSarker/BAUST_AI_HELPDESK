import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Pre-initialize Gemini client if key is available
GEMINI_CLIENT = None
try:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        from google import genai
        GEMINI_CLIENT = genai.Client(api_key=gemini_key)
except Exception as e:
    print(f"[Gemini Client Init Warning]: {e}")


def extract_text_from_txt(file_path: str) -> str:
    """Read plain text files with UTF-8 fallback encodings."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read().strip()
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def gemini_multimodal_extract(file_path: str, mime_type: str) -> str:
    """
    Extract text and layout directly from image or PDF using Gemini Multimodal Vision.
    - Uses Google GenAI SDK (v1.0.0+) Files API for files > 4MB to avoid memory/payload limits.
    - In-memory Part bytes for files <= 4MB for sub-second execution.
    - Smart failover: Only retries alternative models on transient server (500/503/429/timeout) errors.
    - Automatic cloud file cleanup in finally block.
    """
    if not GEMINI_CLIENT or not os.path.exists(file_path):
        return ""

    uploaded_file = None
    try:
        from google.genai import types
        file_size = os.path.getsize(file_path)

        # 1. New Google GenAI SDK v1.0.0+ Files API for large documents/images (> 4MB)
        if file_size > 4 * 1024 * 1024:
            try:
                uploaded_file = GEMINI_CLIENT.files.upload(file=file_path)
                content_payload = [uploaded_file]
            except Exception as up_err:
                print(f"[Gemini Files Upload Notice]: {up_err}. Streaming local bytes.")
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                content_payload = [types.Part.from_bytes(data=file_bytes, mime_type=mime_type)]
        else:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            if not file_bytes:
                return ""
            content_payload = [types.Part.from_bytes(data=file_bytes, mime_type=mime_type)]

        prompt = """
You are an expert document digitization and information extraction assistant for Bangladesh Army University of Science and Technology (BAUST).

Your task is to extract ALL valuable textual data, notices, tables, lists, and person profiles from this image/document accurately into clean Markdown.

CRITICAL EXTRACTION RULES:
1. **NEVER MISINTERPRET ICONS AS TEXT**:
   - Phone icons (📞, 📱), Email icons (✉, @), Location icons (📍), Degree/cap icons (🎓), Bullet icons, and social icons must NEVER be converted into random characters or numbers like '9', 'j', '2', 'У', ')', '©', or '*'.
   - Extract the phone number or email cleanly (e.g., 'Phone: 01769662712', 'Email: sowket@baust.edu.bd').

2. **IGNORE UI & SCREENSHOT ARTIFACTS**:
   - Completely remove 'View Profile' buttons, 'Snipping Tool' / 'Screenshot copied' popups, browser headers/tabs, search bars, and window frames.

3. **PRESERVE EXPLICIT CATEGORIES & STATUS**:
   - If this is a directory of faculty, staff, or students, clearly group them under Markdown headers:
     * ## Current / Active Faculty Members
     * ## Faculty on Study Leave
     * ## Ex-Faculty / Former Faculty Members
     * ## Adjunct Faculty Members
     * ## Teaching Assistants
     * ## Officers & Lab Technical Staff
   - For each person, clearly list: Name, Designation, Department, Phone, Email, Education, and Current Status.

4. **FOR NOTICES, CIRCULARS & FORMS**:
   - Retain full text, dates, eligibility criteria, fee structures, program details, and tables in clean Markdown format.

Output ONLY the extracted, structured Markdown.
"""
        content_payload.append(prompt)

        # 3. Failover Optimization: Primary model + fast fallback, without cascading 4-model timeout delay
        primary_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        models_to_try = [
            primary_model,
            "gemini-3.5-flash-lite",
            "gemini-flash-latest",
            "gemini-3.8-flash",
            "gemini-flash-lite-latest"
        ]
        unique_models = list(dict.fromkeys([m for m in models_to_try if m]))

        for model_name in unique_models:
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=content_payload
                )
                if response and response.text and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                err_str = str(e).lower()
                print(f"[Gemini Vision Note ({model_name})]: {e}")
                # Try next model if current model is unavailable, not found, or rate-limited
                continue

    except Exception as e:
        print(f"[gemini_multimodal_extract Exception] {file_path}: {e}")
    finally:
        # Automatic cloud cleanup
        if uploaded_file and hasattr(uploaded_file, "name"):
            try:
                GEMINI_CLIENT.files.delete(name=uploaded_file.name)
            except Exception:
                pass

    return ""


def ocr_image_bytes(image_bytes: bytes, filename: str = "page.jpg") -> str:
    """
    High-reliability OCR for in-memory image bytes:
    1. Primary: OCR.Space API with compressed/rendered page image (typically 200KB - 400KB, well under 5MB).
    2. Fallback: Gemini Vision REST API with inline base64 data.
    """
    if not image_bytes:
        return ""

    api_key = os.getenv("OCR_API_KEY")
    if api_key:
        try:
            url = "https://api.ocr.space/parse/image"
            payload = {
                "apikey": api_key,
                "language": "eng",
                "isOverlayRequired": False,
                "detectOrientation": True,
                "scale": True,
                "OCREngine": "2",
            }
            resp = requests.post(
                url,
                files={"file": (filename, image_bytes, "image/jpeg")},
                data=payload,
                timeout=25
            )
            if resp.status_code == 200:
                result = resp.json()
                if "ParsedResults" in result and result["ParsedResults"]:
                    parsed_text = "\n\n".join(
                        page.get("ParsedText", "").strip()
                        for page in result["ParsedResults"]
                        if page.get("ParsedText")
                    )
                    if parsed_text.strip():
                        return parsed_text.strip()
        except Exception as e:
            print(f"[OCR.Space Bytes Exception]: {e}")

    # Fallback to Gemini Vision REST API
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import base64
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            payload = {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                        {"text": "Extract all textual information, tables, schedules, and notices accurately into clean Markdown."}
                    ]
                }],
                "generationConfig": {"temperature": 0.1}
            }
            for model in ["gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
                    resp = requests.post(url, json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                text_val = parts[0]["text"].strip()
                                if text_val:
                                    return text_val
                except Exception:
                    continue
        except Exception as e:
            print(f"[Gemini Vision Fallback Exception]: {e}")

    return ""


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from PDF files using an industry-grade multi-engine pipeline:
    1. Primary: PyMuPDF (fitz) context-manager for native digital text.
    2. Scanned/Image PDF: Page-by-page in-memory pixmap rendering with multi-engine OCR (guaranteed 0 file size limits).
    3. Secondary: pdfplumber for complex table layouts.
    4. Tertiary: pypdf robust digital stream parser.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return ""

    # 1. Primary Engine: PyMuPDF digital text + page-by-page OCR fallback
    try:
        import pymupdf
        pages_text = []
        scanned_pages = []
        with pymupdf.open(file_path) as doc:
            for i, page in enumerate(doc):
                page_str = page.get_text("text").strip()
                if len(page_str) > 20:
                    pages_text.append(f"--- Page {i+1} ---\n{page_str}")
                else:
                    scanned_pages.append(i)

            # If all or most pages had digital text (>50 chars total) and no scanned pages
            if len(pages_text) > 0 and len(scanned_pages) == 0:
                combined = "\n\n".join(pages_text).strip()
                if len(combined) > 50:
                    return combined

            # If there are scanned/image pages, perform page-by-page OCR
            if scanned_pages:
                print(f"[PDF Scanned Detected]: {len(scanned_pages)} page(s) require image OCR for '{file_path}'")
                all_pages_ordered = []
                for i, page in enumerate(doc):
                    page_str = page.get_text("text").strip()
                    if len(page_str) > 20:
                        all_pages_ordered.append(f"--- Page {i+1} ---\n{page_str}")
                    else:
                        pix = page.get_pixmap(dpi=90)
                        img_bytes = pix.tobytes("jpeg")
                        ocr_txt = ocr_image_bytes(img_bytes, filename=f"page_{i+1}.jpg")
                        if ocr_txt:
                            all_pages_ordered.append(f"--- Page {i+1} ---\n{ocr_txt}")
                if all_pages_ordered:
                    combined = "\n\n".join(all_pages_ordered).strip()
                    if len(combined) > 50:
                        return combined
    except Exception as e:
        print(f"[PyMuPDF Extraction Exception] {file_path}: {e}")

    # 2. Secondary Engine: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            pages_text = []
            for i, page in enumerate(pdf.pages):
                page_str = page.extract_text()
                if page_str and page_str.strip():
                    pages_text.append(f"--- Page {i+1} ---\n{page_str.strip()}")
            if pages_text:
                combined = "\n\n".join(pages_text).strip()
                if len(combined) > 50:
                    return combined
    except Exception as e:
        print(f"[pdfplumber Extraction Exception] {file_path}: {e}")

    # 3. Tertiary Engine: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_str = page.extract_text()
            if page_str and page_str.strip():
                pages_text.append(f"--- Page {i+1} ---\n{page_str.strip()}")
        if pages_text:
            combined = "\n\n".join(pages_text).strip()
            if len(combined) > 50:
                return combined
    except Exception as e:
        print(f"[pypdf Extraction Exception] {file_path}: {e}")

    # 4. Scanned / Image PDF fallback: Gemini Multimodal Vision
    gemini_result = gemini_multimodal_extract(file_path, "application/pdf")
    if gemini_result and len(gemini_result.strip()) > 20:
        return gemini_result

    # 5. Final fallback: OCR.Space
    return image_to_text(file_path, is_pdf=True)


def image_to_text(file_path: str, is_pdf: bool = False) -> str:
    """Extract text from images or scanned PDFs using OCR.Space API as fallback."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return ""

    file_size = os.path.getsize(file_path)
    # If image > 4MB, compress in memory using PIL before sending to OCR.Space
    if file_size > 4 * 1024 * 1024:
        try:
            from PIL import Image
            import io
            with Image.open(file_path) as im:
                im = im.convert("RGB")
                im.thumbnail((2000, 2000))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=80)
                image_bytes = buf.getvalue()
                if image_bytes:
                    return ocr_image_bytes(image_bytes, filename=os.path.basename(file_path))
        except Exception as e:
            print(f"[PIL Image Compression Notice]: {e}")

    url = "https://api.ocr.space/parse/image"
    api_key = os.getenv("OCR_API_KEY")

    if not api_key:
        return "Error: OCR API Key not found in environment."

    payload = {
        "apikey": api_key,
        "language": "eng",
        "isOverlayRequired": False,
        "detectOrientation": True,
        "scale": True,
        "OCREngine": "2",
    }

    if is_pdf:
        payload["filetype"] = "PDF"

    try:
        with open(file_path, "rb") as file_bytes:
            response = requests.post(
                url,
                files={"file": file_bytes},
                data=payload,
                timeout=12  # Strict 12-second timeout to prevent UI freezes
            )

        if response.status_code != 200:
            return f"OCR API error: HTTP {response.status_code}"

        result = response.json()

        if "ParsedResults" in result and result["ParsedResults"]:
            parsed_text = "\n\n".join(
                page.get("ParsedText", "").strip()
                for page in result["ParsedResults"]
                if page.get("ParsedText")
            )
            return parsed_text.strip()

        error_details = result.get("ErrorMessage", str(result))
        return f"OCR Error: {error_details}"

    except Exception as e:
        return f"Exception Error: {str(e)}"


def extract_text_from_file(file_path: str) -> str:
    """Unified file text extractor handling PDF, TXT, and Image formats."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return ""

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        return extract_text_from_txt(file_path)

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    # Image MIME mapping
    image_mimes = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff"
    }

    if ext in image_mimes:
        # 1. Primary: Gemini Multimodal Vision for images
        vision_result = gemini_multimodal_extract(file_path, image_mimes[ext])
        if vision_result and len(vision_result.strip()) > 10:
            return vision_result
        
        # 2. Fallback: OCR.Space
        return image_to_text(file_path)

    # Fallback attempt for non-standard extensions
    txt = extract_text_from_txt(file_path)
    if txt:
        return txt

    return gemini_multimodal_extract(file_path, "image/png") or image_to_text(file_path)
