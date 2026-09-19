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
    Eliminates decorative icon artifacts ('9', 'j', '2'), removes UI junk (Snipping tool, 'View Profile'),
    and outputs clean, structured Markdown sections.
    """
    if not GEMINI_CLIENT:
        return ""

    try:
        from google.genai import types
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        if not file_bytes:
            return ""

        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

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

        models_to_try = [
            os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.6-flash"
        ]
        unique_models = list(dict.fromkeys(models_to_try))

        for model_name in unique_models:
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=[part, prompt]
                )
                if response and response.text and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                print(f"[Gemini Vision Extraction Error with {model_name}]: {e}")
                continue

    except Exception as e:
        print(f"[gemini_multimodal_extract Exception] {file_path}: {e}")

    return ""


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from PDF files.
    1. First attempts high-fidelity digital text extraction across ALL pages using pypdf.
    2. If the PDF contains no text (i.e. scanned images/handouts), falls back to Gemini Multimodal Vision.
    3. Final fallback to OCR.Space.
    """
    # 1. Primary: Extract all digital text from all pages using pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                extracted_pages.append(f"--- Page {i+1} ---\n{page_text.strip()}")

        full_pdf_text = "\n\n".join(extracted_pages).strip()
        # If substantial text was extracted, return it (lossless and handles 100+ pages)
        if len(full_pdf_text) > 80:
            return full_pdf_text
    except Exception as e:
        print(f"[pypdf Extraction Exception] {file_path}: {e}")

    # 2. Scanned / image PDF fallback: Gemini Multimodal Vision
    gemini_result = gemini_multimodal_extract(file_path, "application/pdf")
    if gemini_result and len(gemini_result.strip()) > 20:
        return gemini_result

    # 3. Last fallback: OCR.Space
    return image_to_text(file_path, is_pdf=True)


def image_to_text(file_path: str, is_pdf: bool = False) -> str:
    """Extract text from images or scanned PDFs using OCR.Space API as fallback."""
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
                timeout=30
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
    if not os.path.exists(file_path):
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