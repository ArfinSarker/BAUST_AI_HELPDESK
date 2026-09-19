import os
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

# Primary model identifier (Fast, highly capable, and stable)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Intent classifier pattern for instant greetings
GREETING_PATTERNS = [
    r"^(hi|hello|hey|assalamu\s*alaikum|salam|hi\s*there|hlo)[\s!.]*$",
    r"^(kemon\s*achen|kemon\s*acho|how\s*are\s*you|ke\s*tumi|tumi\s*ke|who\s*are\s*you|tumi\s*kara|tumi\s*kon)[\s?!.]*$"
]

def is_greeting(text: str) -> bool:
    clean_text = text.strip().lower()
    return any(re.match(pattern, clean_text) for pattern in GREETING_PATTERNS)


def clean_raw_icon_artifacts(text: str) -> str:
    """Strip OCR icon artifacts where icons were incorrectly converted to stray characters."""
    lines = []
    for line in text.splitlines():
        # Remove single-char icon artifacts before phone numbers (e.g. '9 017...', 'j +880...', '2 013...')
        cleaned = re.sub(r'^[9j2JУyY\)\*\(\#\©\•\-\>\:\;]\s*(?=(\+?880|01[3-9]\d))', '', line.strip())
        # Remove single-char icon artifacts before emails
        cleaned = re.sub(r'^[9j2JУyY\)\*\(\#\©\•\-\>\:\;]\s*(?=[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', '', cleaned)
        # Remove snipping tool or view profile noise
        if re.match(r'^(View Profile|• View Profile|Snipping Tool|Screenshot copied|Automatically saved).*', cleaned, re.IGNORECASE):
            continue
        lines.append(cleaned)
    return "\n".join(lines)


def structure_extracted_text(raw_text: str) -> str:
    """
    Pass raw OCR/PDF text through Gemini to clean noise, remove icon artifacts,
    and organize strictly into clear Markdown sections.
    """
    if not raw_text or len(raw_text.strip()) < 10:
        return raw_text

    cleaned_pretext = clean_raw_icon_artifacts(raw_text)

    prompt = f"""
You are an expert university document digitization and data organization assistant for Bangladesh Army University of Science and Technology (BAUST).

Your task is to take raw pasted or extracted text (which may be unorganized, messy, or copied from websites/notices) and organize it into clean, high-precision, beautifully structured Markdown.

MANDATORY RULES & COMPREHENSIVE CATEGORIZATION:
1. **CLEAN & REMOVE NOISE**:
   - Strip any remnant icon garbage characters ('9', 'j', '2', 'У', ')') before phone numbers or emails.
   - Remove duplicate lines, UI buttons ("View Profile", "Click here"), and website artifacts.

2. **RESIDENTIAL HALLS & HOSTEL ADMINISTRATION**:
   If the document contains hall, hostel, or residential information:
   - Create clear headers for each Hall (e.g. ## Hall Name (Boys' / Girls' / Female Hostel)).
   - State the **Provost** (Name, Designation, Dept, Contact/Phone, Email).
   - List all **Assistant Provosts** with their Department, Phone, Email, and status.
   - Clearly state the **Total Count of Assistant Provosts** (e.g., "Total Assistant Provosts in Girls' Hall: 3").
   - Include Hall Rules, Seat Capacity, Dining/Staff, and Helpline numbers.

3. **FACULTY & DEPARTMENT DIRECTORIES**:
   If the document contains faculty or teacher lists:
   - ## Department & Head / Dean Details
   - ## Current / Active Faculty Members (Name, Designation, Dept, Phone, Email, Qualifications, Status: Active)
   - ## Faculty Members on Study Leave (Mention university/degree pursuing)
   - ## Ex-Faculty / Former Faculty Members (Former teachers)
   - ## Adjunct Faculty Members
   - ## Teaching Assistants & Lab Technical Staff

4. **ADMINISTRATIVE & PROCTORIAL BODIES**:
   If about university administration:
   - Clearly list Proctor, Assistant Proctors, Student Welfare, Registrar, Exam Controller with designations and phone numbers.

5. **FOR GENERAL NOTICES, ADMISSIONS & CIRCULARS**:
   - Structure logically using headings (## Summary, ## Eligibility, ## Important Dates, ## Fee Structure, ## Application Procedure, ## Helplines & Contact).
   - Use Markdown tables for data comparison or fees.

Output ONLY the structured, categorized Markdown text.

RAW INPUT TEXT:
{cleaned_pretext}
"""
    models_to_try = [
        DEFAULT_MODEL,
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash"
    ]
    unique_models = list(dict.fromkeys(models_to_try))

    for model_name in unique_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.1}
            )
            if response and response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            print(f"[Gemini Structuring Error with {model_name}]: {e}")
            continue

    return cleaned_pretext


def find_matching_knowledge_file(new_text: str, knowledge_folder: str) -> tuple[str, str]:
    """
    Intelligently determines if new text belongs to an existing knowledge base file or is a new topic.
    Returns: (matched_filename or None, suggested_snake_case_stem)
    """
    if not os.path.exists(knowledge_folder):
        return None, "document"

    existing_files = [f for f in sorted(os.listdir(knowledge_folder)) if f.endswith(".txt")]
    if not existing_files:
        return None, "document"

    file_summaries = []
    for f in existing_files:
        path = os.path.join(knowledge_folder, f)
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                snippet = file_obj.read()[:250].replace("\n", " ").strip()
                file_summaries.append(f"- {f}: {snippet}")
        except Exception:
            continue

    if not file_summaries:
        return None, "document"

    files_str = "\n".join(file_summaries)
    sample_new_snippet = new_text[:1200]

    prompt = f"""
You are the Knowledge Base Architect for Bangladesh Army University of Science and Technology (BAUST).

EXISTING KNOWLEDGE FILES:
{files_str}

NEWLY INGESTED CONTENT:
{sample_new_snippet}

TASK:
Determine if this new content belongs to one of the EXISTING files (as an update, addition, link, notice, or revision), OR if it represents a completely NEW separate topic.

DECISION CRITERIA:
- If it is about admissions, forms, application links, deadlines -> match existing admission file.
- If it is about CSE teachers, faculty updates, new teachers, leave info -> match cse_faculty_members.txt.
- If it is about Halls, Provosts, Assistant Provosts, hostel facilities -> match hall_administration.txt or residential_facilities.txt.
- If it is about campus wifi, internet -> match campus_wifi.txt.
- Only create NEW if it is a genuinely distinct category (e.g. Bus transport, Blood Donation Club, Library).

OUTPUT FORMAT:
If MATCH:
MATCH: <exact_existing_filename>

If NEW:
NEW: <suggested_snake_case_name_without_extension>
"""
    models_to_try = [
        DEFAULT_MODEL,
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash"
    ]
    unique_models = list(dict.fromkeys(models_to_try))

    for model_name in unique_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.0}
            )
            if response and response.text:
                resp_text = response.text.strip()
                match = re.search(r'MATCH:\s*([a-zA-Z0-9_\-\.]+)', resp_text, re.IGNORECASE)
                if match:
                    matched_file = match.group(1).strip()
                    if matched_file in existing_files:
                        return matched_file, os.path.splitext(matched_file)[0]

                new_match = re.search(r'NEW:\s*([a-zA-Z0-9_\-]+)', resp_text, re.IGNORECASE)
                if new_match:
                    stem = new_match.group(1).strip().lower()
                    return None, stem
        except Exception as e:
            print(f"[Topic Matcher Error with {model_name}]: {e}")
            continue

    return None, "document"


def merge_into_existing_document(existing_text: str, new_text: str) -> str:
    """
    Intelligently merges new information (links, updates, additional sections) into an existing document
    without creating duplicates and preserving clean Markdown hierarchy.
    """
    if not existing_text or not existing_text.strip():
        return new_text
    if not new_text or not new_text.strip():
        return existing_text

    prompt = f"""
You are an expert Knowledge Base Editor for Bangladesh Army University of Science and Technology (BAUST).

Your task is to merge the NEW information into the EXISTING document intelligently.

STRICT MERGING RULES:
1. **NO REDUNDANCY / DUPLICATION**:
   - Do NOT duplicate existing headings, paragraphs, or lists.
   - If the new content updates existing facts (e.g., a new link, updated contact number, new deadline, or a new person), integrate or update them in the appropriate section.
2. **PRESERVE ALL PRE-EXISTING DATA**:
   - Do NOT remove any existing verified facts, faculty details, or notices unless explicitly replaced.
3. **SEAMLESS STRUCTURE**:
   - Maintain clean Markdown formatting (#, ##, ###, bullet lists, tables).

Output ONLY the complete, updated merged Markdown document.

EXISTING DOCUMENT:
{existing_text}

NEW INFORMATION TO INTEGRATE:
{new_text}
"""
    models_to_try = [
        DEFAULT_MODEL,
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash"
    ]
    unique_models = list(dict.fromkeys(models_to_try))

    for model_name in unique_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.1}
            )
            if response and response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            print(f"[Document Merger Error with {model_name}]: {e}")
            continue

    # Fallback: append cleanly
    return f"{existing_text.strip()}\n\n---\n\n## Additional Updates\n{new_text.strip()}"


def generate_answer(context: str, question: str, history: list = None) -> str:
    """
    Generate a high-speed, structured, ChatGPT-style answer using Gemini.
    Combines BAUST RAG knowledge with full ChatGPT general intelligence.
    """
    # 1. Fast path for basic pure greetings
    if is_greeting(question):
        return (
            "হ্যালো! আমি **BAUST (Bangladesh Army University of Science and Technology), Saidpur**-এর অফিসিয়াল AI সহায়ক। 🎓\n\n"
            "আমি ChatGPT-এর মতোই যেকোনো সাধারণ প্রশ্ন, গণিত, কোডিং, বিজ্ঞান, লেখার কাজ এবং বিশেষ করে **BAUST-এর ভর্তি, বিভাগ, শিক্ষক, ফি ও অন্যান্য সকল তথ্যে** সাহায্য করতে পারি।\n\n"
            "আপনাকে কীভাবে সাহায্য করতে পারি?"
        )

    history_str = ""
    if history and isinstance(history, list):
        recent_history = history[-6:]
        history_lines = []
        for msg in recent_history:
            if isinstance(msg, dict) and msg.get("content"):
                role = "User" if msg.get("role") in ["user", "student"] else "Assistant"
                history_lines.append(f"{role}: {str(msg.get('content')).strip()}")
        
        if history_lines:
            history_str = "\nRECENT CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n"

    system_instruction = """
You are an advanced, extremely capable, friendly AI Assistant inspired by ChatGPT, representing Bangladesh Army University of Science and Technology (BAUST), Saidpur Cantonment.

CORE BEHAVIOR & RULES:
1. **ANSWER ANY QUESTION (UNIVERSAL CHATGPT INTELLIGENCE)**:
   - You can answer ANY question asked by the user: general knowledge, math, science, Python/C++/Java/Web programming, essays, translations, physics, logic, current affairs, philosophy, life advice, or casual chatting.
   - NEVER say "I don't have access to this information", "I only answer university questions", or "No record found" for general queries. Answer immediately, thoroughly, and intelligently just like ChatGPT!

2. **ACCURATE BAUST UNIVERSITY EXPERT**:
   - Whenever the question is about BAUST (e.g., admissions, faculties, CSE/EEE/ME/CE/IPE/BBA/English, fees, eligibility GPA, faculty members like Dr. Nakib Hayat, Dr. Jahangir, Al-Hasan, Chancellor, VC, contact numbers, deadlines):
     * CAREFULLY search and prioritize the provided "OFFICIAL BAUST UNIVERSITY RECORDS".
     * Format faculty details cleanly: Designation, Department, Contact (Email/Phone), Education, Status.
     * If the information is in the BAUST records, give precise and complete facts.
     * If an exact detail is not mentioned in the records, provide accurate general context and the official BAUST Helpline: Phone: **01769675588**, **01769675589**, Email: **admission@baust.edu.bd**, Web: **www.baust.edu.bd**.

3. **LANGUAGE & TONE**:
   - If the user asks in Bengali (বাংলা) or Banglish, reply in warm, natural, fluent Bengali.
   - If the user asks in English, reply in articulate, professional English.

4. **PRESENTATION & MARKDOWN**:
   - Use beautiful Markdown: bold text for key points, clean bullet lists, tables for data, and code blocks for programming.
"""

    prompt = f"""{system_instruction}

{history_str}
OFFICIAL BAUST UNIVERSITY RECORDS (Search Results from Knowledge Base):
{context if context and context.strip() else "(No specific internal BAUST document found for this query - use your general intelligence)"}

USER QUESTION:
{question}

ASSISTANT RESPONSE:"""

    # Active Gemini models with fast fallback order
    models_to_try = [DEFAULT_MODEL, "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]
    unique_models = list(dict.fromkeys(models_to_try))

    for model_name in unique_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.3}
            )

            if response and response.text:
                return response.text.strip()

        except Exception as e:
            print(f"[Gemini API Error with {model_name}]: {e}")
            continue

    return (
        "সার্ভারে সাময়িক সমস্যা হচ্ছে। অনুগ্রহ করে পুনরায় প্রশ্নটি করুন অথবা BAUST হেল্পলাইনে সরাসরি যোগাযোগ করুন: **01769675588**, **01769675589**"
    )