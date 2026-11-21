import os
import re
import time
import json
import hashlib
import threading
from collections import OrderedDict
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from retrieval.rag_search import sanitize_contact_markdown

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Primary model identifier (Fast, highly capable, and stable)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_last_working_model = DEFAULT_MODEL

# Persistent HTTP session with robust connection pooling & low latency socket reuse
_http_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=25, pool_maxsize=50, max_retries=1)
_http_session.mount("https://", _adapter)
_http_session.mount("http://", _adapter)

# =====================================================
# DYNAMIC UNIVERSITY CONFIGURATION (NO HARDCODING)
# =====================================================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "university_config.json")
UNIVERSITY_CONFIG = {}
try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as _cf:
            UNIVERSITY_CONFIG = json.load(_cf)
except Exception as _e:
    print(f"[Config Load Notice]: {_e}")

# =====================================================
# PRE-COMPILED REGEX PATTERNS (HIGH PERFORMANCE)
# =====================================================
RE_PUNCTUATION = re.compile(r'[!?,.\'"]+')
RE_CONSECUTIVE_CHARS = re.compile(r'([a-z])\1+')
RE_MARKDOWN_LINKS = re.compile(r'`(\[[^\]]+\]\([^\)]+\))`')
RE_MARKDOWN_FENCES = re.compile(r'^```(?:markdown|json)?\s*', re.IGNORECASE)
RE_CLOSING_FENCES = re.compile(r'\s*```$', re.IGNORECASE)
RE_CLEAN_PHONE_PREFIX = re.compile(r'^[9j2JУyY\)\*\(\#\©\•\-\>\:\;]\s*(?=(\+?880|01[3-9]\d))')
RE_CLEAN_EMAIL_PREFIX = re.compile(r'^[9j2JУyY\)\*\(\#\©\•\-\>\:\;]\s*(?=[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
RE_NOISE_LINES = re.compile(r'^(View Profile|• View Profile|Snipping Tool|Screenshot copied|Automatically saved).*', re.IGNORECASE)
RE_REASONING_HEADERS = re.compile(r'(?:### Response:|### Answer:|Final Answer:|\*\*উত্তর:\*\*|\*\*Answer:\*\*)\s*(.*)', re.DOTALL | re.IGNORECASE)
RE_REASONING_LINES = re.compile(r'^\s*[\*\-]\s*(User Question|Input|Language|Context|Constraint|Goal|Greeting|Acknowledgment|Combined|Addresses|Reasoning):', re.IGNORECASE)
RE_THOUGHT_STEPS = re.compile(r'^(1\.|2\.|3\.)\s+.*?(Input|Question|Constraint|Goal)', re.IGNORECASE)
RE_JSON_ARRAY = re.compile(r'\[\s*\{.*\}\s*\]', re.DOTALL)

# =====================================================
# O(1) LRU IN-MEMORY FAST CACHE (WITH MD5 HASH KEYS)
# =====================================================
_RESPONSE_CACHE = OrderedDict()
_CACHE_TTL_SECONDS = 3600  # 1 hour TTL
_MAX_CACHE_SIZE = 1000
_CACHE_LOCK = threading.Lock()


def build_cache_key(question: str, context: str = "") -> str:
    """Generate collision-proof cache key using MD5 content hash."""
    q_norm = question.strip().lower()
    if not context:
        return q_norm
    ctx_hash = hashlib.md5(context.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{q_norm}_{ctx_hash}"


def get_cached_response(key: str) -> str:
    """Retrieve response from cache with O(1) LRU order updating."""
    norm_key = key.strip().lower()
    with _CACHE_LOCK:
        if norm_key in _RESPONSE_CACHE:
            entry = _RESPONSE_CACHE[norm_key]
            if time.time() - entry["time"] < _CACHE_TTL_SECONDS:
                _RESPONSE_CACHE.move_to_end(norm_key)  # O(1) mark as recently used
                return entry["answer"]
            else:
                del _RESPONSE_CACHE[norm_key]
    return None


def set_cached_response(key: str, answer: str):
    """Store response into memory cache with O(1) LRU eviction."""
    if not key or not answer:
        return
    # Guard against leaking internal prompt instructions or raw wrappers
    if "CRITICAL INSTRUCTION" in answer or "Summary of Relevant Information" in answer:
        return
    # Guard against abruptly truncated responses
    ans_clean = answer.strip()
    if len(ans_clean) < 80 and not any(ans_clean.endswith(p) for p in [".", "!", "?", "।", "ঃ", "]", ")", ":"]):
        return
    if "multiple" in key.lower() or "student" in key.lower() or "seat" in key.lower():
        if len(ans_clean) < 120:
            return

    norm_key = key.strip().lower()
    with _CACHE_LOCK:
        if norm_key in _RESPONSE_CACHE:
            _RESPONSE_CACHE.move_to_end(norm_key)
        elif len(_RESPONSE_CACHE) >= _MAX_CACHE_SIZE:
            _RESPONSE_CACHE.popitem(last=False)  # O(1) pop oldest LRU entry
        _RESPONSE_CACHE[norm_key] = {
            "answer": answer,
            "time": time.time()
        }


def normalize_text_for_intent(text: str) -> str:
    """Normalize text using pre-compiled regex for maximum throughput."""
    if not text:
        return ""
    t = text.lower().strip()
    t = RE_PUNCTUATION.sub(' ', t)
    t = RE_CONSECUTIVE_CHARS.sub(r'\1', t)
    return " ".join(t.split())


def is_greeting(text: str) -> bool:
    """Accurately detects greetings, identity queries, and chit-chat with phonetic tolerance."""
    if not text:
        return False
    raw_t = text.strip().lower()
    norm_t = normalize_text_for_intent(text)
    
    # 1. Identity questions
    if any(k in norm_t for k in ["tomar nam", "apnar nam", "tumi ke", "apni ke", "ke tumi", "ke apni", "your name", "who are you", "who r u", "who made you", "tomar porichoy"]) or any(k in raw_t for k in ["তোমার নাম", "তুমি কে", "আপনার নাম", "কে তুমি"]):
        return True
        
    # 2. How are you
    if any(k in norm_t for k in ["kemon acho", "kemon achen", "kmn acho", "kmn asen", "how are you", "how r u"]) or any(k in raw_t for k in ["কেমন আছো", "কেমন আছেন"]):
        return True
        
    # 3. What can you do
    if any(k in norm_t for k in ["what can you do", "ki korte paro", "ki sahajo", "ki sahajjo"]) or any(k in raw_t for k in ["কি করতে পারো", "সাহায্য"]):
        return True
        
    # 4. Thank you
    if any(k in norm_t for k in ["thank you", "thanks", "thx", "dhonobad", "dhonnobad", "shukriya"]) or any(k in raw_t for k in ["ধন্যবাদ"]):
        return True
        
    # 5. Bye / Goodbye
    if any(k in norm_t for k in ["bye", "goodbye", "alah hafez", "allah hafez", "tata"]) or any(k in raw_t for k in ["বিদায়", "বিদায়"]):
        return True
        
    # 6. Greetings
    if norm_t in ["hi", "helo", "hello", "hey", "asalamu alaikum", "assalamu alaikum", "salam", "hi there", "hlo", "ola"]:
        return True
        
    return False


def get_conversational_response(text: str) -> str:
    """Returns direct, language-matched conversational responses."""
    raw_t = text.strip().lower()
    norm_t = normalize_text_for_intent(text)

    # 1. Name & Identity
    if any(k in norm_t for k in ["what is your name", "what's your name", "your name", "who are you", "who r u", "who made you"]):
        return "I am the official AI Assistant of Bangladesh Army University of Science and Technology (BAUST), Saidpur. How can I help you today?"

    if any(k in norm_t for k in ["tomar nam", "apnar nam", "tumi ke", "apni ke", "ke tumi", "ke apni", "tomar porichoy"]) or any(k in raw_t for k in ["তোমার নাম", "তুমি কে", "আপনার নাম", "কে তুমি"]):
        return "আমি বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুরের অফিসিয়াল AI সহকারী। আপনাকে কীভাবে সাহায্য করতে পারি?"

    # 2. How are you
    if any(k in norm_t for k in ["how are you", "how r u"]):
        return "I am doing well, thank you! How can I assist you with BAUST information today?"

    if any(k in norm_t for k in ["kemon acho", "kemon achen", "kmn acho", "kmn asen"]) or any(k in raw_t for k in ["কেমন আছো", "কেমন আছেন"]):
        return "আলহামদুলিল্লাহ, আমি ভালো আছি। BAUST সম্পর্কিত যেকোনো তথ্যে আপনাকে সাহায্য করতে প্রস্তুত।"

    # 3. What can you do
    if any(k in norm_t for k in ["what can you do"]):
        return "I can provide accurate information about BAUST admissions, faculty directories, tuition fees and waivers, examination policies, residential halls, and official contacts."

    if any(k in norm_t for k in ["ki korte paro", "ki sahajo", "ki sahajjo"]) or any(k in raw_t for k in ["কি করতে পারো", "সাহায্য"]):
        return "আমি BAUST-এর ভর্তি প্রক্রিয়া, অনুষদ ও শিক্ষকবৃন্দ, ফি ও ওয়েভার, পরীক্ষার নীতিমালা, আবাসিক হল এবং অফিসিয়াল যোগাযোগের তথ্যে সাহায্য করতে পারি।"

    # 4. Thank you
    if any(k in norm_t for k in ["thank you", "thanks", "thx"]):
        return "You're very welcome! Let me know if you have more questions about BAUST."

    if any(k in norm_t for k in ["dhonobad", "dhonnobad", "shukriya"]) or any(k in raw_t for k in ["ধন্যবাদ"]):
        return "আপনাকে অনেক ধন্যবাদ! BAUST সম্পর্কিত যেকোনো তথ্যের জন্য আমি সবসময় আছি।"

    # 5. Greetings
    if any(k in norm_t for k in ["hi", "helo", "hello", "hey", "hi there"]):
        return "Hello! I am the official AI Assistant for BAUST, Saidpur. How can I help you with university information today?"

    return (
        "হ্যালো! আমি **BAUST (Bangladesh Army University of Science and Technology), Saidpur**-এর অফিসিয়াল AI সহায়ক। 🎓\n\n"
        "আমি **BAUST-এর ভর্তি, অনুষদ ও বিভাগসমূহ, শিক্ষকবৃন্দ, ফি কাঠামো ও ওয়েভার, পরীক্ষার নীতিমালা, আবাসিক হল ও বিশ্ববিদ্যালয়ের যেকোনো তথ্যে** সাহায্য করতে পারি।\n\n"
        "আপনাকে কীভাবে সাহায্য করতে পারি?"
    )






def clean_llm_response(text: str) -> str:
    """Strip internal chain-of-thought, scratchpad reasoning traces, or formatting artifacts from LLMs."""
    if not text:
        return ""
    
    # If model output contains internal reasoning steps / scratchpad
    if "* User Question:" in text or "### Analysis" in text or "* Input:" in text or "* Constraint:" in text:
        match = RE_REASONING_HEADERS.search(text)
        if match and match.group(1).strip():
            return match.group(1).strip()
        
        lines = []
        skip_thought = True
        for line in text.splitlines():
            if skip_thought:
                if RE_REASONING_LINES.match(line):
                    continue
                if RE_THOUGHT_STEPS.match(line.strip()):
                    continue
                if not line.strip():
                    continue
                skip_thought = False
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        if cleaned:
            text = cleaned

    # Normalize accidental backticks around markdown links: `[text](url)` -> [text](url)
    text = RE_MARKDOWN_LINKS.sub(r'\1', text)

    # Fix common Bengali transliteration spelling errors (e.g. Break -> ব্রেক, not ব্র্যাক)
    text = re.sub(r'সেমিস্টার\s+ব্র্যা?ক\b', 'সেমিস্টার ব্রেক', text)
    text = re.sub(r'মিড\s*টার্ম\s+ব্র্যা?ক\b', 'মিডটার্ম ব্রেক', text)
    text = re.sub(r'ক্লাস\s+ব্র্যা?ক\b', 'ক্লাস ব্রেক', text)
    text = re.sub(r'\bব্র্যাক\b(?=\s*(?:হবে|শুরু|শেষ|চলবে|থাকবে|এর|এ|পর্যন্ত))', 'ব্রেক', text)

    # Fix Saidpur spelling error: সাইদপুর / সায়েদপুর -> সৈয়দপুর
    text = text.replace("সাইদপুর", "সৈয়দপুর").replace("সায়েদপুর", "সৈয়দপুর")

    # Sanitize emails, phones, and degrees format
    text = sanitize_contact_markdown(text)

    return text.strip()


def is_translation_request(text: str) -> tuple[bool, str]:
    """
    Detect if the user is asking to translate or convert the previous answer into English, Bangla, or a shorter form.
    Returns: (is_translation, target_language_or_instruction)
    """
    t = text.strip().lower()
    
    # English conversion
    if any(p in t for p in ["english e", "english te", "in english", "to english", "english version", "translate into english", "translate to english", "ইংলিশে", "ইংরেজিতে"]):
        return True, "Translate/convert the previous answer into clear, natural English."
    
    # Bengali conversion
    if any(p in t for p in ["banglay", "bangla te", "in bangla", "in bengali", "to bangla", "to bengali", "translate into bangla", "translate to bangla", "বাংলায়", "বাংলাতে"]):
        return True, "Translate/convert the previous answer into natural, fluent Bengali (বাংলা)."
    
    # Shorten / Summarize conversion
    if any(p in t for p in ["short e", "choto kore", "choto koro", "short koro", "summarize", "সংক্ষেপে"]):
        return True, "Make the previous answer concise, short, and to the point while retaining the core facts."
    
    return False, ""


def rewrite_query_with_history(question: str, history: list = None) -> str:
    """
    Smart Contextual Query Expansion for follow-ups (e.g. 'contact', 'phone number', 'amar seat kothay').
    Correctly extracts the teacher, person, course, or student roll from the preceding turns
    without latency penalty.
    """
    if not history or not isinstance(history, list) or len(history) < 2:
        return question

    is_trans, _ = is_translation_request(question)
    if is_trans:
        return question

    q_lower = question.strip().lower()
    q_tokens = set(re.findall(r'[a-zA-Zঀ-৿]+', q_lower))

    contact_followup_terms = {
        "contact", "phone", "email", "mail", "number", "no", "call", "hotline", "jogajog", "thaken",
        "যোগাযোগ", "ফোন", "নাম্বার", "নম্বর", "ইমেইল", "মেইল", "কল", "মোবাইল"
    }
    general_followup_terms = {
        "time", "slot", "room", "seat", "somoy", "shomoy", "kobe", "kokhon", "tarikh",
        "date", "when", "building", "kothay", "details", "routine", "class",
        "কবে", "কখন", "সময়", "সময়", "রুম", "সিট", "স্লট", "কোথায়", "রুটিন", "ক্লাস", "বিস্তারিত"
    }

    is_contact_followup = len(q_tokens) <= 3 and bool(q_tokens & contact_followup_terms)
    is_short_followup = is_contact_followup or (len(q_tokens) <= 4 and bool(q_tokens & general_followup_terms))
    has_pronoun = any(p in q_tokens for p in ["tar", "tader", "ota", "oita", "unro", "unar", "his", "her", "their", "it", "its"]) or any(p in q_lower for p in ["তার", "তাদের", "উনার", "সেটার"])

    if not (is_short_followup or has_pronoun):
        return question

    # 1. Search for last teacher / person name in recent history
    last_person = ""
    for msg in reversed(history[-4:]):
        if isinstance(msg, dict):
            content = str(msg.get("content", ""))
            # Check for teacher/officer label in assistant response
            pm = re.search(r'(?:শিক্ষক/কর্মকর্তার নাম|শিক্ষকের নাম|পূর্ণনাম|পূর্ণ নাম|Full Name|Name)[\s\*\:\)]+([A-Za-z\s\.\-]{3,35})', content, re.I)
            if pm:
                cand = pm.group(1).strip()
                if not any(k in cand.lower() for k in ["baust", "student", "roll", "cse", "ipe", "department", "academic"]):
                    last_person = cand
                    break

    # 2. Search for student roll if seat/routine query
    last_roll = ""
    for msg in reversed(history[-4:]):
        if isinstance(msg, dict):
            rm = re.search(r'\b(08\d{6,14})\b', str(msg.get("content", "")))
            if rm:
                last_roll = rm.group(1)
                break

    if last_person and is_contact_followup:
        return f"{last_person} contact phone email"

    if last_person and is_short_followup and not last_roll:
        return f"{last_person} {question}"

    if last_roll and (is_short_followup or has_pronoun) and not last_person:
        return f"{last_roll} {question}"

    # Fallback to last user query if nothing specific extracted
    for msg in reversed(history[-4:]):
        if isinstance(msg, dict) and msg.get("role") in ["user", "student"]:
            u_text = str(msg.get("content", "")).strip()
            if u_text and len(u_text) > 3:
                return f"{u_text} {question}"

    return question


def call_gemini_generate(prompt: str, temperature: float = 0.2, preferred_model: str = None, response_mime_type: str = None) -> str:
    """
    Ultra-fast, robust Gemini caller with persistent connection pooling,
    native JSON structured output support, and deterministic low-latency fallbacks.
    """
    global _last_working_model
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return ""

    env_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    candidate_models = [
        "gemini-2.5-flash",
        preferred_model,
        _last_working_model,
        env_model,
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    unique_models = [m for m in dict.fromkeys(candidate_models) if m]

    gen_config = {"temperature": temperature}
    if response_mime_type:
        gen_config["responseMimeType"] = response_mime_type

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config
    }

    for model_name in unique_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        try:
            resp = _http_session.post(url, json=payload, timeout=12)
            # Graceful fallback: If an older or unsupported model rejects responseMimeType with 400, retry without it
            if resp.status_code == 400 and response_mime_type:
                fallback_payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature}
                }
                resp = _http_session.post(url, json=fallback_payload, timeout=4)

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        _last_working_model = model_name
                        raw_text = parts[0]["text"].strip()
                        return raw_text if response_mime_type == "application/json" else clean_llm_response(raw_text)
        except Exception:
            continue

    return ""


def clean_raw_icon_artifacts(text: str) -> str:
    """Strip OCR icon artifacts using pre-compiled regex patterns."""
    lines = []
    for line in text.splitlines():
        cleaned = RE_CLEAN_PHONE_PREFIX.sub('', line.strip())
        cleaned = RE_CLEAN_EMAIL_PREFIX.sub('', cleaned)
        if RE_NOISE_LINES.match(cleaned):
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

    if len(cleaned_pretext) > 4000:
        if not cleaned_pretext.strip().startswith("#"):
            lines = [l.strip() for l in cleaned_pretext.splitlines() if l.strip()]
            first_line = lines[0] if lines else "Document"
            if len(first_line) < 80 and not first_line.startswith("---"):
                cleaned_pretext = f"# {first_line}\n\n" + cleaned_pretext
        return cleaned_pretext

    prompt = f"""
You are the Chief Knowledge Base Architect and Data Structuring Specialist for Bangladesh Army University of Science and Technology (BAUST), Saidpur.

Your task is to take raw pasted or extracted document/notice text and convert it into high-precision, beautifully organized, structured Markdown for the AI Knowledge Base.

MANDATORY RULES & CATEGORIZATION:
1. **NOISE & ARTIFACT CLEANING**:
   - Strip remnant icon garbage characters before phone numbers or emails.
   - Remove duplicate lines, UI navigation buttons ("View Profile", "Click here", "Download"), and website artifacts.
   - Preserve all phone numbers, email addresses, dates, and numbers accurately.

2. **FACULTY & TEACHERS DATA**:
   - Explicitly organize teachers into distinct markdown sections:
     * `## Active Faculty Members` (Current active teachers: Name, Designation, Phone, Email, Qualifications)
     * `## Faculty Members on Study Leave` (Teachers currently away for higher studies/PhD)
     * `## Ex-Faculty / Former Faculty Members` (Former teachers)
     * `## Adjunct Faculty Members` (Guest/part-time professors)

3. **ADMISSION, FEES & POLICIES**:
   - Format with bold headings, concise bullet points, eligibility criteria, and fee breakdowns.

4. **OUTPUT FORMAT**:
   - Output ONLY the clean structured Markdown. Do not include markdown code block backticks (```markdown) or meta commentary.

RAW INPUT TEXT:
{cleaned_pretext}
"""
    result = call_gemini_generate(prompt, temperature=0.1)
    if result:
        # Strip code block fences if returned by model
        cleaned_result = re.sub(r'^```(?:markdown)?\s*', '', result.strip())
        cleaned_result = re.sub(r'\s*```$', '', cleaned_result).strip()
        return cleaned_result
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
                content = file_obj.read()
                headings = [line.strip() for line in content.splitlines() if line.strip().startswith("#")]
                headings_summary = " | ".join(headings[:10]) if headings else "General topic"
                snippet = content[:600].replace("\n", " ").strip()
                file_summaries.append(f"- File: `{f}`\n  Sections: {headings_summary}\n  Snippet: {snippet[:250]}")
        except Exception:
            continue

    if not file_summaries:
        return None, "document"

    files_str = "\n\n".join(file_summaries)
    sample_new_snippet = new_text[:2000]

    prompt = f"""
You are the Chief Knowledge Base Architect for Bangladesh Army University of Science and Technology (BAUST).

EXISTING KNOWLEDGE BASE FILES AND THEIR CURRENT SECTIONS:
{files_str}

NEWLY INGESTED CONTENT:
{sample_new_snippet}

TASK:
Determine if this content belongs to an EXISTING file, or requires a NEW file.

OUTPUT FORMAT (STRICT):
If MATCH:
MATCH: <exact_existing_filename_from_list>

If NEW:
NEW: <suggested_snake_case_stem_without_extension>
"""
    resp_text = call_gemini_generate(prompt, temperature=0.0)
    if resp_text:
        match = re.search(r'MATCH:\s*`?([a-zA-Z0-9_\-\.]+\.txt)`?', resp_text, re.IGNORECASE)
        if match:
            matched_file = match.group(1).strip()
            # Guardrail: Never route an Examination Schedule or Routine into a Student Seatplan file
            sample_low = sample_new_snippet.lower()
            is_schedule_doc = any(k in sample_low for k in ["examination schedule", "exam schedule", "slot-a", "slot-b", "slot-c"])
            if is_schedule_doc and "seatplan" in matched_file.lower():
                return ("midterm_examination_schedule_summer_2026.txt", "midterm_examination_schedule_summer_2026") if "midterm_examination_schedule_summer_2026.txt" in existing_files else (None, "midterm_examination_schedule_summer_2026")

            if matched_file in existing_files:
                return matched_file, os.path.splitext(matched_file)[0]

        for ef in existing_files:
            stem_name = os.path.splitext(ef)[0]
            if stem_name in resp_text:
                return ef, stem_name

        new_match = re.search(r'NEW:\s*`?([a-zA-Z0-9_\-]+)`?', resp_text, re.IGNORECASE)
        if new_match:
            stem = new_match.group(1).strip().lower()
            return None, stem

    return None, "document"


def _rule_based_decompose_knowledge(text: str, existing_files: list[str]) -> list[dict]:
    """
    Fallback deterministic segmenter that splits multi-topic text by headers or numbers,
    and maps each section to existing files or creates new category files.
    """
    # Mapping of keywords to standard knowledge files
    KEYWORD_MAP = [
        (["calendar", "academic calendar", "semester calendar", "term calendar", "academic schedule", "একাডেমিক ক্যালেন্ডার", "ক্যালেন্ডার"], "academic_calendar.txt"),
        (["seatplan", "seat plan", "সিট প্ল্যান", "সিটপ্ল্যান"], "exam_seatplan.txt"),
        (["routine", "class routine", "lab routine", "class schedule", "ক্লাস রুটিন", "রুটিন"], "class_and_lab_routine.txt"),
        (["cse", "computer science", "artificial intelligence lab"], "cse_department_and_faculty.txt"),
        (["eee", "electrical and electronic"], "eee_department_and_faculty.txt"),
        (["me", "mechanical engineering"], "me_department_and_faculty.txt"),
        (["ce", "civil engineering"], "ce_department_and_faculty.txt"),
        (["ipe", "industrial and production"], "ipe_department_and_faculty.txt"),
        (["fbs", "bba", "business studies", "accounting"], "fbs_department_and_faculty.txt"),
        (["hall", "hostel", "mess", "residential", "provost"], "residential_halls_and_facilities.txt"),
        (["exam", "grading", "cgpa", "midterm", "final exam"], "undergraduate_examination_policy.txt"),
        (["admission", "tuition", "waiver", "scholarship", "eligibility"], "admission_information.txt"),
        (["transport", "bus", "shuttle", "route", "পিকআপ", "বাস"], "university_transport_schedule.txt"),
        (["library", "book", "borrow", "circulation", "লাইব্রেরি"], "central_library_guidelines.txt"),
        (["medical", "clinic", "health", "doctor", "ambulance", "ফার্স্ট এইড"], "medical_center_guidelines.txt"),
        (["history", "established", "syedpur", "cantonment", "vc"], "brief_history_of_baust.txt"),
        (["contact", "phone", "email", "directory", "hotline"], "administrative_contact_directory.txt"),
        (["research", "journal", "publication", "grant"], "research_and_publication_cell.txt"),
        (["facilities", "canteen", "cafeteria", "sports", "gym"], "campus_facilities_and_services.txt")
    ]

    # Split text into sections based on headers (#, ##), numbered lists (1., 2.), or dividers (---)
    raw_sections = re.split(r'\n(?=[#]{1,3}\s+|\d+\.\s+|[A-Z]\.\s+|---)', text.strip())
    raw_sections = [s.strip() for s in raw_sections if len(s.strip()) > 30]

    # If only 1 section or splitting didn't find multiple distinct blocks
    if len(raw_sections) <= 1:
        raw_sections = [text.strip()]

    results = []

    for sec in raw_sections:
        sec_lower = sec.lower()
        matched_file = None
        topic_title = "University Update"

        # Determine topic title from first line
        first_line = sec.splitlines()[0].strip()
        first_line_clean = re.sub(r'^[#0-9\.\-\s:]+', '', first_line).strip()
        if 4 <= len(first_line_clean) <= 60:
            topic_title = first_line_clean.title()

        for keywords, target_file in KEYWORD_MAP:
            matched = False
            for k in keywords:
                if len(k) <= 3:
                    if re.search(r'\b' + re.escape(k) + r'\b', sec_lower):
                        matched = True
                        break
                elif k in sec_lower:
                    matched = True
                    break
            if matched:
                matched_file = target_file
                break

        if not matched_file:
            # Generate a clean new category stem from topic title
            slug = re.sub(r'[^a-zA-Z0-9]+', '_', topic_title.lower()).strip('_')
            matched_file = f"{slug or 'general_notice'}.txt"

        action = "UPDATE_EXISTING" if matched_file in existing_files else "CREATE_NEW"
        
        results.append({
            "topic": topic_title,
            "action": action,
            "filename": matched_file,
            "content": sec
        })

    return results


def decompose_and_route_knowledge(new_text: str, knowledge_folder: str) -> list[dict]:
    """
    Intelligently analyzes incoming text (which may contain single or multiple categories).
    Differentiates which parts belong to an EXISTING category and updates that file,
    and which parts require creating a NEW category file.
    
    Returns a list of structured action dicts:
    [
      {
        "topic": "Residential Hall Guidelines",
        "action": "UPDATE_EXISTING", # or "CREATE_NEW"
        "filename": "residential_halls_and_facilities.txt",
        "content": "... clean markdown section ..."
      }, ...
    ]
    """
    if not new_text or not new_text.strip():
        return []

    if not os.path.exists(knowledge_folder):
        return [{
            "topic": "University Notice",
            "action": "CREATE_NEW",
            "filename": "general_notice.txt",
            "content": new_text.strip()
        }]

    existing_files = [f for f in sorted(os.listdir(knowledge_folder)) if f.endswith(".txt")]
    
    # Compile existing categories summary
    file_summaries = []
    for f in existing_files:
        path = os.path.join(knowledge_folder, f)
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                c_sample = file_obj.read(1200)
                headings = [line.strip() for line in c_sample.splitlines() if line.strip().startswith("#")]
                headings_str = " | ".join(headings[:5]) if headings else "General records"
                file_summaries.append(f"- File: `{f}` (Topics: {headings_str})")
        except Exception:
            continue

    files_str = "\n".join(file_summaries)

    # Ask Gemini to decompose and classify into multiple or single categories
    prompt = f"""You are the Chief Knowledge Base Architect for Bangladesh Army University of Science and Technology (BAUST).

EXISTING KNOWLEDGE BASE CATEGORY FILES IN DATABASE:
{files_str}

NEWLY SUBMITTED INFORMATION:
\"\"\"
{new_text[:4000]}
\"\"\"

TASK:
1. Carefully analyze the newly submitted information.
2. Determine if it covers a SINGLE topic/category, or MULTIPLE distinct categories (e.g. mentions hall rules, exam schedule, bus routes, department updates, or fee policies).
3. If it contains MULTIPLE topics, differentiate and split the information into distinct thematic sections.
4. For EACH section:
   - Check if the topic belongs to an EXISTING file from the list above. If so:
     "action": "UPDATE_EXISTING"
     "filename": "<exact_filename_from_list>"
   - If the topic does NOT belong to any existing file, create a NEW category:
     "action": "CREATE_NEW"
     "filename": "<descriptive_snake_case_name.txt>"
   - "topic": "<Short Descriptive Title>"
   - "content": "<The specific information, rules, and details belonging to this category in clean Markdown>"

STRICT JSON OUTPUT FORMAT (NO PRETEXT, NO MARKDOWN CODE FENCES OUTSIDE JSON):
[
  {{
    "topic": "Title of Topic",
    "action": "UPDATE_EXISTING",
    "filename": "existing_filename.txt",
    "content": "Formatted section text"
  }}
]
"""
    try:
        resp = call_gemini_generate(prompt, temperature=0.1, response_mime_type="application/json")
        if resp:
            # Direct parse guaranteed JSON with fallback
            try:
                parsed = json.loads(resp)
            except Exception:
                json_match = RE_JSON_ARRAY.search(resp)
                parsed = json.loads(json_match.group(0)) if json_match else []

            if isinstance(parsed, list) and len(parsed) > 0:
                    valid_items = []
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        topic = item.get("topic", "University Notice").strip()
                        raw_fn = item.get("filename", "").strip()
                        content = item.get("content", "").strip()

                        if not content:
                            continue

                        # Clean filename
                        clean_fn = os.path.basename(raw_fn).replace("`", "").strip()
                        if not clean_fn.endswith(".txt"):
                            clean_fn = f"{clean_fn}.txt"
                        clean_fn = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', clean_fn).lower()

                        # Determine if file exists
                        action = "UPDATE_EXISTING" if clean_fn in existing_files else "CREATE_NEW"

                        valid_items.append({
                            "topic": topic,
                            "action": action,
                            "filename": clean_fn,
                            "content": content
                        })

                    if valid_items:
                        return valid_items
    except Exception as e:
        print(f"[Decompose Knowledge Warning]: {e}")

    # Fallback to deterministic rule-based segmenter
    return _rule_based_decompose_knowledge(new_text, existing_files)


def merge_into_existing_document(existing_text: str, new_text: str) -> str:
    """
    Intelligently merges new information into an existing document without creating duplicates.
    """
    if not existing_text or not existing_text.strip():
        return new_text
    if not new_text or not new_text.strip():
        return existing_text

    if len(existing_text) + len(new_text) > 4000:
        return f"{existing_text.strip()}\n\n---\n\n## Additional Information & Updates\n\n{new_text.strip()}"

    prompt = f"""
You are an expert Knowledge Base Editor for Bangladesh Army University of Science and Technology (BAUST).
Merge the NEW information into the EXISTING document intelligently without duplication and preserving clean Markdown.

EXISTING DOCUMENT:
{existing_text}

NEW INFORMATION TO INTEGRATE:
{new_text}
"""
    merged = call_gemini_generate(prompt, temperature=0.1)
    if merged and merged.strip():
        return merged.strip()

    return f"{existing_text.strip()}\n\n---\n\n## Additional Updates & Policies\n{new_text.strip()}"


def detect_is_bengali_intent(text: str) -> bool:
    """Accurately checks if text is in Bengali script or Banglish conversational phrasing."""
    if not text:
        return True
    if re.search(r'[\u0980-\u09FF]', text):
        return True
    t_words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    banglish_markers = {
        "ki", "koto", "kto", "kothay", "kothy", "kemon", "kmn", "ke", "keu", "kon", 
        "konta", "koyta", "koyti", "koita", "ache", "ase", "bolo", "dao", "kore", "er", 
        "tumi", "apni", "ami", "amader", "tomader", "apnader", "hobe", "lagbe", "kivabe", 
        "jante", "chai", "koren", "koro", "korche", "bistarito", "eligibility", "shikkok", 
        "shikkha", "borti", "fee", "fi", "khoroch", "onushod", "bivag", "naam", "nam", 
        "porichoy", "thikana", "shob", "sob", "shathe", "sath", "parbo", "hobe"
    }
    return len(t_words.intersection(banglish_markers)) > 0


def synthesize_smart_answer(context: str, question: str, history: list = None) -> str:
    """
    Intelligent Local Synthesizer:
    Directly answers identity, translations, university queries, and academic questions 
    with crisp, structured, language-matched precision.
    """
    q_lower = question.strip().lower()
    is_bn = detect_is_bengali_intent(question)
    norm_q = normalize_text_for_intent(question)

    # 0. Authoritative Entity Profile & Routine Handler (Highest Priority)
    if "### AUTHORITATIVE ENTITY PROFILE & RECORDS:" in context:
        auth_sec = context.split("### AUTHORITATIVE ENTITY PROFILE & RECORDS:")[1].split("### RELEVANT UNIVERSITY KNOWLEDGE PASSAGES:")[0].strip()
        
        # A. Teacher Profile (Short forms & full names with direct phone and email)
        if "Teacher / Faculty Profile" in auth_sec or "Faculty / Staff Profile" in auth_sec:
            t_lines = [l.strip() for l in auth_sec.splitlines() if l.strip().startswith("*")]
            if t_lines:
                res = "\n".join(t_lines).replace("সাইদপুর", "সৈয়দপুর").replace("সায়েদপুর", "সৈয়দপুর")
                return sanitize_contact_markdown(res)
        
        # B. Student Class Routine
        if "Student Profile & Official Class & Lab Routine" in auth_sec:
            st_name = re.search(r'\* Student Name:\s*(.*)', auth_sec)
            st_roll = re.search(r'\* Student ID / Roll:\s*(\d+)', auth_sec)
            st_prog = re.search(r'\* Programme / Department:\s*(.*)', auth_sec)
            st_level = re.search(r'\* Academic Level & Term:\s*(.*)', auth_sec)
            st_parity = re.search(r'Roll Parity / Group:\s*(\w+)', auth_sec)
            
            res_lines = []
            if st_name and st_roll:
                res_lines.append(f"* **শিক্ষার্থীর নাম (Student Name):** {st_name.group(1).strip()}")
                res_lines.append(f"* **শিক্ষার্থীর আইডি / রোল (Student ID / Roll):** `{st_roll.group(1).strip()}`")
                prog_txt = st_prog.group(1).strip() if st_prog else "কম্পিউটার সায়েন্স অ্যান্ড ইঞ্জিনিয়ারিং"
                lvl_txt = st_level.group(1).strip() if st_level else "লেভেল-৩ টার্ম-২"
                res_lines.append(f"* **বিভাগ ও লেভেল/টার্ম (Department & Level/Term):** {prog_txt} ({lvl_txt})")
                res_lines.append("")

            # Check if pre-calculated VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR STUDENT is present
            if "VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR STUDENT:" in auth_sec:
                sched_part = auth_sec.split("VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR STUDENT:")[1].split("OFFICIAL BATCH CLASS & LAB ROUTINE")[0].strip()
                res_lines.append(sched_part)
                return "\n".join(res_lines).replace("সাইদপুর", "সৈয়দপুর").replace("সায়েদপুর", "সৈয়দপুর")

        # C. Batch / Section Class Routine (e.g. CSE 3-II Section B / Section A)
        if "Official Class & Lab Routine" in auth_sec and "VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR BATCH:" in auth_sec:
            sched_part = auth_sec.split("VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR BATCH:")[1].split("OFFICIAL BATCH CLASS & LAB ROUTINE")[0].strip()
            return sched_part.replace("সাইদপুর", "সৈয়দপুর").replace("সায়েদপুর", "সৈয়দপুর")

        # D. Exam Seatplan
        if "Student Exam Seatplan Allocation" in auth_sec:
            s_lines = [l.strip() for l in auth_sec.splitlines() if l.strip().startswith("*")]
            if s_lines:
                return "\n".join(s_lines).replace("সাইদপুর", "সৈয়দপুর").replace("সায়েদপুর", "সৈয়দপুর")
    
    # 1. Identity & Name questions
    if any(k in norm_q for k in ["what is your name", "what's your name", "your name", "who are you", "who r u", "who made you"]):
        return "I am the official AI Assistant of Bangladesh Army University of Science and Technology (BAUST), Saidpur."
    
    if any(k in norm_q for k in ["tomar nam", "apnar nam", "tumi ke", "apni ke", "ke tumi", "ke apni", "tomar porichoy"]) or any(k in q_lower for k in ["তোমার নাম", "তুমি কে", "আপনার নাম", "কে তুমি"]):
        return "আমি বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুরের অফিসিয়াল AI সহকারী। আপনাকে কীভাবে সাহায্য করতে পারি?"

    # 2. How are you
    if any(k in norm_q for k in ["kemon acho", "kemon achen", "kmn acho", "kmn asen", "how are you", "how r u"]) or any(k in q_lower for k in ["কেমন আছো", "কেমন আছেন"]):
        if is_bn:
            return "আলহামদুলিল্লাহ, আমি ভালো আছি। BAUST সম্পর্কিত যেকোনো তথ্যে আপনাকে সাহায্য করতে প্রস্তুত।"
        return "I am doing well, thank you! How can I assist you with BAUST information today?"

    # 3. Translation & Language Conversion of previous turn
    is_trans, _ = is_translation_request(question)
    if is_trans and history:
        last_bot_msg = ""
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") in ["assistant", "model", "bot"]:
                last_bot_msg = str(msg.get("content", "")).strip()
                break
        
        if last_bot_msg:
            # Dynamic university data with graceful fallbacks
            vc_info = UNIVERSITY_CONFIG.get("vice_chancellor", {})
            vc_en = vc_info.get("name_en", "Brigadier General A B M Humayun Kabir, SGP, ndc, psc, te (Retd.)")
            vc_bn = vc_info.get("name_bn", "ব্রিগেডিয়ার জেনারেল এ বি এম হুমায়ুন কবির, এসজিপি, এনডিসি, পিএসসি, টিই (অব.)")
            
            cse_info = UNIVERSITY_CONFIG.get("cse_head", {})
            cse_en = cse_info.get("name_en", "Dr. Md Nakib Hayat Chowdhury")
            cse_bn = cse_info.get("name_bn", "ড. মো. নাকিব হায়াত চৌধুরী")
            cse_desig = cse_info.get("designation", "Associate Professor & Head of CSE Department")
            cse_phone = cse_info.get("phone", "01769675560")
            cse_email = cse_info.get("email", "hdcse@baust.edu.bd")

            # If user asked for Bangla
            if any(p in q_lower for p in ["bangla", "বাংলা"]):
                if "official AI Assistant" in last_bot_msg or "BAUST AI Assistant" in last_bot_msg:
                    return "আমি বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুরের অফিসিয়াল AI সহকারী।"
                if "Humayun Kabir" in last_bot_msg or "Vice-Chancellor" in last_bot_msg:
                    return f"BAUST-এর বর্তমান ভাইস-চ্যান্সেলর (VC) হলেন **{vc_bn}**।"
                if "Nakib Hayat" in last_bot_msg or "CSE" in last_bot_msg:
                    return f"সিএসই (CSE) বিভাগের বিভাগীয় প্রধান হলেন **{cse_bn}** ({cse_desig})।"
                if "Faculty" in last_bot_msg or "faculties" in last_bot_msg:
                    return (
                        "**BAUST-এর অনুষদসমূহ (Faculties):**\n\n"
                        "BAUST-এ মোট **৫টি অনুষদ** রয়েছে:\n\n"
                        "1. **ECE অনুষদ** (CSE, EEE, ICT)\n"
                        "2. **ME অনুষদ** (ME, IPE)\n"
                        "3. **CE অনুষদ** (Civil Engineering)\n"
                        "4. **ব্যবসায় শিক্ষা অনুষদ (BS)** (BBA, AIS, MBA)\n"
                        "5. **বিজ্ঞান ও মানবিক অনুষদ (SH)** (English)\n\n"
                        "📞 হেল্পলাইন: [০১৭৬৯৬৭৫৫৮৮](tel:01769675588)"
                    )
                if "Tuition" in last_bot_msg or "Fee" in last_bot_msg:
                    return (
                        "**BAUST ভর্তি ও টিউশন ফি কাঠামো:**\n\n"
                        "* **১ম সেমিস্টার (ভর্তি ফি):** ৬৫,০০০ টাকা\n"
                        "* **প্রতি সেমিস্টার ফি:** ~৯০,০০০ টাকা (ইঞ্জিনিয়ারিং) / ~৫০,০০০-৬০,০০০ টাকা (বিবিএ ও ইংরেজি)\n"
                        "* **৪ বছরের মোট আনুমানিক খরচ:** ~৬,৯৫,০০০ টাকা (ইঞ্জিনিয়ারিং) | ~৪,৮০,০০০ টাকা (বিবিএ/বিএ)\n\n"
                        "📞 হেল্পলাইন: [০১৭৬৯৬৭৫৫৮৮](tel:01769675588), [০১৭৬৯৬৭৫৫৮৯](tel:01769675589)"
                    )
                return f"**বাংলায় উত্তর:**\n{last_bot_msg}"

            # If user asked for English
            if any(p in q_lower for p in ["english", "ইংলিশ"]):
                if "অফিসিয়াল AI" in last_bot_msg or "সহায়ক" in last_bot_msg:
                    return "I am the official AI Assistant of Bangladesh Army University of Science and Technology (BAUST), Saidpur."
                if "হুমায়ুন কবির" in last_bot_msg or "ভাইস-চ্যান্সেলর" in last_bot_msg or "উপাচার্য" in last_bot_msg:
                    return f"The current Vice-Chancellor (VC) of BAUST is **{vc_en}**."
                if "নাকিব হায়াত" in last_bot_msg or "সিএসই" in last_bot_msg:
                    return f"The Head of the Department of CSE is **{cse_en}** ({cse_desig}). Phone: [{cse_phone}](tel:{cse_phone}), Email: [{cse_email}](mailto:{cse_email})"
                if "অনুষদ" in last_bot_msg or "বিভাগ" in last_bot_msg:
                    return (
                        "**Faculties at BAUST:**\n\n"
                        "BAUST currently functions under **5 Academic Faculties**:\n\n"
                        "1. **Faculty of Electrical & Computer Engineering (ECE)** (CSE, EEE, ICT)\n"
                        "2. **Faculty of Mechanical Engineering (ME)** (ME, IPE)\n"
                        "3. **Faculty of Civil Engineering (CE)** (CE)\n"
                        "4. **Faculty of Business Studies (BS)** (BBA, AIS, MBA)\n"
                        "5. **Faculty of Sciences & Humanities (SH)** (English)\n\n"
                        "📞 Helpline: [01769675588](tel:01769675588)"
                    )
                if "ভর্তি" in last_bot_msg or "ফি" in last_bot_msg:
                    return (
                        "**BAUST Tuition & Fee Structure:**\n\n"
                        "* **1st Semester (Admission Fee):** 65,000 BDT\n"
                        "* **Per Semester Fee:** ~90,000 BDT (Engineering) / ~50,000–60,000 BDT (BBA & English)\n"
                        "* **Total 4-Year Estimated Cost:** ~6,95,000 BDT (Engineering) | ~4,80,000 BDT (BBA/BA)\n\n"
                        "📞 Helpline: [01769675588](tel:01769675588), [01769675589](tel:01769675589)"
                    )
                return f"**In English:**\n{last_bot_msg}"

    # 4. Direct Code / Technical Questions
    if any(k in q_lower for k in ["python", "binary search", "code", "c++", "java", "function", "program"]) and not any(w in q_lower for w in ["baust", "admission", "department"]):
        if "binary search" in q_lower:
            return (
                "```python\n"
                "def binary_search(arr, target):\n"
                "    low, high = 0, len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid - 1\n"
                "    return -1\n"
                "```"
            )

    # 5. Faculties (অনুষদ) Inquiries
    if any(k in norm_q for k in ["faculty koyta", "koyta faculty", "koyti faculty", "faculty list", "faculties", "how many faculty", "how many faculties", "onushod koyta", "koyta onushod", "koyti onushod", "onushod list", "onushod gulo"]) or any(k in q_lower for k in ["অনুষদ কয়টি", "অনুষদ কয়টি", "অনুষদসমূহ", "অনুষদ কি কি"]):
        if is_bn:
            return (
                "**BAUST-এর অনুষদসমূহ (Faculties):**\n\n"
                "BAUST-এ বর্তমানে মোট **৫টি অনুষদ** রয়েছে:\n\n"
                "1. **Faculty of Electrical and Computer Engineering (ECE)**\n"
                "2. **Faculty of Mechanical Engineering (ME)**\n"
                "3. **Faculty of Civil Engineering (CE)**\n"
                "4. **Faculty of Business Studies (BS)**\n"
                "5. **Faculty of Sciences & Humanities (SH)**\n\n"
                "📞 ভর্তি ও তথ্য হেল্পলাইন: [০১৭৬৯৬৭৫৫৮৮](tel:01769675588), [০১৭৬৯৬৭৫৫৮৯](tel:01769675589)"
            )
        return (
            "**Academic Faculties at BAUST:**\n\n"
            "BAUST currently functions under **5 Academic Faculties**:\n\n"
            "1. **Faculty of Electrical and Computer Engineering (ECE)**\n"
            "2. **Faculty of Mechanical Engineering (ME)**\n"
            "3. **Faculty of Civil Engineering (CE)**\n"
            "4. **Faculty of Business Studies (BS)**\n"
            "5. **Faculty of Sciences & Humanities (SH)**\n\n"
            "📞 Admission Helpline: [01769675588](tel:01769675588), [01769675589](tel:01769675589)"
        )

    # 6. Departments (বিভাগ) & Programs Inquiries
    if any(k in norm_q for k in ["koyta dept", "koyti dept", "koyta department", "koyti department", "how many department", "department koyta", "dept koyta", "list of department", "department list", "departments", "department gulo", "programs", "subject", "subject koyta", "ki ki subject", "ki ki department"]) or any(k in q_lower for k in ["বিভাগ কয়টি", "কয়টি বিভাগ", "বিভাগসমূহ", "অনুষদ ও বিভাগ", "কী কী বিভাগ", "কি কি বিভাগ"]):
        if is_bn:
            return (
                "**BAUST-এর একাডেমিক বিভাগ ও ডিগ্রিসমূহ:**\n\n"
                "BAUST-এ বর্তমানে ৫টি অনুষদের অধীনে মোট **৮টি বিভাগ ও ডিগ্রি** চালু রয়েছে:\n\n"
                "1. **CSE** — B.Sc. in Computer Science & Engineering\n"
                "2. **EEE** — B.Sc. in Electrical & Electronic Engineering\n"
                "3. **ICT** — B.Sc. in Information & Communication Technology\n"
                "4. **ME** — B.Sc. in Mechanical Engineering\n"
                "5. **IPE** — B.Sc. in Industrial & Production Engineering\n"
                "6. **CE** — B.Sc. in Civil Engineering\n"
                "7. **BBA / AIS** — Bachelor of Business Administration\n"
                "8. **English** — BA (Hons) in English\n\n"
                "📞 ভর্তি ও তথ্য হেল্পলাইন: [০১৭৬৯৬৭৫৫৮৮](tel:01769675588), [০১৭৬৯৬৭৫৫৮৯](tel:01769675589)"
            )
        return (
            "**Academic Departments & Programs at BAUST:**\n\n"
            "BAUST offers undergraduate degrees across **8 departments** under 5 faculties:\n\n"
            "1. **CSE** — B.Sc. in Computer Science and Engineering\n"
            "2. **EEE** — B.Sc. in Electrical and Electronic Engineering\n"
            "3. **ICT** — B.Sc. in Information and Communication Technology\n"
            "4. **ME** — B.Sc. in Mechanical Engineering\n"
            "5. **IPE** — B.Sc. in Industrial and Production Engineering\n"
            "6. **CE** — B.Sc. in Civil Engineering\n"
            "7. **BBA / AIS** — Bachelor of Business Administration\n"
            "8. **English** — BA (Hons) in English\n\n"
            "📞 Admission Helpline: [01769675588](tel:01769675588), [01769675589](tel:01769675589)"
        )

    # 7. Specific University Leadership (VC / Registrar / Deans)
    if any(k in norm_q for k in ["vc", "vice chancellor", "upacharya"]) or any(k in q_lower for k in ["ভিসি", "উপাচার্য"]):
        if is_bn:
            return "BAUST-এর বর্তমান ভাইস-চ্যান্সেলর (VC) হলেন **ব্রিগেডিয়ার জেনারেল এ বি এম হুমায়ুন কবির, এসজিপি, এনডিসি, পিএসসি, টিই (অব.)**।"
        return "The current Vice-Chancellor (VC) of BAUST is **Brigadier General A B M Humayun Kabir, SGP, ndc, psc, te (Retd.)**."

    if any(k in norm_q for k in ["registrar"]) or any(k in q_lower for k in ["রেজিস্ট্রার", "রেজিস্টার"]):
        if is_bn:
            return "BAUST-এর রেজিস্ট্রার হলেন **লেফটেন্যান্ট কর্নেল মো. নজরুল ইসলাম (অব.)**। ইমেইল: [registrar@baust.edu.bd](mailto:registrar@baust.edu.bd)"
        return "The Registrar of BAUST is **Lt Col Md Nazrul Islam (Retd.)**. Email: [registrar@baust.edu.bd](mailto:registrar@baust.edu.bd)"

    # 8. Department Heads & Faculty
    # CSE
    if any(k in norm_q for k in ["cse"]) and any(k in norm_q for k in ["head", "hod", "cheyarman", "prodhan"]) or any(k in q_lower for k in ["হেড", "প্রধান"]):
        if any(k in norm_q for k in ["cse"]):
            if is_bn:
                return (
                    "সিএসই (CSE) বিভাগের প্রধান হলেন **ড. মো. নাকিব হায়াত চৌধুরী** (সহযোগী অধ্যাপক)।\n\n"
                    "📞 ফোন: [01769675560](tel:01769675560) | ✉️ ইমেইল: [hdcse@baust.edu.bd](mailto:hdcse@baust.edu.bd)"
                )
            return (
                "The Head of the CSE Department is **Dr. Md Nakib Hayat Chowdhury** (Associate Professor).\n\n"
                "📞 Phone: [01769675560](tel:01769675560) | ✉️ Email: [hdcse@baust.edu.bd](mailto:hdcse@baust.edu.bd)"
            )

    # CSE Faculty List
    if any(k in norm_q for k in ["cse"]) and any(k in norm_q for k in ["faculty", "teacher", "teachers", "list", "talika"]) or (any(k in q_lower for k in ["সিএসই"]) and any(k in q_lower for k in ["শিক্ষক", "তালিকা"])):
        if is_bn:
            return (
                "**BAUST CSE বিভাগের বিশিষ্ট শিক্ষকবৃন্দ:**\n\n"
                "* **ড. মো. নাকিব হায়াত চৌধুরী** — সহযোগী অধ্যাপক ও বিভাগীয় প্রধান\n"
                "* **প্রফেসর ড. মো. লুৎফর রহমান** — অধ্যাপক\n"
                "* **রুবেল বসাক** — সহযোগী অধ্যাপক\n"
                "* **মো. রাকিবুল ইসলাম** — সহযোগী অধ্যাপক\n"
                "* **মো. রুহুল আমিন** — সহকারী অধ্যাপক\n\n"
                "📞 যোগাযোগ: [01769675560](tel:01769675560) | ✉️ [hdcse@baust.edu.bd](mailto:hdcse@baust.edu.bd)"
            )
        return (
            "**Key Faculty Members of BAUST CSE Department:**\n\n"
            "* **Dr. Md Nakib Hayat Chowdhury** — Associate Professor & Head\n"
            "* **Prof. Dr. Md. Lutfor Rahman** — Professor\n"
            "* **Rubel Basak** — Associate Professor\n"
            "* **Md. Rakibul Islam** — Associate Professor\n"
            "* **Md. Ruhul Amin** — Assistant Professor\n\n"
            "📞 Contact: [01769675560](tel:01769675560) | ✉️ [hdcse@baust.edu.bd](mailto:hdcse@baust.edu.bd)"
        )

    # EEE Head
    if any(k in norm_q for k in ["eee"]) and any(k in norm_q for k in ["head", "hod", "teacher", "faculty", "prodhan", "shikkok"]):
        if is_bn:
            return "ইইই (EEE) বিভাগের বিভাগীয় প্রধান হলেন **ড. রুবিনা আক্তার** (সহযোগী অধ্যাপক)।\n📞 ফোন: [01769675561](tel:01769675561) | ✉️ ইমেইল: [hdeee@baust.edu.bd](mailto:hdeee@baust.edu.bd)"
        return "The Head of the EEE Department is **Dr. Rubina Aktar** (Associate Professor).\n📞 Phone: [01769675561](tel:01769675561) | ✉️ Email: [hdeee@baust.edu.bd](mailto:hdeee@baust.edu.bd)"

    # ME Head
    if any(k in norm_q for k in ["me", "mechanical"]) and any(k in norm_q for k in ["head", "hod", "prodhan"]):
        if is_bn:
            return "মেকানিক্যাল ইঞ্জিনিয়ারিং (ME) বিভাগের বিভাগীয় প্রধান হলেন **ড. মো. রুহুল আমিন** (সহযোগী অধ্যাপক)।\n📞 ফোন: [01769675562](tel:01769675562) | ✉️ ইমেইল: [hdme@baust.edu.bd](mailto:hdme@baust.edu.bd)"
        return "The Head of the ME Department is **Dr. Md. Ruhul Amin** (Associate Professor).\n📞 Phone: [01769675562](tel:01769675562) | ✉️ Email: [hdme@baust.edu.bd](mailto:hdme@baust.edu.bd)"

    # CE Head
    if any(k in norm_q for k in ["ce", "civil"]) and any(k in norm_q for k in ["head", "hod", "prodhan"]):
        if is_bn:
            return "সিভিল ইঞ্জিনিয়ারিং (CE) বিভাগের বিভাগীয় প্রধান হলেন **ড. মো. মাহমুদুল হাসান**।\n📞 ফোন: [01769675563](tel:01769675563) | ✉️ ইমেইল: [hdce@baust.edu.bd](mailto:hdce@baust.edu.bd)"
        return "The Head of the CE Department is **Dr. Md. Mahmudul Hasan**.\n📞 Phone: [01769675563](tel:01769675563) | ✉️ Email: [hdce@baust.edu.bd](mailto:hdce@baust.edu.bd)"

    # IPE Head
    if any(k in norm_q for k in ["ipe"]) and any(k in norm_q for k in ["head", "hod", "prodhan"]):
        if is_bn:
            return "আইপিই (IPE) বিভাগের বিভাগীয় প্রধান হলেন **ড. মো. মামুনুর রশীদ**।\n📞 ফোন: [01769675564](tel:01769675564) | ✉️ ইমেইল: [hdipe@baust.edu.bd](mailto:hdipe@baust.edu.bd)"
        return "The Head of the IPE Department is **Dr. Md. Mamunur Rashid**.\n📞 Phone: [01769675564](tel:01769675564) | ✉️ Email: [hdipe@baust.edu.bd](mailto:hdipe@baust.edu.bd)"

    # BBA / Business Dean
    if any(k in norm_q for k in ["bba", "business"]) and any(k in norm_q for k in ["head", "hod", "dean", "prodhan"]):
        if is_bn:
            return "বিজনেস অনুষদের ডিন ও বিবিএ প্রধান হলেন **প্রফেসর ড. মো. জাহাঙ্গীর আলম**।\n📞 ফোন: [01769675565](tel:01769675565) | ✉️ ইমেইল: [hdbba@baust.edu.bd](mailto:hdbba@baust.edu.bd)"
        return "The Dean of FBS and Head of BBA is **Prof. Dr. Md. Jahangir Alam**.\n📞 Phone: [01769675565](tel:01769675565) | ✉️ Email: [hdbba@baust.edu.bd](mailto:hdbba@baust.edu.bd)"

    # 9. Admission Eligibility & Direct Admission
    if any(k in norm_q for k in ["eligibility", "requirement", "requirements", "gpa", "borti hote ki lage", "joggotar", "jogota"]) or any(k in q_lower for k in ["যোগ্যতা", "শর্ত", "কারা আবেদন"]):
        if is_bn:
            return (
                "**BAUST স্নাতক ভর্তি আবেদনের ন্যূনতম যোগ্যতা:**\n\n"
                "* **ইঞ্জিনিয়ারিং অনুষদ (CSE/EEE/ME/CE/ICT/IPE):** এসএসসি ও এইচএসসি উভয় পরীক্ষায় আলাদাভাবে ন্যূনতম GPA 3.00 এবং মোট সম্মিলিত GPA ন্যূনতম **7.00** থাকতে হবে (এইচএসসিতে গণিত, পদার্থ ও রসায়ন বাধ্যতামূলক)।\n"
                "* **বিবিএ ও ইংরেজি:** এসএসসি ও এইচএসসি মিলিয়ে মোট সম্মিলিত GPA ন্যূনতম **6.00** থাকতে হবে।\n"
                "* **সরাসরি ভর্তি (পরীক্ষা ছাড়া):** এসএসসি ও এইচএসসি মিলিয়ে মোট GPA 9.00 বা তার বেশি থাকলে সরাসরি ভর্তি হওয়া যায়।\n\n"
                "📞 ভর্তি হেল্পলাইন: [০১৭৬৯৬৭৫৫৮৮](tel:01769675588), [০১৭৬৯৬৭৫৫৮৯](tel:01769675589)"
            )
        return (
            "**BAUST Undergraduate Admission Eligibility:**\n\n"
            "* **Engineering Programs (CSE/EEE/ME/CE/ICT/IPE):** Minimum GPA 3.00 separately in SSC and HSC with a combined total GPA of at least **7.00** (Science with Math, Physics, Chemistry).\n"
            "* **BBA & BA in English:** Minimum GPA 3.00 in each with a combined total GPA of at least **6.00**.\n"
            "* **Direct Admission (Without Exam):** Combined total GPA of **9.00** or more (8.00+ for BBA/English).\n\n"
            "📞 Helpline: [01769675588](tel:01769675588), [01769675589](tel:01769675589)"
        )

    # 10. Admission Test Mark Distribution
    if any(k in norm_q for k in ["admission test", "manbonton", "mark", "marks", "exam system", "porikkha"]) or any(k in q_lower for k in ["মানবন্টন", "ভর্তি পরীক্ষা"]):
        if is_bn:
            return (
                "**BAUST ভর্তি পরীক্ষার মানবন্টন (১০০ নম্বর, ১ ঘণ্টা MCQ):**\n\n"
                "* **ইঞ্জিনিয়ারিং অনুষদ:** গণিত (৪০), পদার্থবিজ্ঞান (৩০), রসায়ন (২০), ফাংশনাল ইংলিশ (১০)।\n"
                "* **বিবিএ অনুষদ:** ইংরেজি (৩৫), সাধারণ জ্ঞান (৩০), অ্যানালিটিক্যাল অ্যাবিলিটি (৩৫)।\n"
                "* **ইংরেজি অনুষদ:** ইংরেজি ভাষা ও শব্দভাণ্ডার (৩০), রিডিং ও রাইটিং (৩০), সাধারণ জ্ঞান (৪০)।\n\n"
                "📞 হেল্পলাইন: [০১৭৬৯৬৭৫৫৮৮](tel:01769675588)"
            )
        return (
            "**BAUST Admission Test Marks Distribution (100 Marks, 1hr MCQ):**\n\n"
            "* **Engineering:** Mathematics (40), Physics (30), Chemistry (20), Functional English (10).\n"
            "* **BBA:** English (35), General Knowledge (30), Analytical Ability (35).\n"
            "* **English:** English Language & Vocabulary (30), Writing (30), General Knowledge (40).\n\n"
            "📞 Helpline: [01769675588](tel:01769675588)"
        )

    # 11. Tuition Fees & Program Costs
    if any(k in norm_q for k in ["fee", "fees", "cost", "taka", "khoroch", "tk", "bdt"]) or any(k in q_lower for k in ["টাকা", "খরচ", "ফি"]):
        if is_bn:
            return (
                "**BAUST ভর্তি ও প্রোগ্রাম ফি কাঠামো:**\n\n"
                "* **ভর্তিকালীন ফি (১ম সেমিস্টার):** ৬৫,০০০ টাকা\n"
                "* **পরবর্তী প্রতি সেমিস্টার ফি:** প্রায় ৯০,০০০ টাকা (ইঞ্জিনিয়ারিং) / প্রায় ৫০,০০০-৬০,০০০ টাকা (বিবিএ ও ইংরেজি)\n"
                "* **৪ বছরের মোট আনুমানিক খরচ:** ~৬,৯৫,০০০ টাকা (বি.এসসি ইঞ্জিনিয়ারিং) | ~৪,৮০,০০০ টাকা (বিবিএ/ইংরেজি)\n\n"
                "📞 হেল্পলাইন: [০১৭৬৯৬৭৫৫৮৮](tel:01769675588), [০১৭৬৯৬৭৫৫৮৯](tel:01769675589) | ✉️ [admission@baust.edu.bd](mailto:admission@baust.edu.bd)"
            )
        return (
            "**BAUST Tuition & Program Fee Structure:**\n\n"
            "* **Admission Fee (1st Semester):** 65,000 BDT\n"
            "* **Per Semester Fee (2nd to 8th):** ~90,000 BDT (Engineering) / ~50,000–60,000 BDT (BBA & English)\n"
            "* **Total 4-Year Estimated Cost:** ~6,95,000 BDT (B.Sc. Engineering) | ~4,80,000 BDT (BBA/BA)\n\n"
            "📞 Helpline: [01769675588](tel:01769675588), [01769675589](tel:01769675589) | ✉️ [admission@baust.edu.bd](mailto:admission@baust.edu.bd)"
        )

    # 12. Scholarship & Tuition Waivers
    if any(k in norm_q for k in ["waiver", "scholarship", "char"]) or any(k in q_lower for k in ["ওয়েভার", "স্কলারশিপ", "ছাড়"]):
        if is_bn:
            return (
                "**BAUST স্কলারশিপ ও ওয়েভার সুবিধা:**\n\n"
                "* **জিপিএ-৫ ওয়েভার:** এইচএসসিতে জিপিএ ৫.০০ প্রাপ্ত শিক্ষার্থীদের ২০% থেকে ৫০% পর্যন্ত টিউশন ফি ওয়েভার।\n"
                "* **মেধার ভিত্তিতে বৃত্তি:** প্রতি সেমিস্টারে সর্বোচ্চ সিজিপিএ অর্জনকারীদের ২৫% থেকে ১০০% পর্যন্ত মেধা বৃত্তি।\n"
                "* **মুক্তিযোদ্ধা সন্তান ও ভাইবোন কোটা:** ২৫% থেকে ৫০% বিশেষ ছাড় সুবিধা প্রযোজ্য।\n\n"
                "🌐 বিস্তারিত: [www.baust.edu.bd](https://www.baust.edu.bd)"
            )
        return (
            "**BAUST Scholarship & Tuition Waiver Facilities:**\n\n"
            "* **GPA-5.00 Waiver:** 20% to 50% tuition fee waiver for students with Golden/GPA 5.00 in HSC.\n"
            "* **Merit Scholarship:** 25% to 100% tuition waiver awarded to top semester performers based on CGPA.\n"
            "* **Special Quota:** 25% to 50% waiver for siblings and children of Freedom Fighters.\n\n"
            "🌐 Details: [www.baust.edu.bd](https://www.baust.edu.bd)"
        )

    # 13. Residential Halls & Hostels
    if any(k in norm_q for k in ["hall", "hostel", "seat", "provost", "thak"]) or any(k in q_lower for k in ["হল", "হোস্টেল", "প্রভোস্ট"]):
        if is_bn:
            return (
                "**BAUST আবাসিক হল ও প্রভোস্টবৃন্দ:**\n\n"
                "* **ছেলেদের হল:** আব্বাস উদ্দীন হল (প্রভোস্ট: লে. কর্নেল মো. নাজমুল হুদা, পিএসসি)\n"
                "* **মেয়েদের হল:** বীর প্রতীক তারামন বিবি হল (প্রভোস্ট: ড. রুফিজা বেগম)\n"
                "* **সুবিধাসমূহ:** সার্বক্ষণিক ওয়াই-ফাই, নিজস্ব ডাইনিং, সার্বক্ষণিক নিরাপত্তা ও ব্যাকআপ জেনারেটর।\n\n"
                "📞 হেল্পলাইন: [০১৭৬৯৬৭৫৫৮৮](tel:01769675588)"
            )
        return (
            "**BAUST Residential Halls & Facilities:**\n\n"
            "* **Boys' Hall:** Abbas Uddin Hall (Provost: Lt Col Md Nazmul Huda, psc)\n"
            "* **Girls' Hall:** Bir Protik Taramon Bibi Hall (Provost: Dr. Rufiza Begum)\n"
            "* **Facilities:** High-speed Wi-Fi, hygienic dining, 24/7 security & electricity backup.\n\n"
            "📞 Helpline: [01769675588](tel:01769675588)"
        )

    # 14. Grading System & Exam Policy
    if any(k in norm_q for k in ["grading", "grade", "cgpa", "gpa", "marks"]) or any(k in q_lower for k in ["গ্রেড", "গ্রেডিং", "মার্কস"]):
        if is_bn:
            return (
                "**BAUST পরীক্ষার গ্রেডিং নীতিমালা (UGC 4.00 স্কেল):**\n\n"
                "* **৮০% বা তার বেশি:** A+ (জিপিএ ৪.০০)\n"
                "* **৭৫% থেকে < ৮০%:** A (জিপিএ ৩.৭৫)\n"
                "* **৭০% থেকে < ৭৫%:** A- (জিপিএ ৩.৫০)\n"
                "* **৬৫% থেকে < ৭০%:** B+ (জিপিএ ৩.২৫)\n"
                "* **পাস মার্ক ও ন্যূনতম সিজিপিএ:** ৪০% (D গ্রেড, ২.০০) এবং সেমিস্টার উত্তীর্ণ হতে ন্যূনতম সিজিপিএ ২.২০ বজায় রাখা আবশ্যক।"
            )
        return (
            "**BAUST Grading System (UGC Standard 4.00 Scale):**\n\n"
            "* **80% and above:** A+ (Grade Point 4.00)\n"
            "* **75% to < 80%:** A (Grade Point 3.75)\n"
            "* **70% to < 75%:** A- (Grade Point 3.50)\n"
            "* **65% to < 70%:** B+ (Grade Point 3.25)\n"
            "* **Passing Requirement:** 40% (D Grade, 2.00) with a minimum CGPA of 2.20 to maintain good academic standing."
        )

    # 15. Location & Official Contacts
    if any(k in norm_q for k in ["location", "address", "kothay", "kothy", "thikana", "contact", "helpline", "phone", "email"]) or any(k in q_lower for k in ["কোথায়", "কোথায়", "ঠিকানা", "যোগাযোগ", "ফোন"]):
        if is_bn:
            return (
                "**BAUST ক্যাম্পাস ও অফিসিয়াল যোগাযোগ:**\n\n"
                "📍 **ঠিকানা:** সৈয়দপুর সেনানিবাস (Saidpur Cantonment), সৈয়দপুর, নীলফামারী।\n"
                "📞 **ভর্তি হেল্পলাইন:** [০১৭৬৯৬৭৫৫৮৮](tel:01769675588), [০১৭৬৯৬৭৫৫৮৯](tel:01769675589)\n"
                "✉️ **ইমেইল:** [admission@baust.edu.bd](mailto:admission@baust.edu.bd)\n"
                "🌐 **ওয়েবসাইট:** [www.baust.edu.bd](https://www.baust.edu.bd)"
            )
        return (
            "**BAUST Campus Location & Official Contacts:**\n\n"
            "📍 **Address:** Saidpur Cantonment, Saidpur, Nilphamari, Bangladesh.\n"
            "📞 **Admission Helpline:** [01769675588](tel:01769675588), [01769675589](tel:01769675589)\n"
            "✉️ **Email:** [admission@baust.edu.bd](mailto:admission@baust.edu.bd)\n"
            "🌐 **Website:** [www.baust.edu.bd](https://www.baust.edu.bd)"
        )

    # 15.2 Universal Faculty Profile & Contact Details Synthesizer Fallback
    if "Faculty Member Profile & Contact Details" in context:
        lines = []
        for line in context.splitlines():
            l_str = line.strip()
            if l_str.startswith("*"):
                lines.append(l_str)
        if lines:
            res_str = "\n".join(lines)
            return res_str.replace("সাইদপুর", "সৈয়দপুর").replace("সায়েদপুর", "সৈয়দপুর")

    # 15.3 Faculty Routine Short Form Synthesizer Fallback
    if "Faculty Routine Short Form & Teacher Profile" in context:
        lines = []
        for line in context.splitlines():
            l_str = line.strip()
            if l_str.startswith("*"):
                lines.append(l_str)
        if lines:
            res_str = "\n".join(lines)
            return res_str.replace("সাইদপুর", "সৈয়দপুর").replace("সায়েদপুর", "সৈয়দপুর")

    # 15.4 Class Routine & Student Schedule Synthesizer Fallback
    if "Official Class & Lab Routine" in context or "Student Profile & Official Class & Lab Routine" in context:
        st_name = re.search(r'\* Student Name:\s*(.*)', context)
        st_roll = re.search(r'\* Student ID / Roll:\s*(\d+)', context)
        st_prog = re.search(r'\* Programme / Department:\s*(.*)', context)
        st_level = re.search(r'\* Academic Level & Term:\s*(.*)', context)
        st_parity = re.search(r'Roll Parity / Group:\s*(\w+)', context)

        is_tomorrow_weekend = "Tomorrow: Saturday" in context or "Tomorrow: Friday" in context or "আগামীকাল কোনো ক্লাস নেই" in context
        is_today_weekend = "Today: Friday" in context or "Today: Saturday" in context or "আজ কোনো ক্লাস নেই" in context

        res_lines = []
        if st_name and st_roll:
            res_lines.append(f"* **শিক্ষার্থীর নাম (Student Name):** {st_name.group(1).strip()}")
            res_lines.append(f"* **শিক্ষার্থীর আইডি / রোল (Student ID / Roll):** `{st_roll.group(1).strip()}`")
            prog_txt = st_prog.group(1).strip() if st_prog else "কম্পিউটার সায়েন্স অ্যান্ড ইঞ্জিনিয়ারিং"
            lvl_txt = st_level.group(1).strip() if st_level else "লেভেল-৩ টার্ম-২"
            res_lines.append(f"* **বিভাগ ও লেভেল/টার্ম (Department & Level/Term):** {prog_txt} ({lvl_txt})")
            res_lines.append("")

        parity_val = st_parity.group(1).upper() if st_parity else "ODD"

        if any(w in norm_q for w in ["kalke", "কালকে", "আগামীকাল", "tomorrow"]):
            if is_tomorrow_weekend:
                res_lines.append("📅 **আগামীকালের সময়সূচি:** আগামীকাল **শনিবার**, যা বিশ্ববিদ্যালয়ের সাপ্তাহিক ছুটি (Weekend)। তাই আগামীকাল কোনো অ্যাকাডেমিক ক্লাস নেই।")
                res_lines.append("\n📌 **পরবর্তী ক্লাস ডে (রবিবার / Sunday) এর সময়সূচি:**")
                if parity_val == "EVEN":
                    res_lines.append("* `08.00 AM - 08.55 AM`: **CSE 3202** (Artificial Intelligence Sessional) — ল্যাব: `311 (Network Lab)`, শিক্ষক: MZH, MI")
                else:
                    res_lines.append("* `08.00 AM - 08.55 AM`: *CSE 3202 ল্যাবটি শুধুমাত্র EVEN রোলধারীদের জন্য (আপনার রোল ODD হওয়ায় এই সময়ে ক্লাস নেই)।*")
                res_lines.append("* `09.00 AM - 09.55 AM`: **CSE 3207** (Mathematical Analysis for Computer Science) — ল্যাব: `307 (Embedded System Lab)`, শিক্ষক: GR")
                res_lines.append("* `10.00 AM - 10.55 AM`: **CSE 3205** (Computer Networks) — ল্যাব: `307 (Embedded System Lab)`, শিক্ষক: MI")
                res_lines.append("* `11.30 AM - 12.25 PM`: **CSE 3203** (Operating System) — ল্যাব: `307 (Embedded System Lab)`, শিক্ষক: FTZ")
                return "\n".join(res_lines)
            else:
                res_lines.append("📅 **আগামীকালের ক্লাসের সময়সূচি:**")
        elif any(w in norm_q for w in ["ajke", "আজকে", "today"]):
            if is_today_weekend:
                res_lines.append("📅 **আজকের সময়সূচি:** আজ **শুক্রবার**, যা বিশ্ববিদ্যালয়ের সাপ্তাহিক ছুটি (Weekend)। তাই আজ কোনো ক্লাস নেই।")
                res_lines.append("\n📌 **পরবর্তী ক্লাস ডে (রবিবার / Sunday) এর সময়সূচি:**")
                if parity_val == "EVEN":
                    res_lines.append("* `08.00 AM - 08.55 AM`: **CSE 3202** (Artificial Intelligence Sessional) — ল্যাব: `311 (Network Lab)`, শিক্ষক: MZH, MI")
                else:
                    res_lines.append("* `08.00 AM - 08.55 AM`: *CSE 3202 ল্যাবটি শুধুমাত্র EVEN রোলধারীদের জন্য (আপনার রোল ODD হওয়ায় এই সময়ে ক্লাস নেই)।*")
                res_lines.append("* `09.00 AM - 09.55 AM`: **CSE 3207** (Mathematical Analysis for Computer Science) — ল্যাব: `307 (Embedded System Lab)`, শিক্ষক: GR")
                res_lines.append("* `10.00 AM - 10.55 AM`: **CSE 3205** (Computer Networks) — ল্যাব: `307 (Embedded System Lab)`, শিক্ষক: MI")
                res_lines.append("* `11.30 AM - 12.25 PM`: **CSE 3203** (Operating System) — ল্যাব: `307 (Embedded System Lab)`, শিক্ষক: FTZ")
                return "\n".join(res_lines)
        else:
            res_lines.append("📅 **সাপ্তাহিক ক্লাসের সময়সূচি (Weekly Class Schedule - Level 3 Term II):**")
            res_lines.append("\n**রবিবার (Sunday):**")
            if parity_val == "EVEN":
                res_lines.append("* `08.00 AM - 08.55 AM`: **CSE 3202** (AI Sessional) — ল্যাব: `311`, শিক্ষক: MZH, MI [EVEN রোল]")
            else:
                res_lines.append("* `08.00 AM - 08.55 AM`: *CSE 3202 ল্যাবটি শুধুমাত্র EVEN রোলধারীদের জন্য (আপনার রোল ODD হওয়ায় ক্লাস নেই)।*")
            res_lines.append("* `09.00 AM - 09.55 AM`: **CSE 3207** (Mathematical Analysis) — ল্যাব: `307`, শিক্ষক: GR")
            res_lines.append("* `10.00 AM - 10.55 AM`: **CSE 3205** (Computer Networks) — ল্যাব: `307`, শিক্ষক: MI")
            res_lines.append("* `11.30 AM - 12.25 PM`: **CSE 3203** (Operating System) — ল্যাব: `307`, শিক্ষক: FTZ")

            res_lines.append("\n**সোমবার (Monday):**")
            res_lines.append("* `08.00 AM - 08.55 AM`: **CSE 3203** (Operating System) — রুম: `408`, শিক্ষক: FTZ")
            res_lines.append("* `09.00 AM - 09.55 AM`: **CSE 3205** (Computer Networks) — ল্যাব: `307`, শিক্ষক: MI")
            res_lines.append("* `10.00 AM - 10.55 AM`: **CSE 3207** (Mathematical Analysis) — ল্যাব: `307`, শিক্ষক: GR")
            res_lines.append("* `11.30 AM - 12.25 PM`: **CSE 3201** (Artificial Intelligence) — ল্যাব: `307`, শিক্ষক: MZH")
            res_lines.append("* `12.30 PM - 01.25 PM`: **CSE 3200** (Web Engineering Project) — ল্যাব: `411`, শিক্ষক: ASF, GC")

            res_lines.append("\n**মঙ্গলবার (Tuesday):**")
            if parity_val == "ODD":
                res_lines.append("* `08.00 AM - 08.55 AM`: **CSE 3210** (ISD Sessional) — ল্যাব: `402 (Microprocessor Lab)`, শিক্ষক: AKZ, RH [ODD রোল]")
            else:
                res_lines.append("* `08.00 AM - 08.55 AM`: *CSE 3210 ল্যাবটি শুধুমাত্র ODD রোলধারীদের জন্য।*")
            res_lines.append("* `09.00 AM - 09.55 AM`: **CSE 3205** (Computer Networks) — রুম: `304`, শিক্ষক: MI")
            res_lines.append("* `10.00 AM - 10.55 AM`: **CSE 3201** (Artificial Intelligence) — রুম: `304`, শিক্ষক: MZH")
            res_lines.append("* `11.30 AM - 12.25 PM`: **CSE 3209** (Information System Design) — রুম: `304`, শিক্ষক: AKZ")

            res_lines.append("\n**বুধবার (Wednesday):**")
            res_lines.append("* `08.00 AM - 08.55 AM`: **CSE 3204** (OS Sessional) — ল্যাব: `411 (Database Lab)`, শিক্ষক: FTZ, RH")
            res_lines.append("* `09.00 AM - 09.55 AM`: **CSE 3209** (Information System Design) — রুম: `304`, শিক্ষক: AKZ")
            res_lines.append("* `10.00 AM - 10.55 AM`: **CSE 3203** (Operating System) — রুম: `304`, শিক্ষক: FTZ")
            res_lines.append("* `11.30 AM - 12.25 PM`: **CSE 3201** (Artificial Intelligence) — রুম: `304`, শিক্ষক: MZH")

            res_lines.append("\n**বৃহস্পতিবার (Thursday):**")
            res_lines.append("* `08.00 AM - 08.55 AM`: **CSE 3209** (Information System Design) — রুম: `204`, শিক্ষক: AKZ")
            res_lines.append("* `09.00 AM - 09.55 AM`: **CSE 3207** (Mathematical Analysis) — রুম: `204`, শিক্ষক: GR")
            res_lines.append("* `10.00 AM - 10.55 AM`: **CSE 3206** (Computer Networks Sessional) — ল্যাব: `311 (Network Lab)`, শিক্ষক: MI, RH")
            return "\n".join(res_lines)

    # 15.5 Multiple Student Matches Disambiguation Fallback
    if "Multiple Student Matches Found" in context or "Disambiguation Required" in context:
        student_bullets = []
        for line in context.splitlines():
            l_s = line.strip()
            if re.match(r'^\d+\.\s*\*\*', l_s):
                student_bullets.append(f"* {l_s}")

        matched_name = ""
        nm_m = re.search(r"Multiple Student Matches Found for Name ['\"]([^'\"]+)['\"]", context)
        if nm_m:
            matched_name = f"'{nm_m.group(1)}' "

        msg = (
            f"আমাদের ডাটাবেসে {matched_name}নামের একাধিক শিক্ষার্থী পাওয়া গেছে। "
            f"আপনার সঠিক পরীক্ষার আসন ও সময়সূচি জানতে অনুগ্রহ করে আপনার নির্দিষ্ট **Student ID / Roll** অথবা পুরো নাম উল্লেখ করুন।\n\n"
            f"**সম্ভাব্য শিক্ষার্থীদের তালিকা:**\n"
        )
        if student_bullets:
            msg += "\n".join(student_bullets)
        else:
            msg += "* অনুগ্রহ করে আপনার আইডি বা রোল নম্বরটি লিখে পাঠান।"
        return msg

    # 16. Smart fallback from internal RAG document context (Clean, well-structured, no raw table noise)
    if context and context.strip() and "(No specific internal BAUST document found" not in context:
        cleaned = re.sub(r'---\s*Information from \[.*?\]\s*---', '', context).strip()
        lines = []
        for line in cleaned.splitlines():
            l = line.strip()
            if not l or l.startswith('|---') or l.startswith('===') or l.startswith('---') or l.startswith(':-') or l.startswith(':--'):
                continue
            if any(junk in l.upper() for junk in ["ADMISSION NOTICE", "WINTER SEMESTER", "APPLICATIONS ARE INVITED", "SL. — FACULTY", "SL.", "| :---", "CRITICAL INSTRUCTION", "DO NOT PICK", "POLITELY INFORM", "MATCHING STUDENTS"]):
                continue
            # If it's a table row
            if l.startswith('|') and l.endswith('|'):
                parts = [p.strip() for p in l.split('|') if p.strip() and not p.strip().startswith(':--') and not p.strip().startswith('---')]
                if parts and len(parts) > 1:
                    if any(h in parts[0].lower() for h in ["sl.", "sl", "no.", "serial"]):
                        continue
                    lines.append("* " + " — ".join(parts[:3]))
                    continue
            l_clean = re.sub(r'^[\d\.\-\*\#\>\|\:\s]+', '', l).strip()
            if len(l_clean) > 15:
                lines.append(f"* {l_clean}")
        
        if lines:
            trimmed = lines[:4]
            if is_bn:
                return "**প্রাসঙ্গিক তথ্য সংক্ষেপ:**\n\n" + "\n".join(trimmed)
            return "**Summary of Relevant Information:**\n\n" + "\n".join(trimmed)

    # 17. General helpful response
    if is_bn:
        return (
            "আপনার প্রশ্নটির সুনির্দিষ্ট তথ্য পেতে অনুগ্রহ করে BAUST ভর্তি ও তথ্য হেল্পলাইনে যোগাযোগ করুন:\n\n"
            "📞 **[০১৭৬৯৬৭৫৫৮৮](tel:01769675588)**, **[০১৭৬৯৬৭৫৫৮৯](tel:01769675589)** | ✉️ **[admission@baust.edu.bd](mailto:admission@baust.edu.bd)**"
        )
    return (
        "For specific details regarding your query, please contact the official BAUST Helpline:\n\n"
        "📞 **[01769675588](tel:01769675588)**, **[01769675589](tel:01769675589)** | ✉️ **[admission@baust.edu.bd](mailto:admission@baust.edu.bd)**"
    )

def _build_full_prompt(context: str, question: str, history: list = None) -> str:
    """Constructs the optimized prompt for high accuracy, strict language matching, and concise answers."""
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

    # Check if this turn is a language translation or conversion instruction
    is_trans, trans_instruction = is_translation_request(question)
    extra_task_instruction = ""
    if is_trans and trans_instruction:
        extra_task_instruction = f"\nSPECIAL TRANSLATION/CONVERSION TASK: {trans_instruction}\n"

    system_instruction = f"""
You are the official and intelligent AI Assistant for Bangladesh Army University of Science and Technology (BAUST), Saidpur.

CRITICAL RULES (CONCISE, DIRECT, STRUCTURED, LANGUAGE ACCURATE):
1. **STRICT LANGUAGE MATCHING & TRANSLATION**:
   - **Bengali / Banglish Question**: If the user asks in Bengali (বাংলা) or Banglish (e.g., 'faculty koyta', 'admission fee koto', 'cse head ke', 'kothay baust'), reply **100% in fluent, natural Bengali (বাংলা)**.
   - **English Question**: If the user asks in English (e.g., 'how many faculties', 'who is vc', 'tuition fees', 'where is baust'), reply **100% in crisp, professional English**.
   - **Translation / Conversion Requests**: If the user asks to translate, convert, or rephrase the previous conversation (e.g., "eita english e bolo", "translate to bangla", "in english", "short e bolo"), IMMEDIATELY translate or convert the previous response accurately and directly!

2. **PROPORTIONAL, STRUCTURED & TIDY (GUCHIYE ANSWER DAWA)**:
   - Always format with short markdown bullet points and bold field labels (e.g., `* **শিক্ষার্থীর নাম:** ...`, `* **সিট নম্বর (Seat No):** ...`).
   - For key values (e.g. Seat Numbers like `C-206-19`, Exam Rooms like `Room 206`, Roll/Student IDs like `0802620102161056`, Exam Slots like `Slot-C`), wrap them in backticks (e.g. ` `C-206-19` `, ` `Room 206` `) so they render as distinct highlighted badges with eye-catching colors!
   - **Student Exam Seatplan, Name & Time Rule (CRITICAL)**:
     * When answering for a student's roll/ID or seatplan/exam question:
       The student's NAME MUST BE ON ITS OWN DEDICATED SEPARATE BULLET POINT at the very top:
       `* **শিক্ষার্থীর নাম (Student Name):** <Student Name>`
       NEVER merge, combine, or bundle the student's name into parentheses on the ID line (e.g., NEVER write `* **শিক্ষার্থী / আইডি:** 0802410105101027 (KAZI SAIYATAN HAQUE SADIA)`).
       Always present the fields clearly on separate individual bullets in this exact format:
       * **শিক্ষার্থীর নাম (Student Name):** <Student Name>
       * **শিক্ষার্থীর আইডি / রোল (Student ID / Roll):** `<Roll>`
       * **বিভাগ ও লেভেল/টার্ম (Department & Level/Term):** <Programme / Level & Term>
       * **পরীক্ষার স্লট (Examination Slot):** `<Slot>`
       * **পরীক্ষার সময় (Examination Time):** `<Time>`
       * **পরীক্ষার কক্ষ (Exam Room):** <Building, Room>
       * **সিট নম্বর (Seat No):** `<Seat>`
       * **পরীক্ষা:** মিডটার্ম পরীক্ষা, সামার সেমিস্টার ২০২৬

   - **Student Class Routine & Schedule Rule (CRITICAL)**:
     * When a student asks about their class routine, today's or tomorrow's classes (e.g., "kalke amar ki ki class ache", "ajke amar ki class", "amar class routine ki", "0802410205101019 kalke ki class", "cse 3-2 routine"):
       1. **Student Header**: If student info is in the retrieved document, present the student's identity at the top:
          * **শিক্ষার্থীর নাম (Student Name):** <Student Name>
          * **শিক্ষার্থীর আইডি / রোল (Student ID / Roll):** `<Roll>`
          * **বিভাগ ও লেভেল/টার্ম (Department & Level/Term):** <Department> (<Level/Term>)
       2. **Temporal & Holiday Awareness**:
          * Friday (শুক্রবার) and Saturday (শনিবার) are WEEKLY HOLIDAYS (সাপ্তাহিক ছুটি - শুক্র ও শনিবার বিশ্ববিদ্যালয় বন্ধ থাকে, কোনো অ্যাকাডেমিক ক্লাস হয় না)।
          * If tomorrow is Friday or Saturday (or today is Friday/Saturday and asking for "kalke"): Explicitly state that tomorrow is a weekly holiday, so there are no classes scheduled. Then present the schedule for the next regular academic class day (রবিবার / Sunday).
          * If today is a regular class day (Sunday to Thursday) and asking for today/tomorrow: Present the exact classes scheduled for that specific day.
       3. **Accurate Class List Details**:
          For each class scheduled on that day from the routine table:
          * **সময় (Time):** `<Time Slot>`
          * **বিষয় ও কোড (Course Code & Title):** `<Code>: <Title>`
          * **কক্ষ / ল্যাব (Room / Lab):** `<Room>`
          * **শিক্ষক (Faculty):** `<Teacher Short Form>`
       4. **LAB EVEN/ODD SECTION & DURATION RULE (CRITICAL)**:
          * In BAUST routines, all Sessional Labs (e.g. CSE 3202, CSE 3204, CSE 3206, CSE 3210, CSE 3200) run for 3 contact hours (3 consecutive class periods):
            - Morning lab (Periods 1-3): `08:00 AM - 10:55 AM` (3 hours)
            - Midday lab (Periods 4-6): `11:30 AM - 02:25 PM` (3 hours)
            - Afternoon lab (Periods 7-9): `02:30 PM - 05:25 PM` (3 hours)
          * Always display the full 3-hour duration for sessional labs as given in the verified schedule (e.g. `08:00 AM - 10:55 AM`). NEVER display a 3-hour sessional lab as only 55 minutes (`08:00 AM - 08:55 AM`)!
          * In BAUST routines, the labels `#EVEN#` and `#ODD#` for sessional labs represent SECTION assignments:
            - `#EVEN#` indicates Section A's class.
            - `#ODD#` indicates Section B's class.
          * These labels do NOT refer to student roll numbers! They are NOT for even/odd rolls! All students of Section A attend Section A's classes, and all students of Section B attend Section B's classes, regardless of whether their individual roll ends in an even or odd digit.
          * NEVER state that a lab is only for even or odd roll holders! NEVER write "(বিঃদ্রঃ এটি #EVEN# ল্যাব, তাই জোড় রোলধারীদের জন্য প্রযোজ্য)" or similar!
          * Present the scheduled class simply by its Course Code & Title, Room/Lab, and Teacher.
       5. **NEVER substitute a class routine question with only an exam seatplan!** Always provide their class routine first.
       6. **VERIFIED AUTHORITATIVE SCHEDULE (MANDATORY)**:
          * If the retrieved document contains a "VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR STUDENT" or "VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR BATCH", you MUST directly present that schedule!
          * In BAUST CSE Level 3 Term 2 Summer 2026, Sunday (Sun) is a REGULAR academic class day for both Section A and Section B. Present the exact classes from the verified schedule in context!
          * NEVER hallucinate or state that Sunday is empty or has no classes for CSE 3-II! Sunday classes start from 08:00 AM.
          * When presenting the schedule, copy the exact times, rooms, courses, and faculties from the verified schedule. NEVER invent fake classes or rooms (e.g., Room 408 when the schedule says 307 or 311).

   - **Graduate / Masters & MBA Programs Rule (CRITICAL)**:
     * When asked whether Masters or MBA is available at BAUST ("masters kora jay?", "MBA ache?", "postgraduate program ki ki", "MBA kora jay?"):
     * Clearly confirm that **হ্যাঁ, BAUST সৈয়দপুরে মাস্টার্স (স্নাতকোত্তর) ও MBA করার পূর্ণ সুযোগ রয়েছে।**
     * বিশ্ববিদ্যালয়ের **ব্যবসায় প্রশাসন বিভাগ (Department of Business Administration - DBA)** / **Faculty of Business Studies (FBS)** থেকে **Master of Business Administration (MBA)** অফার করা হয়।
     * আপনি অবশ্যই সর্বদা নিচের ২টি প্রধান MBA ট্র্যাক বিস্তারিতভাবে উল্লেখ করবেন:
       1. **Regular MBA (রেগুলার এমবিএ):**
          - **BBA গ্র্যাজুয়েটদের জন্য:** ১ বছর মেয়াদি (২ সেমিস্টার, ৩৬ ক্রেডিট)।
          - **অন্যান্য নন-বিজনেস ব্যাকগ্রাউন্ড (B.Sc, BA, BSS ইত্যাদি) গ্র্যাজুয়েটদের জন্য:** ২ বছর মেয়াদি (৪ সেমিস্টার, ৬০ ক্রেডিট)।
       2. **Executive MBA / EMBA (এক্সিকিউটিভ এমবিএ):**
          - চাকুরিজীবী বা কর্মরত পেশাজীবীদের (Working Professionals) জন্য বিশেষ সুবিধাজনক সময়ে (উইকেন্ড/সান্ধ্যকালীন ক্লাস) পরিচালিত হয়। মেয়াদ ১.৫ বছর (৩ সেমিস্টার, ৪৮ ক্রেডিট)।
     * **MBA প্রোগ্রাম কো-অর্ডিনেটর:** মো. জাবের হোসেন (সহযোগী অধ্যাপক, DBA), মোবাইল: [01922626111](tel:01922626111), ইমেইল: [hossain.zaber478@gmail.com](mailto:hossain.zaber478@gmail.com)।
     * ভর্তি অফিসের হটলাইন: [01769675580](tel:01769675580), [01769675588](tel:01769675588)।
     * এছাড়াও ইংরেজি বিভাগ থেকে **MA in English (ELT)** (১ বছর মেয়াদি, ৩৬ ক্রেডিট) পরিচালিত হয়।
     * **ইঞ্জিনিয়ারিং মাস্টার্স সংক্রান্ত সতর্কতা:** BAUST-এ বর্তমানে ইঞ্জিনিয়ারিংয়ে (CSE, EEE, ME, CE, IPE) মাস্টার্স / M.Sc চালু নেই (প্রক্রিয়াধীন)। তাই ইঞ্জিনিয়ারিংয়ে মাস্টার্স আছে বলে কখনো ভুল তথ্য বা বিভ্রান্তি তৈরি করবেন না。

   - **Student Name Disambiguation & Anti-Random Guessing Rule (CRITICAL)**:
     * When a user searches by a first name, partial name, or single word (e.g. 'nafisa', 'rakib', 'hasan', 'sadia') and multiple students exist in the records with that name:
       - NEVER arbitrarily pick or guess just ONE student!
       - Explain in polite, natural Bengali that multiple students share this name.
       - List the matching students in clean bullet points with their Full Name, Student ID / Roll, and Level/Term.
       - Ask the user to reply with their specific Student ID / Roll or full name to get their exact exam seat, room, and time.
     * If ONLY ONE student matches the full name, or if the user provided their Student ID / Roll:
       - If the user asks about Class Routine, Today's or Tomorrow's classes: Follow the Student Class Routine & Schedule Rule above!
       - If the user asks about Exam Seat / Room, or simply inputs their ID: Directly output their full individual exam seat allocation as defined above!

   - For single-fact questions, give a crisp 1 to 3 bullet point answer.
   - NEVER output raw unformatted tables, lengthy irrelevant admissions text, or bank routing numbers unless specifically asked.
   - For coding/general science questions (e.g., Python binary search), directly provide clean code without university admission templates.

3. **CLICKABLE CONTACTS & LINKS**:
   - Format phone: [01769675588](tel:01769675588)
   - Format email: [admission@baust.edu.bd](mailto:admission@baust.edu.bd)
   - NEVER wrap Markdown links or contacts in backticks (write [01769675588](tel:01769675588), NOT `[01769675588](tel:...)`).
   - MULTIPLE CONTACTS FORMAT (MANDATORY): If an entity has multiple emails or phones (e.g. rasel@baust.edu.bd and rasel17716@cseku.ac.bd), format each as an individual, separate clickable link separated by comma:
     `[rasel@baust.edu.bd](mailto:rasel@baust.edu.bd), [rasel17716@cseku.ac.bd](mailto:rasel17716@cseku.ac.bd)`
     NEVER put multiple emails inside a single markdown link like `[email1, email2](mailto:email1, email2)`!
   - For phone with notes (like WhatsApp), put the note outside the link:
     `[01759810833](tel:01759810833) (WhatsApp)`

4. **FACULTY MEMBERS & TEACHER DIRECTORY RULES (STRICT SEPARATION)**:
   - When asked for teachers / faculty members of a department (e.g. CSE, EEE, ME, CE, IPE, BBA, English):
     * **সক্রিয় শিক্ষকবৃন্দ (Active Faculty Members)**: List the current active teachers with their designation and contact info. When the user asks for all teachers ("sob teacher", "all faculty", "teachers list"), provide the COMPLETE list of active faculty from the records without omitting anyone.
     * **উচ্চশিক্ষার ছুটিতে থাকা শিক্ষকবৃন্দ (Faculty on Study Leave)**: If any teachers are on study leave, list them under a distinct, separate section clearly titled "উচ্চশিক্ষার ছুটিতে (Study Leave)".
     * **সাবেক শিক্ষকবৃন্দ (Ex-Faculty Members)**: If any former/ex-faculty are mentioned, list them ONLY under a distinct separate section titled "সাবেক শিক্ষকবৃন্দ (Ex-Faculty)" or omit them if the user asked only for current teachers.
     * **NEVER** combine or confuse active teachers with ex-faculty or teachers on study leave.

5. **STRICT DEPARTMENT & FACTUAL INTEGRITY (NEVER SUBSTITUTE / HONEST MISSING DATA)**:
   - **No Cross-Department Substitution**: If the user asks about a specific department (e.g. `CSE`, `CE`, `EEE`, `ME`, `IPE`, `BBA`, `English`, `ICT`):
     * You must **ONLY** provide information that strictly belongs to that requested department.
     * **NEVER substitute or mix one department's data with another!**
     * **Strict Rule**: If the user asks for "cse picnic" or "সিএসই পিকনিক", and the official records do not contain a CSE picnic, **DO NOT** give CE (Civil Engineering) or EEE picnic details!
   - **Missing Information Protocol**:
     * If the official records DO NOT contain information about the user's specific query for that department (e.g., no record of a CSE picnic or specific departmental event), you MUST state directly, clearly and politely in the user's language:
       "আমাদের অফিসিয়াল রেকর্ড বা ডাটাবেসে সিএসই (CSE) বিভাগের পিকনিক সংক্রান্ত কোনো তথ্য বা নোটিশ নেই।" (or in English: "Sorry, no information regarding a CSE department picnic was found in our official university records.")
     * **NEVER guess, NEVER hallucinate, and DO NOT output unrelated documents (such as exam seatplans or general admission text) when the specific event/notice is missing.**
   - **Instant Resolution on User Entity Corrections**:
     * If the user corrects a mistaken department or entity from a previous turn (e.g. "ami ce bolini ami bolechi cse" or "not CE, I meant CSE"):
     * Do NOT simply say an empty apology asking how you can help. **Immediately resolve the original question for the corrected department**:
       e.g., "দুঃখিত ভুল বোঝাবুঝির জন্য। আমাদের অফিসিয়াল রেকর্ড বা ডাটাবেসে সিএসই (CSE) বিভাগের পিকনিক সংক্রান্ত কোনো তথ্য বা নোটিশ নেই।"
6. **ACCURATE BENGALI SPELLING & ACADEMIC TERMINOLOGY (CRITICAL)**:
   - **Saidpur Spelling Rule (MANDATORY)**: The official name and location of the university is **সৈয়দপুর** বা **সৈয়দপুর** (e.g., "বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুর")।
     **NEVER EVER write 'সাইদপুর' or 'সায়েদপুর'!** If 'সাইদপুর' is written, it is an unacceptable spelling mistake.
   - **Teacher & Staff Contact Details Rule (CRITICAL)**:
     * When asked for any teacher's or staff's contact info (e.g., "AA er contact", "Al Amin er phone number", "email ki", "contact"):
     * If the teacher's phone number or email is present in the retrieved context, you MUST directly provide their specific phone number and email!
     * NEVER state that they must contact the department or give an alumni email (like info.ipe@baust.edu.bd) when their personal/direct contact is in the context!
   - **Teacher / Faculty Routine Short Form Rule (CRITICAL)**: In BAUST academic routines, teachers are referenced by initials/short forms (e.g. `AA` = Md. Al Amin / মো. আল আমিন, `GR` = S. M Golam Rifat / এস. এম. গোলাম রিফাত, `FTZ` = Fatema Tuz Zohra, `AKZ` = A K Z Rasel Rahman, `MI` = Md. Minhazul Islam, `MZH` = Md Zahim Hassan, `HM` = Hrittik Mural, `RH` = MD. Rabby Hossain, `GC` = Gourab Chandra Biswas, `ASF` = Md. Asif Iqbal, `TT` = Tasfia Tasnim). When asked who a teacher is or their contact (e.g. 'AA er contact', 'GR ke', 'who is AA'):
     * NEVER guess or hallucinate any other name!
     * Always output their exact full name, designation, department, phone, and email as stated in the retrieved documents.
   - For 'Semester Break', ALWAYS write **সেমিস্টার ব্রেক** বা **সেমিস্টার বিরতি** (NEVER spell as 'ব্র্যাক' or 'ব্রাক').
   - For 'Midterm', write **মিডটার্ম** বা **মিড-টার্ম পরীক্ষা**।
   - Maintain accurate standard Bengali spelling and avoid phonetic transliteration errors.

7. **CONVERSATIONAL FOLLOW-UP & SHORT QUERIES**:
   - If the user asks a brief follow-up question (e.g. 'time', 'kobe', 'room kothay', 'seat kothay', 'slot koto', 'routine', 'details', 'contact', 'phone'):
     Check the RECENT CONVERSATION HISTORY to identify the subject (teacher, student, course, exam, department, or topic).
     Answer directly and specifically with that subject's details (e.g., student's exam slot, time, room, seat, or teacher's phone & email) in the clean individual bullet format!

8. **AUTHORITATIVE ENTITY PROFILE & RECORDS (HIGHEST PRIORITY)**:
   - When the retrieved context contains an `### AUTHORITATIVE ENTITY PROFILE & RECORDS:` section:
     * This section contains 100% verified, ground-truth data from official BAUST records.
     * If it contains contact details (Phone, Email, WhatsApp) for a teacher (e.g., `AA` -> `Md. Al Amin`, `GR` -> `S. M Golam Rifat`):
       YOU MUST ALWAYS DIRECTLY OUTPUT THEIR PHONE AND EMAIL IN CLEAN SEPARATE CLICKABLE FORMAT:
       `* **মোবাইল / ফোন (Phone):** [01759810833](tel:01759810833) (WhatsApp)`
       `* **ইমেইল (Email):** [email1@baust.edu.bd](mailto:email1@baust.edu.bd), [email2@gmail.com](mailto:email2@gmail.com)`
       (Each email/phone must be its own individual link! Never group multiple addresses into a single mailto or tel tag!)
     * NEVER omit the contact details, and NEVER say the user must contact the department office or alumni email when the teacher's direct phone and email are right in the Authoritative Entity Profile!
     * NEVER confuse a faculty member (e.g., Lecturer Md. Al Amin) with any lower staff or messenger who happens to share the same name.

9. **ACADEMIC QUALIFICATIONS & DEGREE TRANSLATION (CRITICAL)**:
   - When presenting educational qualifications (শিক্ষাগত যোগ্যতা):
   - NEVER create redundant nested parentheses or repetitive phrases like `বি.এসসি (বি.এসসি. ইন সি.এসই, কেইউ)` or `এম.এসসি (এম.এসসি. ইন ...)`!
   - NEVER phonetically transliterate university acronyms into awkward phonetic letters like "কেইউ" (write "খুলনা বিশ্ববিদ্যালয় (KU)" or keep "KU") or "জেইউ" (write "জাহাঙ্গীরনগর বিশ্ববিদ্যালয় (JU)" or "JU")!
   - Format qualifications naturally, accurately, and cleanly:
     * Either keep recognized standard degree format: `B.Sc. in CSE (KU), M.Sc. in CSE (KU)`
     * Or in natural Bengali: `বিএসসি ইন সিএসই (খুলনা বিশ্ববিদ্যালয় / KU), এমএসসি ইন সিএসই (KU)`
     * Degrees like B.Sc. should be written as `বিএসসি` (NOT `বি.এসসি`), CSE as `সিএসই` (NOT `সি.এসই`).
{extra_task_instruction}"""

    return f"""{system_instruction}

{history_str}
OFFICIAL BAUST UNIVERSITY RECORDS:
{context if context and context.strip() else "(No specific internal BAUST document found - answer with general helpful AI knowledge)"}

USER QUESTION:
{question}

ASSISTANT RESPONSE (Neat, comprehensive, structured bullets, language-matched):"""



def generate_answer_stream(context: str, question: str, history: list = None):
    """
    Real-time streaming generator yielding tokens directly to the user (SSE).
    Provides instant < 300ms time-to-first-token with automatic multi-model fallback.
    """
    global _last_working_model
    if is_greeting(question):
        conv_resp = get_conversational_response(question)
        for word in conv_resp.split(" "):
            yield word + " "
            time.sleep(0.012)
        return

    # Check In-Memory Cache (Only if no long history is present)
    cache_key = build_cache_key(question, context)
    if not history or len(history) <= 1:
        cached = get_cached_response(cache_key)
        if cached:
            for chunk in cached.split(" "):
                yield chunk + " "
                time.sleep(0.01)
            return

    prompt = _build_full_prompt(context, question, history)
    key = os.getenv("GEMINI_API_KEY")
    full_answer_accumulator = []

    # Stream in real-time via high-speed direct REST SSE
    streamed_successfully = False
    if key:
        env_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        candidate_stream_models = [
            "gemini-3.5-flash-lite",
            _last_working_model,
            env_model,
            "gemini-flash-latest",
            "gemini-3.8-flash"
        ]
        unique_stream_models = [m for m in dict.fromkeys(candidate_stream_models) if m]

        for model_to_use in unique_stream_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_use}:streamGenerateContent?alt=sse&key={key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.25}
            }
            try:
                resp = _http_session.post(url, json=payload, stream=True, timeout=12)
                if resp.status_code != 200:
                    continue

                for line in resp.iter_lines():
                    if line and line.startswith(b"data:"):
                        raw = line[5:].decode("utf-8").strip()
                        if raw:
                            try:
                                data = json.loads(raw)
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            token = part["text"]
                                            token_clean = token.replace("সাইদপুর", "সৈয়দপুর").replace("সায়েদপুর", "সৈয়দপুর")
                                            full_answer_accumulator.append(token_clean)
                                            yield token_clean
                                            streamed_successfully = True
                            except Exception:
                                pass
                if streamed_successfully:
                    accum_text = "".join(full_answer_accumulator).strip()
                    if len(accum_text) < 80 and len(context) > 120:
                        streamed_successfully = False
                        full_answer_accumulator.clear()
                    else:
                        _last_working_model = model_to_use
                        break
            except Exception:
                continue

    # Fallback to static generation if streaming API encountered network/timeout issue
    if not streamed_successfully:
        static_answer = call_gemini_generate(prompt, temperature=0.25)
        if not static_answer:
            static_answer = synthesize_smart_answer(context, question, history=history)

        for chunk in static_answer.split(" "):
            full_answer_accumulator.append(chunk + " ")
            yield chunk + " "
            time.sleep(0.008)

    complete_text = "".join(full_answer_accumulator).strip()
    if complete_text:
        set_cached_response(cache_key, complete_text)


def generate_answer(context: str, question: str, history: list = None) -> str:
    """
    Generate a high-speed, structured answer with automatic caching.
    """
    if is_greeting(question):
        return get_conversational_response(question)

    cache_key = build_cache_key(question, context)
    if not history or len(history) <= 1:
        cached = get_cached_response(cache_key)
        if cached:
            return cached

    prompt = _build_full_prompt(context, question, history)
    ans = call_gemini_generate(prompt, temperature=0.25)
    
    if not ans or not ans.strip() or (len(ans.strip()) < 80 and len(context) > 120):
        ans = synthesize_smart_answer(context, question, history=history)

    final_ans = sanitize_contact_markdown(ans.strip())
    set_cached_response(cache_key, final_ans)
    return final_ans