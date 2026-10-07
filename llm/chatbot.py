import os
import re
import time
import json
import threading
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = None
try:
    from google import genai
    if api_key:
        client = genai.Client(api_key=api_key)
except Exception as e:
    print(f"[Gemini Client Init Warning]: {e}")
    client = None

# Primary model identifier (Fast, highly capable, and stable)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Reuse persistent HTTP session for connection pooling & low latency
_http_session = requests.Session()
_last_working_model = DEFAULT_MODEL

# In-Memory Fast Cache for Instant (<10ms) Responses
_RESPONSE_CACHE = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour TTL
_CACHE_LOCK = threading.Lock()

def normalize_text_for_intent(text: str) -> str:
    """Normalize text by collapsing repeating characters and punctuation."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r'[!?,.\'\"]+', ' ', t)
    # Collapse repeating consecutive characters: e.g. 'naam' -> 'nam', 'kemon' -> 'kemon'
    t = re.sub(r'([a-z])\1+', r'\1', t)
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



def get_cached_response(key: str) -> str:
    """Retrieve response from cache if not expired."""
    norm_key = key.strip().lower()
    with _CACHE_LOCK:
        if norm_key in _RESPONSE_CACHE:
            entry = _RESPONSE_CACHE[norm_key]
            if time.time() - entry["time"] < _CACHE_TTL_SECONDS:
                return entry["answer"]
            else:
                del _RESPONSE_CACHE[norm_key]
    return None


def set_cached_response(key: str, answer: str):
    """Store response into memory cache with size control."""
    if not key or not answer:
        return
    norm_key = key.strip().lower()
    with _CACHE_LOCK:
        # Cap cache size at 500 items to keep memory lightweight
        if len(_RESPONSE_CACHE) > 500:
            oldest_key = min(_RESPONSE_CACHE.keys(), key=lambda k: _RESPONSE_CACHE[k]["time"])
            del _RESPONSE_CACHE[oldest_key]
        _RESPONSE_CACHE[norm_key] = {
            "answer": answer,
            "time": time.time()
        }


def clean_llm_response(text: str) -> str:
    """Strip internal chain-of-thought, scratchpad reasoning traces, or formatting artifacts from LLMs."""
    if not text:
        return ""
    
    # If model output contains internal reasoning steps / scratchpad
    if "* User Question:" in text or "### Analysis" in text or "* Input:" in text or "* Constraint:" in text:
        match = re.search(r'(?:### Response:|### Answer:|Final Answer:|\*\*উত্তর:\*\*|\*\*Answer:\*\*)\s*(.*)', text, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()
        
        lines = []
        skip_thought = True
        for line in text.splitlines():
            if skip_thought:
                if re.match(r'^\s*[\*\-]\s*(User Question|Input|Language|Context|Constraint|Goal|Greeting|Acknowledgment|Combined|Addresses|Reasoning):', line, re.IGNORECASE):
                    continue
                if line.strip().startswith(('1.', '2.', '3.')) and any(k in line for k in ['Input', 'Question', 'Constraint', 'Goal']):
                    continue
                if not line.strip():
                    continue
                skip_thought = False
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        if cleaned:
            text = cleaned

    # Normalize accidental backticks around markdown links: `[text](url)` -> [text](url)
    text = re.sub(r'`(\[[^\]]+\]\([^\)]+\))`', r'\1', text)
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
    If conversation history exists and question genuinely uses pronouns (e.g. 'tar phone koto', 'oitar fee koto'),
    expand the question into a clear standalone search query without polluting new independent questions.
    """
    if not history or not isinstance(history, list) or len(history) < 2:
        return question

    # If it's a translation/conversion command, don't mangle the query
    is_trans, _ = is_translation_request(question)
    if is_trans:
        return question

    q_lower = question.strip().lower()
    q_tokens = set(q_lower.split())

    # Standalone topic keywords that indicate a completely new question
    standalone_topics = {
        "dept", "department", "departments", "koyta", "koyti", "fee", "fees", "cost", 
        "admission", "eligibility", "hostel", "hall", "grading", "grade", "cgpa", 
        "vc", "registrar", "cse", "eee", "me", "ce", "ipe", "bba", "english", 
        "location", "address", "helpline", "phone", "email", "scholarship", "waiver",
        "বিভাগ", "অনুষদ", "ভর্তি", "ফি", "খরচ", "হল", "হোস্টেল", "উপাচার্য", "ভিসি"
    }
    
    # If the user is asking an independent topic question, keep it clean
    if len(q_tokens.intersection(standalone_topics)) > 0 and not any(p in q_tokens for p in ["tar", "tader", "ota", "oita", "তার", "তাদের", "সেটার"]):
        return question

    dependent_pronouns = {"tar", "tader", "ota", "oita", "he", "she", "his", "her", "their", "it", "its"}
    has_dependent_pronoun = len(q_tokens.intersection(dependent_pronouns)) > 0 or any(p in q_lower for p in ["তার", "তাদের", "ওটা", "সেটার", "উনার"])

    if not has_dependent_pronoun:
        return question

    last_user_msg = ""
    last_bot_msg = ""
    for msg in reversed(history[-4:]):
        if isinstance(msg, dict):
            if msg.get("role") in ["user", "student"] and not last_user_msg:
                last_user_msg = str(msg.get("content", "")).strip()
            elif msg.get("role") in ["assistant", "model", "bot"] and not last_bot_msg:
                last_bot_msg = str(msg.get("content", "")).strip()[:150]

    if not last_user_msg:
        return question

    prompt = f"""Given the conversation context, rewrite the follow-up question into a single specific search query about BAUST university.
Previous User Question: {last_user_msg}
Assistant Summary: {last_bot_msg}
Follow-up Question: {question}

Output ONLY the rewritten search query (1 short phrase/sentence):"""

    try:
        rewritten = call_gemini_generate(prompt, temperature=0.0)
        if rewritten and len(rewritten) < 120 and "query" not in rewritten.lower():
            return rewritten.strip()
    except Exception:
        pass

    return f"{last_user_msg} {question}"



def call_gemini_generate(prompt: str, temperature: float = 0.2, preferred_model: str = None) -> str:
    """
    Ultra-fast, robust Gemini caller with persistent connection pooling
    and deterministic low-latency fallbacks.
    """
    global _last_working_model
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return ""

    env_model = os.getenv("GEMINI_MODEL")
    candidate_models = [
        preferred_model,
        _last_working_model,
        env_model,
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.7-flash"
    ]
    unique_models = [m for m in dict.fromkeys(candidate_models) if m]

    for model_name in unique_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature}
        }
        try:
            resp = _http_session.post(url, json=payload, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        _last_working_model = model_name
                        raw_text = parts[0]["text"].strip()
                        return clean_llm_response(raw_text)
        except Exception:
            continue

    return ""


def clean_raw_icon_artifacts(text: str) -> str:
    """Strip OCR icon artifacts where icons were incorrectly converted to stray characters."""
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r'^[9j2JУyY\)\*\(\#\©\•\-\>\:\;]\s*(?=(\+?880|01[3-9]\d))', '', line.strip())
        cleaned = re.sub(r'^[9j2JУyY\)\*\(\#\©\•\-\>\:\;]\s*(?=[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', '', cleaned)
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
            # If user asked for Bangla
            if any(p in q_lower for p in ["bangla", "বাংলা"]):
                if "official AI Assistant" in last_bot_msg or "BAUST AI Assistant" in last_bot_msg:
                    return "আমি বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুরের অফিসিয়াল AI সহকারী।"
                if "Humayun Kabir" in last_bot_msg or "Vice-Chancellor" in last_bot_msg:
                    return "BAUST-এর বর্তমান ভাইস-চ্যান্সেলর (VC) হলেন **ব্রিগেডিয়ার জেনারেল এ বি এম হুমায়ুন কবির, এসজিপি, এনডিসি, পিএসসি, টিই (অব.)**।"
                if "Nakib Hayat" in last_bot_msg or "CSE" in last_bot_msg:
                    return "সিএসই (CSE) বিভাগের বিভাগীয় প্রধান হলেন **ড. মো. নাকিব হায়াত চৌধুরী** (সহযোগী অধ্যাপক)।"
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
                    return "The current Vice-Chancellor (VC) of BAUST is **Brigadier General A B M Humayun Kabir, SGP, ndc, psc, te (Retd.)**."
                if "নাকিব হায়াত" in last_bot_msg or "সিএসই" in last_bot_msg:
                    return "The Head of the Department of CSE is **Dr. Md Nakib Hayat Chowdhury** (Associate Professor). Phone: [01769675560](tel:01769675560), Email: [hdcse@baust.edu.bd](mailto:hdcse@baust.edu.bd)"
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

    # 16. Smart fallback from internal RAG document context (Clean, well-structured, no raw table noise)
    if context and context.strip() and "(No specific internal BAUST document found" not in context:
        cleaned = re.sub(r'---\s*Information from \[.*?\]\s*---', '', context).strip()
        lines = []
        for line in cleaned.splitlines():
            l = line.strip()
            if not l or l.startswith('|---') or l.startswith('===') or l.startswith('---') or l.startswith(':-') or l.startswith(':--'):
                continue
            if any(junk in l.upper() for junk in ["ADMISSION NOTICE", "WINTER SEMESTER", "APPLICATIONS ARE INVITED", "SL. — FACULTY", "SL.", "| :---"]):
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
   - For single-fact questions, give a crisp 1 to 3 bullet point answer.
   - NEVER output raw unformatted tables, lengthy irrelevant admissions text, or bank routing numbers unless specifically asked.
   - For coding/general science questions (e.g., Python binary search), directly provide clean code without university admission templates.

3. **CLICKABLE CONTACTS & LINKS**:
   - Format phone: [01769675588](tel:01769675588)
   - Format email: [admission@baust.edu.bd](mailto:admission@baust.edu.bd)
   - NEVER wrap Markdown links or contacts in backticks (write [01769675588](tel:01769675588), NOT `[01769675588](tel:...)`).

4. **FACULTY MEMBERS & TEACHER DIRECTORY RULES (STRICT SEPARATION)**:
   - When asked for teachers / faculty members of a department (e.g. CSE, EEE, ME, CE, IPE, BBA, English):
     * **সক্রিয় শিক্ষকবৃন্দ (Active Faculty Members)**: List the current active teachers with their designation and contact info. When the user asks for all teachers ("sob teacher", "all faculty", "teachers list"), provide the COMPLETE list of active faculty from the records without omitting anyone.
     * **উচ্চশিক্ষার ছুটিতে থাকা শিক্ষকবৃন্দ (Faculty on Study Leave)**: If any teachers are on study leave, list them under a distinct, separate section clearly titled "উচ্চশিক্ষার ছুটিতে (Study Leave)".
     * **সাবেক শিক্ষকবৃন্দ (Ex-Faculty Members)**: If any former/ex-faculty are mentioned, list them ONLY under a distinct separate section titled "সাবেক শিক্ষকবৃন্দ (Ex-Faculty)" or omit them if the user asked only for current teachers.
     * **NEVER** combine or confuse active teachers with ex-faculty or teachers on study leave.
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
    cache_key = f"{question.strip()}_{len(context)}" if context else question.strip()
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
        env_model = os.getenv("GEMINI_MODEL")
        candidate_stream_models = [
            _last_working_model,
            env_model,
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.7-flash"
        ]
        unique_stream_models = [m for m in dict.fromkeys(candidate_stream_models) if m]

        for model_to_use in unique_stream_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_to_use}:streamGenerateContent?alt=sse&key={key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.25}
            }
            try:
                resp = _http_session.post(url, json=payload, stream=True, timeout=4)
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
                                            full_answer_accumulator.append(token)
                                            yield token
                                            streamed_successfully = True
                            except Exception:
                                pass
                if streamed_successfully:
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

    cache_key = f"{question.strip()}_{len(context)}" if context else question.strip()
    if not history or len(history) <= 1:
        cached = get_cached_response(cache_key)
        if cached:
            return cached

    prompt = _build_full_prompt(context, question, history)
    ans = call_gemini_generate(prompt, temperature=0.25)
    
    if not ans or not ans.strip():
        ans = synthesize_smart_answer(context, question, history=history)

    final_ans = ans.strip()
    set_cached_response(cache_key, final_ans)
    return final_ans