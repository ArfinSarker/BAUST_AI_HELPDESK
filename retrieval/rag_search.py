import os
import re
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(BASE_DIR, "vector_store", "documents.pkl")
META_PATH = os.path.join(BASE_DIR, "vector_store", "metadata.pkl")

_documents = None
_metadata = None

# Pure greeting regex patterns
GREETING_PATTERNS = [
    r"^(hi|hello|hey|assalamu\s*alaikum|salam|hi\s*there|hlo)[\s!.]*$",
    r"^(kemon\s*achen|kemon\s*acho|how\s*are\s*you|ke\s*tumi|tumi\s*ke|who\s*are\s*you|tumi\s*kara|tumi\s*kon)[\s?!.]*$"
]

# Bilingual synonym mapping to connect Bengali/Banglish user queries to English official records
SYNONYM_MAP = {
    "ভিসি": ["vc", "vice chancellor", "vice-chancellor", "executive head", "chancellor"],
    "উপাচার্য": ["vc", "vice chancellor", "vice-chancellor"],
    "প্রেসিডেন্ট": ["president", "bot", "board of trustees", "chief of army staff"],
    "আর্মি": ["army", "bangladesh army", "saidpur cantonment"],
    "শিক্ষক": ["faculty", "teacher", "professor", "lecturer", "assistant professor", "associate professor"],
    "শিক্ষিকা": ["faculty", "teacher", "professor", "lecturer"],
    "ডিপার্টমেন্ট": ["department", "cse", "eee", "me", "ce", "ipe", "bba", "english", "ict"],
    "বিভাগ": ["department", "cse", "eee", "me", "ce", "ipe", "bba", "english", "ict"],
    "সিএসই": ["cse", "computer science", "engineering"],
    "কম্পিউটার": ["cse", "computer science"],
    "মেকানিকাল": ["me", "mechanical"],
    "সিভিল": ["ce", "civil"],
    "ইইই": ["eee", "electrical"],
    "আইসিটি": ["ict", "information and communication"],
    "ভর্তি": ["admission", "apply", "eligibility", "intake", "batch", "winter", "summer", "application"],
    "এডমিশন": ["admission", "apply", "eligibility", "form"],
    "ফি": ["fee", "tuition", "cost", "taka", "bdt", "tk", "charge", "payment"],
    "খরচ": ["fee", "tuition", "cost", "taka", "bdt", "tk", "charge", "payment"],
    "টাকা": ["fee", "tuition", "cost", "taka", "bdt", "tk", "payment"],
    "স্কলারশিপ": ["scholarship", "waiver", "discount", "financial aid", "merit"],
    "ওয়েভার": ["waiver", "scholarship", "discount"],
    "যোগাযোগ": ["contact", "phone", "email", "helpline", "hotline", "location"],
    "ফোন": ["phone", "mobile", "contact", "call", "01769675588", "01769675589"],
    "ঠিকানা": ["location", "saidpur", "cantonment", "address", "nilphamari"],
    "কোথায়": ["location", "saidpur", "cantonment", "address"],
    "ইতিহাস": ["history", "established", "background", "foundation", "parliament"],
    "কবে": ["established", "date", "year", "founded", "history"],
    "নকীব": ["nakib", "nakib hayat", "head of the department"],
    "জাহাঙ্গীর": ["jahangir", "dr md jahangir alam", "dean"],
    "হাসান": ["al-hasan", "hasan", "lecturer"],
    "রেজিস্ট্রার": ["registrar", "office of the registrar", "administrative"],
    "হোস্টেল": ["hostel", "hall", "dormitory", "accommodation", "residential"],
    "হল": ["hall", "hostel", "dormitory", "residential", "provost"],
    "হলে": ["hall", "hostel", "dormitory", "residential", "provost"],
    "হলের": ["hall", "hostel", "dormitory", "residential", "provost"],
    "প্রভোস্ট": ["provost", "assistant provost", "hall"],
    "সহকারী": ["assistant", "assistant provost", "assistant professor"],
    "মেয়েদের": ["girls", "female", "women", "ladies", "taramon bibi"],
    "মহিলা": ["girls", "female", "women", "ladies"],
    "মেয়ে": ["girls", "female", "women", "ladies"],
    "ছেলেদের": ["boys", "male", "men", "abbas uddin", "zikrul haque"],
    "ছেলে": ["boys", "male", "men"],
    "প্রক্টর": ["proctor", "assistant proctor", "proctorial"],
    "ক্যাম্পাস": ["campus", "saidpur cantonment", "facilities", "labs"],
}


def is_greeting(text: str) -> bool:
    clean_text = text.strip().lower()
    return any(re.match(pattern, clean_text) for pattern in GREETING_PATTERNS)


def _load_store():
    global _documents, _metadata
    if os.path.exists(DOCS_PATH):
        try:
            with open(DOCS_PATH, "rb") as f:
                _documents = pickle.load(f)
            if os.path.exists(META_PATH):
                with open(META_PATH, "rb") as f:
                    _metadata = pickle.load(f)
            else:
                _metadata = [{"source": "official_record"} for _ in _documents]
            print(f"[RAG Store] Loaded {len(_documents)} document chunks.")
        except Exception as e:
            print(f"[RAG Store Error]: {e}")
            _documents, _metadata = [], []
    else:
        _documents, _metadata = [], []


def reload_vector_store():
    global _documents
    _documents = None
    _load_store()


def get_store():
    global _documents, _metadata
    if _documents is None:
        _load_store()
    return _documents, _metadata


def search_information(question: str, top_k: int = 8) -> str:
    """
    Ultra-Fast, High-Precision Bilingual Hybrid Document Retrieval.
    Matches queries in Bengali, Banglish, and English to BAUST knowledge records.
    """
    if not question or not question.strip():
        return ""

    if is_greeting(question):
        return ""

    documents, metadata = get_store()
    if not documents:
        return ""

    q_lower = question.lower().strip()
    clean_q = re.sub(r'[^\w\s\u0980-\u09FF]', ' ', q_lower)
    tokens = [w for w in clean_q.split() if len(w) > 1]

    expanded_terms = set(tokens)

    # Add bilingual synonyms
    for t in tokens:
        for k, syn_list in SYNONYM_MAP.items():
            if k in t or t in k:
                for s in syn_list:
                    expanded_terms.add(s.lower())

    for k, syn_list in SYNONYM_MAP.items():
        if k in q_lower:
            for s in syn_list:
                expanded_terms.add(s.lower())

    stop_words = {
        "কে", "কি", "কার", "এর", "আর", "কোথায়", "কীভাবে", "কত", "আছেন", "আছে", "স্যার", "ম্যাডাম",
        "the", "and", "who", "what", "where", "how", "about", "is", "sir", "are", "was", "were",
        "a", "an", "of", "in", "on", "for", "to", "at", "by", "tell", "me", "please", "info", "details"
    }

    search_terms = [t for t in expanded_terms if t not in stop_words]
    if not search_terms:
        search_terms = list(expanded_terms)

    scored = []
    for idx, doc in enumerate(documents):
        doc_lower = doc.lower()
        score = 0.0

        # Exact query match boost
        if q_lower in doc_lower:
            score += 25.0

        matched_count = 0
        for term in search_terms:
            if term in doc_lower:
                matched_count += 1
                score += (len(term.split()) * 4.0) + (len(term) * 0.1)

        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    # If no specific keyword match but mentions university/baust, include overview chunks
    if not scored:
        if any(w in q_lower for w in ["baust", "বিশ্ববিদ্যালয়", "university", "ক্যান্টনমেন্ট", "সৈয়দপুর"]):
            scored = [(1.0, i) for i in range(min(4, len(documents)))]

    if not scored:
        return ""

    top_matches = scored[:top_k]
    formatted_chunks = []
    for score_val, idx in top_matches:
        source = metadata[idx].get("source", "BAUST Records") if idx < len(metadata) else "BAUST Records"
        formatted_chunks.append(f"--- Information from [{source}] ---\n{documents[idx]}")

    return "\n\n".join(formatted_chunks)