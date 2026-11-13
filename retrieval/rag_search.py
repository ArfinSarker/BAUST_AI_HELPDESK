import os
import re
import json
import math
import pickle
import threading
from collections import Counter
from typing import Optional, List, Dict, Tuple
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
DOCS_PATH = os.path.join(VECTOR_DIR, "documents.pkl")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")
EMBED_PATH = os.path.join(VECTOR_DIR, "embeddings.npy")
FAISS_PATH = os.path.join(VECTOR_DIR, "index.faiss")
SYNONYMS_FILE = os.path.join(BASE_DIR, "data", "synonyms.json")

# =====================================================
# THREAD-SAFE GLOBAL STATE & LOCKING
# =====================================================
_store_lock = threading.RLock()
_documents = None
_metadata = None
_embeddings = None
_faiss_index = None
_bm25_model = None
_last_store_mtime = 0.0

_embed_model = None
_embed_model_attempted = False

# =====================================================
# BM25 OKAPI ALGORITHM IMPLEMENTATION
# =====================================================
try:
    import importlib
    _RankBM25Okapi = getattr(importlib.import_module("rank_bm25"), "BM25Okapi", None)
except Exception:
    _RankBM25Okapi = None


class BuiltinBM25Okapi:
    """
    Pure Python Okapi BM25 implementation.
    Calculates TF-IDF and document length normalization (k1=1.5, b=0.75)
    with zero external dependencies and thread-safe evaluation.
    """
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = [len(doc) for doc in corpus]

        nd = {}
        for document in corpus:
            frequencies = Counter(document)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1

        idf_sum = 0.0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        average_idf = idf_sum / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not self.corpus_size or not query:
            return scores
        for word in query:
            if word not in self.idf:
                continue
            q_idf = self.idf[word]
            for i, doc_freq in enumerate(self.doc_freqs):
                if word in doc_freq:
                    freq = doc_freq[word]
                    doc_len = self.doc_len[i]
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[i] += q_idf * (numerator / denominator)
        return scores


BM25Okapi = _RankBM25Okapi or BuiltinBM25Okapi


def tokenize_text(text: str) -> list[str]:
    """Tokenize English and Bengali text into clean terms."""
    if not text:
        return []
    clean_t = re.sub(r'[^\w\s\u0980-\u09FF]', ' ', text.lower())
    return [w for w in clean_t.split() if len(w) > 1]


# =====================================================
# DYNAMIC SYNONYM MAP (EXTERNAL JSON + LIVE RELOAD)
# =====================================================
_synonym_map = {}
_synonym_mtime = 0.0

DEFAULT_FALLBACK_SYNONYMS = {
    "ভিসি": ["vc", "vice chancellor", "vice-chancellor", "executive head", "humayun kabir"],
    "উপাচার্য": ["vc", "vice chancellor", "vice-chancellor", "humayun kabir"],
    "ডিপার্টমেন্ট": ["department", "cse", "eee", "me", "ce", "ipe", "bba", "english", "ict"],
    "সিএসই": ["cse", "computer science", "nakib hayat"],
    "ভর্তি": ["admission", "apply", "eligibility", "winter", "summer"],
    "ফি": ["fee", "tuition", "cost", "taka", "bdt", "waiver"]
}


def get_synonym_map() -> dict:
    """Dynamically reads synonyms from data/synonyms.json with automatic hot reload."""
    global _synonym_map, _synonym_mtime
    try:
        if os.path.exists(SYNONYMS_FILE):
            mtime = os.path.getmtime(SYNONYMS_FILE)
            if mtime != _synonym_mtime:
                with open(SYNONYMS_FILE, "r", encoding="utf-8") as f:
                    _synonym_map = json.load(f)
                _synonym_mtime = mtime
    except Exception as e:
        print(f"[Synonyms Read Warning]: {e}")
    return _synonym_map or DEFAULT_FALLBACK_SYNONYMS


# =====================================================
# GREETINGS & INTENT PATTERNS
# =====================================================
GREETING_PATTERNS = [
    r"^(hi|hello|hey|assalamu\s*alaikum|salam|hi\s*there|hlo|ola)[\s!.]*$",
    r"^(kemon\s*achen|kemon\s*acho|how\s*are\s*you|how\s*r\s*u|কেমন\s*আছো|কেমন\s*আছেন)[\s?!.]*$",
    r"^(what\s*is\s*your\s*name|what\'?s\s*your\s*name|your\s*name|who\s*are\s*you|who\s*r\s*u|who\s*made\s*you)[\s?!.]*$",
    r"^(tomar\s*nam\s*ki|apnar\s*nam\s*ki|tumi\s*ke|apni\s*ke|ke\s*tumi|ke\s*apni|তোমার\s*নাম\s*কি|তুমি\s*কে|আপনার\s*নাম\s*কি|কে\s*তুমি)[\s?!.]*$",
    r"^(what\s*can\s*you\s*do|tumi\s*ki\s*korte\s*paro|ki\s*sahajjo\s*korte\s*paro|তুমি\s*কি\s*করতে\s*পারো)[\s?!.]*$",
    r"^(thank\s*you|thanks|thx|dhonnobad|ধন্যবাদ|shukriya)[\s!.]*$",
    r"^(bye|goodbye|allah\s*hafez|tata|বিদায়|বিদায়)[\s!.]*$",
    r"^(banglay\s*bolo|english\s*e\s*bolo|translate\s*to\s*english|translate\s*to\s*bangla|বাংলায়\s*বলো|বাংলাতে\s*বলো)[\s!.]*$"
]


def normalize_text_for_intent(text: str) -> str:
    """Normalize text by collapsing repeating characters and punctuation."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r'[!?,.\'\"]+', ' ', t)
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

    # 7. Translation commands
    if any(k in norm_t for k in ["banglay bolo", "english e bolo", "translate to english", "translate to bangla"]) or any(k in raw_t for k in ["বাংলায় বলো", "বাংলাতে বলো"]):
        return True

    return False


def _get_embedding_model():
    """Lazy loader for SentenceTransformer with single-attempt caching to prevent repeated slow import hangs."""
    global _embed_model, _embed_model_attempted
    if _embed_model is None and not _embed_model_attempted:
        _embed_model_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            # Fast offline-first loading: prevents 100s+ timeout hangs when HuggingFace Hub is unreachable
            try:
                _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
            except Exception:
                _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            _embed_model = None
    return _embed_model


def _get_store_mtime() -> float:
    """Returns the newest modification timestamp of vector store artifacts."""
    mtimes = []
    for p in [FAISS_PATH, DOCS_PATH, META_PATH]:
        if os.path.exists(p):
            try:
                mtimes.append(os.path.getmtime(p))
            except OSError:
                pass
    return max(mtimes) if mtimes else 0.0


def reload_vector_store():
    """
    Thread-safe hot reloader for vector store and BM25 index.
    Loads files into local scope first, then atomically swaps global pointers
    to prevent NoneType race conditions during live user queries.
    """
    global _documents, _metadata, _embeddings, _faiss_index, _bm25_model, _last_store_mtime

    with _store_lock:
        new_docs = []
        new_metadata = []
        new_embeddings = None
        new_faiss = None
        new_bm25 = None
        current_mtime = _get_store_mtime()

        if os.path.exists(DOCS_PATH):
            try:
                with open(DOCS_PATH, "rb") as f:
                    new_docs = pickle.load(f)
                if os.path.exists(META_PATH):
                    with open(META_PATH, "rb") as f:
                        new_metadata = pickle.load(f)
                else:
                    new_metadata = [{"source": "official_record"} for _ in new_docs]

                if os.path.exists(EMBED_PATH):
                    new_embeddings = np.load(EMBED_PATH)

                if os.path.exists(FAISS_PATH):
                    try:
                        import faiss
                        new_faiss = faiss.read_index(FAISS_PATH)
                    except Exception:
                        new_faiss = None

                # Initialize Okapi BM25 on the tokenized corpus
                if new_docs:
                    tokenized_corpus = [tokenize_text(doc) for doc in new_docs]
                    new_bm25 = BM25Okapi(tokenized_corpus)

                print(f"[RAG Hybrid Store] Hot-Loaded {len(new_docs)} chunks with FAISS + BM25Okapi support.")
            except Exception as e:
                print(f"[RAG Store Load Error]: {e}")

        # Atomic pointer swap inside critical section
        _documents = new_docs
        _metadata = new_metadata
        _embeddings = new_embeddings
        _faiss_index = new_faiss
        _bm25_model = new_bm25
        _last_store_mtime = current_mtime


def get_store():
    """
    Thread-safe store getter with automatic mtime hot-reloading.
    Detects if background create_vector.py finished updating index files.
    """
    global _documents, _metadata, _embeddings, _faiss_index, _last_store_mtime
    current_mtime = _get_store_mtime()

    # Trigger hot-reload if store is uninitialized or on-disk index was updated
    if _documents is None or (current_mtime > 0 and current_mtime != _last_store_mtime):
        reload_vector_store()

    return _documents, _metadata, _embeddings, _faiss_index


def dense_semantic_search(query: str, top_k: int = 12) -> list[tuple[float, int]]:
    """
    Computes dense semantic similarity using multilingual neural embeddings.
    Returns: list of (similarity_score, chunk_index)
    """
    documents, metadata, embeddings, faiss_index = get_store()
    if not documents or (embeddings is None and faiss_index is None):
        return []

    model = _get_embedding_model()
    if model is None:
        return []

    try:
        query_vector = model.encode([query], show_progress_bar=False)
        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_norm[query_norm == 0] = 1e-10
        normalized_query = (query_vector / query_norm).astype("float32")

        if faiss_index is not None:
            scores, indices = faiss_index.search(normalized_query, min(top_k, len(documents)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(documents):
                    results.append((float(score), int(idx)))
            return results
        elif embeddings is not None:
            sims = np.dot(embeddings, normalized_query.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [(float(sims[i]), int(i)) for i in top_indices]
    except Exception as e:
        print(f"[Dense Semantic Search Error]: {e}")

    return []


def sparse_lexical_search(question: str, top_k: int = 12) -> list[tuple[float, int]]:
    """
    Computes genuine BM25Okapi lexical keyword and bilingual synonym matches.
    Returns: list of (bm25_score, chunk_index)
    """
    documents, metadata, _, _ = get_store()
    if not documents or _bm25_model is None:
        return []

    q_lower = question.lower().strip()
    query_tokens = tokenize_text(q_lower)
    if not query_tokens:
        return []

    syn_map = get_synonym_map()
    expanded_terms = set(query_tokens)

    # Dynamic bilingual synonym expansion
    for t in query_tokens:
        for k, syn_list in syn_map.items():
            if k in t or t in k:
                for s in syn_list:
                    for sw in tokenize_text(s):
                        expanded_terms.add(sw)

    for k, syn_list in syn_map.items():
        if k in q_lower:
            for s in syn_list:
                for sw in tokenize_text(s):
                    expanded_terms.add(sw)

    stop_words = {
        "কে", "কি", "কার", "এর", "আর", "কোথায়", "কীভাবে", "কত", "আছেন", "আছে", "স্যার", "ম্যাডাম",
        "the", "and", "who", "what", "where", "how", "about", "is", "sir", "are", "was", "were",
        "a", "an", "of", "in", "on", "for", "to", "at", "by", "tell", "me", "please", "info", "details"
    }

    search_terms = [t for t in expanded_terms if t not in stop_words]
    if not search_terms:
        search_terms = list(expanded_terms)

    # Calculate Okapi BM25 score array
    bm25_scores = _bm25_model.get_scores(search_terms)

    scored = []
    for idx, raw_score in enumerate(bm25_scores):
        if raw_score <= 0.05:
            continue
        score = float(raw_score)
        doc_lower = documents[idx].lower()

        # Exact query match boost
        if q_lower in doc_lower:
            score += 15.0

        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def get_department_faculty_file_content(question: str) -> str:
    """
    If the question is asking for faculty members / teachers of a specific department,
    return the complete, unabridged official department faculty document.
    """
    q_lower = question.lower()
    norm_q = normalize_text_for_intent(question)

    # Disqualify from pure faculty file lookup if user is asking for exams, routines, seatplans, fees, etc.
    if any(k in q_lower for k in [
        "exam", "examination", "schedule", "routine", "seatplan", "seat plan", "result",
        "syllabus", "grade", "cgpa", "admission", "fees", "fee", "tuition", "hall", "hostel"
    ]):
        return None

    words_set = set(norm_q.split())
    faculty_keywords = {
        "teacher", "teachers", "faculty", "faculties", "sir", "mam", "shikkok", "shikkhok",
        "professor", "lecturer", "dean", "head", "hod"
    }
    is_faculty_query = bool(words_set & faculty_keywords) or any(k in q_lower for k in [
        "শিক্ষক", "শিক্ষিকা", "অধ্যাপক", "প্রভাষক", "হেড", "প্রধান"
    ]) or (
        ("list" in words_set or "talika" in words_set or "তালিকা" in q_lower or "লিস্ট" in q_lower) and
        any(k in norm_q for k in ["member", "staff", "person", "faculty", "teacher"])
    )

    if not is_faculty_query:
        return None

    dept_file_map = [
        (["cse", "computer science", "কম্পিউটার", "সিএসই"], "cse_department_and_faculty.txt"),
        (["eee", "electrical", "ইইই", "ইলেকট্রিক্যাল"], "eee_department_and_faculty.txt"),
        (["ce", "civil", "সিভিল", "পুরকৌশল"], "ce_department_and_faculty.txt"),
        (["me", "mechanical", "মেকানিকাল", "মেকানিক্যাল", "যন্ত্রকৌশল"], "me_department_and_faculty.txt"),
        (["ipe", "industrial", "আইপিই"], "ipe_department_and_faculty.txt"),
        (["bba", "fbs", "business", "ais", "mba", "বিবিএ", "ব্যবসায়"], "fbs_department_and_faculty.txt"),
        (["ict", "ece", "আইসিটি"], "ict_and_ece_department_and_faculty.txt"),
    ]

    kb_dir = os.path.join(BASE_DIR, "knowledge_base")
    for keywords, fname in dept_file_map:
        matched = False
        for kw in keywords:
            if len(kw) <= 3:
                if re.search(r'\b' + re.escape(kw) + r'\b', q_lower) or re.search(r'\b' + re.escape(kw) + r'\b', norm_q):
                    matched = True
                    break
            else:
                if kw in q_lower or kw in norm_q:
                    matched = True
                    break
        if matched:
            fpath = os.path.join(kb_dir, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        return f"--- Document: Official Department Records (Source: {fname} | Status: CURRENT & AUTHORITATIVE) ---\n{content}"
                except Exception as e:
                    print(f"[Faculty File Read Error]: {e}")
    return None


_STUDENT_CACHE = None
_STUDENT_CACHE_MTIME = 0.0

def get_student_database() -> dict:
    """
    Returns an in-memory O(1) dictionary mapping student roll -> student info.
    Auto-invalidates if seatplan files change.
    """
    global _STUDENT_CACHE, _STUDENT_CACHE_MTIME
    kb_dir = os.path.join(BASE_DIR, "knowledge_base")
    student_files = glob.glob(os.path.join(kb_dir, "*seatplan*.txt")) + glob.glob(os.path.join(kb_dir, "*student*.txt"))
    if not student_files:
        return {}

    max_mtime = max((os.path.getmtime(p) for p in student_files if os.path.exists(p)), default=0.0)
    if _STUDENT_CACHE is not None and max_mtime == _STUDENT_CACHE_MTIME:
        return _STUDENT_CACHE

    cache = {}
    slot_map = {
        'A': ('Slot-A', '09:00 AM - 10:00 AM'),
        'B': ('Slot-B', '11:00 AM - 12:00 PM'),
        'C': ('Slot-C', '01:00 PM - 02:00 PM')
    }

    for fpath in student_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            continue

        current_level = ""
        current_slot = ""
        current_prog = "Computer Science and Engineering"

        for idx, l in enumerate(lines):
            pm = re.match(r'^(?:Programme|Program):\s*(.*)', l, re.I)
            if pm: current_prog = pm.group(1).strip()
            lm = re.match(r'^(Level\s*[-–]?\s*\d+\s*Term\s*[-–]?\s*[IVX0-9]+.*)', l, re.I)
            if lm: current_level = lm.group(1).strip()
            sm = re.match(r'^(Slot\s*[-–:]?\s*[A-Z0-9].*)', l, re.I)
            if sm: current_slot = sm.group(1).strip()

            if re.match(r'^\d{8,16}$', l) and idx + 1 < len(lines):
                roll = l
                if roll in cache:
                    continue
                name = lines[idx + 1]
                room = lines[idx + 2] if idx + 2 < len(lines) else ""
                seat = lines[idx + 3] if idx + 3 < len(lines) else ""

                seat_pfx = re.match(r'^([ABC])-\d+-\d+', seat.strip(), re.I)
                if seat_pfx:
                    sk = seat_pfx.group(1).upper()
                    s_name, s_time = slot_map.get(sk, (f"Slot-{sk}", ""))
                else:
                    s_name = current_slot or "Slot-A"
                    s_time = ""

                lt_m = re.search(r'Level\s*[-–]?\s*(\d+)\s*Term\s*[-–]?\s*([IVX0-9]+)', current_level, re.I)
                norm_lt = f"{lt_m.group(1)}-{lt_m.group(2)}" if lt_m else current_level

                last_char = roll[-1] if roll else "0"
                parity = "ODD" if last_char.isdigit() and int(last_char) % 2 != 0 else "EVEN"

                cache[roll] = {
                    "roll": roll,
                    "name": name,
                    "programme": current_prog,
                    "level": current_level,
                    "level_term": norm_lt,
                    "parity": parity,
                    "slot": s_name,
                    "time": s_time,
                    "room": room,
                    "seat": seat,
                    "source": os.path.basename(fpath)
                }

    _STUDENT_CACHE = cache
    _STUDENT_CACHE_MTIME = max_mtime
    return _STUDENT_CACHE


def get_bangladesh_temporal_context() -> dict:
    import datetime
    tz_bd = datetime.timezone(datetime.timedelta(hours=6))
    now_bd = datetime.datetime.now(tz_bd)
    today_name = now_bd.strftime("%A")
    date_str = now_bd.strftime("%d %B %Y")
    time_str = now_bd.strftime("%I:%M %p")
    tomorrow = now_bd + datetime.timedelta(days=1)
    tomorrow_name = tomorrow.strftime("%A")

    is_today_weekend = today_name in ["Friday", "Saturday"]
    is_tomorrow_weekend = tomorrow_name in ["Friday", "Saturday"]

    if today_name in ["Thursday", "Friday", "Saturday"]:
        next_class_day = "Sunday"
    else:
        next_class_day = tomorrow_name

    bn_day_map = {
        "Friday": "শুক্রবার",
        "Saturday": "শনিবার",
        "Sunday": "রবিবার",
        "Monday": "সোমবার",
        "Tuesday": "মঙ্গলবার",
        "Wednesday": "বুধবার",
        "Thursday": "বৃহস্পতিবার"
    }

    return {
        "today_en": today_name,
        "today_bn": bn_day_map.get(today_name, today_name),
        "tomorrow_en": tomorrow_name,
        "tomorrow_bn": bn_day_map.get(tomorrow_name, tomorrow_name),
        "next_class_en": next_class_day,
        "next_class_bn": bn_day_map.get(next_class_day, next_class_day),
        "is_today_weekend": is_today_weekend,
        "is_tomorrow_weekend": is_tomorrow_weekend,
        "date_str": date_str,
        "time_str": time_str
    }


def get_class_routine_content(dept: str = "CSE", level_term: str = "3-II") -> str:
    kb_dir = os.path.join(BASE_DIR, "knowledge_base")
    routine_files = glob.glob(os.path.join(kb_dir, "*routine*.txt"))
    for fpath in routine_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
                norm_lt = (level_term or "").replace(" ", "").upper()
                if "3-II" in norm_lt or "3-2" in norm_lt or not level_term:
                    return text
        except Exception:
            continue
    return ""


ROUTINE_KEYWORDS = [
    "routine", "class", "classes", "lab", "kalke", "ajke", "schedule", "period",
    "somoy", "shomoy", "kobe", "kokhon", "timetable", "time table", "ক্লাস",
    "রুটিন", "ল্যাব", "সময়সূচি", "সময়সূচি", "কি কি ক্লাস", "কখন ক্লাস", "পড়াশোনা",
    "আজকে", "কালকে", "আগামীকাল", "পরশু", "sunday", "monday", "tuesday", "wednesday",
    "thursday", "রবিবার", "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার"
]


def is_routine_query(query: str) -> bool:
    if not query:
        return False
    q_low = query.lower()
    return any(re.search(r'\b' + re.escape(w) + r'\b', q_low) or (len(w) >= 3 and w in q_low) for w in ROUTINE_KEYWORDS)


def resolve_student_or_routine_query(question: str, history: list = None) -> Optional[str]:
    """
    Authoritative Resolver for:
    1. Student Class Routine queries ("kalke amar ki ki class ache", "amar class routine ki").
    2. Student Exam Seatplan queries ("amar seat kothay", "exam room kothay", "0802410205101019").
    3. General Batch Class Routine queries ("CSE 3-II class routine").
    """
    if not question:
        return None

    clean_q = question.strip()
    q_low = clean_q.lower()
    is_routine = is_routine_query(clean_q)

    # 1. Extract student roll from question or history
    roll_match = re.search(r'\b(08\d{6,14})\b', clean_q)
    if not roll_match and history:
        for msg in reversed(history[-4:]):
            if isinstance(msg, dict):
                m = re.search(r'\b(08\d{6,14})\b', str(msg.get("content", "")))
                if m:
                    roll_match = m
                    break

    st_db = get_student_database()

    # Case A: Student Roll Found
    if roll_match:
        roll = roll_match.group(1)
        s = st_db.get(roll)
        if s:
            if is_routine:
                temporal = get_bangladesh_temporal_context()
                routine_text = get_class_routine_content(s["programme"], s["level_term"])
                if routine_text:
                    weekend_today_note = "[সাপ্তাহিক ছুটি / WEEKEND - আজ কোনো ক্লাস নেই]" if temporal['is_today_weekend'] else "[ক্লাস ডে / REGULAR CLASS DAY]"
                    weekend_tmrw_note = "[সাপ্তাহিক ছুটি / WEEKEND - আগামীকাল কোনো ক্লাস নেই]" if temporal['is_tomorrow_weekend'] else "[ক্লাস ডে / REGULAR CLASS DAY]"

                    doc = (
                        f"--- Document: Student Profile & Official Class & Lab Routine (Authoritative Source: class_and_lab_routine.txt) ---\n"
                        f"STUDENT IDENTITY:\n"
                        f"* Student Name: {s['name']}\n"
                        f"* Student ID / Roll: {s['roll']} (Roll Parity / Group: {s['parity']})\n"
                        f"* Programme / Department: {s['programme']}\n"
                        f"* Academic Level & Term: {s['level']} (Level-Term Code: {s['level_term']}, Section: A)\n\n"
                        f"TEMPORAL CONTEXT (BANGLADESH TIME - UTC+6):\n"
                        f"* Current Date: {temporal['date_str']} | Current Time: {temporal['time_str']}\n"
                        f"* Today: {temporal['today_en']} ({temporal['today_bn']}) {weekend_today_note}\n"
                        f"* Tomorrow: {temporal['tomorrow_en']} ({temporal['tomorrow_bn']}) {weekend_tmrw_note}\n"
                        f"* Next Regular Academic Class Day: {temporal['next_class_en']} ({temporal['next_class_bn']})\n\n"
                        f"OFFICIAL BATCH CLASS & LAB ROUTINE (Department: {s['programme']}, {s['level']}, Section: A):\n"
                        f"{routine_text}\n\n"
                        f"STUDENT EXAMINATION SEATPLAN ALLOCATION (REFERENCE ONLY - DO NOT OVERRIDE ROUTINE):\n"
                        f"* Examination Slot: {s['slot']} ({s['time']})\n"
                        f"* Allocated Exam Room: {s['room']}\n"
                        f"* Allocated Seat Number: {s['seat']}\n"
                    )
                    return doc
                else:
                    return (
                        f"--- Document: Student Profile Found (Class Routine Status) ---\n"
                        f"* Student Name: {s['name']}\n"
                        f"* Student ID / Roll: {s['roll']}\n"
                        f"* Programme: {s['programme']}\n"
                        f"* Academic Level & Term: {s['level']}\n"
                        f"* Note: The official class routine for {s['level']} is currently not published/uploaded in the knowledge base. Only CSE Level- 3 Term- II class routine is currently available in the system.\n"
                    )
            else:
                # User asked about seat/room/exam or entered only ID
                single_doc = (
                    f"--- Document: Single Student Match (Source: {s['source']} | Status: CURRENT & AUTHORITATIVE) ---\n"
                    f"* Student Name: {s['name']}\n"
                    f"* Student ID / Roll: {s['roll']}\n"
                    f"* Programme: {s['programme']}\n"
                    f"* Level & Term: {s['level']}\n"
                    f"* Examination Slot: {s['slot']}\n"
                    f"* Examination Time: {s['time']}\n"
                    f"* Allocated Exam Room: {s['room']}\n"
                    f"* Allocated Seat Number: {s['seat']}\n"
                )
                return single_doc

    # Case B: No roll found, but user asks for Class Routine (e.g. "CSE 3-II class routine" or "routine")
    if is_routine:
        lt_match = re.search(r'\b(3[-–]?[iI]{2}|3[-–]?2|level\s*[-–]?\s*3\s*term\s*[-–]?\s*(?:2|ii))\b', q_low)
        dept_match = any(re.search(pat, q_low) for pat in DEPT_PATTERNS.get("cse", []))
        if lt_match or dept_match or "routine" in q_low or "class" in q_low:
            temporal = get_bangladesh_temporal_context()
            routine_text = get_class_routine_content("CSE", "3-II")
            if routine_text:
                weekend_today_note = "[সাপ্তাহিক ছুটি / WEEKEND - আজ কোনো ক্লাস নেই]" if temporal['is_today_weekend'] else "[ক্লাস ডে / REGULAR CLASS DAY]"
                weekend_tmrw_note = "[সাপ্তাহিক ছুটি / WEEKEND - আগামীকাল কোনো ক্লাস নেই]" if temporal['is_tomorrow_weekend'] else "[ক্লাস ডে / REGULAR CLASS DAY]"

                return (
                    f"--- Document: Official Class & Lab Routine (Authoritative Source: class_and_lab_routine.txt) ---\n"
                    f"TEMPORAL CONTEXT (BANGLADESH TIME - UTC+6):\n"
                    f"* Current Date: {temporal['date_str']} | Current Time: {temporal['time_str']}\n"
                    f"* Today: {temporal['today_en']} ({temporal['today_bn']}) {weekend_today_note}\n"
                    f"* Tomorrow: {temporal['tomorrow_en']} ({temporal['tomorrow_bn']}) {weekend_tmrw_note}\n"
                    f"* Next Regular Academic Class Day: {temporal['next_class_en']} ({temporal['next_class_bn']})\n\n"
                    f"OFFICIAL BATCH CLASS & LAB ROUTINE:\n"
                    f"{routine_text}\n"
                )

    return None


def find_student_matches_by_name(query: str, max_results: int = 12) -> tuple[list[dict], str]:
    """
    Scans student records across knowledge base for student names.
    Returns (matches, disambiguation_context_or_empty).
    - If multiple students match: returns a disambiguation list and prompts for ID/Roll.
    - If exactly one student matches: returns that student's complete record (routine or exam seat).
    - If zero match: returns empty so standard RAG hybrid search proceeds.
    """
    clean_q = query.strip()
    if not clean_q or len(clean_q) < 3:
        return [], ""

    q_lower = clean_q.lower()
    non_name_indicators = [
        "fee", "fees", "cost", "taka", "admission", "borti", "calendar",
        "vc", "vice chancellor", "dean", "head", "provost", "hall",
        "hostel", "bus", "transport", "picnic", "phone", "email", "contact",
        "what", "how", "who", "where", "koto", "kivabe", "syllabus",
        "bivag", "credit", "waiver", "scholarship", "grade", "cgpa"
    ]
    if any(re.search(r'\b' + re.escape(w) + r'\b', q_lower) for w in non_name_indicators):
        return [], ""

    # Do not intercept if query already has a roll/ID number
    if re.search(r'\b\d{6,16}\b', clean_q):
        return [], ""

    raw_tokens = [w for w in re.findall(r'[a-zA-Z]{3,}', q_lower) if w not in ["class", "routine", "kalke", "ajke", "amar", "ache", "ki"]]
    if not raw_tokens:
        raw_tokens = [w for w in re.sub(r'[^\u0980-\u09FF\s]', ' ', clean_q).split() if len(w) >= 3 and w not in ["ক্লাস", "রুটিন", "ল্যাব", "আমার", "আছে", "কি"]]
        if not raw_tokens:
            return [], ""

    st_db = get_student_database()
    results = []
    seen_names = set()

    for s in st_db.values():
        name_low = s["name"].lower()
        name_words = set(re.findall(r'[a-zA-Z\u0980-\u09FF]+', name_low))
        if all(tok in name_words for tok in raw_tokens):
            if s["roll"] not in seen_names:
                seen_names.add(s["roll"])
                results.append(s)
                if len(results) >= max_results:
                    break

    if not results:
        return [], ""

    if len(results) > 1:
        doc_lines = [
            f"--- Document: Multiple Student Matches Found for Name '{clean_q}' (Disambiguation Required) ---",
            f"CRITICAL INSTRUCTION: There are {len(results)} distinct students matching the query '{clean_q}'.",
            "DO NOT pick or answer for only one student arbitrarily!",
            "Politely inform the user in Bengali that multiple students have this name, list the matching students with their full names and IDs, and ask the user to provide their Student ID / Roll number or full name to get their specific exam seat and routine.",
            "\nMATCHING STUDENTS LIST:"
        ]
        for i, s in enumerate(results, 1):
            doc_lines.append(f"{i}. **{s['name']}** — Roll / ID: `{s['roll']}` | Department: {s['programme']} | {s['level']}")
        return results, "\n".join(doc_lines)

    s = results[0]
    if is_routine_query(clean_q):
        temporal = get_bangladesh_temporal_context()
        routine_text = get_class_routine_content(s["programme"], s["level_term"])
        if routine_text:
            return results, (
                f"--- Document: Student Profile & Official Class & Lab Routine (Authoritative Source: class_and_lab_routine.txt) ---\n"
                f"STUDENT IDENTITY:\n"
                f"* Student Name: {s['name']}\n"
                f"* Student ID / Roll: {s['roll']} (Roll Parity / Group: {s['parity']})\n"
                f"* Programme / Department: {s['programme']}\n"
                f"* Academic Level & Term: {s['level']} (Level-Term Code: {s['level_term']}, Section: A)\n\n"
                f"TEMPORAL CONTEXT (BANGLADESH TIME - UTC+6):\n"
                f"* Current Date: {temporal['date_str']} | Current Time: {temporal['time_str']}\n"
                f"* Today: {temporal['today_en']} ({temporal['today_bn']})\n"
                f"* Tomorrow: {temporal['tomorrow_en']} ({temporal['tomorrow_bn']})\n"
                f"* Next Regular Academic Class Day: {temporal['next_class_en']} ({temporal['next_class_bn']})\n\n"
                f"OFFICIAL BATCH CLASS & LAB ROUTINE:\n"
                f"{routine_text}\n"
            )

    single_doc = (
        f"--- Document: Single Student Match (Source: {s['source']} | Status: CURRENT & AUTHORITATIVE) ---\n"
        f"* Student Name: {s['name']}\n"
        f"* Student ID / Roll: {s['roll']}\n"
        f"* Programme: {s['programme']}\n"
        f"* Level & Term: {s['level']}\n"
        f"* Examination Slot: {s['slot']}\n"
        f"* Examination Time: {s['time']}\n"
        f"* Allocated Exam Room: {s['room']}\n"
        f"* Allocated Seat Number: {s['seat']}\n"
    )
    return results, single_doc


DEPT_PATTERNS = {
    "cse": [r"\bcse\b", r"\bcomputer\s+science\b", r"সিএসই", r"কম্পিউটার\s+সায়েন্স", r"কম্পিউটার\s+বিজ্ঞান"],
    "ce": [r"\bce\b", r"\bcivil\b", r"\bcivil\s+engineering\b", r"সিই", r"সিভিল", r"পুরকৌশল"],
    "eee": [r"\beee\b", r"\belectrical\b", r"\beee\s+engineering\b", r"ইইই", r"ইলেকট্রিক্যাল", r"তড়িৎ"],
    "me": [r"\bme\b", r"\bmechanical\b", r"এমই", r"মেকানিক্যাল", r"যন্ত্রকৌশল"],
    "ipe": [r"\bipe\b", r"\bindustrial\b", r"আইপিই"],
    "bba": [r"\bbba\b", r"\bfbs\b", r"\bbusiness\b", r"বিবিএ", r"ব্যবসায়"],
    "english": [r"\benglish\b", r"ইংরেজি", r"ইংলিশ"],
    "ict": [r"\bict\b", r"আইসিটি"]
}


def search_information(question: str, top_k: int = 4, history: list = None) -> str:
    """
    State-of-the-Art Hybrid Retrieval Engine:
    Combines Multilingual Dense Semantic Vector Search + BM25Okapi Lexical Ranking
    via Reciprocal Rank Fusion (RRF) and Full Document Inspection for Faculty/Department Queries.
    """
    if not question or not question.strip():
        return ""

    if is_greeting(question):
        return ""

    # Authoritative Resolver for Student Profile, Class Routine and Exam Seatplan
    student_or_routine_doc = resolve_student_or_routine_query(question, history=history)
    if student_or_routine_doc:
        return student_or_routine_doc

    # Check for student name query (disambiguation if multiple, direct if single)
    name_matches, disambiguation_doc = find_student_matches_by_name(question)
    if disambiguation_doc:
        return disambiguation_doc

    # Check for direct department faculty document lookup
    dept_faculty_doc = get_department_faculty_file_content(question)
    if dept_faculty_doc:
        return dept_faculty_doc

    documents, metadata, _, _ = get_store()
    if not documents:
        return ""

    # 1. Dense Semantic Search (Vector Embedding)
    dense_results = dense_semantic_search(question, top_k=8)

    # 2. Sparse Lexical Search (BM25Okapi & Synonyms)
    sparse_results = sparse_lexical_search(question, top_k=8)

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    dense_weight = 1.0
    sparse_weight = 1.2
    k_constant = 60.0

    for rank, (score, idx) in enumerate(dense_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (dense_weight / (k_constant + rank + 1))

    for rank, (score, idx) in enumerate(sparse_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (sparse_weight / (k_constant + rank + 1))

    # Universal Exact Token Booster (Rolls, Course Codes, Phones, Room IDs)
    target_match = re.search(r'\b(08\d{6,14}|[A-Za-z]{2,5}[-\s]?\d{3,4}|01[3-9]\d{8})\b', question)
    if not target_match and history:
        for msg in reversed(history[-4:]):
            if isinstance(msg, dict):
                m = re.search(r'\b(08\d{6,14}|[A-Za-z]{2,5}[-\s]?\d{3,4}|01[3-9]\d{8})\b', str(msg.get("content", "")))
                if m:
                    target_match = m
                    break

    if target_match:
        target_tok = target_match.group(1).lower().replace(" ", "").replace("-", "")
        for idx in range(len(documents)):
            doc_norm = documents[idx].lower().replace(" ", "").replace("-", "")
            if target_tok in doc_norm:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 10.0

    # 4. Contextual Source & Authoritative Metadata Re-ranking
    q_lower = question.lower()
    is_archive_query = any(k in q_lower for k in [
        "archived", "superseded", "old", "previous", "former", "past",
        "আগের", "পুরাতন", "গত বছর", "পূর্বের", "বাতিল", "অতীত"
    ])

    # Detect if user specifically queried for one or more departments
    queried_depts = set()
    for d_code, p_list in DEPT_PATTERNS.items():
        if any(re.search(p, q_lower) for p in p_list):
            queried_depts.add(d_code)

    for idx in list(rrf_scores.keys()):
        if idx < len(metadata):
            meta = metadata[idx]
            src = meta.get("source", "").lower()
            status = meta.get("status", "active").lower()
            is_authoritative = meta.get("is_authoritative", 1)
            academic_year = str(meta.get("academic_year", "")).lower()
            semester = str(meta.get("semester", "")).lower()
            issuer = str(meta.get("issuing_authority", "")).lower()
            title = str(meta.get("title", "")).lower()
            chunk_text = documents[idx].lower() if idx < len(documents) else ""

            # A. Authoritative Version Filter
            if not is_archive_query:
                if status in ["superseded", "archived"] or not is_authoritative:
                    rrf_scores[idx] *= 0.001
                elif status == "active" and is_authoritative:
                    rrf_scores[idx] *= 2.0

            # B. Semester & Academic Year matching boost
            if academic_year and academic_year in q_lower:
                rrf_scores[idx] *= 1.4
            if semester and semester in q_lower:
                rrf_scores[idx] *= 1.4

            # C. Departmental Isolation & Mismatch Filter (CRITICAL)
            if queried_depts:
                # Check if this document chunk is affiliated with the queried department
                matches_queried = False
                for qd in queried_depts:
                    if (f"{qd}_" in src or f"_{qd}." in src or 
                        f"department of {qd}" in issuer or 
                        any(re.search(p, title) for p in DEPT_PATTERNS[qd]) or
                        any(re.search(p, chunk_text) for p in DEPT_PATTERNS[qd])):
                        matches_queried = True
                        break

                # Check if this document chunk belongs explicitly to a different department
                other_depts = set(DEPT_PATTERNS.keys()) - queried_depts
                belongs_to_other = False
                for od in other_depts:
                    if (f"{od}_" in src or f"_{od}." in src or 
                        f"department of {od}" in issuer or 
                        any(re.search(p, title) for p in DEPT_PATTERNS[od])):
                        belongs_to_other = True
                        break

                if matches_queried:
                    rrf_scores[idx] *= 2.5  # Strong boost for matching target department
                elif belongs_to_other:
                    rrf_scores[idx] *= 0.0001  # Severe penalty for wrong department

            # D. Functional Topic Boosts
            if ("exam" in q_lower or "grading" in q_lower or "cgpa" in q_lower or "probation" in q_lower) and "exam" in src:
                rrf_scores[idx] *= 1.5
            elif ("calendar" in q_lower or "ক্যালেন্ডার" in q_lower or "ছুটি" in q_lower or "holiday" in q_lower) and ("calendar" in src or meta.get("doc_type") == "academic_calendar"):
                rrf_scores[idx] *= 2.0
            elif ("hall" in q_lower or "hostel" in q_lower or "হলে" in q_lower or "প্রভোস্ট" in q_lower) and "hall" in src:
                rrf_scores[idx] *= 1.5
            elif ("admission" in q_lower or "ভর্তি" in q_lower or "fee" in q_lower or "ফি" in q_lower) and "admission" in src:
                rrf_scores[idx] *= 1.5

    # Sort final merged candidates by fused score
    ranked_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Fallback if no matches found
    if not ranked_chunks:
        if any(w in q_lower for w in ["baust", "বিশ্ববিদ্যালয়", "university", "ক্যান্টনমেন্ট", "সৈয়দপুর"]):
            ranked_chunks = [(i, 1.0) for i in range(min(3, len(documents)))]

    if not ranked_chunks:
        return ""

    top_matches = ranked_chunks[:top_k]
    formatted_chunks = []
    seen_texts = set()
    total_chars = 0
    max_chars = 8000

    has_active_chunks = any(
        (metadata[idx].get("status") == "active" and metadata[idx].get("is_authoritative", 1))
        for idx, _ in top_matches if idx < len(metadata)
    )

    for idx, score_val in top_matches:
        if idx >= len(documents):
            continue
        text = documents[idx].strip()
        if text in seen_texts:
            continue

        meta = metadata[idx] if idx < len(metadata) else {}
        doc_status = meta.get("status", "active").lower()
        doc_is_auth = meta.get("is_authoritative", 1)

        # Skip outdated/superseded chunks when active authoritative versions exist
        if not is_archive_query and has_active_chunks:
            if doc_status in ["superseded", "archived"] or not doc_is_auth:
                continue

        seen_texts.add(text)

        source = meta.get("source", "BAUST Records")
        title = meta.get("title", source)
        version = meta.get("version", "1.0")
        status = meta.get("status", "active").upper()
        authority = meta.get("issuing_authority", "BAUST Authority")
        year = meta.get("academic_year", "")
        semester = meta.get("semester", "")
        session_parts = [p for p in [semester, year] if p]
        session_str = f" | Session: {' '.join(session_parts)}" if session_parts else ""

        auth_label = "CURRENT & AUTHORITATIVE" if meta.get("is_authoritative", 1) else status
        chunk_header = f"--- Document: {title} (Source: {source} | Status: {auth_label} | Version: {version} | Issuer: {authority}{session_str}) ---"
        chunk_str = f"{chunk_header}\n{text}"

        if total_chars + len(chunk_str) > max_chars and formatted_chunks:
            break
        formatted_chunks.append(chunk_str)
        total_chars += len(chunk_str)

    return "\n\n".join(formatted_chunks)
