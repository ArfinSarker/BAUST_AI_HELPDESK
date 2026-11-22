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
    "ভিসি": ["vc", "vice chancellor", "vice-chancellor", "executive head", "mobarak hossain", "mohammad mobarak hossain majumder"],
    "উপাচার্য": ["vc", "vice chancellor", "vice-chancellor", "mobarak hossain", "mohammad mobarak hossain majumder"],
    "vc": ["vice chancellor", "vice-chancellor", "ভিসি", "উপাচার্য", "mobarak hossain", "mohammad mobarak hossain majumder"],
    "vice chancellor": ["vc", "vice-chancellor", "ভিসি", "উপাচার্য", "mobarak hossain", "mohammad mobarak hossain majumder"],
    "vice-chancellor": ["vc", "vice chancellor", "ভিসি", "উপাচার্য", "mobarak hossain", "mohammad mobarak hossain majumder"],
    "ডিপার্টমেন্ট": ["department", "cse", "eee", "me", "ce", "ipe", "bba", "english", "ict"],
    "সিএসই": ["cse", "computer science", "nakib hayat"],
    "ভর্তি": ["admission", "apply", "eligibility", "winter", "summer"],
    "ফি": ["fee", "tuition", "cost", "taka", "bdt", "waiver"],
    "masters": ["masters", "mba", "emba", "postgraduate", "graduate", "মাস্টার্স", "স্নাতকোত্তর", "dba"],
    "mba": ["mba", "emba", "regular mba", "executive mba", "business administration", "dba", "masters", "মাস্টার্স"],
    "মাস্টার্স": ["masters", "mba", "emba", "postgraduate", "স্নাতকোত্তর", "dba"]
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

                # Ensure backward-compatible key parity: both 'source' and 'filename' are guaranteed
                for m in new_metadata:
                    if isinstance(m, dict):
                        prov = m.get("source") or m.get("filename") or ""
                        m["source"] = prov
                        m["filename"] = prov

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


_ROUTINE_CACHE = None
_ROUTINE_CACHE_MTIME = 0.0


def parse_class_routine_tables() -> dict:
    """
    Dynamically and generically parses official class routines from knowledge_base/*routine*.txt.
    Supports any Section (A, B, etc.), courses, faculty names, and day-by-day schedules with parities.
    Auto-invalidates when routine files are edited or uploaded.
    """
    global _ROUTINE_CACHE, _ROUTINE_CACHE_MTIME
    kb_dir = os.path.join(BASE_DIR, "knowledge_base")
    routine_files = glob.glob(os.path.join(kb_dir, "*routine*.txt"))
    if not routine_files:
        return {}

    max_mtime = max((os.path.getmtime(p) for p in routine_files if os.path.exists(p)), default=0.0)
    if _ROUTINE_CACHE is not None and max_mtime == _ROUTINE_CACHE_MTIME:
        return _ROUTINE_CACHE

    sections_data = {}
    table_pattern = re.compile(r'(\| Day\s*\|[^\n]+\n\|(?:\s*:?---+:?\s*\|)+\n(?:\|[^\n]+\n)+)', re.I)
    course_pattern = re.compile(r'\|\s*(?:\*\*)?([A-Z]{2,4}\s*\d{4})(?:\*\*)?\s*\|\s*([^|]+?)\s*\|', re.I)
    faculty_pattern = re.compile(r'\*\s*\*\*([^\*]+)\*\*\s*\n\s*\*\s*\*\*Designation:\*\*\s*([^\n]+)\s*\n\s*\*\s*\*\*Short Form:\*\*\s*([A-Za-z]+)', re.I)

    for fpath in routine_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        blocks = re.split(r'(?=#+\s*Bangladesh Army University.*?(?:CSE|Computer Science))', content, flags=re.I | re.S)
        for b in blocks:
            sec_m = re.search(r'Section:\s*\*?\*?\s*([AB])', b, re.I)
            lt_m = re.search(r'Level-Term:\s*\*?\*?\s*([^\n|]+)', b, re.I)
            if not sec_m:
                continue
            sec = sec_m.group(1).upper()
            lt = lt_m.group(1).strip() if lt_m else "3-II"

            courses = {}
            for cm in course_pattern.finditer(b):
                c_code = cm.group(1).strip()
                c_title = cm.group(2).strip()
                courses[c_code] = c_title

            teachers = {}
            for fm in faculty_pattern.finditer(b):
                t_name = fm.group(1).strip()
                t_desig = fm.group(2).strip()
                t_short = fm.group(3).strip().upper()
                teachers[t_short] = f"{t_name} ({t_desig})"

            days_map = {}
            m_tab = table_pattern.search(b)
            if m_tab:
                lines = m_tab.group(1).strip().splitlines()
                # Normalize time slots (e.g. 08.00 AM -> 08:00 AM)
                slots = [re.sub(r'(\d{1,2})\.(\d{2})', r'\1:\2', c.strip()) for c in lines[0].split('|')[2:] if c.strip()]
                for row in lines[2:]:
                    cols = [c.strip() for c in row.split('|')]
                    if len(cols) > 2:
                        day_raw = re.sub(r'[*_]', '', cols[1]).strip()
                        day_key = day_raw[:3].capitalize()
                        raw_items = []
                        row_classes = cols[2:]
                        for s_idx, cell in enumerate(row_classes):
                            if not cell or cell == '-' or s_idx >= len(slots):
                                continue
                            slot_time = slots[s_idx]

                            slot_tag = ""
                            if '#ODD#' in cell.upper():
                                slot_tag = 'ODD'
                            elif '#EVEN#' in cell.upper():
                                slot_tag = 'EVEN'

                            code_m = re.search(r'([A-Z]{2,4}\s*\d{4})', cell)
                            code = code_m.group(1) if code_m else cell
                            title = courses.get(code, '')

                            room_m = re.search(r'\[([^\]]+)\]', cell)
                            room = room_m.group(1).strip() if room_m else 'TBA'

                            t_m = re.search(r'\(([^)]+)\)', cell)
                            t_initials = t_m.group(1).strip() if t_m else ''
                            t_full = ', '.join([teachers.get(ti.strip().upper(), ti.strip()) for ti in t_initials.split(',') if ti.strip()]) if t_initials else 'TBA'

                            clean_cell = re.sub(r'#(?:EVEN|ODD)#', '', cell, flags=re.I).strip()

                            raw_items.append({
                                'time': slot_time,
                                'code': code,
                                'title': title,
                                'room': room,
                                'teacher': f"{t_initials} ({t_full})" if t_initials and t_full != t_initials else t_full,
                                'slot_tag': slot_tag,
                                'raw': clean_cell
                            })

                        # Merge consecutive periods of the same course (e.g. 3-hour sessional labs: Period 1-3, Period 4-6, Period 7-9)
                        merged_items = []
                        curr_item = None
                        for item in raw_items:
                            if curr_item and curr_item['code'] == item['code']:
                                curr_start = curr_item['time'].split('-')[0].strip()
                                new_end = item['time'].split('-')[-1].strip()
                                curr_item['time'] = f"{curr_start} - {new_end}"
                            else:
                                if curr_item:
                                    merged_items.append(curr_item)
                                curr_item = dict(item)
                        if curr_item:
                            merged_items.append(curr_item)
                        days_map[day_key] = merged_items

            if days_map or sec not in sections_data:
                sections_data[sec] = {
                    'level_term': lt,
                    'courses': courses,
                    'teachers': teachers,
                    'schedule': days_map,
                    'raw_block': b.strip()
                }

    _ROUTINE_CACHE = sections_data
    _ROUTINE_CACHE_MTIME = max_mtime
    return _ROUTINE_CACHE


def get_student_daily_classes(day_name: str, parity: str = "ALL", dept: str = "CSE", level_term: str = "3-II", section: str = "A") -> list[dict]:
    """
    Returns verified, structured class items for a specific day, section, and roll parity.
    Extracts directly from official parsed routine tables.
    """
    day_key = day_name[:3].capitalize()
    routines = parse_class_routine_tables()
    sec_key = (section or "A").upper()
    sec_data = routines.get(sec_key) or routines.get("A")
    if not sec_data:
        return []
    return sec_data["schedule"].get(day_key, [])


def get_class_routine_content(dept: str = "CSE", level_term: str = "3-II", section: str = "") -> str:
    """
    Returns authoritative Markdown routine content for requested section or all sections.
    """
    routines = parse_class_routine_tables()
    sec_key = (section or "").upper()
    block = ""
    if sec_key and sec_key in routines:
        block = routines[sec_key]["raw_block"]
    elif routines:
        block = "\n\n---\n\n".join(sec["raw_block"] for sec in routines.values())
    if block:
        # Normalize period times
        block = re.sub(r'(\d{1,2})\.(\d{2})\s*(AM|PM)', r'\1:\2 \3', block)
    return block


ROUTINE_KEYWORDS = [
    "routine", "class", "classes", "lab", "kalke", "ajke", "schedule", "period",
    "somoy", "shomoy", "kobe", "kokhon", "timetable", "time table", "ক্লাস",
    "রুটিন", "ল্যাব", "সময়সূচি", "সময়সূচি", "কি কি ক্লাস", "কখন ক্লাস", "পড়াশোনা",
    "আজকে", "কালকে", "আগামীকাল", "পরশু",
    "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "robibar", "robi", "shombar", "sombar", "shom", "som", "mongolbar", "mongol",
    "budhbar", "budbar", "budh", "bud", "brihoshpotibar", "brihospotibar", "brihaspatibar",
    "shukrobar", "sukrobar", "shonibar", "sonibar",
    "রবিবার", "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার",
    "রবি", "সোম", "মঙ্গল", "বুধ", "বৃহস্পতি", "শুক্র", "শনি",
    "কি কি আছে", "কি আছে", "ki ki ache", "ki ache"
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
    3. General Batch / Section Class Routine queries ("CSE 3-II section B kalke ki ki class", "section A routine").
    """
    if not question:
        return None

    clean_q = question.strip()
    q_low = clean_q.lower()
    is_routine = is_routine_query(clean_q)

    # Detect Section mentioned in question (Section A, Section B, Sec A, Sec B)
    sec_m = re.search(r'\b(?:section|sec)\s*[:\s-]?\s*([AB])\b|\b([AB])\s*(?:section|sec)\b', clean_q, re.I)
    query_section = (sec_m.group(1) or sec_m.group(2)).upper() if sec_m else ""

    # Detect personal pronoun (amar, amr, my, tar)
    has_personal_pronoun = any(re.search(r'\b' + re.escape(w) + r'\b', q_low) for w in ["amar", "amr", "my", "tar", "আমার", "আমারটি", "তার"])

    # 1. Extract student roll:
    # Explicit roll in question, OR lookup from history whenever routine or exam is asked
    roll_match = re.search(r'\b(08\d{6,14})\b', clean_q)
    roll_from_history = False
    if not roll_match and history and not query_section:
        student_followup_terms = [
            "seat", "room", "slot", "exam", "routine", "class", "kalke", "ajke", "schedule", "period",
            "সিট", "রুম", "কক্ষ", "পরীক্ষা", "রুটিন", "ক্লাস", "আজকে", "কালকে", "আমার", "amar", "tar", "amr",
            "robibar", "robi", "shombar", "sombar", "mongolbar", "budhbar", "brihoshpotibar",
            "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
            "রবিবার", "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "কি আছে", "ki ache"
        ]
        is_student_q = any(re.search(r'\b' + re.escape(term) + r'\b', q_low) or (len(term) >= 4 and term in q_low) for term in student_followup_terms)
        is_faculty_q = any(k in q_low for k in ["sir", "teacher", "faculty", "madam", "maam", "prof", "head", "hod", "gr", "ftz", "akz", "rifat", "zohra", "কে", "কার", "শিক্ষক", "স্যার"])
        if (is_student_q or is_routine or has_personal_pronoun) and not is_faculty_q:
            for msg in reversed(history[-6:]):
                if isinstance(msg, dict):
                    m = re.search(r'\b(08\d{6,14})\b', str(msg.get("content", "")))
                    if m:
                        roll_match = m
                        roll_from_history = True
                        break

    st_db = get_student_database()

    # Case A: Student Roll Found (Personal Query)
    if roll_match:
        roll = roll_match.group(1)
        s = st_db.get(roll)
        if s:
            if is_routine:
                temporal = get_bangladesh_temporal_context()

                # Determine student's section
                if query_section:
                    st_sec = query_section
                else:
                    last_3 = int(roll[-3:]) if roll[-3:].isdigit() else 0
                    st_sec = "B" if last_3 > 50 else "A"

                routine_text = get_class_routine_content(s["programme"], s["level_term"], section=st_sec)
                if routine_text:
                    weekend_today_note = "[সাপ্তাহিক ছুটি / WEEKEND - আজ কোনো ক্লাস নেই]" if temporal['is_today_weekend'] else "[ক্লাস ডে / REGULAR CLASS DAY]"
                    weekend_tmrw_note = "[সাপ্তাহিক ছুটি / WEEKEND - আগামীকাল কোনো ক্লাস নেই]" if temporal['is_tomorrow_weekend'] else "[ক্লাস ডে / REGULAR CLASS DAY]"

                    # Check for explicit day mentions in query:
                    day_explicit_map = {
                        "Sunday": ["sunday", "sun", "robibar", "robi", "রবিবার", "রবি"],
                        "Monday": ["monday", "mon", "shombar", "sombar", "shom", "som", "সোমবার", "সোম"],
                        "Tuesday": ["tuesday", "tue", "mongolbar", "mongol", "মঙ্গলবার", "মঙ্গল"],
                        "Wednesday": ["wednesday", "wed", "budhbar", "budbar", "budh", "bud", "বুধবার", "বুধ"],
                        "Thursday": ["thursday", "thu", "brihoshpotibar", "brihospotibar", "brihaspatibar", "brihospoti", "বৃহস্পতিবার", "বৃহস্পতি"],
                        "Friday": ["friday", "fri", "shukrobar", "sukrobar", "shukro", "sukro", "শুক্রবার", "শুক্র"],
                        "Saturday": ["saturday", "sat", "shonibar", "sonibar", "shoni", "soni", "শনিবার", "শনি"],
                    }
                    explicit_day = None
                    explicit_day_bn = None
                    bn_names = {
                        "Sunday": "রবিবার", "Monday": "সোমবার", "Tuesday": "মঙ্গলবার",
                        "Wednesday": "বুধবার", "Thursday": "বৃহস্পতিবার", "Friday": "শুক্রবার", "Saturday": "শনিবার"
                    }
                    for day_name, kws in day_explicit_map.items():
                        if any(re.search(r'\b' + re.escape(w) + r'\b', q_low) or (len(w) >= 4 and w in q_low) for w in kws):
                            explicit_day = day_name
                            explicit_day_bn = bn_names[day_name]
                            break

                    is_kalke = any(w in q_low for w in ["kalke", "kal", "tomorrow", "কালকে", "কাল", "আগামীকাল"])
                    is_ajke = any(w in q_low for w in ["ajke", "aj", "today", "আজকে", "আজ"])

                    if explicit_day:
                        target_day = explicit_day
                        target_day_bn = explicit_day_bn
                    elif is_kalke:
                        target_day = temporal["tomorrow_en"]
                        target_day_bn = temporal["tomorrow_bn"]
                    elif is_ajke:
                        target_day = temporal["today_en"]
                        target_day_bn = temporal["today_bn"]
                    else:
                        target_day = temporal["next_class_en"]
                        target_day_bn = temporal["next_class_bn"]

                    sched_items = get_student_daily_classes(target_day, parity=s["parity"], dept=s["programme"], level_term=s["level_term"], section=st_sec)
                    sched_lines = []
                    if target_day in ["Friday", "Saturday"]:
                        sched_lines.append(f"📅 **সাপ্তাহিক ছুটি:** আজ {target_day_bn} বিশ্ববিদ্যালয় বন্ধ রয়েছে। কোনো অ্যাকাডেমিক ক্লাস নেই।")
                        next_items = get_student_daily_classes(temporal["next_class_en"], parity=s["parity"], dept=s["programme"], level_term=s["level_term"], section=st_sec)
                        sched_lines.append(f"\n📌 **পরবর্তী ক্লাস ডে ({temporal['next_class_bn']} / {temporal['next_class_en']})-এর ক্লাসের সময়সূচি (রোল: `{s['roll']}`, সেকশন: {st_sec}):**")
                        for it in next_items:
                            tag_str = f" (#{it['slot_tag']}# ল্যাব)" if it.get('slot_tag') else ""
                            sched_lines.append(f"* `{it['time']}`: **{it['code']}** ({it['title']}){tag_str} — রুম/ল্যাব: `{it['room']}`, শিক্ষক: {it['teacher']}")
                    else:
                        sched_lines.append(f"📌 **{target_day_bn} ({target_day})-এর ক্লাসের সময়সূচি (রোল: `{s['roll']}`, সেকশন: {st_sec}):**")
                        for it in sched_items:
                            tag_str = f" (#{it['slot_tag']}# ল্যাব)" if it.get('slot_tag') else ""
                            sched_lines.append(f"* `{it['time']}`: **{it['code']}** ({it['title']}){tag_str} — রুম/ল্যাব: `{it['room']}`, শিক্ষক: {it['teacher']}")

                    schedule_block = "\n".join(sched_lines)

                    doc = (
                        f"--- Document: Student Profile & Official Class & Lab Routine (Authoritative Source: class_and_lab_routine.txt) ---\n"
                        f"STUDENT IDENTITY:\n"
                        f"* Student Name: {s['name']}\n"
                        f"* Student ID / Roll: {s['roll']}\n"
                        f"* Programme / Department: {s['programme']}\n"
                        f"* Academic Level & Term: {s['level']} (Level-Term Code: {s['level_term']}, Section: {st_sec})\n\n"
                        f"TEMPORAL CONTEXT (BANGLADESH TIME - UTC+6):\n"
                        f"* Current Date: {temporal['date_str']} | Current Time: {temporal['time_str']}\n"
                        f"* Today: {temporal['today_en']} ({temporal['today_bn']}) {weekend_today_note}\n"
                        f"* Tomorrow: {temporal['tomorrow_en']} ({temporal['tomorrow_bn']}) {weekend_tmrw_note}\n"
                        f"* Next Regular Academic Class Day: {temporal['next_class_en']} ({temporal['next_class_bn']})\n\n"
                        f"VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR STUDENT:\n"
                        f"{schedule_block}\n\n"
                        f"CRITICAL FORMATTING INSTRUCTIONS FOR AI ASSISTANT:\n"
                        f"1. You MUST list the student's daily classes EXACTLY as listed in 'VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR STUDENT' above.\n"
                        f"2. Present the exact time for each period (e.g. `08:00 AM - 10:55 AM` for 3-hour sessional labs, `11:30 AM - 12:25 PM`, etc.). In BAUST, sessional labs span 3 consecutive periods (3 hours).\n"
                        f"3. If a lab has an (#EVEN# ল্যাব) or (#ODD# ল্যাব) tag, you MUST explicitly include this label in the course details (e.g. `CSE 3202: Artificial Intelligence Sessional (#EVEN# ল্যাব)`).\n"
                        f"4. For each class, show Course Code & Title, Room/Lab, and Faculty directly.\n\n"
                        f"OFFICIAL BATCH CLASS & LAB ROUTINE (Department: {s['programme']}, {s['level']}, Section: {st_sec}):\n"
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
                if roll_from_history and not any(k in q_low for k in ["seat", "room", "exam", "slot", "সিট", "রুম", "কক্ষ", "পরীক্ষা"]):
                    return None
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

    # Case B: No roll found, but user asks for Batch / Section Class Routine
    # e.g. "cse 3 II section B er kalke ki ki class ache", "section A routine", "routine"
    if is_routine:
        lt_match = re.search(r'\b(3[-–]?[iI]{2}|3[-–]?2|level\s*[-–]?\s*3\s*term\s*[-–]?\s*(?:2|ii))\b', q_low)
        dept_match = any(re.search(pat, q_low) for pat in DEPT_PATTERNS.get("cse", []))
        if lt_match or dept_match or "routine" in q_low or "class" in q_low or query_section:
            temporal = get_bangladesh_temporal_context()
            target_sec = query_section or "A"
            routine_text = get_class_routine_content("CSE", "3-II", section=target_sec)
            if routine_text:
                weekend_today_note = "[সাপ্তাহিক ছুটি / WEEKEND - আজ কোনো ক্লাস নেই]" if temporal['is_today_weekend'] else "[ক্লাস ডে / REGULAR CLASS DAY]"
                weekend_tmrw_note = "[সাপ্তাহিক ছুটি / WEEKEND - আগামীকাল কোনো ক্লাস নেই]" if temporal['is_tomorrow_weekend'] else "[ক্লাস ডে / REGULAR CLASS DAY]"

                day_explicit_map = {
                    "Sunday": ["sunday", "sun", "robibar", "robi", "রবিবার", "রবি"],
                    "Monday": ["monday", "mon", "shombar", "sombar", "shom", "som", "সোমবার", "সোম"],
                    "Tuesday": ["tuesday", "tue", "mongolbar", "mongol", "মঙ্গলবার", "মঙ্গল"],
                    "Wednesday": ["wednesday", "wed", "budhbar", "budbar", "budh", "bud", "বুধবার", "বুধ"],
                    "Thursday": ["thursday", "thu", "brihoshpotibar", "brihospotibar", "brihaspatibar", "brihospoti", "বৃহস্পতিবার", "বৃহস্পতি"],
                    "Friday": ["friday", "fri", "shukrobar", "sukrobar", "shukro", "sukro", "শুক্রবার", "শুক্র"],
                    "Saturday": ["saturday", "sat", "shonibar", "sonibar", "shoni", "soni", "শনিবার", "শনি"],
                }
                explicit_day = None
                explicit_day_bn = None
                bn_names = {
                    "Sunday": "রবিবার", "Monday": "সোমবার", "Tuesday": "মঙ্গলবার",
                    "Wednesday": "বুধবার", "Thursday": "বৃহস্পতিবার", "Friday": "শুক্রবার", "Saturday": "শনিবার"
                }
                for day_name, kws in day_explicit_map.items():
                    if any(re.search(r'\b' + re.escape(w) + r'\b', q_low) or (len(w) >= 4 and w in q_low) for w in kws):
                        explicit_day = day_name
                        explicit_day_bn = bn_names[day_name]
                        break

                is_kalke = any(w in q_low for w in ["kalke", "kal", "tomorrow", "কালকে", "কাল", "আগামীকাল"])
                is_ajke = any(w in q_low for w in ["ajke", "aj", "today", "আজকে", "আজ"])

                if explicit_day:
                    target_day = explicit_day
                    target_day_bn = explicit_day_bn
                elif is_kalke:
                    target_day = temporal["tomorrow_en"]
                    target_day_bn = temporal["tomorrow_bn"]
                elif is_ajke:
                    target_day = temporal["today_en"]
                    target_day_bn = temporal["today_bn"]
                else:
                    target_day = temporal["next_class_en"]
                    target_day_bn = temporal["next_class_bn"]

                sched_items = get_student_daily_classes(target_day, parity="ALL", section=target_sec)
                sched_lines = []
                if target_day in ["Friday", "Saturday"]:
                    sched_lines.append(f"📅 **সাপ্তাহিক ছুটি:** আজ {target_day_bn} বিশ্ববিদ্যালয় বন্ধ রয়েছে। কোনো অ্যাকাডেমিক ক্লাস নেই।")
                    next_items = get_student_daily_classes(temporal["next_class_en"], parity="ALL", section=target_sec)
                    sched_lines.append(f"\n📌 **পরবর্তী ক্লাস ডে ({temporal['next_class_bn']} / {temporal['next_class_en']})-এর ক্লাসের সময়সূচি (CSE 3-II, সেকশন: {target_sec}):**")
                    for it in next_items:
                        tag_str = f" (#{it['slot_tag']}# ল্যাব)" if it.get('slot_tag') else ""
                        sched_lines.append(f"* `{it['time']}`: **{it['code']}** ({it['title']}){tag_str} — রুম/ল্যাব: `{it['room']}`, শিক্ষক: {it['teacher']}")
                else:
                    sched_lines.append(f"📌 **{target_day_bn} ({target_day})-এর ক্লাসের সময়সূচি (CSE 3-II, সেকশন: {target_sec}):**")
                    for it in sched_items:
                        tag_str = f" (#{it['slot_tag']}# ল্যাব)" if it.get('slot_tag') else ""
                        sched_lines.append(f"* `{it['time']}`: **{it['code']}** ({it['title']}){tag_str} — রুম/ল্যাব: `{it['room']}`, শিক্ষক: {it['teacher']}")

                schedule_block = "\n".join(sched_lines)

                return (
                    f"--- Document: Official Class & Lab Routine (Authoritative Source: class_and_lab_routine.txt) ---\n"
                    f"BATCH SPECIFICATION:\n"
                    f"* Department: Computer Science and Engineering (CSE)\n"
                    f"* Academic Level & Term: Level- 3 Term- II\n"
                    f"* Section: {target_sec}\n\n"
                    f"TEMPORAL CONTEXT (BANGLADESH TIME - UTC+6):\n"
                    f"* Current Date: {temporal['date_str']} | Current Time: {temporal['time_str']}\n"
                    f"* Today: {temporal['today_en']} ({temporal['today_bn']}) {weekend_today_note}\n"
                    f"* Tomorrow: {temporal['tomorrow_en']} ({temporal['tomorrow_bn']}) {weekend_tmrw_note}\n"
                    f"* Next Regular Academic Class Day: {temporal['next_class_en']} ({temporal['next_class_bn']})\n\n"
                    f"VERIFIED AUTHORITATIVE DAILY SCHEDULE FOR BATCH:\n"
                    f"{schedule_block}\n\n"
                    f"OFFICIAL BATCH CLASS & LAB ROUTINE (Section: {target_sec}):\n"
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

# =====================================================================
# =====================================================================
# UNIFIED MULTI-HOP ENTITY & KNOWLEDGE GRAPH ENGINE
# =====================================================================
_UNIFIED_CACHE = None
_UNIFIED_MTIME = 0.0

def get_unified_entity_graph():
    """
    Parses and cross-indexes ALL entities across ALL 37 knowledge base documents in memory:
    - 420+ Faculty & Staff members across all 9 departments with verified phone, email, designation.
    - All Routine Teacher Initials / Short Forms (e.g. GR -> S. M Golam Rifat, AA -> Md. Al Amin).
    - 3,180+ Student Profiles from seatplan files (Roll, Name, Department, Level-Term, Parity, Room, Seat, Slot).
    - Course Codes & Titles.
    Ultra-fast (7.5ms), thread-safe, and auto-invalidates when knowledge base files change.
    """
    global _UNIFIED_CACHE, _UNIFIED_MTIME
    kb_dir = os.path.join(BASE_DIR, "knowledge_base")
    all_files = glob.glob(os.path.join(kb_dir, "*.txt"))
    if not all_files:
        return [], {}, {}, {}

    max_mtime = max((os.path.getmtime(p) for p in all_files if os.path.exists(p)), default=0.0)
    if _UNIFIED_CACHE is not None and max_mtime == _UNIFIED_MTIME:
        return _UNIFIED_CACHE

    faculty_list = []
    abbreviations = {}
    students = {}
    courses = {}

    dept_label_map = {
        "CSE": "Computer Science and Engineering (CSE)",
        "EEE": "Electrical and Electronic Engineering (EEE)",
        "ME": "Mechanical Engineering (ME)",
        "IPE": "Industrial and Production Engineering (IPE)",
        "CE": "Civil Engineering (CE)",
        "FBS": "Faculty of Business Studies (BBA/MBA)",
        "ENGLISH": "Department of English",
        "ICT": "Information and Communication Technology (ICT/ICE)",
        "ARTS": "Department of Arts and Sciences"
    }

    # 1. Parse all Faculty & Staff across all departments
    fac_files = glob.glob(os.path.join(kb_dir, "*faculty*.txt")) + glob.glob(os.path.join(kb_dir, "*department*.txt"))
    for fpath in fac_files:
        fname = os.path.basename(fpath)
        raw_dept = fname.split("_")[0].upper()
        full_dept = dept_label_map.get(raw_dept, f"Department of {raw_dept}")
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            continue

        in_table = False
        for l in lines:
            l = l.strip()
            if l.startswith("|") and "---" not in l:
                cells = [c.strip().strip("*") for c in l.split("|")[1:-1]]
                if any(h in cells[0].lower() for h in ["name", "faculty", "teacher"]):
                    in_table = True
                    continue
                if in_table and len(cells) >= 3 and cells[0]:
                    faculty_list.append({
                        "name": cells[0],
                        "designation": cells[1] if len(cells) > 1 else "",
                        "phone": cells[2] if len(cells) > 2 else "",
                        "email": cells[3] if len(cells) > 3 else "",
                        "qualifications": cells[4] if len(cells) > 4 else "",
                        "initials": "",
                        "department": full_dept,
                        "source": fname
                    })
            elif not l.startswith("|"):
                in_table = False

        current_prof = None
        for l in lines:
            l = l.strip()
            hm = re.match(r"^###\s+(?:Prof\w*\.?|Dr\w*\.?|Engr\w*\.?|Mr\w*\.?|Ms\w*\.?|Mst\w*\.?|Mrs\w*\.?)?\s*(.*)", l, re.I)
            if hm and not any(k in l.lower() for k in ["introduction", "overview", "specialization", "laboratory", "facilities", "programs", "members", "alumni", "directory", "committee", "office", "history", "contact"]):
                raw_name = hm.group(1).strip()
                if len(raw_name) > 3 and not raw_name.startswith("Level") and not any(k in raw_name.lower() for k in ["alumni", "directory", "committee"]):
                    current_prof = {
                        "name": l.replace("###", "").strip(),
                        "designation": "",
                        "phone": "",
                        "email": "",
                        "qualifications": "",
                        "initials": "",
                        "department": full_dept,
                        "source": fname
                    }
                    faculty_list.append(current_prof)
            elif current_prof and l.startswith("- **"):
                if "designation" in l.lower():
                    current_prof["designation"] = re.sub(r"-\s*\*\*Designation\*\*:\s*", "", l, flags=re.I).strip()
                elif "phone" in l.lower():
                    current_prof["phone"] = re.sub(r"-\s*\*\*Phone\*\*:\s*", "", l, flags=re.I).strip()
                elif "email" in l.lower():
                    current_prof["email"] = re.sub(r"-\s*\*\*Email\*\*:\s*", "", l, flags=re.I).strip()
                elif "education" in l.lower() or "qualification" in l.lower():
                    current_prof["qualifications"] = re.sub(r"-\s*\*\*(?:Education\s*/\s*Qualifications|Qualifications)\*\*:\s*", "", l, flags=re.I).strip()
                elif "short form" in l.lower() or "initials" in l.lower():
                    sf_m = re.search(r'Short Form(?:\s*/\s*Initials)?\s*(?:\*\*)?[:\s\*]*([A-Za-z0-9]+)', l, re.I)
                    if sf_m:
                        sf = sf_m.group(1).upper()
                        current_prof["initials"] = sf
                        if sf not in abbreviations:
                            abbreviations[sf] = {
                                "name": current_prof["name"],
                                "designation": current_prof.get("designation", "Lecturer"),
                                "phone": current_prof.get("phone", ""),
                                "email": current_prof.get("email", ""),
                                "qualifications": current_prof.get("qualifications", ""),
                                "department": full_dept
                            }

    # 2. Parse all Routine Abbreviations & Courses
    routine_files = glob.glob(os.path.join(kb_dir, "*routine*.txt"))
    for rf in routine_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                rlines = [line.strip() for line in f.readlines()]
            for i in range(len(rlines) - 2):
                l1, l2, l3 = rlines[i], rlines[i+1], rlines[i+2]
                if re.match(r"^[A-Z]{2,5}(?:\([a-z]\))?$", l2) and any(d in l3.lower() for d in ["professor", "lecturer", "engineer", "assistant", "officer"]):
                    if l2 not in abbreviations:
                        abbreviations[l2] = {"name": l1, "designation": l3}
            for i in range(len(rlines) - 2):
                m_name = re.match(r"^\*\s*\*\*(.*?)\*\*", rlines[i])
                if m_name:
                    nm = m_name.group(1).strip()
                    if any(bad in nm.lower() for bad in ["designation", "short form", "note", "department"]):
                        continue
                    for j in range(1, 4):
                        if i + j < len(rlines):
                            sf_m = re.search(r'Short Form(?:\s*/\s*Initials)?\s*(?:\*\*)?[:\s\*]*([A-Za-z0-9]+)', rlines[i+j], re.I)
                            ds_m = re.search(r'Designation\s*(?:\*\*)?[:\s\*]*(.*?)$', rlines[i+j], re.I)
                            if sf_m:
                                sf = sf_m.group(1).upper()
                                if sf not in abbreviations:
                                    abbreviations[sf] = {"name": nm, "designation": ds_m.group(1).strip() if ds_m else ""}

            for l in rlines:
                cm = re.search(r"\|\s*\*\*([A-Z]{2,4}\s*\d{4})\*\*\s*\|\s*([^\|]+)\|", l)
                if cm:
                    ccode = cm.group(1).replace(" ", "").upper()
                    ctitle = cm.group(2).strip()
                    courses[ccode] = {"code": ccode, "title": ctitle, "file": os.path.basename(rf)}
        except Exception:
            pass

    # Enrich abbreviations with faculty phone, email & department
    for abbr, info in abbreviations.items():
        clean_n = [w for w in re.findall(r"[a-zA-Z]{3,}", info["name"].lower()) if w not in ["prof", "dr", "engr", "md", "mst", "mrs", "mr"]]
        matched_prof = None
        if clean_n:
            for f in faculty_list:
                f_low = f["name"].lower()
                if all(w in f_low for w in clean_n):
                    matched_prof = f
                    break
        if matched_prof:
            info["phone"] = matched_prof.get("phone", "")
            info["email"] = matched_prof.get("email", "")
            info["qualifications"] = matched_prof.get("qualifications", "")
            info["department"] = matched_prof.get("department", "")
        else:
            if not info.get("phone"): info["phone"] = ""
            if not info.get("email"): info["email"] = ""
            if not info.get("qualifications"): info["qualifications"] = ""
            if not info.get("department"):
                m_dept = re.search(r"(?:,\s*|\b)(CSE|EEE|ME|IPE|CE|BBA|English|A&S)\b", info.get("designation", ""), re.I)
                info["department"] = dept_label_map.get(m_dept.group(1).upper(), f"Department of {m_dept.group(1).upper()}") if m_dept else "BAUST"

    # 3. Parse all Students across seatplan files
    slot_map = {'A': ('Slot-A', '09:00 AM - 10:00 AM'), 'B': ('Slot-B', '11:00 AM - 12:00 PM'), 'C': ('Slot-C', '01:00 PM - 02:00 PM')}
    for fpath in glob.glob(os.path.join(kb_dir, "*seatplan*.txt")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                slines = [line.strip() for line in f.readlines() if line.strip()]
            c_prog = "Computer Science and Engineering"
            c_lvl = ""
            c_slot = ""
            for idx, l in enumerate(slines):
                pm = re.match(r'^(?:Programme|Program):\s*(.*)', l, re.I)
                if pm: c_prog = pm.group(1).strip()
                lm = re.match(r'^(Level\s*[-–]?\s*\d+\s*Term\s*[-–]?\s*[IVX0-9]+.*)', l, re.I)
                if lm: c_lvl = lm.group(1).strip()
                sm = re.match(r'^(Slot\s*[-–:]?\s*[A-Z0-9].*)', l, re.I)
                if sm: c_slot = sm.group(1).strip()
                if re.match(r'^\d{8,16}$', l) and idx + 1 < len(slines):
                    roll = l
                    if roll not in students:
                        name = slines[idx + 1]
                        room = slines[idx + 2] if idx + 2 < len(slines) else ""
                        seat = slines[idx + 3] if idx + 3 < len(slines) else ""
                        spfx = re.match(r'^([ABC])-\d+-\d+', seat.strip(), re.I)
                        s_name, s_time = slot_map.get(spfx.group(1).upper(), (c_slot or "Slot-A", "")) if spfx else (c_slot or "Slot-A", "")
                        lt_m = re.search(r'Level\s*[-–]?\s*(\d+)\s*Term\s*[-–]?\s*([IVX0-9]+)', c_lvl, re.I)
                        norm_lt = f"{lt_m.group(1)}-{lt_m.group(2)}" if lt_m else c_lvl
                        parity = "ODD" if roll[-1].isdigit() and int(roll[-1]) % 2 != 0 else "EVEN"
                        students[roll] = {
                            "roll": roll,
                            "name": name,
                            "programme": c_prog,
                            "level": c_lvl,
                            "level_term": norm_lt,
                            "parity": parity,
                            "slot": s_name,
                            "time": s_time,
                            "room": room,
                            "seat": seat,
                            "source": os.path.basename(fpath)
                        }
        except Exception:
            pass

    _UNIFIED_CACHE = (faculty_list, abbreviations, students, courses)
    _UNIFIED_MTIME = max_mtime
    return _UNIFIED_CACHE


def get_all_faculty_database() -> list[dict]:
    """Backward compatible wrapper returning the unified faculty list."""
    fac, _, _, _ = get_unified_entity_graph()
    return fac


def get_all_abbreviations_map() -> dict:
    """Backward compatible wrapper returning the unified abbreviations dictionary."""
    _, abbr, _, _ = get_unified_entity_graph()
    return abbr


def format_email_markdown(email_str: str) -> str:
    """Formats one or multiple emails into clean, individual markdown mailto links separated by comma."""
    if not email_str:
        return ""
    emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', email_str)
    if not emails:
        return email_str.strip()
    seen = set()
    deduped = []
    for e in emails:
        el = e.lower()
        if el not in seen:
            seen.add(el)
            deduped.append(e)
    return ", ".join([f"[{e}](mailto:{e})" for e in deduped])


def format_phone_markdown(phone_str: str) -> str:
    """Formats one or multiple phone numbers into clean, individual markdown tel links with notes outside."""
    if not phone_str:
        return ""
    parts = [p.strip() for p in re.split(r'[,/]', phone_str) if p.strip()]
    formatted = []
    for p in parts:
        m = re.search(r'(\+?\d[\d\s\-]{8,15}\d|\d{10,14})', p)
        if m:
            num = m.group(1).strip()
            clean_digits = re.sub(r'[\s\-]', '', num)
            note = p.replace(num, '').strip()
            if note:
                formatted.append(f"[{num}](tel:{clean_digits}) {note}")
            else:
                formatted.append(f"[{num}](tel:{clean_digits})")
        else:
            formatted.append(p)
    return ", ".join(formatted)


def sanitize_contact_markdown(text: str) -> str:
    """Post-processor that cleans up email, phone link formatting, and unnatural Bengali translations in markdown."""
    if not text:
        return ""
    # 1. Un-nest backticks from markdown links: `[text](url)` -> [text](url)
    text = re.sub(r'`(\[[^\]]+\]\([^\)]+\))`', r'\1', text)

    # 2. Fix malformed mailto: [email1, email2](mailto:email1, email2)
    def _fix_mailto(match):
        inner_label = match.group(1)
        inner_href = match.group(2)
        return format_email_markdown(inner_label + " " + inner_href)

    text = re.sub(r'\[([^\]]*@[^\]]*)\]\(mailto:([^\)]*)\)', _fix_mailto, text)

    # 3. Fix malformed tel: [phone (note)](tel:phone (note)) or [phone1, phone2](tel:...)
    def _fix_tel(match):
        inner_label = match.group(1)
        return format_phone_markdown(inner_label)

    text = re.sub(r'\[([^\]]+)\]\(tel:([^)]*(?:\([^)]*\)[^)]*)*)\)', _fix_tel, text)

    # 4. Clean unnatural, repetitive Bengali academic qualification translations
    # Fix dotted acronyms like 'সি.এসই' -> 'সিএসই'
    text = re.sub(r'সি[\.\s]+এসই', 'সিএসই', text)
    text = re.sub(r'ই[\.\s]+ই[\.\s]+ই', 'ইইই', text)
    text = re.sub(r'আই[\.\s]+পি[\.\s]+ই', 'আইপিই', text)

    # University transliterations (e.g. কেইউ -> খুলনা বিশ্ববিদ্যালয় (KU))
    uni_map = {
        r'\bকেইউ\b': 'খুলনা বিশ্ববিদ্যালয় (KU)',
        r'\bজেইউ\b': 'জাহাঙ্গীরনগর বিশ্ববিদ্যালয় (JU)',
        r'\bডিইউ\b': 'ঢাকা বিশ্ববিদ্যালয় (DU)',
        r'\bসিইউ\b': 'চট্টগ্রাম বিশ্ববিদ্যালয় (CU)',
        r'\bআরইউ\b': 'রাজশাহী বিশ্ববিদ্যালয় (RU)',
    }
    for k, v in uni_map.items():
        text = re.sub(k, v, text)

    # Fix 'বি.এসসি (বি.এসসি. ইন ...)' tautology
    text = re.sub(
        r'বি[\.\s]*এসসি\s*\(\s*(?:বি[\.\s]*এসসি[\.\s]*)?ইন\s*([^\)]+)\)',
        r'বিএসসি ইন \1',
        text,
        flags=re.IGNORECASE
    )
    # Fix 'এম.এসসি (এম.এসসি. ইন ...)' tautology
    text = re.sub(
        r'এম[\.\s]*এসসি\s*\(\s*(?:এম[\.\s]*এসসি[\.\s]*)?ইন\s*([^\)]+)\)',
        r'এমএসসি ইন \1',
        text,
        flags=re.IGNORECASE
    )
    # Fix any remaining nested repetition like 'বি.এসসি (বি.এসসি...)'
    text = re.sub(
        r'(বি[\.\s]*এসসি|এম[\.\s]*এসসি)\s*\(\s*\1\s*[,;]?\s*([^\)]*)\)',
        r'\1 (\2)',
        text,
        flags=re.IGNORECASE
    )
    return text


def find_faculty_or_staff_by_name(query: str) -> Optional[str]:
    """Finds a faculty member by name or token match and returns formatted markdown."""
    fac, _, _, _ = get_unified_entity_graph()
    tokens = [w for w in re.findall(r"[a-zA-Z\u0980-\u09FF]{3,}", query.lower()) if w not in ["sir", "teacher", "faculty", "ke", "kake", "er", "who", "is", "কে", "কার", "স্যার", "ম্যাডাম", "শিক্ষক"]]
    if "al amin" in query.lower() or "alamin" in query.lower():
        tokens = ["al", "amin"]
    if not tokens:
        return None
    for f in fac:
        if all(tok in f["name"].lower() for tok in tokens):
            c_parts = []
            if f.get("phone"):
                cp = format_phone_markdown(f["phone"])
                if cp: c_parts.append(f"ফোন: {cp}")
            if f.get("email"):
                ce = format_email_markdown(f["email"])
                if ce: c_parts.append(f"ইমেইল: {ce}")
            c_str = " | ".join(c_parts) if c_parts else "বিভাগীয় অফিসে যোগাযোগ করুন"
            return (
                f"--- Document: Faculty Member Profile (Authoritative Source: {f['source']}) ---\n"
                f"* **শিক্ষক/কর্মকর্তার নাম (Full Name):** {f['name']}\n"
                f"* **পদবি (Designation):** {f['designation']}\n"
                f"* **বিভাগ (Department):** {f['department']}, বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুর।\n"
                f"* **যোগাযোগ (Contact):** {c_str}\n"
            )
    return None


def resolve_faculty_member_or_initials(question: str) -> Optional[str]:
    """Finds a faculty member by initials/short form or alias."""
    _, abbr, _, _ = get_unified_entity_graph()
    for w in re.findall(r'\b[A-Za-z]{2,5}\b', question):
        w_up = w.upper()
        if w_up in abbr:
            info = abbr[w_up]
            c_parts = []
            if info.get("phone"):
                cp = format_phone_markdown(info["phone"])
                if cp: c_parts.append(f"ফোন: {cp}")
            if info.get("email"):
                ce = format_email_markdown(info["email"])
                if ce: c_parts.append(f"ইমেইল: {ce}")
            c_str = " | ".join(c_parts) if c_parts else "বিভাগীয় অফিসে যোগাযোগ করুন"
            return (
                f"--- Document: Faculty Routine Short Form & Teacher Profile (Authoritative Source: class_and_lab_routine.txt) ---\n"
                f"* **শিক্ষক সংক্ষেপণ (Faculty Short Form):** `{w_up}`\n"
                f"* **পূর্ণনাম (Full Name):** {info['name']}\n"
                f"* **পদবি (Designation):** {info.get('designation', 'Lecturer')}\n"
                f"* **বিভাগ (Department):** {info.get('department', 'BAUST')}, বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুর।\n"
                f"* **যোগাযোগ (Contact):** {c_str}\n"
            )
    return None


def search_information(question: str, top_k: int = 4, history: list = None) -> str:
    """
    PERMANENT UNIFIED KNOWLEDGE & ENTITY GRAPH RETRIEVER:
    1. Cross-references multi-hop entities (Faculty, Routine Abbreviations, Students, Courses) in 7ms.
    2. Resolves follow-ups seamlessly across conversation history.
    3. Assembles Authoritative Entity Dossiers (Direct phone, email, live Bangladesh temporal schedule).
    4. Simultaneously executes Dense Semantic Vector (FAISS) + Sparse BM25Okapi across all 37 documents.
    5. Fuses Entity Dossier + Top Hybrid RAG Passages into one comprehensive, anti-hallucination context.
    """
    if not question or not question.strip():
        return ""

    if is_greeting(question):
        return ""

    clean_q = question.strip()
    q_low = clean_q.lower()
    faculty_list, abbreviations, students, courses = get_unified_entity_graph()

    entity_blocks = []
    expanded_search_terms = []

    followup_terms = {
        "contact", "phone", "email", "mail", "number", "no", "call", "hotline", "jogajog", "thaken",
        "time", "slot", "room", "seat", "somoy", "shomoy", "kobe", "kokhon", "tarikh",
        "date", "when", "building", "kothay", "details", "routine", "class",
        "যোগাযোগ", "ফোন", "নাম্বার", "নম্বর", "ইমেইল", "মেইল", "সময়", "সময়", "রুম", "সিট", "স্লট", "কোথায়", "ক্লাস", "রুটিন"
    }
    q_tokens = set(re.findall(r"[a-zA-Z\u0980-\u09FF]{2,}", q_low))
    is_followup = len(q_tokens) <= 4 and bool(q_tokens & followup_terms)

    active_person = ""
    active_roll = ""
    if is_followup and history:
        for msg in reversed(history[-4:]):
            if isinstance(msg, dict):
                content = str(msg.get("content", ""))
                if not active_person:
                    pm = re.search(r'(?:শিক্ষক/কর্মকর্তার নাম|শিক্ষকের নাম|পূর্ণনাম|পূর্ণ নাম|Full Name|Name)[\s\*\:\)]+([A-Za-z\s\.\-]{3,35})', content, re.I)
                    if pm and not any(k in pm.group(1).lower() for k in ["baust", "student", "roll", "cse", "ipe", "department", "academic"]):
                        active_person = pm.group(1).strip()
                if not active_roll:
                    rm = re.search(r'\b(08\d{6,14})\b', content)
                    if rm:
                        active_roll = rm.group(1)

    # 1. Match Initials / Abbreviations (AA, GR, FTZ, GC, MI, MZH, RH, etc.)
    matched_abbr = None
    for w in re.findall(r'\b[A-Za-z]{2,5}\b', clean_q):
        w_up = w.upper()
        if w_up in abbreviations:
            matched_abbr = abbreviations[w_up].copy()
            matched_abbr["abbr"] = w_up
            break

    if matched_abbr:
        entity_lines = [
            f"--- Teacher / Faculty Profile (Abbreviation: `{matched_abbr['abbr']}`) ---",
            f"* **শিক্ষক সংক্ষেপণ (Faculty Short Form):** `{matched_abbr['abbr']}`",
            f"* **পূর্ণনাম (Full Name):** {matched_abbr['name']}",
            f"* **পদবি (Designation):** {matched_abbr.get('designation', 'Lecturer')}",
            f"* **বিভাগ (Department):** {matched_abbr.get('department', 'BAUST')}, বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুর।"
        ]
        if matched_abbr.get("phone"):
            cp = format_phone_markdown(matched_abbr["phone"])
            if cp: entity_lines.append(f"* **মোবাইল / ফোন (Phone):** {cp}")
        if matched_abbr.get("email"):
            ce = format_email_markdown(matched_abbr["email"])
            if ce: entity_lines.append(f"* **ইমেইল (Email):** {ce}")
        if matched_abbr.get("qualifications"):
            entity_lines.append(f"* **শিক্ষাগত যোগ্যতা (Qualifications):** {matched_abbr['qualifications']}")
        entity_blocks.append("\n".join(entity_lines) + "\n")
        expanded_search_terms.append(matched_abbr['name'])

    # 2. Check for Vice Chancellor / Executive Leadership queries
    vc_triggers = ["vc", "vice chancellor", "vice-chancellor", "vice_chancellor", "উপাচার্য", "ভিসি"]
    is_vc_query = any(re.search(r'\b' + re.escape(trig) + r'\b', q_low) for trig in vc_triggers) or "উপাচার্য" in q_low or "ভিসি" in q_low
    if is_vc_query:
        vc_lines = [
            "--- Leadership & Governance: Office of the Vice Chancellor ---",
            "* **উপাচার্যের নাম (Full Name):** ব্রিগেডিয়ার জেনারেল মোহাম্মদ মোবারক হোসেন মজুমদার, পিএসসি (Brig Gen Mohammad Mobarak Hossain Majumder, psc)",
            "* **পদবি (Designation):** উপাচার্য (Vice Chancellor) ও চেয়ারম্যান, একাডেমিক কাউন্সিল",
            "* **বিভাগ/প্রতিষ্ঠান:** বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুর।",
            "* **মোবাইল / ফোন (Phone):** [+8801769675550](tel:+8801769675550)",
            "* **ইমেইল (Email):** [vc@baust.edu.bd](mailto:vc@baust.edu.bd)",
            "* **শিক্ষাগত যোগ্যতা (Qualifications):** বিএসসি ও এমএসসি ইন মেকানিক্যাল ইঞ্জিনিয়ারিং (ME), বুয়েট (BUET)"
        ]
        entity_blocks.append("\n".join(vc_lines) + "\n")
        expanded_search_terms.extend(["Mobarak Hossain", "Vice Chancellor", "উপাচার্য"])

    # 3. Match Faculty by Direct Name or Follow-up Person from History
    target_person_query = active_person if (is_followup and active_person) else clean_q
    if not matched_abbr and not is_vc_query:
        INSTITUTIONAL_STOPWORDS = {
            "baust", "university", "varsity", "campus", "cantonment", "saidpur",
            "department", "dept", "faculty", "teacher", "sir", "madam", "student", "students",
            "alumni", "directory", "office", "section", "wing", "center", "centre", "hall", "hostel",
            "routine", "schedule", "exam", "admission", "notice", "result", "grade", "syllabus", "course",
            "fee", "fees", "cost", "waiver", "rules", "policy", "information", "info", "helpdesk", "assistant",
            "ke", "kake", "er", "who", "is", "the", "what", "where", "how", "when", "which", "tell", "me", "about",
            "vc", "dean", "head", "provost", "proctor", "registrar", "treasurer",
            "কে", "কার", "স্যার", "ম্যাডাম", "শিক্ষক", "শিক্ষিকা", "বায়েস্ট", "বিশ্ববিদ্যালয়", "ক্যান্টনমেন্ট", "সৈয়দপুর", "সৈয়দপুর", "বিভাগ", "অনুষদ", "অফিস", "ভিসি", "উপাচার্য"
        }
        person_tokens = [w for w in re.findall(r"[a-zA-Z\u0980-\u09FF]{3,}", target_person_query.lower()) if w not in followup_terms and w not in INSTITUTIONAL_STOPWORDS]
        if "al amin" in target_person_query.lower() or "alamin" in target_person_query.lower():
            person_tokens = ["al", "amin"]
        if person_tokens:
            for f in faculty_list:
                f_name_low = f["name"].lower()
                if any(bad in f_name_low for bad in ["alumni", "directory", "committee"]):
                    continue
                if all(tok in f_name_low for tok in person_tokens):
                    entity_lines = [
                        f"--- Faculty / Staff Profile (Direct Match: {f['name']}) ---",
                        f"* **শিক্ষক/কর্মকর্তার নাম (Full Name):** {f['name']}",
                        f"* **পদবি (Designation):** {f['designation']}",
                        f"* **বিভাগ (Department):** {f['department']}, বাংলাদেশ আর্মি ইউনিভার্সিটি অব সায়েন্স অ্যান্ড টেকনোলজি (BAUST), সৈয়দপুর।"
                    ]
                    if f.get("phone"):
                        cp = format_phone_markdown(f["phone"])
                        if cp: entity_lines.append(f"* **মোবাইল / ফোন (Phone):** {cp}")
                    if f.get("email"):
                        ce = format_email_markdown(f["email"])
                        if ce: entity_lines.append(f"* **ইমেইল (Email):** {ce}")
                    if f.get("qualifications"):
                        entity_lines.append(f"* **শিক্ষাগত যোগ্যতা (Qualifications):** {f['qualifications']}")
                    entity_blocks.append("\n".join(entity_lines) + "\n")
                    expanded_search_terms.append(f['name'])
                    break

    # 3. Match Student Routine, Exam Seatplan, or Batch / Section Routine
    routine_or_student_doc = resolve_student_or_routine_query(clean_q, history=history)
    if routine_or_student_doc:
        entity_blocks.append(routine_or_student_doc)
    elif not matched_abbr and not entity_blocks:
        # Check student name disambiguation if user searched by a student's partial name
        name_matches, disambiguation_doc = find_student_matches_by_name(clean_q)
        if disambiguation_doc:
            entity_blocks.append(disambiguation_doc)

    # 4. Check for entire department faculty list request
    dept_faculty_doc = get_department_faculty_file_content(clean_q)
    if dept_faculty_doc:
        entity_blocks.append(dept_faculty_doc)

    # 5. Hybrid Vector + BM25 Lexical Ranking across ALL 37 files
    documents, metadata, _, _ = get_store()
    rag_blocks = []

    if documents:
        search_q = f"{clean_q} {' '.join(expanded_search_terms)}".strip()
        dense_res = dense_semantic_search(search_q, top_k=8)
        sparse_res = sparse_lexical_search(search_q, top_k=8)

        rrf_scores = {}
        for rank, (score, idx) in enumerate(dense_res):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60.0 + rank + 1))
        for rank, (score, idx) in enumerate(sparse_res):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.2 / (60.0 + rank + 1))

        # Department filtering & boosting
        queried_depts = set()
        for dept_code, patterns in DEPT_PATTERNS.items():
            if any(re.search(p, q_low) for p in patterns):
                queried_depts.add(dept_code)

        dept_file_prefixes = {
            "cse": ["cse_"],
            "ce": ["ce_"],
            "eee": ["eee_"],
            "me": ["me_"],
            "ipe": ["ipe_"],
            "bba": ["fbs_", "bba_"],
            "english": ["english_"],
            "ict": ["ict_"]
        }

        def _matches_dept(file_source: str, d_code: str) -> bool:
            prefixes = dept_file_prefixes.get(d_code, [f"{d_code}_"])
            return any(p in file_source or f"_{d_code}." in file_source for p in prefixes)

        for idx in list(rrf_scores.keys()):
            meta = metadata[idx] if idx < len(metadata) else {}
            # Resolves metadata-key mismatch: handles both 'source' and 'filename'
            src = (meta.get("source") or meta.get("filename") or "").lower()
            if queried_depts:
                matches_queried = any(_matches_dept(src, qd) for qd in queried_depts)
                belongs_to_other = any(_matches_dept(src, od) for od in set(DEPT_PATTERNS.keys()) - queried_depts)
                if matches_queried:
                    rrf_scores[idx] *= 2.5
                elif belongs_to_other:
                    rrf_scores[idx] *= 0.001

            if ("calendar" in q_low or "ছুটি" in q_low or "holiday" in q_low) and "calendar" in src:
                rrf_scores[idx] *= 2.0
            elif ("admission" in q_low or "ভর্তি" in q_low or "fee" in q_low or "ফি" in q_low) and "admission" in src:
                rrf_scores[idx] *= 1.5

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        seen_texts = set()
        total_chars = 0
        max_chars = 7000

        for idx in sorted_indices:
            if idx < len(documents):
                chunk_text = documents[idx].strip()
                if chunk_text in seen_texts:
                    continue
                seen_texts.add(chunk_text)
                meta = metadata[idx] if idx < len(metadata) else {}
                title = meta.get("title") or meta.get("filename") or meta.get("source") or "BAUST Document"
                src = meta.get("source") or meta.get("filename") or "Record"
                block = f"--- Document: {title} (Source: {src}) ---\n{chunk_text}"
                if total_chars + len(block) > max_chars and rag_blocks:
                    break
                rag_blocks.append(block)
                total_chars += len(block)

    # 6. Compose Unified Context
    final_output = []
    if entity_blocks:
        final_output.append("### AUTHORITATIVE ENTITY PROFILE & RECORDS:\n" + "\n\n".join(entity_blocks))
    if rag_blocks:
        final_output.append("### RELEVANT UNIVERSITY KNOWLEDGE PASSAGES:\n" + "\n\n".join(rag_blocks))

    return "\n\n".join(final_output)
