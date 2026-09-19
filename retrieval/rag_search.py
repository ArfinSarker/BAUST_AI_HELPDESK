import os
import re
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
DOCS_PATH = os.path.join(VECTOR_DIR, "documents.pkl")
META_PATH = os.path.join(VECTOR_DIR, "metadata.pkl")
EMBED_PATH = os.path.join(VECTOR_DIR, "embeddings.npy")
FAISS_PATH = os.path.join(VECTOR_DIR, "index.faiss")

_documents = None
_metadata = None
_embeddings = None
_faiss_index = None
_embed_model = None

# Pure greeting regex patterns
GREETING_PATTERNS = [
    r"^(hi|hello|hey|assalamu\s*alaikum|salam|hi\s*there|hlo)[\s!.]*$",
    r"^(kemon\s*achen|kemon\s*acho|how\s*are\s*you|ke\s*tumi|tumi\s*ke|who\s*are\s*you|tumi\s*kara|tumi\s*kon)[\s?!.]*$"
]

# Comprehensive Bilingual Synonym & Context Map
SYNONYM_MAP = {
    "ভিসি": ["vc", "vice chancellor", "vice-chancellor", "executive head", "chancellor", "humayun kabir"],
    "উপাচার্য": ["vc", "vice chancellor", "vice-chancellor", "humayun kabir"],
    "প্রেসিডেন্ট": ["president", "bot", "board of trustees", "chief of army staff"],
    "আর্মি": ["army", "bangladesh army", "saidpur cantonment"],
    "শিক্ষক": ["faculty", "teacher", "professor", "lecturer", "assistant professor", "associate professor"],
    "শিক্ষিকা": ["faculty", "teacher", "professor", "lecturer"],
    "ডিপার্টমেন্ট": ["department", "cse", "eee", "me", "ce", "ipe", "bba", "english", "ict"],
    "বিভাগ": ["department", "cse", "eee", "me", "ce", "ipe", "bba", "english", "ict"],
    "সিএসই": ["cse", "computer science", "engineering", "nakib hayat"],
    "কম্পিউটার": ["cse", "computer science"],
    "মেকানিকাল": ["me", "mechanical"],
    "সিভিল": ["ce", "civil"],
    "ইইই": ["eee", "electrical", "rubina aktar"],
    "আইসিটি": ["ict", "information and communication", "ece"],
    "আইপিই": ["ipe", "industrial and production"],
    "বিবিএ": ["bba", "business administration", "fbs"],
    "ইংরেজি": ["english", "humanities"],
    "ভর্তি": ["admission", "apply", "eligibility", "intake", "batch", "winter", "summer", "application", "hsc"],
    "এডমিশন": ["admission", "apply", "eligibility", "form"],
    "ফি": ["fee", "tuition", "cost", "taka", "bdt", "tk", "charge", "payment", "waiver"],
    "খরচ": ["fee", "tuition", "cost", "taka", "bdt", "tk", "charge", "payment"],
    "টাকা": ["fee", "tuition", "cost", "taka", "bdt", "tk", "payment"],
    "স্কলারশিপ": ["scholarship", "waiver", "discount", "financial aid", "merit", "gpa 5"],
    "ওয়েভার": ["waiver", "scholarship", "discount"],
    "যোগাযোগ": ["contact", "phone", "email", "helpline", "hotline", "location"],
    "ফোন": ["phone", "mobile", "contact", "call", "01769675588", "01769675589"],
    "ঠিকানা": ["location", "saidpur", "cantonment", "address", "nilphamari"],
    "কোথায়": ["location", "saidpur", "cantonment", "address"],
    "ইতিহাস": ["history", "established", "background", "foundation", "parliament", "2015"],
    "কবে": ["established", "date", "year", "founded", "history"],
    "নকীব": ["nakib", "nakib hayat", "head of the department"],
    "রুহুল": ["ruhul", "dr md ruhul amin"],
    "রুফিজা": ["rufiza", "rufiza begum"],
    "জাহাঙ্গীর": ["jahangir", "dr md jahangir alam", "dean"],
    "হাসান": ["al-hasan", "hasan", "lecturer"],
    "রেজিস্ট্রার": ["registrar", "office of the registrar", "administrative"],
    "হোস্টেল": ["hostel", "hall", "dormitory", "accommodation", "residential", "seat"],
    "হল": ["hall", "hostel", "dormitory", "residential", "provost", "seat", "capacity"],
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
    "ক্যাম্পাস": ["campus", "saidpur cantonment", "facilities", "labs", "wifi", "cafeteria"],
    "লাইব্রেরি": ["library", "central library", "books", "reading room"],
    "পরীক্ষা": ["examination", "grading", "cgpa", "gpa", "term", "promotion", "clearance"],
    "গ্রেড": ["grade", "grading", "cgpa", "gpa", "marks", "scale"],
    "প্রমোশন": ["promotion", "probation", "retake", "improvement"],
    "ল্যাব": ["laboratory", "lab", "facilities", "equipment"],
    "গবেষণা": ["research", "publication", "cell", "journal", "newsletter"],
}


def is_greeting(text: str) -> bool:
    clean_text = text.strip().lower()
    return any(re.match(pattern, clean_text) for pattern in GREETING_PATTERNS)


def _get_embedding_model():
    """Lazy loader for SentenceTransformer to minimize cold-start time."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"[Embedding Model Warning]: {e}")
            _embed_model = None
    return _embed_model


def _load_store():
    """Loads text chunks, metadata, numpy vector embeddings and FAISS index."""
    global _documents, _metadata, _embeddings, _faiss_index
    if os.path.exists(DOCS_PATH):
        try:
            with open(DOCS_PATH, "rb") as f:
                _documents = pickle.load(f)
            if os.path.exists(META_PATH):
                with open(META_PATH, "rb") as f:
                    _metadata = pickle.load(f)
            else:
                _metadata = [{"source": "official_record"} for _ in _documents]

            if os.path.exists(EMBED_PATH):
                _embeddings = np.load(EMBED_PATH)
            else:
                _embeddings = None

            if os.path.exists(FAISS_PATH):
                try:
                    import faiss
                    _faiss_index = faiss.read_index(FAISS_PATH)
                except Exception:
                    _faiss_index = None

            print(f"[RAG Hybrid Store] Loaded {len(_documents)} document chunks with dense vector support.")
        except Exception as e:
            print(f"[RAG Store Error]: {e}")
            _documents, _metadata, _embeddings, _faiss_index = [], [], None, None
    else:
        _documents, _metadata, _embeddings, _faiss_index = [], [], None, None


def reload_vector_store():
    global _documents, _embeddings, _faiss_index
    _documents = None
    _embeddings = None
    _faiss_index = None
    _load_store()


def get_store():
    global _documents, _metadata, _embeddings, _faiss_index
    if _documents is None:
        _load_store()
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
                if idx >= 0 and idx < len(documents):
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
    Computes BM25-style lexical keyword and bilingual synonym matches.
    Returns: list of (keyword_score, chunk_index)
    """
    documents, metadata, _, _ = get_store()
    if not documents:
        return []

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

        # Exact phrase match boost
        if q_lower in doc_lower:
            score += 25.0

        for term in search_terms:
            if term in doc_lower:
                score += (len(term.split()) * 4.0) + (len(term) * 0.1)

        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def search_information(question: str, top_k: int = 8) -> str:
    """
    State-of-the-Art Hybrid Retrieval Engine:
    Combines Multilingual Dense Semantic Vector Search + Sparse Lexical/Synonym Search
    via Reciprocal Rank Fusion (RRF) and Source Domain Re-ranking.
    """
    if not question or not question.strip():
        return ""

    if is_greeting(question):
        return ""

    documents, metadata, _, _ = get_store()
    if not documents:
        return ""

    # 1. Dense Semantic Search (Vector Embedding)
    dense_results = dense_semantic_search(question, top_k=14)

    # 2. Sparse Lexical Search (Keyword & Synonyms)
    sparse_results = sparse_lexical_search(question, top_k=14)

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF Score = (Dense Weight / (60 + Dense Rank)) + (Sparse Weight / (60 + Sparse Rank))
    rrf_scores = {}
    dense_weight = 1.0
    sparse_weight = 1.2
    k_constant = 60.0

    for rank, (score, idx) in enumerate(dense_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (dense_weight / (k_constant + rank + 1))

    for rank, (score, idx) in enumerate(sparse_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (sparse_weight / (k_constant + rank + 1))

    # 4. Contextual Source Re-ranking
    # Boost document if question explicitly targets its domain
    q_lower = question.lower()
    for idx in rrf_scores:
        if idx < len(metadata):
            src = metadata[idx].get("source", "").lower()
            if "cse" in q_lower and "cse" in src:
                rrf_scores[idx] *= 1.4
            elif "eee" in q_lower and "eee" in src:
                rrf_scores[idx] *= 1.4
            elif "me" in q_lower and ("me_" in src or "mechanical" in src):
                rrf_scores[idx] *= 1.4
            elif "ipe" in q_lower and "ipe" in src:
                rrf_scores[idx] *= 1.4
            elif ("exam" in q_lower or "grading" in q_lower or "cgpa" in q_lower or "probation" in q_lower) and "exam" in src:
                rrf_scores[idx] *= 1.4
            elif ("hall" in q_lower or "hostel" in q_lower or "হলে" in q_lower or "প্রভোস্ট" in q_lower) and "hall" in src:
                rrf_scores[idx] *= 1.4
            elif ("admission" in q_lower or "ভর্তি" in q_lower or "fee" in q_lower or "ফি" in q_lower) and "admission" in src:
                rrf_scores[idx] *= 1.4

    # Sort final merged candidates by fused score
    ranked_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Fallback if no matches found
    if not ranked_chunks:
        if any(w in q_lower for w in ["baust", "বিশ্ববিদ্যালয়", "university", "ক্যান্টনমেন্ট", "সৈয়দপুর"]):
            ranked_chunks = [(i, 1.0) for i in range(min(4, len(documents)))]

    if not ranked_chunks:
        return ""

    top_matches = ranked_chunks[:top_k]
    formatted_chunks = []
    seen_texts = set()

    for idx, score_val in top_matches:
        if idx >= len(documents):
            continue
        text = documents[idx].strip()
        if text in seen_texts:
            continue
        seen_texts.add(text)

        source = metadata[idx].get("source", "BAUST Records") if idx < len(metadata) else "BAUST Records"
        formatted_chunks.append(f"--- Information from [{source}] ---\n{text}")

    return "\n\n".join(formatted_chunks)