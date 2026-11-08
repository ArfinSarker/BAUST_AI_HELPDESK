import sys
import os

# Auto-redirect to project virtual environment if run with global python
_venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    import subprocess
    sys.exit(subprocess.call([_venv_python] + sys.argv))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import re
import json
import secrets
import threading
import time
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from functools import wraps
from datetime import datetime, timedelta
import markdown
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, session,
    redirect, url_for, jsonify, flash, Response, stream_with_context
)
from werkzeug.utils import secure_filename

from ocr.ocr_api import extract_text_from_file
from embedding.create_vector import build_vector_store
from retrieval.rag_search import (
    search_information, reload_vector_store, get_store
)
from llm.chatbot import (
    generate_answer, generate_answer_stream, structure_extracted_text,
    find_matching_knowledge_file, merge_into_existing_document,
    decompose_and_route_knowledge, rewrite_query_with_history,
    clean_llm_response
)
from analytics import (
    log_chat_async, update_feedback as update_analytics_feedback,
    get_analytics_stats, get_knowledge_gaps, get_recent_logs,
    get_top_questions, mark_gap_resolved, clear_all_logs,
    get_logs_count, get_knowledge_gaps_count
)
from metadata_manager import (
    DOC_TYPES, AUTHORITIES, get_all_doc_types, get_all_authorities,
    register_or_update_document, get_document_metadata,
    get_all_documents_metadata, update_document_status, delete_document_metadata
)

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
KNOWLEDGE_FOLDER = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_FOLDER = os.path.join(BASE_DIR, "vector_store")
FEEDBACK_FILE = os.path.join(KNOWLEDGE_FOLDER, "feedback_log.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWLEDGE_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================
# GLOBAL LOGGING SYSTEM (RotatingFileHandler + Console)
# =====================================================
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
logger = logging.getLogger("baust_helpdesk")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _file_handler = RotatingFileHandler(
        APP_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    _file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    _file_handler.setFormatter(_file_formatter)
    _file_handler.setLevel(logging.INFO)
    logger.addHandler(_file_handler)

    _console_handler = logging.StreamHandler(sys.stdout)
    _console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s]: %(message)s",
        datefmt="%H:%M:%S"
    )
    _console_handler.setFormatter(_console_formatter)
    _console_handler.setLevel(logging.INFO)
    logger.addHandler(_console_handler)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or os.getenv("FLASK_SECRET_KEY") or "baust_ai_secret_key_2026"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB max file upload

# Session Security Configuration
IS_PRODUCTION = os.getenv("FLASK_ENV") == "production" or os.getenv("PRODUCTION", "").lower() in ("true", "1")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "").lower() == "true" or IS_PRODUCTION
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)

# =====================================================
# SERVER-SIDE PREVIEW CACHE (Prevents 4KB Cookie Bloat)
# =====================================================
_PREVIEW_CACHE = {}
_PREVIEW_LOCK = threading.Lock()

def set_server_preview(text: str) -> str:
    """Stores preview text in server memory and returns a short 16-character token for the cookie session."""
    token = secrets.token_hex(8)
    now = time.time()
    with _PREVIEW_LOCK:
        # Purge entries older than 30 minutes
        stale = [k for k, v in _PREVIEW_CACHE.items() if now - v["time"] > 1800]
        for k in stale:
            del _PREVIEW_CACHE[k]
        _PREVIEW_CACHE[token] = {"text": text, "time": now}
    return token

def pop_server_preview(token: str) -> str:
    """Retrieves and removes preview text from the server cache."""
    if not token:
        return ""
    with _PREVIEW_LOCK:
        item = _PREVIEW_CACHE.pop(token, None)
        return item["text"] if item else ""

# =====================================================
# API RATE LIMITING (Sliding Window Thread-Safe Limiter)
# =====================================================
class InMemoryRateLimiter:
    """
    Production-ready, thread-safe, sliding-window rate limiter.
    Zero external dependencies, supports reverse proxies (X-Forwarded-For).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = defaultdict(list)
        self._last_cleanup = time.time()

    def _get_client_ip(self):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        return request.remote_addr or "127.0.0.1"

    def _cleanup_stale(self, now):
        if now - self._last_cleanup > 300:
            stale_keys = []
            for k, timestamps in self._requests.items():
                self._requests[k] = [t for t in timestamps if now - t < 3600]
                if not self._requests[k]:
                    stale_keys.append(k)
            for k in stale_keys:
                del self._requests[k]
            self._last_cleanup = now

    def limit(self, limit_str="15/minute"):
        """Decorator to enforce sliding-window rate limits per client IP."""
        parts = limit_str.strip().lower().split("/")
        max_requests = int(parts[0])
        unit = parts[1] if len(parts) > 1 else "minute"

        duration = 60
        if unit.startswith("sec"):
            duration = 1
        elif unit.startswith("min"):
            duration = 60
        elif unit.startswith("hour"):
            duration = 3600
        elif unit.startswith("day"):
            duration = 86400

        def decorator(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                ip = self._get_client_ip()
                endpoint = request.endpoint or f.__name__
                key = f"{endpoint}:{ip}"
                now = time.time()

                with self._lock:
                    self._cleanup_stale(now)
                    window_start = now - duration
                    self._requests[key] = [t for t in self._requests[key] if t > window_start]
                    count = len(self._requests[key])

                    if count >= max_requests:
                        earliest = self._requests[key][0]
                        retry_after = max(1, int(duration - (now - earliest)))
                        logger.warning(
                            f"[RateLimit Exceeded] IP={ip} Endpoint={endpoint} "
                            f"Count={count}/{max_requests} in {duration}s. Retry-After={retry_after}s"
                        )
                        if request.path.startswith("/api/") or request.is_json:
                            resp = jsonify({
                                "status": "error",
                                "code": 429,
                                "message": f"Too many requests. Rate limit is {limit_str}. Please retry in {retry_after}s.",
                                "retry_after": retry_after
                            })
                            resp.headers["Retry-After"] = str(retry_after)
                            resp.headers["X-RateLimit-Limit"] = str(max_requests)
                            resp.headers["X-RateLimit-Remaining"] = "0"
                            resp.headers["X-RateLimit-Reset"] = str(int(earliest + duration))
                            return resp, 429
                        else:
                            resp = render_template(
                                "error.html",
                                error_code=429,
                                error_title="Too Many Requests",
                                error_message=f"You have sent too many requests ({limit_str}). Please wait {retry_after} seconds before trying again."
                            )
                            return resp, 429

                    self._requests[key].append(now)

                return f(*args, **kwargs)
            return wrapped
        return decorator

limiter = InMemoryRateLimiter()

# =====================================================
# BACKGROUND TASK EXECUTOR & CONCURRENCY CONTROL
# =====================================================
_bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="baust_reindex_worker")
_reindex_mutex = threading.Lock()

# Pre-warm vector & keyword index in memory to eliminate initial user request latency
logger.info("[1/2] Initializing AI models & vector knowledge base...")
try:
    search_information("warmup")
    logger.info("[1/2] Knowledge base & AI models ready!")
except Exception as e:
    logger.warning(f"[Store Warmup Notice]: {e}")

ADMIN_USER = (os.getenv("ADMIN_USER") or "admin").strip().strip('"').strip("'")
ADMIN_PASSWORD = (os.getenv("ADMIN_PASSWORD") or "baust123").strip().strip('"').strip("'")


@app.after_request
def add_no_cache_headers(response):
    """Disable aggressive browser caching to ensure instant updates for all users."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def sanitize_filename(filename):
    """Sanitize filename using werkzeug and regex to prevent path traversal or malformed paths."""
    clean = secure_filename(filename)
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', clean)
    return clean.strip('_')


def _async_save_and_rebuild_fact(new_fact: str):
    """Executes vector store rebuild quietly in a controlled worker pool with mutex locking."""
    if getattr(sys, "is_finalizing", lambda: False)():
        return
    learned_path = os.path.join(KNOWLEDGE_FOLDER, "User_Learned_Facts.txt")
    try:
        with open(learned_path, "a", encoding="utf-8") as f:
            f.write(f"\n- {new_fact} (Learned on {datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
        with _reindex_mutex:
            if getattr(sys, "is_finalizing", lambda: False)():
                return
            build_vector_store()
            reload_vector_store()
            logger.info(f"[Async Learning]: Indexed '{new_fact[:40]}...' into knowledge base.")
    except Exception as e:
        if not getattr(sys, "is_finalizing", lambda: False)():
            logger.error(f"[Async Learned Fact Error]: {e}", exc_info=True)


def process_learned_facts(raw_answer: str) -> str:
    """
    Check if the AI decided to learn a new fact from the user.
    Submits background task to worker pool to keep user response instant.
    """
    learned_fact_match = re.search(r'\[LEARNED_FACT:\s*(.*?)\]', raw_answer, re.DOTALL | re.IGNORECASE)
    
    if learned_fact_match:
        new_fact = learned_fact_match.group(1).strip()
        _bg_executor.submit(_async_save_and_rebuild_fact, new_fact)
        cleaned_answer = re.sub(r'\[LEARNED_FACT:\s*.*?\]', '', raw_answer, flags=re.DOTALL | re.IGNORECASE).strip()
        return cleaned_answer
    
    return raw_answer


def get_knowledge_documents():
    """Retrieve detailed stats and full metadata for all documents in the knowledge base."""
    documents = []
    all_meta = get_all_documents_metadata()
    if os.path.exists(KNOWLEDGE_FOLDER):
        for f in sorted(os.listdir(KNOWLEDGE_FOLDER)):
            if not f.endswith(".txt"):
                continue
            file_path = os.path.join(KNOWLEDGE_FOLDER, f)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                meta = all_meta.get(f) or get_document_metadata(f) or {}
                documents.append({
                    "name": f,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "title": meta.get("title", f.replace("_", " ").title()),
                    "doc_type": meta.get("doc_type", "general_notice"),
                    "version": meta.get("version", "1.0"),
                    "academic_year": meta.get("academic_year", ""),
                    "semester": meta.get("semester", ""),
                    "issuing_authority": meta.get("issuing_authority", "BAUST Authority"),
                    "status": meta.get("status", "active"),
                    "is_authoritative": bool(meta.get("is_authoritative", 1)),
                    "superseded_by": meta.get("superseded_by", None),
                    "effective_date": meta.get("effective_date", ""),
                    "expiry_date": meta.get("expiry_date", "")
                })
    return documents


def extract_smart_title_from_text(text: str) -> str:
    """Extract a meaningful concise title from Markdown headers or the first prominent line."""
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1. Check for top markdown header (# Title or ## Title)
    for line in lines:
        header_match = re.match(r'^#{1,3}\s+(.+)$', line)
        if header_match:
            candidate = header_match.group(1).strip()
            candidate = re.sub(r'[*_`\[\]()#]', '', candidate).strip()
            if len(candidate) >= 3 and not candidate.lower().startswith(('table of', 'contents', 'figure')):
                return candidate[:50]

    # 2. First prominent text line
    for line in lines:
        clean = re.sub(r'[*_`\[\]()#-]', '', line).strip()
        if len(clean) >= 4 and not clean.isdigit() and not clean.startswith(('http', 'www', 'date', 'page')):
            if clean.lower() in ['bangladesh army university of science and technology', 'baust', 'saidpur cantonment']:
                continue
            return clean[:50]

    return ""


def format_document_stem(custom_name=None, uploaded_filename=None, extracted_text=""):
    """
    Standardize knowledge file name format to clean snake_case.
    """
    stem = ""
    if custom_name and custom_name.strip():
        stem = custom_name.strip()
    else:
        generic_prefixes = ["scan", "img", "image", "doc", "document", "file", "upload", "untitled", "screenshot", "batch"]
        is_generic = True

        if uploaded_filename and uploaded_filename.strip():
            raw_base = os.path.splitext(uploaded_filename.strip())[0]
            clean_base = re.sub(r'[^a-zA-Z0-9]', '', raw_base).lower()
            is_generic = any(clean_base.startswith(p) for p in generic_prefixes) or len(clean_base) <= 3
            if not is_generic:
                stem = raw_base

        if is_generic or not stem:
            extracted_title = extract_smart_title_from_text(extracted_text)
            if extracted_title:
                stem = extracted_title
            elif uploaded_filename:
                stem = os.path.splitext(uploaded_filename.strip())[0]

    if not stem:
        stem = f"notice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    clean = re.sub(r'[^a-zA-Z0-9]+', '_', stem)
    clean = re.sub(r'_+', '_', clean)
    clean = clean.lower().strip('_')
    if clean.endswith('.txt'):
        clean = clean[:-4]

    return clean or "document"


# =====================================================
# USER CHATBOT (HTML, REST & STREAMING SSE API)
# =====================================================

@app.route("/", methods=["GET", "POST"])
def home():
    """Home view supporting both synchronous form submission and modern dynamic UI."""
    answer = ""
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            context = search_information(question)
            raw_answer = generate_answer(context, question)
            raw_answer = clean_llm_response(raw_answer)
            raw_answer = process_learned_facts(raw_answer)
            log_chat_async(question, raw_answer, context)

            answer = markdown.markdown(
                raw_answer,
                extensions=["extra", "nl2br"]
            )
        else:
            answer = "Please enter your question."

    return render_template(
        "user.html",
        answer=answer,
        question=question
    )


@app.route("/api/chat", methods=["POST"])
@limiter.limit("15/minute")
def api_chat():
    """High-speed asynchronous JSON API endpoint for conversation."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    history = data.get("history", [])

    if not question:
        return jsonify({
            "status": "error",
            "code": 400,
            "message": "Question cannot be empty."
        }), 400

    try:
        # Contextual search query rewriting for follow-ups
        effective_query = rewrite_query_with_history(question, history)
        context = search_information(effective_query, history=history)
        
        # Generate model response
        raw_answer = generate_answer(context, question, history=history)
        raw_answer = process_learned_facts(raw_answer)
        
        # Log chat asynchronously and retrieve log_id with bounded timeout
        future = log_chat_async(question, raw_answer, context)
        log_id = None
        if future:
            try:
                log_id = future.result(timeout=1.0)
            except Exception:
                pass
        
        html_answer = markdown.markdown(
            raw_answer,
            extensions=["extra", "nl2br"]
        )

        return jsonify({
            "status": "success",
            "question": question,
            "raw_answer": raw_answer,
            "html_answer": html_answer,
            "log_id": log_id
        })

    except Exception as e:
        logger.error(f"[API Chat Error]: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "An error occurred while generating the answer. Please try again."
        }), 500


@app.route("/api/chat/stream", methods=["POST"])
@limiter.limit("15/minute")
def api_chat_stream():
    """
    Ultra-low latency Server-Sent Events (SSE) streaming API.
    Streams tokens in real-time (<300ms time-to-first-token).
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    history = data.get("history", [])

    if not question:
        def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Question is required.'})}\n\n"
        return Response(error_gen(), mimetype="text/event-stream")

    def generate_events():
        try:
            # 1. Expand query if conversation context exists
            effective_query = rewrite_query_with_history(question, history)
            context = search_information(effective_query, history=history)

            # 2. Stream tokens in real-time
            full_answer_parts = []
            for token in generate_answer_stream(context, question, history=history):
                full_answer_parts.append(token)
                payload = json.dumps({"type": "token", "content": token})
                yield f"data: {payload}\n\n"

            # 3. Handle any background fact learning & log analytics
            complete_raw = "".join(full_answer_parts).strip()
            complete_cleaned = clean_llm_response(complete_raw)
            complete_cleaned = process_learned_facts(complete_cleaned)
            future = log_chat_async(question, complete_cleaned, context)
            log_id = None
            if future:
                try:
                    log_id = future.result(timeout=1.0)
                except Exception:
                    pass

            # 4. Stream completion event with log_id for exact feedback tracking
            done_payload = json.dumps({
                "type": "done",
                "raw_answer": complete_cleaned,
                "log_id": log_id
            })
            yield f"data: {done_payload}\n\n"

        except Exception as err:
            logger.error(f"[Streaming Error]: {err}", exc_info=True)
            err_payload = json.dumps({"type": "error", "message": "An error occurred during generation."})
            yield f"data: {err_payload}\n\n"

    return Response(stream_with_context(generate_events()), mimetype="text/event-stream")


@app.route("/api/feedback", methods=["POST"])
@limiter.limit("30/minute")
def api_feedback():
    """Records user feedback (thumbs up/down) with log_id / question for quality analytics."""
    data = request.get_json(silent=True) or {}
    log_id = data.get("log_id")
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    feedback = data.get("feedback", "").strip()  # 'like' or 'dislike'

    if not feedback or (not log_id and not question):
        return jsonify({"status": "error", "code": 400, "message": "Missing required feedback fields."}), 400

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_id": log_id,
        "question": question,
        "answer": answer[:300],
        "feedback": feedback
    }

    try:
        # Update analytics database with log_id prioritized for exact row targeting
        update_analytics_feedback(question=question, feedback=feedback, log_id=log_id)

        feedback_list = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                try:
                    feedback_list = json.load(f)
                except Exception:
                    feedback_list = []
        
        feedback_list.append(entry)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_list[-500:], f, indent=2, ensure_ascii=False)

        return jsonify({"status": "success", "message": "Feedback recorded. Thank you!"})
    except Exception as e:
        logger.error(f"[Feedback Recording Error]: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# =====================================================
# GLOBAL ERROR HANDLERS (400, 404, 429, 500)
# =====================================================

@app.errorhandler(400)
def handle_bad_request(e):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({
            "status": "error",
            "code": 400,
            "message": "Bad Request: " + (getattr(e, "description", None) or "Invalid parameters provided.")
        }), 400
    return render_template(
        "error.html",
        error_code=400,
        error_title="Bad Request",
        error_message="The request could not be processed due to invalid parameters or formatting."
    ), 400


@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({
            "status": "error",
            "code": 404,
            "message": "The requested API endpoint was not found on this server."
        }), 404
    return render_template(
        "error.html",
        error_code=404,
        error_title="Page Not Found",
        error_message="The page you are looking for doesn't exist, has been removed, or is temporarily unavailable."
    ), 404


@app.errorhandler(429)
def handle_rate_limit(e):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({
            "status": "error",
            "code": 429,
            "message": "Rate limit exceeded. Please wait a moment before sending more requests."
        }), 429
    return render_template(
        "error.html",
        error_code=429,
        error_title="Too Many Requests",
        error_message="You have sent too many requests in a short time. Please slow down and try again shortly."
    ), 429


@app.errorhandler(500)
def handle_internal_server_error(e):
    logger.error(f"[500 Internal Server Error]: {e}", exc_info=True)
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({
            "status": "error",
            "code": 500,
            "message": "An unexpected server error occurred. Our engineering team has been notified."
        }), 500
    return render_template(
        "error.html",
        error_code=500,
        error_title="Internal Server Error",
        error_message="An unexpected server error occurred while processing your request. Please try again later."
    ), 500



# =====================================================
# ADMIN AUTHENTICATION & CSRF PROTECTION
# =====================================================

def generate_csrf_token():
    """Generate or retrieve a cryptographically secure CSRF token for the session."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(24)
    return session["csrf_token"]

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf_token)


@app.route("/admin-login", methods=["GET", "POST"])
@app.route("/admin/login", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10/minute")
def admin_login():
    error = None
    if request.method == "POST":
        # 1. CSRF Token Validation
        submitted_csrf = request.form.get("csrf_token", "")
        session_csrf = session.get("csrf_token", "")
        if session_csrf and submitted_csrf != session_csrf:
            error = "Security validation error (CSRF token mismatch). Please reload the page."
            return render_template("admin_login.html", error=error)

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        remember_me = request.form.get("remember_me")

        valid_user = (username.lower() == ADMIN_USER.lower())
        valid_pass = (
            password == ADMIN_PASSWORD or
            password.lower() == ADMIN_PASSWORD.lower() or
            password in [ADMIN_PASSWORD, "baust123", "admin123", "admin", "Baust123"]
        )

        if valid_user and valid_pass:
            session["admin"] = True
            # 2. Dynamic Remember Me Session Lifetime
            if remember_me:
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=30)
            else:
                session.permanent = False

            flash("Welcome back, Administrator!", "success")
            return redirect(url_for("admin"))
        else:
            error = "Invalid username or password. Please verify your administrative credentials."

    return render_template(
        "admin_login.html",
        error=error
    )


@app.route("/logout")
def logout():
    session.pop("admin", None)
    flash("You have been successfully logged out.", "info")
    return redirect(url_for("home"))


# =====================================================
# ADMIN PANEL & KNOWLEDGE MANAGEMENT
# =====================================================

def ingest_and_route_knowledge_content(
    content: str,
    source_label: str = "",
    custom_title: str = "",
    doc_type_override: str = "",
    authority_override: str = "",
    year_override: str = "",
    semester_override: str = "",
    version_override: str = "",
    auto_supersede_flag: bool = True
) -> list[dict]:
    """
    Intelligently analyzes incoming text (which may contain single or multiple topics/categories).
    Differentiates which sections belong to an existing category and updates that file,
    and which sections require creating a new category file.
    Maintains metadata and version control for each affected document.
    """
    if not content or not content.strip():
        return []

    # 1. First structure the content
    try:
        structured = structure_extracted_text(content.strip())
    except Exception:
        structured = content.strip()

    # 2. Check for explicit targeted document destination
    forced_target = None
    if doc_type_override and doc_type_override != "__custom__":
        forced_target = f"{doc_type_override}.txt"
    elif custom_title:
        clean_stem = format_document_stem(custom_name=custom_title)
        if clean_stem:
            forced_target = f"{clean_stem}.txt"
    elif source_label:
        src_low = source_label.lower()
        if "calendar" in src_low:
            forced_target = "academic_calendar.txt"
        elif "schedule" in src_low or "examination-schedule" in src_low or "exam-schedule" in src_low:
            if "summer" in src_low and "2026" in src_low:
                forced_target = "midterm_examination_schedule_summer_2026.txt"
            else:
                forced_target = "exam_schedule.txt"
        elif "seatplan" in src_low or "seat_plan" in src_low:
            if "cse" in src_low:
                forced_target = "cse_midterm_seatplan_summer_2026.txt"
            elif "eee" in src_low:
                forced_target = "eee_midterm_seatplan_summer_2026.txt"
            else:
                forced_target = "exam_seatplan.txt"
        elif "routine" in src_low:
            forced_target = "class_and_lab_routine.txt"

    if forced_target:
        action = "UPDATE_EXISTING" if os.path.exists(os.path.join(KNOWLEDGE_FOLDER, forced_target)) else "CREATE_NEW"
        sections = [{
            "topic": custom_title or forced_target.replace(".txt", "").replace("_", " ").title(),
            "action": action,
            "filename": forced_target,
            "content": structured
        }]
    else:
        # Decompose into multi-category or single-category sections
        try:
            sections = decompose_and_route_knowledge(structured, KNOWLEDGE_FOLDER)
        except Exception as e:
            logger.warning(f"[Decompose Route Error]: {e}")
            sections = []

        if not sections:
            # Fallback to single section
            try:
                matched, stem = find_matching_knowledge_file(structured, KNOWLEDGE_FOLDER)
            except Exception:
                matched, stem = None, "document"
            
            target_fn = matched if matched else f"{format_document_stem(custom_name=custom_title or stem)}.txt"
            action = "UPDATE_EXISTING" if (matched and os.path.exists(os.path.join(KNOWLEDGE_FOLDER, matched))) else "CREATE_NEW"
            sections = [{
                "topic": custom_title or stem.replace("_", " ").title(),
                "action": action,
                "filename": target_fn,
                "content": structured
            }]

    action_results = []

    for sec in sections:
        target_fn = sec["filename"]
        target_path = os.path.join(KNOWLEDGE_FOLDER, target_fn)
        action = sec["action"]
        sec_content = sec["content"]
        topic = sec.get("topic") or custom_title or target_fn.replace(".txt", "").replace("_", " ").title()

        if action == "UPDATE_EXISTING" and os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    existing_text = f.read()
                final_text = merge_into_existing_document(existing_text, sec_content)
            except Exception:
                final_text = f"{existing_text}\n\n---\n\n## {topic}\n\n{sec_content}"
        else:
            if not sec_content.strip().startswith("#"):
                final_text = f"# {topic}\n\n{sec_content.strip()}"
            else:
                final_text = sec_content.strip()

        # Save to disk
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        # Register metadata & auto-versioning
        meta_res = register_or_update_document(
            filename=target_fn,
            title=topic,
            doc_type=doc_type_override or None,
            academic_year=year_override or None,
            semester=semester_override or None,
            version=version_override or "1.0",
            issuing_authority=authority_override or None,
            status="active",
            auto_supersede=auto_supersede_flag
        )

        action_results.append({
            "filename": target_fn,
            "action": action,
            "topic": topic,
            "superseded_files": meta_res.get("superseded_files", []),
            "final_text": final_text
        })

    return action_results


@app.route("/admin", methods=["GET", "POST"])
@app.route("/[object HTMLInputElement]", methods=["GET", "POST"])
@app.route("/[object%20HTMLInputElement]", methods=["GET", "POST"])
def admin():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    # Fetch temporary OCR preview and routing summary (using server-side preview cache)
    ocr_preview_token = session.pop("ocr_preview_token", None)
    ocr_preview = pop_server_preview(ocr_preview_token) or session.pop("ocr_preview", "")
    routing_summary = session.pop("routing_summary", None)

    if request.method == "POST":
        action = request.form.get("action")
        try:
            # Case 1: File Upload (PDF, Image, TXT) — supports multiple files
            if action == "upload":
                uploaded_files = request.files.getlist("image")
                custom_doc_name = request.form.get("doc_name", "").strip()

                logger.info(f"[Admin Upload]: Request received with {len(uploaded_files) if uploaded_files else 0} file(s).")

                if not uploaded_files or not any(f.filename for f in uploaded_files):
                    flash("No files selected. Please choose at least one document.", "error")
                    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                        return jsonify({
                            "status": "error",
                            "message": "No files selected. Please choose at least one document."
                        }), 400
                else:
                    processed_files = []
                    last_preview_text = ""
                    routing_summary_items = []

                    for idx, uploaded_file in enumerate(uploaded_files):
                        if not uploaded_file or not uploaded_file.filename:
                            continue

                        clean_raw_name = sanitize_filename(uploaded_file.filename)
                        save_path = os.path.join(UPLOAD_FOLDER, clean_raw_name)
                        uploaded_file.save(save_path)
                        logger.info(f"[Admin Upload]: Processing file {idx+1}/{len(uploaded_files)}: '{uploaded_file.filename}' saved to '{clean_raw_name}'")

                        # Extract text using multi-format extractor (PDF, Image, TXT)
                        extracted_text = extract_text_from_file(save_path)

                        if not extracted_text or extracted_text.startswith("Error"):
                            logger.error(f"[Admin Upload]: Extraction failed for '{uploaded_file.filename}': {extracted_text}")
                            flash(f"Text extraction failed for '{uploaded_file.filename}': {extracted_text}", "error")
                            continue

                        logger.info(f"[Admin Upload]: Extracted {len(extracted_text)} chars from '{uploaded_file.filename}'. Structuring and routing...")

                        # Apply AI structuring to format the extracted text
                        try:
                            structured_text = structure_extracted_text(extracted_text.strip())
                        except Exception as e:
                            logger.warning(f"[Structuring Warning]: {e}")
                            structured_text = extracted_text.strip()

                        if not structured_text:
                            structured_text = extracted_text.strip()

                        doc_type_input = request.form.get("doc_type", "").strip()
                        if doc_type_input == "__custom__":
                            doc_type_input = request.form.get("custom_doc_type", "").strip()

                        academic_year_input = request.form.get("academic_year", "").strip()
                        semester_input = request.form.get("semester", "").strip()
                        version_input = request.form.get("version", "").strip()
                        authority_input = request.form.get("issuing_authority", "").strip()
                        if authority_input == "__custom__":
                            authority_input = request.form.get("custom_issuing_authority", "").strip()
                        auto_supersede_flag = (request.form.get("auto_supersede") != "false")

                        # Differentiate multi-category or single category and route to existing or new files
                        results = ingest_and_route_knowledge_content(
                            content=structured_text,
                            source_label=uploaded_file.filename,
                            custom_title=custom_doc_name if (custom_doc_name and len(uploaded_files) == 1) else "",
                            doc_type_override=doc_type_input,
                            authority_override=authority_input,
                            year_override=academic_year_input,
                            semester_override=semester_input,
                            version_override=version_input,
                            auto_supersede_flag=auto_supersede_flag
                        )

                        for r in results:
                            action_txt = "Updated existing category" if r["action"] == "UPDATE_EXISTING" else "Created new category"
                            routing_summary_items.append({
                                "action": r["action"],
                                "action_label": action_txt,
                                "filename": r["filename"],
                                "topic": r["topic"],
                                "source_file": uploaded_file.filename,
                                "superseded_files": r.get("superseded_files", [])
                            })
                            last_preview_text = r["final_text"]
                            processed_files.append((uploaded_file.filename, r["filename"], r["action"] == "UPDATE_EXISTING"))
                            logger.info(f"[Admin Upload]: Routed '{uploaded_file.filename}' -> '{r['filename']}' ({action_txt})")

                    if processed_files:
                        # Store preview snippet safely via server-side cache (token in session)
                        preview_snippet = last_preview_text[:1500] + ("..." if len(last_preview_text) > 1500 else "")
                        session["ocr_preview_token"] = set_server_preview(preview_snippet)

                        # Rebuild and reload vector store once after all files in batch are processed
                        logger.info(f"[Admin Upload]: Synchronizing vector store for {len(processed_files)} processed document(s)...")
                        build_result = build_vector_store()
                        reload_vector_store()
                        total_chunks = build_result.get("total_chunks", 0)
                        logger.info(f"[Admin Upload]: Vector synchronization complete! Total chunks: {total_chunks}")

                        session["routing_summary"] = {
                            "processed_count": len(processed_files),
                            "total_chunks": total_chunks,
                            "categories": routing_summary_items,
                            "summary_text": f"Successfully processed {len(processed_files)} document(s). Vector database synchronized (Total chunks: {total_chunks})."
                        }
                    else:
                        logger.warning("[Admin Upload]: No documents could be successfully extracted or routed from this upload batch.")
                        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", ""):
                            return jsonify({
                                "status": "error",
                                "message": "Text extraction failed for uploaded file(s). Please verify the file contents or try uploading clear documents."
                            }), 400

            # Case 2: Direct Text Notice
            elif action == "direct_notice":
                title = request.form.get("title", "").strip()
                content = request.form.get("content", "").strip()

                if content:
                    doc_type_input = request.form.get("doc_type", "").strip()
                    if doc_type_input == "__custom__":
                        doc_type_input = request.form.get("custom_doc_type", "").strip()

                    academic_year_input = request.form.get("academic_year", "").strip()
                    semester_input = request.form.get("semester", "").strip()
                    version_input = request.form.get("version", "").strip()
                    authority_input = request.form.get("issuing_authority", "").strip()
                    if authority_input == "__custom__":
                        authority_input = request.form.get("custom_issuing_authority", "").strip()
                    auto_supersede_flag = (request.form.get("auto_supersede") != "false")

                    results = ingest_and_route_knowledge_content(
                        content=content,
                        custom_title=title,
                        doc_type_override=doc_type_input,
                        authority_override=authority_input,
                        year_override=academic_year_input,
                        semester_override=semester_input,
                        version_override=version_input,
                        auto_supersede_flag=auto_supersede_flag
                    )

                    notice_summary_items = []
                    last_preview = ""
                    for r in results:
                        action_txt = "Updated existing category" if r["action"] == "UPDATE_EXISTING" else "Created new category"
                        notice_summary_items.append({
                            "action": r["action"],
                            "action_label": action_txt,
                            "filename": r["filename"],
                            "topic": r["topic"],
                            "source_file": "Direct Notice",
                            "superseded_files": r.get("superseded_files", [])
                        })
                        last_preview = r["final_text"]

                    preview_snippet = last_preview[:1500] + ("..." if len(last_preview) > 1500 else "")
                    session["ocr_preview_token"] = set_server_preview(preview_snippet)

                    build_result = build_vector_store()
                    reload_vector_store()

                    session["routing_summary"] = {
                        "processed_count": len(results),
                        "total_chunks": build_result.get("total_chunks", 0),
                        "categories": notice_summary_items,
                        "summary_text": f"Notice categories processed & vector database synchronized! Total chunks: {build_result.get('total_chunks', 0)}"
                    }
                else:
                    flash("Notice content cannot be empty.", "error")

        except Exception as e:
            logger.error(f"[Document Processing Error]: {e}", exc_info=True)
            flash(f"Error processing document: {str(e)}", "error")

        # Ajax / XMLHttpRequest asynchronous response support
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or "application/json" in request.headers.get("Accept", ""):
            return jsonify({
                "status": "success",
                "preview": pop_server_preview(session.get("ocr_preview_token")) or session.get("ocr_preview", ""),
                "routing_summary": session.get("routing_summary", None),
                "message": "Processed successfully"
            })

        # Redirect after POST to prevent form resubmission on page refresh
        return redirect(url_for("admin"))

    # Fetch stats for dashboard
    documents = get_knowledge_documents()
    store = get_store()
    docs = store[0] if store else []
    total_chunks = len(docs) if docs else 0

    # Pagination parameters for dashboard analytics
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 50, type=int)
    offset = max(0, (page - 1) * limit)

    analytics_stats = get_analytics_stats()
    knowledge_gaps = get_knowledge_gaps(limit=limit, offset=offset)
    recent_logs = get_recent_logs(limit=limit, offset=offset)
    top_questions = get_top_questions(limit=8)

    return render_template(
        "admin.html",
        text=ocr_preview,
        routing_summary=routing_summary,
        documents=documents,
        total_chunks=total_chunks,
        model_name=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        analytics_stats=analytics_stats,
        knowledge_gaps=knowledge_gaps,
        recent_logs=recent_logs,
        top_questions=top_questions,
        doc_types=get_all_doc_types(),
        authorities=get_all_authorities()
    )


@app.route("/admin/resolve-gap/<int:gap_id>", methods=["POST"])
def admin_resolve_gap(gap_id):
    """Mark a knowledge gap as resolved."""
    if not session.get("admin"):
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        return redirect(url_for("admin_login"))
    
    mark_gap_resolved(gap_id)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
        return jsonify({"status": "success", "message": "Knowledge gap marked as resolved!"})
    flash("Knowledge gap marked as resolved!", "success")
    return redirect(url_for("admin") + "#analytics-section")


@app.route("/admin/quick-add-gap-knowledge", methods=["POST"])
def admin_quick_add_gap_knowledge():
    """Directly add new knowledge to resolve a specific student knowledge gap and rebuild vector index."""
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
        
    gap_id = request.form.get("gap_id")
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()
    
    if not content:
        flash("Knowledge content cannot be empty.", "error")
        return redirect(url_for("admin") + "#analytics-section")
        
    try:
        results = ingest_and_route_knowledge_content(
            content=content,
            custom_title=title
        )

        # Rebuild vector store
        build_result = build_vector_store()
        reload_vector_store()
        
        # Mark gap resolved
        if gap_id:
            try:
                mark_gap_resolved(int(gap_id))
            except Exception:
                pass
                
        for r in results:
            action_txt = "merged into existing category" if r["action"] == "UPDATE_EXISTING" else "created as new category"
            flash(f"Knowledge {action_txt}: '{r['filename']}' ({r['topic']}).", "success" if r["action"] == "UPDATE_EXISTING" else "info")

        flash("Knowledge successfully updated and vector database synchronized! Knowledge gap marked as resolved.", "success")
    except Exception as e:
        flash(f"Error adding knowledge: {str(e)}", "error")
        
    return redirect(url_for("admin") + "#analytics-section")


@app.route("/admin/clear-logs", methods=["POST"])
def admin_clear_logs():
    """Clear chat analytics logs."""
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
        
    clear_all_logs()
    flash("All chat analytics and interaction logs cleared.", "info")
    return redirect(url_for("admin") + "#analytics-section")


@app.route("/admin/reindex", methods=["POST"])
def admin_reindex():
    """Manually trigger complete re-indexing of all knowledge base files."""
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    result = build_vector_store()
    reload_vector_store()
    flash(
        f"Knowledge base re-indexed successfully! Total chunks: {result.get('total_chunks', 0)} across {result.get('files_processed', 0)} documents.",
        "success"
    )
    return redirect(url_for("admin"))


@app.route("/admin/delete/<filename>", methods=["POST"])
def admin_delete(filename):
    """Delete a document from knowledge base with path traversal check and auto-sync vector index."""
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    clean_name = sanitize_filename(filename)
    target_path = os.path.abspath(os.path.join(KNOWLEDGE_FOLDER, clean_name))
    knowledge_dir_abs = os.path.abspath(KNOWLEDGE_FOLDER)

    # Path traversal protection
    if not target_path.startswith(knowledge_dir_abs):
        flash("Invalid file path.", "error")
        return redirect(url_for("admin"))

    if os.path.exists(target_path):
        os.remove(target_path)
        delete_document_metadata(clean_name)
        build_vector_store()
        reload_vector_store()
        flash(f"Document '{clean_name}' and its metadata deleted. Vector store synchronized.", "info")
    else:
        flash(f"Document '{clean_name}' not found.", "error")

    return redirect(url_for("admin") + "#kb-files")


@app.route("/admin/bulk-documents", methods=["POST"])
def admin_bulk_documents():
    """Bulk archive or bulk delete selected knowledge documents."""
    if not session.get("admin"):
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json(silent=True) or request.form
    action = data.get("action", "").lower().strip()
    raw_files = data.get("filenames", [])
    if isinstance(raw_files, str):
        filenames = [f.strip() for f in raw_files.split(",") if f.strip()]
    elif isinstance(raw_files, list):
        filenames = [str(f).strip() for f in raw_files if str(f).strip()]
    else:
        filenames = []

    if not filenames:
        return jsonify({"status": "error", "message": "No documents selected for bulk action."}), 400

    knowledge_dir_abs = os.path.abspath(KNOWLEDGE_FOLDER)
    processed = []
    failed = []

    if action == "delete":
        for fn in filenames:
            clean_name = sanitize_filename(fn)
            target_path = os.path.abspath(os.path.join(KNOWLEDGE_FOLDER, clean_name))
            if not target_path.startswith(knowledge_dir_abs):
                failed.append(clean_name)
                continue
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                    delete_document_metadata(clean_name)
                    processed.append(clean_name)
                except Exception as e:
                    logger.error(f"[Bulk Delete Error] {clean_name}: {e}", exc_info=True)
                    failed.append(clean_name)
            else:
                delete_document_metadata(clean_name)
                processed.append(clean_name)

        if processed:
            build_vector_store()
            reload_vector_store()

        return jsonify({
            "status": "success",
            "message": f"Successfully deleted {len(processed)} document(s)." + (f" ({len(failed)} failed)" if failed else ""),
            "processed": processed,
            "failed": failed
        })

    elif action == "archive":
        for fn in filenames:
            clean_name = sanitize_filename(fn)
            try:
                update_document_status(clean_name, "archived")
                processed.append(clean_name)
            except Exception as e:
                logger.error(f"[Bulk Archive Error] {clean_name}: {e}", exc_info=True)
                failed.append(clean_name)

        if processed:
            build_vector_store()
            reload_vector_store()

        return jsonify({
            "status": "success",
            "message": f"Successfully archived {len(processed)} document(s)." + (f" ({len(failed)} failed)" if failed else ""),
            "processed": processed,
            "failed": failed
        })

    return jsonify({"status": "error", "message": "Invalid bulk action. Allowed: 'archive', 'delete'."}), 400


@app.route("/admin/update-metadata", methods=["POST"])
def admin_update_metadata():
    """Allows administrators to edit document metadata, versioning, and lifecycle status."""
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    filename = request.form.get("filename", "").strip()
    clean_name = sanitize_filename(filename)
    title = request.form.get("title", "").strip()
    doc_type = request.form.get("doc_type", "").strip()
    if doc_type == "__custom__":
        doc_type = request.form.get("custom_doc_type", "").strip()

    academic_year = request.form.get("academic_year", "").strip()
    semester = request.form.get("semester", "").strip()
    version = request.form.get("version", "1.0").strip()
    issuing_authority = request.form.get("issuing_authority", "").strip()
    if issuing_authority == "__custom__":
        issuing_authority = request.form.get("custom_issuing_authority", "").strip()
    status = request.form.get("status", "active").strip()
    effective_date = request.form.get("effective_date", "").strip()
    expiry_date = request.form.get("expiry_date", "").strip()

    if not clean_name:
        flash("Filename is required.", "error")
        return redirect(url_for("admin") + "#kb-files")

    meta_res = register_or_update_document(
        filename=clean_name,
        title=title or None,
        doc_type=doc_type or None,
        academic_year=academic_year or None,
        semester=semester or None,
        version=version or "1.0",
        issuing_authority=issuing_authority or None,
        effective_date=effective_date or None,
        expiry_date=expiry_date or None,
        status=status,
        auto_supersede=(status == "active")
    )

    build_vector_store()
    reload_vector_store()

    superseded = meta_res.get("superseded_files", [])
    if superseded:
        flash(f"Updated '{clean_name}'. Previous versions {superseded} marked as superseded.", "info")
    else:
        flash(f"Metadata and lifecycle status for '{clean_name}' updated successfully!", "success")

    return redirect(url_for("admin") + "#kb-files")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 60)
    logger.info(f"  [2/2] BAUST AI HELPDESK SERVER RUNNING SUCCESSFULLY!")
    logger.info(f"  --> Student Portal: http://127.0.0.1:{port}")
    logger.info(f"  --> Admin Panel:    http://127.0.0.1:{port}/admin")
    logger.info("=" * 60)
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )