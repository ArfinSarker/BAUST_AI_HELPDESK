import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from datetime import datetime
import markdown
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, session,
    redirect, url_for, jsonify, flash
)
from werkzeug.utils import secure_filename

from ocr.ocr_api import extract_text_from_file
from embedding.create_vector import build_vector_store
from retrieval.rag_search import (
    search_information, reload_vector_store, get_store
)
from llm.chatbot import (
    generate_answer, structure_extracted_text,
    find_matching_knowledge_file, merge_into_existing_document
)

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
KNOWLEDGE_FOLDER = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_FOLDER = os.path.join(BASE_DIR, "vector_store")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or os.getenv("FLASK_SECRET_KEY") or "baust_ai_secret_key_2026"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB max file upload

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWLEDGE_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "baust123")


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


def process_learned_facts(raw_answer):
    """
    Check if the AI decided to learn a new fact from the user.
    If [LEARNED_FACT: ...] is found, extract it, save it, rebuild the index safely,
    and return the cleaned answer.
    """
    learned_fact_match = re.search(r'\[LEARNED_FACT:\s*(.*?)\]', raw_answer, re.DOTALL | re.IGNORECASE)
    
    if learned_fact_match:
        new_fact = learned_fact_match.group(1).strip()
        learned_path = os.path.join(KNOWLEDGE_FOLDER, "User_Learned_Facts.txt")
        
        try:
            # Append the new fact
            with open(learned_path, "a", encoding="utf-8") as f:
                f.write(f"\n- {new_fact} (Learned from user conversation on {datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
            
            # Rebuild vector store quietly
            build_vector_store()
            reload_vector_store()
        except Exception as e:
            print(f"[Learned Fact Error]: {e}")
        
        # Remove the tag from the final answer shown to the user
        cleaned_answer = re.sub(r'\[LEARNED_FACT:\s*.*?\]', '', raw_answer, flags=re.DOTALL | re.IGNORECASE).strip()
        return cleaned_answer
    
    return raw_answer


def get_knowledge_documents():
    """Retrieve detailed stats for all documents in the knowledge base."""
    documents = []
    if os.path.exists(KNOWLEDGE_FOLDER):
        for f in sorted(os.listdir(KNOWLEDGE_FOLDER)):
            file_path = os.path.join(KNOWLEDGE_FOLDER, f)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                documents.append({
                    "name": f,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
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
    If no custom name is given, smartly detects title from content or uploaded filename.
    """
    stem = ""
    if custom_name and custom_name.strip():
        stem = custom_name.strip()
    else:
        # Check if uploaded filename is informative or generic (e.g. Screenshot_..., scan01, img123)
        generic_prefixes = ["scan", "img", "image", "doc", "document", "file", "upload", "untitled", "screenshot", "batch"]
        is_generic = True

        if uploaded_filename and uploaded_filename.strip():
            raw_base = os.path.splitext(uploaded_filename.strip())[0]
            clean_base = re.sub(r'[^a-zA-Z0-9]', '', raw_base).lower()
            is_generic = any(clean_base.startswith(p) for p in generic_prefixes) or len(clean_base) <= 3
            if not is_generic:
                stem = raw_base

        # If generic or missing, intelligently extract title from text
        if is_generic or not stem:
            extracted_title = extract_smart_title_from_text(extracted_text)
            if extracted_title:
                stem = extracted_title
            elif uploaded_filename:
                stem = os.path.splitext(uploaded_filename.strip())[0]

    if not stem:
        stem = f"notice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Convert to standard snake_case (lowercase with underscores)
    clean = re.sub(r'[^a-zA-Z0-9]+', '_', stem)
    clean = re.sub(r'_+', '_', clean)
    clean = clean.lower().strip('_')
    if clean.endswith('.txt'):
        clean = clean[:-4]

    return clean or "document"


# =====================================================
# USER CHATBOT (HTML & REST API)
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
            
            # Extract and process any learned facts
            raw_answer = process_learned_facts(raw_answer)

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
def api_chat():
    """High-speed asynchronous JSON API endpoint for real-time conversation."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    history = data.get("history", [])

    if not question:
        return jsonify({
            "status": "error",
            "message": "Question cannot be empty."
        }), 400

    try:
        # Retrieve context from FAISS
        context = search_information(question)
        
        # Generate model response
        raw_answer = generate_answer(context, question, history=history)
        
        # Extract and process any learned facts
        raw_answer = process_learned_facts(raw_answer)
        
        # Render markdown to HTML for rich formatting
        html_answer = markdown.markdown(
            raw_answer,
            extensions=["extra", "nl2br"]
        )

        return jsonify({
            "status": "success",
            "question": question,
            "raw_answer": raw_answer,
            "html_answer": html_answer
        })

    except Exception as e:
        print("[API Chat Error]:", e)
        return jsonify({
            "status": "error",
            "message": "An error occurred while generating the answer. Please try again."
        }), 500


# =====================================================
# ADMIN AUTHENTICATION
# =====================================================

@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username.lower() == ADMIN_USER.lower() and password == ADMIN_PASSWORD:
            session["admin"] = True
            session.permanent = True
            flash("Welcome back, Administrator!", "success")
            return redirect(url_for("admin"))
        else:
            error = "Invalid username or password. Please try again."

    return render_template("admin_login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("admin", None)
    flash("You have been successfully logged out.", "info")
    return redirect(url_for("home"))


# =====================================================
# ADMIN PANEL & KNOWLEDGE MANAGEMENT
# =====================================================

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    # Fetch temporary OCR preview from session if it exists
    ocr_preview = session.pop("ocr_preview", "")

    if request.method == "POST":
        action = request.form.get("action")

        # Case 1: File Upload (PDF, Image, TXT) — supports multiple files
        if action == "upload":
            uploaded_files = request.files.getlist("image")
            custom_doc_name = request.form.get("doc_name", "").strip()

            if not uploaded_files or not any(f.filename for f in uploaded_files):
                flash("No files selected. Please choose at least one document.", "error")
            else:
                processed_files = []
                last_preview_text = ""

                # Process each uploaded document independently and place it into its appropriate topic
                for uploaded_file in uploaded_files:
                    if not uploaded_file or not uploaded_file.filename:
                        continue

                    clean_raw_name = sanitize_filename(uploaded_file.filename)
                    save_path = os.path.join(UPLOAD_FOLDER, clean_raw_name)
                    uploaded_file.save(save_path)

                    # Extract text using multi-format extractor (PDF, Image, TXT)
                    extracted_text = extract_text_from_file(save_path)

                    if not extracted_text or extracted_text.startswith("Error"):
                        flash(f"Text extraction failed for '{uploaded_file.filename}': {extracted_text}", "error")
                        continue

                    # Apply AI structuring to format the extracted text
                    structured_text = structure_extracted_text(extracted_text.strip())
                    if not structured_text:
                        structured_text = extracted_text.strip()

                    merged_into_existing = False
                    target_filename = None

                    # Check which knowledge base file this specific document belongs to
                    matched_file, suggested_stem = find_matching_knowledge_file(structured_text, KNOWLEDGE_FOLDER)

                    if matched_file:
                        target_filename = matched_file
                        merged_into_existing = True
                    else:
                        # If user gave a custom title and only 1 file was uploaded, use custom title
                        if custom_doc_name and len(uploaded_files) == 1:
                            base_stem = format_document_stem(custom_name=custom_doc_name, extracted_text=structured_text)
                            target_filename = f"{base_stem}.txt"
                        else:
                            file_stem = os.path.splitext(clean_raw_name)[0]
                            stem_to_use = suggested_stem if (suggested_stem and suggested_stem != "document") else file_stem
                            base_stem = format_document_stem(custom_name=stem_to_use, extracted_text=structured_text)
                            target_filename = f"{base_stem}.txt"

                    target_txt_path = os.path.join(KNOWLEDGE_FOLDER, target_filename)

                    if merged_into_existing and os.path.exists(target_txt_path):
                        try:
                            with open(target_txt_path, "r", encoding="utf-8") as f:
                                existing_content = f.read()
                            final_text = merge_into_existing_document(existing_content, structured_text)
                        except Exception:
                            final_text = structured_text
                    else:
                        final_text = structured_text

                    with open(target_txt_path, "w", encoding="utf-8") as f:
                        f.write(final_text)

                    last_preview_text = final_text
                    processed_files.append((uploaded_file.filename, target_filename, merged_into_existing))

                if processed_files:
                    # Store preview of the last structured file in session
                    session["ocr_preview"] = last_preview_text[:1500] + ("..." if len(last_preview_text) > 1500 else "")

                    # Rebuild and reload vector store once after all files in batch are processed
                    build_result = build_vector_store()
                    reload_vector_store()

                    for orig_name, tgt_name, was_merged in processed_files:
                        if was_merged:
                            flash(f"'{orig_name}' automatically identified and merged into '{tgt_name}'", "success")
                        else:
                            flash(f"'{orig_name}' categorized and saved as '{tgt_name}'", "success")

                    flash(f"Successfully processed {len(processed_files)} document(s). Vector database synchronized (Total chunks: {build_result.get('total_chunks', 0)}).", "info")

        # Case 2: Direct Text Notice
        elif action == "direct_notice":
            title = request.form.get("title", "").strip()
            content = request.form.get("content", "").strip()

            if content:
                # Apply AI structuring to format and categorize the pasted content
                structured_content = structure_extracted_text(content)

                merged_into_existing = False
                target_filename = None

                # Step 1: Always check first if the content belongs to an existing knowledge base file
                matched_file, suggested_stem = find_matching_knowledge_file(structured_content, KNOWLEDGE_FOLDER)
                
                if matched_file:
                    # Existing file matched -> Merge into that file regardless of whether user typed a title
                    target_filename = matched_file
                    merged_into_existing = True
                else:
                    # No existing file matched -> Create a new file
                    # If the user supplied a custom title, use it (in clean snake_case); otherwise use suggested_stem
                    if title:
                        clean_stem = format_document_stem(custom_name=title, extracted_text=structured_content)
                        target_filename = f"{clean_stem}.txt"
                    else:
                        target_filename = f"{suggested_stem}.txt"

                target_txt_path = os.path.join(KNOWLEDGE_FOLDER, target_filename)

                if merged_into_existing and os.path.exists(target_txt_path):
                    try:
                        with open(target_txt_path, "r", encoding="utf-8") as f:
                            existing_content = f.read()
                        final_text = merge_into_existing_document(existing_content, structured_content)
                    except Exception:
                        final_text = structured_content
                else:
                    final_text = structured_content

                with open(target_txt_path, "w", encoding="utf-8") as f:
                    f.write(final_text)

                # Store preview in session
                session["ocr_preview"] = final_text[:1500] + ("..." if len(final_text) > 1500 else "")

                build_result = build_vector_store()
                reload_vector_store()

                if merged_into_existing:
                    flash(
                        f"Notice automatically identified as related to '{target_filename}' and merged into existing document! "
                        f"Total chunks: {build_result.get('total_chunks', 0)}",
                        "success"
                    )
                else:
                    flash(
                        f"Notice saved as new document '{target_filename}' and indexed into AI memory! "
                        f"Total chunks: {build_result.get('total_chunks', 0)}",
                        "success"
                    )
            else:
                flash("Notice content cannot be empty.", "error")

        # Redirect after POST to prevent form resubmission on page refresh
        return redirect(url_for("admin"))

    # Fetch stats for dashboard
    documents = get_knowledge_documents()
    store = get_store()
    docs = store[0] if store else []
    total_chunks = len(docs) if docs else 0

    return render_template(
        "admin.html",
        text=ocr_preview,
        documents=documents,
        total_chunks=total_chunks,
        model_name=os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    )


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
        build_vector_store()
        reload_vector_store()
        flash(f"Document '{clean_name}' deleted and vector store updated.", "info")
    else:
        flash(f"Document '{clean_name}' not found.", "error")

    return redirect(url_for("admin"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )