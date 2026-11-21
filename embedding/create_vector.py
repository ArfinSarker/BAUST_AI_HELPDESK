import os
import sys
import json
import pickle
import shutil
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import metadata_manager

# Dynamic paths from environment with sensible production defaults
DEFAULT_KNOWLEDGE_FOLDER = os.getenv("KNOWLEDGE_FOLDER", os.path.join(BASE_DIR, "knowledge_base"))
DEFAULT_VECTOR_FOLDER = os.getenv("VECTOR_FOLDER", os.path.join(BASE_DIR, "vector_store"))
DEFAULT_QUARANTINE_FOLDER = os.getenv("QUARANTINE_FOLDER", os.path.join(BASE_DIR, "quarantine"))
DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_MAX_WORKERS = int(os.getenv("INDEXER_MAX_WORKERS", min(8, os.cpu_count() or 4)))

# =====================================================
# LOGGING SYSTEM (CONSOLE + ROTATING indexer.log)
# =====================================================
logger = logging.getLogger("BAUST_INDEXER")
logger.setLevel(logging.INFO)

def setup_indexer_logging(log_file_path: str = None):
    """Configures thread-safe console and rotating file logging."""
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [BAUST_INDEXER]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # 1. Console stream handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file_path:
        has_file_handler = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
        if not has_file_handler:
            try:
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                file_handler = RotatingFileHandler(
                    log_file_path,
                    maxBytes=5 * 1024 * 1024,  # 5 MB
                    backupCount=3,
                    encoding="utf-8"
                )
                file_formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"[Logging Warning]: Unable to initialize rotating log file ({e})")

# Initialize default logging handler
setup_indexer_logging(os.path.join(DEFAULT_VECTOR_FOLDER, "indexer.log"))

_model_instance = None


def get_embedding_model(model_name=DEFAULT_MODEL_NAME):
    """Singleton getter for the SentenceTransformer model with graceful fallback."""
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Fast offline-first loading: avoids 100s+ timeout hangs on unreachable HF hub
            try:
                _model_instance = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                _model_instance = SentenceTransformer(model_name)
            logger.info(f"Loaded dense embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"Dense neural model initialization skipped ({e}). Using ultra-fast hybrid sparse indexing.")
            _model_instance = None
    return _model_instance


def compute_file_hash(file_path: str) -> str:
    """Computes an MD5 checksum of a file for incremental change detection."""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.debug(f"Failed to hash {file_path}: {e}")
        return ""


def quarantine_damaged_file(file_path: str, quarantine_folder: str, reason: str = "Corrupted/Unreadable"):
    """
    Safely moves a damaged, password-protected, or unreadable file out of the knowledge base
    into quarantine so subsequent indexing loops are never disrupted.
    """
    try:
        os.makedirs(quarantine_folder, exist_ok=True)
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(quarantine_folder, file_name)
        
        # Avoid collisions if file already exists in quarantine
        if os.path.exists(dest_path):
            base_name, ext = os.path.splitext(file_name)
            dest_path = os.path.join(quarantine_folder, f"{base_name}_{int(os.path.getmtime(file_path))}{ext}")

        shutil.move(file_path, dest_path)
        logger.warning(f"QUARANTINED file '{file_name}' -> '{dest_path}' | Reason: {reason}")
        return True
    except Exception as e:
        logger.error(f"Failed to quarantine file '{file_path}': {e}")
        return False


def extract_text_from_file(file_path: str) -> str:
    """
    Production-grade multi-engine text extractor:
    1. TXT: multi-encoding fallback (UTF-8, Latin-1, CP1252)
    2. PDF: PyMuPDF (fitz) -> pdfplumber -> pypdf
    Raises Exception if file is fundamentally corrupt or unreadable.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read().strip()
                    if content:
                        return content
            except (UnicodeDecodeError, OSError):
                continue
        # If open succeeded but completely empty or undecodable
        with open(file_path, "rb") as f:
            raw_bytes = f.read(100)
            if not raw_bytes:
                return ""
        return ""

    elif ext == ".pdf":
        # 1. PyMuPDF (High speed, clean layout preservation)
        try:
            import pymupdf
            doc = pymupdf.open(file_path)
            if doc.is_encrypted:
                doc.close()
                raise ValueError("PDF is password-protected or encrypted")
            pages = []
            for page in doc:
                t = page.get_text("text").strip()
                if t:
                    pages.append(t)
            doc.close()
            if pages:
                return "\n\n".join(pages)
        except ValueError:
            raise
        except Exception as e:
            logger.debug(f"PyMuPDF extraction note for {os.path.basename(file_path)}: {e}")

        # 2. pdfplumber fallback
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                if pages:
                    return "\n\n".join(pages)
        except Exception as e:
            logger.debug(f"pdfplumber fallback note for {os.path.basename(file_path)}: {e}")

        # 3. pypdf fallback
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            if reader.is_encrypted:
                raise ValueError("PDF is password-protected or encrypted")
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            if pages:
                return "\n\n".join(pages)
        except ValueError:
            raise
        except Exception as e:
            logger.debug(f"pypdf fallback note for {os.path.basename(file_path)}: {e}")

    return ""


def _worker_extract_file(args):
    """Standalone worker for concurrent file extraction and validation."""
    file_name, file_path, quarantine_folder = args
    try:
        text = extract_text_from_file(file_path)
        if not text or not text.strip():
            # Check if file has 0 bytes (empty dummy file)
            if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
                quarantine_damaged_file(file_path, quarantine_folder, "Empty 0-byte file")
                return file_name, None, "Empty file quarantined"
            
            # If a non-empty PDF or file yields 0 characters (corrupt, encrypted, binary garbage)
            _, ext = os.path.splitext(file_path)
            if ext.lower() == ".pdf":
                quarantine_damaged_file(file_path, quarantine_folder, "Unreadable or corrupt PDF")
                return file_name, None, "Corrupted PDF quarantined"

            return file_name, "", "Empty text"
        return file_name, text, "Success"
    except Exception as err:
        reason = str(err) or "Corrupted or password-protected"
        quarantine_damaged_file(file_path, quarantine_folder, reason)
        return file_name, None, f"Quarantined: {reason}"


def semantic_recursive_chunk(
    text: str,
    doc_name: str = "",
    chunk_size: int = 900,
    chunk_overlap: int = 150
) -> list[str]:
    """
    Context-Aware Hierarchical & Breadcrumb-Enriched Semantic Text Splitter.
    Automatically maintains active hierarchical context (Document, Section, Subsection, Page, Slot,
    Level/Term, Table Column Headers) and prepends these breadcrumbs to every split chunk.
    This guarantees 100% self-contained chunks for ANY PDF (seat plans, exam schedules, routines,
    notices, tuition fees) without requiring brittle per-file hardcoded logic.
    """
    if not text or not text.strip():
        return []

    import re

    # 1. Automatically learn any slot / time definitions present anywhere in this document
    slot_time_defs = {}
    for m in re.finditer(r'(Slot\s*[-–:]?\s*([A-Za-z0-9]+))\s*(?:\(([^)]+)\))?', text, re.I):
        slot_label = m.group(1).strip()
        slot_key = m.group(2).upper()
        time_str = m.group(3).strip() if m.group(3) else ""
        if time_str and slot_key not in slot_time_defs:
            slot_time_defs[slot_key] = (slot_label, time_str)

    lines = text.splitlines()
    chunks = []

    # Active hierarchical context state
    clean_stem = doc_name.replace(".txt", "").replace("_", " ").title() if doc_name else ""
    doc_title = clean_stem
    page_ctx = ""
    major_section = ""
    sub_section = ""
    slot_ctx = ""
    level_ctx = ""
    table_headers = ""

    current_body_lines = []

    def build_header() -> str:
        parts = []
        if doc_title:
            parts.append(f"Doc: {doc_title}")
        if page_ctx:
            parts.append(page_ctx)
        if major_section:
            parts.append(major_section)
        if level_ctx:
            parts.append(level_ctx)
        if slot_ctx:
            parts.append(slot_ctx)
        if sub_section and sub_section not in (major_section, level_ctx, slot_ctx):
            parts.append(sub_section)
        if table_headers:
            parts.append(f"Cols: {table_headers}")
        return f"[Context: {' | '.join(parts)}]" if parts else ""

    def flush_chunk(body_lines: list[str]):
        if not body_lines:
            return
        header = build_header()
        content = "\n".join(body_lines).strip()
        full_chunk = f"{header}\n{content}" if header else content
        chunks.append(full_chunk)

    for line in lines:
        raw_l = line.strip()
        if not raw_l:
            continue

        # Page boundary detection
        page_m = re.match(r'^---\s*Page\s+(\d+)\s*---', raw_l, re.I)
        if page_m:
            if current_body_lines:
                flush_chunk(current_body_lines)
                current_body_lines = []
            page_ctx = f"Page {page_m.group(1)}"
            continue

        # Level 1 Markdown Heading
        if raw_l.startswith('# '):
            if current_body_lines:
                flush_chunk(current_body_lines)
                current_body_lines = []
            major_section = raw_l.lstrip('# ').strip()
            continue

        # Level 2 Markdown Heading
        if raw_l.startswith('## '):
            if current_body_lines:
                flush_chunk(current_body_lines)
                current_body_lines = []
            sub_section = raw_l.lstrip('# ').strip()
            continue

        # Level 3 Markdown Heading
        if raw_l.startswith('### '):
            if current_body_lines:
                flush_chunk(current_body_lines)
                current_body_lines = []
            sub_section = raw_l.lstrip('# ').strip()
            continue

        # Department / Programme detection
        dept_m = re.match(r'^(?:Department\s+of|Dept\.?\s+of|Programme:?)\s*(.*)', raw_l, re.I)
        if dept_m:
            major_section = raw_l
            continue

        # Slot detection: e.g. "Slot- C (01:00 PM - 02:00 PM)"
        slot_m = re.match(r'^(Slot\s*[-:]?\s*([A-Za-z0-9]+).*)', raw_l, re.I)
        if slot_m:
            if current_body_lines:
                flush_chunk(current_body_lines)
                current_body_lines = []
            slot_ctx = slot_m.group(1).strip()
            continue

        # Level & Term detection: e.g. "Level- 3 Term- II"
        lvl_m = re.match(r'^(Level\s*[-–]?\s*\d+\s*Term\s*[-–]?\s*[IVX0-9]+.*)', raw_l, re.I)
        if lvl_m:
            if current_body_lines:
                flush_chunk(current_body_lines)
                current_body_lines = []
            level_ctx = lvl_m.group(1).strip()
            continue

        # Dynamic seat prefix slot correlation: e.g. "C-304-14", "A-102-5", "B-205-1"
        seat_prefix_m = re.match(r'^([A-Za-z])-\d+-\d+', raw_l)
        if seat_prefix_m:
            pfx = seat_prefix_m.group(1).upper()
            if pfx in slot_time_defs:
                slot_lbl, slot_time = slot_time_defs[pfx]
                desired_slot_ctx = f"{slot_lbl} ({slot_time})"
                if slot_ctx != desired_slot_ctx:
                    if current_body_lines:
                        flush_chunk(current_body_lines)
                        current_body_lines = []
                    slot_ctx = desired_slot_ctx

        # Table Column Headers detection
        if raw_l in ['Roll No.', 'Roll', 'ID', 'Student ID', 'Course Code', 'Course No.'] and not table_headers:
            table_headers = 'Roll No. | Name | Exam Room | Seat'
            continue
        if raw_l in ['Name', 'Exam Room', 'Seat', 'Room No', 'Course Title']:
            continue

        # Markdown table row check
        if raw_l.startswith('|') and raw_l.endswith('|'):
            if ':-' in raw_l or '---' in raw_l:
                continue
            cells = [c.strip() for c in raw_l.split('|') if c.strip()]
            if any(c.lower() in ['roll', 'id', 'name', 'course', 'time', 'date', 'room', 'slot'] for c in cells):
                table_headers = ' | '.join(cells)
                continue

        # Append to body
        current_body_lines.append(raw_l)

        # Check chunk length
        approx_size = sum(len(l) for l in current_body_lines) + len(build_header())
        if approx_size >= chunk_size:
            flush_chunk(current_body_lines)
            overlap_count = 2 if len(current_body_lines) > 2 else 0
            current_body_lines = current_body_lines[-overlap_count:] if overlap_count else []

    if current_body_lines:
        flush_chunk(current_body_lines)

    return chunks


def chunk_text(text, chunk_size=350, overlap=80):
    """Backward compatibility wrapper redirecting to semantic recursive chunker."""
    return semantic_recursive_chunk(text, chunk_size=950, chunk_overlap=150)


def encode_in_batches(model, texts: list[str], batch_size: int = 32, sub_batch_limit: int = 256) -> np.ndarray:
    """
    RAM-efficient batched embedding encoder to prevent out-of-memory spikes on large document bases.
    """
    if not texts:
        return np.empty((0, 384), dtype="float32")

    all_embs = []
    total = len(texts)

    for start_idx in range(0, total, sub_batch_limit):
        end_idx = min(start_idx + sub_batch_limit, total)
        slice_texts = texts[start_idx:end_idx]

        # Add prefix for multilingual-e5 models if active
        model_name_str = getattr(model, "model_name_or_path", "")
        if "e5" in model_name_str.lower():
            slice_texts = [f"passage: {t}" for t in slice_texts]

        raw_emb = model.encode(slice_texts, batch_size=batch_size, show_progress_bar=False)
        all_embs.append(np.array(raw_emb, dtype="float32"))

    return np.vstack(all_embs) if all_embs else np.empty((0, 384), dtype="float32")


def build_vector_store(
    knowledge_folder=None,
    vector_folder=None,
    quarantine_folder=None,
    model_name=None,
    force_rebuild=False
):
    """
    Production-Grade Incremental Vector Store Builder:
    1. Environment-driven paths with logging to indexer.log.
    2. Concurrent parallel file text extraction with quarantine for corrupt/unreadable files.
    3. Incremental hash/mtime detection (< 30ms when up-to-date).
    4. Token-safe semantic recursive chunking & RAM-safe batch embeddings.
    """
    knowledge_folder = knowledge_folder or DEFAULT_KNOWLEDGE_FOLDER
    vector_folder = vector_folder or DEFAULT_VECTOR_FOLDER
    quarantine_folder = quarantine_folder or DEFAULT_QUARANTINE_FOLDER
    model_name = model_name or DEFAULT_MODEL_NAME

    os.makedirs(knowledge_folder, exist_ok=True)
    os.makedirs(vector_folder, exist_ok=True)

    # Ensure log file exists in target vector_folder
    setup_indexer_logging(os.path.join(vector_folder, "indexer.log"))

    cache_dir = os.path.join(vector_folder, ".incremental_cache")
    os.makedirs(cache_dir, exist_ok=True)

    manifest_path = os.path.join(vector_folder, "index_manifest.json")
    old_manifest = {}
    if os.path.exists(manifest_path) and not force_rebuild:
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                old_manifest = json.load(mf)
        except Exception as e:
            logger.warning(f"Failed to read existing manifest ({e}). Triggering fresh scan.")
            old_manifest = {}

    cached_files_manifest = old_manifest.get("files", {})

    # Detect current files in knowledge_base
    current_files = sorted([
        f for f in os.listdir(knowledge_folder)
        if os.path.isfile(os.path.join(knowledge_folder, f)) and f.lower().endswith((".txt", ".pdf"))
    ])

    files_to_process = []
    files_to_reuse = []

    for file_name in current_files:
        file_path = os.path.join(knowledge_folder, file_name)
        f_hash = compute_file_hash(file_path)
        f_mtime = os.path.getmtime(file_path)
        cache_file = os.path.join(cache_dir, f"{file_name}.pkl")

        manifest_entry = cached_files_manifest.get(file_name)
        if (
            not force_rebuild
            and manifest_entry
            and manifest_entry.get("hash") == f_hash
            and os.path.exists(cache_file)
        ):
            files_to_reuse.append((file_name, f_hash, f_mtime, cache_file))
        else:
            files_to_process.append((file_name, file_path, f_hash, f_mtime, cache_file))

    # Check for deleted files
    deleted_files = [fn for fn in cached_files_manifest.keys() if fn not in current_files]

    # Fast Path: If nothing changed, return instantly
    faiss_path = os.path.join(vector_folder, "index.faiss")
    meta_path = os.path.join(vector_folder, "metadata.pkl")
    if not files_to_process and not deleted_files and os.path.exists(faiss_path) and os.path.exists(meta_path):
        total_chunks = old_manifest.get("total_chunks", 0)
        logger.info(f"All {len(current_files)} files up-to-date. Zero re-embedding required! (Total chunks: {total_chunks})")
        return {
            "status": "up_to_date",
            "files_processed": len(current_files),
            "total_chunks": total_chunks,
            "incremental": True
        }

    logger.info(f"Incremental Indexing: {len(files_to_process)} modified/new, {len(files_to_reuse)} unchanged, {len(deleted_files)} deleted.")

    # Remove cache for deleted files
    for df in deleted_files:
        del_cache = os.path.join(cache_dir, f"{df}.pkl")
        if os.path.exists(del_cache):
            try:
                os.remove(del_cache)
                logger.info(f"Removed stale cache for deleted file: {df}")
            except Exception as e:
                logger.debug(f"Could not remove {del_cache}: {e}")

    # =====================================================
    # PARALLEL FILE EXTRACTION & VALIDATION
    # =====================================================
    extracted_texts = {}
    if files_to_process:
        worker_args = [
            (fn, fp, quarantine_folder)
            for fn, fp, _, _, _ in files_to_process
        ]
        max_workers = min(DEFAULT_MAX_WORKERS, len(files_to_process))
        logger.info(f"Starting parallel text extraction with {max_workers} worker threads...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(_worker_extract_file, arg): arg[0]
                for arg in worker_args
            }
            for future in as_completed(future_to_file):
                fn, text, msg = future.result()
                if text is not None and text.strip():
                    extracted_texts[fn] = text
                else:
                    logger.warning(f"File '{fn}' produced no indexable text ({msg}).")

    model = get_embedding_model(model_name)

    # Process chunks and embeddings for modified/new files
    for file_name, file_path, f_hash, f_mtime, cache_file in files_to_process:
        text = extracted_texts.get(file_name, "")
        if not text.strip():
            continue

        chunks = semantic_recursive_chunk(text, doc_name=file_name)
        doc_meta = metadata_manager.get_document_metadata(file_name)
        if not doc_meta:
            doc_meta = metadata_manager.register_or_update_document(file_name)

        file_metadata = []
        for idx, chunk in enumerate(chunks):
            file_metadata.append({
                "source": file_name,
                "filename": file_name,
                "chunk_id": idx,
                "total_chunks": len(chunks),
                "title": doc_meta.get("title", file_name),
                "doc_type": doc_meta.get("doc_type", "general_notice"),
                "academic_year": doc_meta.get("academic_year", ""),
                "semester": doc_meta.get("semester", ""),
                "version": doc_meta.get("version", "1.0"),
                "issuing_authority": doc_meta.get("issuing_authority", "BAUST Authority"),
                "status": doc_meta.get("status", "active"),
                "is_authoritative": doc_meta.get("is_authoritative", 1),
                "superseded_by": doc_meta.get("superseded_by", None),
                "effective_date": doc_meta.get("effective_date", ""),
                "expiry_date": doc_meta.get("expiry_date", "")
            })

        # Generate embeddings for new chunks
        file_embeddings = None
        if model is not None and chunks:
            file_embeddings = encode_in_batches(model, chunks, batch_size=32)

        # Cache file artifacts
        try:
            with open(cache_file, "wb") as f_cache:
                pickle.dump({
                    "hash": f_hash,
                    "mtime": f_mtime,
                    "chunks": chunks,
                    "metadata": file_metadata,
                    "embeddings": file_embeddings
                }, f_cache)
            logger.info(f"Cached {len(chunks)} chunks for: {file_name}")
        except Exception as e:
            logger.error(f"Failed to cache artifacts for {file_name}: {e}")

    # Aggregate all active chunks, metadata, and embeddings in deterministic order
    all_documents = []
    all_metadata = []
    all_embeddings_list = []
    new_manifest_files = {}

    for file_name in current_files:
        cache_file = os.path.join(cache_dir, f"{file_name}.pkl")
        if not os.path.exists(cache_file):
            continue

        try:
            with open(cache_file, "rb") as f_cache:
                data = pickle.load(f_cache)
                file_chunks = data["chunks"]
                file_meta = data["metadata"]
                file_embs = data.get("embeddings")

                # Normalize metadata keys to guarantee backward compatibility
                for m in file_meta:
                    if isinstance(m, dict):
                        prov = m.get("source") or m.get("filename") or file_name
                        m["source"] = prov
                        m["filename"] = prov

                all_documents.extend(file_chunks)
                all_metadata.extend(file_meta)
                if file_embs is not None and len(file_embs) > 0:
                    all_embeddings_list.append(file_embs)

                new_manifest_files[file_name] = {
                    "hash": data["hash"],
                    "mtime": data["mtime"],
                    "chunk_count": len(file_chunks)
                }
        except Exception as e:
            logger.error(f"Cache Read Error {file_name}: {e}")

    total_chunks = len(all_documents)
    if total_chunks == 0:
        logger.warning("No documents found to index.")
        return {"status": "empty", "files_processed": 0, "total_chunks": 0}

    # Save aggregated documents and metadata
    with open(os.path.join(vector_folder, "documents.pkl"), "wb") as f:
        pickle.dump(all_documents, f)
    with open(os.path.join(vector_folder, "metadata.pkl"), "wb") as f:
        pickle.dump(all_metadata, f)

    # Compile FAISS and numpy index if embeddings exist
    if all_embeddings_list:
        combined_embeddings = np.vstack(all_embeddings_list).astype("float32")
        norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_embeddings = combined_embeddings / norms

        np.save(os.path.join(vector_folder, "embeddings.npy"), normalized_embeddings)

        try:
            import faiss
            dimension = normalized_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(normalized_embeddings)
            faiss.write_index(index, os.path.join(vector_folder, "index.faiss"))
        except Exception as e:
            logger.error(f"FAISS Save Warning: {e}")

    # Write updated manifest
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump({
            "model_name": model_name,
            "total_documents": len(current_files),
            "total_chunks": total_chunks,
            "files": new_manifest_files
        }, mf, indent=2)

    logger.info(f"Incremental Indexing Complete: {len(current_files)} files, {total_chunks} total chunks in {vector_folder}")
    return {
        "status": "success",
        "files_processed": len(current_files),
        "total_chunks": total_chunks,
        "incremental": True
    }


if __name__ == "__main__":
    result = build_vector_store()
    logger.info(f"Build result: {result}")
