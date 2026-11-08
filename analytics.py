import os
import re
import sqlite3
import threading
import sys
import atexit
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "analytics.db")

# Dedicated mutex ONLY for write operations (INSERT, UPDATE, DELETE, DDL)
_DB_LOCK = threading.Lock()

# Global worker pool for detached, non-blocking chat logging
# Bounded to max_workers=2 to prevent arbitrary thread explosion during peak traffic
_log_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analytics_log_worker")


def _shutdown_log_executor():
    """Cleanly terminates the log worker executor upon application exit."""
    try:
        _log_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


atexit.register(_shutdown_log_executor)


def get_db_connection():
    """Returns a SQLite connection configured with WAL mode for high concurrency."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db():
    """Initialize the analytics database schema."""
    with _DB_LOCK:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                question TEXT NOT NULL,
                answer TEXT,
                context_found INTEGER DEFAULT 1,
                knowledge_gap INTEGER DEFAULT 0,
                feedback TEXT DEFAULT NULL,
                resolved INTEGER DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_logs (timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_gap ON chat_logs (knowledge_gap, resolved);")
        conn.commit()
        conn.close()


# Ensure DB exists at import time
init_db()


def is_greeting_or_chitchat(text: str) -> bool:
    """Detects simple conversational pleasantries that should never be flagged as knowledge gaps."""
    if not text:
        return True
    t = text.strip().lower()
    
    greetings = [
        "hi", "hello", "hey", "salam", "assalamu alaikum", "kemon acho", "kemon achen",
        "how are you", "who are you", "what is your name", "tomar nam", "apnar nam",
        "tumi ke", "apni ke", "thanks", "thank you", "dhonnobad", "bye", "goodbye",
        "allah hafez", "tata", "কেমন আছো", "কেমন আছেন", "ধন্যবাদ", "তোমার নাম কি",
        "তুমি কে", "আপনার নাম কি", "কে তুমি"
    ]
    if any(t == g or t.startswith(g + " ") or t.endswith(" " + g) for g in greetings):
        return True
    return False


def detect_knowledge_gap(question: str, context: str, answer: str) -> bool:
    """
    Evaluates whether a student question represents a knowledge gap in the university database.
    """
    if is_greeting_or_chitchat(question):
        return False

    # 1. No context found in vector / keyword retrieval
    clean_ctx = context.strip() if context else ""
    if not clean_ctx or clean_ctx == "(No specific internal BAUST document found - answer with general helpful AI knowledge)":
        return True

    # 2. Check for explicit fallback or missing-data patterns in the AI output
    ans_clean = (answer or "").lower()
    missing_patterns = [
        r'পা[ওঅ]য়া\s*যা[য়য]নি',             # পাওয়া যায়নি, পাওয়া যায়নি, পাওয়া যায় নি
        r'পা[ওঅ]য়া\s*সম্ভব\s*হ[য়য]নি',     # পাওয়া সম্ভব হয়নি
        r'খুঁজে\s*পা[ওঅ]য়া\s*যা[য়য]নি',     # খুঁজে পাওয়া যায়নি
        r'তথ্য\s*নেই',                     # তথ্য নেই
        r'তথ্য\s*পা[ওঅ]য়া\s*যা[য়য]নি',     # তথ্য পাওয়া যায়নি
        r'নোটিশ\s*নেই',                    # নোটিশ নেই
        r'নোটিশ\s*পা[ওঅ]য়া\s*যা[য়য]নি',    # নোটিশ পাওয়া যায়নি
        r'তারিখ\s*পা[ওঅ]য়া\s*যা[য়য]নি',    # তারিখ পাওয়া যায়নি
        r'তারিখ\s*বা\s*নোটিশ',             # তারিখ বা নোটিশ পাওয়া যায়নি
        r'ডাটাবেসে?\s*নেই',               # ডাটাবেসে নেই
        r'ডাটাবেজে?\s*নেই',               # ডাটাবেজে নেই
        r'রেকর্ডে?\s*নেই',                 # রেকর্ডে নেই, রেকর্ডে নেই
        r'রেকর্ড\s*পা[ওঅ]য়া\s*যা[য়য]নি',   # রেকর্ড পাওয়া যায়নি
        r'কোনো\s*রেকর্ড\s*নেই',             # কোনো রেকর্ড নেই
        r'কোনো\s*তথ্য\s*নেই',             # কোনো তথ্য নেই
        r'সংরক্ষিত\s*নেই',                 # সংরক্ষিত নেই
        r'নির্দিষ্ট\s*কোনো\s*তথ্য',       # নির্দিষ্ট কোনো তথ্য
        r'not\s+found',                    # not found
        r'no\s+information\s+found',       # no information found
        r'not\s+available\s+in',           # not available in
        r'do\s+not\s+have\s+information',  # do not have information
        r'don\'?t\s+have\s+information',   # don't have information
        r'no\s+record',                    # no record
        r'no\s+specific\s+document',       # no specific document
    ]

    for pat in missing_patterns:
        if re.search(pat, ans_clean):
            return True

    return False


def _write_chat_log(question: str, answer: str, context: str, feedback: str = None) -> int:
    """
    Internal worker function executed in the background pool to insert log into SQLite.
    Returns the newly generated primary key (id) for precise feedback referencing.
    """
    try:
        clean_q = (question or "").strip()
        if not clean_q:
            return None

        has_context = 1 if (context and context.strip()) else 0
        knowledge_gap = 1 if detect_knowledge_gap(clean_q, context, answer) else 0

        with _DB_LOCK:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_logs (timestamp, question, answer, context_found, knowledge_gap, feedback, resolved)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                clean_q,
                (answer or "").strip(),
                has_context,
                knowledge_gap,
                feedback
            ))
            log_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return log_id
    except Exception as e:
        print(f"[Analytics Log Error]: {e}")
        return None


def log_chat_async(question: str, answer: str, context: str, feedback: str = None):
    """
    Asynchronously schedules chat logging to a bounded ThreadPoolExecutor worker.
    Returns the Future object, allowing callers to inspect the generated log_id.
    """
    if getattr(sys, "is_finalizing", lambda: False)():
        return None
    try:
        return _log_executor.submit(_write_chat_log, question, answer, context, feedback)
    except RuntimeError:
        return None


def log_chat(question: str, answer: str, context: str, feedback: str = None) -> int:
    """
    Synchronously inserts chat log and immediately returns the newly inserted log_id.
    """
    return _write_chat_log(question, answer, context, feedback)


def update_feedback(question: str = None, feedback: str = "like", log_id: int = None) -> bool:
    """
    Updates user rating (like/dislike).
    
    CRITICAL: Prioritizes exact `log_id` targeting to completely eliminate concurrency race conditions
    when multiple users query the same question simultaneously.
    Provides graceful fallback to exact and fuzzy text matching if log_id is not supplied.
    """
    try:
        clean_q = (question or "").strip()
        gap_flag = 1 if feedback == "dislike" else None

        with _DB_LOCK:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 1. Primary Strategy: Exact Log ID matching
            if log_id is not None:
                try:
                    target_id = int(log_id)
                    if gap_flag is not None:
                        cursor.execute("""
                            UPDATE chat_logs 
                            SET feedback = ?, knowledge_gap = ?
                            WHERE id = ?
                        """, (feedback, gap_flag, target_id))
                    else:
                        cursor.execute("""
                            UPDATE chat_logs 
                            SET feedback = ?
                            WHERE id = ?
                        """, (feedback, target_id))

                    if cursor.rowcount > 0:
                        conn.commit()
                        conn.close()
                        return True
                except (ValueError, TypeError):
                    pass

            # 2. Secondary Fallback: Exact Question Match (most recent)
            if clean_q:
                if gap_flag is not None:
                    cursor.execute("""
                        UPDATE chat_logs 
                        SET feedback = ?, knowledge_gap = ?
                        WHERE id = (
                            SELECT id FROM chat_logs 
                            WHERE question = ? 
                            ORDER BY id DESC LIMIT 1
                        )
                    """, (feedback, gap_flag, clean_q))
                else:
                    cursor.execute("""
                        UPDATE chat_logs 
                        SET feedback = ?
                        WHERE id = (
                            SELECT id FROM chat_logs 
                            WHERE question = ? 
                            ORDER BY id DESC LIMIT 1
                        )
                    """, (feedback, clean_q))

                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    return True

            # 3. Tertiary Fallback: Substring Match
            if clean_q:
                short_q = clean_q[:40] if len(clean_q) > 40 else clean_q
                if gap_flag is not None:
                    cursor.execute("""
                        UPDATE chat_logs 
                        SET feedback = ?, knowledge_gap = ?
                        WHERE id = (
                            SELECT id FROM chat_logs 
                            WHERE question LIKE ? 
                            ORDER BY id DESC LIMIT 1
                        )
                    """, (feedback, gap_flag, f"%{short_q}%"))
                else:
                    cursor.execute("""
                        UPDATE chat_logs 
                        SET feedback = ?
                        WHERE id = (
                            SELECT id FROM chat_logs 
                            WHERE question LIKE ? 
                            ORDER BY id DESC LIMIT 1
                        )
                    """, (feedback, f"%{short_q}%"))

                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    return True

            # 4. Quaternary Fallback: Latest unrated interaction
            if gap_flag is not None:
                cursor.execute("""
                    UPDATE chat_logs 
                    SET feedback = ?, knowledge_gap = ?
                    WHERE id = (
                        SELECT id FROM chat_logs 
                        WHERE feedback IS NULL OR feedback = ''
                        ORDER BY id DESC LIMIT 1
                    )
                """, (feedback, gap_flag))
            else:
                cursor.execute("""
                    UPDATE chat_logs 
                    SET feedback = ?
                    WHERE id = (
                        SELECT id FROM chat_logs 
                        WHERE feedback IS NULL OR feedback = ''
                        ORDER BY id DESC LIMIT 1
                    )
                """, (feedback,))

            conn.commit()
            conn.close()
            return True

    except Exception as e:
        print(f"[Analytics Feedback Error]: {e}")
        return False


def get_analytics_stats() -> dict:
    """
    Returns top-level KPIs for the Admin Dashboard.
    Executed without _DB_LOCK to allow full concurrent reads under SQLite WAL mode.
    """
    stats = {
        "total_queries": 0,
        "knowledge_gaps_count": 0,
        "resolved_gaps_count": 0,
        "unresolved_gaps_count": 0,
        "likes_count": 0,
        "dislikes_count": 0,
        "satisfaction_rate": 100
    }
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chat_logs;")
        stats["total_queries"] = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE knowledge_gap = 1;")
        stats["knowledge_gaps_count"] = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE knowledge_gap = 1 AND resolved = 1;")
        stats["resolved_gaps_count"] = cursor.fetchone()[0] or 0

        stats["unresolved_gaps_count"] = stats["knowledge_gaps_count"] - stats["resolved_gaps_count"]

        cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE feedback = 'like';")
        stats["likes_count"] = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE feedback = 'dislike';")
        stats["dislikes_count"] = cursor.fetchone()[0] or 0

        total_rated = stats["likes_count"] + stats["dislikes_count"]
        if total_rated > 0:
            stats["satisfaction_rate"] = round((stats["likes_count"] / total_rated) * 100)
        else:
            stats["satisfaction_rate"] = 100

        conn.close()
    except Exception as e:
        print(f"[Analytics Stats Error]: {e}")

    return stats


def get_knowledge_gaps(limit: int = 50, offset: int = 0) -> list:
    """
    Returns unanswered or flagged student questions grouped by occurrence frequency.
    Supports pagination via `limit` and `offset`.
    Runs lock-free to maximize read throughput.
    """
    gaps = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                MAX(id) AS id,
                question,
                COUNT(*) AS frequency,
                MAX(timestamp) AS last_asked,
                MAX(resolved) AS resolved,
                MAX(answer) AS last_answer
            FROM chat_logs
            WHERE knowledge_gap = 1
            GROUP BY LOWER(TRIM(question))
            ORDER BY resolved ASC, frequency DESC, last_asked DESC
            LIMIT ? OFFSET ?
        """, (max(1, int(limit)), max(0, int(offset))))
        rows = cursor.fetchall()
        for r in rows:
            gaps.append({
                "id": r["id"],
                "question": r["question"],
                "frequency": r["frequency"],
                "last_asked": r["last_asked"],
                "resolved": bool(r["resolved"]),
                "last_answer": r["last_answer"] or ""
            })
        conn.close()
    except Exception as e:
        print(f"[Get Gaps Error]: {e}")
    return gaps


def get_recent_logs(limit: int = 50, offset: int = 0) -> list:
    """
    Returns recent student interactions with pagination support (limit & offset).
    Runs lock-free to allow concurrent reads during background writes.
    """
    logs = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, question, answer, context_found, knowledge_gap, feedback, resolved
            FROM chat_logs
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (max(1, int(limit)), max(0, int(offset))))
        rows = cursor.fetchall()
        for r in rows:
            logs.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "question": r["question"],
                "answer": r["answer"] or "",
                "context_found": bool(r["context_found"]),
                "knowledge_gap": bool(r["knowledge_gap"]),
                "feedback": r["feedback"],
                "resolved": bool(r["resolved"])
            })
        conn.close()
    except Exception as e:
        print(f"[Get Recent Logs Error]: {e}")
    return logs


def get_logs_count() -> int:
    """Returns the total number of chat interaction logs in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chat_logs;")
        count = cursor.fetchone()[0] or 0
        conn.close()
        return count
    except Exception:
        return 0


def get_knowledge_gaps_count() -> int:
    """Returns total distinct knowledge gap clusters."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT LOWER(TRIM(question)))
            FROM chat_logs
            WHERE knowledge_gap = 1;
        """)
        count = cursor.fetchone()[0] or 0
        conn.close()
        return count
    except Exception:
        return 0


def get_top_questions(limit: int = 8) -> list:
    """
    Returns the most frequently asked student questions.
    Distinguishes between active unresolved gaps, resolved gaps, and normal questions.
    Runs lock-free under WAL mode.
    """
    top_q = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                question,
                COUNT(*) as count,
                MAX(timestamp) as last_seen,
                MAX(CASE WHEN knowledge_gap = 1 AND (resolved IS NULL OR resolved = 0) THEN 1 ELSE 0 END) as is_unresolved_gap,
                MAX(CASE WHEN knowledge_gap = 1 AND resolved = 1 THEN 1 ELSE 0 END) as is_resolved_gap
            FROM chat_logs
            GROUP BY LOWER(TRIM(question))
            ORDER BY count DESC
            LIMIT ?
        """, (max(1, int(limit)),))
        rows = cursor.fetchall()
        for r in rows:
            if not is_greeting_or_chitchat(r["question"]):
                is_unresolved = bool(r["is_unresolved_gap"])
                is_resolved = bool(r["is_resolved_gap"]) and not is_unresolved
                top_q.append({
                    "question": r["question"],
                    "count": r["count"],
                    "last_seen": r["last_seen"],
                    "had_gap": is_unresolved,           # backward-compatible: only True if active unresolved gap
                    "unresolved_gap": is_unresolved,
                    "resolved_gap": is_resolved
                })
        conn.close()
    except Exception as e:
        print(f"[Get Top Questions Error]: {e}")
    return top_q


def mark_gap_resolved(gap_id: int):
    """Marks a specific knowledge gap question as resolved."""
    try:
        with _DB_LOCK:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Mark all occurrences with the same question as resolved
            cursor.execute("""
                UPDATE chat_logs 
                SET resolved = 1 
                WHERE LOWER(TRIM(question)) = (
                    SELECT LOWER(TRIM(question)) FROM chat_logs WHERE id = ?
                )
            """, (gap_id,))
            conn.commit()
            conn.close()
            return True
    except Exception as e:
        print(f"[Mark Resolved Error]: {e}")
        return False


def clear_all_logs():
    """Clears all records for admin testing/maintenance."""
    try:
        with _DB_LOCK:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_logs;")
            conn.commit()
            conn.close()
            return True
    except Exception as e:
        print(f"[Clear Logs Error]: {e}")
        return False
