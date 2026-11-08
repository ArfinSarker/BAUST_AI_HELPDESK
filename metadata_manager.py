import os
import re
import json
import sqlite3
import threading
from datetime import datetime
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWLEDGE_FOLDER = os.path.join(BASE_DIR, "knowledge_base")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "analytics.db")
META_JSON_BACKUP = os.path.join(DATA_DIR, "document_metadata.json")

# Write lock for serializing version comparison and update transactions
_WRITE_LOCK = threading.Lock()
# Backward compatibility alias
_META_LOCK = _WRITE_LOCK

# Standard Document Types
DOC_TYPES = [
    ("academic_calendar", "📅 Academic Calendar"),
    ("fee_structure", "💰 Tuition & Fee Structure"),
    ("exam_policy", "📝 Examination & Grading Policy"),
    ("seatplan", "🪑 Exam Seatplan"),
    ("admission_notice", "🎓 Admission Notice"),
    ("faculty_directory", "👨‍🏫 Faculty Directory"),
    ("syllabus", "📖 Academic Syllabus / Curriculum"),
    ("hall_guidelines", "🏢 Residential Hall Guidelines"),
    ("transport_schedule", "🚌 Transport Schedule & Routes"),
    ("library_rules", "📚 Library Guidelines & Rules"),
    ("scholarship_notice", "🏆 Scholarship & Waiver Notice"),
    ("medical_guidelines", "🩺 Medical & Healthcare Guidelines"),
    ("code_of_conduct", "⚖️ Student Code of Conduct & Rules"),
    ("club_notice", "🎯 Club & Extracurricular Notice"),
    ("career_placement", "💼 Career & Internship Notice"),
    ("routine", "🗓️ Class & Lab Routine"),
    ("general_notice", "📢 General Notice / Circular"),
]

# Standard Issuing Authorities
AUTHORITIES = [
    "Office of the Registrar",
    "Controller of Examinations",
    "Admission Office",
    "Office of the Proctor",
    "Office of the Provost",
    "Office of the Treasurer",
    "Central Library",
    "Medical Center",
    "Transport Section",
    "Department of CSE",
    "Department of EEE",
    "Department of ME",
    "Department of CE",
    "Department of IPE",
    "Faculty of Business Studies",
    "Department of English",
    "Department of Arts & Sciences",
    "Directorate of Students' Welfare (DSW)",
    "BAUST Authority",
]

# =========================================================================
# PRE-COMPILED REGEX PATTERNS (Anti-Backtracking & Optimized Execution)
# =========================================================================
RE_YEAR = re.compile(r'\b(202[3-9]|203[0-5])(?:-(202[4-9]|203[0-5]))?\b')
RE_DATE = re.compile(
    r'(?:effective\s*(?:from|date)?|published\s*on|তারিখ|date)\s*[:\-–]\s*'
    r'([0-9]{1,2}[-/\.][0-9]{1,2}[-/\.][0-9]{2,4}|[0-9]{1,2}\s+[A-Za-z]+\s+[0-9]{4}|[A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4}|[0-9]{4}-[0-9]{2}-[0-9]{2})',
    re.IGNORECASE
)
RE_SUBJECT_PRIMARY = re.compile(
    r'(?:subject|বিষয়|title|regarding)\s*[:\-–]\s*([^\n\r]{6,100})',
    re.IGNORECASE
)
RE_NOTICE_SECONDARY = re.compile(
    r'(?:notice|circular|ref)\s*[:\-–]\s*([^\n\r]{6,100})',
    re.IGNORECASE
)

# Non-greedy, space-bounded character sets preventing catastrophic backtracking
RE_OFFICE = re.compile(
    r'office\s+of\s+the\s+([A-Za-z0-9(),& -]{3,60}?)(?:[\n\r,\.]|\bbaust\b|$)',
    re.IGNORECASE
)
RE_DEPT = re.compile(
    r'department\s+of\s+([A-Za-z0-9(),& -]{2,50}?)(?:[\n\r,\.]|\bbaust\b|$)',
    re.IGNORECASE
)
RE_FACULTY = re.compile(
    r'faculty\s+of\s+([A-Za-z0-9(),& -]{2,50}?)(?:[\n\r,\.]|\bbaust\b|$)',
    re.IGNORECASE
)
RE_CLUB = re.compile(
    r'\b(baust\s+[A-Za-z0-9 -]{2,35}\s+club)\b',
    re.IGNORECASE
)


# =========================================================================
# DATABASE CONNECTION & CONTEXT MANAGEMENT
# =========================================================================

@contextmanager
def get_db_connection_safe(timeout: float = 30.0):
    """
    Context manager for safe, leak-free SQLite connections configured with WAL mode.
    Guarantees connection closing and rollback on unhandled errors.
    """
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_db_connection():
    """Returns a SQLite connection configured with WAL mode (legacy compatibility)."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def init_metadata_db():
    """Creates the metadata table if it doesn't already exist."""
    with _WRITE_LOCK:
        with get_db_connection_safe() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    filename TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    academic_year TEXT DEFAULT '',
                    semester TEXT DEFAULT '',
                    version TEXT DEFAULT '1.0',
                    issuing_authority TEXT DEFAULT 'BAUST Authority',
                    effective_date TEXT DEFAULT '',
                    expiry_date TEXT DEFAULT '',
                    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    superseded_by TEXT DEFAULT NULL,
                    is_authoritative INTEGER DEFAULT 1,
                    notes TEXT DEFAULT ''
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_meta_type ON document_metadata (doc_type, status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_meta_status ON document_metadata (status, is_authoritative);")
            conn.commit()


init_metadata_db()


# =========================================================================
# LOCK-FREE READ QUERIES (FULL CONCURRENCY UNDER WAL)
# =========================================================================

def get_all_doc_types() -> list:
    """
    Returns standard document types merged with any dynamically
    created custom document types found in the database.
    Lock-free execution under WAL mode.
    """
    types_map = {code: label for code, label in DOC_TYPES}
    try:
        with get_db_connection_safe() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT doc_type FROM document_metadata WHERE doc_type IS NOT NULL AND doc_type != '';")
            rows = cursor.fetchall()
            for r in rows:
                code = r[0]
                if code and code not in types_map:
                    clean_name = code.replace("_", " ").title()
                    types_map[code] = f"📄 {clean_name}"
    except Exception as e:
        print(f"[Doc Types Fetch Error]: {e}")
    return list(types_map.items())


def get_all_authorities() -> list:
    """
    Returns standard authorities merged with any custom authorities
    detected or registered in the metadata database.
    Lock-free execution under WAL mode.
    """
    auth_list = list(AUTHORITIES)
    try:
        with get_db_connection_safe() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT issuing_authority FROM document_metadata WHERE issuing_authority IS NOT NULL AND issuing_authority != '';")
            rows = cursor.fetchall()
            for r in rows:
                auth = r[0]
                if auth and auth.strip() and auth.strip() not in auth_list:
                    auth_list.append(auth.strip())
    except Exception as e:
        print(f"[Authorities Fetch Error]: {e}")
    return auth_list


def get_document_metadata(filename: str) -> dict:
    """Fetches metadata dictionary for a specific file (Lock-free)."""
    try:
        with get_db_connection_safe() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM document_metadata WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            if row:
                return dict(row)
    except Exception as e:
        print(f"[Get Metadata Error]: {e}")
    return None


def get_all_documents_metadata() -> dict:
    """Returns a dict mapping filename -> metadata dictionary (Lock-free)."""
    result = {}
    try:
        with get_db_connection_safe() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM document_metadata ORDER BY doc_type, is_authoritative DESC, upload_date DESC")
            rows = cursor.fetchall()
            for r in rows:
                result[r["filename"]] = dict(r)
    except Exception as e:
        print(f"[Get All Metadata Error]: {e}")
    return result


# =========================================================================
# HEURISTIC & NLP METADATA INFERENCE (With Backtracking Protections)
# =========================================================================

def infer_metadata_from_text(filename: str, text: str = "") -> dict:
    """
    Intelligently infers document title, type, year, semester, and issuing authority
    from filename and text content using heuristic semantic matching and pre-compiled regex NLP.
    
    Protections:
    - Inputs bounded to 3000 chars to avoid memory exhaustion
    - Whitespace collapsed to prevent catastrophic regex backtracking
    - Non-greedy bounded character sets
    """
    f_lower = filename.lower()
    
    # 1. Bounded text sanitation & normalization
    text_sample = text[:3000] if text else ""
    normalized_text = re.sub(r'[ \t]+', ' ', text_sample)
    t_lower = normalized_text.lower()
    combined = f_lower + " " + t_lower

    # 2. Infer Academic Year
    year_match = RE_YEAR.search(combined)
    academic_year = year_match.group(0) if year_match else datetime.now().strftime("%Y")

    # 3. Infer Semester
    semester = ""
    if "summer" in combined or "সামার" in combined:
        semester = "Summer"
    elif "fall" in combined or "ফল" in combined:
        semester = "Fall"
    elif "spring" in combined or "স্প্রিং" in combined:
        semester = "Spring"
    elif "winter" in combined:
        semester = "Winter"

    # 4. Infer Effective Date from text if present
    effective_date = ""
    date_match = RE_DATE.search(normalized_text)
    if date_match:
        raw_d = date_match.group(1).strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %B %Y", "%B %d, %Y", "%B %d %Y"):
            try:
                dt = datetime.strptime(raw_d, fmt)
                effective_date = dt.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

    if not effective_date:
        effective_date = datetime.now().strftime("%Y-%m-%d")

    # 5. Infer Document Title
    raw_stem = os.path.splitext(filename)[0]
    title = raw_stem.replace("_", " ").title()

    subj_match = RE_SUBJECT_PRIMARY.search(normalized_text)
    if not subj_match:
        subj_match = RE_NOTICE_SECONDARY.search(normalized_text)

    if subj_match:
        candidate_title = subj_match.group(1).strip()
        candidate_title = re.sub(r'[\.\;\:]+$', '', candidate_title).strip()
        if len(candidate_title) >= 6:
            title = candidate_title

    # 6. Infer Issuing Authority
    authority = "BAUST Authority"
    
    office_match = RE_OFFICE.search(normalized_text)
    dept_match = RE_DEPT.search(normalized_text)
    faculty_match = RE_FACULTY.search(normalized_text)
    club_match = RE_CLUB.search(normalized_text)

    if office_match:
        raw_off = re.split(r'[\n\r,\.]|\bbaust\b', office_match.group(1), flags=re.IGNORECASE)[0].strip()
        off_name = raw_off.title()
        if "Proctor" in off_name:
            authority = "Office of the Proctor"
        elif "Registrar" in off_name:
            authority = "Office of the Registrar"
        elif "Provost" in off_name:
            authority = "Office of the Provost"
        elif "Treasurer" in off_name:
            authority = "Office of the Treasurer"
        elif "Controller" in off_name or "Examination" in off_name:
            authority = "Controller of Examinations"
        elif "Admission" in off_name:
            authority = "Admission Office"
        elif "Transport" in off_name:
            authority = "Transport Section"
        elif "Sports" in off_name or "Physical Education" in off_name:
            authority = "Office of the Physical Education & Sports Committee"
        elif "Dean" in off_name:
            clean_dean = off_name.replace("Office Of The Dean", "").strip()
            authority = f"Office of the Dean, {clean_dean or 'Faculty'}"
        else:
            authority = f"Office of the {off_name}"

    elif dept_match:
        raw_dept = re.split(r'[\n\r,\.]|\bbaust\b', dept_match.group(1), flags=re.IGNORECASE)[0].strip()
        dept_name = raw_dept.title()
        if "Cse" in dept_name or "Computer Science" in dept_name:
            authority = "Department of CSE"
        elif "Eee" in dept_name or "Electrical" in dept_name:
            authority = "Department of EEE"
        elif "Me" in dept_name or "Mechanical" in dept_name:
            authority = "Department of ME"
        elif "Ce" in dept_name or "Civil" in dept_name:
            authority = "Department of CE"
        elif "Ipe" in dept_name or "Industrial" in dept_name:
            authority = "Department of IPE"
        elif "English" in dept_name:
            authority = "Department of English"
        elif "Business" in dept_name or "Bba" in dept_name:
            authority = "Faculty of Business Studies"
        else:
            authority = f"Department of {dept_name}"

    elif faculty_match:
        raw_fac = re.split(r'[\n\r,\.]|\bbaust\b', faculty_match.group(1), flags=re.IGNORECASE)[0].strip()
        fac_name = raw_fac.title()
        authority = f"Faculty of {fac_name}"

    elif club_match:
        raw_club = re.split(r'[\n\r,\.]', club_match.group(1))[0].strip()
        authority = raw_club.title()

    elif "central library" in combined or "baust library" in combined:
        authority = "Central Library"
    elif "medical center" in combined or "medical centre" in combined or "health care" in combined:
        authority = "Medical Center"
    elif "transport section" in combined or "transport committee" in combined or "bus schedule" in combined:
        authority = "Transport Section"
    elif "proctor" in combined:
        authority = "Office of the Proctor"
    elif "provost" in combined or "hall authority" in combined:
        authority = "Office of the Provost"
    elif "dsw" in combined or "student welfare" in combined or "students' welfare" in combined:
        authority = "Directorate of Students' Welfare (DSW)"
    elif "admission" in combined:
        authority = "Admission Office"
    elif "exam" in combined or "grading" in combined or "seatplan" in combined:
        authority = "Controller of Examinations"
    elif "calendar" in combined or "syllabus" in combined:
        authority = "Office of the Registrar"

    # 7. Infer Document Type
    doc_type = "general_notice"

    if "calendar" in f_lower or "academic_calendar" in f_lower or "academic calendar" in t_lower:
        doc_type = "academic_calendar"
        if authority == "BAUST Authority": authority = "Office of the Registrar"
        if not subj_match: title = f"Academic Calendar {semester} {academic_year}".strip()

    elif "seatplan" in f_lower or "seat_plan" in f_lower or "seat plan" in t_lower or "সিট প্ল্যান" in t_lower:
        doc_type = "seatplan"
        if authority == "BAUST Authority": authority = "Controller of Examinations"
        if not subj_match: title = f"Exam Seatplan {semester} {academic_year}".strip()

    elif any(k in f_lower for k in ["sports", "gym", "fitness"]) or any(k in t_lower[:300] for k in ["sports", "gym", "fitness", "tournament"]):
        doc_type = "sports_notice"
        if authority == "BAUST Authority": authority = "Office of the Physical Education & Sports Committee"
        if not subj_match: title = "Sports & Gymnasium Guidelines"

    elif "tuition" in combined or "fee_structure" in f_lower or "fee structure" in t_lower or "tuition fee" in combined or ("fee" in f_lower and not any(k in f_lower for k in ["gym", "library", "hall", "transport", "bus"])):
        doc_type = "fee_structure"
        if authority == "BAUST Authority": authority = "Admission Office"
        if not subj_match: title = f"Tuition & Fee Structure {academic_year}".strip()

    elif "exam" in combined or "grading" in combined or "cgpa" in combined or "পরীক্ষা" in combined:
        doc_type = "exam_policy"
        if authority == "BAUST Authority": authority = "Controller of Examinations"
        if not subj_match: title = "Undergraduate Examination Policy & Regulations"

    elif "admission" in combined or "ভর্তি" in combined:
        doc_type = "admission_notice"
        if authority == "BAUST Authority": authority = "Admission Office"
        if not subj_match: title = f"Admission Information {academic_year}".strip()

    elif "faculty" in f_lower or ("teacher" in combined and "directory" in combined):
        doc_type = "faculty_directory"
        if not subj_match: title = f"{authority} Faculty Directory"

    elif "hall" in combined or "hostel" in combined or "residential" in combined:
        doc_type = "hall_guidelines"
        if authority == "BAUST Authority": authority = "Office of the Provost"
        if not subj_match: title = "Residential Halls & Facilities Guidelines"

    elif "syllabus" in combined or "curriculum" in combined:
        doc_type = "syllabus"
        if authority == "BAUST Authority": authority = "Office of the Registrar"
        if not subj_match: title = f"Undergraduate Syllabus {academic_year}".strip()

    elif any(k in combined for k in ["transport", "bus", "route", "shuttle", "পিকআপ", "বাস"]):
        doc_type = "transport_schedule"
        if authority == "BAUST Authority": authority = "Transport Section"
        if not subj_match: title = f"University Transport Schedule & Routes {academic_year}".strip()

    elif any(k in combined for k in ["library", "book", "borrow", "circulation", "লাইব্রেরি"]):
        doc_type = "library_rules"
        if authority == "BAUST Authority": authority = "Central Library"
        if not subj_match: title = "Central Library Rules & Book Borrowing Policy"

    elif any(k in combined for k in ["scholarship", "waiver", "stipend", "financial aid", "মেধাবৃত্তি", "ছাড়"]):
        doc_type = "scholarship_notice"
        if authority == "BAUST Authority": authority = "Office of the Registrar"
        if not subj_match: title = f"Scholarship & Tuition Waiver Notice {academic_year}".strip()

    elif any(k in combined for k in ["medical", "clinic", "doctor", "health", "ambulance", "ফার্স্ট এইড"]):
        doc_type = "medical_guidelines"
        if authority == "BAUST Authority": authority = "Medical Center"
        if not subj_match: title = "Medical Center & Healthcare Guidelines"

    elif any(k in combined for k in ["proctor", "discipline", "ragging", "code of conduct", "dress code", "id card", "শৃঙ্খলা"]):
        doc_type = "code_of_conduct"
        if authority == "BAUST Authority": authority = "Office of the Proctor"
        if not subj_match: title = "Student Code of Conduct & Disciplinary Policy"

    elif any(k in combined for k in ["routine", "class routine", "lab routine", "class schedule"]):
        doc_type = "routine"
        if not subj_match: title = f"Class & Lab Routine {semester} {academic_year}".strip()

    elif any(k in combined for k in ["career", "placement", "internship", "job fair", "নিয়োগ"]):
        doc_type = "career_placement"
        if authority == "BAUST Authority": authority = "Directorate of Students' Welfare (DSW)"
        if not subj_match: title = "Career & Internship Placement Circular"

    elif any(k in combined for k in ["club", "fest", "hackathon", "contest", "competition", "workshop", "seminar"]):
        doc_type = "club_notice"
        if not subj_match: title = f"Event & Extracurricular Notice {academic_year}".strip()

    elif any(k in combined for k in ["research", "journal", "conference", "publication", "grant"]):
        doc_type = "research_notice"
        if not subj_match: title = "Research & Publication Circular"

    else:
        clean_stem = re.sub(r'[^a-zA-Z0-9]+', '_', raw_stem.lower()).strip('_')
        if clean_stem and len(clean_stem) >= 4 and not clean_stem.startswith(("file", "doc", "upload", "image", "pdf")):
            doc_type = clean_stem
            title = raw_stem.replace("_", " ").title()
        else:
            doc_type = "general_notice"

    now_date = datetime.now().strftime("%Y-%m-%d")

    return {
        "title": title,
        "doc_type": doc_type,
        "academic_year": academic_year,
        "semester": semester,
        "version": "1.0",
        "issuing_authority": authority,
        "effective_date": now_date,
        "expiry_date": "",
        "status": "active",
        "is_authoritative": 1
    }


SEMESTER_WEIGHTS = {
    "spring": 1, "স্প্রিং": 1, "বসন্ত": 1,
    "summer": 2, "সামার": 2, "গ্রীষ্ম": 2,
    "fall": 3, "ফল": 3, "autumn": 3, "শরৎ": 3,
    "winter": 4, "উইন্টার": 4, "শীত": 4
}


def parse_chronological_key(academic_year: str = "", semester: str = "", effective_date: str = "", upload_date: str = "") -> tuple:
    """Computes a strictly comparable tuple for version chronology comparison."""
    year_val = 2000
    if academic_year:
        m = re.search(r'\b(20[2-3]\d)\b', str(academic_year))
        if m:
            year_val = int(m.group(1))
    elif effective_date:
        m = re.search(r'\b(20[2-3]\d)\b', str(effective_date))
        if m:
            year_val = int(m.group(1))

    sem_val = SEMESTER_WEIGHTS.get(str(semester).strip().lower(), 0)
    eff_val = str(effective_date).strip() if effective_date else "1970-01-01"
    up_val = str(upload_date).strip() if upload_date else "1970-01-01 00:00:00"

    return (year_val, sem_val, eff_val, up_val)


# =========================================================================
# LIFECYCLE & VERSION MANAGEMENT (Safe Writes via _WRITE_LOCK)
# =========================================================================

def auto_manage_version_and_lifecycle(
    filename: str,
    doc_type: str,
    academic_year: str = "",
    semester: str = "",
    effective_date: str = "",
    issuing_authority: str = "",
    requested_version: str = None
) -> tuple:
    """
    Automatically tracks document chronology, updates version numbers, and manages lifecycle.
    Serialized via _WRITE_LOCK and get_db_connection_safe.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_chrono = parse_chronological_key(academic_year, semester, effective_date, now_str)

    superseded_files = []
    assigned_version = requested_version or "1.0"
    status = "active"
    is_authoritative = 1

    try:
        with _WRITE_LOCK:
            with get_db_connection_safe() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT filename, title, doc_type, academic_year, semester, version, 
                           issuing_authority, effective_date, upload_date, status 
                    FROM document_metadata 
                    WHERE status = 'active'
                """)
                active_docs = cursor.fetchall()

                matching_predecessors = []
                base_new = re.sub(r'(20[2-3]\d|_summer|_spring|_fall|_winter|_v\d+)', '', filename.lower()).strip('_-')

                PERIODIC_SERIES_TYPES = [
                    "academic_calendar", "seatplan", "routine", "annual_report"
                ]

                for doc in active_docs:
                    doc_fn = doc["filename"]
                    if doc_fn == filename:
                        matching_predecessors.append(doc)
                        continue

                    base_old = re.sub(r'(20[2-3]\d|_summer|_spring|_fall|_winter|_v\d+)', '', doc_fn.lower()).strip('_-')

                    if base_old and base_old == base_new and len(base_old) >= 4:
                        matching_predecessors.append(doc)
                        continue

                    if doc["doc_type"] == doc_type and doc_type in PERIODIC_SERIES_TYPES:
                        matching_predecessors.append(doc)
                        continue

                if matching_predecessors:
                    highest_existing_v = 1.0
                    all_older = True
                    newer_existing_doc = None

                    for pred in matching_predecessors:
                        try:
                            v_num = float(str(pred["version"]).replace("v", "").replace("V", "").strip())
                            highest_existing_v = max(highest_existing_v, v_num)
                        except Exception:
                            pass

                        pred_chrono = parse_chronological_key(
                            pred["academic_year"], pred["semester"], 
                            pred["effective_date"], pred["upload_date"]
                        )
                        
                        if new_chrono < pred_chrono:
                            all_older = False
                            newer_existing_doc = pred["filename"]

                    if all_older:
                        if not requested_version or requested_version in ["1.0", ""]:
                            old_year = matching_predecessors[0]["academic_year"]
                            if academic_year and old_year and academic_year > old_year:
                                assigned_version = f"{int(highest_existing_v) + 1.0:.1f}"
                            else:
                                assigned_version = f"{highest_existing_v + 0.1:.1f}"

                        status = "active"
                        is_authoritative = 1

                        for pred in matching_predecessors:
                            old_fn = pred["filename"]
                            if old_fn != filename:
                                cursor.execute("""
                                    UPDATE document_metadata 
                                    SET status = 'superseded', 
                                        superseded_by = ?, 
                                        is_authoritative = 0,
                                        last_updated = ?
                                    WHERE filename = ?
                                """, (filename, now_str, old_fn))
                                superseded_files.append(old_fn)
                                print(f"[Auto Version Control]: Older document '{old_fn}' (v{pred['version']}) marked as superseded by newer '{filename}' (v{assigned_version})")

                    else:
                        status = "archived"
                        is_authoritative = 0
                        assigned_version = requested_version or "0.9"
                        print(f"[Auto Version Control]: Uploaded '{filename}' is older than active '{newer_existing_doc}'. Marked as archived history.")

                conn.commit()
    except Exception as e:
        print(f"[Auto Version Control Error]: {e}")

    return assigned_version, status, is_authoritative, superseded_files


def supersede_previous_documents(doc_type: str, new_filename: str, semester: str = "", academic_year: str = "") -> list:
    """Wrapper for backward compatibility."""
    ver, stat, auth, superseded = auto_manage_version_and_lifecycle(
        filename=new_filename,
        doc_type=doc_type,
        academic_year=academic_year,
        semester=semester
    )
    return superseded


def register_or_update_document(
    filename: str,
    title: str = None,
    doc_type: str = None,
    academic_year: str = None,
    semester: str = None,
    version: str = "1.0",
    issuing_authority: str = None,
    effective_date: str = None,
    expiry_date: str = None,
    status: str = "active",
    notes: str = "",
    auto_supersede: bool = True
) -> dict:
    """
    Registers a new document or updates an existing one with full metadata.
    Automatically supersedes older active versions of the same document type if auto_supersede is True.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_date = datetime.now().strftime("%Y-%m-%d")

    if not title or not doc_type:
        content_preview = ""
        full_path = os.path.join(KNOWLEDGE_FOLDER, filename)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content_preview = f.read(2500)
            except Exception:
                pass
        inferred = infer_metadata_from_text(filename, content_preview)
        title = title or inferred["title"]
        doc_type = doc_type or inferred["doc_type"]
        academic_year = academic_year if academic_year is not None else inferred["academic_year"]
        semester = semester if semester is not None else inferred["semester"]
        issuing_authority = issuing_authority or inferred["issuing_authority"]
        effective_date = effective_date or inferred["effective_date"]

    if doc_type:
        clean_code = re.sub(r'[^a-zA-Z0-9_]+', '_', doc_type.strip().lower()).strip('_')
        doc_type = clean_code or "general_notice"

    if issuing_authority:
        issuing_authority = issuing_authority.strip()

    superseded_list = []
    if auto_supersede:
        assigned_ver, calc_status, calc_auth, superseded_list = auto_manage_version_and_lifecycle(
            filename=filename,
            doc_type=doc_type,
            academic_year=academic_year,
            semester=semester,
            effective_date=effective_date,
            issuing_authority=issuing_authority,
            requested_version=version if (version and version != "1.0") else None
        )
        version = assigned_ver
        status = calc_status
        is_auth = calc_auth
    else:
        is_auth = 1 if status == "active" else 0

    with _WRITE_LOCK:
        with get_db_connection_safe() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO document_metadata (
                    filename, title, doc_type, academic_year, semester,
                    version, issuing_authority, effective_date, expiry_date,
                    upload_date, last_updated, status, superseded_by, is_authoritative, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    title = excluded.title,
                    doc_type = excluded.doc_type,
                    academic_year = excluded.academic_year,
                    semester = excluded.semester,
                    version = excluded.version,
                    issuing_authority = excluded.issuing_authority,
                    effective_date = excluded.effective_date,
                    expiry_date = excluded.expiry_date,
                    last_updated = excluded.last_updated,
                    status = excluded.status,
                    is_authoritative = excluded.is_authoritative,
                    notes = excluded.notes;
            """, (
                filename,
                title,
                doc_type,
                academic_year or "",
                semester or "",
                version or "1.0",
                issuing_authority or "BAUST Authority",
                effective_date or now_date,
                expiry_date or "",
                now_str,
                now_str,
                status,
                is_auth,
                notes or ""
            ))
            conn.commit()

    meta_dict = get_document_metadata(filename)
    if meta_dict:
        meta_dict["superseded_files"] = superseded_list
    sync_metadata_json_backup()
    return meta_dict


def update_document_status(filename: str, new_status: str, superseded_by: str = None) -> bool:
    """Updates the status of a document (e.g. 'active', 'archived', 'superseded')."""
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_auth = 1 if new_status == "active" else 0
        with _WRITE_LOCK:
            with get_db_connection_safe() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE document_metadata 
                    SET status = ?, 
                        superseded_by = ?, 
                        is_authoritative = ?,
                        last_updated = ?
                    WHERE filename = ?
                """, (new_status, superseded_by, is_auth, now_str, filename))
                conn.commit()
        sync_metadata_json_backup()
        return True
    except Exception as e:
        print(f"[Update Status Error]: {e}")
        return False


def delete_document_metadata(filename: str) -> bool:
    """Removes a metadata entry when a document is deleted from knowledge_base/."""
    try:
        with _WRITE_LOCK:
            with get_db_connection_safe() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM document_metadata WHERE filename = ?", (filename,))
                conn.commit()
        sync_metadata_json_backup()
        return True
    except Exception as e:
        print(f"[Delete Metadata Error]: {e}")
        return False


_JSON_BACKUP_LOCK = threading.Lock()


def sync_metadata_json_backup():
    """Exports all document metadata to a JSON backup file in data/ using safe atomic write."""
    if not _JSON_BACKUP_LOCK.acquire(blocking=False):
        # A sync is already in flight; latest state will be captured
        return
    try:
        all_meta = get_all_documents_metadata()
        tmp_path = META_JSON_BACKUP + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(all_meta, f, indent=2, ensure_ascii=False)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, META_JSON_BACKUP)
    except Exception as e:
        print(f"[Sync Metadata JSON Error]: {e}")
    finally:
        _JSON_BACKUP_LOCK.release()


# =========================================================================
# ASYNCHRONOUS BACKGROUND SEEDING
# =========================================================================
_seeding_lock = threading.Lock()
_is_seeding = False


def auto_seed_existing_knowledge_files():
    """
    Initializes metadata for any files in knowledge_base/ that do not yet have records.
    Ensures complete, authoritative metadata coverage without blocking server startup.
    Guarded to prevent duplicate concurrent runs.
    """
    global _is_seeding
    if not _seeding_lock.acquire(blocking=False):
        # A seeding process is already active
        return

    try:
        _is_seeding = True
        if not os.path.exists(KNOWLEDGE_FOLDER):
            return

        all_meta = get_all_documents_metadata()
        files = [f for f in os.listdir(KNOWLEDGE_FOLDER) if f.endswith(".txt") and os.path.isfile(os.path.join(KNOWLEDGE_FOLDER, f))]

        for f in sorted(files):
            if f not in all_meta and not f.endswith((".json", ".pkl", ".npy", ".faiss", ".tmp", ".log")):
                full_path = os.path.join(KNOWLEDGE_FOLDER, f)
                text_sample = ""
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as fp:
                        text_sample = fp.read(3000)
                except Exception:
                    pass
                inferred = infer_metadata_from_text(f, text_sample)
                register_or_update_document(
                    filename=f,
                    title=inferred["title"],
                    doc_type=inferred["doc_type"],
                    academic_year=inferred["academic_year"],
                    semester=inferred["semester"],
                    version="1.0",
                    issuing_authority=inferred["issuing_authority"],
                    effective_date=inferred["effective_date"],
                    status="active",
                    auto_supersede=False
                )
                print(f"[Metadata Seed]: Initialized metadata for '{f}' as '{inferred['doc_type']}' ({inferred['title']})")

        sync_metadata_json_backup()
    except Exception as e:
        print(f"[Metadata Seed Error]: {e}")
    finally:
        _is_seeding = False
        _seeding_lock.release()


def seed_metadata_async():
    """
    Executes metadata seeding in a non-blocking background daemon thread.
    Guarantees that application startup (Gunicorn/Flask/WSGI) remains instantaneous (<50ms).
    """
    t = threading.Thread(
        target=auto_seed_existing_knowledge_files,
        name="MetadataSeederWorker",
        daemon=True
    )
    t.start()
    return t


# Non-blocking asynchronous seeding on module import
seed_metadata_async()
