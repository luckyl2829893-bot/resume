import sqlite3
from datetime import datetime
from pathlib import Path
import sys

# Setup package import alignments
sys.path.append(str(Path(__file__).parent.parent))
from config import DB_PATH

def init_db():
    """
    Initializes the SQLite database and creates the applications table if it doesn't exist.
    """
    db_file = Path(DB_PATH)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            role TEXT NOT NULL,
            url TEXT,
            status TEXT DEFAULT 'draft',
            original_score REAL,
            tailored_score REAL,
            notes TEXT,
            applied_date TEXT,
            updated_date TEXT
        )
    """)
    conn.commit()

    # MLOps experiment tracking table — one row per tailoring/generation run
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mlops_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TEXT NOT NULL,
            model_used TEXT,
            model_label TEXT,
            job_title TEXT,
            company TEXT,
            original_score REAL,
            tailored_score REAL,
            score_delta REAL,
            grade TEXT,
            missing_keywords TEXT,
            section_order TEXT,
            run_type TEXT DEFAULT 'tailor'
        )
    """)
    conn.commit()
    conn.close()

def add_application(company: str, role: str, url: str = "", status: str = "draft",
                    original_score: float = 0.0, tailored_score: float = 0.0,
                    notes: str = "") -> int:
    """
    Adds a new job application record to the tracker.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO applications 
        (company, role, url, status, original_score, tailored_score, notes, applied_date, updated_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (company, role, url, status, original_score, tailored_score, notes, 
          now_str if status == "applied" else None, now_str))
          
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id

def update_status(app_id: int, status: str, notes: str = None) -> bool:
    """
    Updates the application status and notes, adjusting applied_date or updated_date accordingly.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if notes is not None:
        cursor.execute("""
            UPDATE applications 
            SET status = ?, notes = ?, updated_date = ?,
                applied_date = CASE WHEN ? = 'applied' AND applied_date IS NULL THEN ? ELSE applied_date END
            WHERE id = ?
        """, (status, notes, now_str, status, now_str, app_id))
    else:
        cursor.execute("""
            UPDATE applications 
            SET status = ?, updated_date = ?,
                applied_date = CASE WHEN ? = 'applied' AND applied_date IS NULL THEN ? ELSE applied_date END
            WHERE id = ?
        """, (status, now_str, status, now_str, app_id))
        
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def get_all() -> list:
    """
    Retrieves all application records ordered by recent updates.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM applications ORDER BY updated_date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete(app_id: int) -> bool:
    """
    Deletes an application record from the database.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM applications WHERE id = ?", (app_id,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def get_stats() -> dict:
    """
    Retrieves high-level summary counters of application statuses.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT status, COUNT(*) FROM applications GROUP BY status")
    rows = cursor.fetchall()
    conn.close()
    
    counts = {r[0]: r[1] for r in rows}
    total = sum(counts.values())
    
    # Summary of stages
    applied = counts.get("applied", 0)
    screening = counts.get("screening", 0)
    interviews = counts.get("interview", 0)
    offers = counts.get("offer", 0)
    
    return {
        "total": total,
        "applied": applied,
        "interviews": interviews,
        "offers": offers,
        "counts": counts
    }


# ── MLOps Experiment Logging ───────────────────────────────────────────────────

def log_experiment(
    model_used: str,
    model_label: str,
    job_title: str,
    company: str,
    original_score: float,
    tailored_score: float,
    grade: str,
    missing_keywords: list,
    section_order: list = None,
    run_type: str = "tailor",
) -> int:
    """
    Logs a single tailoring/generation run to the mlops_runs table.
    Returns the new row ID.
    """
    import json
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO mlops_runs
        (run_ts, model_used, model_label, job_title, company,
         original_score, tailored_score, score_delta, grade,
         missing_keywords, section_order, run_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str, model_used, model_label, job_title, company,
        original_score, tailored_score,
        round(tailored_score - original_score, 2),
        grade,
        json.dumps(missing_keywords[:10]),
        json.dumps(section_order or []),
        run_type,
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_experiments(limit: int = 50) -> list:
    """Returns the most recent MLOps experiment runs."""
    import json
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM mlops_runs ORDER BY run_ts DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["missing_keywords"] = json.loads(d.get("missing_keywords") or "[]")
            d["section_order"] = json.loads(d.get("section_order") or "[]")
        except Exception:
            pass
        result.append(d)
    return result
