"""
Database module for AI-Based Surface Defect Detection.
Manages SQLite storage for inspection records, history viewing, and deletion.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inspections.db")


def get_connection(db_path: str = DB_FILE) -> sqlite3.Connection:
    """Returns a SQLite connection with row factory configured as dictionary."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_FILE) -> None:
    """Initializes the database schema if it doesn't already exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_code TEXT NOT NULL,
                date_str TEXT NOT NULL,
                time_str TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                detected_material TEXT NOT NULL,
                material_confidence REAL NOT NULL,
                crack_result TEXT NOT NULL,
                crack_confidence REAL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def insert_inspection(
    image_filename: str,
    detected_material: str,
    material_confidence: float,
    crack_result: str,
    crack_confidence: Optional[float] = None,
    status: str = "COMPLETED",
    db_path: str = DB_FILE
) -> Dict[str, Any]:
    """Inserts a new inspection record into SQLite and returns the record data."""
    init_db(db_path)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    inspection_code = f"INSP-{now.strftime('%Y%m%d-%H%M%S')}"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inspections (
                inspection_code, date_str, time_str, image_filename,
                detected_material, material_confidence, crack_result,
                crack_confidence, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inspection_code,
            date_str,
            time_str,
            image_filename,
            detected_material,
            round(float(material_confidence), 2),
            crack_result,
            round(float(crack_confidence), 2) if crack_confidence is not None else None,
            status
        ))
        conn.commit()
        record_id = cursor.lastrowid

    return {
        "id": record_id,
        "inspection_code": inspection_code,
        "date_str": date_str,
        "time_str": time_str,
        "image_filename": image_filename,
        "detected_material": detected_material,
        "material_confidence": round(float(material_confidence), 2),
        "crack_result": crack_result,
        "crack_confidence": round(float(crack_confidence), 2) if crack_confidence is not None else None,
        "status": status
    }


def get_all_inspections(db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    """Retrieves all inspection records ordered by newest first."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspections ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_inspection_by_id(inspection_id: int, db_path: str = DB_FILE) -> Optional[Dict[str, Any]]:
    """Retrieves a single inspection record by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_inspection(inspection_id: int, db_path: str = DB_FILE) -> bool:
    """Deletes an inspection record by its ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_inspections(db_path: str = DB_FILE) -> int:
    """Clears all inspection records from the database. Returns count of deleted rows."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM inspections")
        conn.commit()
        return cursor.rowcount


def get_statistics(db_path: str = DB_FILE) -> Dict[str, Any]:
    """Returns aggregated summary statistics for the dashboard."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM inspections")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM inspections WHERE detected_material = 'METAL'")
        metal_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM inspections WHERE crack_result = 'CRACK DETECTED'")
        crack_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM inspections WHERE crack_result = 'NO VISIBLE CRACK DETECTED'")
        no_crack_count = cursor.fetchone()[0]

        return {
            "total_inspections": total,
            "metal_inspections": metal_count,
            "cracks_detected": crack_count,
            "normal_surfaces": no_crack_count
        }

