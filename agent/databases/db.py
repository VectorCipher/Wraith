"""
WRAITH Persistence Database.

Handles saving and loading scan state to a local SQLite database.
This ensures findings survive terminal restarts and can be used for reporting.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Any, List, Optional
from datetime import datetime

from config import settings
from utils.logger import get_logger
from models.vulnerability import Vulnerability
from models.target import Endpoint

logger = get_logger("databases.db")

class DatabaseManager:
    """Manages SQLite database connections and operations."""

    def __init__(self, db_path: Optional[str] = None):
        # Default to ~/.wraith/wraith.db if not provided
        if not db_path:
            home = Path.home()
            wraith_dir = home / ".wraith"
            wraith_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(wraith_dir / "wraith.db")
        else:
            self.db_path = db_path
            
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a configured database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema if it doesn't exist."""
        try:
            with self._get_conn() as conn:
                conn.executescript('''
                    CREATE TABLE IF NOT EXISTS scans (
                        scan_id TEXT PRIMARY KEY,
                        target_url TEXT NOT NULL,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        status TEXT,
                        config_json TEXT
                    );

                    CREATE TABLE IF NOT EXISTS endpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id TEXT,
                        method TEXT,
                        path TEXT,
                        data_json TEXT,
                        FOREIGN KEY (scan_id) REFERENCES scans (scan_id)
                    );

                    CREATE TABLE IF NOT EXISTS vulnerabilities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id TEXT,
                        severity TEXT,
                        vuln_type TEXT,
                        title TEXT,
                        endpoint TEXT,
                        data_json TEXT,
                        FOREIGN KEY (scan_id) REFERENCES scans (scan_id)
                    );
                ''')
            logger.debug(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    # ===================================================================
    # Scan Management
    # ===================================================================
    def save_scan(self, scan_id: str, target_url: str, status: str, config: dict = None):
        """Save or update a scan record."""
        config_str = json.dumps(config) if config else "{}"
        
        with self._get_conn() as conn:
            # Upsert
            conn.execute('''
                INSERT INTO scans (scan_id, target_url, start_time, status, config_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scan_id) DO UPDATE SET
                    status = excluded.status,
                    config_json = excluded.config_json
            ''', (scan_id, target_url, datetime.now().isoformat(), status, config_str))
            
    def update_scan_status(self, scan_id: str, status: str, complete: bool = False):
        """Update scan status and optionally set end_time."""
        with self._get_conn() as conn:
            if complete:
                conn.execute('''
                    UPDATE scans SET status = ?, end_time = ? WHERE scan_id = ?
                ''', (status, datetime.now().isoformat(), scan_id))
            else:
                conn.execute('''
                    UPDATE scans SET status = ? WHERE scan_id = ?
                ''', (status, scan_id))

    def get_scan(self, scan_id: str) -> Optional[dict]:
        """Retrieve a scan record."""
        with self._get_conn() as conn:
            row = conn.execute('SELECT * FROM scans WHERE scan_id = ?', (scan_id,)).fetchone()
            return dict(row) if row else None

    # ===================================================================
    # Data Persistence (Endpoints & Vulnerabilities)
    # ===================================================================
    def save_endpoint(self, scan_id: str, endpoint: Endpoint):
        """Save a discovered endpoint."""
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO endpoints (scan_id, method, path, data_json)
                VALUES (?, ?, ?, ?)
            ''', (scan_id, endpoint.method, endpoint.path, endpoint.model_dump_json()))

    def get_endpoints(self, scan_id: str) -> List[Endpoint]:
        """Retrieve all endpoints for a scan."""
        with self._get_conn() as conn:
            rows = conn.execute('SELECT data_json FROM endpoints WHERE scan_id = ?', (scan_id,)).fetchall()
            return [Endpoint.model_validate_json(row['data_json']) for row in rows]

    def save_vulnerability(self, scan_id: str, vuln: Vulnerability):
        """Save a confirmed vulnerability."""
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO vulnerabilities (scan_id, severity, vuln_type, title, endpoint, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (scan_id, vuln.severity.value, vuln.vuln_type.value, vuln.title, vuln.endpoint, vuln.model_dump_json()))

    def get_vulnerabilities(self, scan_id: str) -> List[Vulnerability]:
        """Retrieve all vulnerabilities for a scan."""
        with self._get_conn() as conn:
            rows = conn.execute('SELECT data_json FROM vulnerabilities WHERE scan_id = ? ORDER BY severity', (scan_id,)).fetchall()
            return [Vulnerability.model_validate_json(row['data_json']) for row in rows]
