"""
WRAITH Episodic Memory — Per-Target Scan History

Stores structured records of past scans against specific targets.
When WRAITH encounters a target it has scanned before, episodic memory
provides the AI with historical context: what tech stack was detected,
which endpoints were found, what attacks succeeded or failed, and an
LLM-written plain-English summary of the previous engagement.

This is how WRAITH avoids repeating dead-end attack vectors and
builds on previous knowledge about a specific target.

Storage: SQLite `episodes` table (managed by DatabaseManager).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger("memory.episodic")


# ===================================================================
# Data Models
# ===================================================================

class EpisodeRecord(BaseModel):
    """
    A single episodic memory entry — one past scan of a target.

    Contains everything the AI needs to know about what happened
    last time it encountered this target.
    """

    id: int | None = None
    target_host: str
    scan_id: str
    tech_stack: dict[str, Any] = Field(default_factory=dict)
    endpoints: list[str] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

    def to_context_string(self) -> str:
        """Format this episode as a context string for LLM injection."""
        parts = [
            f"### Prior Scan: `{self.scan_id}` ({self.created_at.strftime('%Y-%m-%d %H:%M')})",
        ]

        # Tech stack
        if self.tech_stack:
            tech_parts = []
            for key in ["language", "framework", "database", "web_server", "waf"]:
                val = self.tech_stack.get(key)
                if val:
                    tech_parts.append(f"{key}={val}")
            if tech_parts:
                parts.append(f"- **Tech Stack**: {', '.join(tech_parts)}")

        # Endpoints
        if self.endpoints:
            shown = self.endpoints[:10]
            parts.append(f"- **Endpoints discovered** ({len(self.endpoints)} total): {', '.join(shown)}")
            if len(self.endpoints) > 10:
                parts.append(f"  ... and {len(self.endpoints) - 10} more")

        # Summary
        if self.summary:
            parts.append(f"- **Summary**: {self.summary}")

        return "\n".join(parts)


# ===================================================================
# EpisodicMemory
# ===================================================================

class EpisodicMemory:
    """
    SQLite-backed episodic memory for per-target scan history.

    This class does NOT manage its own database connection — it
    receives a DatabaseManager instance and uses it for all
    persistence operations.

    Usage:
        em = EpisodicMemory(db_manager)

        # Save after a scan
        em.save_episode(
            target_host="example.com",
            scan_id="wraith-abc123",
            tech_stack={"language": "PHP", "database": "MySQL"},
            endpoints=["/login", "/api/users", "/admin"],
            summary="Found SQLi on /login via POST parameter 'username'."
        )

        # Retrieve before a new scan
        history = em.get_episodes("example.com")
        for episode in history:
            print(episode.to_context_string())
    """

    def __init__(self, db_manager: Any) -> None:
        """
        Initialize episodic memory with a database manager.

        Args:
            db_manager: An instance of databases.db.DatabaseManager.
        """
        self._db = db_manager
        logger.info("EpisodicMemory initialized")

    # ===================================================================
    # Write Operations
    # ===================================================================

    def save_episode(
        self,
        target_host: str,
        scan_id: str,
        tech_stack: dict[str, Any] | None = None,
        endpoints: list[str] | None = None,
        summary: str = "",
    ) -> EpisodeRecord:
        """
        Save a new episodic memory entry for a target.

        Called at the end of a scan to persist what was learned.

        Args:
            target_host: The target's hostname (e.g., "example.com").
            scan_id: The scan ID that generated this episode.
            tech_stack: Detected technology stack as a dict.
            endpoints: List of discovered endpoint paths.
            summary: LLM-generated plain-English summary of the scan.

        Returns:
            The created EpisodeRecord.
        """
        tech_json = json.dumps(tech_stack or {})
        endpoints_json = json.dumps(endpoints or [])

        try:
            with self._db._get_conn() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO episodes (target_host, scan_id, tech_stack, endpoints, summary)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (target_host, scan_id, tech_json, endpoints_json, summary),
                )
                row_id = cursor.lastrowid

            record = EpisodeRecord(
                id=row_id,
                target_host=target_host,
                scan_id=scan_id,
                tech_stack=tech_stack or {},
                endpoints=endpoints or [],
                summary=summary,
            )

            logger.info(
                f"Episode saved — target={target_host}, scan={scan_id}, "
                f"endpoints={len(endpoints or [])}"
            )
            return record

        except Exception as e:
            logger.error(f"Failed to save episode: {e}")
            raise

    # ===================================================================
    # Read Operations
    # ===================================================================

    def get_episodes(
        self,
        target_host: str,
        limit: int = 10,
    ) -> list[EpisodeRecord]:
        """
        Retrieve all episodic memories for a target host.

        Returns episodes sorted by most recent first.

        Args:
            target_host: The target's hostname.
            limit: Maximum number of episodes to return.

        Returns:
            List of EpisodeRecord objects.
        """
        try:
            with self._db._get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT id, target_host, scan_id, tech_stack, endpoints, summary, created_at
                    FROM episodes
                    WHERE target_host = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (target_host, limit),
                ).fetchall()

            episodes = []
            for row in rows:
                try:
                    tech = json.loads(row["tech_stack"]) if row["tech_stack"] else {}
                except (json.JSONDecodeError, TypeError):
                    tech = {}

                try:
                    eps = json.loads(row["endpoints"]) if row["endpoints"] else []
                except (json.JSONDecodeError, TypeError):
                    eps = []

                episodes.append(EpisodeRecord(
                    id=row["id"],
                    target_host=row["target_host"],
                    scan_id=row["scan_id"],
                    tech_stack=tech,
                    endpoints=eps,
                    summary=row["summary"] or "",
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                ))

            logger.debug(
                f"Retrieved {len(episodes)} episodes for target={target_host}"
            )
            return episodes

        except Exception as e:
            logger.error(f"Failed to retrieve episodes for {target_host}: {e}")
            return []

    def get_episode_by_scan_id(self, scan_id: str) -> EpisodeRecord | None:
        """Retrieve a specific episode by scan ID."""
        try:
            with self._db._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM episodes WHERE scan_id = ?",
                    (scan_id,),
                ).fetchone()

            if not row:
                return None

            try:
                tech = json.loads(row["tech_stack"]) if row["tech_stack"] else {}
            except (json.JSONDecodeError, TypeError):
                tech = {}

            try:
                eps = json.loads(row["endpoints"]) if row["endpoints"] else []
            except (json.JSONDecodeError, TypeError):
                eps = []

            return EpisodeRecord(
                id=row["id"],
                target_host=row["target_host"],
                scan_id=row["scan_id"],
                tech_stack=tech,
                endpoints=eps,
                summary=row["summary"] or "",
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            )

        except Exception as e:
            logger.error(f"Failed to retrieve episode for scan {scan_id}: {e}")
            return None

    def has_prior_scans(self, target_host: str) -> bool:
        """Check if any prior scans exist for a target."""
        try:
            with self._db._get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM episodes WHERE target_host = ?",
                    (target_host,),
                ).fetchone()
            return (row["cnt"] or 0) > 0
        except Exception:
            return False

    def get_all_targets(self) -> list[str]:
        """Get a list of all unique target hosts with episodic memory."""
        try:
            with self._db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT target_host FROM episodes ORDER BY target_host"
                ).fetchall()
            return [row["target_host"] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get target list: {e}")
            return []

    # ===================================================================
    # Context Building
    # ===================================================================

    def build_episodic_context(
        self,
        target_host: str,
        max_episodes: int = 3,
        max_chars: int = 3000,
    ) -> str:
        """
        Build a formatted context string from episodic memory.

        This is injected into LLM prompts at scan start so the AI
        knows what happened in previous engagements against this target.

        Args:
            target_host: The target's hostname.
            max_episodes: Maximum number of past episodes to include.
            max_chars: Maximum total characters for the context string.

        Returns:
            Formatted context string, or empty string if no history.
        """
        episodes = self.get_episodes(target_host, limit=max_episodes)

        if not episodes:
            return ""

        parts = [
            f"## Prior Engagement History — {target_host}",
            f"WRAITH has scanned this target {len(episodes)} time(s) before.",
            "",
        ]

        char_count = sum(len(p) for p in parts)

        for episode in episodes:
            episode_str = episode.to_context_string()
            if char_count + len(episode_str) > max_chars:
                parts.append("... older episodes omitted for brevity")
                break
            parts.append(episode_str)
            parts.append("")
            char_count += len(episode_str) + 1

        return "\n".join(parts)
