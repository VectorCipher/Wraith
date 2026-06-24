"""
WRAITH Skill Reader — Retrieve & Search Skill Documents

Provides semantic search (via ChromaDB) and structured lookup (via SQLite)
for skill documents. Used by the Memory Manager during scan initialization
to find relevant attack knowledge for the current target.

Usage:
    reader = SkillReader(longterm_memory=ltm, db_manager=db, skills_dir="./data/skills")

    # Semantic search
    results = reader.search("SQL injection WAF bypass", top_k=5)

    # Get specific skill
    skill = reader.get_by_id("wraith-skill-0042")

    # List all skills
    all_skills = reader.list_all()

    # Filter by attack class
    sqli_skills = reader.list_all(attack_class="SQL Injection")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skills.indexer import SkillDocument, SkillIndexer
from utils.logger import get_logger

logger = get_logger("skills.reader")


class SkillReader:
    """
    Read and search skill documents from ChromaDB and SQLite.

    Provides two search modes:
        1. Semantic (ChromaDB) — "find skills about WAF bypass on PHP"
        2. Structured (SQLite) — exact match on attack_class, tags, etc.

    The semantic search is the primary retrieval path used during scans.
    Structured queries are for the CLI (`wraith skills list/search`).
    """

    def __init__(
        self,
        longterm_memory: Any,
        db_manager: Any,
        skills_dir: str = "./data/skills",
    ) -> None:
        """
        Args:
            longterm_memory: LongTermMemory instance for semantic search.
            db_manager: DatabaseManager instance for structured queries.
            skills_dir: Directory containing skill .md files.
        """
        self._ltm = longterm_memory
        self._db = db_manager
        self._skills_dir = Path(skills_dir)

        logger.info(f"SkillReader initialized — dir={skills_dir}")

    # ===================================================================
    # Semantic Search (ChromaDB)
    # ===================================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        attack_class: str | None = None,
        min_confidence: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search for skills.

        This is the main retrieval path used during scans. Finds
        skills that are semantically similar to the query string.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.
            attack_class: Optional filter by attack class.
            min_confidence: Optional minimum confidence filter.

        Returns:
            List of result dicts with id, text, metadata, distance.
        """
        where_filter = {"type": "skill"}
        if attack_class:
            where_filter["attack_class"] = attack_class

        results = self._ltm.search(
            query=query,
            top_k=top_k,
            where=where_filter,
        )

        # Post-filter by confidence if requested
        if min_confidence and results:
            confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            min_level = confidence_order.get(min_confidence.upper(), 0)
            results = [
                r for r in results
                if confidence_order.get(
                    r.get("metadata", {}).get("confidence", "LOW").upper(), 0
                ) >= min_level
            ]

        logger.debug(
            f"Skill search '{query[:50]}...' → {len(results)} results"
        )
        return results

    # ===================================================================
    # Exact Lookup
    # ===================================================================

    def get_by_id(self, skill_id: str) -> SkillDocument | None:
        """
        Retrieve a specific skill document by ID.

        Looks up the file path from SQLite, then parses the
        Markdown file from disk.

        Args:
            skill_id: The unique skill identifier.

        Returns:
            SkillDocument if found, None otherwise.
        """
        try:
            with self._db._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM skills WHERE skill_id = ?",
                    (skill_id,),
                ).fetchone()

            if not row:
                # Try ChromaDB as fallback
                chroma_result = self._ltm.get_by_id(skill_id)
                if chroma_result:
                    return SkillDocument(
                        skill_id=skill_id,
                        body=chroma_result.get("text", ""),
                        **{
                            k: v for k, v in chroma_result.get("metadata", {}).items()
                            if k in ("attack_class", "target_profile", "confidence", "scan_id")
                        },
                    )
                return None

            # Parse the file from disk if it exists
            file_path = row["file_path"]
            if file_path and Path(file_path).exists():
                return SkillIndexer.parse_file(file_path)

            # Otherwise construct from SQLite data
            tags = row["tags"].split(", ") if row["tags"] else []
            return SkillDocument(
                skill_id=row["skill_id"],
                scan_id=row["scan_id"] or "",
                attack_class=row["attack_class"] or "",
                target_profile=row["target_profile"] or "",
                confidence=row["confidence"] or "MEDIUM",
                reuse_score=row["reuse_score"] or 0.5,
                tags=tags,
                file_path=file_path or "",
            )

        except Exception as e:
            logger.error(f"Failed to get skill {skill_id}: {e}")
            return None

    # ===================================================================
    # Structured Listing (SQLite)
    # ===================================================================

    def list_all(
        self,
        attack_class: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List all skills from the SQLite index.

        Args:
            attack_class: Optional filter by attack class.
            limit: Maximum number of results.

        Returns:
            List of skill record dicts.
        """
        try:
            with self._db._get_conn() as conn:
                if attack_class:
                    rows = conn.execute(
                        """
                        SELECT * FROM skills
                        WHERE attack_class = ?
                        ORDER BY reuse_score DESC
                        LIMIT ?
                        """,
                        (attack_class, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM skills
                        ORDER BY reuse_score DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to list skills: {e}")
            return []

    def count(self) -> int:
        """Return the total number of indexed skills."""
        try:
            with self._db._get_conn() as conn:
                row = conn.execute("SELECT COUNT(*) as cnt FROM skills").fetchone()
            return row["cnt"] or 0
        except Exception:
            return 0

    def search_by_tags(
        self,
        tags: list[str],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search skills by tag match (structured, not semantic).

        Args:
            tags: List of tags to search for.
            limit: Maximum results.

        Returns:
            List of matching skill records.
        """
        if not tags:
            return []

        try:
            # Build a LIKE query for each tag
            conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            params = [f"%{tag}%" for tag in tags]
            params.append(limit)

            with self._db._get_conn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT * FROM skills
                    WHERE {conditions}
                    ORDER BY reuse_score DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Tag search failed: {e}")
            return []

    # ===================================================================
    # Display Formatting
    # ===================================================================

    def format_skill_summary(self, skill: dict[str, Any]) -> str:
        """Format a skill record for CLI display."""
        parts = [
            f"  {skill.get('skill_id', '?')}",
            f"  Attack: {skill.get('attack_class', 'unknown')}",
            f"  Profile: {skill.get('target_profile', 'any')}",
            f"  Confidence: {skill.get('confidence', '?')} "
            f"(reuse: {skill.get('reuse_score', 0):.2f})",
        ]
        tags = skill.get("tags", "")
        if tags:
            parts.append(f"  Tags: {tags}")
        return "\n".join(parts)
