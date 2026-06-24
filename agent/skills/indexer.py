"""
WRAITH Skill Indexer — Parse & Index Skill Markdown Documents

Parses skill Markdown files (with YAML frontmatter) and indexes them
into ChromaDB for semantic retrieval and into the SQLite `skills`
table for structured queries.

Skill document format:
    ---
    skill_id: wraith-skill-0042
    created: 2026-06-01T14:32:00Z
    scan_id: wraith-scan-0099
    target_profile: PHP 7.x + MySQL + Apache + ModSecurity WAF
    attack_class: SQL Injection
    confidence: HIGH
    reuse_score: 0.91
    tags: sqli, blind-sqli, waf-bypass, user-agent, php, modsecurity
    ---

    # Skill Title
    Body text...

The frontmatter is YAML between --- delimiters. The body is free-form
Markdown describing the technique, payloads, evidence, and reuse
instructions.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger("skills.indexer")

# Regex to extract YAML frontmatter from Markdown
_FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    re.DOTALL,
)


class SkillDocument:
    """
    Parsed representation of a skill Markdown file.

    Attributes:
        skill_id: Unique identifier.
        scan_id: The scan that generated this skill.
        target_profile: Tech stack description this skill applies to.
        attack_class: Vulnerability class (e.g., "SQL Injection").
        confidence: Confidence level (HIGH, MEDIUM, LOW).
        reuse_score: Float 0-1 indicating reusability.
        tags: List of searchable tags.
        title: Extracted from the first # heading.
        body: Full Markdown body text.
        file_path: Path to the source .md file.
        created: Creation timestamp.
    """

    def __init__(
        self,
        skill_id: str = "",
        scan_id: str = "",
        target_profile: str = "",
        attack_class: str = "",
        confidence: str = "MEDIUM",
        reuse_score: float = 0.5,
        tags: list[str] | None = None,
        title: str = "",
        body: str = "",
        file_path: str = "",
        created: str = "",
    ) -> None:
        self.skill_id = skill_id
        self.scan_id = scan_id
        self.target_profile = target_profile
        self.attack_class = attack_class
        self.confidence = confidence
        self.reuse_score = reuse_score
        self.tags = tags or []
        self.title = title
        self.body = body
        self.file_path = file_path
        self.created = created or datetime.now().isoformat()

    def to_metadata(self) -> dict[str, Any]:
        """Convert to a metadata dict for ChromaDB indexing."""
        return {
            "skill_id": self.skill_id,
            "scan_id": self.scan_id,
            "target_profile": self.target_profile,
            "attack_class": self.attack_class,
            "confidence": self.confidence,
            "reuse_score": self.reuse_score,
            "tags": ", ".join(self.tags),
            "type": "skill",
            "created": self.created,
        }

    def to_db_record(self) -> dict[str, Any]:
        """Convert to a dict for SQLite `skills` table insertion."""
        return {
            "skill_id": self.skill_id,
            "scan_id": self.scan_id,
            "attack_class": self.attack_class,
            "target_profile": self.target_profile,
            "confidence": self.confidence,
            "reuse_score": self.reuse_score,
            "tags": ", ".join(self.tags),
            "file_path": self.file_path,
        }

    def to_search_text(self) -> str:
        """
        Build the text string that gets embedded in ChromaDB.
        Combines title, attack class, target profile, and body
        for maximum semantic retrieval quality.
        """
        parts = []
        if self.title:
            parts.append(self.title)
        if self.attack_class:
            parts.append(f"Attack class: {self.attack_class}")
        if self.target_profile:
            parts.append(f"Target profile: {self.target_profile}")
        if self.body:
            # Truncate body to keep embedding focused
            body_preview = self.body[:2000]
            parts.append(body_preview)
        return "\n".join(parts)


class SkillIndexer:
    """
    Parses skill Markdown files and indexes them into ChromaDB
    and the SQLite skills table.

    Usage:
        indexer = SkillIndexer(
            skills_dir="./data/skills",
            longterm_memory=ltm,
            db_manager=db,
        )

        # Index a single file
        doc = indexer.index_file("./data/skills/wraith-skill-0042.md")

        # Rebuild entire index from disk
        count = indexer.reindex_all()
    """

    def __init__(
        self,
        skills_dir: str,
        longterm_memory: Any,
        db_manager: Any,
    ) -> None:
        """
        Args:
            skills_dir: Directory containing skill .md files.
            longterm_memory: LongTermMemory instance for ChromaDB.
            db_manager: DatabaseManager instance for SQLite.
        """
        self._skills_dir = Path(skills_dir)
        self._ltm = longterm_memory
        self._db = db_manager

        # Ensure directory exists
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SkillIndexer initialized — dir={skills_dir}")

    # ===================================================================
    # Parsing
    # ===================================================================

    @staticmethod
    def parse_file(file_path: str) -> SkillDocument | None:
        """
        Parse a skill Markdown file into a SkillDocument.

        Args:
            file_path: Absolute or relative path to the .md file.

        Returns:
            SkillDocument if parsing succeeds, None otherwise.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read skill file {file_path}: {e}")
            return None

        return SkillIndexer.parse_content(content, file_path=file_path)

    @staticmethod
    def parse_content(content: str, file_path: str = "") -> SkillDocument | None:
        """
        Parse skill Markdown content (with frontmatter) into a SkillDocument.

        Args:
            content: Raw Markdown string with YAML frontmatter.
            file_path: Optional path for metadata tracking.

        Returns:
            SkillDocument if parsing succeeds, None otherwise.
        """
        match = _FRONTMATTER_PATTERN.match(content.strip())
        if not match:
            logger.warning(
                f"No frontmatter found in skill document: {file_path or 'inline'}"
            )
            # Still try to parse as a body-only document
            return SkillDocument(
                body=content.strip(),
                file_path=file_path,
            )

        frontmatter_raw = match.group(1)
        body = match.group(2).strip()

        # Parse YAML frontmatter manually (avoid PyYAML dependency for this)
        meta = _parse_simple_yaml(frontmatter_raw)

        # Extract title from first # heading in body
        title = ""
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break

        # Parse tags
        tags_raw = meta.get("tags", "")
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []

        # Parse reuse_score
        try:
            reuse_score = float(meta.get("reuse_score", 0.5))
        except (ValueError, TypeError):
            reuse_score = 0.5

        return SkillDocument(
            skill_id=meta.get("skill_id", ""),
            scan_id=meta.get("scan_id", ""),
            target_profile=meta.get("target_profile", ""),
            attack_class=meta.get("attack_class", ""),
            confidence=meta.get("confidence", "MEDIUM"),
            reuse_score=reuse_score,
            tags=tags,
            title=title,
            body=body,
            file_path=file_path,
            created=meta.get("created", ""),
        )

    # ===================================================================
    # Indexing
    # ===================================================================

    def index_file(self, file_path: str) -> SkillDocument | None:
        """
        Parse and index a single skill file.

        Adds the document to both ChromaDB (for semantic search)
        and the SQLite skills table (for structured queries).

        Args:
            file_path: Path to the .md file.

        Returns:
            SkillDocument if successful, None otherwise.
        """
        doc = self.parse_file(file_path)
        if not doc or not doc.skill_id:
            logger.warning(f"Skipping invalid skill file: {file_path}")
            return None

        # Index into ChromaDB
        search_text = doc.to_search_text()
        if search_text:
            self._ltm.add_document(
                doc_id=doc.skill_id,
                text=search_text,
                metadata=doc.to_metadata(),
            )

        # Index into SQLite
        try:
            db_record = doc.to_db_record()
            with self._db._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO skills
                    (skill_id, scan_id, attack_class, target_profile,
                     confidence, reuse_score, tags, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_record["skill_id"],
                        db_record["scan_id"],
                        db_record["attack_class"],
                        db_record["target_profile"],
                        db_record["confidence"],
                        db_record["reuse_score"],
                        db_record["tags"],
                        db_record["file_path"],
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to save skill to SQLite: {e}")

        logger.info(
            f"Skill indexed: {doc.skill_id} — "
            f"{doc.attack_class} ({doc.confidence})"
        )
        return doc

    def index_document(self, doc: SkillDocument) -> None:
        """
        Index a pre-parsed SkillDocument (used by SkillWriter after
        generating a skill without saving to a file first).
        """
        if not doc.skill_id:
            logger.warning("Cannot index skill without skill_id")
            return

        # Index into ChromaDB
        search_text = doc.to_search_text()
        if search_text:
            self._ltm.add_document(
                doc_id=doc.skill_id,
                text=search_text,
                metadata=doc.to_metadata(),
            )

        # Index into SQLite
        try:
            db_record = doc.to_db_record()
            with self._db._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO skills
                    (skill_id, scan_id, attack_class, target_profile,
                     confidence, reuse_score, tags, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_record["skill_id"],
                        db_record["scan_id"],
                        db_record["attack_class"],
                        db_record["target_profile"],
                        db_record["confidence"],
                        db_record["reuse_score"],
                        db_record["tags"],
                        db_record["file_path"],
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to save skill to SQLite: {e}")

        logger.debug(f"Skill document indexed: {doc.skill_id}")

    def reindex_all(self) -> int:
        """
        Rebuild the entire skill index from .md files on disk.

        Scans the skills directory, parses every .md file, and
        re-indexes them into ChromaDB and SQLite.

        Returns:
            Number of skills successfully indexed.
        """
        if not self._skills_dir.exists():
            logger.warning(f"Skills directory does not exist: {self._skills_dir}")
            return 0

        md_files = list(self._skills_dir.glob("*.md"))
        if not md_files:
            logger.info("No skill files found to index")
            return 0

        logger.info(f"Re-indexing {len(md_files)} skill files...")

        indexed = 0
        for md_file in md_files:
            doc = self.index_file(str(md_file))
            if doc:
                indexed += 1

        logger.info(f"Re-index complete: {indexed}/{len(md_files)} skills indexed")
        return indexed

    def get_next_skill_id(self) -> str:
        """
        Generate the next sequential skill ID.

        Scans existing files to find the highest number and increments.
        Format: wraith-skill-XXXX
        """
        existing = list(self._skills_dir.glob("wraith-skill-*.md"))
        max_num = 0

        for f in existing:
            try:
                # Extract number from filename: wraith-skill-0042.md -> 42
                name = f.stem  # wraith-skill-0042
                num_str = name.split("-")[-1]
                num = int(num_str)
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue

        return f"wraith-skill-{max_num + 1:04d}"


def _parse_simple_yaml(raw: str) -> dict[str, str]:
    """
    Minimal YAML parser for skill frontmatter.

    Only handles flat key: value pairs (no nesting, no lists).
    This avoids requiring PyYAML for such a simple use case,
    though PyYAML is already in requirements.txt if needed.
    """
    result = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            # Strip surrounding quotes
            if value and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            result[key] = value
    return result
