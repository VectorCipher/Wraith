"""
WRAITH Memory Package (v2)

Three-tier memory system that gives WRAITH persistent, cross-scan intelligence.

Tiers:
    1. Long-Term (ChromaDB)  — semantic vector store for skills, techniques, CVE knowledge
    2. Episodic  (SQLite)    — per-target scan history and outcomes
    3. Working   (In-process) — current scan state (wraps existing ScanMemory)

Usage:
    from memory import MemoryManager

    mm = MemoryManager(db_manager=db, scan_id="wraith-abc123")
    await mm.initialize(target_host="example.com")

    # Retrieve relevant knowledge before planning
    skills = await mm.retrieve("SQL injection on PHP with ModSecurity")

    # Store knowledge after a scan
    await mm.store_episode(scan_id, target, summary)
    await mm.store_skill(skill_doc)
"""

from memory.manager import MemoryManager

__all__ = ["MemoryManager"]
