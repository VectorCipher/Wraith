"""
WRAITH Working Memory — Current Scan State Adapter

Wraps the existing ScanMemory class (core/memory.py) with v2 extensions:
    - to_episode(): Serializes current scan state into an episodic record
    - inject_skills(): Prepends retrieved skill context into LLM prompts
    - skill_context: Holds retrieved skills for the current scan session

The existing ScanMemory is excellent and handles all within-scan state
management. This adapter adds the bridge between ScanMemory and the
new persistent memory tiers (Long-Term and Episodic) without modifying
the original class.

Design principle: Composition over inheritance. We wrap ScanMemory
rather than subclassing it to keep the original module untouched.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from core.memory import ScanMemory
from models.scan import ScanConfig
from utils.logger import get_logger

logger = get_logger("memory.working")


class WorkingMemory:
    """
    Adapter that wraps ScanMemory with v2 memory capabilities.

    Provides all existing ScanMemory functionality via delegation,
    plus new methods for cross-tier memory integration.

    Usage:
        wm = WorkingMemory(scan_id="wraith-abc123", config=scan_config)

        # Use exactly like ScanMemory
        wm.scan_memory.set_target(target)
        wm.scan_memory.add_endpoint(endpoint)

        # v2 additions
        wm.inject_skills(retrieved_skills)
        episode = wm.to_episode()
    """

    def __init__(self, scan_id: str, config: ScanConfig) -> None:
        """
        Initialize working memory.

        Args:
            scan_id: Unique scan identifier.
            config: Scan configuration.
        """
        # The original v1 memory system — all existing behavior preserved
        self._scan_memory = ScanMemory(scan_id=scan_id, config=config)

        # v2 additions
        self._retrieved_skills: list[dict[str, Any]] = []
        self._episodic_context: str = ""
        self._memory_queries: list[str] = []

        logger.info(f"WorkingMemory initialized — scan_id={scan_id}")

    # ===================================================================
    # Properties
    # ===================================================================

    @property
    def scan_memory(self) -> ScanMemory:
        """Access the underlying ScanMemory instance directly."""
        return self._scan_memory

    @property
    def retrieved_skills(self) -> list[dict[str, Any]]:
        """Skills retrieved from long-term memory for this scan."""
        return self._retrieved_skills

    @property
    def episodic_context(self) -> str:
        """Episodic context string loaded at scan start."""
        return self._episodic_context

    @property
    def has_prior_knowledge(self) -> bool:
        """Whether any prior knowledge was loaded for this scan."""
        return bool(self._retrieved_skills) or bool(self._episodic_context)

    # ===================================================================
    # Skill Injection
    # ===================================================================

    def inject_skills(self, skills: list[dict[str, Any]]) -> None:
        """
        Store retrieved skills in working memory for this scan session.

        These skills were retrieved from ChromaDB's long-term memory
        based on the target's tech stack and are used to enrich LLM
        prompts during attack planning.

        Args:
            skills: List of skill dicts from LongTermMemory.search().
                    Each dict contains: id, text, metadata, distance.
        """
        self._retrieved_skills = skills
        logger.info(
            f"Injected {len(skills)} skills into working memory "
            f"(best match distance: {skills[0]['distance']:.3f})"
            if skills else
            "No skills to inject"
        )

    def inject_episodic_context(self, context: str) -> None:
        """
        Store episodic context string for this scan session.

        This context string is prepended to LLM prompts to give
        the AI awareness of prior scans against this target.

        Args:
            context: Formatted context string from EpisodicMemory.
        """
        self._episodic_context = context
        if context:
            logger.info(
                f"Episodic context injected ({len(context)} chars)"
            )

    def build_skill_context(self, max_chars: int = 4000) -> str:
        """
        Build a formatted context string from retrieved skills.

        Used to inject skill knowledge into LLM prompts during
        attack planning.

        Args:
            max_chars: Maximum characters for the context string.

        Returns:
            Formatted string with skill summaries.
        """
        if not self._retrieved_skills:
            return ""

        parts = [
            f"## Retrieved Attack Knowledge ({len(self._retrieved_skills)} skills)",
            "The following techniques were retrieved from WRAITH's memory based on the target profile:",
            "",
        ]

        char_count = sum(len(p) for p in parts)

        for i, skill in enumerate(self._retrieved_skills, 1):
            meta = skill.get("metadata", {})
            text = skill.get("text", "")

            header = f"### Skill {i}: {meta.get('attack_class', 'unknown').upper()}"
            confidence = f"Confidence: {meta.get('confidence', 'unknown')}"
            profile = f"Target Profile: {meta.get('target_profile', 'any')}"

            # Truncate skill text if too long
            if len(text) > 500:
                text = text[:500] + "..."

            skill_block = f"{header}\n- {confidence}\n- {profile}\n{text}\n"

            if char_count + len(skill_block) > max_chars:
                parts.append(f"... {len(self._retrieved_skills) - i + 1} more skills omitted")
                break

            parts.append(skill_block)
            char_count += len(skill_block)

        return "\n".join(parts)

    def build_enhanced_context(self, max_chars: int = 16000) -> str:
        """
        Build the complete v2 context string for LLM prompts.

        Combines: episodic context + skill context + standard ScanMemory context.
        This is the "supercharged" version of ScanMemory.build_full_context().

        Args:
            max_chars: Maximum total characters.

        Returns:
            Complete context string with all memory tiers.
        """
        sections = []

        # Episodic context first (what we know about this target)
        if self._episodic_context:
            sections.append(self._episodic_context)

        # Skill context (relevant attack techniques)
        skill_ctx = self.build_skill_context()
        if skill_ctx:
            sections.append(skill_ctx)

        # Standard ScanMemory context (current scan state)
        scan_ctx = self._scan_memory.build_full_context()
        sections.append(scan_ctx)

        full = "\n\n".join(sections)

        if len(full) > max_chars:
            full = full[:max_chars - 50] + "\n\n... [context truncated]"

        return full

    # ===================================================================
    # Episode Serialization
    # ===================================================================

    def to_episode(self) -> dict[str, Any]:
        """
        Serialize the current scan state into an episodic memory record.

        Called at the end of a scan to persist what was learned
        into the episodic memory tier.

        Returns:
            Dict ready to pass to EpisodicMemory.save_episode().
        """
        sm = self._scan_memory
        target = sm.target

        # Extract hostname from target URL
        target_host = ""
        if target and target.url:
            parsed = urlparse(target.url)
            target_host = parsed.hostname or parsed.netloc or target.url

        # Serialize tech stack
        ts = sm.tech_stack
        tech_stack = {
            "language": ts.language,
            "framework": ts.framework,
            "database": ts.database,
            "web_server": ts.web_server,
            "template_engine": ts.template_engine,
        }
        # Remove None values
        tech_stack = {k: v for k, v in tech_stack.items() if v}

        # Serialize endpoint paths
        endpoints = [
            entry.endpoint.path
            for entry in sm.get_all_endpoints()
        ]

        # Build a summary string
        summary_parts = []
        vulns = sm.get_vulnerabilities()
        if vulns:
            vuln_strs = [
                f"[{v.severity.value.upper()}] {v.vuln_type.value} on {v.method} {v.endpoint}"
                for v in vulns
            ]
            summary_parts.append(f"Found {len(vulns)} vulnerabilities: {'; '.join(vuln_strs)}")
        else:
            summary_parts.append("No vulnerabilities confirmed.")

        summary_parts.append(
            f"Tested {sum(1 for e in sm.get_all_endpoints() if e.attack_types_tested)}"
            f"/{sm.endpoint_count} endpoints with {sm.attack_count} attacks."
        )

        return {
            "target_host": target_host,
            "scan_id": sm.scan_id,
            "tech_stack": tech_stack,
            "endpoints": endpoints,
            "summary": " ".join(summary_parts),
        }
