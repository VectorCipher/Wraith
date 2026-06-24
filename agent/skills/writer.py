"""
WRAITH Skill Writer — Post-Scan Knowledge Extraction

Runs as a post-scan pass after every completed scan. Takes the full scan
log and asks the LLM to extract the single most reusable technique
discovered, then writes it as a Markdown skill document.

This is how WRAITH compounds over time — every scan teaches it something,
and that knowledge is available for all future scans.

Pipeline:
    1. Receive scan log from MemoryManager
    2. Build extraction prompt (Prompt 4 from design doc)
    3. Send to LLM
    4. Parse LLM response into skill Markdown format
    5. Write .md file to ./data/skills/
    6. Index into ChromaDB + SQLite via SkillIndexer

Key design decision:
    The Skill Writer runs AFTER the scan, not during. During the scan,
    the LLM is focused on attack reasoning. Asking it to simultaneously
    write documentation degrades attack quality. The post-scan pass gives
    it a clean, complete log to work from and produces better skills.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from skills.indexer import SkillDocument, SkillIndexer
from utils.logger import get_logger

logger = get_logger("skills.writer")

# Template for the skill extraction prompt
_SKILL_EXTRACTION_PROMPT = """# POST-SCAN SKILL EXTRACTION

You are WRAITH's knowledge extraction engine. You have just completed a penetration test.
Your job is to extract the single most reusable, interesting technique discovered during
this scan and write it as a skill document.

## Scan Log
{scan_log}

## Instructions

Analyze the scan log and extract the single most valuable, reusable technique.
Focus on techniques that would help in future scans against similar targets.

Write the skill document in this EXACT format (including the --- delimiters):

---
skill_id: {skill_id}
created: {timestamp}
scan_id: {scan_id}
target_profile: [tech stack this applies to, e.g. "PHP 7.x + MySQL + Apache"]
attack_class: [e.g. SQL Injection, XSS, SSRF, SSTI, etc.]
confidence: [HIGH, MEDIUM, or LOW]
reuse_score: [0.0 to 1.0 — how reusable is this across targets]
tags: [comma-separated tags]
---

# [Descriptive Title of the Technique]

## What Was Discovered
[What vulnerability or technique was found]

## Payload
```
[Exact payload that worked, if applicable]
```

## Evidence
[How the vulnerability was confirmed — response codes, timing, content changes]

## Target Profile Match
- [List conditions when this technique is applicable]

## Reuse Instructions
1. [Step-by-step instructions for using this in future scans]

## Why This Works
[Technical explanation of why the target was vulnerable]

## Related CVEs
[Any related CVEs, or "None — technique, not CVE-specific"]

## Tags
[Same comma-separated tags from frontmatter]

IMPORTANT:
- If no interesting techniques were discovered, write a skill about what was tested
  and WHY it failed — this is also valuable knowledge (negative skills).
- Be specific — include exact payloads, exact response codes, exact timing.
- Focus on REUSABILITY — another scan should be able to use this skill directly.
"""

# Template for generating memory search queries
_MEMORY_QUERY_PROMPT = """# MEMORY RETRIEVAL QUERY GENERATION

You are WRAITH's memory system. Before starting a penetration test, you need to
search your knowledge base for relevant attack techniques.

## Target Information
Target URL: {target_url}
Detected Tech Stack: {tech_stack}

## Instructions

Generate 3-5 specific search queries to retrieve relevant attack knowledge from
WRAITH's memory store. Each query should be:
- Specific to the detected technology stack
- Focused on known vulnerability patterns for this stack
- Including any relevant WAF/protection bypass techniques

Format your response as a numbered list:
1. [query]
2. [query]
3. [query]
...

Be specific about the tech stack components. Generic queries like "web vulnerabilities"
are useless. Good queries mention specific technologies, frameworks, and attack classes.
"""

# Enhanced attack planning prompt with skill context
_ATTACK_PLAN_WITH_SKILLS_PROMPT = """# ATTACK PLANNING WITH PRIOR KNOWLEDGE

You are WRAITH, an autonomous penetration testing AI. You are planning your attack
strategy for a target, and you have access to knowledge from prior engagements.

## Prior Knowledge (Retrieved Skills)
{skill_context}

## Episodic Memory (Past Scans of This Target)
{episodic_context}

## Current Target
URL: {target_url}
Tech Stack: {tech_stack}
Endpoints Discovered: {endpoint_count}

## Discovered Endpoints
{endpoints}

## Instructions

Using your prior knowledge AND the current target information, create a prioritized
attack plan. For each attack:

1. **Endpoint**: Which endpoint to target
2. **Attack Type**: SQLi, XSS, SSRF, SSTI, etc.
3. **Rationale**: WHY this attack, especially citing any relevant skills or past experience
4. **Priority**: CRITICAL, HIGH, MEDIUM, or LOW
5. **Special Notes**: Any WAF bypass techniques or specific payloads to try first

Order attacks by priority (highest first). Cite specific skills when applicable.
If you have prior knowledge about this target from episodic memory, use it to
avoid repeating failed approaches.
"""


class SkillWriter:
    """
    Post-scan knowledge extraction engine.

    Takes a completed scan's data and uses the LLM to extract
    reusable attack knowledge, writing it as skill documents.

    Usage:
        writer = SkillWriter(
            llm_client=llm,
            indexer=indexer,
            skills_dir="./data/skills",
        )

        skill = await writer.extract_and_save(
            scan_id="wraith-abc123",
            scan_log=memory.export_for_report(),
        )
    """

    def __init__(
        self,
        llm_client: Any,
        indexer: SkillIndexer,
        skills_dir: str = "./data/skills",
    ) -> None:
        """
        Args:
            llm_client: LLMClient instance for LLM calls.
            indexer: SkillIndexer for indexing generated skills.
            skills_dir: Directory to write skill .md files.
        """
        self._llm = llm_client
        self._indexer = indexer
        self._skills_dir = Path(skills_dir)
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SkillWriter initialized — dir={skills_dir}")

    # ===================================================================
    # Main Entry Point
    # ===================================================================

    async def extract_and_save(
        self,
        scan_id: str,
        scan_log: dict[str, Any],
    ) -> SkillDocument | None:
        """
        Extract a skill from a completed scan and save it.

        Pipeline:
            1. Format scan log for the LLM
            2. Ask LLM to extract the most reusable technique
            3. Parse the LLM's response
            4. Write .md file to disk
            5. Index into ChromaDB + SQLite

        Args:
            scan_id: The completed scan's ID.
            scan_log: Scan data from memory.export_for_report().

        Returns:
            SkillDocument if extraction succeeded, None otherwise.
        """
        logger.info(f"Starting skill extraction for scan {scan_id}")

        # Step 1: Format scan log
        log_text = self._format_scan_log(scan_log)

        if len(log_text) < 100:
            logger.info("Scan log too short for skill extraction — skipping")
            return None

        # Step 2: Generate skill ID and build prompt
        skill_id = self._indexer.get_next_skill_id()
        timestamp = datetime.now().isoformat()

        prompt = _SKILL_EXTRACTION_PROMPT.format(
            scan_log=log_text,
            skill_id=skill_id,
            timestamp=timestamp,
            scan_id=scan_id,
        )

        # Step 3: Ask LLM
        try:
            response = await self._llm.generate(
                role="coding",
                prompt=prompt,
                temperature=0.2,  # Low temp for precise extraction
            )

            if not response.content or len(response.content.strip()) < 50:
                logger.warning("LLM returned empty/short skill extraction")
                return None

            skill_content = response.content.strip()

        except Exception as e:
            logger.error(f"LLM skill extraction failed: {e}")
            return None

        # Step 4: Parse the response
        doc = SkillIndexer.parse_content(skill_content)
        if not doc:
            logger.warning("Failed to parse LLM skill output")
            return None

        # Ensure required fields
        if not doc.skill_id:
            doc.skill_id = skill_id
        if not doc.scan_id:
            doc.scan_id = scan_id

        # Step 5: Write to disk
        file_path = self._skills_dir / f"{doc.skill_id}.md"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(skill_content)
            doc.file_path = str(file_path)
            logger.info(f"Skill written to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write skill file: {e}")
            return None

        # Step 6: Index
        self._indexer.index_document(doc)

        logger.info(
            f"Skill extraction complete — {doc.skill_id}: "
            f"{doc.attack_class} ({doc.confidence})"
        )
        return doc

    # ===================================================================
    # Prompt Builders (static, used by Orchestrator/PromptEngine)
    # ===================================================================

    @staticmethod
    def build_skill_extraction_prompt(
        scan_log: str,
        skill_id: str,
        scan_id: str,
    ) -> str:
        """Build the skill extraction prompt (Prompt 4)."""
        return _SKILL_EXTRACTION_PROMPT.format(
            scan_log=scan_log,
            skill_id=skill_id,
            timestamp=datetime.now().isoformat(),
            scan_id=scan_id,
        )

    @staticmethod
    def build_memory_query_prompt(
        target_url: str,
        tech_stack: str,
    ) -> str:
        """Build the memory query generation prompt (Prompt 1)."""
        return _MEMORY_QUERY_PROMPT.format(
            target_url=target_url,
            tech_stack=tech_stack,
        )

    @staticmethod
    def build_attack_plan_with_skills_prompt(
        skill_context: str,
        episodic_context: str,
        target_url: str,
        tech_stack: str,
        endpoint_count: int,
        endpoints: str,
    ) -> str:
        """Build the enhanced attack planning prompt (Prompt 2)."""
        return _ATTACK_PLAN_WITH_SKILLS_PROMPT.format(
            skill_context=skill_context or "No prior skills available.",
            episodic_context=episodic_context or "First engagement — no prior history.",
            target_url=target_url,
            tech_stack=tech_stack,
            endpoint_count=endpoint_count,
            endpoints=endpoints,
        )

    # ===================================================================
    # Internal Helpers
    # ===================================================================

    @staticmethod
    def _format_scan_log(scan_log: dict[str, Any]) -> str:
        """
        Format a scan log dict into a readable text for the LLM.

        Extracts the most relevant information and formats it
        as a structured text block.
        """
        parts = []

        # Target info
        target = scan_log.get("target", {})
        if target:
            parts.append(f"Target: {target.get('url', 'unknown')}")

        # Tech stack
        tech = scan_log.get("tech_stack", {})
        if tech:
            tech_parts = [
                f"{k}={v}" for k, v in tech.items()
                if v and k != "other"
            ]
            if tech_parts:
                parts.append(f"Tech Stack: {', '.join(tech_parts)}")

        # Summary stats
        summary = scan_log.get("summary", {})
        if summary:
            parts.append(f"\nScan Summary:")
            eps = summary.get("endpoints", {})
            if eps:
                parts.append(
                    f"  Endpoints: {eps.get('total', 0)} total, "
                    f"{eps.get('tested', 0)} tested"
                )
            attacks = summary.get("attacks", {})
            if attacks:
                parts.append(
                    f"  Attacks: {attacks.get('total', 0)} executed, "
                    f"{attacks.get('requests_sent', 0)} requests"
                )
            vulns = summary.get("vulnerabilities", {})
            if vulns:
                parts.append(
                    f"  Vulnerabilities: {vulns.get('total', 0)} found "
                    f"({vulns.get('critical', 0)} critical, "
                    f"{vulns.get('high', 0)} high, "
                    f"{vulns.get('medium', 0)} medium)"
                )

        # Endpoints tested
        endpoints = scan_log.get("endpoints", [])
        if endpoints:
            parts.append(f"\nEndpoints ({len(endpoints)}):")
            for ep in endpoints[:20]:  # Limit to 20 for context
                line = f"  {ep.get('method', 'GET')} {ep.get('path', '/')}"
                tested = ep.get("attacks_tested", [])
                if tested:
                    line += f" — tested: {', '.join(tested)}"
                params = ep.get("parameters", [])
                if params:
                    param_names = [p.get("name", "?") for p in params[:5]]
                    line += f" — params: {', '.join(param_names)}"
                parts.append(line)

        # Vulnerabilities found
        vulns_data = scan_log.get("vulnerabilities", [])
        if vulns_data:
            parts.append(f"\nConfirmed Vulnerabilities ({len(vulns_data)}):")
            for v in vulns_data:
                parts.append(
                    f"  [{v.get('severity', '?').upper()}] "
                    f"{v.get('vuln_type', '?')} — "
                    f"{v.get('method', '?')} {v.get('endpoint', '?')}"
                )
                desc = v.get("description", "")
                if desc:
                    parts.append(f"    {desc[:200]}")

        return "\n".join(parts)
