"""
WRAITH CVE Converter — Convert Raw CVEs/Exploits to Skill Documents

Takes a raw CVE, exploit, or Nuclei template and sends it to the
LLM for analysis. The LLM generates a WRAITH skill document that
captures the vulnerability class, affected tech stack, attack
playbook, and detection fingerprint.

This is how WRAITH's knowledge base grows even between scans.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from skills.indexer import SkillDocument, SkillIndexer
from utils.logger import get_logger

logger = get_logger("feed.converter")

_CVE_CONVERSION_PROMPT = """# CVE → WRAITH SKILL CONVERSION

You are WRAITH's knowledge ingestion engine. Convert the following
vulnerability data into a reusable WRAITH skill document.

## Raw Vulnerability Data
{vuln_data}

## Instructions

Analyze the vulnerability and generate a WRAITH skill document.
Focus on:
1. HOW to detect this vulnerability class in a target
2. WHAT payloads or techniques to use
3. WHICH technology stacks are affected
4. HOW to confirm exploitation

Write the skill document in this EXACT format:

---
skill_id: {skill_id}
created: {timestamp}
scan_id: feed-ingestion
target_profile: [affected technology stack]
attack_class: [vulnerability class — e.g., SQL Injection, RCE, SSRF]
confidence: [HIGH if well-documented with PoC, MEDIUM if theoretical, LOW if vague]
reuse_score: [0.0 to 1.0 — based on how broadly applicable this is]
tags: [comma-separated relevant tags]
---

# [Descriptive technique title]

## Vulnerability Summary
[What this vulnerability is and why it matters]

## Detection
[How to detect if a target is vulnerable]

## Exploitation
```
[Exact payloads or techniques to exploit]
```

## Target Profile Match
- [Technology/version conditions]

## Mitigation
[How targets should fix this]

## References
- [Source links]

Be specific and actionable. Vague skills are useless.
"""


class CVEConverter:
    """
    Converts raw CVEs, exploits, and templates into WRAITH skill documents.

    Usage:
        converter = CVEConverter(llm_client=llm, indexer=indexer)

        skill = await converter.convert_cve(cve_record)
        skill = await converter.convert_exploit(exploit_record)
        skill = await converter.convert_template(nuclei_template)
    """

    def __init__(
        self,
        llm_client: Any,
        indexer: SkillIndexer,
        skills_dir: str = "./data/skills",
    ) -> None:
        self._llm = llm_client
        self._indexer = indexer
        self._skills_dir = skills_dir

        logger.info("CVEConverter initialized")

    async def convert_cve(self, cve_record: Any) -> SkillDocument | None:
        """Convert an NVD CVE record into a skill document."""
        return await self._convert(
            vuln_data=cve_record.to_text(),
            source_id=cve_record.cve_id,
        )

    async def convert_exploit(self, exploit_record: Any) -> SkillDocument | None:
        """Convert an ExploitDB record into a skill document."""
        return await self._convert(
            vuln_data=exploit_record.to_text(),
            source_id=f"EDB-{exploit_record.exploit_id}",
        )

    async def convert_template(self, template: Any) -> SkillDocument | None:
        """Convert a Nuclei template into a skill document."""
        return await self._convert(
            vuln_data=template.to_text(),
            source_id=f"NUCLEI-{template.template_id}",
        )

    async def _convert(
        self,
        vuln_data: str,
        source_id: str,
    ) -> SkillDocument | None:
        """
        Internal: Send vulnerability data to LLM and parse the result.
        """
        skill_id = self._indexer.get_next_skill_id()
        timestamp = datetime.now().isoformat()

        prompt = _CVE_CONVERSION_PROMPT.format(
            vuln_data=vuln_data,
            skill_id=skill_id,
            timestamp=timestamp,
        )

        try:
            response = await self._llm.generate(
                role="coding",
                prompt=prompt,
                temperature=0.2,
            )

            if not response.content or len(response.content.strip()) < 50:
                logger.warning(f"LLM returned empty conversion for {source_id}")
                return None

            skill_content = response.content.strip()

        except Exception as e:
            logger.error(f"LLM conversion failed for {source_id}: {e}")
            return None

        # Parse and index
        doc = SkillIndexer.parse_content(skill_content)
        if not doc:
            logger.warning(f"Failed to parse LLM output for {source_id}")
            return None

        if not doc.skill_id:
            doc.skill_id = skill_id

        # Write to disk
        from pathlib import Path
        file_path = Path(self._skills_dir) / f"{doc.skill_id}.md"
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(skill_content)
            doc.file_path = str(file_path)
        except Exception as e:
            logger.error(f"Failed to write skill file for {source_id}: {e}")
            return None

        # Index into ChromaDB + SQLite
        self._indexer.index_document(doc)

        logger.info(
            f"Converted {source_id} → {doc.skill_id} "
            f"({doc.attack_class}, {doc.confidence})"
        )
        return doc
