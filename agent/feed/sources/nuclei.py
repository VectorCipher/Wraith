"""
WRAITH Nuclei Templates Source — Monitor for New Templates

Monitors the Nuclei Templates GitHub repository for newly added
YAML templates. Each template defines a specific vulnerability
check that WRAITH can learn from.

Source: https://github.com/projectdiscovery/nuclei-templates
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

logger = get_logger("feed.sources.nuclei")

_NUCLEI_API_BASE = "https://api.github.com/repos/projectdiscovery/nuclei-templates"


class NucleiTemplate:
    """Parsed Nuclei template metadata."""

    def __init__(
        self,
        template_id: str,
        name: str = "",
        severity: str = "",
        description: str = "",
        tags: list[str] | None = None,
        cve_id: str = "",
        reference: list[str] | None = None,
        file_path: str = "",
    ) -> None:
        self.template_id = template_id
        self.name = name
        self.severity = severity
        self.description = description
        self.tags = tags or []
        self.cve_id = cve_id
        self.reference = reference or []
        self.file_path = file_path

    def to_text(self) -> str:
        """Format as text for LLM processing."""
        parts = [
            f"Nuclei Template: {self.template_id}",
            f"Name: {self.name}",
            f"Severity: {self.severity}",
            f"Description: {self.description}",
        ]
        if self.cve_id:
            parts.append(f"CVE: {self.cve_id}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(parts)


class NucleiSource:
    """
    Fetches new Nuclei templates from GitHub.

    Uses the GitHub API to list recently committed YAML files
    in the nuclei-templates repository.

    Usage:
        source = NucleiSource()
        templates = await source.fetch_recent(days=7)
    """

    def __init__(self, github_token: str = "") -> None:
        self._token = github_token
        logger.info("Nuclei templates source initialized")

    async def fetch_recent(
        self,
        days: int = 7,
        max_results: int = 30,
    ) -> list[NucleiTemplate]:
        """
        Fetch recently added Nuclei templates.

        Uses the GitHub Commits API to find new .yaml files.

        Args:
            days: Look-back window in days.
            max_results: Maximum templates to return.

        Returns:
            List of NucleiTemplate objects.
        """
        try:
            import httpx
            from datetime import datetime, timedelta
        except ImportError:
            logger.error("httpx not installed — cannot fetch Nuclei templates")
            return []

        since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        headers = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get recent commits
                response = await client.get(
                    f"{_NUCLEI_API_BASE}/commits",
                    params={"since": since, "per_page": 50},
                    headers=headers,
                )
                response.raise_for_status()
                commits = response.json()

            templates = []
            seen_files = set()

            for commit in commits[:30]:  # Limit commit processing
                sha = commit.get("sha", "")
                if not sha:
                    continue

                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        detail_resp = await client.get(
                            f"{_NUCLEI_API_BASE}/commits/{sha}",
                            headers=headers,
                        )
                        detail_resp.raise_for_status()
                        detail = detail_resp.json()

                    for file_info in detail.get("files", []):
                        filename = file_info.get("filename", "")
                        status = file_info.get("status", "")

                        if (
                            filename.endswith(".yaml")
                            and status in ("added", "modified")
                            and filename not in seen_files
                        ):
                            seen_files.add(filename)
                            template = self._parse_file_entry(filename, file_info)
                            if template:
                                templates.append(template)

                        if len(templates) >= max_results:
                            break
                except Exception:
                    continue

                if len(templates) >= max_results:
                    break

            logger.info(
                f"Fetched {len(templates)} Nuclei templates (last {days}d)"
            )
            return templates

        except Exception as e:
            logger.error(f"Nuclei fetch failed: {e}")
            return []

    def _parse_file_entry(
        self,
        filename: str,
        file_info: dict[str, Any],
    ) -> NucleiTemplate | None:
        """Parse a file entry from the GitHub API into a NucleiTemplate."""
        # Extract template ID from filename
        # e.g., "cves/2024/CVE-2024-1234.yaml" → "CVE-2024-1234"
        parts = filename.replace(".yaml", "").split("/")
        template_id = parts[-1] if parts else filename

        # Try to parse the patch content for metadata
        patch = file_info.get("patch", "")
        name = template_id
        severity = ""
        description = ""
        tags = []
        cve_id = ""

        if patch:
            for line in patch.split("\n"):
                line = line.strip().lstrip("+").strip()
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("severity:"):
                    severity = line.split(":", 1)[1].strip()
                elif line.startswith("description:"):
                    description = line.split(":", 1)[1].strip()
                elif line.startswith("tags:"):
                    tags = [t.strip() for t in line.split(":", 1)[1].split(",")]
                elif "CVE-" in line:
                    import re
                    cve_match = re.search(r"CVE-\d{4}-\d+", line)
                    if cve_match:
                        cve_id = cve_match.group()

        return NucleiTemplate(
            template_id=template_id,
            name=name,
            severity=severity,
            description=description,
            tags=tags,
            cve_id=cve_id,
            file_path=filename,
        )
