"""
WRAITH NVD Source — Fetch CVEs from the National Vulnerability Database

Uses the NVD REST API v2 to fetch recently published CVEs.
Supports optional API key for higher rate limits.

API docs: https://nvd.nist.gov/developers/vulnerabilities
Rate limits:
    - Without key: 5 requests per 30 seconds
    - With key:    50 requests per 30 seconds
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from utils.logger import get_logger

logger = get_logger("feed.sources.nvd")

_NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"


class CVERecord:
    """Parsed CVE record from NVD."""

    def __init__(
        self,
        cve_id: str,
        description: str = "",
        severity: str = "",
        cvss_score: float = 0.0,
        published: str = "",
        affected_products: list[str] | None = None,
        references: list[str] | None = None,
        cwe_ids: list[str] | None = None,
    ) -> None:
        self.cve_id = cve_id
        self.description = description
        self.severity = severity
        self.cvss_score = cvss_score
        self.published = published
        self.affected_products = affected_products or []
        self.references = references or []
        self.cwe_ids = cwe_ids or []

    def to_text(self) -> str:
        """Format as text for LLM processing."""
        parts = [
            f"CVE: {self.cve_id}",
            f"Severity: {self.severity} (CVSS: {self.cvss_score})",
            f"Published: {self.published}",
            f"Description: {self.description}",
        ]
        if self.affected_products:
            parts.append(f"Affected: {', '.join(self.affected_products[:10])}")
        if self.cwe_ids:
            parts.append(f"CWE: {', '.join(self.cwe_ids)}")
        if self.references:
            parts.append(f"References: {', '.join(self.references[:5])}")
        return "\n".join(parts)


class NVDSource:
    """
    Fetches recent CVEs from the NVD REST API.

    Usage:
        source = NVDSource(api_key="your-key-here")
        cves = await source.fetch_recent(hours=24)
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        logger.info(
            f"NVD source initialized — "
            f"API key: {'configured' if api_key else 'none (rate-limited)'}"
        )

    async def fetch_recent(
        self,
        hours: int = 24,
        max_results: int = 50,
    ) -> list[CVERecord]:
        """
        Fetch CVEs published in the last N hours.

        Args:
            hours: Look-back window in hours.
            max_results: Maximum CVEs to return.

        Returns:
            List of CVERecord objects.
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed — cannot fetch NVD data")
            return []

        now = datetime.utcnow()
        start = now - timedelta(hours=hours)

        params = {
            "pubStartDate": start.strftime("%Y-%m-%dT%H:%M:%S.000"),
            "pubEndDate": now.strftime("%Y-%m-%dT%H:%M:%S.000"),
            "resultsPerPage": min(max_results, 100),
        }

        headers = {}
        if self._api_key:
            headers["apiKey"] = self._api_key

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    _NVD_API_BASE,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            cves = self._parse_response(data)
            logger.info(f"Fetched {len(cves)} CVEs from NVD (last {hours}h)")
            return cves

        except Exception as e:
            logger.error(f"NVD fetch failed: {e}")
            return []

    def _parse_response(self, data: dict[str, Any]) -> list[CVERecord]:
        """Parse NVD API response into CVERecord objects."""
        records = []

        for vuln in data.get("vulnerabilities", []):
            cve = vuln.get("cve", {})
            cve_id = cve.get("id", "")

            # Description
            descriptions = cve.get("descriptions", [])
            description = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break

            # CVSS score and severity
            metrics = cve.get("metrics", {})
            cvss_score = 0.0
            severity = "UNKNOWN"

            # Try CVSS 3.1 first, then 3.0, then 2.0
            for version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                if version in metrics:
                    metric_list = metrics[version]
                    if metric_list:
                        cvss_data = metric_list[0].get("cvssData", {})
                        cvss_score = cvss_data.get("baseScore", 0.0)
                        severity = cvss_data.get("baseSeverity", "UNKNOWN")
                        break

            # Published date
            published = cve.get("published", "")

            # CWE IDs
            cwe_ids = []
            for weakness in cve.get("weaknesses", []):
                for desc in weakness.get("description", []):
                    value = desc.get("value", "")
                    if value.startswith("CWE-"):
                        cwe_ids.append(value)

            # References
            references = [
                ref.get("url", "")
                for ref in cve.get("references", [])[:10]
            ]

            # Affected products (CPE)
            products = []
            for config in cve.get("configurations", []):
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        criteria = match.get("criteria", "")
                        if criteria:
                            products.append(criteria)

            records.append(CVERecord(
                cve_id=cve_id,
                description=description,
                severity=severity,
                cvss_score=cvss_score,
                published=published,
                affected_products=products[:10],
                references=references,
                cwe_ids=cwe_ids,
            ))

        return records
