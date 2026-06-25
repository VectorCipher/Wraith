"""
WRAITH Feed Package (v2)

Background service that ingests CVEs, exploits, and Nuclei templates
from public sources, converts them into WRAITH skill documents via
LLM analysis, and indexes them for future retrieval.

Sources:
    - NVD (National Vulnerability Database) — REST API v2
    - ExploitDB — Public exploit database
    - Nuclei Templates — GitHub repository
"""

from feed.ingester import FeedIngester

__all__ = ["FeedIngester"]
