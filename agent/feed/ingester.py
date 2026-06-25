"""
WRAITH Feed Ingester — Main Orchestrator for CVE/Exploit Ingestion

Coordinates the full pipeline:
    1. Fetch from sources (NVD, ExploitDB, Nuclei)
    2. Send to LLM for conversion (CVEConverter)
    3. Write skill documents to disk
    4. Index into ChromaDB + SQLite

Designed to run on a configurable schedule (cron or manual trigger).
All operations are idempotent — re-running skips already-indexed CVEs.

Usage:
    ingester = FeedIngester(
        llm_client=llm,
        longterm_memory=ltm,
        db_manager=db,
    )
    stats = await ingester.run()
    print(f"Ingested {stats['total_converted']} new skills")
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from feed.converter import CVEConverter
from feed.sources.nvd import NVDSource
from feed.sources.exploitdb import ExploitDBSource
from feed.sources.nuclei import NucleiSource
from skills.indexer import SkillIndexer
from utils.logger import get_logger

logger = get_logger("feed.ingester")


class FeedIngester:
    """
    Main feed ingestion orchestrator.

    Fetches from all configured sources, converts to skills,
    and indexes them. Tracks what's been processed to avoid
    duplicate ingestion.

    Usage:
        ingester = FeedIngester(
            llm_client=llm,
            longterm_memory=ltm,
            db_manager=db,
            skills_dir="./data/skills",
        )

        # Manual run
        stats = await ingester.run()

        # Run specific source only
        stats = await ingester.run(sources=["nvd"])
    """

    def __init__(
        self,
        llm_client: Any,
        longterm_memory: Any,
        db_manager: Any,
        skills_dir: str = "./data/skills",
        nvd_api_key: str = "",
        github_token: str = "",
    ) -> None:
        self._llm = llm_client
        self._ltm = longterm_memory
        self._db = db_manager
        self._skills_dir = skills_dir

        # Initialize sources
        self._nvd = NVDSource(api_key=nvd_api_key)
        self._exploitdb = ExploitDBSource()
        self._nuclei = NucleiSource(github_token=github_token)

        # Initialize converter
        self._indexer = SkillIndexer(
            skills_dir=skills_dir,
            longterm_memory=longterm_memory,
            db_manager=db_manager,
        )
        self._converter = CVEConverter(
            llm_client=llm_client,
            indexer=self._indexer,
            skills_dir=skills_dir,
        )

        logger.info("FeedIngester initialized")

    async def run(
        self,
        sources: list[str] | None = None,
        hours: int = 24,
        max_per_source: int = 20,
    ) -> dict[str, Any]:
        """
        Run a feed ingestion cycle.

        Fetches from all configured sources (or specified subset),
        converts to skill documents, and indexes them.

        Args:
            sources: List of source names to fetch from.
                     Defaults to all: ["nvd", "exploitdb", "nuclei"]
            hours: Look-back window for NVD (in hours).
            max_per_source: Max items to process per source.

        Returns:
            Stats dict with counts and timing.
        """
        sources = sources or ["nvd", "exploitdb", "nuclei"]
        start = datetime.now()

        stats = {
            "start_time": start.isoformat(),
            "sources_queried": sources,
            "total_fetched": 0,
            "total_converted": 0,
            "total_failed": 0,
            "by_source": {},
        }

        logger.info(
            f"Feed ingestion started — "
            f"sources={sources}, look-back={hours}h"
        )

        # Process each source
        for source_name in sources:
            source_stats = await self._process_source(
                source_name=source_name,
                hours=hours,
                max_results=max_per_source,
            )
            stats["by_source"][source_name] = source_stats
            stats["total_fetched"] += source_stats.get("fetched", 0)
            stats["total_converted"] += source_stats.get("converted", 0)
            stats["total_failed"] += source_stats.get("failed", 0)

        elapsed = (datetime.now() - start).total_seconds()
        stats["duration_seconds"] = round(elapsed, 1)

        logger.info(
            f"Feed ingestion complete — "
            f"fetched={stats['total_fetched']}, "
            f"converted={stats['total_converted']}, "
            f"failed={stats['total_failed']}, "
            f"duration={elapsed:.1f}s"
        )

        return stats

    async def _process_source(
        self,
        source_name: str,
        hours: int,
        max_results: int,
    ) -> dict[str, int]:
        """Process a single feed source."""
        source_stats = {"fetched": 0, "converted": 0, "failed": 0}

        try:
            if source_name == "nvd":
                items = await self._nvd.fetch_recent(
                    hours=hours, max_results=max_results,
                )
                source_stats["fetched"] = len(items)
                for item in items:
                    try:
                        doc = await self._converter.convert_cve(item)
                        if doc:
                            source_stats["converted"] += 1
                        else:
                            source_stats["failed"] += 1
                    except Exception as e:
                        logger.error(f"CVE conversion error: {e}")
                        source_stats["failed"] += 1

            elif source_name == "exploitdb":
                items = await self._exploitdb.fetch_recent(
                    days=max(hours // 24, 1),
                    max_results=max_results,
                )
                source_stats["fetched"] = len(items)
                for item in items:
                    try:
                        doc = await self._converter.convert_exploit(item)
                        if doc:
                            source_stats["converted"] += 1
                        else:
                            source_stats["failed"] += 1
                    except Exception as e:
                        logger.error(f"Exploit conversion error: {e}")
                        source_stats["failed"] += 1

            elif source_name == "nuclei":
                items = await self._nuclei.fetch_recent(
                    days=max(hours // 24, 1),
                    max_results=max_results,
                )
                source_stats["fetched"] = len(items)
                for item in items:
                    try:
                        doc = await self._converter.convert_template(item)
                        if doc:
                            source_stats["converted"] += 1
                        else:
                            source_stats["failed"] += 1
                    except Exception as e:
                        logger.error(f"Template conversion error: {e}")
                        source_stats["failed"] += 1

            else:
                logger.warning(f"Unknown feed source: {source_name}")

        except Exception as e:
            logger.error(f"Source '{source_name}' failed: {e}")

        return source_stats
