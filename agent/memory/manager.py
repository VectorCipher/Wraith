"""
WRAITH Memory Manager — Unified Three-Tier Memory Interface

The central nervous system of WRAITH v2. Sits between every major
component and is consulted before any scan begins and written to
after every scan ends.

Three tiers:
    1. Long-Term  (ChromaDB)   — semantic knowledge store
    2. Episodic   (SQLite)     — per-target scan history
    3. Working    (In-process)  — current scan state

Key design principle:
    The Memory Manager abstracts storage completely. The AI Orchestrator
    never calls ChromaDB or SQLite directly — it calls:
        memory_manager.retrieve(query)
        memory_manager.store(entry)
    This makes the storage backend swappable.

Usage:
    from memory import MemoryManager

    mm = MemoryManager(db_manager=db, scan_id="wraith-abc123", config=scan_config)
    await mm.initialize(target_host="example.com")

    # Before attack planning
    skills = mm.retrieve_skills("SQL injection on PHP with ModSecurity")
    context = mm.build_full_context()

    # After scan completes
    mm.store_episode()
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from memory.longterm import LongTermMemory
from memory.episodic import EpisodicMemory, EpisodeRecord
from memory.working import WorkingMemory
from models.scan import ScanConfig
from models.target import Target, TechStack
from core.memory import ScanMemory
from utils.logger import get_logger

logger = get_logger("memory.manager")


class MemoryManager:
    """
    Unified interface to WRAITH's three-tier memory system.

    Coordinates Long-Term (ChromaDB), Episodic (SQLite), and Working
    (in-process) memory tiers. All memory operations go through this
    class — no component should access tiers directly.

    Lifecycle:
        1. __init__()        — Create manager with config
        2. initialize()      — Load prior knowledge for the target
        3. (scan runs)       — Working memory accumulates scan data
        4. store_episode()   — Persist what was learned
        5. (optional) store_skill()  — Index a new skill document

    Attributes:
        working: WorkingMemory instance (wraps ScanMemory)
        longterm: LongTermMemory instance (ChromaDB)
        episodic: EpisodicMemory instance (SQLite)
    """

    def __init__(
        self,
        db_manager: Any,
        scan_id: str,
        config: ScanConfig,
        chroma_path: str = "./data/chroma",
    ) -> None:
        """
        Initialize the Memory Manager.

        Args:
            db_manager: DatabaseManager instance for SQLite operations.
            scan_id: Unique scan identifier.
            config: Scan configuration.
            chroma_path: Directory for ChromaDB persistence.
        """
        self._scan_id = scan_id
        self._config = config
        self._target_host: str = ""

        # Initialize all three tiers
        self.working = WorkingMemory(scan_id=scan_id, config=config)
        self.longterm = LongTermMemory(persist_dir=chroma_path)
        self.episodic = EpisodicMemory(db_manager=db_manager)

        # Track initialization state
        self._initialized = False
        self._prior_episodes: list[EpisodeRecord] = []

        logger.info(
            f"MemoryManager created — scan_id={scan_id}, "
            f"chroma_path={chroma_path}"
        )

    # ===================================================================
    # Properties
    # ===================================================================

    @property
    def scan_memory(self) -> ScanMemory:
        """Direct access to the underlying ScanMemory for backward compat."""
        return self.working.scan_memory

    @property
    def scan_id(self) -> str:
        return self._scan_id

    @property
    def target_host(self) -> str:
        return self._target_host

    @property
    def has_prior_knowledge(self) -> bool:
        """Whether any prior knowledge was loaded for this scan."""
        return self.working.has_prior_knowledge

    @property
    def prior_episodes(self) -> list[EpisodeRecord]:
        """Episodes loaded from episodic memory at initialization."""
        return self._prior_episodes

    # ===================================================================
    # Initialization — Called at Scan Start
    # ===================================================================

    def initialize(
        self,
        target_url: str,
        max_episodes: int = 3,
    ) -> dict[str, Any]:
        """
        Load prior knowledge for the target.

        Called during Phase 1 (INIT) of the scan lifecycle. Queries
        episodic memory for past scans and returns a summary of
        what was found so the orchestrator can decide how to proceed.

        Args:
            target_url: The target URL being scanned.
            max_episodes: Maximum number of past episodes to load.

        Returns:
            Dict with initialization results:
                - has_history: bool
                - episode_count: int
                - episodic_context: str (formatted for LLM)
        """
        # Extract hostname from URL
        parsed = urlparse(target_url)
        self._target_host = parsed.hostname or parsed.netloc or target_url

        logger.info(f"Initializing memory for target: {self._target_host}")

        # Query episodic memory for prior scans
        self._prior_episodes = self.episodic.get_episodes(
            target_host=self._target_host,
            limit=max_episodes,
        )

        has_history = len(self._prior_episodes) > 0

        # Build and inject episodic context
        episodic_context = ""
        if has_history:
            episodic_context = self.episodic.build_episodic_context(
                target_host=self._target_host,
                max_episodes=max_episodes,
            )
            self.working.inject_episodic_context(episodic_context)

            logger.info(
                f"Loaded {len(self._prior_episodes)} prior episodes "
                f"for {self._target_host}"
            )
        else:
            logger.info(
                f"No prior scans found for {self._target_host} — "
                f"first engagement"
            )

        self._initialized = True

        return {
            "has_history": has_history,
            "episode_count": len(self._prior_episodes),
            "episodic_context": episodic_context,
        }

    # ===================================================================
    # Retrieval — Query Long-Term Memory
    # ===================================================================

    def retrieve_skills(
        self,
        query: str,
        top_k: int = 10,
        attack_class: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant skills from long-term memory.

        Uses semantic similarity search to find attack techniques,
        WAF bypass methods, and CVE knowledge relevant to the query.

        Args:
            query: Natural language query describing what to look for.
            top_k: Maximum number of results.
            attack_class: Optional filter by attack class (e.g., "sqli").

        Returns:
            List of skill dicts from ChromaDB, injected into working memory.
        """
        where_filter = None
        if attack_class:
            where_filter = {"attack_class": attack_class}

        try:
            results = self.longterm.search(
                query=query,
                top_k=top_k,
                where=where_filter,
            )

            # Inject into working memory for this scan session
            self.working.inject_skills(results)

            if results:
                logger.info(
                    f"Retrieved {len(results)} skills for query: "
                    f"'{query[:50]}...' "
                    f"(best distance: {results[0]['distance']:.3f})"
                )
            else:
                logger.info(f"No skills found for query: '{query[:50]}...'")

            return results

        except Exception as e:
            logger.error(f"Skill retrieval failed: {e}")
            return []

    def retrieve_for_tech_stack(
        self,
        tech_stack: TechStack,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve skills relevant to a specific technology stack.

        Builds a semantic query from the tech stack and searches
        long-term memory. Called after fingerprinting.

        Args:
            tech_stack: The detected technology stack.
            top_k: Maximum number of results.

        Returns:
            List of relevant skill dicts.
        """
        # Build a descriptive query from the tech stack
        parts = []
        if tech_stack.language:
            parts.append(tech_stack.language)
        if tech_stack.framework:
            parts.append(tech_stack.framework)
        if tech_stack.database:
            parts.append(tech_stack.database)
        if tech_stack.web_server:
            parts.append(tech_stack.web_server)

        if not parts:
            return []

        query = f"attack techniques for {' '.join(parts)} web application"
        return self.retrieve_skills(query=query, top_k=top_k)

    # ===================================================================
    # Storage — Called at Scan End
    # ===================================================================

    def store_episode(self, llm_summary: str = "") -> EpisodeRecord | None:
        """
        Persist the current scan's results as an episodic memory entry.

        Called at the end of a scan (Phase 6: POST-SCAN) to save
        what was learned for future scans against this target.

        Args:
            llm_summary: Optional LLM-generated summary of the scan.
                         If not provided, a basic summary is auto-generated.

        Returns:
            The created EpisodeRecord, or None if storage failed.
        """
        episode_data = self.working.to_episode()

        if not episode_data.get("target_host"):
            logger.warning("Cannot store episode — no target host")
            return None

        # Use LLM summary if provided, otherwise use auto-generated
        summary = llm_summary or episode_data.get("summary", "")

        try:
            record = self.episodic.save_episode(
                target_host=episode_data["target_host"],
                scan_id=episode_data["scan_id"],
                tech_stack=episode_data.get("tech_stack"),
                endpoints=episode_data.get("endpoints"),
                summary=summary,
            )

            logger.info(
                f"Episode stored for {episode_data['target_host']} — "
                f"scan {episode_data['scan_id']}"
            )
            return record

        except Exception as e:
            logger.error(f"Failed to store episode: {e}")
            return None

    def store_skill(
        self,
        skill_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Index a skill document into long-term memory.

        Called after the Skill Writer generates a new skill document
        from a completed scan.

        Args:
            skill_id: Unique skill identifier (e.g., "wraith-skill-0042").
            text: The skill document text content.
            metadata: Skill metadata (attack_class, confidence, tags, etc.).
        """
        try:
            self.longterm.add_document(
                doc_id=skill_id,
                text=text,
                metadata=metadata,
            )
            logger.info(f"Skill indexed in long-term memory: {skill_id}")
        except Exception as e:
            logger.error(f"Failed to index skill {skill_id}: {e}")

    # ===================================================================
    # Context Building — For LLM Prompts
    # ===================================================================

    def build_full_context(self, max_chars: int = 16000) -> str:
        """
        Build the complete v2 context string for LLM prompts.

        Combines all three memory tiers into a single context string:
        episodic history + retrieved skills + current scan state.

        Args:
            max_chars: Maximum total characters.

        Returns:
            Complete context string.
        """
        return self.working.build_enhanced_context(max_chars=max_chars)

    def build_attack_planning_context(self, endpoint_key: str) -> str:
        """
        Build context for planning an attack on a specific endpoint.

        Includes standard ScanMemory context plus any relevant skills.

        Args:
            endpoint_key: The endpoint identifier (e.g., "POST /login").

        Returns:
            Rich context string for attack planning.
        """
        # Standard ScanMemory context for this endpoint
        base_context = self.scan_memory.build_attack_planning_context(endpoint_key)

        # Add skill context if available
        skill_context = self.working.build_skill_context()

        if skill_context:
            return f"{skill_context}\n\n{base_context}"

        return base_context

    # ===================================================================
    # Delegate Common ScanMemory Operations
    # ===================================================================
    # These are convenience passthrough methods so the orchestrator
    # doesn't need to access working.scan_memory directly for
    # the most common operations.

    def set_target(self, target: Target) -> None:
        """Delegate to ScanMemory.set_target()."""
        self.scan_memory.set_target(target)

    def update_tech_stack(self, tech: TechStack) -> None:
        """Delegate to ScanMemory.update_tech_stack()."""
        self.scan_memory.update_tech_stack(tech)

    def add_endpoints_from_crawl(self, crawl_results: list[dict]) -> int:
        """Delegate to ScanMemory.add_endpoints_from_crawl()."""
        return self.scan_memory.add_endpoints_from_crawl(crawl_results)

    def record_attack(self, attack_result: Any, endpoint_key: str) -> Any:
        """Delegate to ScanMemory.record_attack()."""
        return self.scan_memory.record_attack(attack_result, endpoint_key=endpoint_key)

    def add_vulnerability(self, vuln: Any) -> None:
        """Delegate to ScanMemory.add_vulnerability()."""
        self.scan_memory.add_vulnerability(vuln)

    def log_reasoning(self, **kwargs) -> None:
        """Delegate to ScanMemory.log_reasoning()."""
        self.scan_memory.log_reasoning(**kwargs)

    def get_vulnerabilities(self, **kwargs) -> list:
        """Delegate to ScanMemory.get_vulnerabilities()."""
        return self.scan_memory.get_vulnerabilities(**kwargs)

    def get_untested_endpoints(self, **kwargs) -> list:
        """Delegate to ScanMemory.get_untested_endpoints()."""
        return self.scan_memory.get_untested_endpoints(**kwargs)

    def get_all_endpoints(self) -> list:
        """Delegate to ScanMemory.get_all_endpoints()."""
        return self.scan_memory.get_all_endpoints()

    def was_attack_tried(self, endpoint_key: str, attack_type: str) -> bool:
        """Delegate to ScanMemory.was_attack_tried()."""
        return self.scan_memory.was_attack_tried(endpoint_key, attack_type)

    def mark_endpoint_baseline(self, **kwargs) -> None:
        """Delegate to ScanMemory.mark_endpoint_baseline()."""
        self.scan_memory.mark_endpoint_baseline(**kwargs)

    def export_for_report(self) -> dict[str, Any]:
        """Delegate to ScanMemory.export_for_report()."""
        return self.scan_memory.export_for_report()
        
    def build_target_context(self) -> str:
        """Delegate to ScanMemory.build_target_context()."""
        return self.scan_memory.build_target_context()

    def build_endpoint_context(self) -> str:
        """Delegate to ScanMemory.build_endpoint_context()."""
        return self.scan_memory.build_endpoint_context()

    def get_progress_display(self) -> str:
        """Delegate to ScanMemory.get_progress_display()."""
        return self.scan_memory.get_progress_display()

    @property
    def target(self):
        return self.scan_memory.target

    @property
    def tech_stack(self):
        return self.scan_memory.tech_stack

    @property
    def endpoint_count(self):
        return self.scan_memory.endpoint_count

    @property
    def vulnerability_count(self):
        return self.scan_memory.vulnerability_count

    @property
    def attack_count(self):
        return self.scan_memory.attack_count

    @property
    def total_requests_sent(self):
        return self.scan_memory.total_requests_sent
