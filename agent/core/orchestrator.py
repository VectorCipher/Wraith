"""
WRAITH Orchestrator

The autonomous AI brain loop that drives the entire penetration test.
This is where everything comes together — the LLM, the scanner, the
memory system, and the task tree.

The orchestrator implements the core agent loop:

    ┌──────────────────────────────────────────────┐
    │              ORCHESTRATOR LOOP               │
    │                                              │
    │   ┌──────┐    ┌────────┐    ┌──────┐        │
    │   │THINK │ →  │ DECIDE │ →  │ ACT  │        │
    │   │      │    │        │    │      │        │
    │   │ LLM  │    │ LLM    │    │ Go   │        │
    │   │      │    │        │    │Scanner│        │
    │   └──┬───┘    └────────┘    └──┬───┘        │
    │      │                         │             │
    │      │      ┌────────┐         │             │
    │      └──────│ LEARN  │←────────┘             │
    │             │        │                       │
    │             │ Memory │                       │
    │             └────────┘                       │
    └──────────────────────────────────────────────┘

Scan lifecycle:
    1. INITIALIZE  — Connect to scanner, verify LLM, set up memory
    2. RECON       — Fingerprint + crawl + optional code analysis
    3. ANALYSIS    — AI analyzes attack surface, plans attack strategy
    4. EXPLOIT     — Execute attacks, analyze results, confirm vulns
    5. POST_EXPLOIT— Chain vulns, escalate findings, deep-dive
    6. REPORT      — Generate final report

Each phase calls the LLM with memory context, acts on the decision,
records results back into memory, and moves to the next step.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from config import settings
from core.task_tree import TaskTree
from databases.db import DatabaseManager
from memory.manager import MemoryManager
from llm.client import LLMClient
from llm.prompt_engine import PromptEngine
from scanner_client.client import ScannerClient
from models.scan import ScanConfig, ScanState, ScanStatus, ScanMode
from models.target import Target, Endpoint, Parameter, TechStack
from models.vulnerability import Vulnerability, Severity, VulnerabilityType
from models.attack_result import AttackRequest, AttackResult, AttackStatus
from utils.exception import (
    WraithError,
    ScannerConnectionError,
    LLMConnectionError,
    ScanAbortedError,
    TargetUnreachableError,
)
from utils.logger import get_logger

logger = get_logger("core.orchestrator")


# ===================================================================
# Phase Callbacks — for CLI to hook into scan events
# ===================================================================

class ScanCallbacks:
    """
    Callback hooks the CLI can register to receive live scan events.
    All methods are no-ops by default — override what you need.
    """

    async def on_phase_start(self, phase: str, description: str) -> None:
        """Called when a scan phase begins."""
        pass

    async def on_phase_complete(self, phase: str, summary: str) -> None:
        """Called when a scan phase finishes."""
        pass

    async def on_task_start(self, task_id: str, name: str) -> None:
        """Called when a task begins."""
        pass

    async def on_task_complete(self, task_id: str, summary: str) -> None:
        """Called when a task finishes."""
        pass

    async def on_vulnerability_found(self, vuln: Vulnerability) -> None:
        """Called when a vulnerability is confirmed."""
        pass

    async def on_endpoint_discovered(self, method: str, path: str) -> None:
        """Called when a new endpoint is discovered."""
        pass

    async def on_progress_update(self, progress: float, status: str) -> None:
        """Called periodically with progress updates."""
        pass

    async def on_ai_reasoning(self, phase: str, decision: str) -> None:
        """Called when the AI makes a decision."""
        pass

    async def on_error(self, error: str) -> None:
        """Called when a non-fatal error occurs."""
        pass


# ===================================================================
# Orchestrator — The Brain
# ===================================================================

class Orchestrator:
    """
    The autonomous AI pentesting brain.

    Manages the full lifecycle of a penetration test by coordinating
    the LLM, scanner, memory, and task tree. Each phase calls the AI
    with accumulated context, executes the AI's decisions, and feeds
    results back into memory for the next iteration.

    Usage:
        orchestrator = Orchestrator(config)
        result = await orchestrator.run()
    """

    def __init__(
        self,
        config: ScanConfig,
        callbacks: ScanCallbacks | None = None,
    ) -> None:
        self._config = config
        self._callbacks = callbacks or ScanCallbacks()

        # Generate scan ID
        self._scan_id = f"wraith-{uuid.uuid4().hex[:8]}"

        # Core components
        self._llm = LLMClient()
        self._scanner = ScannerClient()
        self._prompt_engine = PromptEngine()
        self._task_tree = TaskTree(scan_id=self._scan_id, target_url=config.target_url)

        # v2: Database and Memory Manager (replaces raw ScanMemory)
        self._db = DatabaseManager(db_path=settings.db_path)
        self._memory = MemoryManager(
            db_manager=self._db,
            scan_id=self._scan_id,
            config=config,
            chroma_path=settings.chroma_path,
        )

        # Scan state
        self._state = ScanState(scan_id=self._scan_id)
        self._aborted = False

        logger.info(
            f"Orchestrator initialized — "
            f"scan_id={self._scan_id}, "
            f"target={config.target_url}, "
            f"mode={config.mode.value}"
        )

    # ===================================================================
    # Properties
    # ===================================================================

    @property
    def scan_id(self) -> str:
        return self._scan_id

    @property
    def state(self) -> ScanState:
        return self._state

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def task_tree(self) -> TaskTree:
        return self._task_tree

    @property
    def is_running(self) -> bool:
        return self._state.is_running

    # ===================================================================
    # Main Entry Point
    # ===================================================================

    async def run(self) -> ScanState:
        """
        Execute the full penetration test lifecycle.

        Returns the final ScanState when complete.
        Raises ScanAbortedError if the user cancels.
        """
        logger.info(f"{'=' * 60}")
        logger.info(f"WRAITH SCAN STARTED — {self._scan_id}")
        logger.info(f"Target: {self._config.target_url}")
        logger.info(f"Mode: {self._config.mode.value}")
        logger.info(f"{'=' * 60}")

        try:
            # Phase 1: Initialize
            await self._phase_initialize()
            if self._aborted:
                return self._finalize(ScanStatus.ABORTED)

            # Phase 2: Reconnaissance
            await self._phase_recon()
            if self._aborted:
                return self._finalize(ScanStatus.ABORTED)

            # Phase 3: Analysis
            await self._phase_analysis()
            if self._aborted:
                return self._finalize(ScanStatus.ABORTED)

            # Phase 4: Exploitation
            await self._phase_exploitation()
            if self._aborted:
                return self._finalize(ScanStatus.ABORTED)

            # Phase 5: Post-Exploitation
            await self._phase_post_exploitation()
            if self._aborted:
                return self._finalize(ScanStatus.ABORTED)

            # Phase 6: Reporting
            await self._phase_reporting()

            return self._finalize(ScanStatus.COMPLETED)

        except ScanAbortedError:
            logger.warning("Scan aborted by user")
            return self._finalize(ScanStatus.ABORTED)
        except Exception as e:
            logger.error(f"Scan failed with unexpected error: {e}")
            self._state.errors.append(str(e))
            return self._finalize(ScanStatus.FAILED)
        finally:
            await self._cleanup()

    async def abort(self) -> None:
        """Signal the orchestrator to stop after the current task."""
        self._aborted = True
        logger.warning("Abort signal received — stopping after current task")
        try:
            await self._scanner.abort_all(reason="user abort")
        except Exception:
            pass

    # ===================================================================
    # Phase 1: Initialization
    # ===================================================================

    async def _phase_initialize(self) -> None:
        """Connect to services, verify readiness, set up target."""
        phase = "init"
        self._update_state(ScanStatus.INITIALIZING, "Initializing")
        self._task_tree.start_phase(phase)
        await self._callbacks.on_phase_start(phase, "Initializing services")

        # Task: Connect to scanner
        task_id = self._task_tree.add_task(phase, "Connect to Go scanner")
        self._task_tree.start_task(task_id)
        try:
            await self._scanner.connect()
            healthy = await self._scanner.health_check()
            if not healthy:
                raise ScannerConnectionError(
                    message="Scanner is not healthy",
                    details="Health check returned unhealthy status",
                )
            self._task_tree.complete_task(task_id, summary="Connected and healthy")
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))
            raise

        # Task: Verify LLM
        task_id = self._task_tree.add_task(phase, "Verify LLM connection")
        self._task_tree.start_task(task_id)
        try:
            connected = await self._llm.check_connection()
            if not connected:
                raise LLMConnectionError(
                    message="Cannot connect to Ollama",
                    details="Ensure Ollama is running: ollama serve",
                )
            self._task_tree.complete_task(
                task_id,
                summary=f"Connected — model: {settings.model}",
            )
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))
            raise

        # Task: Initialize target
        task_id = self._task_tree.add_task(phase, "Initialize target")
        self._task_tree.start_task(task_id)

        target = Target(
            url=self._config.target_url,
            source_path=self._config.source_path,
        )
        self._memory.set_target(target)
        self._task_tree.complete_task(task_id, summary=self._config.target_url)

        # v2 Task: Load episodic memory (prior scans of this target)
        task_id = self._task_tree.add_task(phase, "Load prior knowledge")
        self._task_tree.start_task(task_id)
        try:
            init_result = self._memory.initialize(target_url=self._config.target_url)
            if init_result["has_history"]:
                summary = (
                    f"Loaded {init_result['episode_count']} prior episodes "
                    f"for this target"
                )
                self._memory.log_reasoning(
                    phase="init",
                    action="load_episodic_memory",
                    reasoning="Checked for prior scans of this target",
                    decision=f"Found {init_result['episode_count']} prior episodes",
                    outcome="Episodic context injected into working memory",
                )
            else:
                summary = "First engagement — no prior history"
            self._task_tree.complete_task(task_id, summary=summary)
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))
            logger.warning(f"Episodic memory load failed (non-fatal): {e}")

        # v2 Task: Retrieve relevant skills from long-term memory
        task_id = self._task_tree.add_task(phase, "Retrieve attack knowledge")
        self._task_tree.start_task(task_id)
        try:
            # Build a query from the target URL (tech stack will be refined after fingerprinting)
            skills = self._memory.retrieve_skills(
                query=f"attack techniques for {self._config.target_url}",
                top_k=settings.max_skill_retrieval,
            )
            if skills:
                summary = f"{len(skills)} relevant skills retrieved from memory"
            else:
                summary = "No prior skills in memory (will learn from this scan)"
            self._task_tree.complete_task(task_id, summary=summary)
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))
            logger.warning(f"Skill retrieval failed (non-fatal): {e}")

        self._task_tree.complete_phase(phase, summary="All services ready")
        await self._callbacks.on_phase_complete(phase, "All services ready")

    # ===================================================================
    # Phase 2: Reconnaissance
    # ===================================================================

    async def _phase_recon(self) -> None:
        """Discover target technology, endpoints, and attack surface."""
        phase = "recon"
        self._update_state(ScanStatus.RECONNAISSANCE, "Reconnaissance")
        self._task_tree.start_phase(phase)
        await self._callbacks.on_phase_start(phase, "Discovering attack surface")

        # Task: Fingerprint
        await self._recon_fingerprint(phase)
        self._check_abort()

        # Task: Crawl
        await self._recon_crawl(phase)
        self._check_abort()

        # Task: Source code analysis (if whitebox/full mode)
        if self._config.mode in (ScanMode.WHITEBOX, ScanMode.FULL):
            if self._config.source_path:
                await self._recon_source_analysis(phase)
                self._check_abort()
            else:
                tid = self._task_tree.add_task(phase, "Source code analysis")
                self._task_tree.skip_task(tid, reason="No source path provided")

        # Task: AI recon analysis — ask the AI to reason about what we found
        await self._recon_ai_analysis(phase)

        summary = (
            f"{self._memory.endpoint_count} endpoints, "
            f"tech: {self._memory.tech_stack.framework or 'unknown'}"
        )
        self._task_tree.complete_phase(phase, summary=summary)
        await self._callbacks.on_phase_complete(phase, summary)

    async def _recon_fingerprint(self, phase: str) -> None:
        """Run technology fingerprinting via Go scanner."""
        task_id = self._task_tree.add_task(phase, "Fingerprint target")
        self._task_tree.start_task(task_id)
        await self._callbacks.on_task_start(task_id, "Fingerprint target")

        try:
            tech = await self._scanner.fingerprint_target(
                target_url=self._config.target_url,
            )
            self._memory.update_tech_stack(tech)

            summary = ", ".join(filter(None, [
                tech.language, tech.framework, tech.database, tech.web_server,
            ])) or "No technologies detected"

            self._task_tree.complete_task(task_id, summary=summary)
            await self._callbacks.on_task_complete(task_id, summary)
        except ScannerConnectionError as e:
            self._task_tree.fail_task(task_id, reason=e.message)
            logger.warning(f"Fingerprint failed: {e.message}")

    async def _recon_crawl(self, phase: str) -> None:
        """Crawl the target to discover endpoints."""
        task_id = self._task_tree.add_task(phase, "Crawl target")
        self._task_tree.start_task(task_id)
        await self._callbacks.on_task_start(task_id, "Crawl target")

        try:
            crawl_results = []
            async for page in self._scanner.crawl_target(
                target_url=self._config.target_url,
                max_depth=3,
                max_pages=100,
            ):
                crawl_results.append(page)
                url = page.get("url", "")
                method = page.get("method", "GET")
                await self._callbacks.on_endpoint_discovered(method, url)

            new_count = self._memory.add_endpoints_from_crawl(crawl_results)
            summary = f"{new_count} new endpoints from {len(crawl_results)} pages"
            self._task_tree.complete_task(task_id, summary=summary, findings_count=new_count)
            await self._callbacks.on_task_complete(task_id, summary)
        except ScannerConnectionError as e:
            self._task_tree.fail_task(task_id, reason=e.message)
            logger.warning(f"Crawl failed: {e.message}")

    async def _recon_source_analysis(self, phase: str) -> None:
        """Analyze source code for vulnerabilities (whitebox)."""
        task_id = self._task_tree.add_task(phase, "Source code analysis")
        self._task_tree.start_task(task_id)
        await self._callbacks.on_task_start(task_id, "Source code analysis")

        # This phase uses the LLM to analyze source files
        # For now, record intent — full implementation will scan files
        # with tree-sitter and send interesting functions to the LLM
        target = self._memory.target
        if not target or not target.source_path:
            self._task_tree.skip_task(task_id, reason="No source path")
            return

        self._memory.log_reasoning(
            phase="recon",
            action="source_analysis",
            reasoning="Source code is available for white-box analysis",
            decision="Analyze source files for vulnerable patterns",
        )

        # Placeholder — the actual implementation will:
        # 1. Walk the source directory
        # 2. Parse files with tree-sitter to find routes/handlers
        # 3. Send each handler to the LLM for security review
        # 4. Register discovered endpoints and potential vulns

        self._task_tree.complete_task(
            task_id,
            summary="Source analysis framework ready",
        )
        await self._callbacks.on_task_complete(task_id, "Source analysis framework ready")

    async def _recon_ai_analysis(self, phase: str) -> None:
        """Ask the AI to analyze everything we discovered in recon."""
        task_id = self._task_tree.add_task(phase, "AI reconnaissance analysis")
        self._task_tree.start_task(task_id)

        target = self._memory.target
        if not target:
            self._task_tree.skip_task(task_id, reason="No target set")
            return

        # Build prompt with full memory context
        prompt = self._prompt_engine.build_recon_prompt(target)

        # Inject memory context
        memory_context = self._memory.build_target_context()
        endpoint_context = self._memory.build_endpoint_context()
        full_prompt = f"{memory_context}\n\n{endpoint_context}\n\n{prompt}"

        try:
            response = await self._llm.generate(
                role="reasoning",
                prompt=full_prompt,
                temperature=0.3,
            )

            self._memory.log_reasoning(
                phase="recon",
                action="ai_recon_analysis",
                reasoning="Analyzed target tech stack and attack surface",
                decision="Identified priority attack vectors",
                outcome=response.content[:200] if response.content else "empty response",
            )

            self._task_tree.complete_task(task_id, summary="Attack surface analyzed")
            await self._callbacks.on_ai_reasoning("recon", "Attack surface analyzed")
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))
            logger.warning(f"AI recon analysis failed: {e}")

    # ===================================================================
    # Phase 3: Analysis — AI Plans the Attack Strategy
    # ===================================================================

    async def _phase_analysis(self) -> None:
        """AI analyzes the recon data and creates an attack plan."""
        phase = "analysis"
        self._update_state(ScanStatus.ANALYSIS, "Analyzing attack surface")
        self._task_tree.start_phase(phase)
        await self._callbacks.on_phase_start(phase, "Planning attack strategy")

        # Task: Capture baselines for priority endpoints
        await self._analysis_capture_baselines(phase)
        self._check_abort()

        # Task: AI attack planning
        await self._analysis_plan_attacks(phase)

        self._task_tree.complete_phase(phase, summary="Attack strategy ready")
        await self._callbacks.on_phase_complete(phase, "Attack strategy ready")

    async def _analysis_capture_baselines(self, phase: str) -> None:
        """Capture baseline responses for endpoints we'll attack."""
        task_id = self._task_tree.add_task(phase, "Capture baseline responses")
        self._task_tree.start_task(task_id)

        endpoints = self._memory.get_untested_endpoints()
        captured = 0

        for entry in endpoints[:20]:  # Top 20 by priority
            ep = entry.endpoint
            if entry.is_baseline_captured:
                continue

            try:
                baseline = await self._scanner.send_baseline(
                    url=f"{self._config.target_url}{ep.path}",
                    method=ep.method,
                )
                if not baseline.get("error"):
                    self._memory.mark_endpoint_baseline(
                        method=ep.method,
                        path=ep.path,
                        status_code=baseline["status_code"],
                        content_length=baseline["content_length"],
                        response_time_ms=baseline["response_time_ms"],
                    )
                    captured += 1
            except ScannerConnectionError:
                continue

            self._check_abort()

        self._task_tree.complete_task(
            task_id,
            summary=f"{captured} baselines captured",
        )

    async def _analysis_plan_attacks(self, phase: str) -> None:
        """Ask the AI to plan the attack strategy."""
        task_id = self._task_tree.add_task(phase, "AI attack planning")
        self._task_tree.start_task(task_id)

        # Build rich context for the AI
        target = self._memory.target
        if not target:
            self._task_tree.skip_task(task_id, reason="No target")
            return

        # Build individual context blocks for v2 prompt
        skill_context = self._memory.working.build_skill_context()
        episodic_context = self._memory.working.episodic_context
        target_url = self._config.target_url
        tech_stack = self._prompt_engine._format_tech_stack(self._memory.tech_stack)
        endpoint_count = self._memory.endpoint_count
        endpoints = self._prompt_engine._format_endpoints(self._memory.get_untested_endpoints())

        full_prompt = self._prompt_engine.build_attack_plan_with_skills_prompt(
            skill_context=skill_context,
            episodic_context=episodic_context,
            target_url=target_url,
            tech_stack=tech_stack,
            endpoint_count=endpoint_count,
            endpoints=endpoints,
        )

        try:
            response = await self._llm.generate(
                role="reasoning",
                prompt=full_prompt,
                temperature=0.3,
            )

            self._memory.log_reasoning(
                phase="analysis",
                action="attack_planning",
                reasoning="Analyzed endpoints and tech stack for attack strategy",
                decision="Created prioritized attack plan",
                outcome=response.content[:200] if response.content else "empty",
            )

            self._task_tree.complete_task(task_id, summary="Attack plan created")
            await self._callbacks.on_ai_reasoning("analysis", "Attack plan created")
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))

    # ===================================================================
    # Phase 4: Exploitation — Execute Attacks
    # ===================================================================

    async def _phase_exploitation(self) -> None:
        """Execute attacks against discovered endpoints."""
        phase = "exploitation"
        self._update_state(ScanStatus.EXPLOITATION, "Executing attacks")
        self._task_tree.start_phase(phase)
        await self._callbacks.on_phase_start(phase, "Testing for vulnerabilities")

        # Get endpoints sorted by priority
        endpoints = self._memory.get_untested_endpoints()

        if not endpoints:
            self._task_tree.complete_phase(phase, summary="No testable endpoints")
            return

        # Attack each endpoint with relevant attack types
        attack_types = ["sqli", "xss", "ssrf", "ssti", "path_traversal", "idor"]

        for entry in endpoints:
            self._check_abort()
            ep = entry.endpoint

            for attack_type in attack_types:
                self._check_abort()

                if self._memory.was_attack_tried(entry.key, attack_type):
                    continue

                # Check if this attack type is relevant
                if not self._is_attack_relevant(entry, attack_type):
                    continue

                await self._execute_single_attack(
                    phase=phase,
                    entry=entry,
                    attack_type=attack_type,
                )

        vuln_count = self._memory.vulnerability_count
        summary = f"{self._memory.attack_count} attacks, {vuln_count} vulnerabilities"
        self._task_tree.complete_phase(phase, summary=summary)
        await self._callbacks.on_phase_complete(phase, summary)

    async def _execute_single_attack(
        self,
        phase: str,
        entry: Any,
        attack_type: str,
    ) -> None:
        """Generate payloads via AI, execute via scanner, analyze results."""
        ep = entry.endpoint
        endpoint_key = entry.key
        task_name = f"{attack_type.upper()} on {endpoint_key}"

        task_id = self._task_tree.add_task(phase, task_name)
        self._task_tree.start_task(task_id)
        await self._callbacks.on_task_start(task_id, task_name)

        try:
            # Step 1: Ask AI to generate payloads
            context = self._memory.build_attack_planning_context(endpoint_key)
            target_param = self._get_target_param(ep)
            payload_prompt = self._prompt_engine.build_payload_prompt(
                attack_type=attack_type,
                target=self._memory.target,
                endpoint=ep,
                parameter_name=target_param,
            )

            payload_response = await self._llm.generate(
                role="coding",
                prompt=f"{context}\n\n{payload_prompt}",
                temperature=0.2,
            )

            # Parse payloads from AI response
            payloads = self._parse_payloads(payload_response.content)

            if not payloads:
                self._task_tree.complete_task(task_id, summary="No payloads generated")
                return

            # Step 2: Execute via Go scanner
            attack_id = f"atk-{uuid.uuid4().hex[:8]}"
            request = AttackRequest(
                attack_id=attack_id,
                attack_type=attack_type,
                target_url=f"{self._config.target_url}{ep.path}",
                method=ep.method,
                payloads=payloads,
                injection_point=self._get_injection_point(ep, attack_type),
                parameter_name=self._get_target_param(ep),
                timeout_seconds=self._config.rate_limit,
            )

            if entry.is_baseline_captured:
                request.baseline_response = str(entry.baseline_content_length)

            result = await self._scanner.execute_attack(request)

            # Step 3: Record in memory
            self._memory.record_attack(result, endpoint_key=endpoint_key)

            # Step 4: Ask AI to analyze results
            vuln = await self._analyze_attack_results(
                entry=entry,
                attack_type=attack_type,
                result=result,
            )

            if vuln:
                self._memory.add_vulnerability(vuln)
                self._task_tree.complete_task(
                    task_id,
                    summary=f"⚠️ VULNERABLE — {vuln.severity.value.upper()}",
                    findings_count=1,
                )
                await self._callbacks.on_vulnerability_found(vuln)
            else:
                # v2: Try payload mutations if initial payloads failed
                mutation_vuln = await self._try_mutations(
                    entry=entry,
                    attack_type=attack_type,
                    original_payloads=payloads,
                    result=result,
                    phase=phase,
                )

                if mutation_vuln:
                    self._memory.add_vulnerability(mutation_vuln)
                    self._task_tree.complete_task(
                        task_id,
                        summary=f"⚠️ VULNERABLE (via mutation) — {mutation_vuln.severity.value.upper()}",
                        findings_count=1,
                    )
                    await self._callbacks.on_vulnerability_found(mutation_vuln)
                else:
                    self._task_tree.complete_task(
                        task_id,
                        summary=f"{len(payloads)} payloads — not vulnerable",
                    )

            await self._callbacks.on_task_complete(task_id, "Attack completed")

        except ScannerConnectionError as e:
            self._task_tree.fail_task(task_id, reason=e.message)
            await self._callbacks.on_error(f"Scanner error: {e.message}")
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))
            logger.error(f"Attack execution error: {e}")

    async def _try_mutations(
        self,
        entry: Any,
        attack_type: str,
        original_payloads: list[str],
        result: AttackResult,
        phase: str,
        max_rounds: int = 3,
    ) -> Vulnerability | None:
        """
        v2: Try payload mutations when initial payloads fail.

        Checks if the failure looks like WAF blocking, then uses the
        PayloadMutator to generate alternative payloads and retries.

        Args:
            entry: The endpoint entry being attacked.
            attack_type: Type of attack (sqli, xss, etc.)
            original_payloads: The payloads that failed.
            result: The AttackResult from the initial attempt.
            phase: Current scan phase name.
            max_rounds: Maximum mutation rounds to try.

        Returns:
            Vulnerability if mutation succeeds, None otherwise.
        """
        from payload_engine.mutator import PayloadMutator
        from payload_engine.logger import PayloadLogger, PayloadResult

        # Check if responses suggest WAF blocking
        blocked_responses = [
            pr for pr in result.payload_results
            if pr.status_code in (403, 406, 429, 503)
        ]

        if not blocked_responses:
            # No WAF signals — mutations unlikely to help
            return None

        ep = entry.endpoint
        endpoint_key = entry.key

        # Initialize mutation engine
        payload_logger = PayloadLogger(db_manager=self._db)
        mutator = PayloadMutator(payload_logger=payload_logger)

        # Log the original failures
        for pr in result.payload_results:
            payload_logger.log_result(PayloadResult(
                payload_raw=pr.payload,
                target_url=f"{self._config.target_url}{ep.path}",
                scan_id=self._scan_id,
                attack_class=attack_type,
                response_code=pr.status_code,
                response_body=pr.response_body or "",
                failure_reason="blocked" if pr.status_code in (403, 406, 429, 503) else "",
            ))

        logger.info(
            f"WAF detected on {endpoint_key} — "
            f"{len(blocked_responses)} blocked responses, "
            f"trying mutations (max {max_rounds} rounds)"
        )

        # Build failure context for smart mutation ranking
        failure_context = {
            "response_code": blocked_responses[0].status_code,
            "attack_class": attack_type,
        }

        # Try mutation rounds
        for round_num in range(1, max_rounds + 1):
            # Pick a payload that was blocked
            blocked_payload = blocked_responses[0].payload
            if round_num > 1 and len(blocked_responses) > 1:
                blocked_payload = blocked_responses[min(round_num - 1, len(blocked_responses) - 1)].payload

            # Get mutation suggestions
            suggestions = mutator.suggest_mutations(
                payload=blocked_payload,
                failure_context=failure_context,
                max_mutations=5,
            )

            if not suggestions:
                break

            # Extract mutated payloads (skip HEADER: and FRAGMENT: special formats)
            mutation_payloads = [
                s["payload"] for s in suggestions
                if not s["payload"].startswith(("HEADER:", "FRAGMENT:", "HPP:"))
            ][:5]

            if not mutation_payloads:
                break

            # Execute mutated payloads via scanner
            try:
                mutation_request = AttackRequest(
                    attack_id=f"mut-{uuid.uuid4().hex[:8]}",
                    attack_type=attack_type,
                    target_url=f"{self._config.target_url}{ep.path}",
                    method=ep.method,
                    payloads=mutation_payloads,
                    injection_point=self._get_injection_point(ep, attack_type),
                    parameter_name=self._get_target_param(ep),
                    timeout_seconds=self._config.rate_limit,
                )

                mutation_result = await self._scanner.execute_attack(mutation_request)

                # Log mutation results
                for pr in mutation_result.payload_results:
                    succeeded = pr.status_code not in (403, 406, 429, 503)
                    strategy = suggestions[0]["strategy"] if suggestions else "unknown"
                    payload_logger.log_result(PayloadResult(
                        payload_raw=pr.payload,
                        target_url=f"{self._config.target_url}{ep.path}",
                        scan_id=self._scan_id,
                        attack_class=attack_type,
                        response_code=pr.status_code,
                        response_body=pr.response_body or "",
                        mutation_applied=strategy,
                        mutation_succeeded=succeeded,
                    ))

                # Record in memory
                self._memory.record_attack(mutation_result, endpoint_key=endpoint_key)

                # Analyze mutation results
                vuln = await self._analyze_attack_results(
                    entry=entry,
                    attack_type=attack_type,
                    result=mutation_result,
                )

                if vuln:
                    self._memory.log_reasoning(
                        phase=phase,
                        action="mutation_success",
                        reasoning=f"Initial payload blocked, mutation round {round_num} succeeded",
                        decision=f"Vulnerability confirmed via payload mutation",
                        outcome=f"{vuln.severity.value.upper()} — {vuln.title}",
                    )
                    logger.info(
                        f"🎯 Mutation round {round_num} succeeded on {endpoint_key}!"
                    )
                    return vuln

            except Exception as e:
                logger.debug(f"Mutation round {round_num} error: {e}")
                continue

        logger.debug(
            f"All {max_rounds} mutation rounds exhausted on {endpoint_key}"
        )
        return None

    async def _analyze_attack_results(
        self,
        entry: Any,
        attack_type: str,
        result: AttackResult,
    ) -> Vulnerability | None:
        """Ask the AI to analyze attack results and determine if vulnerable."""
        if result.status != AttackStatus.SUCCESS:
            return None

        if not result.payload_results:
            return None

        # Build analysis prompt with results
        target = self._memory.target
        if not target:
            return None

        # Build payloads_and_responses list for the prompt engine
        payloads_and_responses = [
            {
                "payload": pr.payload,
                "status_code": pr.status_code,
                "response_body": pr.response_body,
                "response_time_ms": pr.response_time_ms,
            }
            for pr in result.payload_results
        ]
        baseline_str = (
            f"HTTP {entry.baseline_status_code}, {entry.baseline_content_length} bytes"
            if entry.is_baseline_captured else None
        )
        prompt = self._prompt_engine.build_result_analysis_prompt(
            attack_type=attack_type,
            endpoint=f"{entry.endpoint.method} {entry.endpoint.path}",
            payloads_and_responses=payloads_and_responses,
            baseline_response=baseline_str,
        )

        try:
            response = await self._llm.generate(
                role="reasoning",
                prompt=prompt,
                temperature=0.1,  # Low temperature for precise analysis
            )

            # Parse the AI's verdict
            content = response.content.lower() if response.content else ""

            # Look for confirmation signals
            is_vulnerable = any(word in content for word in [
                "confirmed", "vulnerable", "exploitable",
                "successful", "injection confirmed",
                "vulnerability found", "is vulnerable",
            ])

            # Look for false positive signals
            is_false_positive = any(word in content for word in [
                "false positive", "not vulnerable", "no vulnerability",
                "safely handled", "properly sanitized", "not exploitable",
            ])

            if is_vulnerable and not is_false_positive:
                # Determine severity
                severity = self._determine_severity(attack_type, content)

                vuln = Vulnerability(
                    vuln_type=self._map_attack_to_vuln_type(attack_type),
                    severity=severity,
                    title=f"{attack_type.upper()} in {entry.endpoint.path}",
                    description=response.content[:500] if response.content else "",
                    endpoint=entry.endpoint.path,
                    method=entry.endpoint.method,
                )

                self._memory.log_reasoning(
                    phase="exploitation",
                    action=f"analyze_{attack_type}",
                    reasoning="AI analyzed attack results",
                    decision="Vulnerability confirmed",
                    outcome=f"{severity.value} {attack_type}",
                )

                return vuln

            self._memory.log_reasoning(
                phase="exploitation",
                action=f"analyze_{attack_type}",
                reasoning="AI analyzed attack results",
                decision="Not vulnerable / false positive",
            )
            return None

        except Exception as e:
            logger.warning(f"Analysis failed: {e}")
            return None

    # ===================================================================
    # Phase 5: Post-Exploitation
    # ===================================================================

    async def _phase_post_exploitation(self) -> None:
        """Chain vulnerabilities and escalate findings."""
        phase = "post_exploit"
        self._update_state(ScanStatus.POST_EXPLOITATION, "Post-exploitation analysis")
        self._task_tree.start_phase(phase)
        await self._callbacks.on_phase_start(phase, "Analyzing vulnerability chains")

        vulns = self._memory.get_vulnerabilities()
        if not vulns:
            self._task_tree.skip_phase(phase, reason="No vulnerabilities to chain")
            return

        # Task: Ask AI to find attack chains
        task_id = self._task_tree.add_task(phase, "Vulnerability chain analysis")
        self._task_tree.start_task(task_id)

        context = self._memory.build_full_context()
        # Build a chain analysis prompt inline (no dedicated method yet)
        vuln_list = "\n".join(
            f"- [{v.severity.value.upper()}] {v.title} — {v.method} {v.endpoint}"
            for v in vulns
        )
        chain_prompt = (
            "# VULNERABILITY CHAIN ANALYSIS\n\n"
            "Given the following confirmed vulnerabilities, identify:\n"
            "1. Which vulns can be chained for higher impact\n"
            "2. Possible privilege escalation paths\n"
            "3. Data exfiltration scenarios\n\n"
            f"## Confirmed Vulnerabilities:\n{vuln_list}"
        )

        try:
            response = await self._llm.generate(
                role="reasoning",
                prompt=f"{context}\n\n{chain_prompt}",
                temperature=0.3,
            )

            self._memory.log_reasoning(
                phase="post_exploit",
                action="chain_analysis",
                reasoning="Analyzed confirmed vulns for chaining potential",
                decision="Identified attack chains",
                outcome=response.content[:200] if response.content else "empty",
            )

            self._task_tree.complete_task(task_id, summary="Chain analysis complete")
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))

        # Task: Generate remediation for each vuln
        task_id = self._task_tree.add_task(phase, "Generate remediation code")
        self._task_tree.start_task(task_id)

        remediated = 0
        for vuln in vulns:
            if vuln.remediation:
                continue

            try:
                lang = self._memory.tech_stack.language or "unknown"
                remediation_prompt = self._prompt_engine.build_remediation_prompt(
                    vulnerability=vuln,
                    language=lang,
                )

                response = await self._llm.generate(
                    role="coding",
                    prompt=remediation_prompt,
                    temperature=0.1,
                )

                if response.content:
                    from models.vulnerability import Remediation
                    vuln.remediation = Remediation(
                        description=response.content[:1000],
                    )
                    remediated += 1
            except Exception:
                continue

        self._task_tree.complete_task(
            task_id,
            summary=f"Remediation generated for {remediated}/{len(vulns)} vulns",
        )

        self._task_tree.complete_phase(phase, summary="Post-exploitation complete")
        await self._callbacks.on_phase_complete(phase, "Post-exploitation complete")

    # ===================================================================
    # Phase 6: Reporting
    # ===================================================================

    async def _phase_reporting(self) -> None:
        """Generate the final scan report and persist knowledge."""
        phase = "reporting"
        self._update_state(ScanStatus.REPORTING, "Generating report")
        self._task_tree.start_phase(phase)
        await self._callbacks.on_phase_start(phase, "Generating report")

        # v2 Task: Store episodic memory for this scan
        task_id = self._task_tree.add_task(phase, "Store scan episode")
        self._task_tree.start_task(task_id)
        try:
            episode = self._memory.store_episode()
            if episode:
                summary = f"Episode saved for {episode.target_host}"
            else:
                summary = "Episode storage skipped"
            self._task_tree.complete_task(task_id, summary=summary)
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))
            logger.warning(f"Episode storage failed (non-fatal): {e}")

        # v2 Task: Extract and index skill from this scan
        task_id = self._task_tree.add_task(phase, "Extract reusable skill")
        self._task_tree.start_task(task_id)
        try:
            from skills.writer import SkillWriter
            from skills.indexer import SkillIndexer

            indexer = SkillIndexer(
                skills_dir=settings.skills_path,
                longterm_memory=self._memory.longterm,
                db_manager=self._db,
            )
            writer = SkillWriter(
                llm_client=self._llm,
                indexer=indexer,
                skills_dir=settings.skills_path,
            )

            scan_log = self._memory.export_for_report()
            skill_doc = await writer.extract_and_save(
                scan_id=self._scan_id,
                scan_log=scan_log,
            )

            if skill_doc:
                summary = (
                    f"Skill extracted: {skill_doc.skill_id} "
                    f"({skill_doc.attack_class}, {skill_doc.confidence})"
                )
                self._memory.log_reasoning(
                    phase="reporting",
                    action="skill_extraction",
                    reasoning="Extracted reusable technique from scan results",
                    decision=f"Generated skill {skill_doc.skill_id}",
                    outcome=f"{skill_doc.attack_class} — {skill_doc.title}",
                )
            else:
                summary = "No skill extracted (insufficient data)"
            self._task_tree.complete_task(task_id, summary=summary)
        except Exception as e:
            self._task_tree.fail_task(task_id, reason=str(e))
            logger.warning(f"Skill extraction failed (non-fatal): {e}")

        task_id = self._task_tree.add_task(phase, "Build report data")
        self._task_tree.start_task(task_id)

        report_data = self._memory.export_for_report()
        report_data["task_tree"] = self._task_tree.to_display_string()
        report_data["progress"] = self._memory.get_progress_display()

        self._task_tree.complete_task(task_id, summary="Report data assembled")

        # The actual report rendering (HTML/PDF/JSON) will be done by
        # the reporters package — this phase just prepares the data
        self._task_tree.complete_phase(phase, summary="Report ready")
        await self._callbacks.on_phase_complete(phase, "Report ready")

    # ===================================================================
    # Internal Helpers
    # ===================================================================

    def _update_state(self, status: ScanStatus, phase: str) -> None:
        """Update the live scan state."""
        self._state.status = status
        self._state.current_phase = phase
        self._state.endpoints_tested = sum(
            1 for e in self._memory.get_all_endpoints() if e.attack_types_tested
        )
        self._state.vulnerabilities_found = self._memory.vulnerability_count
        self._state.total_requests_sent = self._memory.total_requests_sent

    def _check_abort(self) -> None:
        """Raise if abort was requested."""
        if self._aborted:
            raise ScanAbortedError(
                message="Scan aborted",
                details="User requested scan cancellation",
            )

    def _finalize(self, status: ScanStatus) -> ScanState:
        """Finalize the scan state."""
        self._state.status = status
        self._state.end_time = datetime.now()
        self._state.vulnerabilities_found = self._memory.vulnerability_count
        self._state.endpoints_tested = sum(
            1 for e in self._memory.get_all_endpoints() if e.attack_types_tested
        )
        self._state.total_requests_sent = self._memory.total_requests_sent

        logger.info(f"{'=' * 60}")
        logger.info(f"SCAN {status.value.upper()} — {self._scan_id}")
        logger.info(f"Duration: {self._state.duration_seconds:.1f}s")
        logger.info(f"Endpoints: {self._state.endpoints_tested}")
        logger.info(f"Vulnerabilities: {self._state.vulnerabilities_found}")
        logger.info(f"Requests: {self._state.total_requests_sent}")
        logger.info(f"{'=' * 60}")

        return self._state

    async def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self._scanner.close()
        except Exception:
            pass
        try:
            await self._llm.close()
        except Exception:
            pass

    def _parse_payloads(self, ai_response: str) -> list[str]:
        """
        Parse payload list from AI response.
        The AI typically returns payloads as a numbered or bulleted list.
        """
        if not ai_response:
            return []

        payloads = []
        for line in ai_response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Strip numbering: "1. payload" → "payload"
            # Strip bullets: "- payload" → "payload"
            # Strip backticks: "`payload`" → "payload"
            for prefix in ["- ", "* ", "• "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break

            # Strip numbered lists
            if line and line[0].isdigit():
                parts = line.split(". ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    line = parts[1]

            # Strip backtick wrapping
            line = line.strip("`").strip("'").strip('"')

            if line and len(line) > 1:
                payloads.append(line)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in payloads:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        return unique[:50]  # Cap at 50 payloads

    def _is_attack_relevant(self, entry: Any, attack_type: str) -> bool:
        """Check if an attack type is relevant for this endpoint."""
        ep = entry.endpoint

        # SQLi needs parameters
        if attack_type == "sqli" and not ep.parameters:
            return False

        # XSS needs user-facing input params
        if attack_type == "xss" and not ep.parameters:
            return False

        # SSRF typically targets URL params
        if attack_type == "ssrf":
            url_params = [
                p for p in ep.parameters
                if any(kw in p.name.lower() for kw in ["url", "uri", "link", "href", "path", "redirect"])
            ]
            if not url_params and ep.parameters:
                return True  # Still try if there are any params
            if not ep.parameters:
                return False

        # SSTI needs template-like params
        if attack_type == "ssti":
            ts = self._memory.tech_stack
            if ts.template_engine or ts.framework:
                return True
            return len(ep.parameters) > 0

        return True

    def _get_injection_point(self, ep: Any, attack_type: str) -> str:
        """Determine where to inject payloads."""
        if ep.method in ("POST", "PUT", "PATCH"):
            return "body"
        return "query"

    def _get_target_param(self, ep: Any) -> str:
        """Get the best parameter to target for injection."""
        if not ep.parameters:
            return ""
        # Prefer body params, then query params
        for p in ep.parameters:
            if p.location == "body":
                return p.name
        return ep.parameters[0].name

    def _determine_severity(self, attack_type: str, analysis: str) -> Severity:
        """Determine vulnerability severity based on type and AI analysis."""
        # Default severity mapping by attack type
        severity_map = {
            "sqli": Severity.CRITICAL,
            "command_injection": Severity.CRITICAL,
            "ssrf": Severity.HIGH,
            "ssti": Severity.HIGH,
            "xss": Severity.MEDIUM,
            "idor": Severity.MEDIUM,
            "path_traversal": Severity.HIGH,
            "xxe": Severity.HIGH,
            "cors": Severity.LOW,
        }

        base = severity_map.get(attack_type, Severity.MEDIUM)

        # AI might upgrade/downgrade
        if "critical" in analysis:
            return Severity.CRITICAL
        if "high" in analysis and base.value not in ("critical",):
            return Severity.HIGH

        return base

    def _map_attack_to_vuln_type(self, attack_type: str) -> VulnerabilityType:
        """Map attack type string to VulnerabilityType enum."""
        mapping = {
            "sqli": VulnerabilityType.SQLI,
            "xss": VulnerabilityType.XSS,
            "ssrf": VulnerabilityType.SSRF,
            "ssti": VulnerabilityType.SSTI,
            "idor": VulnerabilityType.IDOR,
            "path_traversal": VulnerabilityType.PATH_TRAVERSAL,
            "command_injection": VulnerabilityType.COMMAND_INJECTION,
            "xxe": VulnerabilityType.XXE,
            "cors": VulnerabilityType.CORS,
            "csrf": VulnerabilityType.CSRF,
        }
        return mapping.get(attack_type, VulnerabilityType.OTHER)
