"""
WRAITH Memory System

The AI's working memory during a penetration test. This is the heart of
the harness — it stores everything the AI discovers and provides rich,
structured context for every LLM call.

Without memory, the AI forgets what it already tried, repeats work,
and can't chain findings together. Memory turns a stateless LLM into
a persistent, learning agent.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                    ScanMemory                            │
    │                                                          │
    │  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
    │  │ EndpointStore│  │ AttackLedger  │  │ VulnStore    │  │
    │  │              │  │               │  │              │  │
    │  │ discovered   │  │ what we tried │  │ confirmed    │  │
    │  │ endpoints    │  │ per endpoint  │  │ findings     │  │
    │  └──────────────┘  └───────────────┘  └──────────────┘  │
    │                                                          │
    │  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
    │  │ ReasoningLog │  │ ContextEngine │  │ FindingLinks │  │
    │  │              │  │               │  │              │  │
    │  │ AI thoughts  │  │ builds LLM    │  │ correlates   │  │
    │  │ & decisions  │  │ context strs  │  │ findings     │  │
    │  └──────────────┘  └───────────────┘  └──────────────┘  │
    └──────────────────────────────────────────────────────────┘

Key design decisions:
    - All stores are in-memory (dict/list). No DB dependency.
    - Every store produces formatted context strings for LLM prompts.
    - Deduplication at every layer — never test the same thing twice.
    - Priority scoring on endpoints to focus the AI on what matters.
    - Thread-safe via asyncio lock (not threading — we use asyncio).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from models.target import Target, Endpoint, TechStack, Parameter
from models.vulnerability import Vulnerability, Severity, VulnerabilityType
from models.attack_result import AttackRequest, AttackResult, PayloadResult
from models.scan import ScanConfig, ScanMode
from utils.logger import get_logger

logger = get_logger("core.memory")


# ===================================================================
# Constants
# ===================================================================

# Maximum context sizes for LLM prompts (characters)
MAX_ENDPOINT_CONTEXT = 4000
MAX_ATTACK_CONTEXT = 3000
MAX_VULN_CONTEXT = 4000
MAX_REASONING_CONTEXT = 3000
MAX_FULL_CONTEXT = 12000


# ===================================================================
# Endpoint Priority Scoring
# ===================================================================

class EndpointPriority(str, Enum):
    """Priority level for testing an endpoint."""
    CRITICAL = "critical"   # Auth endpoints, admin panels, file uploads
    HIGH = "high"           # Data modification, user input handling
    MEDIUM = "medium"       # Standard CRUD operations
    LOW = "low"             # Static content, health checks
    SKIP = "skip"           # Already tested or explicitly excluded


# Patterns that make endpoints high-priority attack targets
_HIGH_PRIORITY_PATTERNS = [
    "login", "auth", "signin", "signup", "register",
    "admin", "dashboard", "upload", "import", "export",
    "password", "reset", "token", "api/key", "oauth",
    "graphql", "search", "query", "exec", "eval",
    "file", "download", "include", "template", "render",
    "redirect", "callback", "webhook", "proxy", "fetch",
]

# HTTP methods ranked by attack surface
_METHOD_PRIORITY = {
    "POST": 4,
    "PUT": 4,
    "PATCH": 3,
    "DELETE": 3,
    "GET": 2,
    "OPTIONS": 1,
    "HEAD": 1,
}


# ===================================================================
# EndpointEntry — Enriched Endpoint with Attack Metadata
# ===================================================================

class EndpointEntry(BaseModel):
    """
    An endpoint enriched with attack metadata.

    Wraps the base Endpoint model with additional tracking fields
    that the memory system uses to prioritize and track testing.
    """
    endpoint: Endpoint
    priority: EndpointPriority = EndpointPriority.MEDIUM
    priority_score: float = 0.0
    attack_types_tested: list[str] = []
    is_baseline_captured: bool = False
    baseline_status_code: int | None = None
    baseline_content_length: int | None = None
    baseline_response_time_ms: float | None = None
    notes: list[str] = []
    discovered_at: datetime = Field(default_factory=datetime.now)
    discovered_by: str = ""  # "crawl", "source_code", "manual", "ai_inference"

    @property
    def is_fully_tested(self) -> bool:
        """Has this endpoint been tested with all common attack types?"""
        common_attacks = {"sqli", "xss", "ssrf", "ssti", "idor", "path_traversal"}
        return common_attacks.issubset(set(self.attack_types_tested))

    @property
    def untested_attack_types(self) -> list[str]:
        """Attack types that haven't been tried on this endpoint yet."""
        common = ["sqli", "xss", "ssrf", "ssti", "idor", "path_traversal",
                  "command_injection", "xxe", "cors"]
        return [a for a in common if a not in self.attack_types_tested]

    @property
    def key(self) -> str:
        """Unique identifier: METHOD + PATH."""
        return f"{self.endpoint.method} {self.endpoint.path}"


# ===================================================================
# AttackEntry — Record of a Single Attack Attempt
# ===================================================================

class AttackEntry(BaseModel):
    """Record of a single attack attempt against an endpoint."""
    attack_id: str
    attack_type: str
    endpoint_key: str           # "POST /api/login"
    payloads_sent: int = 0
    interesting_responses: int = 0
    confirmed_vuln: bool = False
    vuln_id: str | None = None
    duration_ms: float = 0.0
    notes: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


# ===================================================================
# ReasoningEntry — AI's Thought Process
# ===================================================================

class ReasoningEntry(BaseModel):
    """A single entry in the AI's reasoning journal."""
    phase: str             # "recon", "analysis", "attack_planning", "exploitation"
    action: str            # "analyze_tech_stack", "plan_sqli_attack", etc.
    reasoning: str         # What the AI was thinking
    decision: str          # What the AI decided to do
    outcome: str = ""      # What actually happened
    timestamp: datetime = Field(default_factory=datetime.now)


# ===================================================================
# ScanMemory — The Complete Working Memory
# ===================================================================

class ScanMemory:
    """
    The AI agent's complete working memory for a single scan.

    This is the core harness that turns a stateless LLM into a
    persistent, context-aware penetration tester. Every LLM call
    gets injected with relevant memory context so the AI:

    1. Knows what tech stack the target runs
    2. Remembers which endpoints exist
    3. Recalls what attacks were already tried
    4. Can build on previous findings
    5. Doesn't repeat work
    6. Can chain vulnerabilities together

    Usage:
        memory = ScanMemory(scan_id="wraith-001", config=scan_config)
        memory.set_target(target)
        memory.add_endpoint(endpoint, discovered_by="crawl")
        memory.record_attack(attack_result)
        memory.add_vulnerability(vuln)

        # Get context for LLM prompts
        context = memory.build_attack_context(endpoint)
    """

    def __init__(self, scan_id: str, config: ScanConfig) -> None:
        self._scan_id = scan_id
        self._config = config
        self._lock = asyncio.Lock()

        # Target knowledge
        self._target: Target | None = None
        self._tech_stack: TechStack = TechStack()
        self._crawl_data: list[dict] = []

        # Endpoint registry — keyed by "METHOD /path"
        self._endpoints: dict[str, EndpointEntry] = {}

        # Attack ledger — complete record of what we tried
        self._attacks: list[AttackEntry] = []

        # Vulnerability store
        self._vulnerabilities: list[Vulnerability] = []

        # AI reasoning journal
        self._reasoning: list[ReasoningEntry] = []

        # Stats
        self._total_requests_sent: int = 0
        self._scan_start: datetime = datetime.now()

        logger.info(f"Scan memory initialized — scan_id={scan_id}")

    # ===================================================================
    # Properties
    # ===================================================================

    @property
    def scan_id(self) -> str:
        return self._scan_id

    @property
    def config(self) -> ScanConfig:
        return self._config

    @property
    def target(self) -> Target | None:
        return self._target

    @property
    def tech_stack(self) -> TechStack:
        return self._tech_stack

    @property
    def endpoint_count(self) -> int:
        return len(self._endpoints)

    @property
    def vulnerability_count(self) -> int:
        return len(self._vulnerabilities)

    @property
    def attack_count(self) -> int:
        return len(self._attacks)

    @property
    def total_requests_sent(self) -> int:
        return self._total_requests_sent

    @property
    def duration_seconds(self) -> float:
        return (datetime.now() - self._scan_start).total_seconds()

    # ===================================================================
    # Target Management
    # ===================================================================

    def set_target(self, target: Target) -> None:
        """Set the scan target and initialize tech stack."""
        self._target = target
        if target.tech_stack:
            self._tech_stack = target.tech_stack

        # Auto-register any endpoints already on the target
        for ep in target.endpoints:
            self.add_endpoint(ep, discovered_by="initial")

        logger.info(
            f"Target set: {target.url} "
            f"({len(target.endpoints)} initial endpoints)"
        )

    def update_tech_stack(self, tech: TechStack) -> None:
        """Update the known tech stack (from fingerprinting or AI analysis)."""
        # Merge — don't overwrite fields that already have values
        if tech.language and not self._tech_stack.language:
            self._tech_stack.language = tech.language
        if tech.framework and not self._tech_stack.framework:
            self._tech_stack.framework = tech.framework
        if tech.database and not self._tech_stack.database:
            self._tech_stack.database = tech.database
        if tech.web_server and not self._tech_stack.web_server:
            self._tech_stack.web_server = tech.web_server
        if tech.template_engine and not self._tech_stack.template_engine:
            self._tech_stack.template_engine = tech.template_engine
        if tech.other:
            # Deduplicate
            existing = set(self._tech_stack.other)
            for item in tech.other:
                if item not in existing:
                    self._tech_stack.other.append(item)

        # Update the target model too
        if self._target:
            self._target.tech_stack = self._tech_stack

        logger.debug(
            f"Tech stack updated: lang={self._tech_stack.language}, "
            f"framework={self._tech_stack.framework}, "
            f"db={self._tech_stack.database}"
        )

    # ===================================================================
    # Endpoint Registry
    # ===================================================================

    def add_endpoint(
        self,
        endpoint: Endpoint,
        discovered_by: str = "unknown",
    ) -> EndpointEntry:
        """
        Register a discovered endpoint.
        Deduplicates by METHOD+PATH. Returns the entry.
        """
        key = f"{endpoint.method} {endpoint.path}"

        if key in self._endpoints:
            # Merge parameters if new ones are discovered
            existing = self._endpoints[key]
            existing_param_names = {p.name for p in existing.endpoint.parameters}
            for param in endpoint.parameters:
                if param.name not in existing_param_names:
                    existing.endpoint.parameters.append(param)
            logger.debug(f"Endpoint already known, merged params: {key}")
            return existing

        # Score and prioritize
        priority, score = self._score_endpoint(endpoint)

        entry = EndpointEntry(
            endpoint=endpoint,
            priority=priority,
            priority_score=score,
            discovered_by=discovered_by,
        )
        self._endpoints[key] = entry

        # Also update the target's endpoint list
        if self._target:
            target_keys = {f"{e.method} {e.path}" for e in self._target.endpoints}
            if key not in target_keys:
                self._target.endpoints.append(endpoint)

        logger.debug(
            f"Endpoint registered: {key} "
            f"(priority={priority.value}, score={score:.1f}, "
            f"params={len(endpoint.parameters)}, by={discovered_by})"
        )
        return entry

    def add_endpoints_from_crawl(self, crawl_results: list[dict]) -> int:
        """
        Bulk-register endpoints from crawler output.
        Returns the number of NEW endpoints added.
        """
        new_count = 0
        for cr in crawl_results:
            self._crawl_data.append(cr)
            url = cr.get("url", "")
            method = cr.get("method", "GET")

            # Extract path from URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path = parsed.path or "/"

            ep = Endpoint(path=path, method=method)

            # Add form fields as parameters
            for form in cr.get("forms", []):
                for field in form.get("fields", []):
                    ep.parameters.append(Parameter(
                        name=field.get("name", ""),
                        location="body" if form.get("method", "").upper() == "POST" else "query",
                        param_type=field.get("type", "string"),
                        required=field.get("required", False),
                    ))

            key = f"{ep.method} {ep.path}"
            if key not in self._endpoints:
                new_count += 1
            self.add_endpoint(ep, discovered_by="crawl")

        logger.info(f"Crawl data ingested: {new_count} new endpoints from {len(crawl_results)} pages")
        return new_count

    def get_endpoint(self, method: str, path: str) -> EndpointEntry | None:
        """Look up an endpoint by METHOD and PATH."""
        return self._endpoints.get(f"{method} {path}")

    def get_untested_endpoints(self, attack_type: str | None = None) -> list[EndpointEntry]:
        """
        Get endpoints that haven't been fully tested yet.
        If attack_type is given, only returns endpoints that haven't
        been tested with that specific attack type.
        Sorted by priority score (highest first).
        """
        untested = []
        for entry in self._endpoints.values():
            if entry.priority == EndpointPriority.SKIP:
                continue
            if attack_type:
                if attack_type not in entry.attack_types_tested:
                    untested.append(entry)
            else:
                if not entry.is_fully_tested:
                    untested.append(entry)

        # Sort by priority score descending
        untested.sort(key=lambda e: e.priority_score, reverse=True)
        return untested

    def get_all_endpoints(self) -> list[EndpointEntry]:
        """Get all registered endpoints sorted by priority."""
        entries = list(self._endpoints.values())
        entries.sort(key=lambda e: e.priority_score, reverse=True)
        return entries

    def mark_endpoint_baseline(
        self,
        method: str,
        path: str,
        status_code: int,
        content_length: int,
        response_time_ms: float,
    ) -> None:
        """Record baseline response data for an endpoint."""
        entry = self.get_endpoint(method, path)
        if entry:
            entry.is_baseline_captured = True
            entry.baseline_status_code = status_code
            entry.baseline_content_length = content_length
            entry.baseline_response_time_ms = response_time_ms
            logger.debug(
                f"Baseline captured: {method} {path} → "
                f"{status_code}, {content_length}B, {response_time_ms:.0f}ms"
            )

    # ===================================================================
    # Attack Ledger
    # ===================================================================

    def record_attack(
        self,
        attack_result: AttackResult,
        endpoint_key: str,
        notes: str = "",
    ) -> AttackEntry:
        """
        Record a completed attack attempt in the ledger.
        Also marks the attack type as tested on the endpoint.
        """
        entry = AttackEntry(
            attack_id=attack_result.attack_id,
            attack_type=attack_result.attack_type,
            endpoint_key=endpoint_key,
            payloads_sent=attack_result.total_requests,
            interesting_responses=len([
                r for r in attack_result.payload_results
                if r.error is None and r.status_code != 200
            ]),
            duration_ms=attack_result.duration_ms,
            notes=notes,
        )

        self._attacks.append(entry)
        self._total_requests_sent += attack_result.total_requests

        # Mark this attack type as tested on the endpoint
        ep_entry = self._endpoints.get(endpoint_key)
        if ep_entry and attack_result.attack_type not in ep_entry.attack_types_tested:
            ep_entry.attack_types_tested.append(attack_result.attack_type)

        logger.debug(
            f"Attack recorded: {attack_result.attack_type} on {endpoint_key} "
            f"({attack_result.total_requests} payloads, "
            f"{entry.interesting_responses} interesting)"
        )
        return entry

    def get_attacks_for_endpoint(self, endpoint_key: str) -> list[AttackEntry]:
        """Get all attack records for a specific endpoint."""
        return [a for a in self._attacks if a.endpoint_key == endpoint_key]

    def was_attack_tried(self, endpoint_key: str, attack_type: str) -> bool:
        """Check if a specific attack type was already tried on an endpoint."""
        return any(
            a.endpoint_key == endpoint_key and a.attack_type == attack_type
            for a in self._attacks
        )

    # ===================================================================
    # Vulnerability Store
    # ===================================================================

    def add_vulnerability(self, vuln: Vulnerability) -> None:
        """
        Record a confirmed vulnerability.
        Auto-generates an ID if not set.
        """
        if not vuln.id:
            vuln.id = f"WRAITH-{self._scan_id[:8]}-{len(self._vulnerabilities) + 1:03d}"

        # Dedup check — same type + endpoint
        for existing in self._vulnerabilities:
            if (existing.vuln_type == vuln.vuln_type and
                    existing.endpoint == vuln.endpoint and
                    existing.method == vuln.method):
                logger.debug(
                    f"Duplicate vulnerability skipped: "
                    f"{vuln.vuln_type.value} on {vuln.method} {vuln.endpoint}"
                )
                return

        self._vulnerabilities.append(vuln)

        # Link back to the attack entry
        endpoint_key = f"{vuln.method} {vuln.endpoint}"
        for attack in reversed(self._attacks):
            if (attack.endpoint_key == endpoint_key and
                    attack.attack_type == vuln.vuln_type.value):
                attack.confirmed_vuln = True
                attack.vuln_id = vuln.id
                break

        logger.info(
            f"🔴 Vulnerability confirmed: [{vuln.severity.value.upper()}] "
            f"{vuln.title} ({vuln.id})"
        )

    def get_vulnerabilities(
        self,
        severity: Severity | None = None,
        vuln_type: VulnerabilityType | None = None,
    ) -> list[Vulnerability]:
        """Get vulnerabilities, optionally filtered."""
        results = self._vulnerabilities
        if severity:
            results = [v for v in results if v.severity == severity]
        if vuln_type:
            results = [v for v in results if v.vuln_type == vuln_type]
        return results

    def get_vulnerable_endpoints(self) -> set[str]:
        """Get endpoint keys that have confirmed vulnerabilities."""
        return {f"{v.method} {v.endpoint}" for v in self._vulnerabilities}

    # ===================================================================
    # Reasoning Journal
    # ===================================================================

    def log_reasoning(
        self,
        phase: str,
        action: str,
        reasoning: str,
        decision: str,
        outcome: str = "",
    ) -> None:
        """Record an AI reasoning step in the journal."""
        entry = ReasoningEntry(
            phase=phase,
            action=action,
            reasoning=reasoning,
            decision=decision,
            outcome=outcome,
        )
        self._reasoning.append(entry)
        logger.debug(f"Reasoning logged: [{phase}] {action} → {decision}")

    def update_last_reasoning_outcome(self, outcome: str) -> None:
        """Update the outcome of the most recent reasoning entry."""
        if self._reasoning:
            self._reasoning[-1].outcome = outcome

    def get_recent_reasoning(self, count: int = 5) -> list[ReasoningEntry]:
        """Get the most recent reasoning entries."""
        return self._reasoning[-count:]

    # ===================================================================
    # Context Engine — Builds Formatted Strings for LLM Prompts
    # ===================================================================
    # This is the critical harness layer. Every LLM call gets memory
    # context injected via these methods.

    def build_target_context(self) -> str:
        """
        Build a context string describing what we know about the target.
        Used in virtually every LLM prompt.
        """
        if not self._target:
            return "## Target\nNo target information available yet."

        parts = [
            "## Target Information",
            f"- **URL**: {self._target.url}",
            f"- **Source code**: {'Available' if self._target.has_source else 'Not available'}",
        ]

        ts = self._tech_stack
        if ts.language or ts.framework or ts.database or ts.web_server:
            parts.append("")
            parts.append("### Detected Technology Stack")
            if ts.language:
                parts.append(f"- **Language**: {ts.language}")
            if ts.framework:
                parts.append(f"- **Framework**: {ts.framework}")
            if ts.database:
                parts.append(f"- **Database**: {ts.database}")
            if ts.web_server:
                parts.append(f"- **Web Server**: {ts.web_server}")
            if ts.template_engine:
                parts.append(f"- **Template Engine**: {ts.template_engine}")
            if ts.auth_mechanism and ts.auth_mechanism.value != "unknown":
                parts.append(f"- **Auth**: {ts.auth_mechanism.value}")
            if ts.other:
                parts.append(f"- **Other**: {', '.join(ts.other)}")

        parts.append("")
        parts.append(
            f"### Scan Progress: {self.endpoint_count} endpoints, "
            f"{self.attack_count} attacks, "
            f"{self.vulnerability_count} vulnerabilities"
        )

        return "\n".join(parts)

    def build_endpoint_context(self, max_chars: int = MAX_ENDPOINT_CONTEXT) -> str:
        """
        Build a context string listing all known endpoints.
        Sorted by priority. Truncated to fit context window.
        """
        endpoints = self.get_all_endpoints()
        if not endpoints:
            return "## Endpoints\nNo endpoints discovered yet."

        parts = [f"## Discovered Endpoints ({len(endpoints)} total)"]
        vuln_eps = self.get_vulnerable_endpoints()
        char_count = len(parts[0])

        for entry in endpoints:
            ep = entry.endpoint
            line = f"- `{ep.method} {ep.path}`"

            # Add parameter info
            if ep.parameters:
                param_names = [p.name for p in ep.parameters[:5]]
                line += f" — params: {', '.join(param_names)}"
                if len(ep.parameters) > 5:
                    line += f" (+{len(ep.parameters) - 5} more)"

            # Add testing status
            if entry.key in vuln_eps:
                line += " ⚠️ VULNERABLE"
            elif entry.is_fully_tested:
                line += " ✓ tested"
            elif entry.attack_types_tested:
                line += f" (tested: {', '.join(entry.attack_types_tested)})"
            else:
                line += f" [{entry.priority.value}]"

            if char_count + len(line) + 1 > max_chars:
                parts.append(f"... and {len(endpoints) - len(parts) + 1} more endpoints")
                break

            parts.append(line)
            char_count += len(line) + 1

        return "\n".join(parts)

    def build_attack_history_context(
        self,
        endpoint_key: str | None = None,
        max_chars: int = MAX_ATTACK_CONTEXT,
    ) -> str:
        """
        Build a context string summarizing attack history.
        If endpoint_key is given, shows history for that endpoint only.
        """
        attacks = (
            self.get_attacks_for_endpoint(endpoint_key)
            if endpoint_key
            else self._attacks
        )

        if not attacks:
            scope = f"on {endpoint_key}" if endpoint_key else "yet"
            return f"## Attack History\nNo attacks executed {scope}."

        # Show most recent first
        recent = attacks[-15:]  # Last 15 attacks
        scope = f"on `{endpoint_key}`" if endpoint_key else ""
        parts = [f"## Attack History {scope} ({len(attacks)} total)"]
        char_count = len(parts[0])

        for entry in reversed(recent):
            vuln_flag = " → ⚠️ VULNERABLE" if entry.confirmed_vuln else ""
            line = (
                f"- [{entry.attack_type}] {entry.endpoint_key} — "
                f"{entry.payloads_sent} payloads, "
                f"{entry.interesting_responses} interesting"
                f"{vuln_flag}"
            )
            if entry.notes:
                line += f" | {entry.notes}"

            if char_count + len(line) + 1 > max_chars:
                break
            parts.append(line)
            char_count += len(line) + 1

        return "\n".join(parts)

    def build_vulnerability_context(self, max_chars: int = MAX_VULN_CONTEXT) -> str:
        """
        Build a context string summarizing all confirmed vulnerabilities.
        The AI uses this to avoid re-testing and to find attack chains.
        """
        vulns = self._vulnerabilities
        if not vulns:
            return "## Confirmed Vulnerabilities\nNone confirmed yet."

        # Group by severity
        by_severity = {"critical": [], "high": [], "medium": [], "low": [], "info": []}
        for v in vulns:
            by_severity[v.severity.value].append(v)

        parts = [f"## Confirmed Vulnerabilities ({len(vulns)} total)"]
        char_count = len(parts[0])

        for sev in ["critical", "high", "medium", "low", "info"]:
            group = by_severity[sev]
            if not group:
                continue

            parts.append(f"\n### {sev.upper()} ({len(group)})")
            for v in group:
                line = (
                    f"- **{v.title}** — `{v.method} {v.endpoint}` "
                    f"[{v.vuln_type.value}]"
                )
                if v.cvss_score:
                    line += f" (CVSS: {v.cvss_score})"

                if char_count + len(line) + 1 > max_chars:
                    parts.append("... additional vulnerabilities truncated")
                    return "\n".join(parts)

                parts.append(line)
                char_count += len(line) + 1

        return "\n".join(parts)

    def build_reasoning_context(self, max_chars: int = MAX_REASONING_CONTEXT) -> str:
        """
        Build a context string with the AI's recent reasoning history.
        Helps the AI stay consistent and build on previous thoughts.
        """
        recent = self.get_recent_reasoning(count=8)
        if not recent:
            return ""

        parts = ["## Recent AI Reasoning"]
        char_count = len(parts[0])

        for entry in recent:
            line = (
                f"- [{entry.phase}] **{entry.action}**: {entry.decision}"
            )
            if entry.outcome:
                line += f" → {entry.outcome}"

            if char_count + len(line) + 1 > max_chars:
                break
            parts.append(line)
            char_count += len(line) + 1

        return "\n".join(parts)

    def build_full_context(self, max_chars: int = MAX_FULL_CONTEXT) -> str:
        """
        Build the complete context string for a general LLM prompt.
        Combines all context sections, prioritized by importance.
        """
        sections = [
            self.build_target_context(),
            self.build_vulnerability_context(),
            self.build_endpoint_context(),
            self.build_attack_history_context(),
            self.build_reasoning_context(),
        ]

        full = "\n\n".join(s for s in sections if s)

        if len(full) > max_chars:
            full = full[:max_chars - 50] + "\n\n... [context truncated for brevity]"

        return full

    def build_attack_planning_context(self, endpoint_key: str) -> str:
        """
        Build context specifically for planning an attack on an endpoint.
        Includes: target info, endpoint details, prior attacks on this
        endpoint, and any related vulnerabilities found elsewhere.
        """
        entry = self._endpoints.get(endpoint_key)
        if not entry:
            return f"No information available for endpoint: {endpoint_key}"

        ep = entry.endpoint
        parts = [
            self.build_target_context(),
            "",
            f"## Current Attack Target: `{endpoint_key}`",
            f"- **Priority**: {entry.priority.value} (score: {entry.priority_score:.1f})",
            f"- **Auth required**: {ep.auth_required}",
        ]

        if ep.parameters:
            parts.append("- **Parameters**:")
            for p in ep.parameters:
                parts.append(
                    f"  - `{p.name}` ({p.location}, {p.param_type})"
                    + (f" = {p.example_value}" if p.example_value else "")
                    + (" [required]" if p.required else "")
                )

        if ep.source_file:
            parts.append(f"- **Source**: {ep.source_file}:{ep.source_line or '?'}")

        if entry.is_baseline_captured:
            parts.append(
                f"- **Baseline**: {entry.baseline_status_code}, "
                f"{entry.baseline_content_length}B, "
                f"{entry.baseline_response_time_ms:.0f}ms"
            )

        # What we've tried on this endpoint
        prior = self.build_attack_history_context(endpoint_key=endpoint_key)
        if "No attacks" not in prior:
            parts.extend(["", prior])

        # What's left to try
        untested = entry.untested_attack_types
        if untested:
            parts.extend([
                "",
                f"### Untested Attack Types: {', '.join(untested)}",
            ])

        # Related vulns on similar endpoints
        related = self._find_related_vulnerabilities(endpoint_key)
        if related:
            parts.extend(["", "### Related Findings on Similar Endpoints"])
            for v in related[:3]:
                parts.append(
                    f"- {v.vuln_type.value} on `{v.method} {v.endpoint}` "
                    f"— might indicate similar pattern here"
                )

        return "\n".join(parts)

    # ===================================================================
    # Summary & Stats
    # ===================================================================

    def get_summary(self) -> dict[str, Any]:
        """Get a summary dict for CLI display and logging."""
        vuln_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for v in self._vulnerabilities:
            vuln_counts[v.severity.value] += 1

        tested = sum(1 for e in self._endpoints.values() if e.attack_types_tested)
        untested = sum(1 for e in self._endpoints.values() if not e.attack_types_tested)

        return {
            "scan_id": self._scan_id,
            "target": self._target.url if self._target else "not set",
            "mode": self._config.mode.value,
            "duration_seconds": round(self.duration_seconds, 1),
            "endpoints": {
                "total": self.endpoint_count,
                "tested": tested,
                "untested": untested,
            },
            "attacks": {
                "total": self.attack_count,
                "requests_sent": self._total_requests_sent,
            },
            "vulnerabilities": {
                "total": self.vulnerability_count,
                **vuln_counts,
            },
            "tech_stack": {
                "language": self._tech_stack.language,
                "framework": self._tech_stack.framework,
                "database": self._tech_stack.database,
                "web_server": self._tech_stack.web_server,
            },
            "reasoning_steps": len(self._reasoning),
        }

    def get_progress_display(self) -> str:
        """One-line progress string for CLI live display."""
        ep_total = self.endpoint_count
        tested = sum(1 for e in self._endpoints.values() if e.attack_types_tested)
        vulns = self.vulnerability_count
        dur = self.duration_seconds

        return (
            f"Endpoints: {tested}/{ep_total} tested | "
            f"Attacks: {self.attack_count} | "
            f"Vulns: {vulns} | "
            f"Requests: {self._total_requests_sent} | "
            f"Time: {dur:.0f}s"
        )

    # ===================================================================
    # Internal: Endpoint Priority Scoring
    # ===================================================================

    def _score_endpoint(self, endpoint: Endpoint) -> tuple[EndpointPriority, float]:
        """
        Calculate a priority score for an endpoint.

        Higher scores = more likely to be vulnerable = test first.

        Scoring factors:
        - URL pattern matches (login, admin, upload, etc.)
        - HTTP method (POST/PUT > GET)
        - Number of parameters (more params = more injection points)
        - Auth requirement (auth endpoints are high-value targets)
        - Tech stack awareness (e.g., PHP + query params = SQLi likely)
        """
        score = 0.0
        path_lower = endpoint.path.lower()

        # Pattern matching (+3 per matching pattern)
        for pattern in _HIGH_PRIORITY_PATTERNS:
            if pattern in path_lower:
                score += 3.0
                break  # Only count once

        # HTTP method score
        score += _METHOD_PRIORITY.get(endpoint.method.upper(), 1)

        # Parameter count (each param is an injection point)
        param_count = len(endpoint.parameters)
        score += min(param_count * 1.5, 6.0)  # Cap at 6

        # Body parameters are especially interesting
        body_params = [p for p in endpoint.parameters if p.location == "body"]
        score += len(body_params) * 0.5

        # Auth endpoints are high-value
        if endpoint.auth_required:
            score += 2.0

        # Tech-stack-aware scoring
        if self._tech_stack.language:
            lang = self._tech_stack.language.lower()
            # PHP is historically more vulnerable to injection
            if lang == "php" and param_count > 0:
                score += 2.0
            # Java endpoints with path params might have deserialization
            if lang == "java" and any(p.location == "body" for p in endpoint.parameters):
                score += 1.5

        # Determine priority level
        if score >= 8.0:
            priority = EndpointPriority.CRITICAL
        elif score >= 5.0:
            priority = EndpointPriority.HIGH
        elif score >= 2.5:
            priority = EndpointPriority.MEDIUM
        else:
            priority = EndpointPriority.LOW

        return priority, score

    # ===================================================================
    # Internal: Finding Correlation
    # ===================================================================

    def _find_related_vulnerabilities(self, endpoint_key: str) -> list[Vulnerability]:
        """
        Find vulnerabilities on endpoints that are structurally similar
        to the given endpoint. Used for attack planning — if SQLi was
        found on /api/users, it's worth testing /api/products too.
        """
        parts = endpoint_key.split(" ", 1)
        if len(parts) != 2:
            return []

        _, target_path = parts
        target_segments = set(target_path.strip("/").split("/"))

        related = []
        for vuln in self._vulnerabilities:
            vuln_segments = set(vuln.endpoint.strip("/").split("/"))
            # Check if they share path structure (e.g., both under /api/)
            shared = target_segments & vuln_segments
            if shared and vuln.endpoint != target_path:
                related.append(vuln)

        return related

    # ===================================================================
    # Export / Serialization
    # ===================================================================

    def export_for_report(self) -> dict[str, Any]:
        """Export memory contents in a format suitable for report generation."""
        return {
            "scan_id": self._scan_id,
            "target": self._target.model_dump() if self._target else {},
            "tech_stack": self._tech_stack.model_dump(),
            "endpoints": [
                {
                    "path": entry.endpoint.path,
                    "method": entry.endpoint.method,
                    "parameters": [p.model_dump() for p in entry.endpoint.parameters],
                    "priority": entry.priority.value,
                    "attacks_tested": entry.attack_types_tested,
                    "baseline_status": entry.baseline_status_code,
                }
                for entry in self.get_all_endpoints()
            ],
            "vulnerabilities": [
                v.model_dump() for v in self._vulnerabilities
            ],
            "summary": self.get_summary(),
        }

    def reset(self) -> None:
        """Clear all memory. Used for test isolation."""
        self._target = None
        self._tech_stack = TechStack()
        self._crawl_data.clear()
        self._endpoints.clear()
        self._attacks.clear()
        self._vulnerabilities.clear()
        self._reasoning.clear()
        self._total_requests_sent = 0
        self._scan_start = datetime.now()
        logger.info("Scan memory reset")
