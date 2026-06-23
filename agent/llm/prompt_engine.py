"""
WRAITH Prompt Engine

Constructs rich, context-aware prompts for every phase of a penetration test.
This module sits between the orchestrator and the LLM client — it takes raw
scan data (target info, tech stack, findings) and builds structured prompts
that guide the AI to produce useful, actionable output.

The prompt engine does NOT call the LLM directly. It builds prompt strings
that are then passed to LLMClient.generate() or LLMClient.chat().

Architecture:
    Orchestrator → PromptEngine.build_*() → prompt string → LLMClient

Responsibilities:
    - Build phase-specific prompts (recon, analysis, attack planning, etc.)
    - Inject target context (tech stack, endpoints, findings) into prompts
    - Manage conversation history for multi-turn interactions
    - Enforce context window limits (truncation when needed)
    - Format prompts for structured output parsing (JSON, lists, etc.)
"""

from typing import Any

from config import settings, get_model_config
from models.target import Target, Endpoint
from models.vulnerability import Vulnerability, Severity
from models.scan import ScanConfig, ScanMode
from utils.logger import get_logger
from utils.exception import WraithError

logger = get_logger("llm.prompt_engine")


# ===========================================================================
# Constants
# ===========================================================================

# Maximum characters per prompt section to prevent context overflow
_MAX_ENDPOINT_CHARS = 3000
_MAX_FINDINGS_CHARS = 4000
_MAX_CODE_CHARS = 8000
_MAX_HISTORY_CHARS = 4000

# Output format markers the LLM should use
_JSON_INSTRUCTION = (
    "Respond ONLY with valid JSON. No markdown, no explanation, no code fences. "
    "Start your response with { and end with }."
)

_LIST_INSTRUCTION = (
    "Respond with a numbered list. One item per line. "
    "Format: 1. <item>"
)

_SECTION_DIVIDER = "\n" + "=" * 60 + "\n"


# ===========================================================================
# PromptEngine
# ===========================================================================
class PromptEngine:
    """
    Builds context-rich prompts for every phase of a WRAITH penetration test.

    The engine maintains awareness of the current scan state and injects
    relevant context into every prompt — target info, tech stack,
    previously discovered vulnerabilities, and endpoint data.

    Attributes:
        _conversation_history: Running history of LLM interactions.
        _max_history: Maximum number of history entries to retain.
        _include_tech_context: Whether to inject tech stack info.
        _include_findings_context: Whether to inject prior findings.
    """

    def __init__(self) -> None:
        """Initialize the Prompt Engine with defaults from config."""
        # Load prompt settings from models.yaml if available
        try:
            from config import _models_yaml
            prompt_settings = _models_yaml.get("prompt_settings", {})
        except Exception:
            prompt_settings = {}

        self._max_history: int = prompt_settings.get("max_conversation_history", 10)
        self._truncation_strategy: str = prompt_settings.get(
            "truncation_strategy", "oldest_first"
        )
        self._include_tech_context: bool = prompt_settings.get(
            "include_tech_context", True
        )
        self._include_findings_context: bool = prompt_settings.get(
            "include_findings_context", True
        )
        self._prefer_json: bool = prompt_settings.get("prefer_json_output", False)

        # Conversation history for multi-turn interactions
        self._conversation_history: list[dict[str, str]] = []

        logger.debug(
            f"Prompt Engine initialized — "
            f"Max history: {self._max_history}, "
            f"Tech context: {self._include_tech_context}, "
            f"Findings context: {self._include_findings_context}"
        )

    # ===================================================================
    # PUBLIC: Reconnaissance Prompts
    # ===================================================================
    def build_recon_prompt(self, target: Target) -> str:
        """
        Build a prompt for the reconnaissance phase.

        Asks the AI to analyze the target and identify:
        - Technology stack
        - Potential attack vectors
        - Interesting endpoints
        - Authentication mechanisms
        - Known vulnerability patterns for the detected stack

        Args:
            target: Target being scanned.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [
            "# RECONNAISSANCE ANALYSIS",
            "",
            "Analyze the following target and provide a comprehensive "
            "reconnaissance report.",
            "",
            self._format_target_context(target),
            "",
            "## Your Tasks:",
            "",
            "1. **Technology Identification**: Identify the web framework, "
            "programming language, database, template engine, and web server.",
            "",
            "2. **Attack Surface Mapping**: List all potential attack vectors "
            "based on the technology stack and available endpoints.",
            "",
            "3. **Authentication Analysis**: Identify the authentication "
            "mechanism (JWT, session, OAuth, etc.) and potential weaknesses.",
            "",
            "4. **Priority Targets**: Rank the top 5 endpoints most likely "
            "to contain vulnerabilities, with reasoning.",
            "",
            "5. **Known Vulnerabilities**: List known CVEs or vulnerability "
            "patterns associated with the detected technology stack.",
            "",
            "## Output Format:",
            "",
            "Structure your response with clear headers for each section. "
            "Be specific — reference exact endpoints, parameters, and "
            "technology versions where possible.",
        ]

        prompt = "\n".join(prompt_parts)
        logger.debug(f"Built recon prompt: {len(prompt)} chars")
        return prompt

    # ===================================================================
    # PUBLIC: Code Analysis Prompts
    # ===================================================================
    def build_code_analysis_prompt(
        self,
        source_code: str,
        file_path: str,
        language: str = "unknown",
        focus: str | None = None,
    ) -> str:
        """
        Build a prompt for analyzing source code for vulnerabilities.

        Asks the AI to perform a security-focused code review,
        tracing data flow from user inputs to dangerous sinks.

        Args:
            source_code: The source code to analyze.
            file_path: Path to the source file.
            language: Programming language of the code.
            focus: Optional specific vulnerability type to focus on
                  (e.g., "sqli", "xss"). If None, checks for all types.

        Returns:
            Formatted prompt string.
        """
        # Truncate code if too long
        truncated_code = self._truncate(source_code, _MAX_CODE_CHARS)
        was_truncated = len(source_code) > _MAX_CODE_CHARS

        prompt_parts = [
            "# SOURCE CODE SECURITY ANALYSIS",
            "",
            f"**File**: `{file_path}`",
            f"**Language**: {language}",
        ]

        if was_truncated:
            prompt_parts.append(
                f"**Note**: Code truncated to {_MAX_CODE_CHARS} characters. "
                "Focus on the visible portion."
            )

        if focus:
            prompt_parts.extend([
                "",
                f"**Focus Area**: Specifically analyze for **{focus}** vulnerabilities.",
            ])

        prompt_parts.extend([
            "",
            "## Source Code:",
            "",
            f"```{language}",
            truncated_code,
            "```",
            "",
            "## Your Tasks:",
            "",
            "1. **Vulnerability Detection**: Identify ALL security vulnerabilities "
            "in this code. For each vulnerability:",
            "   - Type (SQLi, XSS, SSRF, etc.)",
            "   - Exact line number(s)",
            "   - Severity (Critical/High/Medium/Low)",
            "   - Explanation of why it's vulnerable",
            "",
            "2. **Data Flow Tracing**: For each vulnerability, trace the data flow:",
            "   - SOURCE: Where does user input enter?",
            "   - TRANSFORMATIONS: How is the data processed?",
            "   - SINK: Where does it reach a dangerous function?",
            "",
            "3. **Exploit Scenario**: For each vulnerability, describe a realistic "
            "attack scenario with an example payload.",
            "",
            "4. **Remediation**: For each vulnerability, provide the exact code fix. "
            "Write real, runnable code in the same language and framework.",
        ])

        prompt = "\n".join(prompt_parts)
        logger.debug(f"Built code analysis prompt: {len(prompt)} chars for {file_path}")
        return prompt

    # ===================================================================
    # PUBLIC: Attack Planning Prompts
    # ===================================================================
    def build_attack_plan_prompt(
        self,
        target: Target,
        scan_config: ScanConfig | None = None,
        findings: list[Vulnerability] | None = None,
    ) -> str:
        """
        Build a prompt for the AI to plan the attack strategy.

        Given the target info and any existing findings, asks the AI
        to create an ordered attack plan prioritized by risk.

        Args:
            target: Target being scanned.
            scan_config: Current scan configuration.
            findings: Previously discovered vulnerabilities (if any).

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [
            "# ATTACK STRATEGY PLANNING",
            "",
            "Based on the target information below, create a prioritized "
            "attack plan. Think like a senior penetration tester — focus on "
            "the highest-impact, most likely vulnerabilities first.",
            "",
            self._format_target_context(target),
        ]

        # Add scan mode context
        if scan_config:
            prompt_parts.extend([
                "",
                f"**Scan Mode**: {scan_config.mode.value}",
                f"**Time Budget**: {scan_config.max_duration_minutes} minutes",
            ])

            if scan_config.enabled_attacks:
                prompt_parts.append(
                    f"**Enabled Attacks**: {', '.join(scan_config.enabled_attacks)}"
                )

        # Add existing findings context
        if findings and self._include_findings_context:
            prompt_parts.extend([
                "",
                self._format_findings_context(findings),
            ])

        prompt_parts.extend([
            "",
            "## Your Tasks:",
            "",
            "1. **Attack Prioritization**: List attacks in order of priority. "
            "For each attack:",
            "   - Attack type (SQLi, XSS, SSRF, etc.)",
            "   - Target endpoint and parameter",
            "   - Why this is a high-priority target",
            "   - Expected severity if successful",
            "",
            "2. **Attack Chains**: Identify potential multi-step attack chains. "
            "Example: SSRF → internal API access → SQLi → data extraction.",
            "",
            "3. **Bypasses**: For each attack, suggest potential WAF/filter "
            "bypass techniques based on the detected technology stack.",
            "",
            "4. **Time Allocation**: Given the time budget, recommend how "
            "many minutes to spend on each attack category.",
            "",
            "Be specific. Reference exact endpoints and parameters from "
            "the target information above.",
        ])

        prompt = "\n".join(prompt_parts)
        logger.debug(f"Built attack plan prompt: {len(prompt)} chars")
        return prompt

    # ===================================================================
    # PUBLIC: Payload Generation Prompts
    # ===================================================================
    def build_payload_prompt(
        self,
        attack_type: str,
        target: Target,
        endpoint: Endpoint,
        parameter_name: str,
        context_clues: str | None = None,
    ) -> str:
        """
        Build a prompt for generating context-aware attack payloads.

        Unlike static wordlists, this asks the AI to generate payloads
        specifically tailored to the target's tech stack, endpoint
        behavior, and parameter context.

        Args:
            attack_type: Type of attack (e.g., "sqli", "xss", "ssti").
            target: Target being scanned.
            endpoint: Specific endpoint to target.
            parameter_name: Parameter to inject into.
            context_clues: Any additional context (e.g., error messages,
                          response patterns, source code snippets).

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [
            f"# PAYLOAD GENERATION: {attack_type.upper()}",
            "",
            "Generate targeted attack payloads for the following context. "
            "These payloads should be specifically crafted for this target — "
            "not generic payloads from a wordlist.",
            "",
            "## Target Context:",
            "",
            f"- **URL**: {target.url}",
            f"- **Endpoint**: {endpoint.method} {endpoint.path}",
            f"- **Parameter**: `{parameter_name}` "
            f"(location: {self._get_param_location(endpoint, parameter_name)})",
        ]

        # Add tech stack context
        if target.tech_stack.framework:
            prompt_parts.append(f"- **Framework**: {target.tech_stack.framework}")
        if target.tech_stack.database:
            prompt_parts.append(f"- **Database**: {target.tech_stack.database}")
        if target.tech_stack.template_engine:
            prompt_parts.append(f"- **Template Engine**: {target.tech_stack.template_engine}")
        if target.tech_stack.language:
            prompt_parts.append(f"- **Language**: {target.tech_stack.language}")

        if context_clues:
            prompt_parts.extend([
                "",
                "## Additional Context:",
                "",
                context_clues,
            ])

        prompt_parts.extend([
            "",
            "## Requirements:",
            "",
            "Generate 15-20 payloads organized by technique:",
            "",
            "1. **Basic Detection**: Simple payloads to confirm the vulnerability exists.",
            "2. **Filter Bypass**: Payloads designed to bypass common filters, WAFs, "
            "and input validation for this specific tech stack.",
            "3. **Exploitation**: Payloads that demonstrate real impact "
            "(data extraction, code execution, etc.).",
            "",
            "## Output Format:",
            "",
            "For each payload:",
            "- The payload string (ready to use, properly encoded if needed)",
            "- Which technique category it belongs to",
            "- What response to look for to confirm success",
            "- Brief explanation of why it works against this target",
            "",
            "Format as a numbered list. Put each payload on its own line "
            "wrapped in backticks.",
        ])

        prompt = "\n".join(prompt_parts)
        logger.debug(
            f"Built payload prompt: {attack_type} → "
            f"{endpoint.method} {endpoint.path} [{parameter_name}]"
        )
        return prompt

    # ===================================================================
    # PUBLIC: Result Analysis Prompts
    # ===================================================================
    def build_result_analysis_prompt(
        self,
        attack_type: str,
        endpoint: str,
        payloads_and_responses: list[dict[str, str]],
        baseline_response: str | None = None,
    ) -> str:
        """
        Build a prompt for the AI to analyze attack results.

        Given the payloads sent and responses received, asks the AI
        to determine which (if any) payloads successfully exploited
        a vulnerability and to eliminate false positives.

        Args:
            attack_type: Type of attack that was executed.
            endpoint: The endpoint that was tested.
            payloads_and_responses: List of dicts with keys:
                "payload", "status_code", "response_body", "response_time_ms"
            baseline_response: Normal response for comparison.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [
            f"# RESULT ANALYSIS: {attack_type.upper()}",
            "",
            "Analyze the following attack results and determine which "
            "payloads (if any) successfully exploited a vulnerability. "
            "Your job is to eliminate false positives and confirm real findings.",
            "",
            f"**Attack Type**: {attack_type}",
            f"**Endpoint**: {endpoint}",
        ]

        # Add baseline for comparison
        if baseline_response:
            truncated_baseline = self._truncate(baseline_response, 1000)
            prompt_parts.extend([
                "",
                "## Baseline Response (normal, non-malicious request):",
                "",
                f"```",
                truncated_baseline,
                f"```",
            ])

        # Add payload results
        prompt_parts.extend([
            "",
            "## Attack Results:",
            "",
        ])

        for i, result in enumerate(payloads_and_responses[:20], 1):
            payload = result.get("payload", "N/A")
            status = result.get("status_code", "N/A")
            body = self._truncate(result.get("response_body", ""), 500)
            time_ms = result.get("response_time_ms", "N/A")

            prompt_parts.extend([
                f"### Payload {i}:",
                f"- **Payload**: `{payload}`",
                f"- **Status Code**: {status}",
                f"- **Response Time**: {time_ms}ms",
                f"- **Response Body** (truncated):",
                f"```",
                body,
                f"```",
                "",
            ])

        prompt_parts.extend([
            "## Your Analysis:",
            "",
            "For each payload, determine:",
            "",
            "1. **Vulnerable?** (YES / NO / POSSIBLY)",
            "2. **Confidence** (HIGH / MEDIUM / LOW)",
            "3. **Evidence**: What specific response characteristics confirm "
            "or deny the vulnerability?",
            "4. **False Positive Check**: Could this response occur normally "
            "without a vulnerability? Compare to the baseline.",
            "5. **Severity**: If confirmed, what is the severity? "
            "(Critical/High/Medium/Low)",
            "",
            "Be conservative — only mark as VULNERABLE with HIGH confidence "
            "if the evidence is clear and unambiguous.",
        ])

        prompt = "\n".join(prompt_parts)
        logger.debug(f"Built result analysis prompt: {len(prompt)} chars")
        return prompt

    # ===================================================================
    # PUBLIC: Remediation Prompts
    # ===================================================================
    def build_remediation_prompt(
        self,
        vulnerability: Vulnerability,
        source_code: str | None = None,
        language: str = "unknown",
    ) -> str:
        """
        Build a prompt for generating remediation code.

        Given a confirmed vulnerability, asks the AI to write
        the actual code fix — not just a description, but real
        runnable code in the target's language and framework.

        Args:
            vulnerability: The confirmed vulnerability to fix.
            source_code: Original vulnerable code (if available).
            language: Programming language of the target.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [
            "# REMEDIATION: CODE FIX GENERATION",
            "",
            "Generate a complete code fix for the following vulnerability. "
            "The fix must be production-ready — no placeholders, no TODOs, "
            "no pseudo-code.",
            "",
            "## Vulnerability Details:",
            "",
            f"- **Type**: {vulnerability.vuln_type.value}",
            f"- **Severity**: {vulnerability.severity.value.upper()}",
            f"- **Title**: {vulnerability.title}",
            f"- **Endpoint**: {vulnerability.method} {vulnerability.endpoint}",
            f"- **Description**: {vulnerability.description}",
        ]

        # Add evidence
        if vulnerability.evidence:
            evidence = vulnerability.evidence[0]
            prompt_parts.extend([
                "",
                "## Attack Evidence:",
                "",
                f"- **Payload Used**: `{evidence.payload_used}`",
                f"- **Response**: {self._truncate(evidence.response_received, 500)}",
            ])

        # Add original source code
        if source_code:
            truncated = self._truncate(source_code, _MAX_CODE_CHARS)
            prompt_parts.extend([
                "",
                "## Vulnerable Code:",
                "",
                f"```{language}",
                truncated,
                f"```",
            ])

        prompt_parts.extend([
            "",
            "## Requirements:",
            "",
            f"1. Write the fix in **{language}**. Use the same framework "
            "and coding style as the original code.",
            "",
            "2. The fix must:",
            "   - Completely eliminate the vulnerability",
            "   - Not break existing functionality",
            "   - Follow security best practices",
            "   - Include comments explaining the security fix",
            "",
            "3. Provide:",
            "   - The complete fixed code (not just the changed lines)",
            "   - A brief explanation of what was changed and why",
            "   - Any additional security recommendations",
            "",
            "4. If applicable, mention any libraries or functions "
            "that should be used (e.g., parameterized queries for SQLi, "
            "DOMPurify for XSS, etc.).",
        ])

        prompt = "\n".join(prompt_parts)
        logger.debug(
            f"Built remediation prompt: {vulnerability.vuln_type.value} "
            f"at {vulnerability.endpoint}"
        )
        return prompt

    # ===================================================================
    # PUBLIC: Report Generation Prompts
    # ===================================================================
    def build_executive_summary_prompt(
        self,
        target: Target,
        vulnerabilities: list[Vulnerability],
        scan_duration_seconds: float,
    ) -> str:
        """
        Build a prompt for generating the executive summary of a report.

        The executive summary is written for non-technical stakeholders
        (CTO, CISO, management) — it should explain risk in business
        terms, not technical jargon.

        Args:
            target: Target that was scanned.
            vulnerabilities: All discovered vulnerabilities.
            scan_duration_seconds: Total scan time.

        Returns:
            Formatted prompt string.
        """
        # Count severities
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for vuln in vulnerabilities:
            severity_counts[vuln.severity.value] += 1

        prompt_parts = [
            "# EXECUTIVE SUMMARY GENERATION",
            "",
            "Write a professional executive summary for a penetration test report. "
            "The audience is C-level executives and security management — "
            "use business language, not technical jargon.",
            "",
            "## Scan Overview:",
            "",
            f"- **Target**: {target.url}",
            f"- **Endpoints Tested**: {target.endpoint_count}",
            f"- **Scan Duration**: {scan_duration_seconds / 60:.1f} minutes",
            f"- **Source Code Available**: {'Yes' if target.has_source else 'No'}",
            "",
            "## Findings Summary:",
            "",
            f"- **Critical**: {severity_counts['critical']}",
            f"- **High**: {severity_counts['high']}",
            f"- **Medium**: {severity_counts['medium']}",
            f"- **Low**: {severity_counts['low']}",
            f"- **Informational**: {severity_counts['info']}",
            f"- **Total**: {len(vulnerabilities)}",
        ]

        # Add top findings
        if vulnerabilities:
            prompt_parts.extend([
                "",
                "## Top Findings (most severe):",
                "",
            ])

            # Sort by severity, show top 5
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
            sorted_vulns = sorted(
                vulnerabilities,
                key=lambda v: severity_order.get(v.severity.value, 5),
            )

            for i, vuln in enumerate(sorted_vulns[:5], 1):
                prompt_parts.append(
                    f"{i}. **[{vuln.severity.value.upper()}]** "
                    f"{vuln.title} — {vuln.endpoint}"
                )

        prompt_parts.extend([
            "",
            "## Output Requirements:",
            "",
            "Write 3-5 paragraphs covering:",
            "",
            "1. **Overall Risk Assessment**: What is the overall security "
            "posture? Use terms like 'critical risk', 'significant exposure', etc.",
            "",
            "2. **Key Findings**: Summarize the most impactful vulnerabilities "
            "in business terms (data breach risk, compliance violations, etc.).",
            "",
            "3. **Business Impact**: What could happen if these vulnerabilities "
            "are exploited? (financial loss, reputation damage, regulatory fines)",
            "",
            "4. **Recommendations**: Top 3-5 prioritized actions to take immediately.",
            "",
            "Keep it concise, professional, and actionable. No code. "
            "No technical exploitation details.",
        ])

        prompt = "\n".join(prompt_parts)
        logger.debug(f"Built executive summary prompt: {len(prompt)} chars")
        return prompt

    # ===================================================================
    # PUBLIC: Chat History Management
    # ===================================================================
    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role — "user", "assistant", or "system".
            content: Message content.
        """
        self._conversation_history.append({
            "role": role,
            "content": content,
        })

        # Enforce max history limit
        if len(self._conversation_history) > self._max_history:
            if self._truncation_strategy == "oldest_first":
                # Keep the system message (first) and trim oldest user/assistant
                self._conversation_history = (
                    self._conversation_history[:1] +
                    self._conversation_history[-(self._max_history - 1):]
                )

        logger.debug(
            f"History updated: {len(self._conversation_history)} messages "
            f"(max: {self._max_history})"
        )

    def get_history(self) -> list[dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            List of message dicts with "role" and "content" keys.
        """
        return list(self._conversation_history)

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self._conversation_history.clear()
        logger.debug("Conversation history cleared")

    def get_history_for_chat(
        self,
        new_message: str,
    ) -> list[dict[str, str]]:
        """
        Get conversation history with a new user message appended.

        Useful for building a chat() call with history:
            messages = engine.get_history_for_chat("Analyze this endpoint...")
            response = await client.chat(role="reasoning", messages=messages)
            engine.add_to_history("assistant", response.content)

        Args:
            new_message: The new user message to append.

        Returns:
            Complete message list ready for LLMClient.chat().
        """
        messages = self.get_history()
        messages.append({
            "role": "user",
            "content": new_message,
        })
        return messages

    # ===================================================================
    # PUBLIC: Utility — Wrap for JSON Output
    # ===================================================================
    def wrap_for_json(self, prompt: str) -> str:
        """
        Append JSON output instructions to a prompt.

        Args:
            prompt: Original prompt string.

        Returns:
            Prompt with JSON formatting instructions appended.
        """
        return f"{prompt}\n\n{_SECTION_DIVIDER}\n{_JSON_INSTRUCTION}"

    def wrap_for_list(self, prompt: str) -> str:
        """
        Append list output instructions to a prompt.

        Args:
            prompt: Original prompt string.

        Returns:
            Prompt with list formatting instructions appended.
        """
        return f"{prompt}\n\n{_SECTION_DIVIDER}\n{_LIST_INSTRUCTION}"

    # ===================================================================
    # INTERNAL: Format Target Context Block
    # ===================================================================
    def _format_target_context(self, target: Target) -> str:
        """
        Format target information into a prompt context block.

        Includes URL, tech stack, endpoints, and any notes.

        Args:
            target: Target to format.

        Returns:
            Formatted string block.
        """
        parts = [
            "## Target Information:",
            "",
            f"- **URL**: {target.url}",
            f"- **Source Code**: {'Available' if target.has_source else 'Not available'}",
        ]

        # Tech stack
        ts = target.tech_stack
        if self._include_tech_context and ts:
            parts.append("")
            parts.append("### Technology Stack:")

            if ts.language:
                parts.append(f"- **Language**: {ts.language}")
            if ts.framework:
                parts.append(f"- **Framework**: {ts.framework}")
            if ts.database:
                parts.append(f"- **Database**: {ts.database}")
            if ts.web_server:
                parts.append(f"- **Web Server**: {ts.web_server}")
            if ts.auth_mechanism and ts.auth_mechanism.value != "unknown":
                parts.append(f"- **Authentication**: {ts.auth_mechanism.value}")
            if ts.template_engine:
                parts.append(f"- **Template Engine**: {ts.template_engine}")
            if ts.other:
                parts.append(f"- **Other**: {', '.join(ts.other)}")

        # Endpoints
        if target.endpoints:
            parts.append("")
            parts.append("### Discovered Endpoints:")
            parts.append("")

            endpoint_text = self._format_endpoints(target.endpoints)
            parts.append(endpoint_text)

        # Notes
        if target.notes:
            parts.append("")
            parts.append("### Notes:")
            for note in target.notes[:10]:
                parts.append(f"- {note}")

        return "\n".join(parts)

    # ===================================================================
    # INTERNAL: Format Endpoints
    # ===================================================================
    def _format_endpoints(self, endpoints: list[Endpoint]) -> str:
        """
        Format a list of endpoints into a readable text block.

        Truncates if the total exceeds _MAX_ENDPOINT_CHARS.

        Args:
            endpoints: List of endpoints to format.

        Returns:
            Formatted endpoint listing.
        """
        lines: list[str] = []
        total_chars = 0

        for ep in endpoints:
            # Build endpoint line
            line = f"- `{ep.method} {ep.path}`"

            if ep.auth_required:
                line += " 🔒"

            if ep.parameters:
                param_names = [p.name for p in ep.parameters]
                line += f" — params: [{', '.join(param_names)}]"

            if ep.description:
                line += f" — {ep.description}"

            total_chars += len(line)
            if total_chars > _MAX_ENDPOINT_CHARS:
                remaining = len(endpoints) - len(lines)
                lines.append(f"- ... and {remaining} more endpoints")
                break

            lines.append(line)

        return "\n".join(lines)

    # ===================================================================
    # INTERNAL: Format Findings Context
    # ===================================================================
    def _format_findings_context(
        self,
        findings: list[Vulnerability],
    ) -> str:
        """
        Format previously discovered findings into a context block.

        This enables the AI to:
        - Avoid re-testing already-confirmed vulnerabilities
        - Chain attacks using prior findings
        - Prioritize based on what's already been found

        Args:
            findings: List of discovered vulnerabilities.

        Returns:
            Formatted findings block.
        """
        if not findings:
            return "### Previous Findings:\nNo vulnerabilities discovered yet."

        parts = [
            f"### Previous Findings ({len(findings)} total):",
            "",
        ]

        total_chars = 0

        # Sort by severity (most severe first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_findings = sorted(
            findings,
            key=lambda v: severity_order.get(v.severity.value, 5),
        )

        for i, vuln in enumerate(sorted_findings, 1):
            line = (
                f"{i}. **[{vuln.severity.value.upper()}]** "
                f"{vuln.vuln_type.value} — "
                f"`{vuln.method} {vuln.endpoint}` — "
                f"{vuln.title}"
            )

            total_chars += len(line)
            if total_chars > _MAX_FINDINGS_CHARS:
                remaining = len(findings) - i + 1
                parts.append(f"... and {remaining} more findings")
                break

            parts.append(line)

        return "\n".join(parts)

    # ===================================================================
    # INTERNAL: Get Parameter Location
    # ===================================================================
    @staticmethod
    def _get_param_location(endpoint: Endpoint, param_name: str) -> str:
        """
        Find where a parameter lives in an endpoint definition.

        Args:
            endpoint: Endpoint containing the parameter.
            param_name: Parameter name to look up.

        Returns:
            Location string (e.g., "query", "body", "header").
        """
        for param in endpoint.parameters:
            if param.name == param_name:
                return param.location
        return "unknown"

    # ===================================================================
    # INTERNAL: Truncate Text
    # ===================================================================
    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """
        Truncate text to a maximum character count.

        Adds a truncation notice if text was cut.

        Args:
            text: Text to truncate.
            max_chars: Maximum character limit.

        Returns:
            Truncated text string.
        """
        if not text:
            return ""

        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]

        # Try to cut at a line break for cleaner output
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.8:
            truncated = truncated[:last_newline]

        remaining = len(text) - len(truncated)
        truncated += f"\n\n... [TRUNCATED — {remaining} characters omitted]"

        return truncated

    # ===================================================================
    # v2: Memory & Skill Prompt Builders
    # ===================================================================

    def build_skill_extraction_prompt(
        self,
        scan_log: str,
        skill_id: str,
        scan_id: str,
    ) -> str:
        """
        Build the skill extraction prompt (Prompt 4 from v2 design doc).

        Used post-scan to extract the most reusable technique
        discovered during the scan.

        Args:
            scan_log: Formatted scan log text.
            skill_id: The skill ID to assign.
            scan_id: The scan that generated this data.

        Returns:
            Complete prompt string.
        """
        from skills.writer import SkillWriter
        return SkillWriter.build_skill_extraction_prompt(
            scan_log=scan_log,
            skill_id=skill_id,
            scan_id=scan_id,
        )

    def build_memory_query_prompt(
        self,
        target_url: str,
        tech_stack: str,
    ) -> str:
        """
        Build the memory query generation prompt (Prompt 1 from v2 design doc).

        Used at scan start to generate semantic queries for
        retrieving relevant skills from long-term memory.

        Args:
            target_url: The target being scanned.
            tech_stack: Detected technology stack description.

        Returns:
            Complete prompt string.
        """
        from skills.writer import SkillWriter
        return SkillWriter.build_memory_query_prompt(
            target_url=target_url,
            tech_stack=tech_stack,
        )

    def build_attack_plan_with_skills_prompt(
        self,
        skill_context: str,
        episodic_context: str,
        target_url: str,
        tech_stack: str,
        endpoint_count: int,
        endpoints: str,
    ) -> str:
        """
        Build the enhanced attack planning prompt (Prompt 2 from v2 design doc).

        Includes prior skill knowledge and episodic memory to
        produce smarter, more targeted attack plans.

        Args:
            skill_context: Formatted skill context from WorkingMemory.
            episodic_context: Formatted episodic context.
            target_url: Current target URL.
            tech_stack: Detected tech stack description.
            endpoint_count: Number of discovered endpoints.
            endpoints: Formatted endpoint list.

        Returns:
            Complete prompt string.
        """
        from skills.writer import SkillWriter
        return SkillWriter.build_attack_plan_with_skills_prompt(
            skill_context=skill_context,
            episodic_context=episodic_context,
            target_url=target_url,
            tech_stack=tech_stack,
            endpoint_count=endpoint_count,
            endpoints=endpoints,
        )