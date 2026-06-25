"""
WRAITH Payload Logger — Record Every Payload Attempt

Logs every single payload sent during a scan into the SQLite
`payload_results` table. This data drives the mutation engine's
learning — over time, WRAITH knows which mutations succeed against
which WAF products.

Design decisions:
    - Every payload is logged, success or failure
    - WAF detection is based on response patterns (403, specific headers)
    - Response body is hashed (not stored) to save space but enable dedup
    - Failure reasons are categorized for statistical analysis
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from utils.logger import get_logger

logger = get_logger("payload_engine.logger")

# Common WAF response patterns
_WAF_SIGNATURES = {
    "cloudflare":   ["cf-ray", "cloudflare", "__cfduid"],
    "modsecurity":  ["mod_security", "modsecurity", "NOYB"],
    "aws_waf":      ["awselb", "x-amzn-requestid"],
    "akamai":       ["akamai", "x-akamai"],
    "imperva":      ["incapsula", "visid_incap", "x-iinfo"],
    "sucuri":       ["x-sucuri", "sucuri"],
    "f5_bigip":     ["bigip", "f5", "x-wa-info"],
    "barracuda":    ["barra_counter_session", "barracuda"],
}


class PayloadResult:
    """A single payload execution result for logging."""

    def __init__(
        self,
        payload_raw: str,
        target_url: str,
        scan_id: str,
        attack_class: str = "",
        response_code: int = 0,
        response_body: str = "",
        response_headers: dict[str, str] | None = None,
        failure_reason: str = "",
        mutation_applied: str = "",
        mutation_succeeded: bool = False,
    ) -> None:
        self.payload_raw = payload_raw
        self.target_url = target_url
        self.scan_id = scan_id
        self.attack_class = attack_class
        self.response_code = response_code
        self.response_body_hash = self._hash_body(response_body)
        self.response_headers = response_headers or {}
        self.failure_reason = failure_reason
        self.mutation_applied = mutation_applied
        self.mutation_succeeded = mutation_succeeded

        # Auto-detect WAF
        self.waf_detected = False
        self.waf_signature = ""
        self._detect_waf(response_code, response_headers or {}, response_body)

    @staticmethod
    def _hash_body(body: str) -> str:
        """Hash response body for dedup without storing full content."""
        if not body:
            return ""
        return hashlib.md5(body.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _detect_waf(
        self,
        status_code: int,
        headers: dict[str, str],
        body: str,
    ) -> None:
        """Detect WAF presence from response signals."""
        # Status code signals
        if status_code in (403, 406, 429, 503):
            self.waf_detected = True

        # Header / body fingerprinting
        combined = " ".join(
            list(headers.keys()) + list(headers.values())
        ).lower()
        if body:
            combined += " " + body[:2000].lower()

        for waf_name, patterns in _WAF_SIGNATURES.items():
            for pattern in patterns:
                if pattern.lower() in combined:
                    self.waf_detected = True
                    self.waf_signature = waf_name
                    return


class PayloadLogger:
    """
    Records every payload attempt into SQLite for analysis.

    Usage:
        pl = PayloadLogger(db_manager=db)

        # Log a result
        pl.log_result(PayloadResult(
            payload_raw="' OR 1=1 --",
            target_url="http://target.com/login",
            scan_id="wraith-abc123",
            attack_class="sqli",
            response_code=403,
            response_body="Access Denied",
        ))

        # Query failure patterns
        patterns = pl.get_failure_patterns(waf_signature="modsecurity")
        success_rate = pl.get_success_rate(mutation_type="url_encode")
    """

    def __init__(self, db_manager: Any) -> None:
        self._db = db_manager
        logger.info("PayloadLogger initialized")

    # ===================================================================
    # Write
    # ===================================================================

    def log_result(self, result: PayloadResult) -> None:
        """Log a payload result to the database."""
        try:
            with self._db._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO payload_results
                    (scan_id, target_url, payload_raw, attack_class,
                     response_code, response_body_hash, waf_detected,
                     waf_signature, failure_reason, mutation_applied,
                     mutation_succeeded)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.scan_id,
                        result.target_url,
                        result.payload_raw,
                        result.attack_class,
                        result.response_code,
                        result.response_body_hash,
                        result.waf_detected,
                        result.waf_signature,
                        result.failure_reason,
                        result.mutation_applied,
                        result.mutation_succeeded,
                    ),
                )
            logger.debug(
                f"Payload logged: {result.attack_class} → "
                f"{result.response_code} "
                f"(WAF: {result.waf_signature or 'none'})"
            )
        except Exception as e:
            logger.error(f"Failed to log payload result: {e}")

    def log_batch(self, results: list[PayloadResult]) -> int:
        """Log multiple payload results. Returns count of successful logs."""
        logged = 0
        for result in results:
            try:
                self.log_result(result)
                logged += 1
            except Exception:
                continue
        return logged

    # ===================================================================
    # Read / Analysis
    # ===================================================================

    def get_failure_patterns(
        self,
        waf_signature: str | None = None,
        attack_class: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get payload failure patterns for analysis.

        Returns payloads that were blocked, grouped by failure reason
        and WAF signature. Used by the mutation engine to avoid
        repeating strategies that don't work.

        Args:
            waf_signature: Filter by WAF product.
            attack_class: Filter by attack class.
            limit: Max results.

        Returns:
            List of failure pattern dicts.
        """
        conditions = ["waf_detected = 1"]
        params = []

        if waf_signature:
            conditions.append("waf_signature = ?")
            params.append(waf_signature)
        if attack_class:
            conditions.append("attack_class = ?")
            params.append(attack_class)

        params.append(limit)

        try:
            with self._db._get_conn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT payload_raw, attack_class, response_code,
                           waf_signature, failure_reason, mutation_applied
                    FROM payload_results
                    WHERE {' AND '.join(conditions)}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()

            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get failure patterns: {e}")
            return []

    def get_success_rate(self, mutation_type: str) -> float:
        """
        Get the historical success rate for a specific mutation type.

        Returns a float 0.0-1.0 representing how often this mutation
        type results in a successful bypass.
        """
        try:
            with self._db._get_conn() as conn:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN mutation_succeeded = 1 THEN 1 ELSE 0 END) as successes
                    FROM payload_results
                    WHERE mutation_applied = ?
                    """,
                    (mutation_type,),
                ).fetchone()

            total = row["total"] or 0
            successes = row["successes"] or 0
            return successes / total if total > 0 else 0.0
        except Exception:
            return 0.0

    def get_waf_stats(self) -> dict[str, int]:
        """Get a count of encounters per WAF product."""
        try:
            with self._db._get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT waf_signature, COUNT(*) as cnt
                    FROM payload_results
                    WHERE waf_detected = 1 AND waf_signature != ''
                    GROUP BY waf_signature
                    ORDER BY cnt DESC
                    """
                ).fetchall()
            return {row["waf_signature"]: row["cnt"] for row in rows}
        except Exception:
            return {}

    def get_mutation_stats(self) -> list[dict[str, Any]]:
        """Get success/failure stats for each mutation type."""
        try:
            with self._db._get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        mutation_applied,
                        COUNT(*) as total,
                        SUM(CASE WHEN mutation_succeeded = 1 THEN 1 ELSE 0 END) as successes
                    FROM payload_results
                    WHERE mutation_applied != ''
                    GROUP BY mutation_applied
                    ORDER BY successes DESC
                    """
                ).fetchall()

            return [
                {
                    "mutation": row["mutation_applied"],
                    "total": row["total"],
                    "successes": row["successes"],
                    "rate": (row["successes"] or 0) / (row["total"] or 1),
                }
                for row in rows
            ]
        except Exception:
            return []
