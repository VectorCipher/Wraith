"""
WRAITH Attack Result Data Models

Represents data flowing between the Python AI brain and the
Go scanner during attack execution.

The flow:
  Python AI decides what to attack
      ↓
  AttackRequest sent to Go via gRPC
      ↓
  Go executes hundreds of payloads concurrently
      ↓
  PayloadResult streamed back for each payload
      ↓
  AttackResult collected with all payload results
      ↓
  Python AI analyzes results to confirm vulnerabilities

Models:
  AttackStatus   → Did the attack succeed/fail/timeout/get blocked?
  AttackRequest  → Instructions sent FROM Python TO Go scanner
  PayloadResult  → Result of a SINGLE payload execution
  AttackResult   → Complete result of an entire attack run

These models are the BRIDGE between Python intelligence
and Go speed. Python says WHAT to attack, Go does it FAST,
and sends results back in this format.

Usage:
    request = AttackRequest(
        attack_id="atk-001",
        attack_type="sqli",
        target_url="http://localhost:5000/api/login",
        method="POST",
        payloads=["' OR 1=1--", "admin'--", "1; DROP TABLE users--"],
        parameter_name="username",
        injection_point="body",
    )
"""

from enum import Enum

from pydantic import BaseModel


class AttackStatus(str, Enum):
    """
    Outcome status of an attack execution.

    SUCCESS:  Attack executed normally (doesn't mean vulnerable —
              just means the requests were sent and responses received).
    FAILED:   Attack couldn't execute (code error, config problem).
    TIMEOUT:  Target didn't respond within the time limit.
    BLOCKED:  WAF or firewall blocked the attack payloads.
    ERROR:    Unexpected error during execution.

    Note: SUCCESS means "the attack RAN successfully", NOT that
    a vulnerability was found. Vulnerability determination happens
    AFTER the AI analyzes the results.
    """

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    ERROR = "error"


class AttackRequest(BaseModel):
    """
    Instructions sent from the Python AI agent to the Go scanner.

    The Python side (AI brain) figures out:
    - Which endpoint to attack
    - Which attack type to use
    - Which payloads to inject
    - Where to inject them

    Then packages it all into this request and sends it to Go
    via gRPC. Go receives this, fires all payloads concurrently
    using goroutines, and streams results back.

    Fields explained:
    - attack_id: Unique ID to track this specific attack run
    - attack_type: Category (sqli, xss, ssrf, etc.)
    - target_url: Full URL to attack
    - method: HTTP method to use
    - payloads: List of payloads to inject (LLM-generated)
    - injection_point: WHERE to inject (query param, body, header, etc.)
    - parameter_name: WHICH parameter to inject into
    - headers: Custom HTTP headers to include
    - body: Request body template (for POST/PUT requests)
    - cookies: Cookies to include (for authenticated testing)
    - timeout_seconds: Per-request timeout
    - follow_redirects: Whether to follow HTTP redirects
    - baseline_response: Normal response for comparison
                         (helps detect anomalies)
    """

    attack_id: str
    attack_type: str
    target_url: str
    method: str = "GET"
    payloads: list[str] = []
    injection_point: str = "query"
    parameter_name: str = ""
    headers: dict[str, str] = {}
    body: str | None = None
    cookies: dict[str, str] = {}
    timeout_seconds: int = 30
    follow_redirects: bool = False
    baseline_response: str | None = None


class PayloadResult(BaseModel):
    """
    Result of executing a SINGLE payload against the target.

    For each payload in the AttackRequest, Go sends back one
    PayloadResult. If the request had 100 payloads, you get
    100 PayloadResults streamed back.

    The AI analyzes these individually to detect anomalies:
    - Status code changed? (200 → 500 might mean error-based SQLi)
    - Response body different? (extra data might mean union SQLi)
    - Response time spiked? (5 second delay might mean blind SQLi)
    - Content length changed? (different size might mean XSS reflected)
    - Error message appeared? (stack trace might mean injection worked)

    By comparing each PayloadResult against the baseline_response,
    the AI can detect subtle signs of vulnerability.
    """

    payload: str
    status_code: int
    response_body: str
    response_headers: dict[str, str] = {}
    response_time_ms: float
    content_length: int
    error: str | None = None


class AttackResult(BaseModel):
    """
    Complete result of an entire attack execution.

    Bundles together:
    - The attack metadata (ID, type, status)
    - ALL individual payload results
    - Timing and count statistics

    This is what the Python AI receives back from Go after
    an attack completes. The AI then:
    1. Loops through payload_results
    2. Compares each against baseline
    3. Uses LLM to determine if any payload succeeded
    4. Creates a Vulnerability object if confirmed

    The journey of an attack:
      AttackRequest (Python → Go)
          ↓
      Go fires all payloads concurrently
          ↓
      PayloadResult streamed back one by one
          ↓
      AttackResult bundled when all done
          ↓
      AI analyzes → Vulnerability confirmed or not
    """

    attack_id: str
    attack_type: str
    status: AttackStatus
    payload_results: list[PayloadResult] = []
    total_requests: int = 0
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def successful_payloads(self) -> list[PayloadResult]:
        """Get payloads that got a response (no errors)."""
        return [r for r in self.payload_results if r.error is None]

    @property
    def failed_payloads(self) -> list[PayloadResult]:
        """Get payloads that errored out."""
        return [r for r in self.payload_results if r.error is not None]

    @property
    def avg_response_time_ms(self) -> float:
        """Average response time across all payloads."""
        successful = self.successful_payloads
        if not successful:
            return 0.0
        return sum(r.response_time_ms for r in successful) / len(successful)

    @property
    def status_code_distribution(self) -> dict[int, int]:
        """Count how many times each status code appeared."""
        dist: dict[int, int] = {}
        for r in self.payload_results:
            dist[r.status_code] = dist.get(r.status_code, 0) + 1
        return dist