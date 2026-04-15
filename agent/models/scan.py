"""
WRAITH Scan Data Models

Represents the scan lifecycle from start to finish:
  ScanMode   → What kind of test (blackbox/whitebox/full)
  ScanStatus → Current phase of the scan
  ScanConfig → User's configuration for the scan
  ScanState  → Live state tracking during scan execution
  ScanResult → Final output after scan completes

These three models capture the ENTIRE journey:
  Config = "What the user wants"
  State  = "What's happening right now"
  Result = "What we ended up with"

Usage:
    config = ScanConfig(target_url="http://localhost:5000", mode=ScanMode.FULL)
    state = ScanState(scan_id="wraith-001", status=ScanStatus.RECONNAISSANCE)
    result = ScanResult(scan_id="wraith-001", config=config, target=target, ...)
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from models.target import Target
from models.vulnerability import Vulnerability


class ScanMode(str, Enum):
    """
    Type of penetration test to perform.

    BLACKBOX:  No source code available.
               WRAITH attacks from the outside like a real attacker.
               Crawls, fuzzes, and exploits the live application.

    WHITEBOX:  Source code IS available.
               WRAITH reads code first to find vulnerable patterns,
               then attacks the live app to confirm them.

    FULL:      Both modes combined (recommended).
               Reads source code AND attacks from outside.
               Maximum coverage — finds things neither mode
               would catch alone.
    """

    BLACKBOX = "blackbox"
    WHITEBOX = "whitebox"
    FULL = "full"


class ScanStatus(str, Enum):
    """
    Current phase of the scan lifecycle.

    Flows in order (though some phases may be skipped):
      PENDING → INITIALIZING → RECONNAISSANCE → ANALYSIS
      → EXPLOITATION → POST_EXPLOITATION → REPORTING → COMPLETED

    Can also end with:
      FAILED  → Unrecoverable error occurred
      ABORTED → User cancelled the scan (Ctrl+C)

    The orchestrator transitions between these states.
    The CLI displays the current status to the user.
    """

    PENDING = "pending"
    INITIALIZING = "initializing"
    RECONNAISSANCE = "reconnaissance"
    ANALYSIS = "analysis"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class ScanConfig(BaseModel):
    """
    User's configuration for a scan.

    Created from CLI arguments when the user launches a scan.
    Immutable after creation — represents what the user ASKED for.

    Example CLI command and resulting config:
      $ wraith scan --target http://localhost:5000 --mode full --source ./src

      ScanConfig(
          target_url="http://localhost:5000",
          mode=ScanMode.FULL,
          source_path="./src",
      )
    """

    target_url: str
    mode: ScanMode = ScanMode.FULL
    source_path: str | None = None
    max_duration_minutes: int = 60
    rate_limit: int = 100
    verbose: bool = False
    output_format: str = "html"
    output_path: str = "./reports"
    enabled_attacks: list[str] | None = None


class ScanState(BaseModel):
    """
    Live state of a running scan.

    Updated continuously by the orchestrator as the scan
    progresses. The CLI reads this to show real-time progress.

    Mutable — changes throughout the scan lifecycle.
    """

    scan_id: str
    status: ScanStatus = ScanStatus.PENDING
    current_phase: str = ""
    current_task: str = ""
    progress_percent: float = 0.0
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None
    total_requests_sent: int = 0
    endpoints_tested: int = 0
    vulnerabilities_found: int = 0
    errors: list[str] = []

    @property
    def is_running(self) -> bool:
        """Check if scan is still active."""
        return self.status not in (
            ScanStatus.COMPLETED,
            ScanStatus.FAILED,
            ScanStatus.ABORTED,
        )

    @property
    def duration_seconds(self) -> float:
        """Calculate how long the scan has been running."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def status_display(self) -> str:
        """Human-readable status string for CLI display."""
        if self.current_task:
            return f"{self.status.value} → {self.current_task}"
        if self.current_phase:
            return f"{self.status.value} → {self.current_phase}"
        return self.status.value


class ScanResult(BaseModel):
    """
    Final result of a completed scan.

    Created when the scan finishes (successfully or not).
    Contains everything: config, target info, all findings,
    and metadata about the scan run.

    This is what gets:
    - Saved to the database
    - Passed to the report generator
    - Displayed as the final CLI summary
    """

    scan_id: str
    config: ScanConfig
    target: Target
    vulnerabilities: list[Vulnerability] = []
    state: ScanState
    total_duration_seconds: float = 0.0
    report_path: str | None = None

    @property
    def severity_counts(self) -> dict[str, int]:
        """Count vulnerabilities by severity level."""
        counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        for vuln in self.vulnerabilities:
            counts[vuln.severity.value] += 1
        return counts

    @property
    def total_findings(self) -> int:
        """Total number of confirmed vulnerabilities."""
        return len(self.vulnerabilities)

    @property
    def has_critical(self) -> bool:
        """Check if any critical findings exist."""
        return any(
            v.severity.value == "critical" for v in self.vulnerabilities
        )

    @property
    def summary(self) -> str:
        """One-line summary of scan results for CLI display."""
        counts = self.severity_counts
        parts = []
        if counts["critical"]:
            parts.append(f"{counts['critical']} CRITICAL")
        if counts["high"]:
            parts.append(f"{counts['high']} HIGH")
        if counts["medium"]:
            parts.append(f"{counts['medium']} MEDIUM")
        if counts["low"]:
            parts.append(f"{counts['low']} LOW")
        if counts["info"]:
            parts.append(f"{counts['info']} INFO")

        if not parts:
            return "No vulnerabilities found"

        return f"Found {self.total_findings} vulnerabilities: {', '.join(parts)}"