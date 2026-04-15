"""
WRAITH Report Data Models

Represents the structure of a penetration test report.
Reports are the FINAL DELIVERABLE of WRAITH — what the user
actually takes away after a scan completes.

A professional pentest report has a standard structure:
  1. Executive Summary — high-level overview for non-technical readers
  2. Methodology — what was tested and how
  3. Findings — each vulnerability with evidence and remediation
  4. Risk Summary — severity breakdown and overall risk rating
  5. Compliance — which standards/frameworks are violated
  6. Appendix — raw data, full request/response logs

Models:
  ReportSection    → A single section/chapter in the report
  ReportFinding    → A vulnerability formatted for the report
  ReportMetadata   → Report metadata (title, date, author, etc.)
  Report           → Complete report combining all sections

The report generator (reporters/) takes ScanResult and produces
a Report object, which is then rendered to HTML/PDF/JSON/Markdown
by the appropriate reporter.

Usage:
    report = Report(
        metadata=ReportMetadata(title="Penetration Test Report", ...),
        executive_summary="We tested http://localhost:5000 and found...",
        sections=[methodology_section, findings_section, ...],
    )
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ReportMetadata(BaseModel):
    """
    Metadata about the report itself.

    Appears on the report cover page and header/footer.
    Contains identifying information about the test engagement.

    Example:
        ReportMetadata(
            title="Penetration Test Report — Acme Web Application",
            scan_id="wraith-001",
            target_url="http://localhost:5000",
            scan_mode="full",
            generated_at=datetime.now(),
            wraith_version="1.0.0",
        )
    """

    title: str = "WRAITH Penetration Test Report"
    scan_id: str = ""
    target_url: str = ""
    scan_mode: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)
    scan_started_at: datetime | None = None
    scan_ended_at: datetime | None = None
    scan_duration_seconds: float = 0.0
    wraith_version: str = "0.1.0"
    total_endpoints_tested: int = 0
    total_requests_sent: int = 0


class ReportSection(BaseModel):
    """
    A single section/chapter in the report.

    Sections can be nested — a section can contain subsections.
    This allows flexible report structure:

      Section: "Findings"
      ├── Subsection: "Critical Findings"
      │   └── content: details of each critical vuln
      ├── Subsection: "High Findings"
      │   └── content: details of each high vuln
      └── Subsection: "Medium Findings"
          └── content: details of each medium vuln

    The content field holds the actual text/HTML/markdown.
    For findings sections, the content is AI-generated narrative
    describing the vulnerability, its impact, and remediation.

    Example:
        ReportSection(
            title="Methodology",
            content="WRAITH followed the PTES methodology...",
            subsections=[],
        )
    """

    title: str
    content: str = ""
    subsections: list["ReportSection"] = []
    order: int = 0


class ReportFinding(BaseModel):
    """
    A vulnerability formatted specifically for report display.

    This is a PRESENTATION layer on top of the Vulnerability model.
    While Vulnerability stores raw data, ReportFinding formats it
    for human reading in a report.

    Adds formatted fields that the Vulnerability model doesn't have:
    - risk_rating: Human-readable risk description
    - impact_description: AI-generated impact narrative
    - steps_to_reproduce: Numbered reproduction steps
    - affected_component: Which part of the app is affected
    - formatted_evidence: Evidence formatted as readable text

    Example:
        ReportFinding(
            title="SQL Injection in Authentication Endpoint",
            severity="critical",
            cvss_score=9.8,
            risk_rating="CRITICAL — Immediate action required",
            impact_description="An attacker can bypass authentication...",
            steps_to_reproduce=[
                "Navigate to /api/login",
                "Enter payload: admin' OR 1=1--",
                "Observe that a valid JWT token is returned",
            ],
            ...
        )
    """

    title: str
    severity: str
    vuln_type: str
    cvss_score: float | None = None
    risk_rating: str = ""
    endpoint: str = ""
    method: str = "GET"
    description: str = ""
    impact_description: str = ""
    affected_component: str = ""
    steps_to_reproduce: list[str] = []
    formatted_evidence: str = ""
    remediation_description: str = ""
    remediation_code: str | None = None
    compliance_mappings: list[str] = []
    references: list[str] = []


class Report(BaseModel):
    """
    Complete penetration test report.

    This is the TOP-LEVEL model that the report generator builds
    and the reporter templates render into HTML/PDF/JSON/Markdown.

    Structure mirrors a professional pentest report:
      metadata           → Cover page info
      executive_summary  → 1-2 paragraph overview for executives
      methodology        → How the test was conducted
      findings           → List of all vulnerabilities (formatted)
      sections           → Additional custom sections
      severity_counts    → Quick stats breakdown

    Lifecycle:
    1. Scan completes → ScanResult created
    2. Report generator takes ScanResult
    3. AI writes executive_summary and methodology narrative
    4. Each Vulnerability is converted to ReportFinding
    5. Compliance mapper adds framework mappings
    6. Report object assembled
    7. Template renderer converts to HTML/PDF/JSON/Markdown
    8. File saved to disk

    Example:
        report = Report(
            metadata=ReportMetadata(title="Pentest Report", ...),
            executive_summary="We conducted a comprehensive...",
            methodology="Following PTES methodology...",
            findings=[finding1, finding2, ...],
        )
    """

    metadata: ReportMetadata = ReportMetadata()
    executive_summary: str = ""
    methodology: str = ""
    findings: list[ReportFinding] = []
    sections: list[ReportSection] = []
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0

    @property
    def total_findings(self) -> int:
        """Total number of findings in the report."""
        return len(self.findings)

    @property
    def overall_risk(self) -> str:
        """
        Determine overall risk rating based on findings.

        Logic:
        - Any critical → "CRITICAL"
        - Any high → "HIGH"
        - Any medium → "MEDIUM"
        - Only low/info → "LOW"
        - No findings → "INFORMATIONAL"
        """
        if self.critical_count > 0:
            return "CRITICAL"
        if self.high_count > 0:
            return "HIGH"
        if self.medium_count > 0:
            return "MEDIUM"
        if self.low_count > 0:
            return "LOW"
        return "INFORMATIONAL"

    @property
    def risk_color(self) -> str:
        """Color for the overall risk rating (used by CLI and HTML reports)."""
        colors = {
            "CRITICAL": "red",
            "HIGH": "orange",
            "MEDIUM": "yellow",
            "LOW": "blue",
            "INFORMATIONAL": "green",
        }
        return colors.get(self.overall_risk, "gray")

    @property
    def severity_summary(self) -> str:
        """One-line severity breakdown for display."""
        parts = []
        if self.critical_count:
            parts.append(f"{self.critical_count} Critical")
        if self.high_count:
            parts.append(f"{self.high_count} High")
        if self.medium_count:
            parts.append(f"{self.medium_count} Medium")
        if self.low_count:
            parts.append(f"{self.low_count} Low")
        if self.info_count:
            parts.append(f"{self.info_count} Info")

        if not parts:
            return "No findings"

        return " | ".join(parts)

    @property
    def has_findings(self) -> bool:
        """Check if report has any findings."""
        return self.total_findings > 0