"""
WRAITH Data Models Package

Pydantic models that define the shape of ALL data flowing
through the application. Every component uses these models
to ensure data consistency and validation.
"""

from models.target import (
    AuthType,
    TechStack,
    Parameter,
    Endpoint,
    Target,
)

from models.vulnerability import (
    Severity,
    VulnerabilityType,
    Evidence,
    Remediation,
    ComplianceMapping,
    Vulnerability,
)

from models.scan import (
    ScanMode,
    ScanStatus,
    ScanConfig,
    ScanState,
    ScanResult,
)

from models.attack_result import (
    AttackStatus,
    AttackRequest,
    PayloadResult,
    AttackResult,
)

from models.task import (
    TaskType,
    TaskStatus,
    TaskNode,
)

from models.report import (
    ReportMetadata,
    ReportSection,
    ReportFinding,
    Report,
)