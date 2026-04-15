"""
WRAITH Target Data Models

Represents the application being tested.
This is the foundational model — almost every other component
needs to know about the target.

Models (bottom-up):
  AuthType   → How the app handles authentication
  TechStack  → Detected technology stack
  Parameter  → A single request parameter
  Endpoint   → A single API route/endpoint
  Target     → Complete target representation

"""

from enum import Enum
from pydantic import BaseModel


class AuthType(str, Enum):
    """
    Authentication mechanisms a target application might use.

    Detected during reconnaissance phase by:
    - Analyzing source code (white-box)
    - Observing response headers and cookies (black-box)
    - Checking for JWT tokens, session cookies, etc.

    Used by the AI to decide which auth-related attacks to try.
    """

    NONE = "none"
    JWT = "jwt"
    SESSION = "session"
    BASIC = "basic"
    OAUTH = "oauth"
    API_KEY = "api_key"
    UNKNOWN = "unknown"


class TechStack(BaseModel):
    """
    Detected technology stack of the target application.

    Populated during the reconnaissance phase.
    The AI uses this to generate SMART payloads instead of
    blasting generic ones.
    """

    language: str | None = None
    framework: str | None = None
    database: str | None = None
    web_server: str | None = None
    auth_mechanism: AuthType = AuthType.UNKNOWN
    template_engine: str | None = None
    other: list[str] = []


class Parameter(BaseModel):
    """
    A single request parameter for an endpoint.
    Each parameter is a potential injection point for attacks.
    """

    name: str
    location: str       # "query", "body", "header", "path", "cookie"
    param_type: str = "string"
    required: bool = False
    example_value: str | None = None


class Endpoint(BaseModel):
    """
    A single API endpoint / route in the target application.
    Each endpoint is a potential attack target.
    """

    path: str
    method: str = "GET"
    parameters: list[Parameter] = []
    auth_required: bool = False
    description: str | None = None
    source_file: str | None = None
    source_line: int | None = None


class Target(BaseModel):
    """
    Complete representation of the target application.
    Starts mostly empty, gets populated as WRAITH discovers info.
    """

    url: str
    source_path: str | None = None
    tech_stack: TechStack = TechStack()
    endpoints: list[Endpoint] = []
    open_ports: list[int] = []
    notes: list[str] = []

    @property
    def has_source(self) -> bool:
        """Check if source code is available for white-box testing."""
        return self.source_path is not None

    @property
    def endpoint_count(self) -> int:
        """Get total number of discovered endpoints."""
        return len(self.endpoints)

    @property
    def attack_surface_summary(self) -> str:
        """Generate a brief summary of the attack surface."""
        parts = [f"URL: {self.url}"]

        if self.tech_stack.framework:
            parts.append(f"Framework: {self.tech_stack.framework}")

        if self.tech_stack.database:
            parts.append(f"Database: {self.tech_stack.database}")

        parts.append(f"Endpoints: {self.endpoint_count}")
        parts.append(f"Source code: {'Yes' if self.has_source else 'No'}")

        return " | ".join(parts)