"""
WRAITH Settings Module

Loads and validates ALL application settings from environment variables.
Uses pydantic-settings to automatically read from the .env file at the
project root, with type validation, default values, and clear error
messages when required values are missing or invalid.

This is the SINGLE SOURCE OF TRUTH for runtime configuration.
No module should ever call os.getenv() directly — use settings instead.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.logger import get_logger

logger = get_logger("config.settings")


class WraithSettings(BaseSettings):
    """
    Application-wide settings loaded from environment variables.

    Pydantic-settings automatically reads from .env file and
    maps UPPER_SNAKE_CASE env vars to lower_snake_case attributes.

    """

    # -------------------------------------------------------------------
    # Pydantic-settings configuration
    # -------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",              # Ignore unknown env vars silently
    )

    # -------------------------------------------------------------------
    # Ollama / LLM Settings
    # -------------------------------------------------------------------
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="URL of the Ollama inference server",
    )
    ollama_timeout: int = Field(
        default=120,
        description="Timeout in seconds for LLM requests",
        ge=10,
        le=600,
    )

    # -------------------------------------------------------------------
    # Model Names
    # These are quick-access defaults. The full model config (temperature,
    # system prompts, etc.) lives in models.yaml and is accessed via
    # get_model_config(). These env vars let you override the model NAME
    # without editing YAML.
    # -------------------------------------------------------------------
    reasoning_model: str = Field(
        default="nemotron-cascade-2:latest",
        description="Ollama model name for reasoning/strategy tasks",
    )
    coding_model: str = Field(
        default="qwen2.5-coder:14b",
        description="Ollama model name for code analysis/generation tasks",
    )

    # -------------------------------------------------------------------
    # Go Scanner (gRPC) Settings
    # -------------------------------------------------------------------
    scanner_grpc_host: str = Field(
        default="localhost",
        description="Hostname of the Go gRPC scanner service",
    )
    scanner_grpc_port: int = Field(
        default=9090,
        description="Port of the Go gRPC scanner service",
        ge=1,
        le=65535,
    )

    # -------------------------------------------------------------------
    # Scanning Behavior
    # -------------------------------------------------------------------
    max_concurrent_requests: int = Field(
        default=50,
        description="Maximum concurrent HTTP requests during scanning",
        ge=1,
        le=500,
    )
    request_timeout: int = Field(
        default=30,
        description="Timeout in seconds for individual HTTP requests to target",
        ge=1,
        le=300,
    )
    scan_rate_limit: int = Field(
        default=100,
        description="Maximum requests per second to the target",
        ge=1,
        le=1000,
    )

    # -------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------
    db_path: str = Field(
        default="./data/wraith.db",
        description="Path to the SQLite database file",
    )
    report_output_dir: str = Field(
        default="./reports",
        description="Directory where generated reports are saved",
    )

    # -------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    log_file: str | None = Field(
        default=None,
        description="Optional file path for logging output",
    )

    # -------------------------------------------------------------------
    # Application Metadata
    # -------------------------------------------------------------------
    wraith_version: str = Field(
        default="0.1.0",
        description="Current WRAITH version",
    )

    # ===================================================================
    # VALIDATORS
    # ===================================================================

    @field_validator("ollama_host")
    @classmethod
    def validate_ollama_host(cls, v: str) -> str:
        """Ensure Ollama host is a valid HTTP URL."""
        v = v.strip().rstrip("/")

        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"ollama_host must start with http:// or https://, got: '{v}'"
            )

        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a recognized Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper().strip()

        if v not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got: '{v}'"
            )

        return v

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """Ensure the parent directory for the database can exist."""
        v = v.strip()
        parent = Path(v).parent

        # We don't require the dir to exist yet — we'll create it.
        # But the path shouldn't be empty or obviously wrong.
        if not v:
            raise ValueError("db_path cannot be empty")

        return v

    @field_validator("report_output_dir")
    @classmethod
    def validate_report_output_dir(cls, v: str) -> str:
        """Basic validation on report output directory."""
        v = v.strip()

        if not v:
            raise ValueError("report_output_dir cannot be empty")

        return v

    # ===================================================================
    # COMPUTED PROPERTIES
    # ===================================================================

    @property
    def scanner_grpc_address(self) -> str:
        """
        Full gRPC address string for connecting to the Go scanner.

        Returns:
            Address in "host:port" format — e.g. "localhost:9090"
        """
        return f"{self.scanner_grpc_host}:{self.scanner_grpc_port}"

    @property
    def ollama_api_url(self) -> str:
        """
        Full URL for the Ollama API endpoint.

        Returns:
            URL string — e.g. "http://localhost:11434/api"
        """
        return f"{self.ollama_host}/api"

    @property
    def db_full_path(self) -> Path:
        """
        Resolved absolute path to the SQLite database file.

        Returns:
            Path object for the database file.
        """
        return Path(self.db_path).resolve()

    @property
    def report_full_path(self) -> Path:
        """
        Resolved absolute path to the report output directory.

        Returns:
            Path object for the report directory.
        """
        return Path(self.report_output_dir).resolve()

    # ===================================================================
    # UTILITY METHODS
    # ===================================================================

    def ensure_directories(self) -> None:
        """
        Create required directories if they don't exist.
        Called once at application startup.

        Creates:
            - Database parent directory (e.g. ./data/)
            - Report output directory (e.g. ./reports/)
        """
        db_dir = self.db_full_path.parent
        report_dir = self.report_full_path

        db_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured database directory: {db_dir}")

        report_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured report directory: {report_dir}")

    def display_summary(self) -> dict[str, str]:
        """
        Return a dictionary of key settings for display in the CLI.
        Useful for the startup banner / status command.

        Returns:
            Dict mapping setting names to their current values.
        """
        return {
            "Ollama Host": self.ollama_host,
            "Ollama Timeout": f"{self.ollama_timeout}s",
            "Reasoning Model": self.reasoning_model,
            "Coding Model": self.coding_model,
            "Scanner gRPC": self.scanner_grpc_address,
            "Rate Limit": f"{self.scan_rate_limit} req/s",
            "Concurrency": str(self.max_concurrent_requests),
            "Request Timeout": f"{self.request_timeout}s",
            "Database": self.db_path,
            "Reports Dir": self.report_output_dir,
            "Log Level": self.log_level,
            "Version": self.wraith_version,
        }


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
# This is created ONCE when settings.py is first imported.
# Every module that does `from config import settings` gets the SAME object.
# If .env is missing, defaults apply. If values are invalid, Pydantic raises
# a clear validation error at startup — fail fast.
# ---------------------------------------------------------------------------

try:
    settings = WraithSettings()
    logger.debug(
        f"Settings loaded — Ollama: {settings.ollama_host}, "
        f"Models: {settings.reasoning_model} / {settings.coding_model}"
    )
except Exception as e:
    # If settings fail to load, this is a fatal configuration error.
    # We log it and re-raise so the app doesn't start in a broken state.
    logger.error(f"Failed to load settings: {e}")
    raise