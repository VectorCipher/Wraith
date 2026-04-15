"""
WRAITH Input Validators

Validates all user inputs before processing.
Called early in the pipeline (CLI commands, scan config)
so bad inputs fail fast with clear error messages.

"""
import os
from pathlib import Path
from urllib.parse import urlparse

from utils.exception import ConfigurationError, TargetUnreachableError
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_target_url(url: str) -> str:
    """
    Validate and normalize a target URL.

    Handles common user mistakes:
    - Missing http:// scheme
    - Trailing slashes
    - Empty input
    - Completely invalid URLs

    Args:
        url: Raw URL string from user input

    Returns:
        Normalized URL string (e.g., "http://localhost:5000")

    Raises:
        TargetUnreachableError: If URL is empty or unparseable
    """
    if not url or not url.strip():
        raise TargetUnreachableError("Target URL cannot be empty")

    url = url.strip()

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
        logger.debug(f"Added http:// scheme: {url}")

    # Parse and validate
    parsed = urlparse(url)

    if not parsed.hostname:
        raise TargetUnreachableError(
            message=f"Invalid URL: {url}",
            details="URL must have a valid hostname (e.g., localhost, 192.168.1.1, example.com)",
        )

    if not parsed.scheme in ("http", "https"):
        raise TargetUnreachableError(
            message=f"Invalid URL scheme: {parsed.scheme}",
            details="Only http:// and https:// are supported",
        )

    # Normalize: remove trailing slash
    url = url.rstrip("/")

    logger.debug(f"Validated target URL: {url}")
    return url


def validate_source_path(path: str) -> str:
    """
    Validate source code directory path for white-box testing.

    Checks:
    - Path is not empty
    - Path exists on disk
    - Path is a directory (not a file)
    - Directory is not empty
    - Directory is readable

    Args:
        path: Raw path string from user input

    Returns:
        Absolute resolved path string

    Raises:
        ConfigurationError: If path is invalid, missing, or not a directory
    """
    if not path or not path.strip():
        raise ConfigurationError("Source code path cannot be empty")

    p = Path(path.strip()).resolve()

    if not p.exists():
        raise ConfigurationError(
            message=f"Source path does not exist: {path}",
            details=f"Resolved to: {p}",
        )

    if not p.is_dir():
        raise ConfigurationError(
            message=f"Source path is not a directory: {path}",
            details="Provide the root directory of the application source code",
        )

    # Check if directory has any files
    files = list(p.iterdir())
    if not files:
        raise ConfigurationError(
            message=f"Source directory is empty: {path}",
            details="Directory must contain application source code files",
        )

    # Check if readable
    if not os.access(p, os.R_OK):
        raise ConfigurationError(
            message=f"Source directory is not readable: {path}",
            details="Check file permissions",
        )

    logger.debug(f"Validated source path: {p}")
    return str(p)


def validate_scan_mode(mode: str) -> str:
    """
    Validate scan mode selection.

    Valid modes:
    - blackbox: Attack from outside, no source code
    - whitebox: Analyze source code + attack
    - full: Both modes combined

    Args:
        mode: Raw mode string from user input

    Returns:
        Normalized lowercase mode string

    Raises:
        ConfigurationError: If mode is not one of the valid options
    """
    valid_modes = {"blackbox", "whitebox", "full"}
    mode = mode.lower().strip()

    if mode not in valid_modes:
        raise ConfigurationError(
            message=f"Invalid scan mode: '{mode}'",
            details=f"Must be one of: {', '.join(sorted(valid_modes))}",
        )

    logger.debug(f"Validated scan mode: {mode}")
    return mode


def validate_output_format(fmt: str) -> str:
    """
    Validate report output format.

    Valid formats:
    - html: Styled HTML report
    - pdf: PDF document
    - json: Machine-readable JSON
    - markdown: Markdown text

    Args:
        fmt: Raw format string from user input

    Returns:
        Normalized lowercase format string

    Raises:
        ConfigurationError: If format is not supported
    """
    valid_formats = {"html", "pdf", "json", "markdown"}
    fmt = fmt.lower().strip()

    if fmt not in valid_formats:
        raise ConfigurationError(
            message=f"Invalid output format: '{fmt}'",
            details=f"Must be one of: {', '.join(sorted(valid_formats))}",
        )

    logger.debug(f"Validated output format: {fmt}")
    return fmt


def validate_rate_limit(rate: int) -> int:
    """
    Validate scan rate limit (requests per second).

    Guards against:
    - Zero or negative values
    - Absurdly high values that could DoS the target

    Args:
        rate: Requests per second

    Returns:
        Validated rate value

    Raises:
        ConfigurationError: If rate is out of safe range
    """
    if rate <= 0:
        raise ConfigurationError(
            message=f"Rate limit must be positive, got: {rate}",
        )

    if rate > 1000:
        raise ConfigurationError(
            message=f"Rate limit too high: {rate} req/sec",
            details="Maximum allowed is 1000 requests per second to prevent accidental DoS",
        )

    logger.debug(f"Validated rate limit: {rate} req/sec")
    return rate


def validate_timeout(seconds: int) -> int:
    """
    Validate request timeout value.

    Args:
        seconds: Timeout in seconds

    Returns:
        Validated timeout value

    Raises:
        ConfigurationError: If timeout is out of reasonable range
    """
    if seconds <= 0:
        raise ConfigurationError(
            message=f"Timeout must be positive, got: {seconds}",
        )

    if seconds > 300:
        raise ConfigurationError(
            message=f"Timeout too high: {seconds} seconds",
            details="Maximum allowed is 300 seconds (5 minutes)",
        )

    logger.debug(f"Validated timeout: {seconds}s")
    return seconds