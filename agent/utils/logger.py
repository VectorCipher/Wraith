"""
WRAITH Logging System

"""

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


WRAITH_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "debug": "dim white",
        "scan": "bold green",
        "attack": "bold magenta",
        "finding": "bold yellow",
        "llm": "bold blue",
        "success": "bold green",
        "dim": "dim white",
    }
)

# Global console instance — importable by any module that needs rich output
# Usage: from utils.logger import console
#        console.print("[scan]Starting scan...[/scan]")
console = Console(theme=WRAITH_THEME)

# Module-level flag — ensures setup_logging() only runs once
_logging_configured = False


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure logging for the entire application.
    Call this ONCE at startup (in main.py), before anything else.

    Args:
        level: Log level string — "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        log_file: Optional path to also write logs to a file
    """
    global _logging_configured

    if _logging_configured:
        return

    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # ---- Rich handler for terminal ----
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        log_time_format="[%H:%M:%S]",
    )
    rich_handler.setLevel(log_level)

    handlers: list[logging.Handler] = [rich_handler]

    # ---- Optional file handler ----
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # ---- Configure root logger ----
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
    )

    # ---- Suppress noisy third-party loggers ----
    # These libraries log too much at INFO level
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    All WRAITH loggers are prefixed with 'wraith.' so they're
    easy to identify and filter in log output.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        A configured logging.Logger instance

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)

        logger.info("Starting scan on target")
        logger.debug("Verbose debug info")
        logger.warning("Something looks off")
        logger.error("Something failed")
        logger.critical("Fatal error, aborting")
    """
    return logging.getLogger(f"wraith.{name}")