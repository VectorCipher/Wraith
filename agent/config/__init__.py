"""
WRAITH Configuration Package

Centralized configuration management for the entire application.
Loads settings from environment variables (.env file), LLM model
definitions from models.yaml, and attack module configuration
from attacks.yaml.

"""

from pathlib import Path

import yaml

from utils.logger import get_logger
from utils.exception import ConfigurationError

logger = get_logger("config")

CONFIG_DIR: Path = Path(__file__).resolve().parent
AGENT_DIR: Path = CONFIG_DIR.parent
PROJECT_DIR: Path = AGENT_DIR.parent


# ---------------------------------------------------------------------------
# YAML Loader Helper
# ---------------------------------------------------------------------------
def _load_yaml(file_path: Path) -> dict:
    """
    Safely load a YAML file and return its contents as a dict.

    Args:
        file_path: Absolute path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ConfigurationError: If file is missing, unreadable, or invalid YAML.
    """
    if not file_path.exists():
        raise ConfigurationError(
            message=f"Configuration file not found: {file_path.name}",
            details=f"Expected at: {file_path}",
        )

    if not file_path.is_file():
        raise ConfigurationError(
            message=f"Configuration path is not a file: {file_path.name}",
            details=f"Path: {file_path}",
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            message=f"Invalid YAML in {file_path.name}",
            details=str(e),
        )
    except OSError as e:
        raise ConfigurationError(
            message=f"Cannot read {file_path.name}",
            details=str(e),
        )

    if data is None:
        raise ConfigurationError(
            message=f"Configuration file is empty: {file_path.name}",
            details=f"Path: {file_path}",
        )

    if not isinstance(data, dict):
        raise ConfigurationError(
            message=f"Configuration file must contain a YAML mapping (dict): {file_path.name}",
            details=f"Got type: {type(data).__name__}",
        )

    logger.debug(f"Loaded config: {file_path.name} ({len(data)} top-level keys)")
    return data


# ---------------------------------------------------------------------------
# Load YAML configs at import time
# ---------------------------------------------------------------------------
_models_yaml: dict = {}
_attacks_yaml: dict = {}


def _load_all_yaml_configs() -> None:
    """Load models.yaml and attacks.yaml into module-level caches."""
    global _models_yaml, _attacks_yaml

    models_path = CONFIG_DIR / "models.yaml"
    attacks_path = CONFIG_DIR / "attacks.yaml"

    try:
        _models_yaml = _load_yaml(models_path)
        logger.debug("models.yaml loaded successfully")
    except ConfigurationError as e:
        logger.warning(f"Could not load models.yaml: {e.message}")
        _models_yaml = {}

    try:
        _attacks_yaml = _load_yaml(attacks_path)
        logger.debug("attacks.yaml loaded successfully")
    except ConfigurationError as e:
        logger.warning(f"Could not load attacks.yaml: {e.message}")
        _attacks_yaml = {}


# ---------------------------------------------------------------------------
# Public Accessor: Model Configuration
# ---------------------------------------------------------------------------
def get_model_config(role: str) -> dict:
    """
    Retrieve configuration for an LLM model by its role.

    Args:
        role: The model role — "reasoning" or "coding".

    Returns:
        Dictionary with model configuration (name, temperature, etc.)

    Raises:
        ConfigurationError: If role is not found in models.yaml.

    """
    if not _models_yaml:
        raise ConfigurationError(
            message="Model configuration not loaded",
            details="models.yaml is missing or empty. Run setup first.",
        )

    models = _models_yaml.get("models", {})

    if role not in models:
        available = ", ".join(models.keys()) if models else "none"
        raise ConfigurationError(
            message=f"Unknown model role: '{role}'",
            details=f"Available roles: {available}",
        )

    logger.debug(f"Retrieved model config for role: {role}")
    return models[role]


# ---------------------------------------------------------------------------
# Public Accessor: Attack Configuration
# ---------------------------------------------------------------------------
def get_attack_config(attack_name: str) -> dict:
    """
    Retrieve configuration for an attack module by name.

    Args:
        attack_name: Attack identifier — e.g. "sqli", "xss", "ssrf".

    Returns:
        Dictionary with attack configuration (enabled, severity, etc.)

    Raises:
        ConfigurationError: If attack is not found in attacks.yaml.

"""
    if not _attacks_yaml:
        raise ConfigurationError(
            message="Attack configuration not loaded",
            details="attacks.yaml is missing or empty. Run setup first.",
        )

    attacks = _attacks_yaml.get("attacks", {})

    if attack_name not in attacks:
        available = ", ".join(attacks.keys()) if attacks else "none"
        raise ConfigurationError(
            message=f"Unknown attack module: '{attack_name}'",
            details=f"Available attacks: {available}",
        )

    logger.debug(f"Retrieved attack config for: {attack_name}")
    return attacks[attack_name]


# ---------------------------------------------------------------------------
# Public Accessor: List all available attacks
# ---------------------------------------------------------------------------
def get_all_attack_names() -> list[str]:
    """
    Return a list of all attack module names defined in attacks.yaml.

    Returns:
        List of attack name strings.
    """
    attacks = _attacks_yaml.get("attacks", {})
    return list(attacks.keys())


# ---------------------------------------------------------------------------
# Public Accessor: List all enabled attacks
# ---------------------------------------------------------------------------
def get_enabled_attacks() -> list[str]:
    """
    Return only the attack modules that are enabled in attacks.yaml.

    Returns:
        List of enabled attack name strings.
    """
    attacks = _attacks_yaml.get("attacks", {})
    return [
        name
        for name, cfg in attacks.items()
        if cfg.get("enabled", True)
    ]


# ---------------------------------------------------------------------------
# Reload function (useful for tests or dynamic config updates)
# ---------------------------------------------------------------------------
def reload_configs() -> None:
    """Force-reload all YAML configuration files from disk."""
    logger.info("Reloading all YAML configuration files...")
    _load_all_yaml_configs()
    logger.info("Configuration reload complete.")


# ---------------------------------------------------------------------------
# Initialize on import — load YAML files when anyone does:
#   from config import ...
# ---------------------------------------------------------------------------
_load_all_yaml_configs()

from config.settings import settings

# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------
__all__ = [
    "settings",
    "get_model_config",
    "get_attack_config",
    "get_all_attack_names",
    "get_enabled_attacks",
    "reload_configs",
    "CONFIG_DIR",
    "AGENT_DIR",
    "PROJECT_DIR",
]