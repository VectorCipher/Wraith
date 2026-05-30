"""
WRAITH Model Manager

Manages LLM model lifecycle, health checking, and availability verification.
This module ensures that the configured model is pulled, healthy, and ready
before WRAITH starts a scan.

WRAITH uses a SINGLE-MODEL architecture: one model handles all tasks,
with role-specific system prompts providing specialised behaviour.
The model is configured via MODEL= in .env (or the default in settings.py).

"""

from enum import Enum

from pydantic import BaseModel

from config import settings, get_model_config
from utils.exception import (
    LLMConnectionError,
    ModelNotFoundError,
    ConfigurationError,
)
from utils.logger import get_logger

# Type hint only — avoid circular import at runtime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from llm.client import LLMClient

logger = get_logger("llm.model_manager")


# ===========================================================================
# Model Status Types
# ===========================================================================
class ModelStatus(str, Enum):
    """Possible states of a model in the local Ollama instance."""
    READY = "ready"             # Pulled and verified working
    AVAILABLE = "available"     # Pulled but not yet verified
    MISSING = "missing"         # Not pulled — needs ollama pull
    UNHEALTHY = "unhealthy"     # Pulled but failing to generate
    UNKNOWN = "unknown"         # Cannot determine status
    CHECKING = "checking"       # Currently being verified


# ===========================================================================
# Single Model Info
# ===========================================================================
class ModelInfo(BaseModel):
    """Status information for a single model."""
    name: str
    role: str
    status: ModelStatus = ModelStatus.UNKNOWN
    size_gb: float | None = None
    parameter_count: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    error: str | None = None

    @property
    def is_ready(self) -> bool:
        """Model is available and verified working."""
        return self.status in (ModelStatus.READY, ModelStatus.AVAILABLE)

    @property
    def is_missing(self) -> bool:
        """Model needs to be pulled."""
        return self.status == ModelStatus.MISSING

    @property
    def status_icon(self) -> str:
        """Icon for CLI display."""
        icons = {
            ModelStatus.READY: "✅",
            ModelStatus.AVAILABLE: "🟡",
            ModelStatus.MISSING: "❌",
            ModelStatus.UNHEALTHY: "⚠️",
            ModelStatus.UNKNOWN: "❓",
            ModelStatus.CHECKING: "⏳",
        }
        return icons.get(self.status, "❓")

    @property
    def status_display(self) -> str:
        """Human-readable status string for CLI."""
        display = f"{self.status_icon} {self.name} [{self.role}] — {self.status.value}"

        if self.size_gb is not None:
            display += f" ({self.size_gb:.1f} GB)"

        if self.parameter_count:
            display += f" | {self.parameter_count} params"

        if self.error:
            display += f" | Error: {self.error}"

        return display

    @property
    def pull_command(self) -> str:
        """The ollama pull command to install this model."""
        return f"ollama pull {self.name}"


# ===========================================================================
# Overall Verification Result
# ===========================================================================
class VerificationResult(BaseModel):
    """Result of verifying all required models."""
    ollama_connected: bool = False
    models: list[ModelInfo] = []
    available_models: list[str] = []

    @property
    def all_ready(self) -> bool:
        """All required models are available and working."""
        if not self.ollama_connected:
            return False
        return all(m.is_ready for m in self.models)

    @property
    def missing(self) -> list[ModelInfo]:
        """Models that need to be pulled."""
        return [m for m in self.models if m.is_missing]

    @property
    def unhealthy(self) -> list[ModelInfo]:
        """Models that are pulled but not working."""
        return [m for m in self.models if m.status == ModelStatus.UNHEALTHY]

    @property
    def ready(self) -> list[ModelInfo]:
        """Models that are ready to use."""
        return [m for m in self.models if m.is_ready]

    @property
    def missing_count(self) -> int:
        """Number of models that need to be pulled."""
        return len(self.missing)

    @property
    def summary(self) -> str:
        """One-line summary for logging."""
        if not self.ollama_connected:
            return "❌ Ollama not connected"

        total = len(self.models)
        ready = len(self.ready)
        missing = len(self.missing)
        unhealthy_count = len(self.unhealthy)

        parts = [f"{ready}/{total} models ready"]

        if missing:
            parts.append(f"{missing} missing")

        if unhealthy_count:
            parts.append(f"{unhealthy_count} unhealthy")

        return " | ".join(parts)

    @property
    def pull_commands(self) -> list[str]:
        """List of commands needed to install missing models."""
        return [m.pull_command for m in self.missing]


# ===========================================================================
# ModelManager — Model Lifecycle Manager
# ===========================================================================
class ModelManager:
    """
    Manages the lifecycle and health of LLM models used by WRAITH.

    This class coordinates with the LLMClient to verify that all
    models defined in models.yaml are available and functioning
    before WRAITH begins scanning.

    Attributes:
        _client: LLMClient instance for communicating with Ollama.
        _required_roles: Model roles that WRAITH needs to function.
    """

    # The task roles WRAITH uses internally.
    # Both map to the SAME model — they differ only in system prompt / params.
    REQUIRED_ROLES: list[str] = ["reasoning", "coding"]

    def __init__(self, client: "LLMClient") -> None:
        """
        Initialize the Model Manager.

        Args:
            client: An initialized LLMClient instance.
        """
        self._client = client
        logger.debug("Model Manager initialized")

    # ===================================================================
    # PUBLIC: Full Verification Pipeline
    # ===================================================================
    async def verify_models(self, health_check: bool = True) -> VerificationResult:
        """
        Verify all required models are available and healthy.

        This is the main entry point — called at WRAITH startup.
        It checks Ollama connectivity, model availability, and
        optionally runs a health check on each model.

        Args:
            health_check: If True, send a test prompt to each model
                         to verify it can actually generate responses.
                         Slower but more thorough.

        Returns:
            VerificationResult with status of all models.
        """
        result = VerificationResult()

        # Step 1: Check Ollama connectivity
        logger.info("Checking Ollama connection...")
        result.ollama_connected = await self._client.check_connection()

        if not result.ollama_connected:
            logger.error(
                "Cannot connect to Ollama. "
                "Ensure it's running: ollama serve"
            )
            # Mark all models as unknown since we can't check
            for role in self.REQUIRED_ROLES:
                model_config = self._get_role_config(role)
                result.models.append(ModelInfo(
                    name=model_config.get("model_name", "unknown"),
                    role=role,
                    status=ModelStatus.UNKNOWN,
                    error="Ollama not connected",
                ))
            return result

        logger.info("✅ Ollama connected")

        # Step 2: Get list of available models
        try:
            result.available_models = await self._client.list_models()
            logger.info(
                f"Found {len(result.available_models)} models in Ollama: "
                f"{result.available_models}"
            )
        except LLMConnectionError as e:
            logger.error(f"Failed to list models: {e.message}")
            result.ollama_connected = False
            return result

        # Step 3: Check each required model
        for role in self.REQUIRED_ROLES:
            model_info = await self._check_model(
                role=role,
                available_models=result.available_models,
                health_check=health_check,
            )
            result.models.append(model_info)

        # Step 4: Log summary
        logger.info(f"Model verification: {result.summary}")

        if result.missing:
            logger.warning(
                "Missing models! Install them with:\n  " +
                "\n  ".join(result.pull_commands)
            )

        if result.unhealthy:
            for m in result.unhealthy:
                logger.warning(
                    f"Model {m.name} is unhealthy: {m.error}"
                )

        return result

    # ===================================================================
    # PUBLIC: Check Single Model by Role
    # ===================================================================
    async def check_model_by_role(self, role: str) -> ModelInfo:
        """
        Check the status of a single model by its role.

        Args:
            role: Model role — "reasoning" or "coding".

        Returns:
            ModelInfo with current status.
        """
        available_models = []

        try:
            available_models = await self._client.list_models()
        except LLMConnectionError:
            pass

        return await self._check_model(
            role=role,
            available_models=available_models,
            health_check=True,
        )

    # ===================================================================
    # PUBLIC: Quick Readiness Check
    # ===================================================================
    async def is_ready(self) -> bool:
        """
        Quick check: are all required models available?

        Does NOT run health checks — just verifies models are pulled.
        Use this for fast checks during scan execution.

        Returns:
            True if all required models exist in Ollama.
        """
        try:
            available = await self._client.list_models()
        except LLMConnectionError:
            return False

        for role in self.REQUIRED_ROLES:
            config = self._get_role_config(role)
            model_name = config.get("model_name", "")

            if not self._model_in_list(model_name, available):
                logger.warning(f"Model not available for role '{role}': {model_name}")
                return False

        return True

    # ===================================================================
    # PUBLIC: Get Required Model Names
    # ===================================================================
    def get_required_models(self) -> list[dict[str, str]]:
        """
        Return a list of required models with their roles.

        Returns:
            List of dicts with "role" and "model_name" keys.
        """
        models = []

        for role in self.REQUIRED_ROLES:
            config = self._get_role_config(role)
            models.append({
                "role": role,
                "model_name": config.get("model_name", "unknown"),
            })

        return models

    # ===================================================================
    # PUBLIC: Pull a Missing Model
    # ===================================================================
    async def pull_model(
        self,
        model_name: str,
        progress_callback: Callable | None = None,
    ) -> bool:
        """
        Pull (download) a model from the Ollama registry.

        This downloads the model weights to the local machine.
        Can take several minutes depending on model size and
        internet speed.

        Args:
            model_name: Full model name (e.g. "qwen2.5:7b").
            progress_callback: Optional async callable that receives
                              progress updates as dicts with keys:
                              "status", "completed", "total".

        Returns:
            True if model was pulled successfully.
        """
        logger.info(f"Pulling model: {model_name} ...")

        try:
            stream = await self._client._client.pull(
                model=model_name,
                stream=True,
            )

            async for chunk in stream:
                if progress_callback:
                    status = self._safe_get(chunk, "status", "")
                    completed = self._safe_get(chunk, "completed", 0)
                    total = self._safe_get(chunk, "total", 0)

                    await progress_callback({
                        "status": status,
                        "completed": completed,
                        "total": total,
                        "model": model_name,
                    })

                # Log progress milestones
                status = self._safe_get(chunk, "status", "")
                if status and "pulling" not in status.lower():
                    logger.debug(f"Pull {model_name}: {status}")

            logger.info(f"✅ Model pulled successfully: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    # ===================================================================
    # PUBLIC: Pull All Missing Models
    # ===================================================================
    async def pull_missing_models(
        self,
        progress_callback: Callable | None = None,
    ) -> dict[str, bool]:
        """
        Pull all models that are missing from Ollama.

        Args:
            progress_callback: Optional progress callback (see pull_model).

        Returns:
            Dict mapping model names to pull success (True/False).
        """
        verification = await self.verify_models(health_check=False)
        results: dict[str, bool] = {}

        if not verification.missing:
            logger.info("No missing models — nothing to pull")
            return results

        for model_info in verification.missing:
            success = await self.pull_model(
                model_name=model_info.name,
                progress_callback=progress_callback,
            )
            results[model_info.name] = success

        return results

    # ===================================================================
    # PUBLIC: Get Status Display for CLI
    # ===================================================================
    async def get_status_display(self) -> list[dict[str, str]]:
        """
        Get model status formatted for CLI display.

        Returns a list of dicts suitable for rendering in a
        rich.table.Table.

        Returns:
            List of dicts with keys: "role", "model", "status", "icon".
        """
        verification = await self.verify_models(health_check=False)
        display_rows: list[dict[str, str]] = []

        # Ollama connection row
        display_rows.append({
            "role": "Ollama Server",
            "model": settings.ollama_host,
            "status": "Connected" if verification.ollama_connected else "Disconnected",
            "icon": "✅" if verification.ollama_connected else "❌",
        })

        # Each model row
        for model in verification.models:
            display_rows.append({
                "role": model.role.capitalize(),
                "model": model.name,
                "status": model.status.value.capitalize(),
                "icon": model.status_icon,
            })

        return display_rows

    # ===================================================================
    # INTERNAL: Check a Single Model
    # ===================================================================
    async def _check_model(
        self,
        role: str,
        available_models: list[str],
        health_check: bool = True,
    ) -> ModelInfo:
        """
        Check the status of a single model.

        Args:
            role: Model role from models.yaml.
            available_models: List of models currently in Ollama.
            health_check: Whether to send a test prompt.

        Returns:
            ModelInfo with the model's current status.
        """
        config = self._get_role_config(role)
        model_name = config.get("model_name", "unknown")

        info = ModelInfo(
            name=model_name,
            role=role,
            status=ModelStatus.CHECKING,
        )

        logger.debug(f"Checking model: {model_name} (role: {role})")

        # Check if model exists in Ollama
        if not self._model_in_list(model_name, available_models):
            info.status = ModelStatus.MISSING
            info.error = f"Not found. Run: ollama pull {model_name}"
            logger.warning(f"Model missing: {model_name}")
            return info

        # Try to get model details
        try:
            model_details = await self._client.get_model_info(model_name)
            info = self._extract_model_details(info, model_details)
        except (ModelNotFoundError, LLMConnectionError) as e:
            logger.warning(f"Cannot get details for {model_name}: {e.message}")
            # Model exists but can't get details — still might work

        # Mark as available (exists but not yet health-checked)
        info.status = ModelStatus.AVAILABLE

        # Optional: run health check
        if health_check:
            healthy = await self._health_check(model_name, role)
            if healthy:
                info.status = ModelStatus.READY
                logger.info(f"✅ Model ready: {model_name} ({role})")
            else:
                info.status = ModelStatus.UNHEALTHY
                info.error = "Failed health check — model cannot generate responses"
                logger.warning(f"⚠️ Model unhealthy: {model_name}")

        return info

    # ===================================================================
    # INTERNAL: Health Check (Test Generation)
    # ===================================================================
    async def _health_check(self, model_name: str, role: str) -> bool:
        """
        Send a minimal test prompt to verify the model can generate.

        Uses a tiny, fast prompt to minimize resource usage.

        Args:
            model_name: Model to test.
            role: Role for the model (uses role's config).

        Returns:
            True if the model responds successfully.
        """
        test_prompt = "Respond with exactly one word: OK"

        logger.debug(f"Health check for {model_name}...")

        try:
            response = await self._client.generate(
                role=role,
                prompt=test_prompt,
                temperature=0.0,    # Deterministic for testing
                max_tokens=10,      # Tiny response
            )

            if response.is_empty:
                logger.warning(f"Health check: {model_name} returned empty response")
                return False

            logger.debug(
                f"Health check passed: {model_name} — "
                f"Response: '{response.content[:50]}' "
                f"({response.total_duration_ms:.0f}ms)"
            )
            return True

        except Exception as e:
            logger.warning(f"Health check failed for {model_name}: {e}")
            return False

    # ===================================================================
    # INTERNAL: Extract Model Details from Ollama Show Response
    # ===================================================================
    def _extract_model_details(
        self,
        info: ModelInfo,
        details: dict,
    ) -> ModelInfo:
        """
        Extract useful model metadata from Ollama's show response.

        The response format varies between Ollama versions, so
        this method handles multiple formats gracefully.

        Args:
            info: Existing ModelInfo to update.
            details: Raw response from ollama show.

        Returns:
            Updated ModelInfo with extracted details.
        """
        try:
            # Try to extract parameter count
            model_info = details.get("model_info", {}) or {}
            details_section = details.get("details", {}) or {}

            # Parameter count (e.g. "7B", "14B")
            param_count = (
                details_section.get("parameter_size", "") or
                model_info.get("general.parameter_count", "")
            )
            if param_count:
                info.parameter_count = str(param_count)

            # Quantization level (e.g. "Q4_K_M", "Q5_0")
            quant = (
                details_section.get("quantization_level", "") or
                model_info.get("general.quantization_version", "")
            )
            if quant:
                info.quantization = str(quant)

            # Context length
            ctx_length = model_info.get(
                "llama.context_length",
                model_info.get("general.context_length", None),
            )
            if ctx_length:
                info.context_length = int(ctx_length)

            # Size — try to calculate from the response
            size = details.get("size", 0) or 0
            if size > 0:
                info.size_gb = round(size / (1024 ** 3), 1)

        except Exception as e:
            logger.debug(f"Could not extract all model details: {e}")

        return info

    # ===================================================================
    # INTERNAL: Check if Model Name Exists in Available List
    # ===================================================================
    @staticmethod
    def _model_in_list(model_name: str, available: list[str]) -> bool:
        """
        Check if a model name matches any model in the available list.

        Handles variations in naming:
            - Exact match: "qwen2.5:7b" in ["qwen2.5:7b"]
            - Base match: "qwen2.5:7b" matches "qwen2.5:7b-q4_0"
            - Tag variations: "model:latest" matches "model:latest"

        Args:
            model_name: Model name to search for.
            available: List of available model names from Ollama.

        Returns:
            True if the model is found.
        """
        if not model_name or not available:
            return False

        # Exact match
        if model_name in available:
            return True

        # Base name match (before the colon)
        base_name = model_name.split(":")[0]
        target_tag = model_name.split(":")[-1] if ":" in model_name else ""

        for avail in available:
            avail_base = avail.split(":")[0]
            avail_tag = avail.split(":")[-1] if ":" in avail else ""

            # Same base name
            if base_name == avail_base:
                # If no specific tag requested, any tag matches
                if not target_tag or target_tag == "latest":
                    return True

                # Specific tag match
                if target_tag == avail_tag:
                    return True

                # Tag starts with our target (handles quantization suffixes)
                if avail_tag.startswith(target_tag):
                    return True

        return False

    # ===================================================================
    # INTERNAL: Get Model Config for a Role
    # ===================================================================
    def _get_role_config(self, role: str) -> dict:
        """
        Get model configuration for a role from models.yaml.

        Falls back to a minimal config if the role isn't found,
        using the model name from settings (.env).

        Args:
            role: Model role — "reasoning" or "coding".

        Returns:
            Model configuration dictionary.
        """
        try:
            return get_model_config(role)
        except ConfigurationError:
            # Fallback: use the single model name from settings
            fallback_name = settings.model

            logger.warning(
                f"Role '{role}' not in models.yaml — "
                f"falling back to settings.model: {fallback_name}"
            )

            return {
                "model_name": fallback_name or "unknown",
                "temperature": 0.3,
                "system_prompt": "",
            }

    # ===================================================================
    # INTERNAL: Safe Attribute/Key Access
    # ===================================================================
    @staticmethod
    def _safe_get(obj: object, key: str, default: object = None) -> object:
        """
        Safely get a value from dict or object.

        Args:
            obj: Source object.
            key: Key or attribute name.
            default: Fallback value.

        Returns:
            The value, or default if not found.
        """
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)