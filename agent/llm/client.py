"""
WRAITH LLM Client

Core module for communicating with the Ollama inference server.
This is the ONLY module that makes HTTP calls to Ollama.
All other modules use this client to access LLM capabilities.

Supports:
    - Dual-model architecture (reasoning + coding roles)
    - Model-agnostic design (model names from config, never hardcoded)
    - Async non-blocking I/O via Ollama's AsyncClient
    - Streaming responses for real-time CLI output
    - Automatic system prompt injection from models.yaml
    - Token usage and performance tracking
    - Chain-of-thought extraction (<think> tag parsing)
    - Structured LLMResponse output

Usage:
    client = LLMClient()

    # Simple generation with a role
    response = await client.generate(
        role="reasoning",
        prompt="Analyze this endpoint for SQL injection vectors...",
    )
    print(response.content)

    # Chat with message history
    response = await client.chat(
        role="coding",
        messages=[
            {"role": "user", "content": "Find SQLi sinks in this code..."}
        ],
    )

    # Streaming for real-time CLI display
    async for chunk in client.generate_stream(
        role="reasoning", prompt="Plan an attack strategy..."
    ):
        print(chunk, end="", flush=True)
"""

import re
import time
from typing import AsyncGenerator

from ollama import AsyncClient
from pydantic import BaseModel

from config import settings, get_model_config
from utils.exception import (
    LLMConnectionError,
    ModelNotFoundError,
    WraithError,
)
from utils.logger import get_logger

logger = get_logger("llm.client")

# ---------------------------------------------------------------------------
# Regex pattern for extracting chain-of-thought <think> blocks.
# Models like Nemotron-Cascade and DeepSeek-R1 wrap their reasoning
# inside <think>...</think> tags before giving the final answer.
# ---------------------------------------------------------------------------
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ===========================================================================
# LLMResponse — Structured output from every LLM call
# ===========================================================================
class LLMResponse(BaseModel):
    """
    Structured response from an LLM call.

    Wraps raw Ollama output into a clean, typed object
    that the rest of the application can work with.

    Attributes:
        content: The main generated text (thinking tags removed).
        model: Name of the model that produced this response.
        role: Role used for the request ("reasoning" or "coding").
        thinking: Chain-of-thought content extracted from <think> tags.
        total_duration_ms: Total wall-clock time for the request.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated.
        tokens_per_second: Generation speed.
    """

    content: str = ""
    model: str = ""
    role: str = ""
    thinking: str | None = None
    total_duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def has_thinking(self) -> bool:
        """Check if the response contains chain-of-thought reasoning."""
        return self.thinking is not None and len(self.thinking.strip()) > 0

    @property
    def is_empty(self) -> bool:
        """Check if the response has no meaningful content."""
        return not self.content or not self.content.strip()

    @property
    def summary(self) -> str:
        """Short summary string for logging."""
        preview = self.content[:100].replace("\n", " ")
        if len(self.content) > 100:
            preview += "..."

        return (
            f"Model: {self.model} | "
            f"Tokens: {self.total_tokens} ({self.prompt_tokens}+{self.completion_tokens}) | "
            f"Duration: {self.total_duration_ms:.0f}ms | "
            f"Speed: {self.tokens_per_second:.1f} tok/s"
        )


# ===========================================================================
# LLMClient — The single gateway to Ollama
# ===========================================================================
class LLMClient:
    """
    Async client for communicating with the Ollama inference server.

    This class is the single gateway to ALL LLM operations in WRAITH.
    It reads model configuration from models.yaml via get_model_config(),
    ensuring model names and parameters are never hardcoded.

    Lifecycle:
        client = LLMClient()
        # ... use client ...
        await client.close()

    Attributes:
        _client: Ollama AsyncClient instance.
        _host: Ollama server URL (e.g. "http://localhost:11434").
        _timeout: Request timeout in seconds.
        _total_requests: Running count of all LLM requests made.
        _total_tokens_used: Running count of all tokens consumed.
    """

    def __init__(
        self,
        host: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            host: Ollama server URL. Defaults to settings.ollama_host.
            timeout: Request timeout in seconds. Defaults to settings.ollama_timeout.
        """
        self._host = host or settings.ollama_host
        self._timeout = timeout or settings.ollama_timeout

        self._client = AsyncClient(
            host=self._host,
            timeout=float(self._timeout),
        )

        # Usage tracking
        self._total_requests: int = 0
        self._total_tokens_used: int = 0

        logger.info(
            f"LLM client initialized — "
            f"Host: {self._host}, Timeout: {self._timeout}s"
        )

    # ===================================================================
    # PUBLIC API: Generate (single prompt → response)
    # ===================================================================
    async def generate(
        self,
        role: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        context: list[int] | None = None,
        raw: bool = False,
    ) -> LLMResponse:
        """
        Generate a response from a single prompt.

        Uses Ollama's /api/generate endpoint. The model is selected
        based on the role ("reasoning" or "coding") from models.yaml.

        Args:
            role: Model role — "reasoning", "coding", or a direct model name.
            prompt: The user prompt to send.
            system_prompt: Override the default system prompt from models.yaml.
                          If None, uses the system prompt from config.
            temperature: Override the config temperature for this call.
            max_tokens: Override max output tokens for this call.
            context: Previous conversation context (Ollama context array).
            raw: If True, skip Ollama's template formatting.

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            LLMConnectionError: Cannot reach Ollama server.
            ModelNotFoundError: Model is not pulled in Ollama.
            WraithError: Other LLM-related errors.
        """
        model_config = self._get_model_config(role)
        model_name = model_config["model_name"]

        options = self._build_options(model_config, temperature, max_tokens)
        effective_system = system_prompt or model_config.get("system_prompt", "")

        logger.debug(
            f"[generate] Role: {role} | Model: {model_name} | "
            f"Prompt: {len(prompt)} chars"
        )

        start_time = time.monotonic()

        try:
            raw_response = await self._client.generate(
                model=model_name,
                prompt=prompt,
                system=effective_system,
                options=options,
                context=context,
                raw=raw,
            )
        except Exception as e:
            self._handle_error(e, model_name)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result = self._parse_generate_response(raw_response, role, elapsed_ms)

        self._track_usage(result)
        logger.info(f"[generate] {result.summary}")
        return result

    # ===================================================================
    # PUBLIC API: Chat (message list → response)
    # ===================================================================
    async def chat(
        self,
        role: str,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate a response from a chat conversation.

        Uses Ollama's /api/chat endpoint. Supports multi-turn
        conversations with message history.

        Args:
            role: Model role — "reasoning", "coding", or a direct model name.
            messages: List of message dicts with "role" and "content" keys.
                     Example: [{"role": "user", "content": "Find vulns..."}]
            system_prompt: Override the default system prompt.
            temperature: Override the config temperature.
            max_tokens: Override max output tokens.

        Returns:
            LLMResponse with the generated content and metadata.

        Raises:
            LLMConnectionError: Cannot reach Ollama server.
            ModelNotFoundError: Model is not pulled in Ollama.
            WraithError: Other LLM-related errors.
        """
        model_config = self._get_model_config(role)
        model_name = model_config["model_name"]

        options = self._build_options(model_config, temperature, max_tokens)
        effective_system = system_prompt or model_config.get("system_prompt", "")
        full_messages = self._build_chat_messages(messages, effective_system)

        logger.debug(
            f"[chat] Role: {role} | Model: {model_name} | "
            f"Messages: {len(full_messages)} | "
            f"Last msg: {len(messages[-1]['content'])} chars"
        )

        start_time = time.monotonic()

        try:
            raw_response = await self._client.chat(
                model=model_name,
                messages=full_messages,
                options=options,
            )
        except Exception as e:
            self._handle_error(e, model_name)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result = self._parse_chat_response(raw_response, role, elapsed_ms)

        self._track_usage(result)
        logger.info(f"[chat] {result.summary}")
        return result

    # ===================================================================
    # PUBLIC API: Streaming Generate (token-by-token)
    # ===================================================================
    async def generate_stream(
        self,
        role: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response token-by-token from /api/generate.

        Used by the CLI to display LLM output in real-time as it's
        generated. Yields content strings as they arrive from Ollama.

        Args:
            role: Model role — "reasoning" or "coding".
            prompt: The user prompt to send.
            system_prompt: Override the default system prompt.
            temperature: Override the config temperature.
            max_tokens: Override max output tokens.

        Yields:
            String chunks of the generated response.

        Raises:
            LLMConnectionError: Cannot reach Ollama server.
            ModelNotFoundError: Model is not pulled in Ollama.
        """
        model_config = self._get_model_config(role)
        model_name = model_config["model_name"]

        options = self._build_options(model_config, temperature, max_tokens)
        effective_system = system_prompt or model_config.get("system_prompt", "")

        logger.debug(
            f"[stream] Role: {role} | Model: {model_name} | "
            f"Prompt: {len(prompt)} chars"
        )

        try:
            stream = await self._client.generate(
                model=model_name,
                prompt=prompt,
                system=effective_system,
                options=options,
                stream=True,
            )

            async for chunk in stream:
                token = self._safe_get(chunk, "response", "")
                if token:
                    yield token

        except Exception as e:
            self._handle_error(e, model_name)

    # ===================================================================
    # PUBLIC API: Streaming Chat (token-by-token)
    # ===================================================================
    async def chat_stream(
        self,
        role: str,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response token-by-token from /api/chat.

        Args:
            role: Model role — "reasoning" or "coding".
            messages: Chat message history.
            system_prompt: Override the default system prompt.
            temperature: Override temperature.
            max_tokens: Override max tokens.

        Yields:
            String chunks of the generated response.

        Raises:
            LLMConnectionError: Cannot reach Ollama server.
            ModelNotFoundError: Model is not pulled in Ollama.
        """
        model_config = self._get_model_config(role)
        model_name = model_config["model_name"]

        options = self._build_options(model_config, temperature, max_tokens)
        effective_system = system_prompt or model_config.get("system_prompt", "")
        full_messages = self._build_chat_messages(messages, effective_system)

        logger.debug(
            f"[chat_stream] Role: {role} | Model: {model_name} | "
            f"Messages: {len(full_messages)}"
        )

        try:
            stream = await self._client.chat(
                model=model_name,
                messages=full_messages,
                options=options,
                stream=True,
            )

            async for chunk in stream:
                message = self._safe_get(chunk, "message", {})
                if isinstance(message, dict):
                    token = message.get("content", "")
                else:
                    token = getattr(message, "content", "")
                if token:
                    yield token

        except Exception as e:
            self._handle_error(e, model_name)

    # ===================================================================
    # PUBLIC API: Connection & Model Health Checks
    # ===================================================================
    async def check_connection(self) -> bool:
        """
        Check if the Ollama server is reachable.

        Returns:
            True if server responds, False otherwise.
        """
        try:
            await self._client.list()
            logger.debug("Ollama connection check: OK")
            return True
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
            return False

    async def list_models(self) -> list[str]:
        """
        List all models available in the local Ollama instance.

        Returns:
            List of model name strings (e.g. ["llama3.1:8b", "qwen2.5-coder:14b"]).

        Raises:
            LLMConnectionError: Cannot reach Ollama server.
        """
        try:
            response = await self._client.list()

            # Handle both dict-style and object-style responses
            # (Ollama Python lib versions differ in return types)
            if isinstance(response, dict):
                models_data = response.get("models", [])
            else:
                models_data = getattr(response, "models", [])

            model_names: list[str] = []
            for m in models_data:
                name = self._safe_get(m, "name", "") or self._safe_get(m, "model", "")
                if name:
                    model_names.append(name)

            logger.debug(f"Available models ({len(model_names)}): {model_names}")
            return model_names

        except Exception as e:
            raise LLMConnectionError(
                message="Cannot list models from Ollama",
                details=f"Host: {self._host} | Error: {e}",
            )

    async def model_exists(self, model_name: str) -> bool:
        """
        Check if a specific model is available locally.

        Args:
            model_name: Full model name (e.g. "nemotron-cascade:latest").

        Returns:
            True if model is pulled and available.
        """
        try:
            models = await self.list_models()

            # Exact match first
            if model_name in models:
                return True

            # Partial match: "qwen2.5-coder:14b" should match
            # "qwen2.5-coder:14b" even if Ollama stores a full hash tag
            base_name = model_name.split(":")[0]
            return any(m.startswith(base_name) for m in models)

        except LLMConnectionError:
            return False

    async def get_model_info(self, model_name: str) -> dict:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Full model name.

        Returns:
            Dict with model details (parameters, template, size, etc.)

        Raises:
            ModelNotFoundError: Model is not available.
            LLMConnectionError: Cannot reach Ollama.
        """
        try:
            info = await self._client.show(model_name)

            # Normalize to dict
            if not isinstance(info, dict):
                info = dict(info) if hasattr(info, "__iter__") else {"raw": str(info)}

            return info

        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                raise ModelNotFoundError(
                    message=f"Model not found: {model_name}",
                    details=f"Pull it with: ollama pull {model_name}",
                )
            raise LLMConnectionError(
                message=f"Cannot get info for model: {model_name}",
                details=f"Host: {self._host} | Error: {e}",
            )

    # ===================================================================
    # PUBLIC API: Usage Statistics
    # ===================================================================
    @property
    def total_requests(self) -> int:
        """Total number of LLM requests made through this client."""
        return self._total_requests

    @property
    def total_tokens_used(self) -> int:
        """Total tokens consumed across all requests."""
        return self._total_tokens_used

    def get_usage_stats(self) -> dict[str, int]:
        """
        Return usage statistics for display.

        Returns:
            Dict with request count and token usage.
        """
        return {
            "total_requests": self._total_requests,
            "total_tokens_used": self._total_tokens_used,
        }

    # ===================================================================
    # PUBLIC API: Lifecycle
    # ===================================================================
    async def close(self) -> None:
        """
        Clean up the underlying HTTP client.

        Call this when the LLMClient is no longer needed.
        Safe to call multiple times.
        """
        try:
            if hasattr(self._client, "_client") and self._client._client:
                await self._client._client.aclose()
                logger.debug("LLM client HTTP connection closed")
        except Exception as e:
            logger.debug(f"LLM client close (non-critical): {e}")

        logger.info(
            f"LLM client closed — "
            f"Total requests: {self._total_requests}, "
            f"Total tokens: {self._total_tokens_used}"
        )

    # ===================================================================
    # INTERNAL: Config Resolution
    # ===================================================================
    def _get_model_config(self, role: str) -> dict:
        """
        Resolve a role name to its full model configuration.

        Tries to find the role in models.yaml first. If not found,
        treats the role string as a direct model name and wraps it
        in a minimal configuration dict.

        Args:
            role: "reasoning", "coding", or a direct model name.

        Returns:
            Model configuration dictionary.
        """
        try:
            config = get_model_config(role)
            return config
        except Exception:
            # Role not in models.yaml — treat as a direct model name.
            # Wrap in minimal config so the rest of the pipeline works.
            logger.warning(
                f"Role '{role}' not in models.yaml — "
                f"using as direct model name with defaults"
            )
            return {
                "model_name": role,
                "temperature": 0.3,
                "context_window": 8192,
                "max_output_tokens": 4096,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "system_prompt": "",
                "ollama_options": {},
            }

    # ===================================================================
    # INTERNAL: Build Ollama Options Dict
    # ===================================================================
    def _build_options(
        self,
        model_config: dict,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
    ) -> dict:
        """
        Build the options dict sent to Ollama's API.

        Priority (highest to lowest):
            1. Explicit overrides (function arguments)
            2. ollama_options from models.yaml
            3. Top-level model config values
            4. Hardcoded defaults

        Args:
            model_config: Model configuration from models.yaml.
            temperature_override: Override temperature for this call.
            max_tokens_override: Override max tokens for this call.

        Returns:
            Options dict ready for the Ollama API.
        """
        # Start with ollama_options from config (most specific)
        options = dict(model_config.get("ollama_options", {}))

        # Fill in from top-level config values if not already set
        options.setdefault("temperature", model_config.get("temperature", 0.3))
        options.setdefault("num_ctx", model_config.get("context_window", 8192))
        options.setdefault("num_predict", model_config.get("max_output_tokens", 4096))
        options.setdefault("top_p", model_config.get("top_p", 0.9))
        options.setdefault("repeat_penalty", model_config.get("repeat_penalty", 1.1))

        # Apply explicit overrides (highest priority)
        if temperature_override is not None:
            options["temperature"] = temperature_override

        if max_tokens_override is not None:
            options["num_predict"] = max_tokens_override

        return options

    # ===================================================================
    # INTERNAL: Build Chat Message List
    # ===================================================================
    def _build_chat_messages(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> list[dict[str, str]]:
        """
        Build the full message list for a chat request.

        Prepends the system prompt as a system message if provided,
        then appends all user/assistant messages.

        Args:
            messages: User-provided message list.
            system_prompt: System prompt to prepend.

        Returns:
            Complete message list ready for Ollama's /api/chat.
        """
        full_messages: list[dict[str, str]] = []

        # Add system prompt as the first message
        if system_prompt and system_prompt.strip():
            full_messages.append({
                "role": "system",
                "content": system_prompt.strip(),
            })

        # Append conversation messages
        for msg in messages:
            full_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        return full_messages

    # ===================================================================
    # INTERNAL: Parse /api/generate Response
    # ===================================================================
    def _parse_generate_response(
        self,
        raw_response: object,
        role: str,
        elapsed_ms: float,
    ) -> LLMResponse:
        """
        Parse the raw Ollama /api/generate response into an LLMResponse.

        Handles both dict-style and object-style responses from
        different versions of the Ollama Python library.

        Args:
            raw_response: Raw response from Ollama.
            role: The role used for the request.
            elapsed_ms: Wall-clock time for the request.

        Returns:
            Parsed LLMResponse object.
        """
        # Extract raw text
        raw_text = self._safe_get(raw_response, "response", "")

        # Extract thinking blocks and clean content
        thinking, content = self._extract_thinking(raw_text)

        # Extract token counts from Ollama's response
        prompt_tokens = self._safe_get(raw_response, "prompt_eval_count", 0)
        completion_tokens = self._safe_get(raw_response, "eval_count", 0)

        # Calculate speed
        # Ollama reports total_duration in nanoseconds
        ollama_duration_ns = self._safe_get(raw_response, "total_duration", 0)
        if ollama_duration_ns > 0:
            duration_ms = ollama_duration_ns / 1_000_000
        else:
            duration_ms = elapsed_ms

        tokens_per_second = 0.0
        if completion_tokens > 0 and duration_ms > 0:
            tokens_per_second = (completion_tokens / duration_ms) * 1000

        # Get model name from response (Ollama includes it)
        model_name = self._safe_get(raw_response, "model", "unknown")

        return LLMResponse(
            content=content,
            model=model_name,
            role=role,
            thinking=thinking,
            total_duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
        )

    # ===================================================================
    # INTERNAL: Parse /api/chat Response
    # ===================================================================
    def _parse_chat_response(
        self,
        raw_response: object,
        role: str,
        elapsed_ms: float,
    ) -> LLMResponse:
        """
        Parse the raw Ollama /api/chat response into an LLMResponse.

        The chat endpoint nests the content inside a "message" field.

        Args:
            raw_response: Raw response from Ollama.
            role: The role used for the request.
            elapsed_ms: Wall-clock time for the request.

        Returns:
            Parsed LLMResponse object.
        """
        # Chat responses nest content inside "message"
        message = self._safe_get(raw_response, "message", {})

        if isinstance(message, dict):
            raw_text = message.get("content", "")
        else:
            raw_text = getattr(message, "content", "")

        # Extract thinking blocks and clean content
        thinking, content = self._extract_thinking(raw_text)

        # Extract token counts
        prompt_tokens = self._safe_get(raw_response, "prompt_eval_count", 0)
        completion_tokens = self._safe_get(raw_response, "eval_count", 0)

        # Calculate speed
        ollama_duration_ns = self._safe_get(raw_response, "total_duration", 0)
        if ollama_duration_ns > 0:
            duration_ms = ollama_duration_ns / 1_000_000
        else:
            duration_ms = elapsed_ms

        tokens_per_second = 0.0
        if completion_tokens > 0 and duration_ms > 0:
            tokens_per_second = (completion_tokens / duration_ms) * 1000

        model_name = self._safe_get(raw_response, "model", "unknown")

        return LLMResponse(
            content=content,
            model=model_name,
            role=role,
            thinking=thinking,
            total_duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
        )

    # ===================================================================
    # INTERNAL: Extract Chain-of-Thought <think> Blocks
    # ===================================================================
    def _extract_thinking(self, raw_text: str) -> tuple[str | None, str]:
        """
        Extract chain-of-thought content from <think> tags.

        Models like Nemotron-Cascade and DeepSeek-R1 wrap their
        reasoning process inside <think>...</think> tags:

            <think>
            The user wants me to analyze SQL injection.
            The endpoint uses string concatenation...
            </think>

            Based on my analysis, this endpoint is vulnerable to
            SQL injection because...

        This method separates the thinking from the final answer.

        Args:
            raw_text: Raw text from the model response.

        Returns:
            Tuple of (thinking_content, clean_content).
            thinking_content is None if no <think> tags found.
        """
        if not raw_text:
            return None, ""

        # Find all <think> blocks
        think_matches = _THINK_PATTERN.findall(raw_text)

        if not think_matches:
            # No thinking blocks — return raw text as content
            return None, raw_text.strip()

        # Combine all thinking blocks (some models output multiple)
        thinking = "\n\n".join(match.strip() for match in think_matches)

        # Remove all <think>...</think> blocks from the content
        content = _THINK_PATTERN.sub("", raw_text).strip()

        # Clean up any leftover whitespace from tag removal
        content = re.sub(r"\n{3,}", "\n\n", content)

        logger.debug(
            f"Extracted thinking: {len(thinking)} chars, "
            f"Content: {len(content)} chars"
        )

        return thinking, content

    # ===================================================================
    # INTERNAL: Usage Tracking
    # ===================================================================
    def _track_usage(self, response: LLMResponse) -> None:
        """
        Update running usage statistics.

        Args:
            response: The LLMResponse from a completed request.
        """
        self._total_requests += 1
        self._total_tokens_used += response.total_tokens

    # ===================================================================
    # INTERNAL: Error Handler
    # ===================================================================
    def _handle_error(self, error: Exception, model_name: str) -> None:
        """
        Convert raw exceptions into WRAITH-specific exceptions.

        Inspects the error message to determine the root cause and
        raises the appropriate WraithError subclass with helpful
        guidance for the user.

        Args:
            error: The raw exception from Ollama.
            model_name: The model that was being used.

        Raises:
            ModelNotFoundError: Model is not pulled.
            LLMConnectionError: Server unreachable.
            WraithError: Other LLM errors.
        """
        error_str = str(error).lower()

        logger.error(f"LLM error with model '{model_name}': {error}")

        # Model not found / not pulled
        if any(phrase in error_str for phrase in [
            "not found",
            "model not found",
            "does not exist",
            "404",
            "pull",
        ]):
            raise ModelNotFoundError(
                message=f"Model '{model_name}' is not available in Ollama",
                details=(
                    f"Pull it with: ollama pull {model_name}\n"
                    f"Then verify with: ollama list"
                ),
            )

        # Connection refused / timeout
        if any(phrase in error_str for phrase in [
            "connection refused",
            "connect error",
            "connection error",
            "timeout",
            "timed out",
            "unreachable",
            "cannot connect",
        ]):
            raise LLMConnectionError(
                message="Cannot connect to Ollama server",
                details=(
                    f"Host: {self._host}\n"
                    f"Ensure Ollama is running: ollama serve\n"
                    f"Error: {error}"
                ),
            )

        # Out of memory
        if any(phrase in error_str for phrase in [
            "out of memory",
            "oom",
            "cuda error",
            "memory",
            "alloc",
        ]):
            raise WraithError(
                message=f"Model '{model_name}' requires more memory than available",
                details=(
                    f"Try a smaller model. Update REASONING_MODEL or CODING_MODEL in .env\n"
                    f"Smaller alternatives: qwen2.5:7b, llama3.2:3b\n"
                    f"Error: {error}"
                ),
            )

        # Generic fallback
        raise WraithError(
            message=f"LLM request failed with model '{model_name}'",
            details=f"Error: {error}",
        )

    # ===================================================================
    # INTERNAL: Safe Attribute/Key Access
    # ===================================================================
    @staticmethod
    def _safe_get(obj: object, key: str, default: object = None) -> object:
        """
        Safely get a value from an object that might be a dict OR a dataclass.

        The Ollama Python library sometimes returns dicts and sometimes
        returns Pydantic-like objects depending on the version. This
        method handles both transparently.

        Args:
            obj: The object to read from (dict, dataclass, or any).
            key: The attribute/key name.
            default: Fallback value if key is not found.

        Returns:
            The value, or default if not found.
        """
        # Try dict access first
        if isinstance(obj, dict):
            return obj.get(key, default)

        # Try attribute access
        return getattr(obj, key, default)