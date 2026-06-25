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

    """

    content: str = ""
    model: str = ""
    role: str = ""         # Task role used — "reasoning" or "coding"
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

        Routes to Ollama or OpenRouter based on settings.llm_provider.
        """
        if settings.llm_provider == "openrouter":
            # Delegate to chat logic for OpenRouter (it only exposes chat completions API)
            messages = [{"role": "user", "content": prompt}]
            return await self.chat(
                role=role,
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
        model_config = self._get_model_config(role)
        model_name = settings.model if settings.llm_provider == "openrouter" and settings.model else model_config["model_name"]

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

        Routes to Ollama or OpenRouter based on settings.llm_provider.
        """
        model_config = self._get_model_config(role)
        model_name = settings.model if settings.llm_provider == "openrouter" and settings.model else model_config["model_name"]

        options = self._build_options(model_config, temperature, max_tokens)
        effective_system = system_prompt or model_config.get("system_prompt", "")
        full_messages = self._build_chat_messages(messages, effective_system)

        logger.debug(
            f"[chat] Role: {role} | Model: {model_name} | "
            f"Provider: {settings.llm_provider} | "
            f"Messages: {len(full_messages)} | "
            f"Last msg: {len(messages[-1]['content'])} chars"
        )

        start_time = time.monotonic()

        try:
            if settings.llm_provider == "openrouter":
                raw_response = await self._openrouter_chat(
                    model=model_name,
                    messages=full_messages,
                    options=options,
                )
            else:
                raw_response = await self._client.chat(
                    model=model_name,
                    messages=full_messages,
                    options=options,
                )
        except Exception as e:
            self._handle_error(e, model_name)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        
        if settings.llm_provider == "openrouter":
            result = self._parse_openrouter_response(raw_response, model_name, role, elapsed_ms)
        else:
            result = self._parse_chat_response(raw_response, role, elapsed_ms)

        self._track_usage(result)
        logger.info(f"[chat] {result.summary}")
        return result

    # ===================================================================
    # INTERNAL: OpenRouter Integration
    # ===================================================================
    async def _openrouter_chat(self, model: str, messages: list[dict], options: dict) -> dict:
        """Make a request to OpenRouter API using httpx."""
        import httpx
        
        if not settings.openrouter_api_key:
            raise LLMConnectionError("OpenRouter API key is not configured in settings or CLI prompt.")
            
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/VectorCipher/Wraith",
            "X-Title": "WRAITH Autonomous Pentester",
        }
        
        # OpenRouter accepts OpenAI-compatible payloads
        payload = {
            "model": model,
            "messages": messages,
            "temperature": options.get("temperature", 0.3),
        }
        
        # Add optional params if set
        if "num_predict" in options:
            payload["max_tokens"] = options["num_predict"]
        if "top_p" in options:
            payload["top_p"] = options["top_p"]
            
        async with httpx.AsyncClient(timeout=float(self._timeout)) as client:
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                # Get the detailed error from OpenRouter if available
                error_body = getattr(e, "response", None)
                if error_body:
                    try:
                        error_data = error_body.json()
                        error_msg = error_data.get("error", {}).get("message", str(e))
                        raise LLMConnectionError(f"OpenRouter API error: {error_msg}")
                    except Exception:
                        pass
                raise LLMConnectionError(f"HTTP error communicating with OpenRouter: {e}")
                
    def _parse_openrouter_response(
        self,
        raw_response: dict,
        model_name: str,
        role: str,
        elapsed_ms: float,
    ) -> LLMResponse:
        """Parse OpenAI-compatible response from OpenRouter."""
        try:
            message = raw_response["choices"][0]["message"]
            raw_text = message.get("content", "")
            
            thinking, content = self._extract_thinking(raw_text)
            
            usage = raw_response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            tokens_per_second = 0.0
            if completion_tokens > 0 and elapsed_ms > 0:
                tokens_per_second = (completion_tokens / elapsed_ms) * 1000
                
            return LLMResponse(
                content=content,
                model=model_name,
                role=role,
                thinking=thinking,
                total_duration_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_per_second=tokens_per_second,
            )
        except (KeyError, IndexError) as e:
            raise WraithError(f"Failed to parse OpenRouter response: {e}")

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

        """
        model_config = self._get_model_config(role)
        model_name = settings.model if settings.llm_provider == "openrouter" and settings.model else model_config["model_name"]

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

        """
        model_config = self._get_model_config(role)
        model_name = settings.model if settings.llm_provider == "openrouter" and settings.model else model_config["model_name"]

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
        Check if the LLM server is reachable.

        Returns:
            True if server responds, False otherwise.
        """
        if settings.llm_provider == "openrouter":
            if not settings.openrouter_api_key:
                logger.warning("OpenRouter API key is missing")
                return False
            return True
            
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
        If using OpenRouter, returns the configured model name to bypass checks.

        Returns:
            List of model name strings.

        Raises:
            LLMConnectionError: Cannot reach Ollama server.
        """
        if settings.llm_provider == "openrouter":
            # For OpenRouter, assume the configured model is available
            # since there are too many to fetch and filter easily.
            return [settings.model]
            
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

        """
        if settings.llm_provider == "openrouter":
            # Mock response for OpenRouter since we don't need local details
            return {"model": model_name, "details": {"family": "cloud", "format": "api"}}
            
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
                    f"Try a smaller model. Update MODEL= in .env\n"
                    f"Smaller alternatives: qwen3.5:4b, qwen2.5:7b, llama3.2:3b\n"
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