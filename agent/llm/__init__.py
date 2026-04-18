"""
WRAITH LLM Package

Provides all communication with the local Ollama LLM inference server.
This package is the ONLY place in the codebase that talks to Ollama.
Every other module that needs AI capabilities goes through this package.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                   LLM Package                       │
    │                                                     │
    │  client.py          — Low-level Ollama API calls    │
    │  model_manager.py   — Model health, availability    │
    │  prompt_engine.py   — Prompt construction & context │
    │  prompts/           — Role-specific prompt templates │
    └──────────────┬──────────────────────────────────────┘
                   │
                   ▼
            Ollama (localhost:11434)

Usage:
    from llm import LLMClient, ModelManager, PromptEngine

    # Initialize
    client = LLMClient()
    manager = ModelManager(client)
    engine = PromptEngine(client)

    # Check models are available
    await manager.verify_models()

    # Generate a response using the reasoning model
    response = await client.generate(
        role="reasoning",
        prompt="Analyze this target for SQL injection vectors...",
    )

    # Generate a response using the coding model
    code = await client.generate(
        role="coding",
        prompt="Generate SQLi payloads for a MySQL backend...",
    )
"""

from llm.client import LLMClient, LLMResponse
from llm.model_manager import ModelManager

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ModelManager",
]