"""
WRAITH LLM Package

Provides all communication with the local Ollama LLM inference server.
This package is the ONLY place in the codebase that talks to Ollama.
Every other module that needs AI capabilities goes through this package.
    )
"""

from llm.client import LLMClient, LLMResponse
from llm.model_manager import ModelManager

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ModelManager",
]