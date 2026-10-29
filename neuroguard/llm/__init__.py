"""LLM integration package for NeuroGuard-X."""

from neuroguard.llm.backends import (
    LLMBackend,
    StubBackend,
    OllamaBackend,
    APIBackend,
    get_backend,
)
from neuroguard.llm.prompts import (
    build_trojan_analysis_prompt,
    parse_llm_response,
)
from neuroguard.llm.consistency import (
    compute_agreement,
    ab_swap_test,
    temperature_sweep,
)

__all__ = [
    "LLMBackend",
    "StubBackend",
    "OllamaBackend",
    "APIBackend",
    "get_backend",
    "build_trojan_analysis_prompt",
    "parse_llm_response",
    "compute_agreement",
    "ab_swap_test",
    "temperature_sweep",
]
