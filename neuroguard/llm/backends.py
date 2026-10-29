"""LLM backend abstraction for NeuroGuard-X."""

import json
import os
import time
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def query(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the raw text response."""
        ...


class StubBackend(LLMBackend):
    """Returns a fixed JSON verdict – used for unit tests and offline demos."""

    _FIXED_RESPONSE = {
        "verdict": "trojan_detected",
        "confidence_0_100": 75,
        "evidence": ["High fan-in AND gate pattern", "Low toggle rate"],
        "explanation": "Stub response for testing",
        "fix": "Remove suspicious gates G0-G2",
    }

    def query(self, prompt: str) -> str:  # noqa: ARG002
        return json.dumps(self._FIXED_RESPONSE)


class OllamaBackend(LLMBackend):
    """Queries a local Ollama instance via its REST API.

    Parameters
    ----------
    model : str
        Ollama model tag (default ``qwen3:14b``).
    host : str
        Base URL of the Ollama server (default ``http://localhost:11434``).
    """

    def __init__(self, model: str = "qwen3:14b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def query(self, prompt: str) -> str:
        import urllib.request
        import urllib.error

        url = f"{self.host}/api/generate"
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode()

        max_retries = 2
        last_err: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode())
                    return body.get("response", json.dumps(body))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
                last_err = exc
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        # Graceful fallback when Ollama is not running
        return json.dumps({
            "verdict": "inconclusive",
            "confidence_0_100": 0,
            "evidence": [],
            "explanation": f"Ollama backend unavailable: {last_err}",
            "fix": "Ensure Ollama is running and the model is pulled.",
        })


class APIBackend(LLMBackend):
    """Generic OpenAI-compatible API backend.

    Reads configuration from environment variables:
    - ``NEUROGUARD_LLM_API_KEY``
    - ``NEUROGUARD_LLM_API_URL``

    Raises
    ------
    ValueError
        If the required environment variables are not set.
    """

    def __init__(self):
        self.api_key = os.environ.get("NEUROGUARD_LLM_API_KEY")
        self.api_url = os.environ.get("NEUROGUARD_LLM_API_URL")
        if not self.api_key or not self.api_url:
            raise ValueError(
                "Both NEUROGUARD_LLM_API_KEY and NEUROGUARD_LLM_API_URL "
                "environment variables must be set to use APIBackend."
            )

    def query(self, prompt: str) -> str:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }).encode()

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
                # OpenAI-compatible response shape
                choices = body.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", json.dumps(body))
                return json.dumps(body)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            return json.dumps({
                "verdict": "inconclusive",
                "confidence_0_100": 0,
                "evidence": [],
                "explanation": f"API request failed: {exc}",
                "fix": "Check API key, URL, and network connectivity.",
            })


# ── Factory ──────────────────────────────────────────────────────────────────

_BACKENDS = {
    "stub": StubBackend,
    "ollama": OllamaBackend,
    "api": APIBackend,
}


def get_backend(name: str = "stub") -> LLMBackend:
    """Instantiate and return the requested LLM backend.

    Parameters
    ----------
    name : str
        One of ``"stub"``, ``"ollama"``, ``"api"``.

    Returns
    -------
    LLMBackend
    """
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"Unknown backend '{name}'. Choose from {list(_BACKENDS)}")
    return cls()
