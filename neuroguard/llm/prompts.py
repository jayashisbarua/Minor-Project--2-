"""Structured prompt templates for Trojan analysis."""

import json
import re

# Required fields in a valid LLM response
_REQUIRED_FIELDS = {"verdict", "confidence_0_100", "evidence", "explanation", "fix"}

# Valid verdict values
_VALID_VERDICTS = {"trojan_detected", "likely_clean", "inconclusive"}


def build_trojan_analysis_prompt(context_json: dict) -> str:
    """Build a structured prompt for LLM-based Trojan analysis.

    Parameters
    ----------
    context_json : dict
        Anonymized context produced by Phase 7's ``build_llm_context()``.
        Must contain only anonymous gate IDs (G0, G1, …) – never real
        node/net/instance names.

    Returns
    -------
    str
        The full prompt string ready to send to an LLM backend.
    """
    # Serialise the context block
    context_block = json.dumps(context_json, indent=2)

    prompt = f"""\
You are a hardware security expert analyzing a digital circuit for potential \
hardware Trojans. Your task is to examine the anonymized gate-level netlist \
information below and determine whether the circuit contains suspicious \
structures that may indicate the presence of a hardware Trojan.

=== ANONYMIZED CIRCUIT CONTEXT ===
{context_block}
=== END CONTEXT ===

Analyze the circuit above for Trojan indicators. Look for:
- Rare-event trigger logic (gates with very low toggle rates)
- Unusual fan-in / fan-out patterns
- Suspicious connectivity to primary outputs or scan chains
- Redundant logic that does not contribute to normal function
- Sequential trigger structures (counters, shift-register patterns)

Respond with ONLY valid JSON (no markdown fences, no extra text). Use this \
exact schema:

{{
  "verdict": "<one of: trojan_detected, likely_clean, inconclusive>",
  "confidence_0_100": <integer 0-100>,
  "evidence": ["<reason 1>", "<reason 2>", "..."],
  "explanation": "<brief paragraph>",
  "fix": "<recommended remediation or 'No action needed'>"
}}

Output ONLY valid JSON, no markdown fences."""

    # ── Safety assertion: no real node/net/instance names leaked ──────────
    # The word "Trojan" (capital T, in instruction text) is fine.
    # We forbid "troj" or "trig" appearing as *substrings of identifiers*
    # inside the context block only.
    _assert_no_name_leakage(prompt, context_json)

    return prompt


def _assert_no_name_leakage(prompt: str, context_json: dict) -> None:
    """Verify that no node/net names leaked into the prompt's context block.

    Guards against netlist identifiers like ``troj_trig_01`` or instance
    names sneaking in through the context dict.
    Standard role hints (like 'possible_trigger') and analysis metadata
    are legitimate.
    """
    context_str = json.dumps(context_json).lower()
    # Mask out allowed standard keywords
    cleaned = context_str.replace("possible_trigger", "").replace("hardware_trojan", "").replace("trigger", "").replace("trojan", "")
    for forbidden in ("troj", "trig"):
        if forbidden in cleaned:
            raise ValueError(
                f"Name-leakage detected: context contains '{forbidden}' identifier fragment. "
                f"Only anonymous gate IDs (G0, G1, …) are allowed."
            )


def parse_llm_response(response_text: str) -> dict:
    """Extract and validate JSON from a raw LLM response string.

    Handles common LLM quirks:
    - Markdown fences (````` ```json ... ``` `````)
    - Preamble / postamble text around the JSON object
    - Whitespace variations

    Parameters
    ----------
    response_text : str
        Raw text returned by the LLM backend.

    Returns
    -------
    dict
        Parsed response with all required fields, or an error dict.
    """
    text = response_text.strip()

    # 1. Strip markdown code fences if present
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        text = match.group(1).strip()

    # 2. Try to find a JSON object in the text
    # Look for the outermost { ... }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start : brace_end + 1]

    # 3. Attempt JSON parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "verdict": "inconclusive",
            "confidence_0_100": 0,
            "evidence": [],
            "explanation": f"Failed to parse LLM response as JSON: {exc}",
            "fix": "Retry with a clearer prompt or different model.",
            "_parse_error": True,
        }

    # 4. Validate required fields
    missing = _REQUIRED_FIELDS - set(parsed.keys())
    if missing:
        return {
            **{k: parsed.get(k) for k in _REQUIRED_FIELDS},
            "verdict": parsed.get("verdict", "inconclusive"),
            "confidence_0_100": parsed.get("confidence_0_100", 0),
            "evidence": parsed.get("evidence", []),
            "explanation": parsed.get("explanation", f"Missing fields: {missing}"),
            "fix": parsed.get("fix", ""),
            "_missing_fields": list(missing),
        }

    # 5. Validate verdict value
    if parsed.get("verdict") not in _VALID_VERDICTS:
        parsed["_warning"] = (
            f"Unknown verdict '{parsed.get('verdict')}'; "
            f"expected one of {_VALID_VERDICTS}"
        )

    return parsed
