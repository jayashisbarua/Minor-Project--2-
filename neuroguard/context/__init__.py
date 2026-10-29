"""Context expansion package for NeuroGuard-X.

Extracts anonymized 2-hop subgraphs and role hints around suspicious gates
for safe consumption by downstream LLM reasoning modules without leaking
any circuit/node/net identifiers.
"""

from neuroguard.context.expand import extract_suspicious_regions, Region
from neuroguard.context.roles import assign_role_hints
from neuroguard.context.to_json import regions_to_json, build_llm_context

__all__ = [
    "extract_suspicious_regions",
    "Region",
    "assign_role_hints",
    "regions_to_json",
    "build_llm_context",
]

