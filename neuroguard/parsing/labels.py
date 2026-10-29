"""Label extraction and ground-truth generation for NeuroGuard-X.

NOTE: Per project rules, gate and signal names may ONLY be used inside this module
to generate ground-truth labels. Model features and LLM prompts must NEVER receive
original names.
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set, Any


def parse_log(log_path: str | Path) -> Dict[str, Any]:
    """Parse a TRIT log.txt file.

    Extracts:
        - num_trojans: int
        - trojan_type: str
        - effects: List[str] (payload nets altered by the Trojan)
        - triggers: List[str] (internal trigger condition signals)
        - body_instances: List[Tuple[str, str, str]] (cell_type, instance_name, port_string)
        - body_instance_names: Set[str]
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return {
            "num_trojans": 0,
            "trojan_type": "clean",
            "effects": [],
            "triggers": [],
            "body_instances": [],
            "body_instance_names": set(),
        }

    with open(log_path, "r", errors="ignore") as f:
        text = f.read()

    num_trojans = 1
    m_num = re.search(r"Number of Trojans:\s*(\d+)", text)
    if m_num:
        num_trojans = int(m_num.group(1))

    trojan_type = "unknown"
    m_type = re.search(r"Type:\s*(.+)", text)
    if m_type:
        trojan_type = m_type.group(1).strip()

    effects: List[str] = []
    eff_match = re.search(
        r"Effect:.*?signal\(s\)\.\.\.\s*\n(.*?)(?=\nActivation Condition:|\n\*{10,}|\Z)",
        text,
        flags=re.DOTALL,
    )
    if eff_match:
        for line in eff_match.group(1).split():
            eff = line.strip()
            if eff:
                effects.append(eff)

    triggers: List[str] = []
    trig_match = re.search(
        r"Activation Condition:.*?signals\.\.\.\s*\n(.*?)(?=\n\*{10,}|\nTROJAN BODY:|\Z)",
        text,
        flags=re.DOTALL,
    )
    if trig_match:
        for item in trig_match.group(1).split():
            item = item.strip()
            if item:
                triggers.append(item)

    body_instances: List[Tuple[str, str, str]] = []
    body_instance_names: Set[str] = set()

    body_parts = text.split("TROJAN BODY:")
    if len(body_parts) > 1:
        body_content = body_parts[1].split("****************************************************************")[0]
        # Match cell inst_name ( ports );
        matches = re.findall(
            r"^\s*([a-zA-Z0-9_]+)\s+(\\\S+|\S+)\s*\((.*?)\)\s*;",
            body_content,
            flags=re.MULTILINE | re.DOTALL,
        )
        for cell, inst, ports in matches:
            body_instances.append((cell, inst, ports.strip()))
            body_instance_names.add(inst)

    return {
        "num_trojans": num_trojans,
        "trojan_type": trojan_type,
        "effects": effects,
        "triggers": triggers,
        "body_instances": body_instances,
        "body_instance_names": body_instance_names,
    }


def is_trojan_name(name: str) -> bool:
    """Check if an instance name belongs to a Trojan insertion."""
    clean_name = name.lstrip("\\").lower()
    return clean_name.startswith(("troj", "trig"))


def build_gate_labels(
    netlist_instances: List[Tuple[str, str, Dict[str, List[str]]]],
    log_info: Optional[Dict[str, Any]] = None,
    graph: Optional[Any] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build per-gate is_trojan (binary) and role (4-class) labels.

    Roles:
        0: Clean
        1: Trigger (activation detection logic)
        2: Body (payload generation, counters, FSMs)
        3: Payload (gate that directly alters functional logic, e.g. XOR on effect net)

    Parameters
    ----------
    netlist_instances: List of (cell_type, instance_name, ports_dict)
        where ports_dict has 'outputs': [nets...], 'inputs': [nets...]
    log_info: Optional parsed log.txt dictionary
    graph: Optional networkx MultiDiGraph for topology-based role resolution

    Returns
    -------
    (is_trojan_dict, role_dict) mapping instance_name -> int
    """
    is_trojan: Dict[str, int] = {}
    roles: Dict[str, int] = {}

    log_body_names = log_info.get("body_instance_names", set()) if log_info else set()
    effects = set(log_info.get("effects", [])) if log_info else set()
    triggers = set(log_info.get("triggers", [])) if log_info else set()

    for cell_type, inst_name, ports in netlist_instances:
        troj = 1 if (is_trojan_name(inst_name) or inst_name in log_body_names) else 0
        is_trojan[inst_name] = troj

        if not troj:
            roles[inst_name] = 0
            continue

        # Trojan gate role assignment
        clean_inst = inst_name.lstrip("\\").lower()
        inst_outputs = set(ports.get("outputs", []))
        inst_inputs = set(ports.get("inputs", []))

        # Check if payload:
        # 1. Drives an effect signal
        # 2. Named trojan0_0 (typical TRIT payload XOR)
        # 3. Output feeds a non-trojan gate
        is_payload = bool(inst_outputs & effects) or clean_inst.startswith("trojan")

        if not is_payload and graph is not None and inst_name in graph:
            # Check if any successor in the graph is clean
            for succ in graph.successors(inst_name):
                succ_type = graph.nodes[succ].get("type", "")
                if succ_type == "GATE" and not is_trojan_name(succ) and succ not in log_body_names:
                    is_payload = True
                    break

        if is_payload:
            roles[inst_name] = 3
            continue

        # Check if trigger:
        # 1. Name starts with 'trig'
        # 2. Output is Trigger_en*
        # 3. Input directly connects to triggers
        is_trigger = (
            clean_inst.startswith("trig")
            or any("trigger_en" in out.lower() for out in inst_outputs)
            or bool(inst_inputs & triggers)
        )
        if is_trigger:
            roles[inst_name] = 1
        else:
            roles[inst_name] = 2  # Trojan body (FSM/counter/intermediate)

    # Cross-check verification if log_info available
    if log_info and log_body_names:
        trojan_from_names = sum(is_trojan.values())
        trojan_from_log = len(log_body_names)
        if trojan_from_names != trojan_from_log:
            # We don't silently fix, we report/record it
            print(
                f"[WARNING] Label count mismatch: {trojan_from_names} gates from netlist, "
                f"{trojan_from_log} from log.txt"
            )

    return is_trojan, roles
