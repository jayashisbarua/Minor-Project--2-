"""Robust Verilog gate-level netlist parser for NeuroGuard-X.

Fixes the critical bugs present in legacy implementations:
1. Correctly parses escaped identifiers (e.g. `\\DFF_637/Q_reg `).
2. Recognizes both Q and QN as output ports on sequential flip-flops.
3. Maps all TRIT standard cells and positional primitives via cell_library.
4. Correctly models constant assigns (`1'b0`, `1'b1`) as constant driver nodes.
5. Preserves Primary Inputs (PI) and Primary Outputs (PO) pseudo-nodes without spurious PIs.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import networkx as nx

from neuroguard.parsing.cell_library import classify_cell, CellInfo
from neuroguard.parsing.labels import parse_log, build_gate_labels


class NetlistParser:
    """Parses gate-level Verilog files into NetworkX directed graphs."""

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = data_root

    def parse(
        self,
        netlist_path: str | Path,
        log_path: Optional[str | Path] = None,
    ) -> nx.MultiDiGraph:
        """Parse a netlist file into a directed graph.

        Parameters
        ----------
        netlist_path: Path to the .v file.
        log_path: Optional path to log.txt. If omitted and in TRIT directory, auto-discovered.

        Returns
        -------
        nx.MultiDiGraph with node & edge attributes.
        """
        netlist_path = Path(netlist_path)
        if log_path is None:
            candidate_log = netlist_path.parent / "log.txt"
            if candidate_log.exists():
                log_path = candidate_log

        with open(netlist_path, "r", errors="ignore") as f:
            content = f.read()

        # Remove line and block comments
        content = re.sub(r"//.*", "", content)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

        # Split into statements by semicolon
        statements = content.split(";")

        G = nx.MultiDiGraph()
        G.graph["name"] = netlist_path.stem
        G.graph["path"] = str(netlist_path)

        primary_inputs: Set[str] = set()
        primary_outputs: Set[str] = set()
        declared_wires: Set[str] = set()

        # Net -> (driver_node, driver_port)
        net_drivers: Dict[str, Tuple[str, str]] = {}
        # List of (load_node, load_port, net_name)
        gate_loads: List[Tuple[str, str, str]] = []
        # Net aliases from assign lhs = rhs
        aliases: List[Tuple[str, str]] = []

        parsed_instances: List[Tuple[str, str, Dict[str, List[str]]]] = []

        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue

            if (
                stmt.startswith(("module", "endmodule", "always", "initial", "task", "function"))
                or stmt == "endmodule"
            ):
                continue

            # Primary Inputs
            m_in = re.match(r"^input\s+(.*)", stmt, flags=re.DOTALL)
            if m_in:
                for item in m_in.group(1).split(","):
                    item = item.strip()
                    if item:
                        primary_inputs.add(item)
                        pi_node = f"PI_{item}"
                        G.add_node(
                            pi_node,
                            type="PI",
                            cell_type="PI",
                            family="PI",
                            drive_strength=1,
                            is_sequential=False,
                            is_pi=True,
                            is_po=False,
                            is_const=False,
                        )
                        net_drivers[item] = (pi_node, "out")
                continue

            # Primary Outputs
            m_out = re.match(r"^output\s+(.*)", stmt, flags=re.DOTALL)
            if m_out:
                for item in m_out.group(1).split(","):
                    item = item.strip()
                    if item:
                        primary_outputs.add(item)
                        po_node = f"PO_{item}"
                        G.add_node(
                            po_node,
                            type="PO",
                            cell_type="PO",
                            family="PO",
                            drive_strength=1,
                            is_sequential=False,
                            is_pi=False,
                            is_po=True,
                            is_const=False,
                        )
                continue

            # Wire declarations
            m_wire = re.match(r"^wire\s+(.*)", stmt, flags=re.DOTALL)
            if m_wire:
                for item in m_wire.group(1).split(","):
                    item = item.strip()
                    if item:
                        declared_wires.add(item)
                continue

            # Continuous Assigns
            m_assign = re.match(r"^assign\s+(\S+)\s*=\s*(\S+)", stmt)
            if m_assign:
                lhs, rhs = m_assign.group(1), m_assign.group(2)
                if rhs in ("1'b0", "0"):
                    const_node = "CONST_0"
                    if const_node not in G:
                        G.add_node(
                            const_node,
                            type="CONST",
                            cell_type="CONST",
                            family="CONST",
                            drive_strength=1,
                            is_sequential=False,
                            is_pi=False,
                            is_po=False,
                            is_const=True,
                        )
                    net_drivers[lhs] = (const_node, "out")
                elif rhs in ("1'b1", "1"):
                    const_node = "CONST_1"
                    if const_node not in G:
                        G.add_node(
                            const_node,
                            type="CONST",
                            cell_type="CONST",
                            family="CONST",
                            drive_strength=1,
                            is_sequential=False,
                            is_pi=False,
                            is_po=False,
                            is_const=True,
                        )
                    net_drivers[lhs] = (const_node, "out")
                else:
                    aliases.append((lhs, rhs))
                continue

            # Gate / Cell Instance
            # Format: cell_type inst_name ( ... )
            m_inst = re.match(r"^(\w+)\s+(\\\S+|\S+)\s*\((.*)\)", stmt, flags=re.DOTALL)
            if m_inst:
                cell_type = m_inst.group(1)
                inst_name = m_inst.group(2)
                port_str = m_inst.group(3)

                info = classify_cell(cell_type)
                G.add_node(
                    inst_name,
                    type="GATE",
                    cell_type=cell_type,
                    family=info.function_family,
                    drive_strength=info.drive_strength,
                    is_sequential=info.is_sequential,
                    is_pi=False,
                    is_po=False,
                    is_const=False,
                )

                inst_ports: Dict[str, List[str]] = {"outputs": [], "inputs": []}

                # Check if named ports: .port(net)
                named_ports = re.findall(r"\.([a-zA-Z0-9_]+)\s*\(\s*([^)]*)\s*\)", port_str)
                if named_ports:
                    for port, net in named_ports:
                        net = net.strip()
                        if not net:
                            continue
                        if port in info.output_ports or port in ("Q", "QN"):
                            net_drivers[net] = (inst_name, port)
                            inst_ports["outputs"].append(net)
                        else:
                            gate_loads.append((inst_name, port, net))
                            inst_ports["inputs"].append(net)
                else:
                    # Positional ports: primitive (out, in1, in2, ...)
                    ports = [p.strip() for p in port_str.split(",") if p.strip()]
                    if ports:
                        out_net = ports[0]
                        net_drivers[out_net] = (inst_name, "Q")
                        inst_ports["outputs"].append(out_net)
                        for idx, in_net in enumerate(ports[1:]):
                            port_id = f"DIN{idx+1}"
                            gate_loads.append((inst_name, port_id, in_net))
                            inst_ports["inputs"].append(in_net)

                parsed_instances.append((cell_type, inst_name, inst_ports))

        # Resolve net aliases
        for lhs, rhs in aliases:
            if rhs in net_drivers:
                net_drivers[lhs] = net_drivers[rhs]

        # Connect gate load edges: driver -> load
        for inst_name, port, net in gate_loads:
            if net in net_drivers:
                driver, driver_port = net_drivers[net]
                G.add_edge(driver, inst_name, net=net, port=port, driver_port=driver_port)

        # Connect PO edges: driver -> PO
        for po in primary_outputs:
            po_node = f"PO_{po}"
            if po in net_drivers:
                driver, driver_port = net_drivers[po]
                G.add_edge(driver, po_node, net=po, port="in", driver_port=driver_port)

        # Build labels if log or Trojan names present
        log_info = parse_log(log_path) if log_path else None
        is_trojan, roles = build_gate_labels(parsed_instances, log_info, G)

        for node in G.nodes():
            if G.nodes[node]["type"] == "GATE":
                G.nodes[node]["is_trojan"] = is_trojan.get(node, 0)
                G.nodes[node]["role"] = roles.get(node, 0)
            else:
                G.nodes[node]["is_trojan"] = 0
                G.nodes[node]["role"] = 0

        # Store graph level summary in G.graph
        trojan_gate_count = sum(d.get("is_trojan", 0) for _, d in G.nodes(data=True))
        G.graph["is_trojan"] = int(trojan_gate_count > 0)
        G.graph["trojan_gate_count"] = trojan_gate_count
        G.graph["num_gates"] = len(parsed_instances)
        G.graph["num_inputs"] = len(primary_inputs)
        G.graph["num_outputs"] = len(primary_outputs)

        return G


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse netlist and print graph statistics.")
    parser.add_argument("netlist", type=str, help="Path to .v file")
    parser.add_argument("--log", type=str, default=None, help="Path to log.txt")
    args = parser.parse_args()

    p = NetlistParser()
    g = p.parse(args.netlist, args.log)
    print(f"Graph: {g.graph.get('name')}")
    print(f"Nodes: {g.number_of_nodes()} (Gates: {g.graph.get('num_gates')}, Inputs: {g.graph.get('num_inputs')}, Outputs: {g.graph.get('num_outputs')})")
    print(f"Edges: {g.number_of_edges()}")
    print(f"Trojan gates: {g.graph.get('trojan_gate_count')} (Graph label: {g.graph.get('is_trojan')})")
