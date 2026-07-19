#!/usr/bin/env python3
"""Convert Neurolucida ASC (Liu 2026) to SWC for navis / morph scoring.

SWC types: 1=soma, 2=axon, 3=basal dendrite, 4=apical dendrite.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

POINT_RE = re.compile(
    r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"
)

# Map ASC section keywords → SWC type
SECTION_TYPE = {
    "cellbody": 1,
    "soma": 1,
    "axon": 2,
    "dendrite": 3,
    "basal": 3,
    "apical": 4,
}


@dataclass
class Node:
    n: int
    t: int
    x: float
    y: float
    z: float
    r: float
    parent: int


@dataclass
class TreeState:
    nodes: list[Node] = field(default_factory=list)
    next_id: int = 1
    soma_id: int | None = None
    soma_xyz: tuple[float, float, float] | None = None


def _strip_comments(line: str) -> str:
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()


def _section_type(header_chunk: str) -> int | None:
    low = header_chunk.lower()
    for key, t in SECTION_TYPE.items():
        if f"({key})" in low or f'"{key}"' in low:
            return t
    # Contours named CellBody
    if "cellbody" in low:
        return 1
    return None


def parse_asc(path: Path) -> TreeState:
    """Parse ASC into SWC nodes (best-effort Neurolucida V3)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Tokenize roughly by parentheses depth for trees
    state = TreeState()
    # First pass: soma contour → mean center as soma node
    soma_pts: list[tuple[float, float, float, float]] = []
    in_cellbody = False
    depth = 0
    cellbody_depth = None

    # Simple line scan for CellBody contour
    for raw in text.splitlines():
        line = _strip_comments(raw)
        if not line:
            continue
        if "cellbody" in line.lower() and "(" in line:
            in_cellbody = True
            cellbody_depth = depth + line.count("(") - line.count(")")
            # continue collecting
        if in_cellbody:
            for m in POINT_RE.finditer(line):
                x, y, z, d = map(float, m.groups())
                soma_pts.append((x, y, z, d / 2.0))  # diameter → radius
            depth_delta = line.count("(") - line.count(")")
            depth += depth_delta
            if cellbody_depth is not None and depth <= cellbody_depth - 1:
                in_cellbody = False
                cellbody_depth = None

    if soma_pts:
        xs = [p[0] for p in soma_pts]
        ys = [p[1] for p in soma_pts]
        zs = [p[2] for p in soma_pts]
        rs = [p[3] for p in soma_pts]
        cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
        # equivalent radius from mean contour radius or extent
        r_mean = sum(rs) / len(rs)
        state.soma_xyz = (cx, cy, cz)
        state.nodes.append(Node(1, 1, cx, cy, cz, max(r_mean, 1.0), -1))
        state.soma_id = 1
        state.next_id = 2
    else:
        # placeholder; first tree root becomes soma
        state.nodes.append(Node(1, 1, 0.0, 0.0, 0.0, 1.0, -1))
        state.soma_id = 1
        state.next_id = 2
        state.soma_xyz = (0.0, 0.0, 0.0)

    # Second pass: parse Axon / Dendrite / Apical trees with | bifurcations
    # Strategy: find each top-level tree starting with (Axon)/(Dendrite)/(Apical)
    # and walk character stream with a stack of parent node ids.
    _parse_trees(text, state)
    return state


def _parse_trees(text: str, state: TreeState) -> None:
    # Remove comments
    lines = []
    for raw in text.splitlines():
        lines.append(_strip_comments(raw))
    joined = "\n".join(lines)

    # Find tree starts
    for m in re.finditer(r"\(\s*(Axon|Dendrite|Apical|Basal)\s*\)", joined, re.I):
        kind = m.group(1).lower()
        swc_t = SECTION_TYPE.get(kind, 3)
        # Find opening paren of the enclosing tree: walk back to matching '('
        start = m.start()
        # The tree usually looks like: ( (Color ...) (Dendrite) ... )
        # Walk left to find the outer '(' of this tree block
        i = start
        while i > 0 and joined[i] != "(":
            i -= 1
        # Prefer the paren that opens the tree group: go further left one level if nested
        # Find content after the type tag until the matching close of the tree's outer paren
        # Simpler: from m.end(), parse nested structure with stack
        _parse_one_tree(joined, m.end(), swc_t, state)


def _parse_one_tree(text: str, pos: int, swc_t: int, state: TreeState) -> int:
    """Parse points/bifurcations after a (Dendrite)/(Axon)/(Apical) tag.

    Neurolucida uses:
      (x y z d)
      |          ← sibling branch at same parent
    Nested parentheses group subtrees.
    """
    parent_stack: list[int] = [state.soma_id if state.soma_id is not None else 1]
    last_node = parent_stack[-1]
    i = pos
    n = len(text)
    depth0 = 0
    # We are inside the tree; track paren depth relative to entry
    # Skip until we see first point or nested structure
    while i < n:
        # skip whitespace
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        ch = text[i]
        if ch == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "|":
            # bifurcation: next point attaches to same parent as last sibling
            if len(parent_stack) >= 2:
                last_node = parent_stack[-2]
            else:
                last_node = parent_stack[-1]
            i += 1
            continue
        if ch == "(":
            # Could be point, color, or subtree
            close = text.find(")", i)
            if close < 0:
                break
            chunk = text[i : close + 1]
            pm = POINT_RE.match(chunk)
            if pm:
                x, y, z, d = map(float, pm.groups())
                nid = state.next_id
                state.next_id += 1
                parent = last_node
                state.nodes.append(Node(nid, swc_t, x, y, z, max(d / 2.0, 0.01), parent))
                # push as current tip
                # If next non-ws is '(', this may start a child group — keep stack
                last_node = nid
                # Maintain stack: replace tip at this depth
                parent_stack.append(nid)
                i = close + 1
                continue
            # Non-point paren: Color / Dendrite already handled — skip tag or recurse subtree
            inner = chunk[1:-1].strip().lower()
            if inner.startswith("color") or inner in ("axon", "dendrite", "apical", "basal", "cellbody"):
                i = close + 1
                continue
            # Nested group: recurse into children with current last_node as parent
            parent_stack.append(last_node)
            i = _parse_subtree_group(text, i + 1, swc_t, state, last_node)
            if parent_stack:
                parent_stack.pop()
            if parent_stack:
                last_node = parent_stack[-1]
            continue
        if ch == ")":
            # end of enclosing group
            return i + 1
        # unknown token — skip char
        i += 1
    return i


def _parse_subtree_group(
    text: str, pos: int, swc_t: int, state: TreeState, parent: int
) -> int:
    """Parse content of a nested (...) group of branches."""
    last_node = parent
    branch_parent = parent
    i = pos
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        ch = text[i]
        if ch == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "|":
            last_node = branch_parent
            i += 1
            continue
        if ch == ")":
            return i + 1
        if ch == "(":
            close = text.find(")", i)
            if close < 0:
                return i
            chunk = text[i : close + 1]
            pm = POINT_RE.match(chunk)
            if pm:
                x, y, z, d = map(float, pm.groups())
                nid = state.next_id
                state.next_id += 1
                state.nodes.append(
                    Node(nid, swc_t, x, y, z, max(d / 2.0, 0.01), last_node)
                )
                last_node = nid
                i = close + 1
                continue
            inner = chunk[1:-1].strip().lower()
            if inner.startswith("color") or inner in (
                "axon",
                "dendrite",
                "apical",
                "basal",
                "cellbody",
            ):
                i = close + 1
                continue
            # nested
            i = _parse_subtree_group(text, i + 1, swc_t, state, last_node)
            continue
        i += 1
    return i


def write_swc(state: TreeState, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated from Neurolucida ASC (Liu 2026 VEN atlas)",
        "# n T X Y Z R P",
    ]
    for nd in state.nodes:
        lines.append(
            f"{nd.n} {nd.t} {nd.x:.4f} {nd.y:.4f} {nd.z:.4f} {nd.r:.4f} {nd.parent}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_file(asc: Path, out: Path) -> dict:
    state = parse_asc(asc)
    write_swc(state, out)
    n_by_t: dict[int, int] = {}
    for nd in state.nodes:
        n_by_t[nd.t] = n_by_t.get(nd.t, 0) + 1
    return {
        "asc": str(asc),
        "swc": str(out),
        "n_nodes": len(state.nodes),
        "n_by_type": n_by_t,
    }


def convert_tree(
    morph_root: Path,
    out_root: Path,
    classes: list[str] | None = None,
) -> list[dict]:
    mapping = {
        "VENL": "PatchClamp_morph/VENL",
        "VENS": "PatchClamp_morph/VENS",
        "PC-L5_ET": "PatchClamp_morph/PC-L5_ET",
    }
    if classes is None:
        classes = list(mapping.keys())
    results = []
    for cls in classes:
        rel = mapping[cls]
        src = morph_root / rel
        dst = out_root / cls
        for asc in sorted(src.glob("*.ASC")):
            out = dst / (asc.stem + ".swc")
            info = convert_file(asc, out)
            info["class"] = cls
            results.append(info)
            print(f"OK {cls}/{asc.name} -> {out.name} nodes={info['n_nodes']} types={info['n_by_type']}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--morph-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "liu2026"
        / "morph"
        / "NeuroMorph_upload260215",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "liu2026" / "swc",
    )
    ap.add_argument(
        "--classes",
        nargs="*",
        default=["VENL", "VENS", "PC-L5_ET"],
    )
    args = ap.parse_args()
    convert_tree(args.morph_root, args.out_root, args.classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
