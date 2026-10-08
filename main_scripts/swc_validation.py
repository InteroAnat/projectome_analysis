"""Parse SWC rows and reject malformed topology before any traversal.

Validation is structural only. It preserves node order, custom nonnegative
SWC types, and transitions between dendrite and axon types; it does not repair
parents or infer whether a reconstruction is biologically complete.
"""

from decimal import Decimal, InvalidOperation
import math


def _integer(token: str, field: str, location: str) -> int:
    """Accept integer-valued numeric notation without rounding identifiers."""
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError(f"{location}: {field} must be an integer") from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"{location}: {field} must be a finite integer")
    return int(value)


def parse_swc(text: str, source: str = "SWC") -> list[tuple]:
    """Return validated ``(id, type, x, y, z, radius, parent)`` rows.

    Rows may be unsorted and use integer-valued decimal/scientific notation.
    Blank/comment lines and trailing comments are allowed. Extra columns are
    left to SWC extensions; only the seven standard fields drive traversal.
    Each non-root parent must exist, and every node must reach the one root.
    """
    if not isinstance(text, str):
        raise ValueError(f"{source}: SWC data must be text")

    rows = []
    parents = {}
    roots = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        location = f"{source}, line {line_number}"
        if len(fields) < 7:
            raise ValueError(f"{location}: expected at least 7 SWC columns")
        node_id = _integer(fields[0], "node ID", location)
        node_type = _integer(fields[1], "node type", location)
        parent = _integer(fields[6], "parent ID", location)
        if node_id <= 0:
            raise ValueError(f"{location}: node ID must be positive")
        if node_id in parents:
            raise ValueError(f"{location}: duplicate node ID {node_id}")
        if node_type < 0:
            raise ValueError(f"{location}: node type must be nonnegative")
        if parent != -1 and parent <= 0:
            raise ValueError(f"{location}: parent ID must be -1 or positive")
        try:
            coordinates = tuple(float(value) for value in fields[2:6])
        except ValueError as exc:
            raise ValueError(f"{location}: coordinates and radius must be numeric") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{location}: coordinates and radius must be finite")
        if coordinates[3] < 0:
            raise ValueError(f"{location}: radius must be nonnegative")
        rows.append((node_id, node_type, *coordinates, parent))
        parents[node_id] = parent
        if parent == -1:
            roots.append(node_id)

    if len(roots) != 1:
        raise ValueError(f"{source}: expected exactly one root (parent -1), found {len(roots)}")
    for node_id, parent in parents.items():
        if parent != -1 and parent not in parents:
            raise ValueError(f"{source}: node {node_id} references missing parent {parent}")

    # Iterative parent walks avoid recursion limits on densely sampled axons.
    # A completed walk is memoized, so validation remains linear in node count.
    complete = set()
    for start in parents:
        path = []
        active = set()
        node_id = start
        while node_id != -1 and node_id not in complete:
            if node_id in active:
                raise ValueError(f"{source}: cycle detected at node {node_id}")
            active.add(node_id)
            path.append(node_id)
            node_id = parents[node_id]
        complete.update(path)
    return rows
