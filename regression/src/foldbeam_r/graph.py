from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Predicate:
    feature: int
    threshold: float

    @property
    def predicate_id(self) -> str:
        return f"f{self.feature:04d}_le_{self.threshold:.10g}"

    def key(self) -> tuple[int, float]:
        return int(self.feature), float(self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {"feature": int(self.feature), "threshold": float(self.threshold)}


@dataclass
class Node:
    node_id: int
    kind: str
    feature: int | None = None
    threshold: float | None = None
    left: int | None = None
    right: int | None = None
    value: float = 0.0
    n: int = 0
    sum_z: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.kind == "terminal"

    @property
    def is_open(self) -> bool:
        return self.kind == "open"

    @property
    def has_open_left(self) -> bool:
        return self.kind == "internal" and self.left is None

    @property
    def has_open_right(self) -> bool:
        return self.kind == "internal" and self.right is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Graph:
    def __init__(self, nodes: dict[int, Node] | None = None, root: int = 0):
        if nodes is None:
            nodes = {0: Node(0, "open")}
        self.nodes = nodes
        self.root = int(root)
        self.validate()

    def clone(self) -> "Graph":
        return copy.deepcopy(self)

    def next_id(self) -> int:
        return max(self.nodes) + 1 if self.nodes else 0

    def reachable_ids(self) -> list[int]:
        seen: set[int] = set()
        stack = [self.root]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.nodes[node_id]
            for child in (node.left, node.right):
                if child is not None:
                    stack.append(int(child))
        return sorted(seen)

    def reachable_count(self) -> int:
        return len(self.reachable_ids())

    def open_edges(self) -> list[tuple[int, str]]:
        edges: list[tuple[int, str]] = []
        for node_id in self.reachable_ids():
            node = self.nodes[node_id]
            if node.kind == "open":
                edges.append((node_id, "self"))
                continue
            if node.kind != "internal":
                continue
            if node.left is None:
                edges.append((node_id, "left"))
            if node.right is None:
                edges.append((node_id, "right"))
        return edges

    def is_complete(self) -> bool:
        return not self.open_edges()

    def parent_map(self) -> dict[int, list[tuple[int, str]]]:
        parents = {node_id: [] for node_id in self.reachable_ids()}
        for parent_id in self.reachable_ids():
            node = self.nodes[parent_id]
            for branch, child in (("left", node.left), ("right", node.right)):
                if child is not None and child in parents:
                    parents[int(child)].append((parent_id, branch))
        return parents

    def multi_parent_count(self) -> int:
        return sum(len(parents) > 1 for parents in self.parent_map().values())

    def would_create_cycle(self, parent_id: int, child_id: int) -> bool:
        stack = [int(child_id)]
        seen: set[int] = set()
        while stack:
            node_id = stack.pop()
            if node_id == int(parent_id):
                return True
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.nodes[node_id]
            for child in (node.left, node.right):
                if child is not None:
                    stack.append(int(child))
        return False

    def route(self, X: np.ndarray, *, stop_at_open: bool = False) -> np.ndarray:
        X = np.asarray(X)
        out = np.full(X.shape[0], self.root, dtype=np.int64)
        for row in range(X.shape[0]):
            node_id = self.root
            for _ in range(len(self.nodes) + 1):
                node = self.nodes[node_id]
                if node.is_terminal:
                    out[row] = node_id
                    break
                if node.is_open:
                    if stop_at_open:
                        out[row] = node_id
                        break
                    raise RuntimeError(f"Routing reached open node {node_id}.")
                if node.feature is None or node.threshold is None:
                    if stop_at_open:
                        out[row] = node_id
                        break
                    raise RuntimeError(f"Internal node {node_id} has no predicate.")
                branch = "left" if X[row, int(node.feature)] <= float(node.threshold) else "right"
                child = node.left if branch == "left" else node.right
                if child is None:
                    if stop_at_open:
                        out[row] = node_id
                        break
                    raise RuntimeError(f"Routing reached open edge {node_id}/{branch}.")
                node_id = int(child)
            else:
                raise RuntimeError("Routing exceeded graph depth bound.")
        return out

    def route_edges(self, X: np.ndarray) -> dict[tuple[int, str], np.ndarray]:
        rows: dict[tuple[int, str], list[int]] = {edge: [] for edge in self.open_edges()}
        for row in range(X.shape[0]):
            node_id = self.root
            for _ in range(len(self.nodes) + 1):
                node = self.nodes[node_id]
                if node.is_open:
                    rows.setdefault((node_id, "self"), []).append(row)
                    break
                if node.is_terminal:
                    break
                if node.feature is None or node.threshold is None:
                    break
                branch = "left" if X[row, int(node.feature)] <= float(node.threshold) else "right"
                child = node.left if branch == "left" else node.right
                if child is None:
                    rows.setdefault((node_id, branch), []).append(row)
                    break
                node_id = int(child)
        return {edge: np.asarray(values, dtype=np.int64) for edge, values in rows.items()}

    def rows_for_edge(self, X: np.ndarray, edge: tuple[int, str]) -> np.ndarray:
        return self.route_edges(X).get(edge, np.empty(0, dtype=np.int64))

    def rows_for_node(self, X: np.ndarray, node_id: int) -> np.ndarray:
        node_id = int(node_id)
        hits: list[int] = []
        for row in range(X.shape[0]):
            current = self.root
            for _ in range(len(self.nodes) + 1):
                if current == node_id:
                    hits.append(row)
                    break
                node = self.nodes[current]
                if node.is_terminal or node.is_open:
                    break
                if node.feature is None or node.threshold is None:
                    break
                branch = "left" if X[row, int(node.feature)] <= float(node.threshold) else "right"
                child = node.left if branch == "left" else node.right
                if child is None:
                    break
                current = int(child)
        return np.asarray(hits, dtype=np.int64)

    def add_new(self, edge: tuple[int, str], predicate: Predicate) -> None:
        node_id, branch = edge
        if branch != "self":
            raise ValueError("FoldBeam-R expands open nodes only.")
        node = self.nodes[int(node_id)]
        if not node.is_open:
            raise ValueError("NEW can only expand an open node.")
        left_id = self.next_id()
        right_id = left_id + 1
        self.nodes[int(node_id)] = Node(
            int(node_id),
            "internal",
            feature=int(predicate.feature),
            threshold=float(predicate.threshold),
            left=left_id,
            right=right_id,
        )
        self.nodes[left_id] = Node(left_id, "open")
        self.nodes[right_id] = Node(right_id, "open")
        self.validate()

    def add_reuse(self, edge: tuple[int, str], target_id: int) -> None:
        node_id, branch = edge
        if branch != "self":
            raise ValueError("FoldBeam-R reuses from open nodes only.")
        if int(node_id) == self.root:
            raise ValueError("The root open node cannot be reused.")
        if not self.nodes[int(node_id)].is_open:
            raise ValueError("REUSE can only close an open node.")
        target_id = int(target_id)
        if target_id == int(node_id) or self.would_create_cycle(int(node_id), target_id):
            raise ValueError("REUSE would create a cycle.")
        for parent_id, parent_branch in self.parent_map()[int(node_id)]:
            parent = self.nodes[parent_id]
            if parent_branch == "left":
                parent.left = target_id
            else:
                parent.right = target_id
        del self.nodes[int(node_id)]
        self.validate()

    def stop(self, edge: tuple[int, str]) -> None:
        node_id, branch = edge
        if branch != "self":
            raise ValueError("FoldBeam-R stops open nodes only.")
        node_id = int(node_id)
        if not self.nodes[node_id].is_open:
            raise ValueError("STOP can only close an open node.")
        self.nodes[node_id] = Node(node_id, "terminal")
        self.validate()

    def close_all(self) -> None:
        for edge in list(self.open_edges()):
            self.stop(edge)

    def fit_terminals(self, X: np.ndarray, z: np.ndarray, lam: float) -> None:
        if not self.is_complete():
            clone = self.clone()
            clone.close_all()
            self.nodes = clone.nodes
        leaves = self.route(X)
        global_mean = float(np.mean(z)) if len(z) else 0.0
        for node_id in self.reachable_ids():
            node = self.nodes[node_id]
            if not node.is_terminal:
                continue
            values = np.asarray(z[leaves == node_id], dtype=np.float64)
            node.n = int(values.size)
            node.sum_z = float(values.sum())
            node.value = float((node.sum_z + float(lam) * global_mean) / (node.n + float(lam)))

    def predict_z(self, X: np.ndarray) -> np.ndarray:
        leaves = self.route(X)
        return np.asarray([self.nodes[int(node_id)].value for node_id in leaves], dtype=np.float64)

    def path_conditions(self, node_id: int, p_max: int = 8) -> list[list[tuple[int, str, float]]]:
        paths: list[list[tuple[int, str, float]]] = []

        def walk(current: int, path: list[tuple[int, str, float]]) -> None:
            if len(paths) >= p_max:
                return
            if current == node_id:
                paths.append(list(path))
                return
            node = self.nodes[current]
            if node.kind != "internal" or node.feature is None or node.threshold is None:
                return
            for branch, child in (("left", node.left), ("right", node.right)):
                if child is None:
                    continue
                op = "<=" if branch == "left" else ">"
                walk(int(child), path + [(int(node.feature), op, float(node.threshold))])

        walk(self.root, [])
        return paths or [[]]

    def canonical_hash(self) -> str:
        payload = []
        for node_id in self.reachable_ids():
            node = self.nodes[node_id]
            payload.append(
                (
                    node.node_id,
                    node.kind,
                    node.feature,
                    None if node.threshold is None else round(float(node.threshold), 12),
                    node.left,
                    node.right,
                    round(float(node.value), 12),
                    int(node.n),
                    round(float(node.sum_z), 12),
                )
            )
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "nodes": [self.nodes[i].to_dict() for i in sorted(self.nodes)]}

    def validate(self) -> None:
        if self.root not in self.nodes:
            raise ValueError("Root node is missing.")
        for node in self.nodes.values():
            if node.kind not in {"internal", "terminal", "open"}:
                raise ValueError(f"Unknown node kind: {node.kind}")
            if node.kind in {"terminal", "open"}:
                continue
            if (node.feature is None) != (node.threshold is None):
                raise ValueError(f"Internal node {node.node_id} has partial predicate.")
        temp: set[int] = set()
        done: set[int] = set()

        def visit(node_id: int) -> None:
            if node_id in done:
                return
            if node_id in temp:
                raise ValueError("Graph contains a directed cycle.")
            temp.add(node_id)
            node = self.nodes[node_id]
            for child in (node.left, node.right):
                if child is None:
                    continue
                if child not in self.nodes:
                    raise ValueError(f"Node {node_id} points to missing child {child}.")
                visit(int(child))
            temp.remove(node_id)
            done.add(node_id)

        visit(self.root)

    @staticmethod
    def _set_child(parent: Node, branch: str, child_id: int) -> None:
        if branch == "left":
            if parent.left is not None:
                raise ValueError("Left edge is already closed.")
            parent.left = int(child_id)
        elif branch == "right":
            if parent.right is not None:
                raise ValueError("Right edge is already closed.")
            parent.right = int(child_id)
        else:
            raise ValueError(f"Unknown branch: {branch}")


def fresh_graph() -> Graph:
    return Graph()
