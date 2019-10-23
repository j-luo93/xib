"""
All language families are conjoined to form a single tree.
"""
from __future__ import annotations

from typing import Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple


class NodeFactory:

    _nodes = dict()

    def get_node(self, name: str, is_lang: bool) -> Node:
        if (name, is_lang) not in self._nodes:
            node = Node(name, is_lang)
            self._nodes[(name, is_lang)] = node
        return self._nodes[(name, is_lang)]

    def get_all_nodes(self) -> Dict[Tuple[str, bool], Node]:
        return self._nodes


node_factory = NodeFactory()


@dataclass(unsafe_hash=True)
class Node:
    name: str
    is_lang: bool  # NOTE(j_luo) Whether this node is a language or a family.
    parent: Node = field(init=False, default=None, repr=False, hash=False)
    children: Set[Node] = field(init=False, default_factory=set, hash=False)

    def add_parent(self, parent: Node):
        if self.parent is not None:
            assert self.parent is parent
        else:
            self.parent = parent
        self.parent.children.add(self)


def get_families(path: str) -> Node:
    root = node_factory.get_node('root', False)
    with Path(path).open('r', encoding='utf8') as fin:
        for line in fin:
            names = line.strip().split('|')
            nodes = [node_factory.get_node(name, i == 0) for i, name in enumerate(names)]
            for node, parent in zip(nodes, nodes[1:] + [root]):
                node.add_parent(parent)
    return root


def get_distance(lang1: str, lang2: str) -> int:
    """Get the tree distance between lang1 and lang2."""
    node1 = node_factory.get_node(lang1, True)
    node2 = node_factory.get_node(lang2, True)

    def _get_path(node: Node) -> List[Node]:
        path = list()
        while node.name != 'root':
            path.append(node)
            node = node.parent
        return list(reversed(path))

    path1 = _get_path(node1)
    path2 = _get_path(node2)

    i = 0
    while i < len(path1) and i < len(path2) and path1[i] is path2[i]:
        i += 1

    if i == len(path1):
        return len(path2) - i
    elif i == len(path2):
        return len(path1) - i
    else:
        length1 = len(path1) - i
        length2 = len(path2) - i
        return length1 + length2


def get_all_distances() -> Dict[str, Node]:
    nodes = node_factory.get_all_nodes()
    distances = dict()
    for (name1, is_lang1), node1 in nodes.items():
        if name1 != 'root' and is_lang1:
            for (name2, is_lang2), node2 in nodes.items():
                if name2 != 'root' and is_lang2:
                    distances[(name1, name2)] = get_distance(name1, name2)
    return distances
