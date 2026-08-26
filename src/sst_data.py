"""Parsing utilities for the Stanford Sentiment Treebank PTB-tree format
(sentiment-treebank-master/binary/*.txt).

Each line is a fully bracketed binary parse tree, e.g.:
    (1 (-1 (-1 The) (-1 Rock)) (1 ...))
Leaf nodes are (label word); internal nodes are (label child child).
label is 0 (negative), 1 (positive), or -1 (neutral / no label).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


@dataclass
class Node:
    label: int
    word: Optional[str] = None
    children: List["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.word is not None

    def text(self) -> str:
        if self.is_leaf:
            return self.word
        return " ".join(child.text() for child in self.children)


def _parse(tokens: List[str], pos: int) -> Tuple[Node, int]:
    assert tokens[pos] == "(", f"expected '(' at token {pos}, got {tokens[pos]!r}"
    pos += 1
    label = int(tokens[pos])
    pos += 1
    if tokens[pos] == "(":
        children = []
        while tokens[pos] == "(":
            child, pos = _parse(tokens, pos)
            children.append(child)
        node = Node(label=label, children=children)
    else:
        # Leaves are usually a single word, but a few are multi-word atomic
        # phrases with only one label, e.g. (-1 8 1\/2) for the film "8 1/2".
        words = []
        while tokens[pos] not in ("(", ")"):
            # the corpus backslash-escapes a few punctuation chars (\/ \*)
            words.append(re.sub(r"\\(.)", r"\1", tokens[pos]))
            pos += 1
        node = Node(label=label, word=" ".join(words))
    assert tokens[pos] == ")", f"expected ')' at token {pos}, got {tokens[pos]!r}"
    pos += 1
    return node, pos


def parse_line(line: str) -> Node:
    tokens = _TOKEN_RE.findall(line.strip())
    node, pos = _parse(tokens, 0)
    assert pos == len(tokens), "trailing tokens after parsing tree"
    return node


def load_trees(path: Path) -> List[Node]:
    trees = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(parse_line(line))
    return trees


def _iter_labeled_phrases(node: Node) -> Iterator[Tuple[str, int]]:
    if node.label != -1:
        yield node.text(), node.label
    for child in node.children:
        yield from _iter_labeled_phrases(child)


def phrase_level_dataset(trees: List[Node]) -> Tuple[List[str], List[int]]:
    """Every non-neutral phrase (all subtrees) in the given trees.

    Used for the training split, matching the standard SST-2 recipe where
    the training set is augmented with every labeled sub-phrase, not just
    full sentences (this is why SST-2 train is ~67k examples vs ~8.5k trees).
    """
    texts, labels = [], []
    for tree in trees:
        for text, label in _iter_labeled_phrases(tree):
            texts.append(text)
            labels.append(label)
    return texts, labels


def sentence_level_dataset(trees: List[Node]) -> Tuple[List[str], List[int]]:
    """Only non-neutral root sentences. Used for dev/test evaluation."""
    texts, labels = [], []
    for tree in trees:
        if tree.label != -1:
            texts.append(tree.text())
            labels.append(tree.label)
    return texts, labels


def load_split(data_dir: Path, name: str) -> List[Node]:
    return load_trees(data_dir / f"{name}-binary.txt")
