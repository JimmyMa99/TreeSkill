"""Skill Tree — hierarchical skill management with folder-based structure.

A "skill tree" is a directory on disk where each sub-directory represents
a child skill group.  The layout looks like::

    my-skills/
    ├── _meta.yaml          # package-level metadata
    ├── root.yaml           # root skill (the "catch-all" prompt)
    ├── social/
    │   ├── _meta.yaml
    │   ├── root.yaml       # social sub-skill
    │   ├── moments.yaml    # WeChat Moments specialty
    │   └── weibo.yaml
    └── business/
        ├── _meta.yaml
        ├── root.yaml
        ├── email.yaml
        └── product.yaml

At runtime, the tree is loaded into a ``SkillNode`` object that mirrors the
folder hierarchy.  Nodes can be added, split, merged, and pruned — either
manually via CLI commands or automatically during APO optimization.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from evoskill.schema import Skill, SkillMeta
from evoskill import skill as skill_module

logger = logging.getLogger(__name__)

_META_FILE = "_meta.yaml"
_ROOT_SKILL = "root.yaml"


# ---------------------------------------------------------------------------
# SkillNode — in-memory representation of one tree node
# ---------------------------------------------------------------------------

@dataclass
class SkillNode:
    """One node in the skill tree.

    Attributes
    ----------
    name : str
        Human-readable name (from ``_meta.yaml`` or directory name).
    skill : Skill
        The ``root.yaml`` skill attached to this node.
    children : dict[str, SkillNode]
        Ordered mapping of child-name → child-node.
    meta : SkillMeta | None
        Extra metadata loaded from ``_meta.yaml``.
    path : Path | None
        Disk path this node was loaded from (``None`` for in-memory nodes).
    age : int
        Number of optimization rounds this node has survived. Used for progressive disclosure.
    usage_count : int
        Number of times this node has been used/routed to.
    collapsed : bool
        If True, this node is "folded" (hidden from routing) but not deleted.
        Implements progressive disclosure - preserve knowledge for future use.
    """

    name: str
    skill: Skill
    children: Dict[str, "SkillNode"] = field(default_factory=dict)
    meta: Optional[SkillMeta] = None
    path: Optional[Path] = None
    age: int = 0  # For progressive disclosure - tracks node maturity
    usage_count: int = 0  # Track how often this node is used
    collapsed: bool = False  # Progressive disclosure - hide without deleting

    # -- convenience -------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaf_count(self) -> int:
        """Return the total number of leaf skills."""
        if self.is_leaf:
            return 1
        return sum(c.leaf_count() for c in self.children.values())

    def all_skills(self) -> List[Skill]:
        """Collect every Skill in the subtree (DFS, root-first)."""
        result = [self.skill]
        for child in self.children.values():
            result.extend(child.all_skills())
        return result


# ---------------------------------------------------------------------------
# SkillTree — the public API
# ---------------------------------------------------------------------------

class SkillTree:
    """Manages a hierarchical skill package stored as a directory tree.

    Parameters
    ----------
    root : SkillNode
        Root node of the tree.
    base_path : Path
        Top-level directory on disk.
    """

    def __init__(self, root: SkillNode, base_path: Path) -> None:
        self.root = root
        self.base_path = base_path

    # ------------------------------------------------------------------
    # Factory: load from disk
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SkillTree":
        """Recursively load a skill tree from a directory.

        If *path* points to a single ``.yaml`` file instead of a directory,
        a trivial one-node tree is returned (backwards-compatible with flat
        skill files).
        """
        path = Path(path)
        if path.is_file():
            # Flat skill file → wrap in a single-node tree
            sk = skill_module.load(path)
            node = SkillNode(name=sk.name, skill=sk, path=path.parent)
            return cls(root=node, base_path=path.parent)

        root = _load_node(path)
        return cls(root=root, base_path=path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the entire tree to disk.

        Returns the directory path.
        """
        dest = Path(path) if path else self.base_path
        _save_node(self.root, dest)
        self.base_path = dest
        return dest

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def get(self, dotpath: str) -> SkillNode:
        """Retrieve a node by dot-separated path, e.g. ``social.moments``.

        Raises ``KeyError`` if the path does not exist.
        """
        parts = [p for p in dotpath.split(".") if p]
        node = self.root
        for p in parts:
            if p not in node.children:
                raise KeyError(
                    f"Child '{p}' not found under '{node.name}'. "
                    f"Available: {list(node.children.keys())}"
                )
            node = node.children[p]
        return node

    def list_tree(self, node: Optional[SkillNode] = None, prefix: str = "") -> str:
        """Return a pretty-printed tree string."""
        node = node or self.root
        lines: List[str] = []
        _format_tree(node, lines, prefix, is_last=True)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mutation: add / split / merge / prune
    # ------------------------------------------------------------------

    def add_child(
        self,
        parent_path: str,
        child_name: str,
        skill: Skill,
        description: Optional[str] = None,
    ) -> SkillNode:
        """Add a new child skill under *parent_path*.

        Returns the newly created ``SkillNode``.
        """
        parent = self.get(parent_path) if parent_path else self.root
        if child_name in parent.children:
            raise ValueError(f"Child '{child_name}' already exists under '{parent.name}'")

        meta = SkillMeta(name=child_name, description=description)
        child = SkillNode(name=child_name, skill=skill, meta=meta)
        parent.children[child_name] = child
        logger.info("Added child '%s' under '%s'", child_name, parent.name)
        return child

    def split(
        self,
        node_path: str,
        children_specs: List[Dict],
    ) -> List[SkillNode]:
        """Split a node into multiple children.

        Each item in *children_specs* is a dict with keys:
        ``name``, ``system_prompt``, and optionally ``description``.

        The original node keeps its prompt (as the "fallback") and gains
        the new children.

        Returns the list of newly created ``SkillNode``s.
        """
        parent = self.get(node_path) if node_path else self.root
        created: List[SkillNode] = []
        for spec in children_specs:
            child_skill = parent.skill.model_copy(
                update={
                    "name": spec["name"],
                    "system_prompt": spec["system_prompt"],
                    "version": "v1.0",
                }
            )
            meta = SkillMeta(
                name=spec["name"],
                description=spec.get("description"),
            )
            child = SkillNode(name=spec["name"], skill=child_skill, meta=meta)
            parent.children[spec["name"]] = child
            created.append(child)

        logger.info(
            "Split '%s' into %d children: %s",
            parent.name,
            len(created),
            [c.name for c in created],
        )
        return created

    def merge(self, node_paths: List[str], merged_name: str, merged_prompt: str) -> SkillNode:
        """Merge multiple sibling nodes into one.

        The merged node replaces the siblings under their common parent.
        """
        if len(node_paths) < 2:
            raise ValueError("Need at least 2 nodes to merge")

        # Find common parent
        nodes = [self.get(p) for p in node_paths]
        parent_path = ".".join(node_paths[0].split(".")[:-1])
        parent = self.get(parent_path) if parent_path else self.root

        # Remove old children
        for np in node_paths:
            child_name = np.split(".")[-1]
            parent.children.pop(child_name, None)

        # Create merged child
        merged_skill = nodes[0].skill.model_copy(
            update={
                "name": merged_name,
                "system_prompt": merged_prompt,
                "version": "v1.0",
            }
        )
        merged_node = SkillNode(
            name=merged_name,
            skill=merged_skill,
            meta=SkillMeta(name=merged_name, description=f"Merged from: {node_paths}"),
        )
        parent.children[merged_name] = merged_node
        logger.info("Merged %s → '%s'", node_paths, merged_name)
        return merged_node

    def prune(self, node_path: str) -> None:
        """Remove a leaf node. Its responsibilities revert to the parent."""
        parts = node_path.split(".")
        child_name = parts[-1]
        parent_path = ".".join(parts[:-1])
        parent = self.get(parent_path) if parent_path else self.root

        if child_name not in parent.children:
            raise KeyError(f"'{child_name}' not found under '{parent.name}'")

        removed = parent.children.pop(child_name)
        logger.info(
            "Pruned '%s' (was under '%s'). "
            "Its responsibilities revert to the parent.",
            removed.name,
            parent.name,
        )


# ---------------------------------------------------------------------------
# Internal helpers — disk I/O
# ---------------------------------------------------------------------------

def _load_node(directory: Path) -> SkillNode:
    """Recursively load a single tree node from *directory*."""
    # Load meta
    meta: Optional[SkillMeta] = None
    meta_path = directory / _META_FILE
    if meta_path.is_file():
        raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        meta = SkillMeta.model_validate(raw)

    # Load root skill
    root_path = directory / _ROOT_SKILL
    if root_path.is_file():
        sk = skill_module.load(root_path)
    else:
        # Fallback: any single .yaml file that isn't _meta.yaml
        yamls = [
            f for f in directory.glob("*.yaml")
            if f.name != _META_FILE
        ]
        if yamls:
            sk = skill_module.load(yamls[0])
        else:
            raise FileNotFoundError(
                f"No root.yaml or skill YAML found in {directory}"
            )

    name = meta.name if meta else directory.name
    node = SkillNode(name=name, skill=sk, meta=meta, path=directory)

    # Recurse into subdirectories
    for sub in sorted(directory.iterdir()):
        if sub.is_dir() and not sub.name.startswith((".", "_")):
            # Only load if it contains at least one yaml
            has_yaml = any(sub.glob("*.yaml"))
            if has_yaml:
                child = _load_node(sub)
                node.children[sub.name] = child

    return node


def _save_node(node: SkillNode, directory: Path) -> None:
    """Recursively save a tree node to *directory*."""
    directory.mkdir(parents=True, exist_ok=True)

    # Save meta
    if node.meta:
        meta_data = node.meta.model_dump(mode="json")
        meta_path = directory / _META_FILE
        meta_path.write_text(
            yaml.dump(meta_data, default_flow_style=False,
                      sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    # Save root skill
    logger.info(f"💾 Saving skill for node '{node.name}':")
    logger.info(f"   Version: {node.skill.version}")
    logger.info(f"   Prompt: {node.skill.system_prompt[:100]}...")
    skill_module.save(node.skill, directory / _ROOT_SKILL)

    # Save children (and clean up removed ones)
    existing_dirs = {
        d.name for d in directory.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    }
    current_children = set(node.children.keys())

    # Remove directories for pruned children
    for removed in existing_dirs - current_children:
        removed_path = directory / removed
        if (removed_path / _ROOT_SKILL).exists() or (removed_path / _META_FILE).exists():
            shutil.rmtree(removed_path)
            logger.debug("Removed pruned child directory: %s", removed_path)

    # Save current children
    for child_name, child_node in node.children.items():
        _save_node(child_node, directory / child_name)


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _format_tree(
    node: SkillNode,
    lines: List[str],
    prefix: str,
    is_last: bool,
) -> None:
    """Build the tree visualization lines."""
    connector = "└── " if is_last else "├── "
    version = node.skill.version
    leaf_tag = "" if node.children else " 🍂"
    desc = f" — {node.meta.description}" if node.meta and node.meta.description else ""
    lines.append(f"{prefix}{connector}{node.name} ({version}){leaf_tag}{desc}")

    child_prefix = prefix + ("    " if is_last else "│   ")
    children = list(node.children.values())
    for i, child in enumerate(children):
        _format_tree(child, lines, child_prefix, i == len(children) - 1)
