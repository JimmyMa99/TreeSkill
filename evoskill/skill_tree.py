"""Skill Tree — hierarchical skill management following Agent Skills standard.

A "skill tree" is a directory on disk where each sub-directory represents
a child skill.  Each directory contains a ``SKILL.md`` (and optional
``config.yaml``), following the Agent Skills specification::

    my-skills/
    ├── SKILL.md            # root skill
    ├── config.yaml         # optional root config
    ├── social/
    │   ├── SKILL.md        # social sub-skill
    │   ├── moments/
    │   │   └── SKILL.md    # WeChat Moments specialty
    │   └── weibo/
    │       └── SKILL.md
    └── business/
        ├── SKILL.md
        ├── email/
        │   └── SKILL.md
        └── product/
            └── SKILL.md

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

from evoskill.schema import Skill
from evoskill.skill import SKILL_FILE, load as _load_skill, save as _save_skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SkillNode — in-memory representation of one tree node
# ---------------------------------------------------------------------------

@dataclass
class SkillNode:
    """One node in the skill tree.

    Attributes
    ----------
    name : str
        Human-readable name (from SKILL.md frontmatter or directory name).
    skill : Skill
        The skill loaded from this node's SKILL.md.
    children : dict[str, SkillNode]
        Ordered mapping of child-name → child-node.
    path : Path | None
        Disk path this node was loaded from (``None`` for in-memory nodes).
    age : int
        Number of optimization rounds this node has survived.
    usage_count : int
        Number of times this node has been used/routed to.
    collapsed : bool
        If True, this node is "folded" (hidden from routing) but not deleted.
    """

    name: str
    skill: Skill
    children: Dict[str, "SkillNode"] = field(default_factory=dict)
    path: Optional[Path] = None
    age: int = 0
    usage_count: int = 0
    collapsed: bool = False

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

        *path* must be a directory containing a SKILL.md at the root.
        """
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Skill tree path must be a directory: {path}"
            )

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

        if description and not skill.description:
            skill = skill.model_copy(update={"description": description})
        child = SkillNode(name=child_name, skill=skill)
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
                    "description": spec.get("description", ""),
                    "system_prompt": spec["system_prompt"],
                    "version": "v1.0",
                }
            )
            child = SkillNode(name=spec["name"], skill=child_skill)
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

        nodes = [self.get(p) for p in node_paths]
        parent_path = ".".join(node_paths[0].split(".")[:-1])
        parent = self.get(parent_path) if parent_path else self.root

        for np in node_paths:
            child_name = np.split(".")[-1]
            parent.children.pop(child_name, None)

        merged_skill = nodes[0].skill.model_copy(
            update={
                "name": merged_name,
                "description": f"Merged from: {[n.name for n in nodes]}",
                "system_prompt": merged_prompt,
                "version": "v1.0",
            }
        )
        merged_node = SkillNode(name=merged_name, skill=merged_skill)
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

    # ------------------------------------------------------------------
    # Graft: attach another skill / tree onto this tree
    # ------------------------------------------------------------------

    def graft(
        self,
        target_path: str,
        source: Union["SkillTree", SkillNode, Skill],
        *,
        name: Optional[str] = None,
    ) -> SkillNode:
        """Attach another Skill / SkillNode / SkillTree at a target location in this tree.

        Parameters
        ----------
        target_path : str
            Path of the parent node. An empty string refers to the root node.
        source : SkillTree | SkillNode | Skill
            Source to attach:
            - ``SkillTree`` — attach the full subtree (its root becomes a child)
            - ``SkillNode`` — attach the node, including its descendants
            - ``Skill``     — create a new leaf node
        name : str | None
            Name of the attached node. Defaults to ``source.name``.

        Returns
        -------
        SkillNode
            The newly attached node.

        Raises
        ------
        ValueError
            If the target location already contains a child with the same name.

        Examples
        --------
        Attach an independent skill::

            external = load_skill("./another-skill")
            tree.graft("social", external)
            # → social now has one more child node

        Attach an entire subtree::

            other_tree = SkillTree.load("./other-skills")
            tree.graft("business", other_tree)
            # → business now includes all nodes from other_tree

        Attach a branch from another tree::

            node = other_tree.get("email")
            tree.graft("business", node, name="imported-email")
        """
        parent = self.get(target_path) if target_path else self.root

        # Normalize source → SkillNode
        if isinstance(source, SkillTree):
            graft_node = source.root
        elif isinstance(source, Skill):
            graft_node = SkillNode(name=source.name, skill=source)
        else:
            graft_node = source

        graft_name = name or graft_node.name

        if graft_name in parent.children:
            raise ValueError(
                f"'{graft_name}' already exists under '{parent.name}'. "
                f"Use name= to specify a different name, or prune the existing node first."
            )

        # Rebuild the node tree to avoid shared references across trees.
        copied = _deep_copy_node(graft_node, new_name=graft_name)
        parent.children[graft_name] = copied

        child_count = copied.leaf_count()
        logger.info(
            "Grafted '%s' under '%s' (%d leaf skills)",
            graft_name,
            parent.name,
            child_count,
        )
        return copied

    # ------------------------------------------------------------------
    # Collect tools: resolve all tools for a given skill node
    # ------------------------------------------------------------------

    def collect_tools(self, dotpath: str = "") -> List:
        """Collect the full tool list for a node, including inherited tools.

        Tool inheritance rules:
        - Child nodes inherit all tool declarations from their ancestors
        - Child-level tool declarations take precedence over inherited ones
        - Functions in script.py also count as a tool source

        Parameters
        ----------
        dotpath : str
            Dot path of the node. An empty string refers to the root node.

        Returns
        -------
        List[ToolRef]
            Merged tool declarations, deduplicated with children overriding parents.
        """
        from evoskill.schema import ToolRef

        # Build the chain from the root to the target node.
        parts = [p for p in dotpath.split(".") if p]
        chain: List[SkillNode] = [self.root]
        node = self.root
        for p in parts:
            node = node.children[p]
            chain.append(node)

        # Merge tools from root to leaf so later nodes override earlier ones.
        merged: Dict[str, "ToolRef"] = {}
        for n in chain:
            for tool_ref in n.skill.tools:
                merged[tool_ref.name] = tool_ref

        return list(merged.values())


# ---------------------------------------------------------------------------
# Internal helpers — disk I/O
# ---------------------------------------------------------------------------

def _deep_copy_node(
    node: SkillNode,
    new_name: Optional[str] = None,
) -> SkillNode:
    """Deep-copy a SkillNode, including all descendants, to avoid shared references."""
    copied = SkillNode(
        name=new_name or node.name,
        skill=node.skill.model_copy(
            update={"name": new_name} if new_name else {},
        ),
        path=None,  # Attached nodes need to be saved again; save() determines the path.
        age=node.age,
        usage_count=0,  # Reset usage count after attaching.
        collapsed=node.collapsed,
    )
    for child_name, child_node in node.children.items():
        copied.children[child_name] = _deep_copy_node(child_node)
    return copied


def _load_node(directory: Path) -> SkillNode:
    """Recursively load a single tree node from *directory*."""
    sk = _load_skill(directory)
    node = SkillNode(name=sk.name, skill=sk, path=directory)

    # Recurse into subdirectories that contain a SKILL.md
    for sub in sorted(directory.iterdir()):
        if sub.is_dir() and not sub.name.startswith((".", "_")):
            skill_md = sub / SKILL_FILE
            if skill_md.is_file():
                child = _load_node(sub)
                node.children[sub.name] = child

    return node


def _save_node(node: SkillNode, directory: Path) -> None:
    """Recursively save a tree node to *directory*."""
    directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving skill for node '{node.name}':")
    logger.info(f"   Version: {node.skill.version}")
    logger.info(f"   Prompt: {node.skill.system_prompt[:100]}...")
    _save_skill(node.skill, directory)

    # Save children (and clean up removed ones)
    existing_dirs = {
        d.name for d in directory.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    }
    current_children = set(node.children.keys())

    # Remove directories for pruned children
    for removed in existing_dirs - current_children:
        removed_path = directory / removed
        if (removed_path / SKILL_FILE).exists():
            shutil.rmtree(removed_path)
            logger.debug("Removed pruned child directory: %s", removed_path)

    # Save current children
    for child_name, child_node in node.children.items():
        _save_node(child_node, directory / child_name)


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def resolve_skill_tools(
    skill: Skill,
    skill_dir: Optional[Path] = None,
) -> Dict[str, "BaseTool"]:
    """Resolve Skill tool declarations + script.py into executable BaseTool instances.

    Tool sources, ordered from lower to higher precedence:
    1. ToolRef declarations in ``skill.tools`` (http / mcp)
    2. Public functions from ``script.py`` (when ``skill_dir`` is provided)

    Functions in script.py override ToolRef declarations with the same name.

        Parameters
        ----------
        skill : Skill
            Skill object.
        skill_dir : Path | None
            Skill directory path, used to load script.py.

        Returns
        -------
        Dict[str, BaseTool]
            Mapping of tool names to executable tools.
    """
    from evoskill.tools import BaseTool, HTTPTool, MCPTool

    tools: Dict[str, BaseTool] = {}

    # 1. Build tools from ToolRef declarations.
    for ref in skill.tools:
        if ref.type == "http" and ref.endpoint:
            tools[ref.name] = HTTPTool(
                _name=ref.name,
                _description=ref.description,
                endpoint=ref.endpoint,
                method=ref.method,
                headers=ref.headers,
            )
        elif ref.type == "mcp" and ref.mcp_server:
            tools[ref.name] = MCPTool(
                _name=ref.name,
                _description=ref.description,
                mcp_server=ref.mcp_server,
                tool_name=ref.tool_name or ref.name,
                auth_token=ref.auth_token,
            )

    # 2. Load tools from script.py, overriding name collisions.
    if skill_dir:
        try:
            from evoskill.script import load_script_as_tools
            script_tools = load_script_as_tools(skill_dir)
            tools.update(script_tools)
        except (ValueError, Exception) as exc:
            logger.warning("Failed to load tools from script.py: %s", exc)

    return tools


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
    desc = f" — {node.skill.description}" if node.skill.description else ""
    lines.append(f"{prefix}{connector}{node.name} ({version}){leaf_tag}{desc}")

    child_prefix = prefix + ("    " if is_last else "│   ")
    children = list(node.children.values())
    for i, child in enumerate(children):
        _format_tree(child, lines, child_prefix, i == len(children) - 1)
