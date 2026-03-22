"""Resume State — track optimization progress for crash recovery.

When optimizing a skill tree, the process can be interrupted at any point
(timeout, crash, Ctrl+C).  This module persists a lightweight state file
(``.evo_resume.json``) inside the skill directory so that the next run can
pick up where it left off.

State is saved **after each node** completes optimization, so the worst-case
data loss is a single node's work.

Lifecycle::

    1. Before optimization starts → ``create()``
    2. After each node finishes   → ``mark_node_done()`` (auto-saves)
    3. After full optimization    → ``clear()``
    4. On next startup            → ``load()`` to detect incomplete run

Resume file lives at ``<skill_dir>/.evo_resume.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

RESUME_FILE = ".evo_resume.json"


class ResumeState:
    """Tracks which nodes have been optimized in the current run.

    Parameters
    ----------
    skill_dir : Path
        The skill tree root directory.
    round_num : int
        Current optimization round (1-based).
    total_rounds : int
        Total planned rounds.
    completed_nodes : set[str]
        Dot-paths of nodes whose optimization has finished.
    split_nodes : dict[str, list[str]]
        Dot-paths of nodes that were split, mapping to child names.
    started_at : str
        ISO timestamp when this optimization run began.
    metadata : dict
        Arbitrary extra info (e.g. trace count, config snapshot).
    """

    def __init__(
        self,
        skill_dir: Union[str, Path],
        round_num: int = 1,
        total_rounds: int = 1,
        completed_nodes: Optional[Set[str]] = None,
        split_nodes: Optional[Dict[str, List[str]]] = None,
        started_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.skill_dir = Path(skill_dir)
        self.round_num = round_num
        self.total_rounds = total_rounds
        self.completed_nodes: Set[str] = completed_nodes or set()
        self.split_nodes: Dict[str, List[str]] = split_nodes or {}
        self.started_at = started_at or datetime.now(timezone.utc).isoformat()
        self.metadata: Dict[str, Any] = metadata or {}

    @property
    def resume_path(self) -> Path:
        return self.skill_dir / RESUME_FILE

    def is_node_done(self, dotpath: str) -> bool:
        """Check whether a node has already been optimized."""
        return dotpath in self.completed_nodes

    def mark_node_done(self, dotpath: str) -> None:
        """Record that a node's optimization is complete and persist to disk."""
        self.completed_nodes.add(dotpath)
        self.save()

    def mark_node_split(self, dotpath: str, child_names: List[str]) -> None:
        """Record that a node was split into children."""
        self.split_nodes[dotpath] = child_names
        self.save()

    def advance_round(self) -> None:
        """Move to the next round, resetting per-round tracking."""
        self.round_num += 1
        self.completed_nodes.clear()
        self.split_nodes.clear()
        self.save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write resume state to disk (atomic via temp file)."""
        data = {
            "round_num": self.round_num,
            "total_rounds": self.total_rounds,
            "completed_nodes": sorted(self.completed_nodes),
            "split_nodes": self.split_nodes,
            "started_at": self.started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
        }
        tmp = self.resume_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.resume_path)  # atomic on POSIX
        logger.debug("Resume state saved → %s (%d nodes done)", self.resume_path, len(self.completed_nodes))

    def clear(self) -> None:
        """Remove the resume file (optimization completed successfully)."""
        if self.resume_path.exists():
            self.resume_path.unlink()
            logger.info("Resume state cleared — optimization completed.")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        skill_dir: Union[str, Path],
        total_rounds: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ResumeState":
        """Create a fresh resume state for a new optimization run."""
        state = cls(
            skill_dir=skill_dir,
            round_num=1,
            total_rounds=total_rounds,
            metadata=metadata or {},
        )
        state.save()
        return state

    @classmethod
    def load(cls, skill_dir: Union[str, Path]) -> Optional["ResumeState"]:
        """Load an existing resume state, or return None if none exists."""
        skill_dir = Path(skill_dir)
        resume_path = skill_dir / RESUME_FILE
        if not resume_path.is_file():
            return None

        try:
            data = json.loads(resume_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt resume file %s: %s — ignoring.", resume_path, e)
            return None

        return cls(
            skill_dir=skill_dir,
            round_num=data.get("round_num", 1),
            total_rounds=data.get("total_rounds", 1),
            completed_nodes=set(data.get("completed_nodes", [])),
            split_nodes=data.get("split_nodes", {}),
            started_at=data.get("started_at"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def exists(cls, skill_dir: Union[str, Path]) -> bool:
        """Check if a resume file exists without loading it."""
        return (Path(skill_dir) / RESUME_FILE).is_file()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary for CLI display."""
        lines = [
            f"Unfinished optimization run (started: {self.started_at})",
            f"  Round: {self.round_num}/{self.total_rounds}",
            f"  Completed nodes: {len(self.completed_nodes)}",
        ]
        if self.completed_nodes:
            for node in sorted(self.completed_nodes):
                lines.append(f"    ✓ {node}")
        if self.split_nodes:
            lines.append(f"  Split nodes: {len(self.split_nodes)}")
        return "\n".join(lines)
