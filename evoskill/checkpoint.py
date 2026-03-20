"""Checkpoint Manager — folder-based snapshots of skill + memory state.

Each checkpoint is a self-contained directory that can be used to restore
a training session::

    ckpt/
    └── writing-assistant_v1.2_20260306_140000/
        ├── skill/          # complete skill tree (Agent Skills format)
        │   ├── SKILL.md
        │   └── social/
        │       └── SKILL.md
        └── mem/
            ├── traces.jsonl
            └── meta.json   # optimization round, config, etc.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from evoskill.schema import Skill
from evoskill import skill as skill_module

logger = logging.getLogger(__name__)

_SKILL_DIR = "skill"
_MEM_DIR = "mem"
_META_FILE = "meta.json"
_TRACES_FILE = "traces.jsonl"


class CheckpointManager:
    """Save and restore training checkpoints.

    Parameters
    ----------
    ckpt_dir : str | Path
        Root directory for all checkpoints (default: ``./ckpt``).
    """

    def __init__(self, ckpt_dir: Union[str, Path] = "./ckpt") -> None:
        self.ckpt_dir = Path(ckpt_dir)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        skill_source: Union[Skill, Path],
        trace_path: Optional[Path] = None,
        name: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Create a checkpoint snapshot.

        Parameters
        ----------
        skill_source :
            A ``Skill`` object (single skill) **or** a ``Path`` to a
            skill-tree directory.
        trace_path :
            Path to the traces JSONL file.  Copied into ``mem/``.
        name :
            Override for the checkpoint folder name.  If *None*, one is
            generated from the skill name + version + timestamp.
        extra_meta :
            Additional key-value pairs to store in ``meta.json``.

        Returns
        -------
        Path
            The checkpoint directory that was created.
        """
        # Determine checkpoint name
        if name is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            if isinstance(skill_source, Skill):
                name = f"{skill_source.name}_{skill_source.version}_{ts}"
            else:
                name = f"{skill_source.name}_{ts}"

        ckpt_path = self.ckpt_dir / name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # -- skill/ --------------------------------------------------------
        skill_dir = ckpt_path / _SKILL_DIR
        if isinstance(skill_source, Path) and skill_source.is_dir():
            # Copy the entire skill-tree directory
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
            shutil.copytree(skill_source, skill_dir)
        elif isinstance(skill_source, Path) and skill_source.is_file():
            # Single skill file
            skill_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill_source, skill_dir / skill_source.name)
        else:
            # Skill object → save as SKILL.md
            skill_module.save(skill_source, skill_dir)

        # -- mem/ ----------------------------------------------------------
        mem_dir = ckpt_path / _MEM_DIR
        mem_dir.mkdir(parents=True, exist_ok=True)

        if trace_path and trace_path.exists():
            shutil.copy2(trace_path, mem_dir / _TRACES_FILE)

        # meta.json
        meta: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_name": name,
        }
        if isinstance(skill_source, Skill):
            meta["skill_name"] = skill_source.name
            meta["skill_version"] = skill_source.version
        if extra_meta:
            meta.update(extra_meta)

        (mem_dir / _META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("Checkpoint saved → %s", ckpt_path)
        return ckpt_path

    # ------------------------------------------------------------------
    # Load / Restore
    # ------------------------------------------------------------------

    def load(self, ckpt_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a checkpoint, returning skill(s) and trace path.

        Returns
        -------
        dict with keys:
            - ``skill_path``: Path to skill dir or file inside the ckpt
            - ``trace_path``: Path to traces.jsonl (or None)
            - ``meta``: dict from meta.json
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        skill_dir = ckpt_path / _SKILL_DIR
        mem_dir = ckpt_path / _MEM_DIR

        # Determine if it's a tree or single skill
        skill_md = skill_dir / "SKILL.md"
        if skill_md.is_file():
            skill_path = skill_dir
        else:
            # Fallback: check for any SKILL.md in subdirectories
            skill_path = skill_dir

        # Traces
        trace_path = mem_dir / _TRACES_FILE
        if not trace_path.exists():
            trace_path = None

        # Meta
        meta_path = mem_dir / _META_FILE
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        return {
            "skill_path": skill_path,
            "trace_path": trace_path,
            "meta": meta,
        }

    def restore_to(
        self,
        ckpt_path: Union[str, Path],
        skill_dest: Union[str, Path],
        trace_dest: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Restore a checkpoint — copy skill and traces to working directories.

        Parameters
        ----------
        ckpt_path :
            The checkpoint directory to restore from.
        skill_dest :
            Where to copy the skill files.
        trace_dest :
            Where to copy traces.jsonl (optional).

        Returns
        -------
        dict : The loaded meta information.
        """
        info = self.load(ckpt_path)

        skill_dest = Path(skill_dest)
        src = Path(info["skill_path"])

        if src.is_dir():
            if skill_dest.exists():
                shutil.rmtree(skill_dest)
            shutil.copytree(src, skill_dest)
        else:
            skill_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, skill_dest)

        if trace_dest and info["trace_path"]:
            trace_dest = Path(trace_dest)
            trace_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(info["trace_path"], trace_dest)

        logger.info("Restored checkpoint '%s' → skill=%s", ckpt_path, skill_dest)
        return info["meta"]

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with their metadata.

        Returns a list of dicts, each with ``name``, ``path``, and ``meta``.
        Sorted by creation time (newest first).
        """
        if not self.ckpt_dir.exists():
            return []

        results: List[Dict[str, Any]] = []
        for d in sorted(self.ckpt_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / _MEM_DIR / _META_FILE
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            results.append({
                "name": d.name,
                "path": d,
                "meta": meta,
            })

        return results
