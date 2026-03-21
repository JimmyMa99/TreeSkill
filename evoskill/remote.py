"""Remote Skill Loading -- fetch skills from GitHub repositories.

Supports loading skills directly from GitHub without manual cloning::

    python -m evoskill.main --skill github://user/repo/path/to/skill
    python -m evoskill.main --skill github://user/repo@branch/path/to/skill

Skills are cached locally in ``~/.evoskill/cache/`` to avoid repeated
downloads.  Use ``force=True`` or ``/fetch --force`` to refresh.

Implementation uses ``git clone --depth 1 --sparse`` to fetch only the
target directory, keeping downloads minimal.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".evoskill" / "cache"


def parse_github_uri(uri: str) -> Tuple[str, Optional[str], str]:
    """Parse a ``github://`` URI into (repo, branch, path).

    Formats::

        github://user/repo/path/to/skill
        github://user/repo@branch/path/to/skill

    Returns
    -------
    (repo, branch, path) : tuple
        repo: ``user/repo``
        branch: branch/tag name or None (default branch)
        path: path within the repo (may be empty string for root)
    """
    raw = uri
    if raw.startswith("github://"):
        raw = raw[len("github://"):]

    # Split user/repo from the rest
    parts = raw.split("/", 2)
    if len(parts) < 2:
        raise ValueError(
            f"Invalid github URI: {uri}\n"
            f"Expected: github://user/repo[/path] or github://user/repo@branch[/path]"
        )

    user = parts[0]
    repo_part = parts[1]
    rest = parts[2] if len(parts) > 2 else ""

    # Check for @branch in repo part
    branch = None
    if "@" in repo_part:
        repo_name, branch = repo_part.split("@", 1)
    else:
        repo_name = repo_part

    repo = f"{user}/{repo_name}"
    return repo, branch, rest


def _cache_key(repo: str, branch: Optional[str], path: str) -> str:
    """Generate a deterministic cache directory name."""
    key = f"{repo}:{branch or 'HEAD'}:{path}"
    short_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
    safe_name = repo.replace("/", "_")
    return f"{safe_name}_{short_hash}"


def fetch_skill(
    uri: str,
    *,
    force: bool = False,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Fetch a skill from a GitHub repository.

    Parameters
    ----------
    uri : str
        GitHub URI (``github://user/repo/path/to/skill``).
    force : bool
        If True, re-fetch even if cached.
    cache_dir : Path | None
        Override the default cache directory.

    Returns
    -------
    Path
        Local path to the fetched skill directory (containing SKILL.md).
    """
    repo, branch, path = parse_github_uri(uri)
    cache_root = cache_dir or _CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_name = _cache_key(repo, branch, path)
    dest = cache_root / cache_name

    # Check cache
    skill_md = dest / "SKILL.md" if not path else dest / path / "SKILL.md"
    if skill_md.is_file() and not force:
        skill_dir = skill_md.parent
        logger.info("Using cached skill: %s", skill_dir)
        return skill_dir

    # Clean old cache
    if dest.exists():
        shutil.rmtree(dest)

    # Clone with sparse checkout
    clone_url = f"https://github.com/{repo}.git"
    logger.info("Fetching skill from %s ...", clone_url)

    clone_cmd = [
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--sparse",
    ]
    if branch:
        clone_cmd.extend(["--branch", branch])
    clone_cmd.extend([clone_url, str(dest)])

    try:
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to clone {clone_url}: {e.stderr.strip()}"
        ) from e

    # Set sparse checkout to only fetch the target path
    if path:
        try:
            subprocess.run(
                ["git", "sparse-checkout", "set", path],
                cwd=str(dest),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to set sparse checkout for '{path}': {e.stderr.strip()}"
            ) from e

    # Verify SKILL.md exists
    skill_dir = dest / path if path else dest
    if not (skill_dir / "SKILL.md").is_file():
        # Try to find SKILL.md in subdirectories
        found = list(skill_dir.rglob("SKILL.md"))
        if found:
            skill_dir = found[0].parent
        else:
            shutil.rmtree(dest)
            raise FileNotFoundError(
                f"No SKILL.md found in {uri}\n"
                f"Make sure the path points to a valid Agent Skills directory."
            )

    logger.info("Skill fetched: %s", skill_dir)
    return skill_dir


def list_cached_skills(cache_dir: Optional[Path] = None) -> list[dict]:
    """List all cached skills.

    Returns a list of dicts with keys: ``name``, ``path``, ``has_skill_md``.
    """
    cache_root = cache_dir or _CACHE_DIR
    if not cache_root.exists():
        return []

    results = []
    for d in sorted(cache_root.iterdir()):
        if not d.is_dir():
            continue
        skill_files = list(d.rglob("SKILL.md"))
        results.append({
            "name": d.name,
            "path": str(d),
            "has_skill_md": len(skill_files) > 0,
        })
    return results


def clear_cache(cache_dir: Optional[Path] = None) -> int:
    """Remove all cached skills. Returns the number of entries removed."""
    cache_root = cache_dir or _CACHE_DIR
    if not cache_root.exists():
        return 0
    entries = list(cache_root.iterdir())
    count = len(entries)
    shutil.rmtree(cache_root)
    return count
