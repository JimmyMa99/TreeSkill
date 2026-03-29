"""Primary TreeSkill pipeline entrypoint.

This entrypoint makes the current recommended workflow explicit:
- Kode performs the forward pass
- ASO evolves programs / skills
- SealQA lifecycle demo is the default reproducible example

Legacy interactive APO/chat flows remain available via ``python -m treeskill.main``
or the ``legacy-chat`` subcommand here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEMO_LIFECYCLE = ROOT / "demo" / "demo_sealqa_tree_lifecycle.py"
DEMO_ASO = ROOT / "demo" / "demo_sealqa_aso.py"


def _run_script(path: Path, passthrough: list[str]) -> int:
    command = [sys.executable, str(path), *passthrough]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="treeskill",
        description=(
            "TreeSkill primary pipeline: Kode forward execution + ASO skill evolution."
        ),
    )
    parser.add_argument(
        "pipeline",
        nargs="?",
        default="sealqa-lifecycle",
        choices=["sealqa-lifecycle", "sealqa-aso", "legacy-chat"],
        help=(
            "Pipeline to run. Default is the SealQA lifecycle demo that shows "
            "root -> generate -> evolve -> prune -> merge."
        ),
    )
    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected pipeline.",
    )
    args = parser.parse_args(argv)

    passthrough = list(args.pipeline_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.pipeline == "sealqa-lifecycle":
        return _run_script(DEMO_LIFECYCLE, passthrough)
    if args.pipeline == "sealqa-aso":
        return _run_script(DEMO_ASO, passthrough)

    from treeskill.main import main as legacy_main

    legacy_main(passthrough)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
