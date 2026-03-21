"""Evo-Framework — Entry Point.

Usage::

    python -m evoskill.main --skill <name-or-path>
    python -m evoskill.main --skill <name-or-path> --optimize
    python -m evoskill.main --ckpt <checkpoint-path>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from evoskill import skill as skill_module
from evoskill.checkpoint import CheckpointManager
from evoskill.cli import ChatCLI
from evoskill.config import GlobalConfig
from evoskill.llm import LLMClient
from evoskill.optimizer import APOEngine
from evoskill.resume import ResumeState
from evoskill.skill_tree import SkillTree
from evoskill.storage import TraceStorage

console = Console()


def _resolve_skill_path(name_or_path: str, config: GlobalConfig) -> Path:
    """Resolve a skill name or path to a directory containing SKILL.md.

    Resolution order:
    1. Exact path (file or directory)
    2. Named directory inside ``config.storage.skill_path``
    3. Create a new default skill directory
    """
    p = Path(name_or_path)
    # Direct path
    if p.is_dir():
        return p.resolve()
    if p.is_file() and p.name.lower().endswith(".md"):
        return p.parent.resolve()

    # Try as a named skill inside the skills directory
    candidate_dir = Path(config.storage.skill_path) / name_or_path
    if candidate_dir.is_dir() and (candidate_dir / "SKILL.md").is_file():
        return candidate_dir.resolve()

    # Fall back — create a minimal default skill directory
    default_dir = Path(config.storage.skill_path) / name_or_path
    default_dir.mkdir(parents=True, exist_ok=True)
    from evoskill.schema import Skill

    default_skill = Skill(
        name=name_or_path,
        description=f"Default skill: {name_or_path}",
        system_prompt="You are a helpful assistant.",
    )
    skill_module.save(default_skill, default_dir)
    console.print(f"[dim]Created default skill → {default_dir}/SKILL.md[/dim]")
    return default_dir.resolve()


def _handle_resume(skill_path: Path) -> ResumeState | None:
    """Check for an incomplete optimization and ask the user what to do.

    Returns the ResumeState to continue, or None to start fresh.
    """
    state = ResumeState.load(skill_path)
    if state is None:
        return None

    console.print(f"\n[bold yellow]⚠ 发现未完成的优化任务[/bold yellow]")
    console.print(state.summary())
    console.print()

    try:
        from rich.prompt import Prompt
        choice = Prompt.ask(
            "继续上次的优化 (resume) 还是重新开始 (restart)?",
            choices=["resume", "restart"],
            default="resume",
        )
    except (EOFError, KeyboardInterrupt):
        choice = "resume"

    if choice == "resume":
        console.print("[green]✓ 从断点继续[/green]")
        return state
    else:
        state.clear()
        console.print("[dim]已清除旧进度，重新开始[/dim]")
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="evo",
        description="Evo-Framework: Multimodal Self-Evolving Agent",
    )
    parser.add_argument(
        "--skill",
        default="default",
        help="Skill name (resolved in skills/) or path to a skill directory containing SKILL.md.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (see demo/example/config.yaml for template).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run APO optimization on stored traces instead of starting chat.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Restore from a checkpoint path (e.g. ckpt/writing-assistant_v1.2_20260306_140000).",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="./ckpt",
        help="Directory for storing checkpoints (default: ./ckpt).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a ChatML JSONL dataset file. Used with --optimize to "
        "auto-evaluate the skill and generate traces before running APO.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    # --- Config ---
    if args.config:
        config = GlobalConfig.from_yaml(args.config)
        console.print(f"[dim]Config loaded from {args.config}[/dim]")
    else:
        config = GlobalConfig()
    if args.verbose:
        config = config.model_copy(update={"verbose": True})
        logging.basicConfig(level=logging.DEBUG, format="%(name)s  %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # --- Checkpoint restore ---
    if args.ckpt:
        ckpt_mgr = CheckpointManager(args.ckpt_dir)
        info = ckpt_mgr.load(args.ckpt)
        skill_path = info["skill_path"]
        if info["trace_path"]:
            config = config.model_copy(
                update={"storage": config.storage.model_copy(
                    update={"trace_path": info["trace_path"]}
                )}
            )
        console.print(
            f"[green]✓[/green] Restored from checkpoint: [cyan]{args.ckpt}[/cyan]"
        )
    else:
        skill_path = _resolve_skill_path(args.skill, config)

    # --- Load skill (always a directory with SKILL.md) ---
    skill_tree = SkillTree.load(skill_path)
    loaded_skill = skill_tree.root.skill
    leaf_count = skill_tree.root.leaf_count()
    if leaf_count > 1:
        console.print(
            f"[dim]Loaded skill tree from {skill_path} "
            f"({leaf_count} leaves)[/dim]"
        )
    else:
        console.print(
            f"[dim]Loaded skill from {skill_path}/SKILL.md[/dim]"
        )

    # --- Optimize mode ---
    if args.optimize:
        llm = LLMClient(config)
        engine = APOEngine(config, llm)

        if args.dataset:
            # Dataset-driven evaluation → APO pipeline
            from evoskill.dataset import DataLoader
            from evoskill.evaluator import Evaluator

            dataset = DataLoader(args.dataset)
            evaluator = Evaluator(config, llm)

            console.print(
                f"[bold]Evaluating[/bold] {len(dataset)} samples "
                f"from {args.dataset} …"
            )
            traces = evaluator.evaluate(loaded_skill, dataset)
            scored = [t for t in traces if t.feedback is not None]
            avg = sum(t.feedback.score for t in scored) / len(scored) if scored else 0
            console.print(
                f"[dim]Evaluation: {len(scored)} scored, "
                f"avg_score={avg:.2f}[/dim]"
            )
        else:
            # Legacy: load traces from storage
            storage = TraceStorage(config.storage)
            traces = storage.get_feedback_samples()

        if not traces:
            console.print("[yellow]No feedback traces found — nothing to optimize.[/yellow]")
            sys.exit(0)

        # Check for resume state
        resume = _handle_resume(skill_path)
        if resume is None:
            resume = ResumeState.create(
                skill_dir=skill_path,
                metadata={"trace_count": len(traces)},
            )

        tagged = sum(1 for t in traces if t.node_path is not None)
        console.print(
            f"[bold]Running APO[/bold] on {len(traces)} feedback traces "
            f"({tagged} node-routed, {len(traces) - tagged} global) …"
        )

        try:
            engine.evolve_tree(
                skill_tree, traces,
                resume=resume,
            )
            skill_tree.save()
            resume.clear()  # all done — remove resume file
        except KeyboardInterrupt:
            console.print(
                "\n[yellow]⚠ 优化被中断，进度已保存。"
                "下次运行 --optimize 可从断点继续。[/yellow]"
            )
            sys.exit(1)
        except Exception:
            console.print(
                "\n[red]✗ 优化出错，进度已保存。"
                "下次运行 --optimize 可从断点继续。[/red]"
            )
            raise

        new_skill = skill_tree.root.skill

        # Save checkpoint
        ckpt_mgr = CheckpointManager(args.ckpt_dir)
        ckpt_path = ckpt_mgr.save(
            skill_path,
            trace_path=Path(config.storage.trace_path),
        )

        console.print(
            f"[green]✓ Skill updated:[/green] {new_skill.name} "
            f"({loaded_skill.version} → {new_skill.version})"
        )
        console.print(f"[dim]Checkpoint saved → {ckpt_path}[/dim]")
        sys.exit(0)

    # --- Chat mode ---
    chat = ChatCLI(
        config,
        loaded_skill,
        skill_path,
        skill_tree=skill_tree,
        ckpt_dir=args.ckpt_dir,
    )
    chat.run()


if __name__ == "__main__":
    main()
