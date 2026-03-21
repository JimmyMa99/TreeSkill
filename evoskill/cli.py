"""CLI Interface — Rich-based terminal chat with multimodal & feedback support.

Commands
--------
/image <path>   — attach a local image (base64-encoded) to the next message.
/bad <reason>   — mark the *previous* interaction with a low score + critique.
/rewrite <txt>  — mark the *previous* interaction with a correction.
/target <text>  — set a one-line optimization target (e.g. "更像人").
/save           — force-save the current skill to disk.
/optimize       — trigger an APO optimization cycle.
/help           — show available commands and brief descriptions.
/tree           — show the skill tree hierarchy.
/select <path>  — switch active skill (e.g. /select social.moments).
/split          — analyze and split current skill into sub-skills.
/tools          — list built-in tools available to the model.
/ckpt           — list available checkpoints.
/restore <name> — restore from a checkpoint.
/quit           — exit the chat loop.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

from evoskill.builtin_tools import build_builtin_tools
from evoskill import skill as skill_module
from evoskill.checkpoint import CheckpointManager
from evoskill.config import GlobalConfig
from evoskill.llm import LLMClient
from evoskill.optimizer import APOEngine
from evoskill.resume import ResumeState
from evoskill.schema import (
    ContentPart,
    Feedback,
    ImageContent,
    ImageURL,
    Message,
    Skill,
    TextContent,
    Trace,
)
from evoskill.skill_tree import SkillTree
from evoskill.storage import TraceStorage

_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "bold magenta",
        "success": "bold green",
        "error": "bold red",
    }
)

_COMMAND_SPECS = [
    ("/", "", "显示所有命令和简要说明"),
    ("/help", "", "显示所有命令和简要说明"),
    ("/image", "<path>", "给下一条消息附加本地图片"),
    ("/bad", "<reason>", "给上一轮回答打差评并记录原因"),
    ("/rewrite", "<ideal response>", "给上一轮回答提供理想改写（同时收集 DPO 偏好数据）"),
    ("/export-dpo", "<output.jsonl>", "导出 DPO 偏好数据（用于微调）"),
    ("/target", "<text>", "设置长期优化方向"),
    ("/save", "", "立即保存当前 skill"),
    ("/optimize", "", "基于已记录反馈优化当前 skill"),
    ("/tools", "", "查看模型可用的内置工具"),
    ("/tree", "", "查看当前 skill 树"),
    ("/select", "<path>", "切换到 skill 树中的某个节点"),
    ("/split", "", "分析当前 skill 是否适合拆分"),
    ("/ckpt", "", "查看可用 checkpoint"),
    ("/restore", "<name>", "从 checkpoint 恢复"),
    ("/quit", "", "退出 CLI"),
]


class ChatCLI:
    """Interactive terminal chat powered by Evo-Framework.

    Parameters
    ----------
    config : GlobalConfig
    skill_obj : Skill
        The loaded skill to converse with.
    skill_path : str | Path
        Where to persist skill updates.
    """

    def __init__(
        self,
        config: GlobalConfig,
        skill_obj: Skill,
        skill_path: str | Path,
        skill_tree: Optional[SkillTree] = None,
        ckpt_dir: str | Path = "./ckpt",
    ) -> None:
        self._config = config
        self._skill = skill_obj
        self._skill_path = Path(skill_path)
        self._skill_tree = skill_tree

        self._console = Console(theme=_THEME)
        self._llm = LLMClient(config)
        self._storage = TraceStorage(config.storage)
        self._optimizer = APOEngine(config, self._llm)
        self._ckpt = CheckpointManager(ckpt_dir)
        self._builtin_tools = build_builtin_tools()

        self._history: List[Message] = []
        self._last_trace: Optional[Trace] = None
        self._pending_images: List[ContentPart] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive chat loop."""
        # Count existing DPO pairs for display
        dpo_count = len(self._storage.get_dpo_pairs())
        dpo_tip = (
            f"\n[dim]DPO 偏好数据: {dpo_count} 条 | "
            f"用 /rewrite 提供理想回复可积累 DPO 训练数据, "
            f"/export-dpo 导出[/dim]"
        )

        self._console.print(
            Panel(
                f"[bold]EvoSkill Chat[/bold]\n"
                f"Skill: [cyan]{self._skill.name}[/cyan] "
                f"({self._skill.version})\n"
                f"Model: [cyan]{self._config.llm.model}[/cyan]\n"
                f"Built-in tools: [cyan]{', '.join(sorted(self._builtin_tools))}[/cyan]\n\n"
                f"[dim]输入 / 或 /help 查看命令说明[/dim]"
                + dpo_tip,
                title="🧬 EvoSkill",
                border_style="bright_blue",
            )
        )

        while True:
            try:
                raw = Prompt.ask("\n[bold green]You[/bold green] [dim](/ 查看命令)[/dim]")
            except (EOFError, KeyboardInterrupt):
                self._console.print("\n[info]Goodbye![/info]")
                break

            raw = raw.strip()
            if not raw:
                continue

            # --- command dispatch ---
            if raw.startswith("/"):
                if self._handle_command(raw):
                    continue  # command handled; no generation needed
                else:
                    continue  # unknown command; message already printed

            # --- build user message ---
            user_msg = self._build_user_message(raw)
            self._history.append(user_msg)

            # --- generate ---
            full_messages = skill_module.compile_messages(
                self._skill, self._history
            )
            # Append tool guidance to the existing system message
            # (some APIs reject multiple system messages)
            if full_messages and full_messages[0].role == "system":
                combined = full_messages[0].content
                if isinstance(combined, str):
                    combined += "\n\n" + self._tool_guidance_text()
                full_messages[0] = Message(role="system", content=combined)
            else:
                full_messages.insert(0, Message(role="system", content=self._tool_guidance_text()))
            with self._console.status("[dim]Thinking…[/dim]"):
                response = self._llm.generate(
                    full_messages,
                    tools=self._builtin_tools,
                    on_tool_event=self._on_tool_event,
                )
            self._history.append(response)

            # --- trace ---
            trace = Trace(inputs=list(self._history[:-1]), prediction=response)
            self._storage.append(trace)
            self._last_trace = trace

            # --- render ---
            self._console.print()
            self._console.print(
                Panel(
                    Markdown(
                        response.content
                        if isinstance(response.content, str)
                        else "[multimodal response]"
                    ),
                    title="🤖 Assistant",
                    border_style="bright_cyan",
                )
            )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_command(self, raw: str) -> bool:
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in {"/", "/help"}:
            return self._cmd_help(arg)
        if cmd == "/quit":
            raise KeyboardInterrupt

        if cmd == "/image":
            return self._cmd_image(arg)
        if cmd == "/bad":
            return self._cmd_bad(arg)
        if cmd == "/rewrite":
            return self._cmd_rewrite(arg)
        if cmd == "/export-dpo":
            return self._cmd_export_dpo(arg)
        if cmd == "/save":
            return self._cmd_save()
        if cmd == "/optimize":
            return self._cmd_optimize()
        if cmd == "/tools":
            return self._cmd_tools()
        if cmd == "/target":
            return self._cmd_target(arg)
        if cmd == "/tree":
            return self._cmd_tree()
        if cmd == "/select":
            return self._cmd_select(arg)
        if cmd == "/split":
            return self._cmd_split()
        if cmd == "/ckpt":
            return self._cmd_ckpt()
        if cmd == "/restore":
            return self._cmd_restore(arg)

        self._console.print(f"[error]Unknown command:[/error] {cmd}")
        self._show_command_help(prefix=cmd)
        return True  # consumed

    def _cmd_image(self, path_str: str) -> bool:
        if not path_str:
            self._console.print("[error]Usage: /image <path>[/error]")
            return True
        p = Path(path_str).expanduser().resolve()
        if not p.is_file():
            self._console.print(f"[error]File not found:[/error] {p}")
            return True
        data_url = _file_to_data_url(p)
        self._pending_images.append(
            ImageContent(image_url=ImageURL(url=data_url))
        )
        self._console.print(
            f"[success]Image attached:[/success] {p.name}  "
            f"({len(self._pending_images)} pending)"
        )
        return True

    def _cmd_bad(self, reason: str) -> bool:
        if self._last_trace is None:
            self._console.print("[warning]No previous interaction to mark.[/warning]")
            return True
        fb = Feedback(score=0.1, critique=reason or "Bad response")
        self._last_trace.feedback = fb
        self._storage.append(self._last_trace)
        self._console.print("[success]Feedback recorded (bad).[/success]")
        self._console.print(
            "[dim]Tip: 用 /rewrite <理想回复> 可同时积累 DPO 偏好数据[/dim]"
        )
        return True

    def _cmd_rewrite(self, text: str) -> bool:
        if self._last_trace is None:
            self._console.print("[warning]No previous interaction to rewrite.[/warning]")
            return True
        if not text:
            self._console.print("[error]Usage: /rewrite <ideal response>[/error]")
            return True
        fb = Feedback(score=0.1, critique="Rewrite provided", correction=text)
        self._last_trace.feedback = fb
        self._storage.append(self._last_trace)
        self._console.print("[success]Feedback recorded (rewrite).[/success]")
        return True

    def _cmd_export_dpo(self, path_str: str) -> bool:
        output = path_str.strip() or "dpo_pairs.jsonl"
        count = self._storage.export_dpo(output)
        if count == 0:
            self._console.print(
                "[warning]没有可导出的 DPO 数据。[/warning]\n"
                "[dim]使用 /rewrite <理想回复> 对模型回答进行改写，即可积累偏好对。[/dim]"
            )
        else:
            self._console.print(
                f"[success]已导出 {count} 条 DPO 偏好对 →[/success] {output}\n"
                f"[dim]格式: {{prompt, chosen, rejected, score, critique}}，"
                f"可直接用于 DPO/RLHF 微调。[/dim]"
            )
        return True

    def _cmd_save(self) -> bool:
        if self._skill_tree:
            self._skill_tree.save()
        else:
            skill_module.save(self._skill, self._skill_path)
        self._console.print(
            f"[success]Skill saved →[/success] {self._skill_path}/SKILL.md"
        )
        return True

    def _cmd_help(self, text: str) -> bool:
        self._show_command_help(prefix=text.strip() if text else None)
        return True

    def _cmd_optimize(self) -> bool:
        traces = self._storage.get_feedback_samples()
        if not traces:
            self._console.print("[warning]No feedback samples to optimize on.[/warning]")
            return True

        # Check for resume state
        resume = ResumeState.load(self._skill_path)
        if resume:
            self._console.print(f"[yellow]⚠ 发现未完成的优化[/yellow]")
            self._console.print(resume.summary())
            choice = Prompt.ask("继续 (resume) 还是重新开始 (restart)?",
                                choices=["resume", "restart"], default="resume")
            if choice == "restart":
                resume.clear()
                resume = None

        if resume is None:
            resume = ResumeState.create(
                skill_dir=self._skill_path,
                metadata={"trace_count": len(traces)},
            )

        def _on_node_done(dotpath: str, node) -> None:
            self._console.print(
                f"  [success]✓[/success] {dotpath} → {node.skill.version}"
            )

        try:
            if self._skill_tree:
                self._optimizer.evolve_tree(
                    self._skill_tree, traces,
                    resume=resume,
                    on_node_done=_on_node_done,
                )
                self._skill = self._skill_tree.root.skill
                self._skill_tree.save()
            else:
                self._skill = self._optimizer.optimize(self._skill, traces)
                skill_module.save(self._skill, self._skill_path)
            resume.clear()  # success — remove resume file
        except KeyboardInterrupt:
            self._console.print(
                "\n[warning]⚠ 优化被中断，进度已保存。再次 /optimize 可从断点继续。[/warning]"
            )
            return True
        except Exception as e:
            self._console.print(
                f"\n[error]✗ 优化出错: {e}[/error]\n"
                f"[dim]进度已保存，再次 /optimize 可从断点继续。[/dim]"
            )
            return True

        # Auto-save checkpoint
        self._ckpt.save(
            self._skill_path,
            trace_path=Path(self._config.storage.trace_path),
        )
        self._console.print(
            f"[success]Skill optimized →[/success] {self._skill.name} "
            f"({self._skill.version}) [dim](checkpoint saved)[/dim]"
        )
        return True

    def _cmd_target(self, text: str) -> bool:
        if not text:
            if self._skill.target:
                self._console.print(
                    f"[info]当前优化方向:[/info] {self._skill.target}"
                )
            else:
                self._console.print("[warning]尚未设置优化方向。用法: /target <方向>[/warning]")
            return True
        self._skill = self._skill.model_copy(update={"target": text})
        if self._skill_tree:
            self._skill_tree.root.skill = self._skill
            self._skill_tree.save()
        else:
            skill_module.save(self._skill, self._skill_path)
        self._console.print(
            f"[success]优化方向已设置:[/success] {text}"
        )
        return True

    def _cmd_tools(self) -> bool:
        lines = []
        for name in sorted(self._builtin_tools):
            tool = self._builtin_tools[name]
            lines.append(f"- {name}: {tool.description}")
        self._console.print(
            Panel("\n".join(lines), title="🛠️ Built-in Tools", border_style="yellow")
        )
        return True

    # ------------------------------------------------------------------
    # Tree & Checkpoint commands
    # ------------------------------------------------------------------

    def _cmd_tree(self) -> bool:
        if not self._skill_tree:
            self._console.print("[warning]当前未加载 Skill 树。请用目录路径启动。[/warning]")
            return True
        tree_str = self._skill_tree.list_tree()
        self._console.print(Panel(tree_str, title="🌳 Skill Tree", border_style="green"))
        return True

    def _cmd_select(self, dotpath: str) -> bool:
        if not self._skill_tree:
            self._console.print("[warning]当前未加载 Skill 树。[/warning]")
            return True
        if not dotpath:
            self._console.print("[error]Usage: /select <dotpath> (e.g. social.moments)[/error]")
            return True
        try:
            node = self._skill_tree.get(dotpath)
            self._skill = node.skill
            self._console.print(
                f"[success]已切换到:[/success] {node.name} ({node.skill.version})"
            )
        except KeyError as e:
            self._console.print(f"[error]{e}[/error]")
        return True

    def _cmd_split(self) -> bool:
        traces = self._storage.get_feedback_samples()
        if len(traces) < 2:
            self._console.print("[warning]至少需要 2 条反馈才能分析拆分。[/warning]")
            return True
        with self._console.status("[dim]分析是否需要拆分…[/dim]"):
            specs = self._optimizer.analyze_split_need(self._skill, traces)
        if not specs:
            self._console.print("[info]当前 Skill 不需要拆分。[/info]")
            return True
        self._console.print(f"[success]建议拆分为 {len(specs)} 个子技能:[/success]")
        for s in specs:
            self._console.print(f"  • {s['name']}: {s.get('description', '')}")
        confirm = Prompt.ask("确认拆分?", choices=["y", "n"], default="y")
        if confirm == "y":
            if not self._skill_tree:
                # Create a tree on the fly
                from evoskill.skill_tree import SkillNode
                root = SkillNode(name=self._skill.name, skill=self._skill)
                self._skill_tree = SkillTree(root=root, base_path=self._skill_path.parent)
            with self._console.status("[dim]生成子技能 prompt…[/dim]"):
                enriched = self._optimizer.generate_child_prompts(self._skill, specs)
            self._skill_tree.split("", enriched)
            self._skill_tree.save()
            self._console.print("[success]拆分完成！用 /tree 查看。[/success]")
        return True

    def _cmd_ckpt(self) -> bool:
        ckpts = self._ckpt.list_checkpoints()
        if not ckpts:
            self._console.print("[warning]没有找到任何 checkpoint。[/warning]")
            return True
        self._console.print(f"[info]找到 {len(ckpts)} 个 checkpoint:[/info]")
        for c in ckpts:
            meta = c["meta"]
            ts = meta.get("created_at", "?")
            ver = meta.get("skill_version", "?")
            self._console.print(f"  • [cyan]{c['name']}[/cyan] (version={ver}, created={ts})")
        return True

    def _cmd_restore(self, name: str) -> bool:
        if not name:
            self._console.print("[error]Usage: /restore <checkpoint-name>[/error]")
            return True
        ckpt_path = self._ckpt.ckpt_dir / name
        if not ckpt_path.is_dir():
            self._console.print(f"[error]Checkpoint not found:[/error] {name}")
            return True
        try:
            meta = self._ckpt.restore_to(
                ckpt_path,
                skill_dest=self._skill_path,
                trace_dest=self._config.storage.trace_path,
            )
            self._skill = skill_module.load(self._skill_path)
            self._console.print(
                f"[success]已恢复 checkpoint:[/success] {name} "
                f"(version={meta.get('skill_version', '?')})"
            )
        except Exception as e:
            self._console.print(f"[error]恢复失败: {e}[/error]")
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, text: str) -> Message:
        """Build a user ``Message``, attaching any pending images."""
        if not self._pending_images:
            return Message(role="user", content=text)

        parts: List[ContentPart] = list(self._pending_images)
        parts.append(TextContent(text=text))
        self._pending_images.clear()
        return Message(role="user", content=parts)

    @staticmethod
    def _tool_guidance_text() -> str:
        return (
            "You have built-in local tools: list_dir, read_file, search_repo, write_file, shell. "
            "For repository or filesystem questions, inspect the real files before answering. "
            "Prefer list_dir/read_file/search_repo over shell when possible. "
            "Only use write_file or destructive shell commands when the user explicitly asks for file changes."
        )

    def _on_tool_event(self, event: str, payload: dict) -> None:
        if event == "start":
            self._console.print(
                f"[info]tool → {payload.get('name')}[/info] {payload.get('arguments', '')}"
            )
            return
        if event == "finish":
            result = str(payload.get("result", ""))
            preview = result.splitlines()[0] if result else ""
            self._console.print(
                f"[success]tool ✓ {payload.get('name')}[/success] {preview}"
            )

    def _show_command_help(self, prefix: Optional[str] = None) -> None:
        prefix = prefix or ""
        if prefix and not prefix.startswith("/"):
            prefix = f"/{prefix}"

        matches = []
        for command, usage, description in _COMMAND_SPECS:
            if prefix and command not in {"/", "/help"} and not command.startswith(prefix):
                continue
            label = f"{command} {usage}".rstrip()
            matches.append(f"{label:<28} {description}")

        if prefix and not matches:
            self._console.print(
                f"[warning]没有匹配到命令:[/warning] {prefix}\n[dim]输入 / 或 /help 查看完整命令列表[/dim]"
            )
            return

        title = "⌨️ Commands"
        if prefix:
            title = f"⌨️ Commands matching {prefix}"
        self._console.print(
            Panel(
                "\n".join(matches),
                title=title,
                border_style="cyan",
            )
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _file_to_data_url(path: Path) -> str:
    """Read a local file and return a ``data:`` URL with base64 encoding."""
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"
