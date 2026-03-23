"""CLI Interface — Rich-based terminal chat with multimodal & feedback support.

Commands
--------
/image <path>   — attach a local image (base64-encoded) to the next message.
/audio <path>   — attach a local audio file (base64-encoded) to the next message.
/bad <reason>   — mark the *previous* interaction with a low score + critique.
/rewrite <txt>  — mark the *previous* interaction with a correction.
/target <text>  — set a one-line optimization target (e.g. "more human").
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

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.shortcuts import CompleteStyle
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

from treeskill.builtin_tools import build_builtin_tools
from treeskill import skill as skill_module
from treeskill.checkpoint import CheckpointManager
from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.resume import ResumeState
from treeskill.schema import (
    AudioContent,
    AudioURL,
    ContentPart,
    Feedback,
    ImageContent,
    ImageURL,
    Message,
    Skill,
    TextContent,
    Trace,
)
from treeskill.skill_tree import SkillTree
from treeskill.storage import TraceStorage

_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "bold magenta",
        "success": "bold green",
        "error": "bold red",
    }
)

_COMMAND_SPECS = [
    ("/", "", "Show all commands and brief descriptions"),
    ("/help", "", "Show all commands and brief descriptions"),
    ("/image", "<path>", "Attach a local image to the next message"),
    ("/audio", "<path>", "Attach a local audio file to the next message"),
    ("/bad", "<reason>", "Mark the previous reply as bad and record the reason"),
    ("/rewrite", "<ideal response>", "Provide an ideal rewrite for the previous reply and collect DPO preference data"),
    ("/export-dpo", "<output.jsonl>", "Export DPO preference data for fine-tuning"),
    ("/target", "<text>", "Set the long-term optimization target"),
    ("/save", "", "Save the current skill immediately"),
    ("/optimize", "", "Optimize the current skill using recorded feedback"),
    ("/tools", "", "Show built-in tools available to the model"),
    ("/tree", "", "Show the current skill tree"),
    ("/select", "<path>", "Switch to a node inside the skill tree"),
    ("/split", "", "Analyze whether the current skill should be split"),
    ("/ckpt", "", "Show available checkpoints"),
    ("/restore", "<name>", "Restore from a checkpoint"),
    ("/quit", "", "Exit the CLI"),
]


def _get_slash_command_suggestions(text: str) -> List[str]:
    """Return matching slash command names for the current input prefix."""
    if not text.startswith("/") or " " in text:
        return []

    return [
        command
        for command, _usage, _description in _COMMAND_SPECS
        if command.startswith(text)
    ]


class _SlashCommandCompleter(Completer):
    """Prompt-toolkit completer for slash command names."""

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return

        for command, usage, description in _COMMAND_SPECS:
            if not command.startswith(text):
                continue
            display = f"{command} {usage}".rstrip()
            yield Completion(
                text=command,
                start_position=-len(text),
                display=display,
                display_meta=description,
            )


def _build_chat_prompt_session() -> PromptSession[str]:
    """Create the interactive prompt session used by the chat loop."""
    return PromptSession(
        completer=_SlashCommandCompleter(),
        complete_while_typing=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        reserve_space_for_menu=10,
    )


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
        self._prompt_session = _build_chat_prompt_session()

        self._history: List[Message] = []
        self._last_trace: Optional[Trace] = None
        self._pending_media_parts: List[ContentPart] = []
        self._stream_live: Optional[Live] = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive chat loop."""
        # Count existing DPO pairs for display
        dpo_count = len(self._storage.get_dpo_pairs())
        dpo_tip = (
            f"\n[dim]DPO preference pairs: {dpo_count} | "
            f"use /rewrite to provide an ideal reply and accumulate DPO training data, "
            f"then export with /export-dpo[/dim]"
        )

        self._console.print(
            Panel(
                f"[bold]TreeSkill Chat[/bold]\n"
                f"Skill: [cyan]{self._skill.name}[/cyan] "
                f"({self._skill.version})\n"
                f"Model: [cyan]{self._config.llm.model}[/cyan]\n"
                f"Built-in tools: [cyan]{', '.join(sorted(self._builtin_tools))}[/cyan]\n\n"
                f"[dim]Type / or /help to view command help[/dim]"
                + dpo_tip,
                title="🧬 TreeSkill",
                border_style="bright_blue",
            )
        )

        while True:
            try:
                raw = self._prompt_session.prompt("You (/ for commands)\n> ")
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
            streamed_text = ""

            def _on_delta(delta: str) -> None:
                nonlocal streamed_text
                streamed_text += delta
                self._render_streaming_assistant(streamed_text)

            with Live(
                self._streaming_assistant_panel(""),
                console=self._console,
                refresh_per_second=12,
                transient=False,
            ) as live:
                self._stream_live = live
                try:
                    response = self._llm.generate_stream(
                        full_messages,
                        tools=self._builtin_tools,
                        on_tool_event=self._on_tool_event,
                        on_delta=_on_delta,
                    )
                    self._stream_live.update(
                        self._final_assistant_panel(response)
                    )
                finally:
                    self._stream_live = None
            self._history.append(response)

            # --- trace ---
            trace = Trace(inputs=list(self._history[:-1]), prediction=response)
            self._storage.append(trace)
            self._last_trace = trace

            # --- render ---
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
        if cmd == "/audio":
            return self._cmd_audio(arg)
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
        self._pending_media_parts.append(
            ImageContent(image_url=ImageURL(url=data_url))
        )
        self._console.print(
            f"[success]Image attached:[/success] {p.name}  "
            f"({len(self._pending_media_parts)} pending)"
        )
        return True

    def _cmd_audio(self, path_str: str) -> bool:
        if not path_str:
            self._console.print("[error]Usage: /audio <path>[/error]")
            return True
        p = Path(path_str).expanduser().resolve()
        if not p.is_file():
            self._console.print(f"[error]File not found:[/error] {p}")
            return True
        data_url = _file_to_data_url(p)
        self._pending_media_parts.append(
            AudioContent(audio_url=AudioURL(url=data_url))
        )
        self._console.print(
            f"[success]Audio attached:[/success] {p.name}  "
            f"({len(self._pending_media_parts)} pending)"
        )
        return True

    def _cmd_bad(self, reason: str) -> bool:
        if self._last_trace is None:
            self._console.print("[warning]No previous interaction to mark.[/warning]")
            return True
        fb = Feedback(score=0.1, critique=reason or "Bad response")
        self._last_trace.feedback = fb
        self._storage.upsert(self._last_trace)
        self._console.print("[success]Feedback recorded (bad).[/success]")
        self._console.print(
            "[dim]Tip: use /rewrite <ideal response> to collect DPO preference data too[/dim]"
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
        self._storage.upsert(self._last_trace)
        self._console.print("[success]Feedback recorded (rewrite).[/success]")
        return True

    def _cmd_export_dpo(self, path_str: str) -> bool:
        output = path_str.strip() or "dpo_pairs.jsonl"
        count = self._storage.export_dpo(output)
        if count == 0:
            self._console.print(
                "[warning]No DPO data available to export.[/warning]\n"
                "[dim]Use /rewrite <ideal response> to rewrite a model reply and collect preference pairs.[/dim]"
            )
        else:
            self._console.print(
                f"[success]Exported {count} DPO preference pairs →[/success] {output}\n"
                f"[dim]Format: {{prompt, chosen, rejected, score, critique}}. "
                f"Ready for DPO/RLHF fine-tuning.[/dim]"
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
            self._console.print(f"[yellow]⚠ Found an unfinished optimization[/yellow]")
            self._console.print(resume.summary())
            choice = Prompt.ask("Resume the previous run or restart from scratch?",
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
                "\n[warning]⚠ Optimization interrupted. Progress has been saved. Run /optimize again to resume.[/warning]"
            )
            return True
        except Exception as e:
            self._console.print(
                f"\n[error]✗ Optimization failed: {e}[/error]\n"
                f"[dim]Progress has been saved. Run /optimize again to resume.[/dim]"
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
                    f"[info]Current optimization target:[/info] {self._skill.target}"
                )
            else:
                self._console.print("[warning]No optimization target has been set yet. Usage: /target <text>[/warning]")
            return True
        self._skill = self._skill.model_copy(update={"target": text})
        if self._skill_tree:
            self._skill_tree.root.skill = self._skill
            self._skill_tree.save()
        else:
            skill_module.save(self._skill, self._skill_path)
        self._console.print(
            f"[success]Optimization target set:[/success] {text}"
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
            self._console.print("[warning]No skill tree is loaded. Start with a skill directory path.[/warning]")
            return True
        tree_str = self._skill_tree.list_tree()
        self._console.print(Panel(tree_str, title="🌳 Skill Tree", border_style="green"))
        return True

    def _cmd_select(self, dotpath: str) -> bool:
        if not self._skill_tree:
            self._console.print("[warning]No skill tree is loaded.[/warning]")
            return True
        if not dotpath:
            self._console.print("[error]Usage: /select <dotpath> (e.g. social.moments)[/error]")
            return True
        try:
            node = self._skill_tree.get(dotpath)
            self._skill = node.skill
            self._console.print(
                f"[success]Switched to:[/success] {node.name} ({node.skill.version})"
            )
        except KeyError as e:
            self._console.print(f"[error]{e}[/error]")
        return True

    def _cmd_split(self) -> bool:
        traces = self._storage.get_feedback_samples()
        if len(traces) < 2:
            self._console.print("[warning]At least 2 feedback samples are required to analyze a split.[/warning]")
            return True
        with self._console.status("[dim]Analyzing whether a split is needed…[/dim]"):
            specs = self._optimizer.analyze_split_need(self._skill, traces)
        if not specs:
            self._console.print("[info]The current skill does not need to be split.[/info]")
            return True
        self._console.print(f"[success]Suggested split into {len(specs)} child skills:[/success]")
        for s in specs:
            self._console.print(f"  • {s['name']}: {s.get('description', '')}")
        confirm = Prompt.ask("Confirm split?", choices=["y", "n"], default="y")
        if confirm == "y":
            if not self._skill_tree:
                # Create a tree on the fly
                from treeskill.skill_tree import SkillNode
                root = SkillNode(name=self._skill.name, skill=self._skill)
                self._skill_tree = SkillTree(root=root, base_path=self._skill_path.parent)
            with self._console.status("[dim]Generating child skill prompts…[/dim]"):
                enriched = self._optimizer.generate_child_prompts(self._skill, specs)
            self._skill_tree.split("", enriched)
            self._skill_tree.save()
            self._console.print("[success]Split complete. Use /tree to inspect it.[/success]")
        return True

    def _cmd_ckpt(self) -> bool:
        ckpts = self._ckpt.list_checkpoints()
        if not ckpts:
            self._console.print("[warning]No checkpoints found.[/warning]")
            return True
        self._console.print(f"[info]Found {len(ckpts)} checkpoints:[/info]")
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
                f"[success]Restored checkpoint:[/success] {name} "
                f"(version={meta.get('skill_version', '?')})"
            )
        except Exception as e:
            self._console.print(f"[error]Restore failed: {e}[/error]")
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, text: str) -> Message:
        """Build a user ``Message``, attaching any pending media parts."""
        if not self._pending_media_parts:
            return Message(role="user", content=text)

        parts: List[ContentPart] = list(self._pending_media_parts)
        parts.append(TextContent(text=text))
        self._pending_media_parts.clear()
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

    def _streaming_assistant_panel(self, text: str) -> Panel:
        body = text if text else "[dim]Thinking...[/dim]"
        return Panel(
            body,
            title="🤖 Assistant",
            border_style="bright_cyan",
        )

    def _final_assistant_panel(self, response: Message) -> Panel:
        return Panel(
            Markdown(
                response.content
                if isinstance(response.content, str)
                else "[multimodal response]"
            ),
            title="🤖 Assistant",
            border_style="bright_cyan",
        )

    def _render_streaming_assistant(self, text: str) -> None:
        if self._stream_live is not None:
            self._stream_live.update(self._streaming_assistant_panel(text))
            return
        self._console.print(self._streaming_assistant_panel(text))

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
                f"[warning]No matching command found:[/warning] {prefix}\n[dim]Type / or /help to view the full command list[/dim]"
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
    if mime == "audio/x-wav":
        mime = "audio/wav"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"
