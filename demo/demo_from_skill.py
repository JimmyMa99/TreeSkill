#!/usr/bin/env python3
"""Demo 2: Start from an existing writing skill — Load预设的写作技能，交互式对话并优化。

Usage:
    cd /Users/mzm/code/evo_agent
    conda activate pr
    python demo/demo_from_skill.py

每个Step都支持交互：你可以Skip、修改输入、自定义Feedback、改变 target 等。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt, Confirm

from evoskill.config import GlobalConfig
from evoskill.schema import Skill, Message, Feedback, Trace
from evoskill.llm import LLMClient
from evoskill.storage import TraceStorage
from evoskill.optimizer import APOEngine
from evoskill.checkpoint import CheckpointManager
from evoskill import skill as skill_module

console = Console()
config = GlobalConfig()


def banner(text: str) -> None:
    console.print(Panel(text, border_style="bright_blue", expand=False))


# ==================================================================
# Step 1: Load已有的 writing-skill
# ==================================================================
banner("📂 Step 1: Load已有的 writing-skills.yaml")

skill_path = Path("demo/writing-skills.yaml")
writing_skill = skill_module.load(skill_path)

console.print(f"[green]✓[/green] 已Load Skill: [cyan]{writing_skill.name}[/cyan] ({writing_skill.version})")
console.print(Panel(
    writing_skill.system_prompt,
    title="Current System Prompt",
    border_style="dim",
))

if not Confirm.ask("Continue?", default=True):
    sys.exit(0)

# ==================================================================
# Step 2: Generate content with the existing Skill
# ==================================================================
banner("💬 Step 2: Generate content with the existing Skill — baseline behavior")

llm = LLMClient(config)
storage = TraceStorage(config.storage)

default_prompts = [
    "帮我写一条朋友圈文案，关于今天加班到很晚终于完成项目的心情",
    "写一段产品描述：一款便携式手冲咖啡壶，适合户外露营",
    "帮我写一封邮件，婉拒一个不想参加的饭局",
]

console.print("[dim]预置了 3 条test prompt，你可以修改或添加自己的。[/dim]\n")
test_prompts = []
for i, default in enumerate(default_prompts, 1):
    console.print(f"[dim]预置 {i}:[/dim] {default}")
    choice = Prompt.ask(f"  操作", choices=["使用", "修改", "Skip"], default="使用")
    if choice == "使用":
        test_prompts.append(default)
    elif choice == "修改":
        custom = Prompt.ask("  Enter your prompt")
        test_prompts.append(custom)
    # "Skip" → do nothing

# 允许添加额外 prompt
while Confirm.ask("Add more prompts?", default=False):
    extra = Prompt.ask("  输入 prompt")
    if extra.strip():
        test_prompts.append(extra.strip())

if not test_prompts:
    console.print("[yellow]没有 prompt，Skip生成Step。[/yellow]")
else:
    traces = []
    for i, prompt_text in enumerate(test_prompts, 1):
        user_msg = Message(role="user", content=prompt_text)
        messages = skill_module.compile_messages(writing_skill, [user_msg])

        console.print(f"\n[bold green]User {i}:[/bold green] {prompt_text}")

        with console.status("[dim]Generating…[/dim]"):
            response = llm.generate(messages)

        console.print(Panel(
            Markdown(response.content if isinstance(response.content, str) else str(response.content)),
            title=f"🤖 Assistant ({writing_skill.version})",
            border_style="bright_cyan",
        ))

        trace = Trace(inputs=[user_msg], prediction=response)
        storage.append(trace)
        traces.append(trace)

    # ==================================================================
    # Step 3: 交互式Feedback
    # ==================================================================
    banner("📋 Step 3: Provide feedback on the generated content")

    console.print("[dim]对每条生成results，你可以Score、给出Critique，或Skip。[/dim]\n")
    for i, (trace, prompt_text) in enumerate(zip(traces, test_prompts), 1):
        console.print(f"[cyan]Task {i}:[/cyan] {prompt_text[:50]}…" if len(prompt_text) > 50 else f"[cyan]Task {i}:[/cyan] {prompt_text}")
        action = Prompt.ask("  Feedback", choices=["Score", "Skip"], default="Score")
        if action == "Skip":
            continue

        score = float(Prompt.ask("  Score (0.0-1.0, 越低越差)", default="0.3"))
        critique = Prompt.ask("  Critique / suggestion (直接回车Skip)", default="")
        correction = Prompt.ask("  Ideal reply (直接回车Skip)", default="")

        fb = Feedback(
            score=max(0.0, min(1.0, score)),
            critique=critique or None,
            correction=correction or None,
        )
        trace.feedback = fb
        storage.append(trace)
        console.print(f"  [green]✓ Feedback已记录 (score={fb.score})[/green]")

    # ==================================================================
    # Step 4: Set优化方向
    # ==================================================================
    banner("🎯 Step 4: Set优化方向")

    current_target = writing_skill.target or "(not set)"
    console.print(f"[dim]Current target:[/dim] {current_target}")
    new_target = Prompt.ask(
        "Enter a new optimization target (press Enter to keep the current one)",
        default=writing_skill.target or "sound more human, avoid boilerplate, and feel warm and distinctive",
    )
    writing_skill = writing_skill.model_copy(update={"target": new_target})
    skill_module.save(writing_skill, skill_path)
    console.print(f"[green]✓[/green] Target 已Set: [bold]{writing_skill.target}[/bold]")

    # ==================================================================
    # Step 5: Run APO optimization
    # ==================================================================
    if Confirm.ask("\nRun APO optimization?", default=True):
        banner("🧠 Step 5: Run APO optimization")

        engine = APOEngine(config, llm)
        feedback_traces = storage.get_feedback_samples()
        console.print(f"[dim]找到 {len(feedback_traces)} 条Feedback样本[/dim]")

        if feedback_traces:
            with console.status("[dim]APO optimization in progress…[/dim]"):
                evolved_skill = engine.optimize(writing_skill, feedback_traces)

            skill_module.save(evolved_skill, skill_path)

            # Save checkpoint
            ckpt = CheckpointManager("./ckpt")
            ckpt_path = ckpt.save(
                evolved_skill,
                trace_path=Path(config.storage.trace_path),
            )

            console.print(f"\n[bold green]✓ Skill 已进化![/bold green] {writing_skill.version} → {evolved_skill.version}")
            console.print(f"[dim]Checkpoint → {ckpt_path}[/dim]")

            # 对比 prompt 变化
            console.print(Panel(
                writing_skill.system_prompt,
                title="📝 Before optimization",
                border_style="dim",
            ))
            console.print(Panel(
                evolved_skill.system_prompt,
                title="📝 After optimization",
                border_style="bright_green",
            ))

            # ==================================================================
            # Step 6: 用进化后的 Skill 重新生成
            # ==================================================================
            if test_prompts and Confirm.ask("\n用进化后的 Skill 重新生成第一个任务做对比?", default=True):
                banner("🔄 Step 6: A/B comparison")

                compare_prompt = test_prompts[0]
                user_msg = Message(role="user", content=compare_prompt)
                console.print(f"[bold green]User:[/bold green] {compare_prompt}\n")

                messages_v2 = skill_module.compile_messages(evolved_skill, [user_msg])
                with console.status("[dim]Generating…[/dim]"):
                    response_v2 = llm.generate(messages_v2)

                old_text = traces[0].prediction.content
                new_text = response_v2.content

                console.print(Panel(
                    Markdown(old_text if isinstance(old_text, str) else str(old_text)),
                    title=f"Before ({writing_skill.version})",
                    border_style="red",
                ))
                console.print(Panel(
                    Markdown(new_text if isinstance(new_text, str) else str(new_text)),
                    title=f"After ({evolved_skill.version})",
                    border_style="bright_green",
                ))
        else:
            console.print("[yellow]没有Feedback样本，Skip优化。[/yellow]")

console.print("\n[bold]✨ Demo complete![/bold]")
console.print("[dim]The evolved skill has been saved to demo/writing-skills.yaml[/dim]")
console.print("[dim]你可以Continue用 CLI 与它交互：python -m evoskill.main --skill demo/example[/dim]\n")
