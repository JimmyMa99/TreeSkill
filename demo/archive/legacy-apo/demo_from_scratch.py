#!/usr/bin/env python3
"""Demo 1: Build from scratch — Create一个全新的写作技能并passed对话+Feedback进化它。

Usage:
    cd /Users/mzm/code/evo_agent
    conda activate pr
    python demo/demo_from_scratch.py

每个Step都支持交互：你可以自定义初始 prompt、修改Feedback、Set target 等。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm

from treeskill.config import GlobalConfig
from treeskill.schema import Skill, Message, Feedback, Trace
from treeskill.llm import LLMClient
from treeskill.storage import TraceStorage
from treeskill.optimizer import APOEngine
from treeskill.checkpoint import CheckpointManager
from treeskill import skill as skill_module

console = Console()
config = GlobalConfig()


def banner(text: str) -> None:
    console.print(Panel(text, border_style="bright_blue", expand=False))


# ==================================================================
# Step 1: Create a minimal Skill
# ==================================================================
banner("📝 Step 1: 从零Create一个极简写作 Skill")

default_prompt = "You are a writing assistant."
console.print(f"[dim]Default initial System Prompt:[/dim] {default_prompt}")
custom = Prompt.ask(
    "Enter your initial System Prompt (press Enter to use the default)",
    default=default_prompt,
)

skill_name = Prompt.ask("Skill name", default="scratch-writer")

scratch_skill = Skill(
    name=skill_name,
    system_prompt=custom,
)

scratch_path = Path(f"demo/{skill_name}.yaml")
skill_module.save(scratch_skill, scratch_path)
console.print(f"[green]✓[/green] 已Create a minimal Skill → {scratch_path}")
console.print(f"[dim]System Prompt:[/dim] {scratch_skill.system_prompt}")

# ==================================================================
# Step 2: 用极简 Skill 生成content
# ==================================================================
if Confirm.ask("\nContinue生成contenttest?", default=True):
    banner("💬 Step 2: Generate content with the minimal Skill and inspect the result")

    llm = LLMClient(config)
    storage = TraceStorage(config.storage)

    default_test = "Write a short passage about spring for me，100字左右"
    test_prompt = Prompt.ask("Enter a test prompt", default=default_test)

    user_msg = Message(role="user", content=test_prompt)
    messages = skill_module.compile_messages(scratch_skill, [user_msg])

    console.print(f"\n[bold green]User:[/bold green] {test_prompt}\n")

    with console.status("[dim]Generating…[/dim]"):
        response = llm.generate(messages)

    console.print(Panel(
        Markdown(response.content if isinstance(response.content, str) else str(response.content)),
        title=f"🤖 Assistant ({scratch_skill.version} 极简 prompt)",
        border_style="bright_cyan",
    ))

    trace1 = Trace(inputs=[user_msg], prediction=response)
    storage.append(trace1)

    # ==================================================================
    # Step 3: 用户Feedback
    # ==================================================================
    banner("📋 Step 3: 对生成results给出Feedback")

    console.print("[dim]你可以Score、给出Critique、提供Ideal reply。[/dim]\n")
    action = Prompt.ask("Feedback action", choices=["Score", "Skip"], default="Score")

    if action == "Score":
        score = float(Prompt.ask("Score (0.0-1.0)", default="0.2"))
        critique = Prompt.ask(
            "Critique / suggestion",
            default="太像AI写的，缺乏真实感和个人色彩，读起来像模板",
        )
        correction = Prompt.ask("Ideal reply (直接回车Skip)", default="")

        feedback = Feedback(
            score=max(0.0, min(1.0, score)),
            critique=critique or None,
            correction=correction or None,
        )
        trace1.feedback = feedback
        storage.append(trace1)
        console.print(f"[green]✓ Feedback已记录 (score={feedback.score})[/green]")

    # 允许添加更多交互+Feedback
    while Confirm.ask("\n再做一轮对话+Feedback?", default=False):
        extra_prompt = Prompt.ask("Enter a new prompt")
        user_msg2 = Message(role="user", content=extra_prompt)
        messages2 = skill_module.compile_messages(scratch_skill, [user_msg2])

        console.print(f"\n[bold green]User:[/bold green] {extra_prompt}")
        with console.status("[dim]Generating…[/dim]"):
            resp2 = llm.generate(messages2)

        console.print(Panel(
            Markdown(resp2.content if isinstance(resp2.content, str) else str(resp2.content)),
            title=f"🤖 Assistant ({scratch_skill.version})",
            border_style="bright_cyan",
        ))

        trace2 = Trace(inputs=[user_msg2], prediction=resp2)
        storage.append(trace2)

        s = float(Prompt.ask("Score (0.0-1.0)", default="0.3"))
        c = Prompt.ask("Critique / suggestion (直接回车Skip)", default="")
        if c:
            trace2.feedback = Feedback(score=s, critique=c)
            storage.append(trace2)
            console.print(f"[green]✓ Feedback已记录[/green]")

    # ==================================================================
    # Step 4: Set优化方向
    # ==================================================================
    banner("🎯 Step 4: Set优化方向")

    default_target = "sound more human, feel lived-in, and avoid AI-sounding phrasing"
    target = Prompt.ask("Enter the optimization target", default=default_target)
    scratch_skill = scratch_skill.model_copy(update={"target": target})
    skill_module.save(scratch_skill, scratch_path)
    console.print(f"[green]✓[/green] Optimization target set: [bold]{scratch_skill.target}[/bold]")

    # ==================================================================
    # Step 5: Run APO optimization
    # ==================================================================
    if Confirm.ask("\nRun APO optimization?", default=True):
        banner("🧠 Step 5: Run APO optimization")

        engine = APOEngine(config, llm)
        traces = storage.get_feedback_samples()
        console.print(f"[dim]找到 {len(traces)} 条Feedback样本[/dim]")

        if traces:
            with console.status("[dim]APO optimization in progress…[/dim]"):
                evolved_skill = engine.optimize(scratch_skill, traces)

            skill_module.save(evolved_skill, scratch_path)

            # Save checkpoint
            ckpt = CheckpointManager("./ckpt")
            ckpt_path = ckpt.save(
                evolved_skill,
                trace_path=Path(config.storage.trace_path),
            )

            console.print(f"\n[bold green]✓ Skill 已进化![/bold green] {scratch_skill.version} → {evolved_skill.version}")
            console.print(f"[dim]Checkpoint → {ckpt_path}[/dim]")
            console.print(Panel(
                evolved_skill.system_prompt,
                title="📝 进化后的 System Prompt",
                border_style="bright_green",
            ))

            # ==================================================================
            # Step 6: 对比
            # ==================================================================
            if Confirm.ask("\n用进化后的 Skill 重新生成做对比?", default=True):
                banner("🔄 Step 6: A/B comparison")

                messages_v2 = skill_module.compile_messages(evolved_skill, [user_msg])
                console.print(f"[bold green]User:[/bold green] {test_prompt}\n")

                with console.status("[dim]Generating…[/dim]"):
                    response_v2 = llm.generate(messages_v2)

                console.print(Panel(
                    Markdown(response.content if isinstance(response.content, str) else str(response.content)),
                    title=f"Before ({scratch_skill.version})",
                    border_style="red",
                ))
                console.print(Panel(
                    Markdown(response_v2.content if isinstance(response_v2.content, str) else str(response_v2.content)),
                    title=f"After ({evolved_skill.version})",
                    border_style="bright_green",
                ))
        else:
            console.print("[yellow]没有Feedback样本，Skip优化。[/yellow]")

console.print("\n[bold]✨ Demo complete![/bold]")
console.print(f"[dim]The evolved skill has been saved to {scratch_path}[/dim]")
console.print(f"[dim]你可以Continue用 CLI 与它交互：python -m treeskill.main --skill {scratch_path}[/dim]\n")
