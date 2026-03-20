#!/usr/bin/env python3
"""Demo 2: 从已有 writing-skill 开始 — 加载预设的写作技能，交互式对话并优化。

使用方式:
    cd /Users/mzm/code/evo_agent
    conda activate pr
    python demo/demo_from_skill.py

每个步骤都支持交互：你可以跳过、修改输入、自定义反馈、改变 target 等。
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
# Step 1: 加载已有的 writing-skill
# ==================================================================
banner("📂 Step 1: 加载已有的 writing-skills.yaml")

skill_path = Path("demo/writing-skills.yaml")
writing_skill = skill_module.load(skill_path)

console.print(f"[green]✓[/green] 已加载 Skill: [cyan]{writing_skill.name}[/cyan] ({writing_skill.version})")
console.print(Panel(
    writing_skill.system_prompt,
    title="当前 System Prompt",
    border_style="dim",
))

if not Confirm.ask("继续?", default=True):
    sys.exit(0)

# ==================================================================
# Step 2: 用已有 Skill 生成内容
# ==================================================================
banner("💬 Step 2: 用已有 Skill 生成内容 — 基线效果")

llm = LLMClient(config)
storage = TraceStorage(config.storage)

default_prompts = [
    "帮我写一条朋友圈文案，关于今天加班到很晚终于完成项目的心情",
    "写一段产品描述：一款便携式手冲咖啡壶，适合户外露营",
    "帮我写一封邮件，婉拒一个不想参加的饭局",
]

console.print("[dim]预置了 3 条测试 prompt，你可以修改或添加自己的。[/dim]\n")
test_prompts = []
for i, default in enumerate(default_prompts, 1):
    console.print(f"[dim]预置 {i}:[/dim] {default}")
    choice = Prompt.ask(f"  操作", choices=["使用", "修改", "跳过"], default="使用")
    if choice == "使用":
        test_prompts.append(default)
    elif choice == "修改":
        custom = Prompt.ask("  输入你的 prompt")
        test_prompts.append(custom)
    # "跳过" → do nothing

# 允许添加额外 prompt
while Confirm.ask("添加更多 prompt?", default=False):
    extra = Prompt.ask("  输入 prompt")
    if extra.strip():
        test_prompts.append(extra.strip())

if not test_prompts:
    console.print("[yellow]没有 prompt，跳过生成步骤。[/yellow]")
else:
    traces = []
    for i, prompt_text in enumerate(test_prompts, 1):
        user_msg = Message(role="user", content=prompt_text)
        messages = skill_module.compile_messages(writing_skill, [user_msg])

        console.print(f"\n[bold green]User {i}:[/bold green] {prompt_text}")

        with console.status("[dim]生成中…[/dim]"):
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
    # Step 3: 交互式反馈
    # ==================================================================
    banner("📋 Step 3: 对生成内容给出反馈")

    console.print("[dim]对每条生成结果，你可以评分、给出批评，或跳过。[/dim]\n")
    for i, (trace, prompt_text) in enumerate(zip(traces, test_prompts), 1):
        console.print(f"[cyan]Task {i}:[/cyan] {prompt_text[:50]}…" if len(prompt_text) > 50 else f"[cyan]Task {i}:[/cyan] {prompt_text}")
        action = Prompt.ask("  反馈", choices=["评分", "跳过"], default="评分")
        if action == "跳过":
            continue

        score = float(Prompt.ask("  评分 (0.0-1.0, 越低越差)", default="0.3"))
        critique = Prompt.ask("  批评/建议 (直接回车跳过)", default="")
        correction = Prompt.ask("  理想回复 (直接回车跳过)", default="")

        fb = Feedback(
            score=max(0.0, min(1.0, score)),
            critique=critique or None,
            correction=correction or None,
        )
        trace.feedback = fb
        storage.append(trace)
        console.print(f"  [green]✓ 反馈已记录 (score={fb.score})[/green]")

    # ==================================================================
    # Step 4: 设置优化方向
    # ==================================================================
    banner("🎯 Step 4: 设置优化方向")

    current_target = writing_skill.target or "(未设置)"
    console.print(f"[dim]当前 target:[/dim] {current_target}")
    new_target = Prompt.ask(
        "输入新的优化方向 (直接回车保持不变)",
        default=writing_skill.target or "更像真人说话，少套话，有温度和个性",
    )
    writing_skill = writing_skill.model_copy(update={"target": new_target})
    skill_module.save(writing_skill, skill_path)
    console.print(f"[green]✓[/green] Target 已设置: [bold]{writing_skill.target}[/bold]")

    # ==================================================================
    # Step 5: 运行 APO 优化
    # ==================================================================
    if Confirm.ask("\n运行 APO 优化?", default=True):
        banner("🧠 Step 5: 运行 APO 优化")

        engine = APOEngine(config, llm)
        feedback_traces = storage.get_feedback_samples()
        console.print(f"[dim]找到 {len(feedback_traces)} 条反馈样本[/dim]")

        if feedback_traces:
            with console.status("[dim]APO 优化中…[/dim]"):
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
                title="📝 优化前",
                border_style="dim",
            ))
            console.print(Panel(
                evolved_skill.system_prompt,
                title="📝 优化后",
                border_style="bright_green",
            ))

            # ==================================================================
            # Step 6: 用进化后的 Skill 重新生成
            # ==================================================================
            if test_prompts and Confirm.ask("\n用进化后的 Skill 重新生成第一个任务做对比?", default=True):
                banner("🔄 Step 6: A/B 对比")

                compare_prompt = test_prompts[0]
                user_msg = Message(role="user", content=compare_prompt)
                console.print(f"[bold green]User:[/bold green] {compare_prompt}\n")

                messages_v2 = skill_module.compile_messages(evolved_skill, [user_msg])
                with console.status("[dim]生成中…[/dim]"):
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
            console.print("[yellow]没有反馈样本，跳过优化。[/yellow]")

console.print("\n[bold]✨ Demo 完成！[/bold]")
console.print("[dim]进化后的 skill 已保存到 demo/writing-skills.yaml[/dim]")
console.print("[dim]你可以继续用 CLI 与它交互：python -m evoskill.main --skill demo/example[/dim]\n")
