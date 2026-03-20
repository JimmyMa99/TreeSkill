#!/usr/bin/env python3
"""Demo 1: 从零开始构建 — 创建一个全新的写作技能并通过对话+反馈进化它。

使用方式:
    cd /Users/mzm/code/evo_agent
    conda activate pr
    python demo/demo_from_scratch.py

每个步骤都支持交互：你可以自定义初始 prompt、修改反馈、设置 target 等。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
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
# Step 1: 创建极简 Skill
# ==================================================================
banner("📝 Step 1: 从零创建一个极简写作 Skill")

default_prompt = "你是一个写作助手。"
console.print(f"[dim]预设的初始 System Prompt:[/dim] {default_prompt}")
custom = Prompt.ask(
    "输入你的初始 System Prompt (直接回车使用预设)",
    default=default_prompt,
)

skill_name = Prompt.ask("Skill 名称", default="scratch-writer")

scratch_skill = Skill(
    name=skill_name,
    system_prompt=custom,
)

scratch_path = Path(f"demo/{skill_name}.yaml")
skill_module.save(scratch_skill, scratch_path)
console.print(f"[green]✓[/green] 已创建极简 Skill → {scratch_path}")
console.print(f"[dim]System Prompt:[/dim] {scratch_skill.system_prompt}")

# ==================================================================
# Step 2: 用极简 Skill 生成内容
# ==================================================================
if Confirm.ask("\n继续生成内容测试?", default=True):
    banner("💬 Step 2: 用极简 Skill 生成内容，观察效果")

    llm = LLMClient(config)
    storage = TraceStorage(config.storage)

    default_test = "帮我写一段关于春天的短文，100字左右"
    test_prompt = Prompt.ask("输入测试 prompt", default=default_test)

    user_msg = Message(role="user", content=test_prompt)
    messages = skill_module.compile_messages(scratch_skill, [user_msg])

    console.print(f"\n[bold green]User:[/bold green] {test_prompt}\n")

    with console.status("[dim]生成中…[/dim]"):
        response = llm.generate(messages)

    console.print(Panel(
        Markdown(response.content if isinstance(response.content, str) else str(response.content)),
        title=f"🤖 Assistant ({scratch_skill.version} 极简 prompt)",
        border_style="bright_cyan",
    ))

    trace1 = Trace(inputs=[user_msg], prediction=response)
    storage.append(trace1)

    # ==================================================================
    # Step 3: 用户反馈
    # ==================================================================
    banner("📋 Step 3: 对生成结果给出反馈")

    console.print("[dim]你可以评分、给出批评、提供理想回复。[/dim]\n")
    action = Prompt.ask("反馈操作", choices=["评分", "跳过"], default="评分")

    if action == "评分":
        score = float(Prompt.ask("评分 (0.0-1.0)", default="0.2"))
        critique = Prompt.ask(
            "批评/建议",
            default="太像AI写的，缺乏真实感和个人色彩，读起来像模板",
        )
        correction = Prompt.ask("理想回复 (直接回车跳过)", default="")

        feedback = Feedback(
            score=max(0.0, min(1.0, score)),
            critique=critique or None,
            correction=correction or None,
        )
        trace1.feedback = feedback
        storage.append(trace1)
        console.print(f"[green]✓ 反馈已记录 (score={feedback.score})[/green]")

    # 允许添加更多交互+反馈
    while Confirm.ask("\n再做一轮对话+反馈?", default=False):
        extra_prompt = Prompt.ask("输入新 prompt")
        user_msg2 = Message(role="user", content=extra_prompt)
        messages2 = skill_module.compile_messages(scratch_skill, [user_msg2])

        console.print(f"\n[bold green]User:[/bold green] {extra_prompt}")
        with console.status("[dim]生成中…[/dim]"):
            resp2 = llm.generate(messages2)

        console.print(Panel(
            Markdown(resp2.content if isinstance(resp2.content, str) else str(resp2.content)),
            title=f"🤖 Assistant ({scratch_skill.version})",
            border_style="bright_cyan",
        ))

        trace2 = Trace(inputs=[user_msg2], prediction=resp2)
        storage.append(trace2)

        s = float(Prompt.ask("评分 (0.0-1.0)", default="0.3"))
        c = Prompt.ask("批评/建议 (直接回车跳过)", default="")
        if c:
            trace2.feedback = Feedback(score=s, critique=c)
            storage.append(trace2)
            console.print(f"[green]✓ 反馈已记录[/green]")

    # ==================================================================
    # Step 4: 设置优化方向
    # ==================================================================
    banner("🎯 Step 4: 设置优化方向")

    default_target = "更像人，有生活气息，避免AI腔"
    target = Prompt.ask("输入优化方向", default=default_target)
    scratch_skill = scratch_skill.model_copy(update={"target": target})
    skill_module.save(scratch_skill, scratch_path)
    console.print(f"[green]✓[/green] 优化方向已设置: [bold]{scratch_skill.target}[/bold]")

    # ==================================================================
    # Step 5: 运行 APO 优化
    # ==================================================================
    if Confirm.ask("\n运行 APO 优化?", default=True):
        banner("🧠 Step 5: 运行 APO 优化")

        engine = APOEngine(config, llm)
        traces = storage.get_feedback_samples()
        console.print(f"[dim]找到 {len(traces)} 条反馈样本[/dim]")

        if traces:
            with console.status("[dim]APO 优化中…[/dim]"):
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
                banner("🔄 Step 6: A/B 对比")

                messages_v2 = skill_module.compile_messages(evolved_skill, [user_msg])
                console.print(f"[bold green]User:[/bold green] {test_prompt}\n")

                with console.status("[dim]生成中…[/dim]"):
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
            console.print("[yellow]没有反馈样本，跳过优化。[/yellow]")

console.print("\n[bold]✨ Demo 完成！[/bold]")
console.print(f"[dim]进化后的 skill 已保存到 {scratch_path}[/dim]")
console.print(f"[dim]你可以继续用 CLI 与它交互：python -m evoskill.main --skill {scratch_path}[/dim]\n")
