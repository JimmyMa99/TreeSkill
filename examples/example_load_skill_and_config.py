#!/usr/bin/env python3
"""
Complete example for loading a saved Skill and using Config

这个Example演示如何：
1. Load已存储的 skill file
2. 使用全局Config
3. 使用适配器调用 LLM
4. SaveAfter optimization的 skill

Flow：
- LoadConfig → Load skill → Create适配器 → 调用 LLM → Save skill
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill import (
    # Config
    GlobalConfig,
    # Skill 管理
    load_skill,
    save_skill,
    compile_messages,
    # 新架构（可选）
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
)

# Set日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_load_config_and_skill():
    """Example 1: LoadConfig和 Skill"""
    print("\n" + "="*60)
    print("Example 1: LoadConfig和 Skill")
    print("="*60 + "\n")

    # Method 1: 从 YAML fileLoadConfig
    config_path = "demo/example/config.yaml"
    if Path(config_path).exists():
        config = GlobalConfig.from_yaml(config_path)
        print(f"✓ 从 {config_path} LoadConfig")
        print(f"  - LLM 模型: {config.llm.model}")
        print(f"  - Judge 模型: {config.llm.judge_model}")
        print(f"  - API 地址: {config.llm.base_url}")
        print(f"  - Trace path: {config.storage.trace_path}")
    else:
        print(f"⚠ Configfile {config_path} 不存在，使用默认Config")
        config = GlobalConfig()

    # Method 2: 从 .env fileLoad（自动）
    # 只需要在项目根目录Create .env file，GlobalConfig 会自动读取
    # .env Example：
    #   EVO_LLM_API_KEY=your-key
    #   EVO_LLM_BASE_URL=https://api.siliconflow.cn/v1
    #   EVO_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct

    # Method 3: 从环境变量Load（最高优先级）
    # export EVO_LLM_API_KEY="your-key"
    # config = GlobalConfig()  # 会自动读取环境变量

    print("\n" + "-"*60)

    # Load Skill
    skill_path = "demo/example/skill.yaml"
    if Path(skill_path).exists():
        skill = load_skill(skill_path)
        print(f"✓ 从 {skill_path} Load Skill")
        print(f"  - 名称: {skill.name}")
        print(f"  - Version: {skill.version}")
        print(f"  - System Prompt length: {len(skill.system_prompt)} 字符")
        print(f"  - Target: {skill.target}")
    else:
        print(f"⚠ Skill file {skill_path} 不存在")
        return None

    return config, skill


def example_2_use_adapter(config, skill):
    """Example 2: 使用适配器调用 LLM"""
    print("\n" + "="*60)
    print("Example 2: 使用适配器调用 LLM")
    print("="*60 + "\n")

    # 延迟导入 OpenAIAdapter（需要安装 openai tiktoken）
    try:
        from evoskill import OpenAIAdapter
    except ImportError:
        print("⚠ OpenAIAdapter 导入failed")
        print("  需要安装依赖:")
        print("    pip install openai tiktoken")
        print("  或:")
        print("    pip install -e .\n")
        print("  Skip此Example\n")
        return None

    # Create OpenAI 适配器
    adapter = OpenAIAdapter(
        api_key=config.llm.api_key.get_secret_value(),
        base_url=config.llm.base_url,
        model=config.llm.model,
    )

    print(f"✓ Create OpenAI 适配器")
    print(f"  - API 地址: {config.llm.base_url}")
    print(f"  - 模型: {config.llm.model}")

    # 准备User input
    user_input = [{"role": "user", "content": "Write a short passage about spring for me"}]

    # 编译消息（system prompt + few-shot + user input）
    messages = compile_messages(skill, user_input)

    print(f"\n✓ 编译消息")
    print(f"  - 消息数量: {len(messages)}")
    print(f"  - 第一条消息角色: {messages[0]['role']}")

    # 调用 LLM
    print(f"\n→ 调用 LLM...")
    try:
        response = adapter.generate(messages, temperature=config.llm.temperature)
        print(f"✓ 收到响应")
        print(f"\nAssistant reply:\n{response.content}\n")
        return response
    except Exception as e:
        print(f"✗ 调用failed: {e}")
        print(f"  提示: 请检查 API Key 和 base_url 是否正确")
        return None


def example_3_save_skill(skill):
    """Example 3: Save Skill"""
    print("\n" + "="*60)
    print("Example 3: Save Skill")
    print("="*60 + "\n")

    # 修改 skill（模拟After optimization）
    skill.version = "v1.1"
    skill.system_prompt = skill.system_prompt + "\n\nNew rule: avoid overly formal polite filler."

    # Save到新path
    output_path = Path("skills/my-skill-v1.1.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_skill(skill, output_path)

    print(f"✓ Skill 已Save到 {output_path}")
    print(f"  - 名称: {skill.name}")
    print(f"  - Version: {skill.version}")
    print(f"  - file大小: {output_path.stat().st_size} 字节")


def example_4_manual_feedback_and_gradient():
    """Example 4: 手动Feedback + 计算梯度（新架构）"""
    print("\n" + "="*60)
    print("Example 4: 手动Feedback + 计算梯度（新架构）")
    print("="*60 + "\n")

    # Create初始 Prompt
    prompt = TextPrompt(
        content="You are a writing assistant who helps users draft many kinds of text."
    )

    print(f"✓ 初始 Prompt:")
    print(f"  {prompt.content}\n")

    # Createfailure experience（模拟 /bad 和 /rewrite 命令）
    experience = ConversationExperience(
        messages=[
            {"role": "user", "content": "Write a short passage about spring"},
        ],
        response="Spring arrives and everything comes back to life...",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CORRECTION,
            critique="回答太简单，缺乏细节和感情",
            correction="春天来了，小区的玉兰花开了，空气中弥漫着淡淡的花香...",
        ),
    )

    print(f"✓ Createfailure experience:")
    print(f"  - User input: {experience.messages[0]['content']}")
    print(f"  - Assistant reply: {experience.response}")
    print(f"  - Feedback类型: {experience.feedback.feedback_type}")
    print(f"  - Critique: {experience.feedback.critique}")
    print(f"  - Ideal reply: {experience.feedback.correction}\n")

    # 使用 Mock 适配器计算梯度
    try:
        from evoskill import MockAdapter
        adapter = MockAdapter()

        print(f"→ 计算文本梯度...")
        gradient = adapter.compute_gradient(
            prompt=prompt,
            failures=[experience],  # failure experience列表
        )

        print(f"✓ 梯度计算完成:")
        print(f"  - 梯度content:\n{gradient.text}\n")

        # 应用梯度更新 Prompt
        print(f"→ 应用梯度更新 Prompt...")
        new_prompt = adapter.apply_gradient(
            prompt=prompt,
            gradient=gradient,
        )

        print(f"✓ 新 Prompt:")
        print(f"  {new_prompt.content}\n")

        print(f"✓ Optimization complete！")
    except ImportError as e:
        print(f"⚠ MockAdapter 导入failed: {e}")
        print(f"  Skip梯度计算Example\n")




def example_5_load_and_continue():
    """Example 5: Load已有 Skill 并Continue优化"""
    print("\n" + "="*60)
    print("Example 5: Load已有 Skill 并Continue优化")
    print("="*60 + "\n")

    skill_path = Path("skills/my-skill-v1.1.yaml")

    if not skill_path.exists():
        print(f"⚠ Skill file {skill_path} 不存在")
        print(f"  请先Run前面的ExampleCreate skill")
        return

    # Load已有 skill
    skill = load_skill(skill_path)

    print(f"✓ Load已有 Skill:")
    print(f"  - 名称: {skill.name}")
    print(f"  - Version: {skill.version}")
    print(f"  - System Prompt length: {len(skill.system_prompt)} 字符\n")

    # Continue优化（模拟）
    skill.version = "v1.2"
    skill.system_prompt = skill.system_prompt + "\n\nNew rule: use more vivid metaphors and concrete detail."

    # Save新Version
    new_path = Path("skills/my-skill-v1.2.yaml")
    save_skill(skill, new_path)

    print(f"✓ After optimizationSave到 {new_path}")
    print(f"  - 新Version: {skill.version}\n")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("EvoSkill - Load已存储 Skill 并使用 Config Example")
    print("="*60)

    # Example 1: LoadConfig和 Skill
    result = example_1_load_config_and_skill()
    if result is None:
        print("\n⚠ 缺少必要file，Example终止")
        return

    config, skill = result

    # Example 2: 使用适配器调用 LLM
    # 注意：需要有效的 API Key 才能成功
    # example_2_use_adapter(config, skill)

    # Example 3: Save Skill
    example_3_save_skill(skill)

    # Example 4: 手动Feedback + 计算梯度
    example_4_manual_feedback_and_gradient()

    # Example 5: Load并Continue优化
    example_5_load_and_continue()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
