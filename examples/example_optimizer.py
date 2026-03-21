#!/usr/bin/env python3
"""
TrainFreeOptimizer completeExample

这个Example演示如何使用新的优化引擎来优化提示词。

Flow：
1. CreateInitial prompt
2. Create模拟的failure experience（带负面Feedback）
3. Set优化器
4. Run优化
5. View results
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill import (
    TextPrompt,
    SimpleGradient,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
    TrainFreeOptimizer,
    OptimizerConfig,
    ConservativeStrategy,
    AggressiveStrategy,
)

# Set日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_mock_failures():
    """Create模拟的failure experience"""
    failures = []

    # Failure case 1: 回答太啰嗦
    exp1 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python是一种高级编程语言，由Guido van Rossum于1991年Create...（后面还有500字）"},
        ],
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.NEGATIVE,
            critique="回答太长太啰嗦，用户只需要简单解释",
            score=0.3,
        ),
    )
    failures.append(exp1)

    # Failure case 2: 没有回答重点
    exp2 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "How do I sort a list?"},
            {"role": "assistant", "content": "排序是一个复杂的话题，涉及计算机科学的基础..."},
        ],
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CORRECTION,
            critique="没有给出直接答案",
            correction="你可以使用 sorted() 函数或 list.sort() 方法。",
        ),
    )
    failures.append(exp2)

    # Failure case 3: 格式混乱
    exp3 = ConversationExperience(
        conversation=[
            {"role": "user", "content": "Explain machine learning"},
            {"role": "assistant", "content": "机器学习是AI的一个分支它使用算法从数据中学习然后做出预测..."},
        ],
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.NEGATIVE,
            critique="格式混乱，难以阅读",
            score=0.4,
        ),
    )
    failures.append(exp3)

    return failures


def mock_adapter_generate(prompt, context=None, **kwargs):
    """模拟适配器的生成函数"""
    # 这里返回一个模拟的梯度
    return SimpleGradient(
        text="分析Failure case发现：1) 回答过于冗长 2) 缺少结构化格式 3) 未直接回答用户问题",
        metadata={"num_failures": 3}
    )


def mock_adapter_apply_gradient(prompt, gradient, conservative=False, **kwargs):
    """模拟适配器的应用梯度函数"""
    # Create新Version的提示词
    new_prompt = prompt.bump_version()

    if conservative:
        new_prompt.content = f"{prompt.content}\n\n注意：回答要简洁直接，格式清晰。"
    else:
        new_prompt.content = (
            "你是一个简洁明了的助手。\n\n"
            "回答原则：\n"
            "1. 直接回答用户问题，不啰嗦\n"
            "2. 使用清晰的结构和格式\n"
            "3. 给出实用的Example和代码\n"
            "4. 避免过度解释"
        )

    return new_prompt


def create_mock_adapter():
    """Create模拟适配器"""
    from evoskill import BaseModelAdapter

    class MockAdapter(BaseModelAdapter):
        def __init__(self):
            super().__init__(model_name="mock-model")

        def generate(self, prompt, context=None, **kwargs):
            return mock_adapter_generate(prompt, context, **kwargs)

        def _call_api(self, messages, system=None, temperature=0.7, **kwargs):
            return "Mock response"

        def _count_tokens_impl(self, text):
            return len(text.split())

        def compute_gradient(self, prompt, failures, target=None, **kwargs):
            return mock_adapter_generate(prompt, failures, **kwargs)

        def apply_gradient(self, prompt, gradient, conservative=False, **kwargs):
            return mock_adapter_apply_gradient(prompt, gradient, conservative, **kwargs)

    return MockAdapter()


def example_basic_optimization():
    """基本优化Example"""
    print("\n" + "="*80)
    print("Example 1: 基本优化Flow")
    print("="*80 + "\n")

    # 1. CreateInitial prompt
    initial_prompt = TextPrompt(
        content="You are a helpful AI assistant.",
        version="v1.0",
    )
    print(f"Initial prompt (v{initial_prompt.version}):\n{initial_prompt.content}\n")

    # 2. Createfailure experience
    failures = create_mock_failures()
    print(f"收集到 {len(failures)} 个Failure case")
    for i, f in enumerate(failures, 1):
        print(f"  {i}. {f.feedback.critique}")
    print()

    # 3. Create适配器和优化器
    adapter = create_mock_adapter()
    config = OptimizerConfig(
        max_steps=3,
        conservative=False,
        validate_every_step=False,  # 暂时Skip验证
    )
    optimizer = TrainFreeOptimizer(adapter, config)

    # 4. Run优化
    result = optimizer.optimize(
        prompt=initial_prompt,
        experiences=failures,
        validator=None,  # 不使用验证器
    )

    # 5. 显示results
    print("\n" + "-"*80)
    print("Optimization results:")
    print("-"*80)
    print(f"Steps: {result.steps_taken}")
    print(f"Converged: {result.converged}")
    print(f"\nFinal prompt (v{result.optimized_prompt.version}):\n{result.optimized_prompt.content}\n")

    # 显示历史
    print("Optimization history:")
    for step in result.history:
        print(f"  Step {step.step_num}: v{step.old_prompt.version} → v{step.new_prompt.version}")
        print(f"    failed数: {step.num_failures}")
        print(f"    梯度: {step.gradient[:80]}...")


def example_with_validation():
    """带验证的优化Example"""
    print("\n" + "="*80)
    print("Example 2: 带验证的优化")
    print("="*80 + "\n")

    # CreateInitial prompt
    initial_prompt = TextPrompt(
        content="Answer the user's questions.",
        version="v1.0",
    )

    # Createfailure experience
    failures = create_mock_failures()

    # Create验证器函数（模拟）
    def mock_validator(prompt):
        """模拟验证器 - 检查提示词length作为简单指标"""
        # 假设更好的提示词会包含更多指导
        score = min(len(prompt.content) / 200, 1.0)
        print(f"  验证score: {score:.3f} (length={len(prompt.content)})")
        return score

    # Create优化器
    adapter = create_mock_adapter()
    config = OptimizerConfig(
        max_steps=3,
        conservative=False,
        validate_every_step=True,
        early_stopping_patience=2,
    )
    optimizer = TrainFreeOptimizer(adapter, config)

    # Run优化
    result = optimizer.optimize(
        prompt=initial_prompt,
        experiences=failures,
        validator=mock_validator,
    )

    # 显示results
    print("\n" + "-"*80)
    print("Optimization results:")
    print("-"*80)
    print(f"初始score: {result.final_score - result.improvement:.3f}")
    print(f"最终score: {result.final_score:.3f}")
    print(f"提升: {result.improvement:+.3f}")
    print(f"\nFinal prompt:\n{result.optimized_prompt.content}\n")


def example_strategies():
    """策略对比Example"""
    print("\n" + "="*80)
    print("Example 3: 不同优化策略对比")
    print("="*80 + "\n")

    initial_prompt = TextPrompt(
        content="帮助用户。",
        version="v1.0",
    )
    failures = create_mock_failures()
    adapter = create_mock_adapter()

    # Conservative strategy
    print("Conservative strategy:")
    config_conservative = OptimizerConfig(
        max_steps=2,
        conservative=True,
        validate_every_step=False,
    )
    optimizer_c = TrainFreeOptimizer(adapter, config_conservative)
    result_c = optimizer_c.optimize(initial_prompt, failures, validator=None)
    print(f"  Final version: {result_c.optimized_prompt.version}")
    print(f"  Final prompt: {result_c.optimized_prompt.content[:100]}...\n")

    # Aggressive strategy
    print("Aggressive strategy:")
    config_aggressive = OptimizerConfig(
        max_steps=2,
        conservative=False,
        validate_every_step=False,
    )
    optimizer_a = TrainFreeOptimizer(adapter, config_aggressive)
    result_a = optimizer_a.optimize(initial_prompt, failures, validator=None)
    print(f"  Final version: {result_a.optimized_prompt.version}")
    print(f"  Final prompt: {result_a.optimized_prompt.content[:100]}...\n")


def main():
    """Run所有Example"""
    print("\n" + "🔍 " * 30)
    print("TrainFreeOptimizer completeExample")
    print("🔍 " * 30)

    # RunExample
    example_basic_optimization()
    example_with_validation()
    example_strategies()

    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
