"""
Fully automated optimization example - no manual feedback required

这个Example展示如何：
1. Createtest集（自动生成或手动定义）
2. 使用 AutoValidator 在test集上评估
3. 完全自动化优化循环 - 无需人工标注

使用场景：
- 你有一批标注好的test数据
- 想要批量优化多个 prompt
- 持续集成/自动化test
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill.core import (
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
    TrainFreeOptimizer,
    OptimizerConfig,
)
from evoskill.adapters.openai import OpenAIAdapter
from evoskill.core.validators import AutoValidator

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ===========================================================================
# Step 1: 定义test集（可以是你的真实数据）
# ===========================================================================

def create_test_dataset():
    """Createtest集 - 这里用简单Example，实际中应该是你的真实数据"""

    test_cases = [
        # Test case 1: 简洁性
        ConversationExperience(
            messages=[{"role": "user", "content": "What is Python?"}],
            response="",  # 将由模型生成
            feedback=CompositeFeedback(
                feedback_type=FeedbackType.SCORE,
                score=0.8,  # 期望score
                metadata={"criteria": "回答应该简洁、有结构"}
            )
        ),
        # Test case 2: 直接回答
        ConversationExperience(
            messages=[{"role": "user", "content": "How do I sort a list?"}],
            response="",
            feedback=CompositeFeedback(
                feedback_type=FeedbackType.SCORE,
                score=0.9,
                metadata={"criteria": "应该直接给出答案和代码"}
            )
        ),
        # Test case 3: 通俗性
        ConversationExperience(
            messages=[{"role": "user", "content": "解释机器学习"}],
            response="",
            feedback=CompositeFeedback(
                feedback_type=FeedbackType.SCORE,
                score=0.7,
                metadata={"criteria": "应该通俗易懂，不要太学术"}
            )
        ),
    ]

    return test_cases


# ===========================================================================
# Step 2: Create自动验证器（用 LLM 作为 Judge）
# ===========================================================================

def create_llm_judge_validator(adapter):
    """Create基于 LLM 的自动验证器"""

    def llm_judge(prompt):
        """使用 LLM 作为 Judge 来评估 prompt 质量"""

        # 这里我们用一个简单的启发式方法
        # 实际中可以调用另一个 LLM 来评估

        score = 0.0

        # 检查 1: 提示词length（不要太短也不要太长）
        length = len(prompt.content)
        if 100 <= length <= 500:
            score += 0.3
        elif 50 <= length < 100 or 500 < length <= 1000:
            score += 0.2

        # 检查 2: 包含关键指导词
        keywords = ['简洁', '直接', '结构', 'Example', '清晰']
        found_keywords = sum(1 for kw in keywords if kw in prompt.content)
        score += found_keywords * 0.14

        # 限制在 1.0 以内
        return min(score, 1.0)

    return llm_judge


# ===========================================================================
# Step 3: 模拟自动收集Failure case（在实际中会用真实交互）
# ===========================================================================

def collect_failures_from_testset(adapter, prompt, test_cases, threshold=0.7):
    """在test集上Run并收集Failure case"""

    failures = []

    print(f"\n{'='*80}")
    print(f"在 {len(test_cases)} 个Test case上评估...")
    print(f"{'='*80}")

    for i, test_case in enumerate(test_cases):
        # 生成回答
        response = adapter.generate(prompt, context=[test_case])

        # 简单评估（实际中可以用 LLM Judge）
        # 这里用length和关键词作为简单Example
        response_text = str(response) if not isinstance(response, str) else response

        score = 0.5  # 基础分
        if 50 <= len(response_text) <= 200:
            score += 0.2
        if any(kw in response_text for kw in ['例如', '比如', '可以']):
            score += 0.2
        if any(kw in response_text for kw in ['首先', '然后', '最后']):
            score += 0.1

        # 判断是否failed
        expected_score = test_case.feedback.score if test_case.feedback else 0.7

        if score < expected_score:
            # Createfailure experience
            failure = ConversationExperience(
                messages=test_case.messages,
                response=response_text,
                feedback=CompositeFeedback(
                    feedback_type=FeedbackType.CRITIQUE,
                    critique=f"score {score:.2f} < 期望 {expected_score:.2f}",
                    score=score
                )
            )
            failures.append(failure)
            print(f"  ❌ 用例 {i+1}: failed (score {score:.2f})")
        else:
            print(f"  ✓ 用例 {i+1}: passed (score {score:.2f})")

    return failures


# ===========================================================================
# Step 4: 完全自动化的优化循环
# ===========================================================================

def fully_automatic_optimization(
    adapter,
    initial_prompt,
    test_cases,
    max_iterations=3,
    target_score=0.8
):
    """完全自动化的优化 - 无需人工Feedback"""

    print("="*80)
    print("🤖 完全自动化优化开始")
    print("="*80)
    print(f"test集大小: {len(test_cases)}")
    print(f"目标score: {target_score}")
    print(f"maximum iterations: {max_iterations}")

    current_prompt = initial_prompt
    validator = create_llm_judge_validator(adapter)

    for iteration in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"迭代 {iteration + 1}/{max_iterations}")
        print(f"{'='*80}")

        # 1. 在test集上评估
        print("\n[1/3] 在test集上评估当前 prompt...")
        failures = collect_failures_from_testset(adapter, current_prompt, test_cases)

        # 2. 检查是否passed
        if len(failures) == 0:
            print(f"\n✅ 所有testpassed！Optimization complete。")
            return current_prompt

        print(f"\n[2/3] 发现 {len(failures)} 个Failure case，开始优化...")

        # 3. Run优化
        config = OptimizerConfig(
            max_steps=2,
            conservative=False,
            validate_every_step=False
        )
        optimizer = TrainFreeOptimizer(adapter, config)

        result = optimizer.optimize(current_prompt, failures)

        # 4. 更新 prompt
        current_prompt = result.optimized_prompt

        print(f"\n[3/3] Optimization complete:")
        print(f"  Version: {current_prompt.version}")
        print(f"  Steps: {result.steps_taken}")

        # 5. 用 validator 评估
        overall_score = validator(current_prompt)
        print(f"  整体score: {overall_score:.2f}")

        if overall_score >= target_score:
            print(f"\n✅ 达到目标score {target_score}！Optimization complete。")
            break

    print(f"\n{'='*80}")
    print("🎉 完全自动化Optimization complete")
    print(f"{'='*80}")
    print(f"Final version: {current_prompt.version}")
    print(f"最终score: {overall_score:.2f}")
    print(f"\n最终 Prompt:\n{current_prompt.content}")

    return current_prompt


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    """Run完全自动化优化"""

    print("\n" + "="*80)
    print("Fully automated optimization example - no manual feedback required")
    print("="*80)

    # 1. Create初始 prompt
    initial_prompt = TextPrompt(
        content="You are an assistant.",
        version="v1.0"
    )

    print(f"\n初始 Prompt (v{initial_prompt.version}):")
    print(f"{initial_prompt.content}")

    # 2. Create适配器（使用你的 API）
    # 注意：这里需要Set环境变量或直接传入参数
    # export OPENAI_API_KEY="your-key"
    # export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"  # 硅流 API

    try:
        adapter = OpenAIAdapter(
            model="Qwen/Qwen2.5-7B-Instruct",  # 使用硅流上的 7B 模型
            # base_url="https://api.siliconflow.cn/v1"  # 或从环境变量读取
        )
    except Exception as e:
        print(f"\n❌ Create适配器failed: {e}")
        print("\n请Set环境变量:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export OPENAI_BASE_URL='https://api.siliconflow.cn/v1'")
        return

    # 3. Createtest集
    test_cases = create_test_dataset()
    print(f"\ntest集: {len(test_cases)} 个用例")

    # 4. Run完全自动化优化
    final_prompt = fully_automatic_optimization(
        adapter=adapter,
        initial_prompt=initial_prompt,
        test_cases=test_cases,
        max_iterations=3,
        target_score=0.8
    )

    # 5. Saveresults
    print(f"\n{'='*80}")
    print("Save the final prompt")
    print(f"{'='*80}")

    with open("optimized_prompt.txt", "w") as f:
        f.write(f"# Version: {final_prompt.version}\n\n")
        f.write(final_prompt.content)

    print(f"✓ Saved to optimized_prompt.txt")


if __name__ == "__main__":
    main()
