"""
论文分类真实 API 优化 Demo

使用真实的 Anthropic API 进行论文分类和优化

训练集: 30 条
验证集: 10 条
测试集: 10 条

优化流程:
1. 加载数据
2. 创建初始 skill (简单 prompt)
3. 在训练集上运行，收集失败案例
4. 使用 TrainFreeOptimizer 优化 prompt
5. 在验证集和测试集上评估
6. 监控优化效果
"""

import csv
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

from tresskill import (
    AnthropicAdapter,
    TrainFreeOptimizer,
    OptimizerConfig,
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
)

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Step 1: 数据加载
# ============================================================================

def load_data(csv_path: str, train_size: int = 30, eval_size: int = 10, test_size: int = 10, seed: int = 42):
    """加载数据并分割"""
    logger.info(f"📂 加载数据: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    logger.info(f"✅ 数据总量: {len(data)} 条")

    # 统计类别
    labels = [item['answer'] for item in data]
    label_counts = Counter(labels)
    logger.info(f"📊 类别数: {len(label_counts)}")
    logger.info(f"   前10个类别: {label_counts.most_common(10)}")

    # 随机打乱
    random.seed(seed)
    random.shuffle(data)

    # 分割
    train = data[:train_size]
    eval_set = data[train_size:train_size + eval_size]
    test = data[train_size + eval_size:train_size + eval_size + test_size]

    logger.info(f"\n✅ 数据分割完成:")
    logger.info(f"   训练集: {len(train)} 条")
    logger.info(f"   验证集: {len(eval_set)} 条")
    logger.info(f"   测试集: {len(test)} 条")

    # 统计分割后的类别分布
    train_labels = [item['answer'] for item in train]
    logger.info(f"\n训练集类别分布: {Counter(train_labels)}")

    return train, eval_set, test


# ============================================================================
# Step 2: 创建分类器
# ============================================================================

def create_initial_prompt() -> str:
    """创建初始分类 prompt"""
    prompt = """You are a scientific paper classification expert. Classify research papers into one of 26 categories.

Categories:
A: Quantum Physics
B: Chemical Physics
C: Atomic Physics
D: Soft Condensed Matter
E: Robotics
F: Computation and Language
G: Software Engineering
H: Information Retrieval
I: High Energy Physics - Theory
J: High Energy Physics - Phenomenology
K: Optics
L: Artificial Intelligence
M: Computer Vision
N: Nuclear Theory
O: Astrophysics
P: Probability
Q: Operating Systems
R: Signal Processing
S: Optimization and Control
T: Dynamical Systems
U: Differential Geometry
V: Mathematical Physics
W: Multimedia
X: Methodology
Y: Combinatorics
Z: Neural and Evolutionary Computing

Instructions:
1. Read the paper carefully
2. Identify the main research domain and methodology
3. Return ONLY the category letter (A-Z), no explanation

Example:
Input: "Quantum Error Correction..."
Output: A
"""
    return prompt


def classify_with_llm(
    adapter: AnthropicAdapter,
    prompt: str,
    question: str
) -> str:
    """使用 LLM 进行分类"""
    # 构建消息
    messages = [
        {"role": "user", "content": f"{prompt}\n\nPaper to classify:\n{question}\n\nCategory:"}
    ]

    # 调用 API
    try:
        response = adapter.client.messages.create(
            model=adapter.model_name,
            max_tokens=10,
            temperature=0.3,
            messages=[{"role": "user", "content": messages[0]["content"]}]
        )

        # 提取答案
        answer = response.content[0].text.strip().upper()

        # 提取字母（如果有多余字符）
        if len(answer) > 0 and answer[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return answer[0]
        else:
            # 默认返回 A
            logger.warning(f"无法解析答案: {answer}, 默认返回 A")
            return 'A'

    except Exception as e:
        logger.error(f"API 调用失败: {e}")
        return 'A'


# ============================================================================
# Step 3: 评估函数
# ============================================================================

def evaluate(
    adapter: AnthropicAdapter,
    prompt: str,
    data: List[Dict],
    dataset_name: str = "Dataset",
    max_samples: int = None
) -> Dict:
    """评估分类器性能"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 评估 {dataset_name}")
    logger.info(f"{'='*60}")

    if max_samples:
        data = data[:max_samples]

    correct = 0
    errors = []

    for i, item in enumerate(data):
        question = item['question']
        true_label = item['answer']

        # 预测
        pred_label = classify_with_llm(adapter, prompt, question)

        # 判断
        if pred_label == true_label:
            correct += 1
            logger.info(f"✅ {i+1}. 正确! 预测={pred_label}, 真实={true_label}")
        else:
            logger.info(f"❌ {i+1}. 错误! 预测={pred_label}, 真实={true_label}")
            errors.append({
                'question': question[:150],
                'true_label': true_label,
                'pred_label': pred_label,
            })

    accuracy = correct / len(data) if data else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"准确率: {accuracy:.2%} ({correct}/{len(data)})")
    logger.info(f"错误数: {len(errors)}/{len(data)}")
    logger.info(f"{'='*60}")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(data),
        'errors': errors,
    }


# ============================================================================
# Step 4: 收集失败案例（用于优化）
# ============================================================================

def collect_experiences(
    adapter: AnthropicAdapter,
    prompt: str,
    data: List[Dict]
) -> List[ConversationExperience]:
    """收集失败案例并转为 Experience 格式"""
    logger.info(f"\n收集训练集上的失败案例...")

    experiences = []

    for item in data:
        question = item['question']
        true_label = item['answer']

        # 预测
        pred_label = classify_with_llm(adapter, prompt, question)

        # 创建 experience
        exp = ConversationExperience(
            conversation=[
                {"role": "user", "content": f"Classify this paper:\n{question}"},
                {"role": "assistant", "content": f"Category: {pred_label}"},
            ]
        )

        # 添加反馈
        if pred_label != true_label:
            # 失败案例 - 添加负面反馈
            exp.feedback = CompositeFeedback(
                feedback_type=FeedbackType.NEGATIVE,
                critique=f"Wrong classification. Predicted {pred_label}, but correct answer is {true_label}. "
                         f"Need to better identify the main research domain.",
                score=0.0,
            )
            experiences.append(exp)
        else:
            # 成功案例 - 添加正面反馈
            exp.feedback = CompositeFeedback(
                feedback_type=FeedbackType.POSITIVE,
                critique="Correct classification",
                score=1.0,
            )
            experiences.append(exp)

    # 统计
    failures = [e for e in experiences if e.feedback and e.feedback.to_score() < 0.5]
    logger.info(f"✅ 收集到 {len(experiences)} 个 experiences")
    logger.info(f"   其中失败案例: {len(failures)} 条")
    logger.info(f"   成功案例: {len(experiences) - len(failures)} 条")

    return experiences


# ============================================================================
# Step 5: 优化循环
# ============================================================================

def optimize_prompt(
    adapter: AnthropicAdapter,
    initial_prompt: str,
    train_data: List[Dict],
    eval_data: List[Dict],
    num_iterations: int = 3
) -> Tuple[str, Dict]:
    """优化 prompt"""

    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 开始优化循环")
    logger.info(f"{'='*60}")

    # 创建优化器
    config = OptimizerConfig(
        max_steps=1,  # 每次迭代只优化1步
        gradient_accumulation_steps=10,
        early_stopping_patience=2,
        target="Improve classification accuracy for scientific papers",
    )

    optimizer = TrainFreeOptimizer(adapter=adapter, config=config)

    current_prompt = initial_prompt
    best_prompt = initial_prompt
    best_accuracy = 0.0

    history = []

    for iteration in range(num_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 优化迭代 {iteration + 1}/{num_iterations}")
        logger.info(f"{'='*60}")

        # 评估当前 prompt
        eval_result = evaluate(adapter, current_prompt, eval_data, f"验证集 (迭代 {iteration+1})")

        # 记录
        history.append({
            'iteration': iteration + 1,
            'accuracy': eval_result['accuracy'],
        })

        # 更新最佳 prompt
        if eval_result['accuracy'] > best_accuracy:
            best_accuracy = eval_result['accuracy']
            best_prompt = current_prompt
            logger.info(f"✨ 新的最佳准确率: {best_accuracy:.2%}")

        # 收集失败案例
        experiences = collect_experiences(adapter, current_prompt, train_data)

        # 创建 OptimizablePrompt
        prompt_obj = TextPrompt(content=current_prompt)

        # 优化
        logger.info(f"\n优化 prompt...")
        result = optimizer.optimize(
            prompt=prompt_obj,
            experiences=experiences,
            validator=None,
        )

        # 更新 prompt
        current_prompt = result.optimized_prompt.content
        logger.info(f"✅ Prompt 已更新")

        # 显示改进
        if result.improvement is not None:
            logger.info(f"   改进: {result.improvement:+.3f}")

    # 最终评估
    logger.info(f"\n{'='*60}")
    logger.info(f"最终评估")
    logger.info(f"{'='*60}")

    final_result = evaluate(adapter, best_prompt, eval_data, "验证集 (最终)")

    optimization_result = {
        'initial_accuracy': history[0]['accuracy'] if history else 0,
        'final_accuracy': final_result['accuracy'],
        'improvement': final_result['accuracy'] - history[0]['accuracy'] if history else 0,
        'num_iterations': num_iterations,
        'history': history,
    }

    return best_prompt, optimization_result


# ============================================================================
# 主流程
# ============================================================================

def main():
    """主流程"""
    logger.info("🚀 论文分类真实 API 优化 Demo")
    logger.info("="*60)

    # Step 1: 加载数据
    train_data, eval_data, test_data = load_data(
        'demo/data/intern_camp5.csv',
        train_size=30,
        eval_size=10,
        test_size=10,
        seed=42
    )

    # Step 2: 创建 adapter
    logger.info(f"\n创建 Anthropic Adapter...")
    adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
    logger.info(f"✅ Adapter 创建成功: {adapter.model_name}")

    # Step 3: 创建初始 prompt
    initial_prompt = create_initial_prompt()
    logger.info(f"\n✅ 初始 prompt 创建完成 ({len(initial_prompt)} 字符)")

    # Step 4: 评估初始模型
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 评估初始模型")
    logger.info(f"{'='*60}")

    # 为了节省 API 调用，只在少量样本上评估
    initial_test_result = evaluate(
        adapter, initial_prompt, test_data[:5], "测试集 (初始, 前5条)"
    )

    # Step 5: 优化
    optimized_prompt, optimization_result = optimize_prompt(
        adapter,
        initial_prompt,
        train_data,
        eval_data,
        num_iterations=3
    )

    # Step 6: 在测试集上评估优化后效果
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 评估优化后模型")
    logger.info(f"{'='*60}")

    final_test_result = evaluate(
        adapter, optimized_prompt, test_data[:5], "测试集 (优化后, 前5条)"
    )

    # Step 7: 总结
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 最终总结")
    logger.info(f"{'='*60}")
    logger.info(f"训练集大小: {len(train_data)}")
    logger.info(f"验证集大小: {len(eval_data)}")
    logger.info(f"测试集大小: {len(test_data)}")
    logger.info(f"\n优化前测试准确率: {initial_test_result['accuracy']:.2%}")
    logger.info(f"优化后测试准确率: {final_test_result['accuracy']:.2%}")
    logger.info(f"准确率变化: {(final_test_result['accuracy'] - initial_test_result['accuracy']):+.2%}")
    logger.info(f"\n验证集最佳准确率: {optimization_result['final_accuracy']:.2%}")
    logger.info(f"验证集提升: +{optimization_result['improvement']:.2%}")

    # 保存结果
    results = {
        'train_size': len(train_data),
        'eval_size': len(eval_data),
        'test_size': len(test_data),
        'initial_test_accuracy': initial_test_result['accuracy'],
        'final_test_accuracy': final_test_result['accuracy'],
        'test_improvement': final_test_result['accuracy'] - initial_test_result['accuracy'],
        'optimization': optimization_result,
    }

    output_path = Path('demo/data/api_optimization_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ 结果已保存到: {output_path}")

    # 保存优化后的 prompt
    prompt_path = Path('demo/data/optimized_prompt.txt')
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(optimized_prompt)
    logger.info(f"✅ 优化后的 prompt 已保存到: {prompt_path}")

    logger.info(f"\n🎉 Demo 完成!")


if __name__ == "__main__":
    main()
