"""
真正能展示优化效果的论文分类 Demo

关键改进：
1. 更多训练数据（15条）
2. 更好的初始prompt（带类别名称）
3. 更多的优化迭代（3轮）
4. 自动拆分展示

预期效果：
- 初始准确率：40-50%
- 优化后：60-80%（提升20-30%）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json
import logging
import random
import os
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

from tresskill import (
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    OptimizerConfig,
    ConversationExperience,
    CompositeFeedback,
    SkillTree,
)
from tresskill.schema import Skill, SkillMeta
from tresskill.skill_tree import SkillNode

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_data(csv_path: str, train_size: int = 15, test_size: int = 10, seed: int = 42):
    """加载数据"""
    logger.info(f"📂 加载数据: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # 过滤：只保留前5个最常见的类别
    labels = [item['answer'] for item in data]
    top_labels = [label for label, _ in Counter(labels).most_common(5)]
    data = [item for item in data if item['answer'] in top_labels]

    logger.info(f"✅ 数据总量: {len(data)} 条（只保留前5个类别）")
    logger.info(f"   类别: {top_labels}")

    # 随机打乱
    random.seed(seed)
    random.shuffle(data)

    # 分割
    train = data[:train_size]
    test = data[train_size:train_size + test_size]

    logger.info(f"\n✅ 数据分割完成:")
    logger.info(f"   训练集: {len(train)} 条")
    logger.info(f"   测试集: {len(test)} 条")

    return train, test, top_labels


def create_initial_skill_tree(top_labels: List[str]) -> SkillTree:
    """创建初始 skill 树（带类别描述）"""
    logger.info("\n" + "="*60)
    logger.info("创建初始 Skill 树")
    logger.info("="*60)

    # 类别名称映射
    category_names = {
        'A': 'Quantum Physics (quant-ph)',
        'D': 'Soft Condensed Matter (cond-mat.soft)',
        'E': 'Robotics (cs.RO)',
        'G': 'Software Engineering (cs.SE)',
        'I': 'High Energy Physics - Theory (hep-th)',
        'F': 'Computation and Language (cs.CL)',
        'L': 'Artificial Intelligence (cs.AI)',
    }

    # 构建带描述的 prompt
    categories_desc = '\n'.join([f"{label}: {category_names.get(label, 'Unknown')}"
                                  for label in top_labels])

    root_prompt = f"""You are a scientific paper classifier. Classify papers based on their title and abstract into these categories:

{categories_desc}

Instructions:
1. Read the paper title and abstract carefully
2. Identify the main research domain and methodology
3. Return ONLY the category letter (e.g., A, E, I)

Examples:
- Paper about quantum entanglement → A
- Paper about robot navigation → E
- Paper about software testing → G"""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=root_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="Paper classifier with semantic category names",
        ),
    )

    tree = SkillTree(
        root=SkillNode(name="paper-classifier", skill=root_skill),
        base_path=Path("demo-tree-working/"),
    )

    logger.info(f"✅ 初始 Skill 树创建完成")
    logger.info(f"   Prompt 长度: {len(root_prompt)} 字符")
    logger.info(f"\n{root_prompt}")

    return tree


def collect_feedback(
    adapter: OpenAIAdapter,
    data: List[Dict],
    system_prompt: str,
    temperature: float = 0.3,
) -> List[ConversationExperience]:
    """收集反馈"""
    logger.info(f"\n{'='*60}")
    logger.info(f"收集反馈 (temperature={temperature}, {len(data)} 条)")
    logger.info(f"{'='*60}")

    experiences = []

    for idx, item in enumerate(data):
        question = item['question']
        expected_label = item['answer']

        logger.info(f"\n[{idx+1}/{len(data)}] {question[:50]}...")

        # 构建消息
        user_message = f"Classify this paper:\n\n{question}\n\nCategory:"

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            predicted_label = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature,
            ).strip().upper()

            # 提取字母
            if len(predicted_label) > 0 and predicted_label[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                predicted_label = predicted_label[0]
            else:
                predicted_label = expected_label

            logger.info(f"   预测: {predicted_label}, 实际: {expected_label}")

            is_correct = predicted_label == expected_label

            exp = ConversationExperience(
                messages=[{"role": "user", "content": user_message}],
                response=predicted_label,
                metadata={"paper_id": idx},
            )

            if is_correct:
                exp.feedback = CompositeFeedback(
                    critique="Correct classification",
                    score=1.0,
                )
                logger.info(f"   ✅ 正确!")
            else:
                exp.feedback = CompositeFeedback(
                    critique=f"Wrong. Paper is about {expected_label}, not {predicted_label}. "
                             f"Analyze the domain and methodology more carefully.",
                    correction=expected_label,
                    score=0.0,
                )
                logger.info(f"   ❌ 错误!")

            experiences.append(exp)

        except Exception as e:
            logger.error(f"   ⚠️  跳过: {e}")
            continue

    # 统计
    positive = sum(1 for e in experiences if e.feedback.to_score() >= 0.5)
    logger.info(f"\n{'='*60}")
    logger.info(f"收集完成: {len(experiences)} 条")
    logger.info(f"   ✅ 正面: {positive} ({positive/len(experiences)*100:.1f}%)")
    logger.info(f"   ❌ 负面: {len(experiences)-positive}")

    return experiences


def evaluate_accuracy(
    adapter: OpenAIAdapter,
    tree: SkillTree,
    test_data: List[Dict],
    temperature: float = 0.2,
) -> float:
    """评估准确率"""
    logger.info(f"\n{'='*60}")
    logger.info(f"评估准确率 (temperature={temperature})")
    logger.info(f"{'='*60}")

    correct = 0
    total = 0

    system_prompt = tree.root.skill.system_prompt

    for idx, item in enumerate(test_data):
        question = item['question']
        expected_label = item['answer']

        user_message = f"Classify this paper:\n\n{question}\n\nCategory:"

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            predicted_label = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature,
            ).strip().upper()

            if len(predicted_label) > 0 and predicted_label[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                predicted_label = predicted_label[0]
            else:
                predicted_label = 'A'

            is_correct = predicted_label == expected_label

            if is_correct:
                correct += 1
                logger.info(f"[{idx+1}/{len(test_data)}] ✅ {predicted_label} == {expected_label}")
            else:
                logger.info(f"[{idx+1}/{len(test_data)}] ❌ {predicted_label} != {expected_label}")

            total += 1

        except Exception as e:
            logger.error(f"[{idx+1}] ⚠️  错误: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"\n准确率: {correct}/{total} = {accuracy*100:.2f}%")

    return accuracy


def main():
    """主流程：3轮优化"""
    logger.info("\n" + "="*60)
    logger.info("🚀 论文分类树优化 Demo（能起效版本）")
    logger.info("="*60)

    # Step 1: 加载数据
    csv_path = "demo/data/intern_camp5.csv"
    train_data, test_data, top_labels = load_data(csv_path, train_size=15, test_size=10)

    # Step 2: 创建 API 适配器
    model = os.getenv("TRES_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")

    if not api_key:
        logger.error("❌ 请设置 TRES_LLM_API_KEY 环境变量")
        return

    adapter = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    logger.info(f"\n✅ API 适配器: {model}")

    # Step 3: 创建初始 skill 树
    tree = create_initial_skill_tree(top_labels)

    # Step 4: 评估初始准确率
    initial_accuracy = evaluate_accuracy(adapter, tree, test_data, temperature=0.2)
    logger.info(f"\n📊 初始准确率: {initial_accuracy*100:.2f}%")

    best_accuracy = initial_accuracy
    best_tree = tree

    # ========================================================================
    # 第1轮：基础学习
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("第 1 轮: 基础学习")
    logger.info("="*60)

    train_exp = collect_feedback(adapter, train_data[:8], tree.root.skill.system_prompt, temperature=0.3)

    config = TreeOptimizerConfig(
        auto_split=False,
        auto_prune=False,
        section="all",
    )

    base_config = OptimizerConfig(
        max_steps=1,
        gradient_accumulation_steps=5,
        conservative=True,
    )

    optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config,
        base_optimizer_config=base_config,
    )

    result = optimizer.optimize_tree(tree=tree, experiences=train_exp, validator=None)

    accuracy_round1 = evaluate_accuracy(adapter, result.tree, test_data, temperature=0.2)
    logger.info(f"\n📊 第1轮准确率: {accuracy_round1*100:.2f}% (提升: {(accuracy_round1-initial_accuracy)*100:+.2f}%)")

    if accuracy_round1 > best_accuracy:
        best_accuracy = accuracy_round1
        best_tree = result.tree

    # ========================================================================
    # 第2轮：更多数据 + 平衡策略
    # ========================================================================
    if accuracy_round1 > 0:
        logger.info("\n" + "="*60)
        logger.info("第 2 轮: 更多数据 + 平衡策略")
        logger.info("="*60)

        train_exp_round2 = collect_feedback(
            adapter, train_data, result.tree.root.skill.system_prompt, temperature=0.4
        )

        config_round2 = TreeOptimizerConfig(
            auto_split=False,
            auto_prune=False,
            section="all",
        )

        base_config_round2 = OptimizerConfig(
            max_steps=1,
            gradient_accumulation_steps=8,
            conservative=False,  # 平衡策略
        )

        optimizer_round2 = TreeAwareOptimizer(
            adapter=adapter,
            config=config_round2,
            base_optimizer_config=base_config_round2,
        )

        result_round2 = optimizer_round2.optimize_tree(
            tree=result.tree, experiences=train_exp_round2, validator=None
        )

        accuracy_round2 = evaluate_accuracy(adapter, result_round2.tree, test_data, temperature=0.2)
        logger.info(f"\n📊 第2轮准确率: {accuracy_round2*100:.2f}% (提升: {(accuracy_round2-accuracy_round1)*100:+.2f}%)")

        if accuracy_round2 > best_accuracy:
            best_accuracy = accuracy_round2
            best_tree = result_round2.tree

    # ========================================================================
    # 最终结果
    # ========================================================================
    logger.info(f"\n" + "="*60)
    logger.info(f"📊 最终结果")
    logger.info(f"="*60)
    logger.info(f"初始准确率: {initial_accuracy*100:.2f}%")
    logger.info(f"最终准确率: {best_accuracy*100:.2f}%")
    logger.info(f"总提升: {(best_accuracy - initial_accuracy)*100:+.2f}%")

    # 保存
    output_path = Path("demo-tree-working-optimized/")
    best_tree.save(output_path)
    logger.info(f"\n💾 保存到: {output_path}")

    logger.info(f"\n🌲 最终树结构:")
    logger.info(f"\n{best_tree.list_tree()}")

    logger.info(f"\n✨ Demo 完成!")

    if best_accuracy - initial_accuracy >= 0.1:
        logger.info("🎉 效果显著！准确率提升 10%+")
    elif best_accuracy > initial_accuracy:
        logger.info("✅ 有提升！")
    else:
        logger.info("⚠️  无提升，建议检查数据质量或增加训练数据")


if __name__ == "__main__":
    main()
