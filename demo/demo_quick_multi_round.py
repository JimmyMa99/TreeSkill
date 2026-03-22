#!/usr/bin/env python3
"""
快速多轮优化 Demo - 简化版，快速展示概念

策略:
1. 小数据集（每个类别10条）
2. 3轮快速优化
3. 展示准确率持续提升
"""

import csv
import logging
import random
import os
from pathlib import Path
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

from tresskill import (
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    OptimizerConfig,
    SkillTree,
    ConversationExperience,
    CompositeFeedback,
)
from tresskill.schema import Skill, SkillMeta
from tresskill.skill_tree import SkillNode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_small_dataset(csv_path: str, samples_per_category: int = 10):
    """创建小数据集"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_data = list(reader)

    categories = ['A', 'E', 'G', 'K', 'M']
    balanced_data = []
    category_data = {cat: [] for cat in categories}

    for item in all_data:
        label = item['answer']
        if label in categories and len(category_data[label]) < samples_per_category:
            category_data[label].append(item)

    for cat, items in category_data.items():
        balanced_data.extend(items)

    random.seed(42)
    random.shuffle(balanced_data)

    train_data = balanced_data[:int(len(balanced_data)*0.7)]
    test_data = balanced_data[int(len(balanced_data)*0.7):]

    return train_data, test_data


def collect_experiences(adapter, data, system_prompt, temperature=0.3):
    """收集经验"""
    experiences = []
    correct = 0

    for idx, item in enumerate(data):
        question = item['question']
        expected = item['answer']

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify:\n\n{question[:400]}\n\nReturn ONLY the letter."},
        ]

        try:
            predicted = adapter._call_api(messages=messages, system=None, temperature=temperature).strip().upper()
            is_correct = predicted == expected.upper()

            exp = ConversationExperience(
                messages=[{"role": "user", "content": question}],
                response=predicted,
                metadata={"paper_id": idx},
            )

            if is_correct:
                exp.feedback = CompositeFeedback(critique="Correct", score=0.9)
                correct += 1
            else:
                exp.feedback = CompositeFeedback(
                    critique=f"Wrong. Should be {expected}, not {predicted}",
                    correction=expected,
                    score=0.1,
                )

            experiences.append(exp)
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

    accuracy = correct / len(experiences) if experiences else 0.0
    return experiences, accuracy


def evaluate(adapter, system_prompt, test_data):
    """评估"""
    correct = 0
    for item in test_data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify:\n\n{item['question'][:400]}\n\nReturn ONLY the letter."},
        ]
        try:
            predicted = adapter._call_api(messages=messages, system=None, temperature=0.3).strip().upper()
            if predicted == item['answer'].upper():
                correct += 1
        except:
            continue
    return correct / len(test_data) if test_data else 0.0


def optimize_round(adapter, tree, experiences, round_name):
    """优化一轮"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 {round_name}")
    logger.info(f"{'='*60}")

    config = TreeOptimizerConfig(
        auto_split=False,
        auto_prune=False,
        optimization_order="bottom_up",
    )

    base_config = OptimizerConfig(
        max_steps=1,
        gradient_accumulation_steps=5,
        conservative=False,
    )

    optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config,
        base_optimizer_config=base_config,
    )

    result = optimizer.optimize_tree(tree=tree, experiences=experiences)

    logger.info(f"✅ 优化完成: 节点优化={result.nodes_optimized}")
    return result.tree


def main():
    logger.info("\n" + "="*60)
    logger.info("⚡ 快速多轮优化 Demo")
    logger.info("="*60)
    logger.info("小数据集 + 3轮快速优化")

    # 加载数据
    csv_path = "demo/data/intern_camp5.csv"
    train_data, test_data = create_small_dataset(csv_path, samples_per_category=10)

    logger.info(f"\n✅ 数据: 训练{len(train_data)}条, 测试{len(test_data)}条")

    # 创建适配器
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    main_model = os.getenv("TRES_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    judge_model = os.getenv("TRES_LLM_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    if not api_key:
        logger.error("❌ 请设置 TRES_LLM_API_KEY")
        return

    main_adapter = OpenAIAdapter(model=main_model, api_key=api_key, base_url=base_url)
    judge_adapter = OpenAIAdapter(model=judge_model, api_key=api_key, base_url=base_url)

    # 创建初始skill树
    initial_prompt = """You are a paper classifier. Classify papers into these categories:

A: Quantum Physics
E: Robotics
G: Software Engineering
K: Mathematics
M: Machine Learning

Return ONLY the category letter (A, E, G, K, or M).
"""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=initial_prompt,
        version="v1.0",
        meta=SkillMeta(name="paper-classifier", description="Paper classifier"),
    )

    tree = SkillTree(
        root=SkillNode(name="root", skill=root_skill),
        base_path=Path("demo-quick-multi-round/"),
    )

    # 评估基准
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 第0轮: 基准准确率")
    logger.info(f"{'='*60}")

    initial_accuracy = evaluate(main_adapter, initial_prompt, test_data)
    logger.info(f"📊 基准准确率: {initial_accuracy*100:.1f}%")

    accuracy_history = [initial_accuracy]
    best_tree = tree
    best_accuracy = initial_accuracy

    # 3轮优化
    num_rounds = 3
    samples_per_round = len(train_data) // num_rounds

    for round_num in range(1, num_rounds + 1):
        start_idx = (round_num - 1) * samples_per_round
        end_idx = start_idx + samples_per_round
        round_data = train_data[start_idx:end_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"第{round_num}轮: 使用数据{start_idx}-{end_idx}")
        logger.info(f"{'='*60}")

        # 收集经验
        experiences, train_acc = collect_experiences(
            main_adapter,
            round_data,
            tree.root.skill.system_prompt,
            temperature=0.3 + round_num * 0.1,
        )
        logger.info(f"📝 训练准确率: {train_acc*100:.1f}%")

        # 优化
        tree = optimize_round(judge_adapter, tree, experiences, f"第{round_num}轮优化")

        # 评估
        test_accuracy = evaluate(main_adapter, tree.root.skill.system_prompt, test_data)
        improvement = (test_accuracy - accuracy_history[-1]) * 100
        logger.info(f"📊 第{round_num}轮测试准确率: {test_accuracy*100:.1f}% ({improvement:+.1f}%)")

        accuracy_history.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_tree = tree
            logger.info(f"   🎯 新最佳!")

    # 总结
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 多轮优化完成")
    logger.info(f"{'='*60}")

    logger.info(f"\n📈 准确率变化:")
    for i, acc in enumerate(accuracy_history):
        if i == 0:
            logger.info(f"   基准: {acc*100:.1f}%")
        else:
            improvement = (acc - accuracy_history[i-1]) * 100
            logger.info(f"   第{i}轮: {acc*100:.1f}% ({improvement:+.1f}%)")

    logger.info(f"\n✅ 最终结果:")
    logger.info(f"   初始: {initial_accuracy*100:.1f}%")
    logger.info(f"   最终: {best_accuracy*100:.1f}%")
    logger.info(f"   总提升: {(best_accuracy-initial_accuracy)*100:+.1f}%")

    # 保存
    output_path = Path("demo-quick-multi-round/")
    best_tree.save(output_path)
    logger.info(f"\n💾 已保存到: {output_path}")

    logger.info(f"\n✅ Demo完成!")


if __name__ == "__main__":
    main()
