#!/usr/bin/env python3
"""
Qwen3-8B 多轮优化 Demo - 使用更小的模型看优化效果

主模型: Qwen/Qwen3-8B (弱模型，有更大优化空间)
Judge模型: Qwen/Qwen2.5-72B (强模型，生成高质量梯度)

策略:
1. 故意很差的初始prompt
2. 5个类别，每个20条数据
3. 3轮迭代优化
4. 观察准确率从低到高的提升
"""

import csv
import logging
import random
import os
import sys
from pathlib import Path
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treeskill import (
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    OptimizerConfig,
    SkillTree,
    ConversationExperience,
    CompositeFeedback,
)
from treeskill.schema import Skill, SkillMeta
from treeskill.skill_tree import SkillNode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_dataset(csv_path: str, samples_per_category: int = 20):
    """Create数据集"""
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
        logger.info(f"   {cat}: {len(items)} 条")

    random.seed(42)
    random.shuffle(balanced_data)

    train_data = balanced_data[:int(len(balanced_data)*0.7)]
    test_data = balanced_data[int(len(balanced_data)*0.7):]

    logger.info(f"✅ 总计: {len(balanced_data)} 条")
    logger.info(f"   训练集: {len(train_data)} 条")
    logger.info(f"   test集: {len(test_data)} 条")

    return train_data, test_data


def collect_experiences(adapter, data, system_prompt, temperature=0.3):
    """收集经验"""
    logger.info(f"📝 收集经验 (n={len(data)}, temp={temperature})")
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

            # 提取第一个字母
            if predicted:
                predicted = predicted[0]

            is_correct = predicted == expected.upper()

            exp = ConversationExperience(
                messages=[{"role": "user", "content": question}],
                response=predicted,
                metadata={"paper_id": idx},
            )

            if is_correct:
                exp.feedback = CompositeFeedback(critique="Correct", score=0.9)
                correct += 1
                logger.info(f"  [{idx+1}] ✅ {predicted}")
            else:
                exp.feedback = CompositeFeedback(
                    critique=f"Wrong. Should be {expected}, not {predicted}",
                    correction=expected,
                    score=0.1,
                )
                logger.info(f"  [{idx+1}] ❌ {predicted} -> {expected}")

            experiences.append(exp)
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

    accuracy = correct / len(experiences) if experiences else 0.0
    logger.info(f"✅ 准确率: {correct}/{len(experiences)} = {accuracy*100:.1f}%")
    return experiences, accuracy


def evaluate(adapter, system_prompt, test_data):
    """评估"""
    logger.info(f"📊 评估 (n={len(test_data)})")
    correct = 0

    for item in test_data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify:\n\n{item['question'][:400]}\n\nReturn ONLY the letter."},
        ]
        try:
            predicted = adapter._call_api(messages=messages, system=None, temperature=0.3).strip().upper()
            if predicted:
                predicted = predicted[0]
            if predicted == item['answer'].upper():
                correct += 1
        except:
            continue

    accuracy = correct / len(test_data) if test_data else 0.0
    logger.info(f"✅ 准确率: {correct}/{len(test_data)} = {accuracy*100:.1f}%")
    return accuracy


def optimize_round(adapter, tree, experiences, round_name):
    """优化一轮"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 {round_name}")
    logger.info(f"{'='*60}")

    config = TreeOptimizerConfig(
        auto_split=False,
        auto_prune=False,
        max_tree_depth=3,
        min_samples_for_split=3,
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

    logger.info(f"✅ Optimization complete: 节点优化={result.nodes_optimized}")
    return result.tree


def main():
    """主Flow"""
    logger.info("\n" + "="*60)
    logger.info("🎯 Qwen3-8B 多轮优化 Demo")
    logger.info("="*60)
    logger.info("主模型: Qwen3-8B (弱模型，有更大优化空间)")
    logger.info("Judge模型: Qwen2.5-72B (强模型，生成高质量梯度)")

    # Load数据
    csv_path = "demo/data/intern_camp5.csv"
    train_data, test_data = create_dataset(csv_path, samples_per_category=20)

    # Create适配器
    api_key = os.getenv("TREE_LLM_API_KEY")
    base_url = os.getenv("TREE_LLM_BASE_URL", "https://api.siliconflow.cn/v1")

    # ⭐ 使用Qwen3-8B作为主模型
    main_model = "Qwen/Qwen3-8B"
    judge_model = "Qwen/Qwen2.5-72B-Instruct"

    if not api_key:
        logger.error("❌ 请Set TREE_LLM_API_KEY")
        return

    main_adapter = OpenAIAdapter(model=main_model, api_key=api_key, base_url=base_url)
    judge_adapter = OpenAIAdapter(model=judge_model, api_key=api_key, base_url=base_url)

    logger.info(f"\n✅ API适配器Create完成")
    logger.info(f"   主模型: {main_model}")
    logger.info(f"   Judge模型: {judge_model}")

    # Create初始skill树 - 使用很差的prompt
    poor_prompt = """Classify papers into categories. Return A, E, G, K, or M."""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=poor_prompt,
        version="v1.0",
        meta=SkillMeta(name="paper-classifier", description="Paper classifier"),
    )

    tree = SkillTree(
        root=SkillNode(name="root", skill=root_skill),
        base_path=Path("demo/outputs/demo-qwen3-8b/"),
    )

    logger.info(f"\n⚠️  使用很差的初始prompt")
    logger.info(f"   Prompt: {poor_prompt}")

    # 评估基准
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 第0轮: 基准准确率")
    logger.info(f"{'='*60}")

    initial_accuracy = evaluate(main_adapter, poor_prompt, test_data)
    logger.info(f"\n📊 基准准确率: {initial_accuracy*100:.1f}%")
    logger.info(f"   预期: Qwen3-8B应该比14B低，有更大优化空间")

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
        logger.info(f"🔄 第{round_num}轮: 使用数据{start_idx}-{end_idx}")
        logger.info(f"{'='*60}")

        # 收集经验
        experiences, train_acc = collect_experiences(
            main_adapter,
            round_data,
            tree.root.skill.system_prompt,
            temperature=0.3 + round_num * 0.05,
        )

        # 优化
        tree = optimize_round(judge_adapter, tree, experiences, f"第{round_num}轮优化")

        # 评估
        test_accuracy = evaluate(main_adapter, tree.root.skill.system_prompt, test_data)
        improvement = (test_accuracy - accuracy_history[-1]) * 100
        logger.info(f"\n📊 第{round_num}轮test准确率: {test_accuracy*100:.1f}% ({improvement:+.1f}%)")

        accuracy_history.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_tree = tree
            logger.info(f"   🎯 新最佳!")

    # 总结
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Qwen3-8B 多轮Optimization complete")
    logger.info(f"{'='*60}")

    logger.info(f"\n📈 准确率变化:")
    for i, acc in enumerate(accuracy_history):
        if i == 0:
            logger.info(f"   基准: {acc*100:.1f}%")
        else:
            improvement = (acc - accuracy_history[i-1]) * 100
            logger.info(f"   第{i}轮: {acc*100:.1f}% ({improvement:+.1f}%)")

    total_improvement = (best_accuracy - initial_accuracy) * 100
    relative_improvement = (best_accuracy / initial_accuracy - 1) * 100 if initial_accuracy > 0 else 0

    logger.info(f"\n✅ 最终results:")
    logger.info(f"   初始: {initial_accuracy*100:.1f}%")
    logger.info(f"   最终: {best_accuracy*100:.1f}%")
    logger.info(f"   Total improvement: {total_improvement:+.1f}% (绝对)")
    logger.info(f"   相对提升: {relative_improvement:+.1f}%")

    # Save
    output_path = Path("demo/outputs/demo-qwen3-8b/")
    best_tree.save(output_path)
    logger.info(f"\n💾 已Save到: {output_path}")

    # 显示After optimization的prompt
    logger.info(f"\n📝 After optimization的prompt:")
    logger.info(f"\n{tree.root.skill.system_prompt[:500]}...")

    logger.info(f"\n✅ Demo完成!")


if __name__ == "__main__":
    main()
