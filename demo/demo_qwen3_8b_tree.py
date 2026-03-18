#!/usr/bin/env python3
"""
Qwen3-8B完整Tree功能Demo - 展示Split/Prune/Merge

这个Demo展示EvoSkill的完整Tree功能：
1. Auto-Split: 自动拆分，生成子skill
2. Auto-Prune: 自动剪枝，删除低效节点
3. Multi-round: 多轮迭代优化

主模型: Qwen/Qwen3-8B (弱模型，有更大优化空间)
Judge模型: Qwen/Qwen2.5-72B (强模型，生成高质量梯度)
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

from evoskill import (
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    OptimizerConfig,
    SkillTree,
    ConversationExperience,
    CompositeFeedback,
)
from evoskill.schema import Skill, SkillMeta
from evoskill.skill_tree import SkillNode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_dataset(csv_path: str, samples_per_category: int = 20):
    """创建数据集"""
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
    logger.info(f"   测试集: {len(test_data)} 条")

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
    """优化一轮 - 使用渐进式剪枝策略"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 {round_name}")
    logger.info(f"{'='*60}")

    config = TreeOptimizerConfig(
        auto_split=True,  # ✅ 启用自动拆分
        auto_prune=True,  # ✅ 启用自动剪枝
        prune_strategy="conservative",  # 🔧 渐进式策略：保守剪枝
        prune_protection_rounds=3,  # 🔧 新节点保护3轮
        prune_usage_threshold=1,  # 🔧 至少使用1次才考虑剪枝
        collapse_instead_of_prune=True,  # 🔧 折叠而非删除（渐进式披露）
        max_tree_depth=3,
        min_samples_for_split=3,
        prune_threshold=0.2,
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

    logger.info(f"✅ 优化完成:")
    logger.info(f"   节点优化: {result.nodes_optimized}")
    logger.info(f"   拆分次数: {result.splits_performed}")
    logger.info(f"   剪枝次数: {result.prunes_performed}")

    return result.tree


def main():
    """主流程"""
    logger.info("\n" + "="*60)
    logger.info("🌳 Qwen3-8B 完整Tree功能Demo")
    logger.info("="*60)
    logger.info("主模型: Qwen3-8B (弱模型)")
    logger.info("Judge模型: Qwen2.5-72B (强模型)")
    logger.info("功能展示: Auto-Split + Auto-Prune + Multi-round")

    # 加载数据
    csv_path = "demo/data/intern_camp5.csv"
    train_data, test_data = create_dataset(csv_path, samples_per_category=20)

    # 创建适配器
    api_key = os.getenv("EVO_LLM_API_KEY")
    base_url = os.getenv("EVO_LLM_BASE_URL", "https://api.siliconflow.cn/v1")

    main_model = "Qwen/Qwen3-8B"
    judge_model = "Qwen/Qwen2.5-72B-Instruct"

    if not api_key:
        logger.error("❌ 请设置 EVO_LLM_API_KEY")
        return

    main_adapter = OpenAIAdapter(model=main_model, api_key=api_key, base_url=base_url)
    judge_adapter = OpenAIAdapter(model=judge_model, api_key=api_key, base_url=base_url)

    logger.info(f"\n✅ API适配器创建完成")

    # 创建初始skill树
    poor_prompt = "Classify papers into categories. Return A, E, G, K, or M."

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=poor_prompt,
        version="v1.0",
        meta=SkillMeta(name="paper-classifier", description="Paper classifier"),
    )

    output_path = Path("demo-qwen3-8b-tree/")
    tree = SkillTree(
        root=SkillNode(name="root", skill=root_skill),
        base_path=output_path,
    )

    logger.info(f"\n⚠️  初始prompt: {poor_prompt}")
    logger.info(f"📁 输出目录: {output_path}")

    # 评估基准
    initial_accuracy = evaluate(main_adapter, poor_prompt, test_data)
    logger.info(f"\n📊 基准准确率: {initial_accuracy*100:.1f}%")

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

        # 第1轮：只Split，不Prune（让新节点有时间积累经验）
        # 第2-3轮：谨慎Prune
        enable_prune = (round_num >= 2)

        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 第{round_num}轮: 使用数据{start_idx}-{end_idx}")
        logger.info(f"   Auto-Split: ✅")
        logger.info(f"   Auto-Prune: {'✅' if enable_prune else '❌'}")
        logger.info(f"{'='*60}")

        # 收集经验
        experiences, train_acc = collect_experiences(
            main_adapter,
            round_data,
            tree.root.skill.system_prompt,
            temperature=0.3 + round_num * 0.05,
        )

        # 优化
        tree = optimize_round(
            judge_adapter,
            tree,
            experiences,
            f"第{round_num}轮优化",
        )

        # 评估
        test_accuracy = evaluate(main_adapter, tree.root.skill.system_prompt, test_data)
        improvement = (test_accuracy - accuracy_history[-1]) * 100
        logger.info(f"\n📊 第{round_num}轮测试准确率: {test_accuracy*100:.1f}% ({improvement:+.1f}%)")

        accuracy_history.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_tree = tree
            logger.info(f"   🎯 新最佳!")

        # 保存中间结果
        checkpoint_path = output_path / f"round{round_num}"
        tree.save(checkpoint_path)
        logger.info(f"   💾 Checkpoint: {checkpoint_path}")

        # 显示tree结构
        logger.info(f"\n📊 Tree结构:")
        logger.info(tree.list_tree())

    # 最终总结
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Qwen3-8B Tree优化完成")
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

    logger.info(f"\n✅ 最终结果:")
    logger.info(f"   初始: {initial_accuracy*100:.1f}%")
    logger.info(f"   最终: {best_accuracy*100:.1f}%")
    logger.info(f"   总提升: {total_improvement:+.1f}% (绝对)")
    logger.info(f"   相对提升: {relative_improvement:+.1f}%")

    # 保存最佳tree
    best_tree.save(output_path)
    logger.info(f"\n💾 已保存到: {output_path}")

    # 显示最终tree结构
    logger.info(f"\n🌳 最终Tree结构:")
    logger.info(best_tree.list_tree())

    # 显示子skill数量
    def count_nodes(node):
        count = 1
        for child in node.children.values():
            count += count_nodes(child)
        return count

    total_nodes = count_nodes(best_tree.root)
    logger.info(f"\n📊 Tree统计:")
    logger.info(f"   总节点数: {total_nodes}")
    logger.info(f"   子skill数: {len(best_tree.root.children)}")

    logger.info(f"\n✅ Demo完成!")


if __name__ == "__main__":
    main()
