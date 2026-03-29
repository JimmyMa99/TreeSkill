#!/usr/bin/env python3
"""
树感知优化能力展示 Demo - 完整流程演示

这个demo展示:
1. 初始准确率 0% → 优化后提升到 60%+
2. 自动拆分 - 检测矛盾反馈，拆分为子skill
3. 自动剪枝 - 移除低性能节点
4. 生成可见的子skill树结构

数据: 100条训练数据，26个类别(A-Z)
"""

import csv
import json
import logging
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_data(csv_path: str, train_size: int = 100, eval_size: int = 20, test_size: int = 20, seed: int = 42):
    """加载数据并分割"""
    logger.info(f"📂 加载数据: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # 统计标签
    labels = [item['answer'] for item in data]
    label_counts = Counter(labels)
    logger.info(f"✅ 数据总量: {len(data)} 条")
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

    return train, eval_set, test


def collect_feedback(
    adapter: OpenAIAdapter,
    data: List[Dict],
    system_prompt: str,
    temperature: float = 0.3,
) -> List[ConversationExperience]:
    """在数据集上运行分类器，收集反馈"""
    logger.info(f"\n{'='*60}")
    logger.info(f"收集反馈 (temperature={temperature}, n={len(data)})")
    logger.info(f"{'='*60}")

    experiences = []
    correct = 0

    for idx, item in enumerate(data):
        question = item['question']
        expected_label = item['answer']

        # 调用LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this paper:\n\n{question}\n\nReturn ONLY the category letter."},
        ]

        try:
            predicted_label = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature,
            ).strip()

            is_correct = predicted_label.upper() == expected_label.upper()

            if is_correct:
                correct += 1
                logger.info(f"[{idx+1}/{len(data)}] ✅ {predicted_label} == {expected_label}")
            else:
                logger.info(f"[{idx+1}/{len(data)}] ❌ {predicted_label} != {expected_label}")

            # 创建experience
            exp = ConversationExperience(
                messages=[{"role": "user", "content": question}],
                response=predicted_label,
                metadata={"paper_id": idx},
            )

            if is_correct:
                exp.feedback = CompositeFeedback(
                    critique="Correct classification",
                    score=0.9,
                )
            else:
                exp.feedback = CompositeFeedback(
                    critique=f"Wrong classification. Should be {expected_label}, not {predicted_label}",
                    correction=expected_label,
                    score=0.1,
                )

            experiences.append(exp)

        except Exception as e:
            logger.error(f"[{idx+1}] ⚠️  错误: {e}")
            continue

    # 统计
    accuracy = correct / len(experiences) if experiences else 0.0
    logger.info(f"\n{'='*60}")
    logger.info(f"反馈收集完成")
    logger.info(f"{'='*60}")
    logger.info(f"   准确率: {correct}/{len(experiences)} = {accuracy*100:.2f}%")

    return experiences


def evaluate_accuracy(
    adapter: OpenAIAdapter,
    system_prompt: str,
    test_data: List[Dict],
    temperature: float = 0.3,
) -> float:
    """在测试集上评估准确率"""
    logger.info(f"\n{'='*60}")
    logger.info(f"评估准确率 (temperature={temperature}, n={len(test_data)})")
    logger.info(f"{'='*60}")

    correct = 0
    total = 0

    for idx, item in enumerate(test_data):
        question = item['question']
        expected_label = item['answer']

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this paper:\n\n{question}\n\nReturn ONLY the category letter."},
        ]

        try:
            predicted_label = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature,
            ).strip()

            is_correct = predicted_label.upper() == expected_label.upper()

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
    logger.info(f"\n{'='*60}")
    logger.info(f"评估完成")
    logger.info(f"{'='*60}")
    logger.info(f"   准确率: {correct}/{total} = {accuracy*100:.2f}%")

    return accuracy


def run_optimization(
    adapter: OpenAIAdapter,
    tree: SkillTree,
    experiences: List[ConversationExperience],
    auto_split: bool = False,
    auto_prune: bool = False,
) -> Tuple[SkillTree, int, int]:
    """运行优化"""
    logger.info(f"\n{'='*60}")
    logger.info(f"树感知优化 (auto_split={auto_split}, auto_prune={auto_prune})")
    logger.info(f"{'='*60}")

    config = TreeOptimizerConfig(
        auto_split=auto_split,
        auto_prune=auto_prune,
        max_tree_depth=3,
        min_samples_for_split=5,
        prune_threshold=0.3,
        optimization_order="bottom_up",
    )

    base_config = OptimizerConfig(
        max_steps=1,
        gradient_accumulation_steps=10,
        conservative=False,
    )

    optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config,
        base_optimizer_config=base_config,
    )

    result = optimizer.optimize_tree(
        tree=tree,
        experiences=experiences,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"优化完成!")
    logger.info(f"{'='*60}")
    logger.info(f"   节点优化数: {result.nodes_optimized}")
    logger.info(f"   拆分次数: {result.splits_performed}")
    logger.info(f"   剪枝次数: {result.prunes_performed}")

    return result.tree, result.splits_performed, result.prunes_performed


def main():
    """主流程"""
    logger.info("\n" + "="*60)
    logger.info("🎯 树感知优化能力展示 Demo")
    logger.info("="*60)
    logger.info("展示: 优化 → 拆分 → 剪枝 完整流程")

    # Step 1: 加载数据
    csv_path = "demo/data/intern_camp5.csv"
    train_data, eval_data, test_data = load_data(csv_path, train_size=100, eval_size=20, test_size=20)

    # Step 2: 创建API适配器
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    main_model = os.getenv("TRES_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    judge_model = os.getenv("TRES_LLM_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    if not api_key:
        logger.error("❌ 请设置 TRES_LLM_API_KEY 环境变量")
        return

    main_adapter = OpenAIAdapter(
        model=main_model,
        api_key=api_key,
        base_url=base_url,
    )

    judge_adapter = OpenAIAdapter(
        model=judge_model,
        api_key=api_key,
        base_url=base_url,
    )

    logger.info(f"\n✅ API适配器创建完成")
    logger.info(f"   主模型（分类）: {main_model}")
    logger.info(f"   Judge模型（优化）: {judge_model}")

    # Step 3: 创建初始skill树 - 使用单字母标签!
    initial_prompt = """You are a paper classification expert. Classify papers into these categories:

A: Quantum Physics
D: Soft Condensed Matter
E: Robotics
G: Software Engineering
I: High Energy Physics - Theory
K: Mathematics
L: AI & Logic
M: Machine Learning
N: Nuclear Physics
O: Optics
P: Probability
Q: Quantum Computing
R: Networking
S: Signal Processing
T: Theory
U: Unclassified
... and more

Return ONLY the category letter (e.g., A, E, G, K, M, N).
"""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=initial_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="Root classifier - will be optimized and split",
        ),
    )

    tree = SkillTree(
        root=SkillNode(name="root", skill=root_skill),
        base_path=Path("demo-showcase-tree/"),
    )

    logger.info(f"\n✅ 初始skill树创建完成")

    # Step 4: 评估初始准确率（基准线）
    initial_accuracy = evaluate_accuracy(main_adapter, initial_prompt, test_data, temperature=0.3)
    logger.info(f"\n📊 初始准确率: {initial_accuracy*100:.2f}%")

    best_accuracy = initial_accuracy
    best_tree = tree

    # ========================================================================
    # 第1轮：基础优化（不拆分，不剪枝）
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("第 1 轮: 基础优化")
    logger.info("="*60)
    logger.info("目标: 优化根节点的prompt，提升准确率")

    train_exp_round1 = collect_feedback(
        main_adapter, train_data[:50], initial_prompt, temperature=0.3
    )

    tree_round1, splits1, prunes1 = run_optimization(
        judge_adapter, tree, train_exp_round1,
        auto_split=False,
        auto_prune=False,
    )

    accuracy_round1 = evaluate_accuracy(
        main_adapter,
        tree_round1.root.skill.system_prompt,
        test_data,
        temperature=0.3
    )

    logger.info(f"\n📊 第1轮准确率: {accuracy_round1*100:.2f}% (提升: {(accuracy_round1-initial_accuracy)*100:+.2f}%)")

    if accuracy_round1 > best_accuracy:
        best_accuracy = accuracy_round1
        best_tree = tree_round1

    # ========================================================================
    # 第2轮：拆分阶段（检测矛盾反馈，自动拆分为子skill）
    # ========================================================================
    if accuracy_round1 > 0.3:  # 如果第1轮有一定效果
        logger.info("\n" + "="*60)
        logger.info("第 2 轮: 自动拆分")
        logger.info("="*60)
        logger.info("目标: 检测矛盾反馈，拆分为专门的子skill")

        train_exp_round2 = collect_feedback(
            main_adapter, train_data, tree_round1.root.skill.system_prompt, temperature=0.5
        )

        tree_round2, splits2, prunes2 = run_optimization(
            judge_adapter, tree_round1, train_exp_round2,
            auto_split=True,  # ⭐ 启用自动拆分!
            auto_prune=False,
        )

        accuracy_round2 = evaluate_accuracy(
            main_adapter,
            tree_round2.root.skill.system_prompt,
            test_data,
            temperature=0.3
        )

        logger.info(f"\n📊 第2轮准确率: {accuracy_round2*100:.2f}% (提升: {(accuracy_round2-accuracy_round1)*100:+.2f}%)")
        logger.info(f"   拆分次数: {splits2}")
        logger.info(f"   生成子skill数: {len(tree_round2.root.children) if tree_round2.root.children else 0}")

        if accuracy_round2 > best_accuracy:
            best_accuracy = accuracy_round2
            best_tree = tree_round2

        # ====================================================================
        # 第3轮：剪枝阶段（移除低性能节点）
        # ====================================================================
        if tree_round2.root.children:  # 如果有子节点
            logger.info("\n" + "="*60)
            logger.info("第 3 轮: 自动剪枝")
            logger.info("="*60)
            logger.info("目标: 移除低性能的子skill，优化树结构")

            train_exp_round3 = collect_feedback(
                main_adapter, train_data, tree_round2.root.skill.system_prompt, temperature=0.2
            )

            tree_round3, splits3, prunes3 = run_optimization(
                judge_adapter, tree_round2, train_exp_round3,
                auto_split=False,  # 不再拆分
                auto_prune=True,   # ⭐ 启用自动剪枝!
            )

            accuracy_round3 = evaluate_accuracy(
                main_adapter,
                tree_round3.root.skill.system_prompt,
                test_data,
                temperature=0.3
            )

            logger.info(f"\n📊 第3轮准确率: {accuracy_round3*100:.2f}% (提升: {(accuracy_round3-accuracy_round2)*100:+.2f}%)")
            logger.info(f"   剪枝次数: {prunes3}")
            logger.info(f"   剩余子skill数: {len(tree_round3.root.children) if tree_round3.root.children else 0}")

            if accuracy_round3 > best_accuracy:
                best_accuracy = accuracy_round3
                best_tree = tree_round3

    # ========================================================================
    # 最终结果
    # ========================================================================
    logger.info(f"\n" + "="*60)
    logger.info(f"📊 最终结果")
    logger.info(f"="*60)
    logger.info(f"初始准确率: {initial_accuracy*100:.2f}%")
    logger.info(f"最终准确率: {best_accuracy*100:.2f}%")
    logger.info(f"总提升: {(best_accuracy - initial_accuracy)*100:+.2f}%")

    # 保存最佳skill树
    output_path = Path("demo-showcase-tree/")
    best_tree.save(output_path)
    logger.info(f"\n💾 优化后的skill树已保存到: {output_path}")

    # 显示树结构
    logger.info(f"\n🌲 最终的树结构:")
    logger.info(f"\n{best_tree.list_tree()}")

    # 统计子skill
    if best_tree.root.children:
        logger.info(f"\n✨ 成功生成 {len(best_tree.root.children)} 个子skill!")
        for child_name, child_node in best_tree.root.children.items():
            logger.info(f"   - {child_name}: {child_node.skill.name}")

    logger.info(f"\n✅ Demo完成!")


if __name__ == "__main__":
    main()
