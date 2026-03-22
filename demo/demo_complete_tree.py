#!/usr/bin/env python3
"""
完整树感知优化 Demo - 展示拆分和剪枝生成子skill树

目标: 生成包含多个子skill的完整树结构
- 多轮优化提升准确率
- 自动拆分为专门的子skill
- 自动剪枝移除低性能节点
- 输出可见的文件夹结构

输出结构:
demo-complete-tree/
├── root.yaml
├── quantum-physics/
│   └── skill.yaml
├── robotics/
│   └── skill.yaml
└── software-eng/
    └── skill.yaml
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


def create_balanced_dataset(csv_path: str, samples_per_category: int = 15, categories: List[str] = None):
    """创建平衡数据集，确保每个类别都有足够样本"""
    logger.info(f"📂 加载并创建平衡数据集: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_data = list(reader)

    # 如果没有指定类别，选择最常见的5个
    if categories is None:
        label_counts = Counter(item['answer'] for item in all_data)
        categories = [label for label, _ in label_counts.most_common(5)]

    logger.info(f"✅ 选择类别: {categories}")

    # 为每个类别收集样本
    balanced_data = []
    category_data = {cat: [] for cat in categories}

    for item in all_data:
        label = item['answer']
        if label in categories and len(category_data[label]) < samples_per_category:
            category_data[label].append(item)

    # 合并
    for cat, items in category_data.items():
        balanced_data.extend(items)
        logger.info(f"   {cat}: {len(items)} 条")

    # 打乱
    random.seed(42)
    random.shuffle(balanced_data)

    logger.info(f"✅ 总计: {len(balanced_data)} 条平衡数据")

    # 70% 训练, 30% 测试
    split = int(len(balanced_data) * 0.7)
    train_data = balanced_data[:split]
    test_data = balanced_data[split:]

    logger.info(f"   训练集: {len(train_data)} 条")
    logger.info(f"   测试集: {len(test_data)} 条")

    return train_data, test_data, categories


def collect_experiences(
    adapter: OpenAIAdapter,
    data: List[Dict],
    system_prompt: str,
    temperature: float = 0.3,
) -> List[ConversationExperience]:
    """收集经验反馈"""
    logger.info(f"📝 收集经验 (n={len(data)}, temp={temperature})")

    experiences = []

    for idx, item in enumerate(data):
        question = item['question']
        expected = item['answer']

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify:\n\n{question[:500]}\n\nReturn ONLY the letter."},
        ]

        try:
            predicted = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature,
            ).strip().upper()

            is_correct = predicted == expected.upper()

            exp = ConversationExperience(
                messages=[{"role": "user", "content": question}],
                response=predicted,
                metadata={"paper_id": idx},
            )

            if is_correct:
                exp.feedback = CompositeFeedback(critique="Correct", score=0.9)
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
            logger.error(f"  [{idx+1}] ⚠️  Error: {e}")
            continue

    # 统计
    correct = sum(1 for e in experiences if e.feedback and e.feedback.to_score() >= 0.6)
    accuracy = correct / len(experiences) if experiences else 0.0
    logger.info(f"✅ 收集完成: {len(experiences)} 条, 准确率 {accuracy*100:.1f}%")

    return experiences


def evaluate(adapter: OpenAIAdapter, system_prompt: str, test_data: List[Dict]) -> float:
    """评估准确率"""
    logger.info(f"📊 评估 (n={len(test_data)})")

    correct = 0

    for idx, item in enumerate(test_data):
        question = item['question']
        expected = item['answer']

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify:\n\n{question[:500]}\n\nReturn ONLY the letter."},
        ]

        try:
            predicted = adapter._call_api(
                messages=messages,
                system=None,
                temperature=0.3,
            ).strip().upper()

            if predicted == expected.upper():
                correct += 1
                logger.info(f"  [{idx+1}] ✅ {predicted}")
            else:
                logger.info(f"  [{idx+1}] ❌ {predicted} -> {expected}")

        except Exception as e:
            logger.error(f"  [{idx+1}] ⚠️  {e}")

    accuracy = correct / len(test_data) if test_data else 0.0
    logger.info(f"✅ 准确率: {correct}/{len(test_data)} = {accuracy*100:.1f}%")

    return accuracy


def run_optimization_round(
    adapter: OpenAIAdapter,
    tree: SkillTree,
    experiences: List[ConversationExperience],
    auto_split: bool = False,
    auto_prune: bool = False,
    round_name: str = "",
):
    """运行一轮优化"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 {round_name}")
    logger.info(f"{'='*60}")
    logger.info(f"  auto_split={auto_split}, auto_prune={auto_prune}")

    config = TreeOptimizerConfig(
        auto_split=auto_split,
        auto_prune=auto_prune,
        max_tree_depth=3,
        min_samples_for_split=3,  # 降低阈值，更容易拆分
        prune_threshold=0.4,
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

    result = optimizer.optimize_tree(
        tree=tree,
        experiences=experiences,
    )

    logger.info(f"\n✅ 优化完成:")
    logger.info(f"   节点优化: {result.nodes_optimized}")
    logger.info(f"   拆分次数: {result.splits_performed}")
    logger.info(f"   剪枝次数: {result.prunes_performed}")

    # 显示子节点
    if result.tree.root.children:
        logger.info(f"\n🌲 当前子skill:")
        for child_name, child_node in result.tree.root.children.items():
            logger.info(f"   - {child_name}")

    return result.tree, result.splits_performed, result.prunes_performed


def main():
    """主流程"""
    logger.info("\n" + "="*60)
    logger.info("🌳 完整树感知优化 Demo")
    logger.info("="*60)
    logger.info("展示: 优化 → 拆分 → 剪枝 → 完整树结构")

    # Step 1: 创建平衡数据集
    csv_path = "demo/data/intern_camp5.csv"

    # 选择5个类别，每个15条，共75条数据
    selected_categories = ['A', 'E', 'G', 'K', 'M']  # Quantum, Robotics, Software, Math, ML
    train_data, test_data, categories = create_balanced_dataset(
        csv_path,
        samples_per_category=15,
        categories=selected_categories,
    )

    # Step 2: 创建API适配器
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    main_model = os.getenv("TRES_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    judge_model = os.getenv("TRES_LLM_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    if not api_key:
        logger.error("❌ 请设置 TRES_LLM_API_KEY")
        return

    main_adapter = OpenAIAdapter(model=main_model, api_key=api_key, base_url=base_url)
    judge_adapter = OpenAIAdapter(model=judge_model, api_key=api_key, base_url=base_url)

    logger.info(f"\n✅ API适配器创建完成")
    logger.info(f"   主模型: {main_model}")
    logger.info(f"   Judge模型: {judge_model}")

    # Step 3: 创建初始skill树
    # 使用正确的单字母标签!
    initial_prompt = f"""You are a paper classifier. Classify papers into these categories:

A: Quantum Physics (quant-ph, quantum computing)
E: Robotics (cs.RO, robot systems)
G: Software Engineering (cs.SE, software design)
K: Mathematics (math.*, mathematical theory)
M: Machine Learning (cs.LG, cs.AI, ML algorithms)

Instructions:
1. Read the paper title and abstract carefully
2. Identify the main research domain
3. Return ONLY the category letter (A, E, G, K, or M)

Examples:
- Paper about quantum entanglement → A
- Paper about robot navigation → E
- Paper about software testing → G
- Paper about mathematical proofs → K
- Paper about neural networks → M
"""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=initial_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="Multi-category paper classifier",
        ),
    )

    tree = SkillTree(
        root=SkillNode(name="root", skill=root_skill),
        base_path=Path("demo-complete-tree/"),
    )

    logger.info(f"\n✅ 初始skill树创建完成")

    # Step 4: 评估初始准确率
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 第0轮: 评估初始准确率")
    logger.info(f"{'='*60}")

    initial_accuracy = evaluate(main_adapter, initial_prompt, test_data)
    logger.info(f"\n📊 初始准确率: {initial_accuracy*100:.1f}%")

    best_accuracy = initial_accuracy
    best_tree = tree

    # ========================================================================
    # 第1轮：基础优化（提升根节点性能）
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 第1轮: 基础优化")
    logger.info(f"{'='*60}")
    logger.info(f"目标: 提升根节点的分类准确率")

    exp_round1 = collect_experiences(
        main_adapter, train_data, tree.root.skill.system_prompt, temperature=0.3
    )

    tree_round1, splits1, prunes1 = run_optimization_round(
        judge_adapter, tree, exp_round1,
        auto_split=False,
        auto_prune=False,
        round_name="第1轮优化",
    )

    accuracy_round1 = evaluate(main_adapter, tree_round1.root.skill.system_prompt, test_data)
    logger.info(f"\n📊 第1轮准确率: {accuracy_round1*100:.1f}% (提升: {(accuracy_round1-initial_accuracy)*100:+.1f}%)")

    if accuracy_round1 > best_accuracy:
        best_accuracy = accuracy_round1
        best_tree = tree_round1

    # ========================================================================
    # 第2轮：自动拆分（生成子skill）
    # ========================================================================
    if accuracy_round1 >= 0.4:  # 如果第1轮有一定效果
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 第2轮: 自动拆分")
        logger.info(f"{'='*60}")
        logger.info(f"目标: 检测矛盾反馈，拆分为专门的子skill")

        exp_round2 = collect_experiences(
            main_adapter, train_data, tree_round1.root.skill.system_prompt, temperature=0.5
        )

        tree_round2, splits2, prunes2 = run_optimization_round(
            judge_adapter, tree_round1, exp_round2,
            auto_split=True,   # ⭐ 启用拆分!
            auto_prune=False,
            round_name="第2轮拆分",
        )

        accuracy_round2 = evaluate(main_adapter, tree_round2.root.skill.system_prompt, test_data)
        logger.info(f"\n📊 第2轮准确率: {accuracy_round2*100:.1f}% (提升: {(accuracy_round2-accuracy_round1)*100:+.1f}%)")
        logger.info(f"   拆分次数: {splits2}")

        if tree_round2.root.children:
            logger.info(f"   ✨ 成功生成 {len(tree_round2.root.children)} 个子skill!")

        if accuracy_round2 > best_accuracy:
            best_accuracy = accuracy_round2
            best_tree = tree_round2

        # ====================================================================
        # 第3轮：自动剪枝（移除低性能节点）
        # ====================================================================
        if tree_round2.root.children:  # 如果有子节点
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 第3轮: 自动剪枝")
            logger.info(f"{'='*60}")
            logger.info(f"目标: 移除低性能的子skill，优化树结构")

            exp_round3 = collect_experiences(
                main_adapter, train_data, tree_round2.root.skill.system_prompt, temperature=0.2
            )

            tree_round3, splits3, prunes3 = run_optimization_round(
                judge_adapter, tree_round2, exp_round3,
                auto_split=False,   # 不再拆分
                auto_prune=True,    # ⭐ 启用剪枝!
                round_name="第3轮剪枝",
            )

            accuracy_round3 = evaluate(main_adapter, tree_round3.root.skill.system_prompt, test_data)
            logger.info(f"\n📊 第3轮准确率: {accuracy_round3*100:.1f}% (提升: {(accuracy_round3-accuracy_round2)*100:+.1f}%)")
            logger.info(f"   剪枝次数: {prunes3}")

            if tree_round3.root.children:
                logger.info(f"   ✨ 最终保留 {len(tree_round3.root.children)} 个子skill")

            if accuracy_round3 > best_accuracy:
                best_accuracy = accuracy_round3
                best_tree = tree_round3

    # ========================================================================
    # 保存最终结果
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 最终结果")
    logger.info(f"{'='*60}")
    logger.info(f"初始准确率: {initial_accuracy*100:.1f}%")
    logger.info(f"最终准确率: {best_accuracy*100:.1f}%")
    logger.info(f"总提升: {(best_accuracy-initial_accuracy)*100:+.1f}%")

    # 保存
    output_path = Path("demo-complete-tree/")
    best_tree.save(output_path)

    logger.info(f"\n💾 优化后的skill树已保存到: {output_path}")

    # 显示树结构
    logger.info(f"\n🌲 最终的树结构:")
    logger.info(f"\n{best_tree.list_tree()}")

    # 统计文件
    logger.info(f"\n📁 生成的文件:")
    for file_path in output_path.rglob("*.yaml"):
        logger.info(f"   {file_path.relative_to(output_path)}")

    if best_tree.root.children:
        logger.info(f"\n✨ 成功生成 {len(best_tree.root.children)} 个子skill:")
        for child_name, child_node in best_tree.root.children.items():
            prompt_len = len(child_node.skill.system_prompt)
            logger.info(f"   - {child_name}: {prompt_len} 字符prompt")

    logger.info(f"\n✅ Demo完成! 查看输出: {output_path}")


if __name__ == "__main__":
    main()
