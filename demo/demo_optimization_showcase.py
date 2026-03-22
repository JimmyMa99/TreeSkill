#!/usr/bin/env python3
"""
展示完整优化能力 Demo - 从差prompt开始，让系统自己优化和拆分

策略:
1. 使用很差的初始prompt（预期准确率很低）
2. 第1轮：基础优化（大幅提升准确率）
3. 第2轮：自动拆分（生成子skill）
4. 保存最终结果

目标: 展示 0% → 60%+ 的优化能力 + 自动拆分
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
    """创建平衡数据集"""
    logger.info(f"📂 加载并创建平衡数据集: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_data = list(reader)

    if categories is None:
        label_counts = Counter(item['answer'] for item in all_data)
        categories = [label for label, _ in label_counts.most_common(5)]

    logger.info(f"✅ 选择类别: {categories}")

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

    logger.info(f"✅ 总计: {len(balanced_data)} 条平衡数据")

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
                metadata={"paper_id": idx, "skill_name": "paper-classifier"},
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
        min_samples_for_split=3,
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

    if result.tree.root.children:
        logger.info(f"\n🌲 当前子skill:")
        for child_name, child_node in result.tree.root.children.items():
            logger.info(f"   - {child_name}")

    return result.tree, result.splits_performed, result.prunes_performed


def main():
    """主流程"""
    logger.info("\n" + "="*60)
    logger.info("🎯 展示完整优化能力 Demo")
    logger.info("="*60)
    logger.info("展示: 差prompt → 优化 → 拆分 → 完整树结构")

    # Step 1: 创建平衡数据集
    csv_path = "demo/data/intern_camp5.csv"

    selected_categories = ['A', 'E', 'G', 'K', 'M']
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

    # Step 3: 创建故意很差的初始skill树
    # ⭐ 故意写得模糊、不完整、没有examples！
    poor_initial_prompt = """Classify the paper into one of these categories:

A - something about physics
E - robotics stuff
G - software
K - math
M - machine learning

Just return a letter."""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=poor_initial_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="Multi-category paper classifier",
        ),
    )

    tree = SkillTree(
        root=SkillNode(name="root", skill=root_skill),
        base_path=Path("demo-optimization-showcase/"),
    )

    logger.info(f"\n✅ 初始skill树创建完成")
    logger.info(f"\n⚠️  使用故意很差的初始prompt（预期准确率很低）")

    # Step 4: 评估初始准确率
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 第0轮: 评估初始准确率（基准线）")
    logger.info(f"{'='*60}")

    initial_accuracy = evaluate(main_adapter, poor_initial_prompt, test_data)
    logger.info(f"\n📊 初始准确率: {initial_accuracy*100:.1f}%")
    logger.info(f"   预期: 应该很低（20-40%），因为prompt很差")

    best_accuracy = initial_accuracy
    best_tree = tree

    # ========================================================================
    # 第1轮：基础优化（大幅提升准确率）
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 第1轮: 基础优化")
    logger.info(f"{'='*60}")
    logger.info(f"目标: 通过TGD优化，大幅提升准确率")

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
    improvement1 = (accuracy_round1 - initial_accuracy) * 100
    logger.info(f"\n📊 第1轮准确率: {accuracy_round1*100:.1f}% (提升: {improvement1:+.1f}%)")

    if accuracy_round1 > best_accuracy:
        best_accuracy = accuracy_round1
        best_tree = tree_round1

    # ========================================================================
    # 第2轮：自动拆分（生成子skill）
    # ========================================================================
    if accuracy_round1 > 0.3:  # 如果第1轮有提升
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 第2轮: 自动拆分")
        logger.info(f"{'='*60}")
        logger.info(f"目标: 检测矛盾反馈，拆分为专门的子skill")

        exp_round2 = collect_experiences(
            main_adapter, train_data, tree_round1.root.skill.system_prompt, temperature=0.5
        )

        tree_round2, splits2, prunes2 = run_optimization_round(
            judge_adapter, tree_round1, exp_round2,
            auto_split=True,
            auto_prune=False,
            round_name="第2轮拆分",
        )

        accuracy_round2 = evaluate(main_adapter, tree_round2.root.skill.system_prompt, test_data)
        improvement2 = (accuracy_round2 - accuracy_round1) * 100
        logger.info(f"\n📊 第2轮准确率: {accuracy_round2*100:.1f}% (提升: {improvement2:+.1f}%)")
        logger.info(f"   拆分次数: {splits2}")

        if tree_round2.root.children:
            logger.info(f"   ✨ 成功生成 {len(tree_round2.root.children)} 个子skill!")

            # 显示每个子skill的详情
            logger.info(f"\n🌲 子skill详情:")
            for child_name, child_node in tree_round2.root.children.items():
                prompt_preview = child_node.skill.system_prompt[:100]
                logger.info(f"   - {child_name}:")
                logger.info(f"     Prompt预览: {prompt_preview}...")

        if accuracy_round2 > best_accuracy:
            best_accuracy = accuracy_round2
            best_tree = tree_round2

    # ========================================================================
    # 保存最终结果
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 最终结果")
    logger.info(f"{'='*60}")
    logger.info(f"初始准确率: {initial_accuracy*100:.1f}%")
    logger.info(f"最终准确率: {best_accuracy*100:.1f}%")
    logger.info(f"总提升: {(best_accuracy-initial_accuracy)*100:+.1f}%")

    output_path = Path("demo-optimization-showcase/")
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
        logger.info(f"\n✨ 成功生成 {len(best_tree.root.children)} 个子skill文件夹:")
        for child_name, child_node in best_tree.root.children.items():
            child_dir = output_path / child_name
            if child_dir.exists():
                logger.info(f"   ✅ {child_name}/")

    logger.info(f"\n✅ Demo完成! 查看输出: {output_path}")
    logger.info(f"\n💡 优化效果: {initial_accuracy*100:.1f}% → {best_accuracy*100:.1f}% (提升 {(best_accuracy-initial_accuracy)*100:+.1f}%)")


if __name__ == "__main__":
    main()
