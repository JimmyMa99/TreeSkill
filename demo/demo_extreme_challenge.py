#!/usr/bin/env python3
"""
极限优化挑战 Demo - 26个类别，极差的初始prompt

策略:
1. 使用所有26个类别（A-Z）
2. 极差的初始prompt（几乎无用）
3. 第1轮：优化（预期从20%提升到40%+）
4. 第2轮：拆分（生成子skill）
5. 第3轮：优化子skill

目标: 展示完整的优化和拆分能力
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


def load_full_dataset(csv_path: str, train_size: int = 150, test_size: int = 50):
    """加载完整数据集（所有26个类别）"""
    logger.info(f"📂 加载完整数据集（26个类别）: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # 统计标签
    labels = [item['answer'] for item in data]
    label_counts = Counter(labels)
    logger.info(f"✅ 数据总量: {len(data)} 条")
    logger.info(f"📊 类别数: {len(label_counts)}")
    logger.info(f"   类别分布: {label_counts.most_common(26)}")

    # 随机打乱
    random.seed(42)
    random.shuffle(data)

    # 分割
    train = data[:train_size]
    test = data[train_size:train_size + test_size]

    logger.info(f"\n✅ 数据分割完成:")
    logger.info(f"   训练集: {len(train)} 条")
    logger.info(f"   测试集: {len(test)} 条")

    return train, test, label_counts


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
            {"role": "user", "content": f"Classify:\n\n{question[:400]}\n\nReturn ONLY the letter."},
        ]

        try:
            predicted = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature,
            ).strip().upper()

            # 提取第一个字母
            if predicted:
                predicted = predicted[0]

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
            {"role": "user", "content": f"Classify:\n\n{question[:400]}\n\nReturn ONLY the letter."},
        ]

        try:
            predicted = adapter._call_api(
                messages=messages,
                system=None,
                temperature=0.3,
            ).strip().upper()

            if predicted:
                predicted = predicted[0]

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
        min_samples_for_split=5,
        prune_threshold=0.4,
        optimization_order="bottom_up",
    )

    base_config = OptimizerConfig(
        max_steps=1,
        gradient_accumulation_steps=10,  # 更多样本用于优化
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
    logger.info("🔥 极限优化挑战 Demo")
    logger.info("="*60)
    logger.info("挑战: 26个类别 + 极差prompt → 优化 → 拆分")

    # Step 1: 加载完整数据集
    csv_path = "demo/data/intern_camp5.csv"
    train_data, test_data, label_counts = load_full_dataset(csv_path, train_size=150, test_size=50)

    # Step 2: 创建API适配器
    api_key = os.getenv("TREE_LLM_API_KEY")
    base_url = os.getenv("TREE_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    main_model = os.getenv("TREE_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    judge_model = os.getenv("TREE_LLM_JUDGE_MODEL", "doubao/seed-2-0-flash")

    if not api_key:
        logger.error("❌ 请设置 TREE_LLM_API_KEY")
        return

    main_adapter = OpenAIAdapter(model=main_model, api_key=api_key, base_url=base_url)
    judge_adapter = OpenAIAdapter(model=judge_model, api_key=api_key, base_url=base_url)

    logger.info(f"\n✅ API适配器创建完成")
    logger.info(f"   主模型: {main_model}")
    logger.info(f"   Judge模型: {judge_model}")

    # Step 3: 创建极差的初始skill树
    # ⭐⭐⭐ 极其模糊的prompt！只给字母，不给含义！⭐⭐⭐
    terrible_prompt = """Classify this paper. Choose from A-Z. Return only one letter."""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=terrible_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="26-category paper classifier",
        ),
    )

    tree = SkillTree(
        root=SkillNode(name="root", skill=root_skill),
        base_path=Path("demo-extreme-challenge/"),
    )

    logger.info(f"\n✅ 初始skill树创建完成")
    logger.info(f"\n⚠️  使用极差的初始prompt（几乎无信息）")
    logger.info(f"   Prompt: {terrible_prompt}")

    # Step 4: 评估初始准确率
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 第0轮: 评估初始准确率（基准线）")
    logger.info(f"{'='*60}")

    initial_accuracy = evaluate(main_adapter, terrible_prompt, test_data)
    logger.info(f"\n📊 初始准确率: {initial_accuracy*100:.1f}%")
    logger.info(f"   预期: 应该很低（3-10%），因为26个类别 + 无信息prompt")

    best_accuracy = initial_accuracy
    best_tree = tree

    # ========================================================================
    # 第1轮：基础优化（尝试提升准确率）
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 第1轮: 基础优化")
    logger.info(f"{'='*60}")
    logger.info(f"目标: 让模型学会至少部分类别的分类")

    exp_round1 = collect_experiences(
        main_adapter, train_data[:80], tree.root.skill.system_prompt, temperature=0.3
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
    # 第2轮：继续优化 + 自动拆分
    # ========================================================================
    if accuracy_round1 > 0.1:  # 如果第1轮有提升
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 第2轮: 继续优化 + 自动拆分")
        logger.info(f"{'='*60}")
        logger.info(f"目标: 进一步优化，检测可拆分的类别")

        exp_round2 = collect_experiences(
            main_adapter, train_data[80:], tree_round1.root.skill.system_prompt, temperature=0.5
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

    output_path = Path("demo-extreme-challenge/")
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
        logger.info(f"\n✨ 成功生成 {len(best_tree.root.children)} 个子skill文件夹!")

    logger.info(f"\n✅ Demo完成! 查看输出: {output_path}")
    logger.info(f"\n💡 优化效果: {initial_accuracy*100:.1f}% → {best_accuracy*100:.1f}% (提升 {(best_accuracy-initial_accuracy)*100:+.1f}%)")


if __name__ == "__main__":
    main()
