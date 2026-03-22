#!/usr/bin/env python3
"""
多轮迭代优化 Demo - 从已有skill树继续优化

策略:
1. 加载已有的skill树（demo-split-showcase）
2. 进行5轮迭代优化
3. 每轮: 收集经验 → 优化 → 评估
4. 观察准确率持续提升
5. 保存最终结果
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


def create_dataset(csv_path: str, samples_per_category: int = 30, categories: List[str] = None):
    """创建更大的数据集，支持多轮训练"""
    logger.info(f"📂 加载数据集: {csv_path}")

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

    logger.info(f"✅ 总计: {len(balanced_data)} 条数据")

    # 60% 训练, 40% 测试
    split = int(len(balanced_data) * 0.6)
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

    return result.tree, result.splits_performed, result.prunes_performed


def main():
    """主流程 - 多轮迭代优化"""
    logger.info("\n" + "="*60)
    logger.info("🔄 多轮迭代优化 Demo")
    logger.info("="*60)
    logger.info("策略: 加载已有skill树 → 多轮迭代优化 → 观察持续提升")

    # Step 1: 创建更大的数据集
    csv_path = "demo/data/intern_camp5.csv"
    selected_categories = ['A', 'E', 'G', 'K', 'M']
    train_data, test_data, categories = create_dataset(
        csv_path,
        samples_per_category=30,  # 每个类别30条，支持多轮训练
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

    # Step 3: 加载已有的skill树（如果存在）
    existing_tree_path = Path("demo-split-showcase/")
    output_path = Path("demo-multi-round/")

    if existing_tree_path.exists():
        logger.info(f"\n📂 加载已有skill树: {existing_tree_path}")
        tree = SkillTree.load(existing_tree_path)
        logger.info(f"✅ 成功加载skill树")
        logger.info(f"\n🌲 当前树结构:")
        logger.info(f"\n{tree.list_tree()}")
    else:
        logger.info(f"\n⚠️  未找到已有skill树，创建新的")

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
            base_path=output_path,
        )

    # Step 4: 评估初始准确率
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 第0轮: 评估基准准确率")
    logger.info(f"{'='*60}")

    initial_accuracy = evaluate(main_adapter, tree.root.skill.system_prompt, test_data)
    logger.info(f"\n📊 基准准确率: {initial_accuracy*100:.1f}%")

    # 跟踪最佳结果
    best_accuracy = initial_accuracy
    best_tree = tree
    accuracy_history = [initial_accuracy]

    # ========================================================================
    # 多轮迭代优化
    # ========================================================================
    num_rounds = 5
    samples_per_round = len(train_data) // num_rounds

    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 第{round_num}轮: 迭代优化")
        logger.info(f"{'='*60}")

        # 使用不同的数据子集
        start_idx = (round_num - 1) * samples_per_round
        end_idx = start_idx + samples_per_round
        round_data = train_data[start_idx:end_idx]

        logger.info(f"   使用数据: {start_idx}-{end_idx} (共{len(round_data)}条)")

        # 收集经验
        exp_round = collect_experiences(
            main_adapter,
            round_data,
            tree.root.skill.system_prompt,
            temperature=0.3 + round_num * 0.05,  # 逐渐增加温度
        )

        # 优化
        tree, splits, prunes = run_optimization_round(
            judge_adapter,
            tree,
            exp_round,
            auto_split=(round_num == 2),  # 只在第2轮拆分
            auto_prune=False,
            round_name=f"第{round_num}轮优化",
        )

        # 评估
        accuracy = evaluate(main_adapter, tree.root.skill.system_prompt, test_data)
        improvement = (accuracy - accuracy_history[-1]) * 100
        logger.info(f"\n📊 第{round_num}轮准确率: {accuracy*100:.1f}% (提升: {improvement:+.1f}%)")

        accuracy_history.append(accuracy)

        # 更新最佳结果
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_tree = tree
            logger.info(f"   🎯 新的最佳准确率!")

        # 如果连续2轮没有提升，提前停止
        if round_num >= 2:
            if accuracy_history[-1] <= accuracy_history[-2] and accuracy_history[-2] <= accuracy_history[-3]:
                logger.info(f"\n⚠️  连续2轮无提升，提前停止")
                break

    # ========================================================================
    # 保存最终结果
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 多轮优化完成")
    logger.info(f"{'='*60}")

    # 显示准确率历史
    logger.info(f"\n📈 准确率变化:")
    for i, acc in enumerate(accuracy_history):
        if i == 0:
            logger.info(f"   第0轮 (基准): {acc*100:.1f}%")
        else:
            improvement = (acc - accuracy_history[i-1]) * 100
            logger.info(f"   第{i}轮: {acc*100:.1f}% ({improvement:+.1f}%)")

    logger.info(f"\n📊 最终统计:")
    logger.info(f"   初始准确率: {initial_accuracy*100:.1f}%")
    logger.info(f"   最终准确率: {best_accuracy*100:.1f}%")
    logger.info(f"   总提升: {(best_accuracy-initial_accuracy)*100:+.1f}%")
    logger.info(f"   相对提升: {(best_accuracy/initial_accuracy - 1)*100:+.1f}%")

    # 保存
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
        logger.info(f"\n✨ 子skill数量: {len(best_tree.root.children)}")

    logger.info(f"\n✅ Demo完成! 查看输出: {output_path}")


if __name__ == "__main__":
    main()
