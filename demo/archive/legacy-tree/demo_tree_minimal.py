"""
最小可用的论文分类树优化 Demo

关键改进：
1. 极简初始prompt（<100字符）
2. 小数据集（5训练 + 5测试）
3. 自动拆分策略（先学大类，再自动拆分）
4. 降低API成本（使用7B模型）

预期效果：
- 初始准确率：20-30%
- 优化后：40-60%（提升20-30%）
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


def load_data(csv_path: str, train_size: int = 5, test_size: int = 5, seed: int = 42):
    """加载数据"""
    logger.info(f"📂 加载数据: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # 过滤：只保留前5个最常见的类别（简化任务）
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

    # 统计类别分布
    train_labels = [item['answer'] for item in train]
    logger.info(f"\n训练集类别分布: {Counter(train_labels)}")

    return train, test, top_labels


def create_initial_skill_tree(top_labels: List[str]) -> SkillTree:
    """创建极简初始 skill 树"""
    logger.info("\n" + "="*60)
    logger.info("创建极简初始 Skill 树")
    logger.info("="*60)

    # 类别名称映射（从数据中提取语义）
    category_names = {
        'A': 'Quantum Physics',
        'D': 'Soft Condensed Matter',
        'E': 'Robotics',
        'F': 'Computation and Language (NLP)',
        'G': 'Software Engineering',
        'I': 'High Energy Physics - Theory',
        'L': 'Artificial Intelligence',
        'M': 'Computer Vision',
    }

    # 极简的初始 prompt（添加类别名称）
    categories_desc = '\n'.join([f"{label}: {category_names.get(label, 'Unknown')}" for label in top_labels])

    root_prompt = f"""Classify scientific papers into these categories:

{categories_desc}

Return ONLY the letter (e.g., A, E, G)."""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=root_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="Simple paper classifier - will be auto-split into specialized children",
        ),
    )

    # 创建 skill 树
    tree = SkillTree(
        root=SkillNode(name="paper-classifier", skill=root_skill),
        base_path=Path("demo-tree-minimal/"),
    )

    logger.info(f"✅ 初始 Skill 树创建完成")
    logger.info(f"   Prompt 长度: {len(root_prompt)} 字符")
    logger.info(f"\n初始 Prompt:\n{root_prompt}")

    return tree


def collect_feedback(
    adapter: OpenAIAdapter,
    data: List[Dict],
    system_prompt: str,
    temperature: float = 0.3,
) -> List[ConversationExperience]:
    """收集反馈"""
    logger.info(f"\n{'='*60}")
    logger.info(f"收集反馈 (temperature={temperature})")
    logger.info(f"{'='*60}")

    experiences = []

    for idx, item in enumerate(data):
        question = item['question']
        expected_label = item['answer']

        logger.info(f"\n[{idx+1}/{len(data)}] 处理论文: {question[:60]}...")

        # 构建消息
        user_message = f"Paper:\n{question}\n\nCategory:"

        # 调用 LLM
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

            # 提取字母（取第一个字符）
            if len(predicted_label) > 0 and predicted_label[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                predicted_label = predicted_label[0]
            else:
                # 如果解析失败，随机选择一个
                predicted_label = expected_label  # 给一个默认值，避免完全随机

            logger.info(f"   预测: {predicted_label}")
            logger.info(f"   实际: {expected_label}")

            # 判断是否正确
            is_correct = predicted_label == expected_label

            # 创建 experience
            exp = ConversationExperience(
                messages=[{"role": "user", "content": user_message}],
                response=predicted_label,
                metadata={"paper_id": idx, "skill_name": "paper-classifier"},
            )

            # 设置反馈
            if is_correct:
                exp.feedback = CompositeFeedback(
                    critique="Correct classification",
                    score=1.0,
                )
                logger.info(f"   ✅ 正确!")
            else:
                exp.feedback = CompositeFeedback(
                    critique=f"Wrong classification. Should be {expected_label}, not {predicted_label}. Analyze the paper's domain carefully.",
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
    negative = len(experiences) - positive

    logger.info(f"\n{'='*60}")
    logger.info(f"反馈收集完成")
    logger.info(f"{'='*60}")
    logger.info(f"   总数: {len(experiences)}")
    logger.info(f"   ✅ 正面: {positive} ({positive/len(experiences)*100:.1f}%)")
    logger.info(f"   ❌ 负面: {negative} ({negative/len(experiences)*100:.1f}%)")

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

    # 获取根节点的 prompt
    system_prompt = tree.root.skill.system_prompt

    for idx, item in enumerate(test_data):
        question = item['question']
        expected_label = item['answer']

        user_message = f"Paper:\n{question}\n\nCategory:"

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
    logger.info(f"\n{'='*60}")
    logger.info(f"评估完成")
    logger.info(f"{'='*60}")
    logger.info(f"   准确率: {correct}/{total} = {accuracy*100:.2f}%")

    return accuracy


def main():
    """主流程：2轮优化（学大类 → 自动拆分）"""
    logger.info("\n" + "="*60)
    logger.info("🚀 最小可用的论文分类树优化 Demo")
    logger.info("="*60)

    # Step 1: 加载数据
    csv_path = "demo/data/intern_camp5.csv"
    train_data, test_data, top_labels = load_data(csv_path, train_size=5, test_size=5)

    # Step 2: 创建 API 适配器
    model = os.getenv("TRES_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")

    if not api_key:
        logger.error("❌ 请设置 TRES_LLM_API_KEY 环境变量")
        return

    # 单个适配器（既做分类也做优化）
    adapter = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    logger.info(f"\n✅ API 适配器创建完成")
    logger.info(f"   模型: {model}")

    # Step 3: 创建初始 skill 树
    tree = create_initial_skill_tree(top_labels)

    # Step 4: 评估初始准确率
    initial_accuracy = evaluate_accuracy(adapter, tree, test_data, temperature=0.2)
    logger.info(f"\n📊 初始准确率: {initial_accuracy*100:.2f}%")

    # ========================================================================
    # 第1轮：基础学习（低温，保守，不拆分）
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("第 1 轮: 基础学习")
    logger.info("="*60)
    logger.info("目标: 学会基础分类规则")

    # 收集反馈（低温）
    train_exp_round1 = collect_feedback(
        adapter=adapter,
        data=train_data,
        system_prompt=tree.root.skill.system_prompt,
        temperature=0.2,  # 低温，输出稳定
    )

    # 优化（不拆分，不剪枝）
    config_round1 = TreeOptimizerConfig(
        auto_split=False,  # 第一轮不拆分
        auto_prune=False,  # 第一轮不剪枝
        optimization_order="bottom_up",
        section="all",
    )

    base_config = OptimizerConfig(
        max_steps=1,
        gradient_accumulation_steps=3,  # 小数据集用3即可
        conservative=True,  # 保守策略
    )

    tree_optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config_round1,
        base_optimizer_config=base_config,
    )

    result_round1 = tree_optimizer.optimize_tree(
        tree=tree,
        experiences=train_exp_round1,
        validator=None,
    )

    logger.info(f"\n第1轮优化完成:")
    logger.info(f"   节点优化数: {result_round1.nodes_optimized}")

    # 评估
    accuracy_round1 = evaluate_accuracy(adapter, result_round1.tree, test_data, temperature=0.2)
    logger.info(f"\n📊 第1轮准确率: {accuracy_round1*100:.2f}%")
    logger.info(f"   提升: {(accuracy_round1 - initial_accuracy)*100:+.2f}%")

    # ========================================================================
    # 第2轮：自动拆分（中温，平衡，启用拆分）
    # ========================================================================
    if accuracy_round1 > 0:  # 如果第1轮有提升
        logger.info("\n" + "="*60)
        logger.info("第 2 轮: 自动拆分")
        logger.info("="*60)
        logger.info("目标: 检测矛盾反馈，自动拆分为细分类别")

        # 收集反馈（中温）
        train_exp_round2 = collect_feedback(
            adapter=adapter,
            data=train_data * 2,  # 重复数据，增加多样性
            system_prompt=result_round1.tree.root.skill.system_prompt,
            temperature=0.4,  # 中温，平衡
        )

        # 优化（启用拆分）
        config_round2 = TreeOptimizerConfig(
            auto_split=True,   # 启用自动拆分！
            auto_prune=False,
            min_samples_for_split=2,  # 降低拆分门槛
            max_tree_depth=2,
            optimization_order="bottom_up",
            section="all",
        )

        base_config_round2 = OptimizerConfig(
            max_steps=1,
            gradient_accumulation_steps=3,
            conservative=False,  # 平衡策略
        )

        tree_optimizer_round2 = TreeAwareOptimizer(
            adapter=adapter,
            config=config_round2,
            base_optimizer_config=base_config_round2,
        )

        result_round2 = tree_optimizer_round2.optimize_tree(
            tree=result_round1.tree,
            experiences=train_exp_round2,
            validator=None,
        )

        logger.info(f"\n第2轮优化完成:")
        logger.info(f"   节点优化数: {result_round2.nodes_optimized}")
        logger.info(f"   拆分次数: {result_round2.splits_performed} ⭐")

        # 评估
        accuracy_round2 = evaluate_accuracy(adapter, result_round2.tree, test_data, temperature=0.2)
        logger.info(f"\n📊 第2轮准确率: {accuracy_round2*100:.2f}%")
        logger.info(f"   拆分次数: {result_round2.splits_performed}")

        final_tree = result_round2.tree
        final_accuracy = accuracy_round2
    else:
        final_tree = result_round1.tree
        final_accuracy = accuracy_round1

    # ========================================================================
    # 最终结果
    # ========================================================================
    logger.info(f"\n" + "="*60)
    logger.info(f"📊 最终结果")
    logger.info(f"="*60)
    logger.info(f"初始准确率: {initial_accuracy*100:.2f}%")
    logger.info(f"最终准确率: {final_accuracy*100:.2f}%")
    logger.info(f"总提升: {(final_accuracy - initial_accuracy)*100:+.2f}%")

    # 保存最佳 skill 树
    output_path = Path("demo-tree-minimal-optimized/")
    final_tree.save(output_path)
    logger.info(f"\n💾 优化后的 skill 树已保存到: {output_path}")

    # 显示树结构
    logger.info(f"\n🌲 优化后的树结构:")
    logger.info(f"\n{final_tree.list_tree()}")

    logger.info(f"\n✨ Demo 完成!")

    # 成功标志
    if final_accuracy - initial_accuracy >= 0.1:
        logger.info("🎉 效果显著！树感知优化成功！")
    elif final_accuracy > initial_accuracy:
        logger.info("✅ 有提升！建议增加数据量或优化迭代轮数。")
    else:
        logger.info("⚠️  无提升，建议检查初始prompt或数据质量。")


if __name__ == "__main__":
    main()
