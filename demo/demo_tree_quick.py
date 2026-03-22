"""
树感知优化精简 Demo - 快速验证版

精简配置：
- 30 训练样本（快速）
- 10 测试样本
- 3 轮优化（大类学习 → 自动拆分 → 剪枝）
- 双模型：7B（主） + 72B（Judge）

关键特性展示：
1. 自动拆分 - 检测矛盾反馈，拆分为子类别
2. 自动剪枝 - 移除低性能子 skill
3. 多轮渐进 - 第1轮学大类，第2轮拆分，第3轮精调
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json
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


def load_data(csv_path: str, train_size: int = 30, test_size: int = 10, seed: int = 42):
    """加载数据"""
    logger.info(f"📂 加载数据: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    logger.info(f"✅ 数据总量: {len(data)} 条")

    # 统计类别
    labels = [item['answer'] for item in data]
    label_counts = Counter(labels)
    logger.info(f"📊 类别数: {len(label_counts)}")
    logger.info(f"   前10个类别: {label_counts.most_common(10)}")

    # 随机打乱
    random.seed(seed)
    random.shuffle(data)

    # 分割
    train = data[:train_size]
    test = data[train_size:train_size + test_size]

    logger.info(f"\n✅ 数据分割完成:")
    logger.info(f"   训练集: {len(train)} 条")
    logger.info(f"   测试集: {len(test)} 条")

    return train, test


def create_initial_skill_tree() -> SkillTree:
    """创建初始 skill 树（简单大类分类器）"""
    logger.info("\n" + "="*60)
    logger.info("创建初始 Skill 树")
    logger.info("="*60)

    # 简单的大类分类 prompt（故意简化，让优化器改进）
    root_prompt = """You are a paper classification expert. Classify papers into major categories:

Categories:
- A-D: Physics (quantum, atomic, condensed matter, optics)
- E-H: CS Core (robotics, NLP, software engineering, IR)
- I-K: High Energy Physics (theory, phenomenology, optics)
- L-O: AI & Vision (AI, CV, nuclear theory, astro)
- P-S: Math & Signals (probability, OS, signal processing, optimization)
- T-Z: Advanced Topics (dynamical systems, geometry, methods, multimedia, combinatorics, neural computing)

Return ONLY the category letter (A-Z).
"""

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=root_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="Root classifier - will be auto-split into specialized children",
        ),
    )

    # 创建 skill 树
    tree = SkillTree(
        root=SkillNode(name="paper-classifier", skill=root_skill),
        base_path=Path("demo-tree-quick/"),
    )

    logger.info(f"✅ 初始 Skill 树创建完成")
    logger.info(f"   根节点: {root_skill.name}")
    logger.info(f"\n初始 Prompt:\n{root_prompt}")

    return tree


def collect_feedback(
    adapter: OpenAIAdapter,
    data: List[Dict],
    system_prompt: str,
    temperature: float = 0.5,
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
        user_message = f"""Classify this paper:

{question}

Return ONLY the category letter (A-Z).
"""

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

            # 提取字母
            if len(predicted_label) > 0 and predicted_label[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                predicted_label = predicted_label[0]
            else:
                predicted_label = 'A'  # 默认

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
                    score=0.9,
                )
                logger.info(f"   ✅ 正确!")
            else:
                exp.feedback = CompositeFeedback(
                    critique=f"Wrong classification. Should be {expected_label}, not {predicted_label}",
                    correction=expected_label,
                    score=0.1,
                )
                logger.info(f"   ❌ 错误!")

            experiences.append(exp)

        except Exception as e:
            logger.error(f"   ⚠️  跳过: {e}")
            continue

    # 统计
    positive = sum(1 for e in experiences if e.feedback.to_score() >= 0.6)
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
    temperature: float = 0.3,
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

        user_message = f"""Classify this paper:

{question}

Return ONLY the category letter (A-Z).
"""

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
    """主流程：3轮渐进式优化"""
    logger.info("\n" + "="*60)
    logger.info("🚀 树感知优化精简 Demo")
    logger.info("="*60)

    # Step 1: 加载数据
    csv_path = "demo/data/intern_camp5.csv"
    train_data, test_data = load_data(csv_path, train_size=30, test_size=10)

    # Step 2: 创建 API 适配器
    main_model = os.getenv("TRES_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    judge_model = os.getenv("TRES_LLM_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")

    if not api_key:
        logger.error("❌ 请设置 TRES_LLM_API_KEY 环境变量")
        return

    # 主适配器（分类）
    main_adapter = OpenAIAdapter(
        model=main_model,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge 适配器（优化）
    judge_adapter = OpenAIAdapter(
        model=judge_model,
        api_key=api_key,
        base_url=base_url,
    )

    logger.info(f"\n✅ API 适配器创建完成")
    logger.info(f"   主模型（分类）: {main_model}")
    logger.info(f"   Judge模型（优化）: {judge_model}")

    # Step 3: 创建初始 skill 树
    tree = create_initial_skill_tree()

    # Step 4: 评估初始准确率
    initial_accuracy = evaluate_accuracy(main_adapter, tree, test_data, temperature=0.3)
    logger.info(f"\n📊 初始准确率: {initial_accuracy*100:.2f}%")

    best_accuracy = initial_accuracy
    best_tree = tree

    # ========================================================================
    # 第1轮：大类分类学习（低温，保守，不拆分不剪枝）
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("第 1 轮: 大类分类学习")
    logger.info("="*60)
    logger.info("目标: 训练根节点学会大类分类")

    # 收集反馈（低温）
    train_exp_round1 = collect_feedback(
        adapter=main_adapter,
        data=train_data[:15],  # 先用15条
        system_prompt=tree.root.skill.system_prompt,
        temperature=0.3,
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
        gradient_accumulation_steps=5,
        conservative=True,  # 保守策略
    )

    tree_optimizer = TreeAwareOptimizer(
        adapter=judge_adapter,
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
    logger.info(f"   拆分次数: {result_round1.splits_performed}")
    logger.info(f"   剪枝次数: {result_round1.prunes_performed}")

    # 评估
    accuracy_round1 = evaluate_accuracy(main_adapter, result_round1.tree, test_data, temperature=0.3)
    logger.info(f"\n📊 第1轮准确率: {accuracy_round1*100:.2f}%")
    logger.info(f"   提升: {(accuracy_round1 - initial_accuracy)*100:+.2f}%")

    if accuracy_round1 > best_accuracy:
        best_accuracy = accuracy_round1
        best_tree = result_round1.tree

    # ========================================================================
    # 第2轮：自动拆分（中温，平衡，启用拆分）
    # ========================================================================
    if accuracy_round1 >= 0.3:  # 如果第1轮还可以
        logger.info("\n" + "="*60)
        logger.info("第 2 轮: 自动拆分")
        logger.info("="*60)
        logger.info("目标: 检测矛盾反馈，自动拆分为细分类别")

        # 收集反馈（中温）
        train_exp_round2 = collect_feedback(
            adapter=main_adapter,
            data=train_data,  # 全部数据
            system_prompt=result_round1.tree.root.skill.system_prompt,
            temperature=0.5,
        )

        # 优化（启用拆分，不剪枝）
        config_round2 = TreeOptimizerConfig(
            auto_split=True,   # 启用自动拆分！
            auto_prune=False,
            prune_threshold=0.3,
            min_samples_for_split=3,  # 降低拆分门槛
            max_tree_depth=2,
            optimization_order="bottom_up",
            section="all",
        )

        base_config_round2 = OptimizerConfig(
            max_steps=1,
            gradient_accumulation_steps=8,
            conservative=False,  # 平衡策略
        )

        tree_optimizer_round2 = TreeAwareOptimizer(
            adapter=judge_adapter,
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
        logger.info(f"   剪枝次数: {result_round2.prunes_performed}")

        # 评估
        accuracy_round2 = evaluate_accuracy(main_adapter, result_round2.tree, test_data, temperature=0.3)
        logger.info(f"\n📊 第2轮准确率: {accuracy_round2*100:.2f}%")
        logger.info(f"   拆分次数: {result_round2.splits_performed}")

        if accuracy_round2 > best_accuracy:
            best_accuracy = accuracy_round2
            best_tree = result_round2.tree
            tree_round2 = result_round2.tree
            accuracy_round2_val = accuracy_round2

        # ====================================================================
        # 第3轮：剪枝优化（低温，保守，启用剪枝）
        # ====================================================================
        if accuracy_round2 >= 0.35:  # 如果第2轮还可以
            logger.info("\n" + "="*60)
            logger.info("第 3 轮: 剪枝优化")
            logger.info("="*60)
            logger.info("目标: 精细优化，剪枝低性能节点")

            # 收集反馈（低温）
            train_exp_round3 = collect_feedback(
                adapter=main_adapter,
                data=train_data,
                system_prompt=result_round2.tree.root.skill.system_prompt,
                temperature=0.2,
            )

            # 优化（不拆分，启用剪枝）
            config_round3 = TreeOptimizerConfig(
                auto_split=False,  # 不再拆分
                auto_prune=True,   # 启用剪枝！
                prune_threshold=0.35,
                max_tree_depth=2,
                optimization_order="bottom_up",
                section="all",
            )

            base_config_round3 = OptimizerConfig(
                max_steps=1,
                gradient_accumulation_steps=8,
                conservative=True,  # 保守策略
            )

            tree_optimizer_round3 = TreeAwareOptimizer(
                adapter=judge_adapter,
                config=config_round3,
                base_optimizer_config=base_config_round3,
            )

            result_round3 = tree_optimizer_round3.optimize_tree(
                tree=tree_round2,
                experiences=train_exp_round3,
                validator=None,
            )

            logger.info(f"\n第3轮优化完成:")
            logger.info(f"   节点优化数: {result_round3.nodes_optimized}")
            logger.info(f"   拆分次数: {result_round3.splits_performed}")
            logger.info(f"   剪枝次数: {result_round3.prunes_performed} ✂️")

            # 评估
            accuracy_round3 = evaluate_accuracy(main_adapter, result_round3.tree, test_data, temperature=0.3)
            logger.info(f"\n📊 第3轮准确率: {accuracy_round3*100:.2f}%")
            logger.info(f"   剪枝次数: {result_round3.prunes_performed}")

            if accuracy_round3 > best_accuracy:
                best_accuracy = accuracy_round3
                best_tree = result_round3.tree

    # ========================================================================
    # 最终结果
    # ========================================================================
    logger.info(f"\n" + "="*60)
    logger.info(f"📊 最终结果")
    logger.info(f"="*60)
    logger.info(f"初始准确率: {initial_accuracy*100:.2f}%")
    logger.info(f"最终准确率: {best_accuracy*100:.2f}%")
    logger.info(f"总提升: {(best_accuracy - initial_accuracy)*100:+.2f}%")

    # 保存最佳 skill 树
    output_path = Path("demo-tree-quick-optimized/")
    best_tree.save(output_path)
    logger.info(f"\n💾 优化后的 skill 树已保存到: {output_path}")

    # 显示树结构
    logger.info(f"\n🌲 优化后的树结构:")
    logger.info(f"\n{best_tree.list_tree()}")

    logger.info(f"\n✨ Demo 完成!")

    # 成功标志
    if best_accuracy - initial_accuracy >= 0.05:
        logger.info("🎉 效果显著！树感知优化成功！")


if __name__ == "__main__":
    main()
