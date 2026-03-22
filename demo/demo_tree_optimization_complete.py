"""
论文分类树感知优化 Demo - 多轮渐进式优化

这个 demo 展示 TreeAwareOptimizer 的多轮渐进式优化:
1. 第一轮：学习大类分类（CS/AI/Bio/Physics等）
2. 第二轮：大类稳定后，拆分为细分类别
3. 第三轮：细分类别优化，剪枝低性能节点

关键特性:
- 自动拆分 - 检测到矛盾反馈时自动拆分 skill 为子 skills
- 自动剪枝 - 根据性能指标自动移除低效子 skill
- 部分优化 - 支持只修改 prompt 的某一部分
- 树感知优化 - 递归优化整棵 skill 树（bottom-up）

```mermaid
graph TD
    A[加载数据] --> B[创建根Skill]
    B --> C[第1轮: 大类分类]
    C --> D[收集反馈]
    D --> E[优化并拆分]
    E --> F{准确率>60%?}
    F -->|是| G[第2轮: 细分类别]
    F -->|否| H[调整温度重试]
    G --> I[收集细类反馈]
    I --> J[优化细类Skills]
    J --> K[剪枝低性能节点]
    K --> L[最终评估]
    H --> D
```
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

# 加载 .env 文件
load_dotenv()

from tresskill import (
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    OptimizerConfig,
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
    SkillTree,
)
from tresskill.schema import Skill, SkillMeta
from tresskill.skill_tree import SkillNode

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Step 1: 数据加载
# ============================================================================

def load_data(csv_path: str, train_size: int = 30, eval_size: int = 10, test_size: int = 10, seed: int = 42):
    """加载数据并分割"""
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
    eval_set = data[train_size:train_size + eval_size]
    test = data[train_size + eval_size:train_size + eval_size + test_size]

    logger.info(f"\n✅ 数据分割完成:")
    logger.info(f"   训练集: {len(train)} 条")
    logger.info(f"   验证集: {len(eval_set)} 条")
    logger.info(f"   测试集: {len(test)} 条")

    return train, eval_set, test


# ============================================================================
# Step 2: 创建 Skill 树（初始版本）
# ============================================================================

def create_initial_skill_tree() -> SkillTree:
    """
    创建初始的 skill 树（单个根节点）

    初始 prompt 设计得比较简单，故意不够精确，
    这样后续可以通过优化来改进它。
    """
    from tresskill.skill_tree import SkillNode

    logger.info("\n" + "="*60)
    logger.info("创建初始 Skill 树")
    logger.info("="*60)

    # 创建根 skill（简单但不够准确的初始 prompt）
    # 故意设计成大类分类器，后续会自动拆分
    root_prompt = """You are a scientific paper classification expert. Your task is to classify research papers into broad scientific categories.

For each paper, read the title and abstract, then determine which major category it belongs to.

Major categories include:
- Computer Science (CS)
- Artificial Intelligence (AI)
- Biology (Bio)
- Physics (Physics)
- Chemistry (Chem)
- Mathematics (Math)
- Medicine (Med)

Return ONLY the category name, nothing else.
    """

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=root_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="Root classifier - will be split into specialized children",
        ),
    )

    # 创建 skill 树
    tree = SkillTree(
        root=SkillNode(name="paper-classifier", skill=root_skill),
        base_path=Path("demo-paper-tree-initial/"),
    )

    logger.info(f"✅ 初始 Skill 树创建完成")
    logger.info(f"   根节点: {root_skill.name}")
    logger.info(f"   版本: {root_skill.version}")
    logger.info(f"\n初始 Prompt:\n{root_prompt}")

    return tree


# ============================================================================
# Step 3: 收集反馈
# ============================================================================

def collect_feedback_with_temperature(
    adapter: OpenAIAdapter,
    data: List[Dict],
    temperature: float = 0.7,
) -> List[ConversationExperience]:
    """
    在数据集上运行分类器，收集反馈

    使用指定的温度参数来控制生成的多样性
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"收集反馈 (temperature={temperature})")
    logger.info(f"{'='*60}")

    experiences = []

    for idx, item in enumerate(data):
        question = item['question']
        expected_label = item['answer']

        logger.info(f"\n[{idx+1}/{len(data)}] 处理论文: {question[:60]}...")

        # 构建分类 prompt
        user_message = f"""Classify this paper:

{question}

Return ONLY the category name.
        """

        # 调用 LLM
        try:
            from tresskill.core.prompts import TextPrompt
            prompt = TextPrompt(content="")  # dummy prompt for generation

            # 直接调用 API
            messages = [
                {"role": "system", "content": """You are a scientific paper classification expert. Your task is to classify research papers into scientific categories.

For each paper. read the title and abstract, then determine which category it belongs to.

Return ONLY the category name, nothing else."""},
                {"role": "user", "content": user_message},
            ]

            predicted_label = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature,
            ).strip()

            logger.info(f"   预测: {predicted_label}")
            logger.info(f"   实际: {expected_label}")

            # 判断是否正确
            is_correct = predicted_label.lower() == expected_label.lower()

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


# ============================================================================
# Step 4: 评估准确率
# ============================================================================

def evaluate_accuracy(
    adapter: OpenAIAdapter,
    tree: SkillTree,
    test_data: List[Dict],
    temperature: float = 0.3,
) -> float:
    """
    在测试集上评估 skill 树的准确率

    使用较低的温度来获得更确定的预测
    """
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

Return ONLY the category name.
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
            ).strip()

            is_correct = predicted_label.lower() == expected_label.lower()

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


# ============================================================================
# Step 5: 树感知优化（核心功能）
# ============================================================================

def run_tree_optimization(
    adapter: OpenAIAdapter,
    tree: SkillTree,
    train_experiences: List[ConversationExperience],
    temperature: float = 0.3,
    auto_split: bool = True,
    auto_prune: bool = True,
    section: str = "all",
) -> Tuple[SkillTree, int]:
    """
    运行树感知优化

    展示:
    1. 单点优化（使用指定温度）
    2. 自动拆分（检测矛盾反馈）
    3. 自动剪枝（移除低性能节点）
    4. 部分优化（可选）
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"树感知优化")
    logger.info(f"{'='*60}")
    logger.info(f"配置:")
    logger.info(f"   temperature: {temperature}")
    logger.info(f"   auto_split: {auto_split}")
    logger.info(f"   auto_prune: {auto_prune}")
    logger.info(f"   section: {section}")

    # 创建 TreeAwareOptimizer
    config = TreeOptimizerConfig(
        auto_split=auto_split,
        auto_prune=auto_prune,
        prune_threshold=0.3,
        min_samples_for_split=5,
        max_tree_depth=3,
        optimization_order="bottom_up",
        section=section,
    )

    # 配置基础优化器（使用指定的温度）
    base_optimizer_config = OptimizerConfig(
        max_steps=1,
        gradient_accumulation_steps=5,
        conservative=False,
        early_stopping_patience=2,
    )

    tree_optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config,
        base_optimizer_config=base_optimizer_config,
    )

    # 运行优化
    result = tree_optimizer.optimize_tree(
        tree=tree,
        experiences=train_experiences,
        validator=None,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"优化完成!")
    logger.info(f"{'='*60}")
    logger.info(f"   节点优化数: {result.nodes_optimized}")
    logger.info(f"   拆分次数: {result.splits_performed}")
    logger.info(f"   剪枝次数: {result.prunes_performed}")

    return result.tree, result.splits_performed + result.prunes_performed


# ============================================================================
# 主流程：渐进式温度调整 + 树感知优化
# ============================================================================

# ============================================================================
# 主流程：多轮渐进式优化 + 树感知优化
# ============================================================================

def main():
    """
    主流程：多轮渐进式优化

    第1轮：大类分类学习
    - 使用低温度(0.3)收集反馈
    - 优化根节点
    - 目标准确率: 60%+

    第2轮：细分类别拆分
    - 如果第1轮效果好，提高温度(0.5)探索
    - 自动拆分为细分类别
    - 优化各个子节点

    第3轮：细类优化 + 剪枝
    - 降低温度(0.2)精细优化
    - 剪枝低性能节点
    - 最终评估
    """
    logger.info("\n" + "="*60)
    logger.info("🚀 论文分类树感知优化 Demo - 多轮渐进式优化")
    logger.info("="*60)

    # Step 1: 加载数据
    csv_path = "demo/data/intern_camp5.csv"
    train_data, eval_data, test_data = load_data(csv_path, train_size=50, eval_size=10, test_size=10)

    # Step 2: 创建 API 适配器
    # 主模型用于分类
    main_model = os.getenv("TRES_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")

    # Judge 和 Rewrite 使用更强的模型
    judge_model = os.getenv("TRES_LLM_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")

    if not api_key:
        logger.error("❌ 请设置 TRES_LLM_API_KEY 环境变量")
        return

    # 主适配器（用于分类）
    main_adapter = OpenAIAdapter(
        model=main_model,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge 适配器（用于梯度计算和重写）
    judge_adapter = OpenAIAdapter(
        model=judge_model,
        api_key=api_key,
        base_url=base_url,
    )

    logger.info(f"\n✅ API 适配器创建完成")
    logger.info(f"   主模型（分类）: {main_model}")
    logger.info(f"   Judge模型（优化）: {judge_model}")
    logger.info(f"   Base URL: {base_url}")

    # Step 3: 创建初始 skill 树
    tree = create_initial_skill_tree()

    # Step 4: 评估初始准确率（基准线）
    initial_accuracy = evaluate_accuracy(main_adapter, tree, test_data, temperature=0.3)
    logger.info(f"\n📊 初始准确率: {initial_accuracy*100:.2f}%")

    best_accuracy = initial_accuracy
    best_tree = tree

    # ========================================================================
    # 第1轮：大类分类学习
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("第 1 轮: 大类分类学习")
    logger.info("="*60)

    logger.info("目标: 训练根节点学会大类分类（准确率>60%）")

    # 收集反馈（低温度，更确定性）
    train_experiences_round1 = collect_feedback_with_temperature(
        adapter=main_adapter,
        data=train_data[:20],  # 先用20条
        temperature=0.3,
    )

    # 优化（不拆分，不剪枝，专注优化根节点）
    tree_round1, changes_round1 = run_tree_optimization(
        adapter=judge_adapter,  # 使用 judge 模型优化
        tree=tree,
        train_experiences=train_experiences_round1,
        temperature=0.3,
        auto_split=False,  # 第一轮不拆分
        auto_prune=False,   # 第一轮不剪枝
        section="all",
    )

    # 评估
    accuracy_round1 = evaluate_accuracy(main_adapter, tree_round1, test_data, temperature=0.3)
    logger.info(f"\n📊 第1轮准确率: {accuracy_round1*100:.2f}%")
    logger.info(f"   提升: {(accuracy_round1 - initial_accuracy)*100:+.2f}%")

    if accuracy_round1 > best_accuracy:
        best_accuracy = accuracy_round1
        best_tree = tree_round1

    # 判断是否继续
    if accuracy_round1 < 0.5:
        logger.warning("\n⚠️  第1轮效果不佳，继续优化根节点...")

        # 再优化一次（更低温度）
        train_experiences_round1b = collect_feedback_with_temperature(
            adapter=main_adapter,
            data=train_data[20:40],  # 用更多数据
            temperature=0.2,
        )

        tree_round1b, _ = run_tree_optimization(
            adapter=judge_adapter,
            tree=tree_round1,
            train_experiences=train_experiences_round1b,
            temperature=0.2,
            auto_split=False,
            auto_prune=False,
            section="instruction",  # 只优化指令部分
        )

        accuracy_round1b = evaluate_accuracy(main_adapter, tree_round1b, test_data, temperature=0.3)
        logger.info(f"\n📊 第1b轮准确率: {accuracy_round1b*100:.2f}%")

        if accuracy_round1b > best_accuracy:
            best_accuracy = accuracy_round1b
            best_tree = tree_round1b
            tree_round1 = tree_round1b
            accuracy_round1 = accuracy_round1b

    # ========================================================================
    # 第2轮：细分类别拆分
    # ========================================================================
    if accuracy_round1 >= 0.5:  # 如果第1轮还可以
        logger.info("\n" + "="*60)
        logger.info("第 2 轮: 细分类别拆分")
        logger.info("="*60)

        logger.info("目标: 检测矛盾反馈，自动拆分为细分类别")

        # 收集更多样化的反馈（提高温度探索）
        train_experiences_round2 = collect_feedback_with_temperature(
            adapter=main_adapter,
            data=train_data,  # 全部数据
            temperature=0.5,  # 提高温度
        )

        # 优化（启用拆分，不剪枝）
        tree_round2, changes_round2 = run_tree_optimization(
            adapter=judge_adapter,
            tree=tree_round1,
            train_experiences=train_experiences_round2,
            temperature=0.4,
            auto_split=True,  # 启用自动拆分！
            auto_prune=False,
            section="all",
        )

        # 评估
        accuracy_round2 = evaluate_accuracy(main_adapter, tree_round2, test_data, temperature=0.3)
        logger.info(f"\n📊 第2轮准确率: {accuracy_round2*100:.2f}%")
        logger.info(f"   拆分次数: {changes_round2}")

        if accuracy_round2 > best_accuracy:
            best_accuracy = accuracy_round2
            best_tree = tree_round2
            tree_round2_current = tree_round2
            accuracy_round2_current = accuracy_round2

        # ====================================================================
        # 第3轮：细类优化 + 剪枝
        # ====================================================================
        if accuracy_round2 >= 0.55:  # 如果第2轮还可以
            logger.info("\n" + "="*60)
            logger.info("第 3 轮: 细类优化 + 剪枝")
            logger.info("="*60)

            logger.info("目标: 精细优化各个子节点，剪枝低性能节点")

            # 收集反馈（低温度，精细）
            train_experiences_round3 = collect_feedback_with_temperature(
                adapter=main_adapter,
                data=train_data,
                temperature=0.2,  # 低温度
            )

            # 优化（不拆分，启用剪枝）
            tree_round3, changes_round3 = run_tree_optimization(
                adapter=judge_adapter,
                tree=tree_round2_current,
                train_experiences=train_experiences_round3,
                temperature=0.2,
                auto_split=False,  # 不再拆分
                auto_prune=True,   # 启用剪枝！
                section="all",
            )

            # 评估
            accuracy_round3 = evaluate_accuracy(main_adapter, tree_round3, test_data, temperature=0.3)
            logger.info(f"\n📊 第3轮准确率: {accuracy_round3*100:.2f}%")
            logger.info(f"   剪枝次数: {changes_round3}")

            if accuracy_round3 > best_accuracy:
                best_accuracy = accuracy_round3
                best_tree = tree_round3

    # ========================================================================
    # 最终结果
    # ========================================================================
    logger.info(f"\n" + "="*60)
    logger.info("📊 最终结果")
    logger.info("="*60)
    logger.info(f"初始准确率: {initial_accuracy*100:.2f}%")
    logger.info(f"最终准确率: {best_accuracy*100:.2f}%")
    logger.info(f"总提升: {(best_accuracy - initial_accuracy)*100:+.2f}%")

    # 保存最佳 skill 树
    output_path = Path("demo-paper-tree-optimized/")
    best_tree.save(output_path)
    logger.info(f"\n💾 优化后的 skill 树已保存到: {output_path}")

    # 显示树结构
    logger.info(f"\n🌲 优化后的树结构:")
    logger.info(f"\n{best_tree.list_tree()}")

    logger.info(f"\n✨ Demo 完成!")

    # 如果连续3次都有提升，可以提前结束
    if best_accuracy - initial_accuracy >= 0.15:  # 提升15%以上
        logger.info("🎉 效果显著！优化成功！")


if __name__ == "__main__":
    main()
