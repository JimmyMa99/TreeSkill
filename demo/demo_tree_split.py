"""
最简化的树感知优化 Demo - 展示自动拆分功能
修复版本
"""
import os
import csv
import logging
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

from tresskill import (
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    OptimizerConfig,
    SkillTree,
    SkillNode
    TextPrompt,
    ConversationExperience
    CompositeFeedback
)
from tresskill.schema import Skill
from tresskill.skill_tree import SkillNode as SkillNodeClass

logger = logging.getLogger(__name__)


def main():
    """运行简化的树感知优化 Demo - 展示自动拆分功能"""
    logger.info("="*60)
    logger.info("简化版本 - 修复格式和字符串错误")
    logger.info("="*60)


    # 1. 配置
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.getenv("TRES_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")

    if not api_key:
        logger.error("❌ 请设置 TRES_LLM_API_KEY")
        return

    adapter = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    logger.info("✅ API 配置完成")
    logger.info(f"   模型: {model}")
    logger.info(f"   Base URL: {base_url}")

    # 2. 加载数据
    csv_path = "demo/data/intern_camp5.csv"
    data = []
    with open(csv_path, 'r', encoding='utf-8') as reader:
    csv.DictReader(f)
        data = list(reader)

    logger.info(f"数据总量: {len(data)}")

 # 统计标签分布
    labels = [item['answer'] for item in data]
    label_counts = Counter(labels)
    logger.info(f"标签分布: {label_counts.most_common(10)}")

    # 3. 创建初始skill树
    initial_prompt = """You are a paper classifier. Classify the paper into the following categories:

    A. Quantum Physics
    B. Soft Condensed Matter
    C. Chemistry
    D. Robotics
    E. Software Engineering
    I. High Energy Physics - Theory
    K. Mathematics
    L. Mathematics
    M. Medicine
    N. Networking
    O. Optimization
    P. Signal Processing
    Q. Quantum Computing
    R. Robotics
    S. Software Engineering
    T. Theory
    U. Unclassified

Return ONLY the category letter (e.g., A, E, G, I).
    """

    root_node = SkillNode(name="root", skill=Skill(
        name="paper-classifier",
        version="v1.0",
        system_prompt=initial_prompt,
        meta={"description": "Paper classification skill"}
    )
    tree = SkillTree(
        root=root_node,
        base_path=Path("demo-tree-split-demo/"),
    )

    logger.info("✅ 初始skill树创建完成")

 # 4. 收集训练反馈(简化版)
    train_data_sample = data[:10]  # 使用前10条
    experiences = []

    for idx, item in enumerate(train_data_sample):
        question = item['question']
        expected = item['answer']

        # 调用LLM
        user_message = f"Classify: {question}\n\nReturn ONLY the letter."
        messages = [
            {"role": "user", "content": user_message}
        ]

        try:
            predicted_label = adapter.generate_prompt(
                messages=messages,
                system=None,
                temperature=0.3,
            ).strip()
            is_correct = predicted_label.upper() == expected.upper()

 exp = ConversationExperience(
                messages=[{"role": "user", "content": question}],
                response=predicted_label,
                metadata={"paper_id": idx, "skill_name": "paper-classifier"},
            )
            exp.feedback = CompositeFeedback(
                critique="Correct classification",
                score=0.9,
            )
        else:
            exp.feedback = CompositeFeedback(
                critique=f"Wrong classification, should be {expected}, not {predicted_label}",
                correction=expected_label,
                score=0.1,
            )
        experiences.append(exp)

 logger.info(f"✅ 收集到 {len(experiences)} 条反馈")

 # 5. 优化skill树
    config = TreeOptimizerConfig(
        auto_split=True,  # 启用拆分
        auto_prune=False,
        max_tree_depth=3,
        min_samples_for_split=5,
        prune_threshold=0.3,
        optimization_order="bottom_up",
        section="all",
    )
    tree_optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config,
    )

    # 6. 执行优化
    result = tree_optimizer.optimize_tree(
        tree=tree,
        experiences=experiences,
        validator=None,
    )

    logger.info("✅ 优化完成!")
    logger.info(f"   节点优化数: {result.nodes_optimized}")
    logger.info(f"   拆分次数: {result.splits_performed}")
    logger.info(f"   剪枝次数: {result.prunes_performed}")
    else:
        logger.warning("   没有拆分和剪枝操作")
    logger.info(f"\n{'='*60})
    logger.info(f"保存优化后的skill树...")

 # 7. 保存优化后的skill树
    output_path = Path("demo-tree-split-demo/")
    result.tree.save(output_path)
    logger.info(f"💾 优化后的skill树已保存到: {output_path}")

    # 显示树结构
    logger.info(f"🌲 优化后的树结构:")
    logger.info(f"\n{result.tree.list_tree()}")
    logger.info(f"\n✅ Demo完成!")


if __name__ == "__main__":
    main()
