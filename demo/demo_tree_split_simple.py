"""
最简化的树感知优化 Demo - 演示自动拆分
只使用10条数据快速验证
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from tresskill import (
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    OptimizerConfig,
    SkillTree,
    SkillNode,
    Skill,
    SkillMeta
)
from tresskill.core.experience import ConversationExperience, CompositeFeedback


)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    """运行简化的树感知优化 Demo"""
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

    # 2. 创建初始skill树
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

Return ONLY the category letter (e.g., A, E, G, I)
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

    # 3. 收集训练反馈
    train_data = [
        {"question": "题目1...", "answer": "A"},
        {"question": "题目2...", "answer": "B"},
        # ... 10条数据
    ]
    experiences = []
    for item in train_data:
        question = item['question']
        expected = item['answer']

        # 直接创建经验
        exp = ConversationExperience(
            messages=[{"role": "user", "content": question}],
            response=expected,
            metadata={"paper_id": 0, "skill_name": "paper-classifier"},
        )
        exp.feedback = CompositeFeedback(
            critique="Correct",
            score=0.9,
        )
        experiences.append(exp)

    logger.info(f"✅ 收集到 {len(experiences)} 条反馈")

    # 4. 优化skill树
    config = TreeOptimizerConfig(
        auto_split=True,  # 启用自动拆分!
        auto_prune=False,
        max_tree_depth=3,
        min_samples_for_split=3,  # 3个样本就拆分
        prune_threshold=0.3,
        optimization_order="bottom_up",
    )
    tree_optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config,
    )
    # 5. 执行优化
    logger.info("\n" + "="*60)
    logger.info("运行树感知优化...")
    result = tree_optimizer.optimize_tree(
        tree=tree,
        experiences=experiences,
        validator=None,
    )
    logger.info("✅ 优化完成!")
    logger.info(f"   节点优化数: {result.nodes_optimized}")
    logger.info(f"   拆分次数: {result.splits_performed}")
    logger.info(f"   剋枝次数: {result.prunes_performed}")

    else:
        logger.warning("   没有拆分和剪枝操作")

    logger.info(f"\n{'='*60}")
    logger.info(f"优化后的skill树:")
    logger.info(f"\n{result.tree.list_tree()}")
    logger.info(f"\n✅ Demo完成!")


if __name__ == "__main__":
    main()
