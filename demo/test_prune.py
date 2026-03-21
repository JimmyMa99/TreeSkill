#!/usr/bin python3
"""
简化版 Tree Demo - test剪枝功能

不需要数据集，直接模拟test
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill import (
    OpenAIAdapter,
    TreeAwareOptimizer,
    TreeOptimizerConfig,
    OptimizerConfig,
    SkillTree,
    ConversationExperience,
    CompositeFeedback,
)
from evoskill.schema import Skill, SkillMeta
from evoskill.skill_tree import SkillNode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_mock_experiences(n=50):
    """Create模拟经验数据"""
    experiences = []
    categories = ['A', 'E', 'G', 'K', 'M']

    for i in range(n):
        exp = ConversationExperience(
            messages=[{"role": "user", "content": f"Test question {i}"}],
            response=categories[i % 5],
            metadata={"paper_id": i},
        )

        # 前 30 条是成功经验
        if i < 30:
            exp.feedback = CompositeFeedback(critique="Good", score=0.9)
        else:
            # 后 20 条是failure experience，显示需要拆分
            exp.feedback = CompositeFeedback(
                critique=f"Wrong category. Should be {categories[(i+1) % 5]}",
                correction=categories[(i+1) % 5],
                score=0.1,
            )

        experiences.append(exp)

    return experiences


def optimize_round(adapter, tree, experiences, round_name, enable_prune=True):
    """优化一轮"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 {round_name}")
    logger.info(f"   Auto-Prune: {'✅' if enable_prune else '❌'}")
    logger.info(f"{'='*60}")

    config = TreeOptimizerConfig(
        auto_split=True,
        auto_prune=enable_prune,
        prune_strategy="moderate",
        prune_protection_rounds=1,
        prune_usage_threshold=1,
        collapse_instead_of_prune=True,
        max_tree_depth=3,
        min_samples_for_split=3,
        prune_threshold=0.3,
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

    result = optimizer.optimize_tree(tree=tree, experiences=experiences)

    logger.info(f"✅ Optimization complete:")
    logger.info(f"   节点优化: {result.nodes_optimized}")
    logger.info(f"   拆分次数: {result.splits_performed}")
    logger.info(f"   剪枝次数: {result.prunes_performed}")

    return result.tree


def count_nodes(node):
    """统计节点数量"""
    count = 1
    for child in node.children.values():
        count += count_nodes(child)
    return count


def main():
    """主Flow"""
    logger.info("\n" + "="*60)
    logger.info("🌳 简化版 Tree 剪枝test")
    logger.info("="*60)

    # Create适配器
    api_key = os.getenv("EVO_LLM_API_KEY")
    base_url = os.getenv("EVO_LLM_BASE_URL", "https://api.siliconflow.cn/v1")

    judge_model = "Qwen/Qwen2.5-72B-Instruct"

    if not api_key:
        logger.error("❌ 请Set EVO_LLM_API_KEY")
        return

    judge_adapter = OpenAIAdapter(model=judge_model, api_key=api_key, base_url=base_url)

    # Create初始skill树
    poor_prompt = "Classify papers into categories. Return A, E, G, K, or M."

    root_skill = Skill(
        name="paper-classifier",
        system_prompt=poor_prompt,
        version="v1.0",
        meta=SkillMeta(name="paper-classifier", description="Paper classifier"),
    )

    output_path = Path("demo/outputs/demo-qwen3-8b-tree-pruned/")
    tree = SkillTree(
        root=SkillNode(name="root", skill=root_skill),
        base_path=output_path,
    )

    logger.info(f"\n⚠️  初始prompt: {poor_prompt}")
    logger.info(f"📁 输出目录: {output_path}")

    # 第1轮：只拆分，不剪枝
    logger.info(f"\n{'='*60}")
    logger.info("🔄 第1轮: 只拆分")
    logger.info(f"   Auto-Prune: ❌")
    logger.info(f"{'='*60}")

    experiences = create_mock_experiences(50)
    tree = optimize_round(judge_adapter, tree, experiences, "第1轮", enable_prune=False)

    logger.info(f"\n📊 Tree统计:")
    logger.info(f"   总节点数: {count_nodes(tree.root)}")
    logger.info(f"   子节点数: {len(tree.root.children)}")
    logger.info(tree.list_tree())

    # 第2轮: 启用剪枝
    logger.info(f"\n{'='*60}")
    logger.info("🔄 第2轮: 启用剪枝")
    logger.info(f"   Auto-Prune: ✅")
    logger.info(f"{'='*60}")

    experiences = create_mock_experiences(30)
    tree = optimize_round(judge_adapter, tree, experiences, "第2轮", enable_prune=True)

    logger.info(f"\n📊 Tree统计:")
    logger.info(f"   总节点数: {count_nodes(tree.root)}")
    logger.info(f"   子节点数: {len(tree.root.children)}")
    logger.info(tree.list_tree())

    # 第3轮: 再次剪枝
    logger.info(f"\n{'='*60}")
    logger.info("🔄 第3轮: 再次剪枝")
    logger.info(f"   Auto-Prune: ✅")
    logger.info(f"{'='*60}")

    experiences = create_mock_experiences(20)
    tree = optimize_round(judge_adapter, tree, experiences, "第3轮", enable_prune=True)

    logger.info(f"\n📊 最终 Tree统计:")
    logger.info(f"   总节点数: {count_nodes(tree.root)}")
    logger.info(f"   子节点数: {len(tree.root.children)}")
    logger.info(tree.list_tree())

    # Save
    tree.save(output_path)
    logger.info(f"\n💾 已Save到: {output_path}")

    logger.info(f"\n✅ Demo完成!")


if __name__ == "__main__":
    main()
