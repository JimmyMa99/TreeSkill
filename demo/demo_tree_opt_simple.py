"""简化版论文分类树感知优化 Demo

这个 demo 使用单个字母标签（A-Z）格式
"""

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
    FeedbackType,
    SkillTree,
)
from tresskill.schema import Skill, SkillMeta
from tresskill.skill_tree import SkillNode

# 日志
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
    logger.info(f"   前5个类别: {label_counts.most_common(5)}")
    
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


def create_initial_tree() -> SkillTree:
    """创建初始 skill 树"""
    root_prompt = """You are a scientific paper classification expert. Classify papers into one of 26 categories (A-Z).

For each paper, read the description carefully and determine which category (A-Z) it belongs to.

Return ONLY a single letter (A-Z), nothing else.
    """
    
    root_skill = Skill(
        name="paper-classifier",
        system_prompt=root_prompt,
        version="v1.0",
        meta=SkillMeta(
            name="paper-classifier",
            description="Root paper classifier",
        ),
    )
    
    tree = SkillTree(
        root=SkillNode(name="paper-classifier", skill=root_skill),
        base_path=Path("demo-paper-tree-simple/"),
    )
    
    logger.info(f"✅ 初始 Skill 树创建完成")
    return tree


def evaluate(adapter, tree, test_data, temperature=0.3):
    """评估准确率"""
    logger.info(f"\n{'='*60}")
    logger.info(f"评估准确率 (temperature={temperature})")
    logger.info(f"{'='*60}")
    
    system_prompt = tree.root.skill.system_prompt
    correct = 0
    total = 0
    
    for idx, item in enumerate(test_data):
        question = item['question']
        expected = item['answer']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        try:
            predicted = adapter._call_api(messages=messages, system=None, temperature=temperature).strip()
            is_correct = predicted.upper() == expected.upper()
            
            if is_correct:
                correct += 1
                logger.info(f"[{idx+1}/{len(test_data)}] ✅ {predicted} == {expected}")
            else:
                logger.info(f"[{idx+1}/{len(test_data)}] ❌ {predicted} != {expected}")
            
            total += 1
        except Exception as e:
            logger.error(f"[{idx+1}] ⚠️  错误: {e}")
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"\n准确率: {correct}/{total} = {accuracy*100:.2f}%")
    return accuracy


def collect_experiences(adapter, data, temperature=0.5):
    """收集经验"""
    logger.info(f"\n{'='*60}")
    logger.info(f"收集经验 (temperature={temperature})")
    logger.info(f"{'='*60}")
    
    system_prompt = """You are a scientific paper classification expert. Classify papers into one of 26 categories (A-Z).

Return ONLY a single letter (A-Z), nothing else."""
    
    experiences = []
    
    for idx, item in enumerate(data):
        question = item['question']
        expected = item['answer']
        
        logger.info(f"\n[{idx+1}/{len(data)}] 处理论文...")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        try:
            predicted = adapter._call_api(messages=messages, system=None, temperature=temperature).strip()
            logger.info(f"   预测: {predicted}, 实际: {expected}")
            
            is_correct = predicted.upper() == expected.upper()
            
            exp = ConversationExperience(
                messages=[{"role": "user", "content": question}],
                response=predicted,
                metadata={"paper_id": idx},
            )
            
            if is_correct:
                exp.feedback = CompositeFeedback(
                    feedback_type=FeedbackType.POSITIVE,
                    critique="Correct classification",
                    score=0.9,
                )
                logger.info(f"   ✅ 正确!")
            else:
                exp.feedback = CompositeFeedback(
                    feedback_type=FeedbackType.NEGATIVE,
                    critique=f"Wrong. Should be {expected}, not {predicted}",
                    correction=expected,
                    score=0.1,
                )
                logger.info(f"   ❌ 错误!")
            
            experiences.append(exp)
        except Exception as e:
            logger.error(f"   ⚠️  跳过: {e}")
    
    positive = sum(1 for e in experiences if e.feedback.to_score() >= 0.6)
    logger.info(f"\n收集完成: {len(experiences)} 条, 正面: {positive}, 负面: {len(experiences)-positive}")
    
    return experiences


def optimize(adapter, tree, experiences, temperature=0.3):
    """优化"""
    logger.info(f"\n{'='*60}")
    logger.info(f"树感知优化 (temperature={temperature})")
    logger.info(f"{'='*60}")
    
    config = TreeOptimizerConfig(
        auto_split=False,  # 简化版不拆分
        auto_prune=False,  # 简化版不剪枝
        section="all",
    )
    
    optimizer = TreeAwareOptimizer(
        adapter=adapter,
        config=config,
        base_optimizer_config=OptimizerConfig(
            max_steps=1,
            gradient_accumulation_steps=5,
            strategy="adaptive",
        ),
    )
    
    result = optimizer.optimize_tree(tree=tree, experiences=experiences, validator=None)
    
    logger.info(f"\n优化完成: 优化了 {result.nodes_optimized} 个节点")
    return result.tree


def main():
    logger.info("\n" + "="*60)
    logger.info("🚀 简化版论文分类树感知优化 Demo")
    logger.info("="*60)
    
    # 加载数据
    train_data, test_data = load_data("demo/data/intern_camp5.csv", train_size=20, test_size=10)
    
    # 创建适配器
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    main_model = "Qwen/Qwen2.5-14B-Instruct"
    judge_model = "Qwen/Qwen2.5-72B-Instruct"
    
    main_adapter = OpenAIAdapter(model=main_model, api_key=api_key, base_url=base_url)
    judge_adapter = OpenAIAdapter(model=judge_model, api_key=api_key, base_url=base_url)
    
    logger.info(f"\n✅ API 适配器创建完成")
    logger.info(f"   主模型: {main_model}")
    logger.info(f"   Judge模型: {judge_model}")
    
    # 创建树
    tree = create_initial_tree()
    
    # 评估初始准确率
    initial_acc = evaluate(main_adapter, tree, test_data, temperature=0.3)
    logger.info(f"\n📊 初始准确率: {initial_acc*100:.2f}%")
    
    # 第1轮优化
    logger.info(f"\n{'='*60}")
    logger.info(f"第 1 轮优化")
    logger.info(f"{'='*60}")
    
    experiences_1 = collect_experiences(main_adapter, train_data[:10], temperature=0.3)
    tree = optimize(judge_adapter, tree, experiences_1, temperature=0.3)
    acc_1 = evaluate(main_adapter, tree, test_data, temperature=0.3)
    logger.info(f"\n📊 第1轮准确率: {acc_1*100:.2f}% (提升 {(acc_1-initial_acc)*100:+.2f}%)")
    
    # 如果第1轮有提升，做第2轮
    if acc_1 > initial_acc:
        logger.info(f"\n{'='*60}")
        logger.info(f"第 2 轮优化（提高温度探索）")
        logger.info(f"{'='*60}")
        
        experiences_2 = collect_experiences(main_adapter, train_data[10:20], temperature=0.5)
        tree = optimize(judge_adapter, tree, experiences_2, temperature=0.5)
        acc_2 = evaluate(main_adapter, tree, test_data, temperature=0.3)
        logger.info(f"\n📊 第2轮准确率: {acc_2*100:.2f}% (提升 {(acc_2-acc_1)*100:+.2f}%)")
        
        final_acc = max(acc_1, acc_2)
    else:
        final_acc = acc_1
    
    # 最终结果
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 最终结果")
    logger.info(f"{'='*60}")
    logger.info(f"初始准确率: {initial_acc*100:.2f}%")
    logger.info(f"最终准确率: {final_acc*100:.2f}%")
    logger.info(f"总提升: {(final_acc-initial_acc)*100:+.2f}%")
    
    # 保存
    tree.save(Path("demo-paper-tree-optimized-simple/"))
    logger.info(f"\n💾 优化后的树已保存")
    
    logger.info(f"\n✨ Demo 完成!")


if __name__ == "__main__":
    main()
