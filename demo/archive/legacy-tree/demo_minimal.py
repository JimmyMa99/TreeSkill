"""最小化 Demo - 只展示基本优化流程

不使用 TreeAwareOptimizer，直接用 TrainFreeOptimizer
"""

import csv
import logging
import random
import os
from pathlib import Path
from typing import List
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

from tresskill import (
    OpenAIAdapter,
    TrainFreeOptimizer,
    OptimizerConfig,
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_data(csv_path: str, train_size: int = 15, test_size: int = 5, seed: int = 42):
    """加载数据"""
    logger.info(f"📂 加载数据: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    logger.info(f"✅ 数据总量: {len(data)} 条")
    
    labels = [item['answer'] for item in data]
    logger.info(f"📊 类别数: {len(Counter(labels))}")
    
    random.seed(seed)
    random.shuffle(data)
    
    train = data[:train_size]
    test = data[train_size:train_size + test_size]
    
    logger.info(f"训练集: {len(train)}, 测试集: {len(test)}")
    return train, test


def evaluate_prompt(adapter, prompt_text, test_data, temperature=0.3):
    """评估准确率"""
    correct = 0
    total = 0
    
    for item in test_data:
        question = item['question']
        expected = item['answer']
        
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": question},
        ]
        
        try:
            predicted = adapter._call_api(messages=messages, system=None, temperature=temperature).strip()
            if predicted.upper() == expected.upper():
                correct += 1
                logger.info(f"✅ {predicted} == {expected}")
            else:
                logger.info(f"❌ {predicted} != {expected}")
            total += 1
        except Exception as e:
            logger.error(f"⚠️  错误: {e}")
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"\n准确率: {correct}/{total} = {accuracy*100:.2f}%")
    return accuracy


def collect_experiences(adapter, prompt_text, data, temperature=0.5):
    """收集经验"""
    experiences = []
    
    for idx, item in enumerate(data):
        question = item['question']
        expected = item['answer']
        
        logger.info(f"\n[{idx+1}/{len(data)}] 处理...")
        
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": question},
        ]
        
        try:
            predicted = adapter._call_api(messages=messages, system=None, temperature=temperature).strip()
            logger.info(f"预测: {predicted}, 实际: {expected}")
            
            is_correct = predicted.upper() == expected.upper()
            
            # 创建经验
            exp = ConversationExperience(
                messages=[{"role": "user", "content": question}],
                response=predicted,
            )
            
            # 添加反馈
            if is_correct:
                exp.feedback = CompositeFeedback(
                    feedback_type=FeedbackType.POSITIVE,
                    critique="Correct",
                    score=0.9,
                )
                logger.info("✅ 正确!")
            else:
                exp.feedback = CompositeFeedback(
                    feedback_type=FeedbackType.NEGATIVE,
                    critique=f"Wrong. Should be {expected}",
                    correction=expected,
                    score=0.1,
                )
                logger.info("❌ 错误!")
            
            experiences.append(exp)
            logger.info(f"添加经验成功 (总数: {len(experiences)})")
            
        except Exception as e:
            logger.error(f"⚠️  跳过: {e}")
            import traceback
            traceback.print_exc()
    
    positive = sum(1 for e in experiences if e.feedback and e.feedback.to_score() >= 0.6)
    logger.info(f"\n收集完成: {len(experiences)} 条, 正面: {positive}")
    
    return experiences


def main():
    logger.info("\n" + "="*60)
    logger.info("🚀 最小化优化 Demo")
    logger.info("="*60)
    
    # 加载数据
    train_data, test_data = load_data("demo/data/intern_camp5.csv", train_size=15, test_size=5)
    
    # 创建适配器
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    
    main_adapter = OpenAIAdapter(model="Qwen/Qwen2.5-14B-Instruct", api_key=api_key, base_url=base_url)
    judge_adapter = OpenAIAdapter(model="Qwen/Qwen2.5-72B-Instruct", api_key=api_key, base_url=base_url)
    
    logger.info(f"\n✅ 适配器创建完成")
    
    # 初始 prompt
    initial_prompt_text = """You are a scientific paper classification expert. Classify papers into categories A-Z.

For each paper, determine which category (A-Z) it belongs to based on the content.

Return ONLY a single letter (A-Z), nothing else."""
    
    initial_prompt = TextPrompt(content=initial_prompt_text)
    
    # 评估初始准确率
    logger.info(f"\n{'='*60}")
    logger.info(f"评估初始准确率")
    logger.info(f"{'='*60}")
    initial_acc = evaluate_prompt(main_adapter, initial_prompt_text, test_data, temperature=0.3)
    
    # 收集经验
    logger.info(f"\n{'='*60}")
    logger.info(f"收集训练经验")
    logger.info(f"{'='*60}")
    experiences = collect_experiences(main_adapter, initial_prompt_text, train_data, temperature=0.5)
    
    if len(experiences) == 0:
        logger.error("❌ 没有收集到经验，退出")
        return
    
    # 优化
    logger.info(f"\n{'='*60}")
    logger.info(f"开始优化")
    logger.info(f"{'='*60}")
    
    optimizer = TrainFreeOptimizer(
        adapter=judge_adapter,
        config=OptimizerConfig(
            max_steps=1,
            gradient_accumulation_steps=5,
            conservative=False,
        ),
    )
    
    result = optimizer.optimize(
        prompt=initial_prompt,
        experiences=experiences,
        validator=None,
    )
    
    optimized_prompt = result.optimized_prompt
    optimized_text = optimized_prompt.content if hasattr(optimized_prompt, 'content') else str(optimized_prompt.to_model_input())
    
    logger.info(f"\n优化完成!")
    logger.info(f"初始 Prompt:\n{initial_prompt_text}")
    logger.info(f"\n优化后 Prompt:\n{optimized_text}")
    
    # 评估优化后准确率
    logger.info(f"\n{'='*60}")
    logger.info(f"评估优化后准确率")
    logger.info(f"{'='*60}")
    final_acc = evaluate_prompt(main_adapter, optimized_text, test_data, temperature=0.3)
    
    # 结果
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 最终结果")
    logger.info(f"{'='*60}")
    logger.info(f"初始准确率: {initial_acc*100:.2f}%")
    logger.info(f"最终准确率: {final_acc*100:.2f}%")
    logger.info(f"提升: {(final_acc-initial_acc)*100:+.2f}%")
    
    logger.info(f"\n✨ Demo 完成!")


if __name__ == "__main__":
    main()
