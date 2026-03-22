"""完全平衡策略 - 5 轮优化

使用固定的温度 0.5 和平衡策略，"""
import csv
import logging
import random
import os
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from tresskill import (
    OpenAIAdapter,
    TrainFreeOptimizer,
    OptimizerConfig,
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_data(path, train=100, test=30):
    """加载数据"""
    with open(path, 'r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    
    random.seed(42)
    random.shuffle(data)
    
    logger.info(f"📂 数据: {len(data)} 条 (训练 {train}, 测试 {test})")
    return data[:train], data[train:train+test]


def evaluate(adapter, prompt_text, test_data, temp=0.3):
    """评估"""
    correct = 0
    for item in test_data:
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": item['question']},
        ]
        pred = adapter._call_api(messages=messages, system=None, temperature=temp).strip()
        if pred.upper() == item['answer'].upper():
            correct += 1
    acc = correct / len(test_data)
    logger.info(f"📊 准确率: {correct}/{len(test_data)} = {acc*100:.1f}%")
    return acc


def collect(adapter, prompt_text, data, temp=0.5):
    """收集经验"""
    exps = []
    for item in data:
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": item['question']},
        ]
        pred = adapter._call_api(messages=messages, system=None, temperature=temp).strip()
        
        exp = ConversationExperience(
            messages=[{"role": "user", "content": item['question']}],
            response=pred,
        )
        
        if pred.upper() == item['answer'].upper():
            exp.feedback = CompositeFeedback(score=0.9, critique="Correct")
        else:
            exp.feedback = CompositeFeedback(
                score=0.1,
                critique=f"Wrong: {pred} != {item['answer']}",
                correction=item['answer']
            )
        
        exps.append(exp)
    
    positive = sum(1 for e in exps if e.feedback.to_score() >= 0.6)
    logger.info(f"收集: {len(exps)} 条 (正面 {positive})")
    return exps


def main():
    logger.info("\n" + "="*60)
    logger.info("🎯 完全平衡策略 - 5 轮优化")
    logger.info("="*60)
    logger.info("策略: 固定温度 0.5 + 平衡策略")
    logger.info("数据: 100 训练 + 30 测试")
    logger.info("轮次: 5 轮优化")
    
    # 加载数据
    train, test = load_data("demo/data/intern_camp5.csv", train=100, test=30)
    
    # 创建适配器
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    
    main_adapter = OpenAIAdapter(
        model="Qwen/Qwen2.5-14B-Instruct",
        api_key=api_key,
        base_url=base_url,
    )
    
    judge_adapter = OpenAIAdapter(
        model="Qwen/Qwen2.5-72B-Instruct",
        api_key=api_key,
        base_url=base_url,
    )
    
    logger.info("✅ 适配器创建完成")
    
    # 初始 prompt
    prompt_text = """You are a scientific paper classifier. Classify papers into categories A-Z.

Read the title and abstract, identify the main research area, and return the appropriate category letter (A-Z).

Return ONLY a single letter (A-Z), nothing else."""
    
    prompt = TextPrompt(content=prompt_text)
    
    # 初始评估
    logger.info(f"\n{'='*60}")
    logger.info("📊 初始评估")
    logger.info(f"{'='*60}")
    initial_acc = evaluate(main_adapter, prompt_text, test, temp=0.3)
    
    best_acc = initial_acc
    best_prompt_text = prompt_text
    history = [{"round": 0, "accuracy": initial_acc, "temperature": 0.3}]
    
    # 5 轮优化
    for round in range(1, 6):
        logger.info(f"\n\n{'='*60}")
        logger.info(f"🔄 第 {round}/5 轮优化")
        logger.info(f"{'='*60}")
        
        # 使用不同子集
        start_idx = (round - 1) * 20
        end_idx = start_idx + 20
        subset = train[start_idx:end_idx]
        
        # 收集经验（固定温度 0.5）
        exps = collect(main_adapter, prompt_text, subset, temp=0.5)
        
        # 优化（平衡策略）
        logger.info(f"🔧 优化中...")
        optimizer = TrainFreeOptimizer(
            adapter=judge_adapter,
            config=OptimizerConfig(
                max_steps=1,
                gradient_accumulation_steps=8,
                conservative=False,  # 平衡策略
            ),
        )
        
        result = optimizer.optimize(
            prompt=prompt,
            experiences=exps,
            validator=None,
        )
        
        # 更新 prompt
        prompt = result.optimized_prompt
        prompt_text = prompt.content if hasattr(prompt, 'content') else str(prompt.to_model_input())
        
        # 评估
        new_acc = evaluate(main_adapter, prompt_text, test, temp=0.3)
        
        # 记录
        history.append({
            "round": round,
            "accuracy": new_acc,
            "temperature": 0.5,
            "delta": new_acc - history[-1]["accuracy"],
        })
        
        # 更新最佳
        if new_acc > best_acc:
            best_acc = new_acc
            best_prompt_text = prompt_text
            logger.info(f"✅ 新最佳: {best_acc*100:.1f}%")
        
        # 检查是否达到 80%
        if new_acc >= 0.8:
            logger.info(f"🎉 达到目标准确率 80%，提前结束！")
            break
    
    # 最终结果
    logger.info(f"\n\n{'='*60}")
    logger.info(f"📊 最终结果")
    logger.info(f"{'='*60}")
    logger.info(f"初始准确率: {initial_acc*100:.1f}%")
    logger.info(f"最终准确率: {best_acc*100:.1f}%")
    logger.info(f"总提升: {(best_acc-initial_acc)*100:+.1f}%")
    
    logger.info(f"\n📈 优化轨迹:")
    for h in history:
        delta = ""
        if "delta" in h:
            sign = "+" if h["delta"] > 0 else ""
            delta = f" ({sign}{h['delta']*100:.1f}%)"
        logger.info(
            f"   Round {h['round']}: {h['accuracy']*100:.1f}% "
            f"(temp={h['temperature']}) {delta}"
        )
    
    # 保存
    with open("demo_balanced_history.json", 'w') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    with open("demo_balanced_best_prompt.txt", 'w') as f:
        f.write(best_prompt_text)
    
    logger.info(f"\n💾 历史已保存到: demo_balanced_history.json")
    logger.info(f"💾 最佳 prompt 已保存到: demo_balanced_best_prompt.txt")
    
    if best_acc > initial_acc:
        logger.info(f"\n✨ 优化成功！提升 {(best_acc-initial_acc)*100:+.1f}%")
    else:
        logger.info(f"\n⚠️  准确率未提升")
    
    logger.info(f"\n✨ Demo 完成!")


if __name__ == "__main__":
    main()
