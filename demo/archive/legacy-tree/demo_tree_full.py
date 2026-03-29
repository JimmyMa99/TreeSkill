"""增强版论文分类优化 Demo

增强特性：
1. 更多的训练数据（50条）
2. 更详细的初始 prompt
3. 3轮渐进式优化（温度调节）
4. 实时展示优化过程
5. 保存优化历史
"""

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


class OptimizationTracker:
    """追踪优化过程"""
    
    def __init__(self):
        self.history = []
        self.start_time = datetime.now()
    
    def record(self, round_num, accuracy, prompt_text, temperature, exp_count):
        """记录一轮优化"""
        self.history.append({
            "round": round_num,
            "accuracy": accuracy,
            "prompt": prompt_text,
            "temperature": temperature,
            "experiences": exp_count,
            "timestamp": datetime.now().isoformat(),
        })
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 第 {round_num} 轮结果")
        logger.info(f"{'='*60}")
        logger.info(f"   准确率: {accuracy*100:.1f}%")
        logger.info(f"   经验数: {exp_count}")
        logger.info(f"   温度: {temperature}")
        logger.info(f"   用时: {elapsed:.1f}秒")
        
        # 显示趋势
        if len(self.history) > 1:
            prev_acc = self.history[-2]["accuracy"]
            delta = accuracy - prev_acc
            logger.info(f"   趋势: {delta*100:+.1f}%")
    
    def save(self, path="demo_optimization_history.json"):
        """保存优化历史"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        logger.info(f"\n💾 优化历史已保存到: {path}")


def load_data(csv_path, train_size=50, test_size=20, seed=42):
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
    
    logger.info(f"\n✅ 数据分割:")
    logger.info(f"   训练集: {len(train)} 条")
    logger.info(f"   测试集: {len(test)} 条")
    
    # 统计训练集类别分布
    train_labels = [item['answer'] for item in train]
    train_dist = Counter(train_labels)
    logger.info(f"   训练集类别分布: {len(train_dist)} 个类别")
    logger.info(f"   最多: {train_dist.most_common(3)}")
    
    return train, test


def evaluate(adapter, prompt_text, test_data, temperature=0.3, show_details=True):
    """评估准确率"""
    if show_details:
        logger.info(f"\n{'='*60}")
        logger.info(f"评估准确率 (temperature={temperature})")
        logger.info(f"{'='*60}")
    
    correct = 0
    total = 0
    results = []
    
    for item in test_data:
        question = item['question']
        expected = item['answer']
        
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": question},
        ]
        
        try:
            predicted = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature
            ).strip()
            
            is_correct = predicted.upper() == expected.upper()
            
            if is_correct:
                correct += 1
                if show_details:
                    logger.info(f"✅ {predicted} == {expected}")
            else:
                if show_details:
                    logger.info(f"❌ {predicted} != {expected}")
            
            total += 1
            results.append({
                "question": question[:80],
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct,
            })
            
        except Exception as e:
            logger.error(f"⚠️  错误: {e}")
    
    accuracy = correct / total if total > 0 else 0.0
    
    if show_details:
        logger.info(f"\n准确率: {correct}/{total} = {accuracy*100:.2f}%")
    
    return accuracy, results


def collect_experiences(adapter, prompt_text, data, temperature=0.5, verbose=True):
    """收集经验"""
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"收集经验 (temperature={temperature})")
        logger.info(f"{'='*60}")
    
    experiences = []
    
    for idx, item in enumerate(data):
        question = item['question']
        expected = item['answer']
        
        if verbose and (idx + 1) % 10 == 0:
            logger.info(f"处理进度: {idx+1}/{len(data)}")
        
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": question},
        ]
        
        try:
            predicted = adapter._call_api(
                messages=messages,
                system=None,
                temperature=temperature
            ).strip()
            
            is_correct = predicted.upper() == expected.upper()
            
            # 创建经验
            exp = ConversationExperience(
                messages=[{"role": "user", "content": question}],
                response=predicted,
            )
            
            # 添加反馈
            if is_correct:
                exp.feedback = CompositeFeedback(
                    score=0.9,
                    critique="Correct classification",
                )
            else:
                exp.feedback = CompositeFeedback(
                    score=0.1,
                    critique=f"Wrong. Should be {expected}, not {predicted}",
                    correction=expected,
                )
            
            experiences.append(exp)
            
        except Exception as e:
            logger.error(f"⚠️  跳过样本 {idx}: {e}")
    
    positive = sum(1 for e in experiences if e.feedback.to_score() >= 0.6)
    
    if verbose:
        logger.info(f"\n收集完成:")
        logger.info(f"   总数: {len(experiences)} 条")
        logger.info(f"   正面: {positive} ({positive/len(experiences)*100:.1f}%)")
        logger.info(f"   负面: {len(experiences)-positive} ({(len(experiences)-positive)/len(experiences)*100:.1f}%)")
    
    return experiences


def optimize_round(adapter, prompt, experiences, round_num):
    """执行一轮优化"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔧 第 {round_num} 轮优化")
    logger.info(f"{'='*60}")
    
    # 根据轮次调整配置
    if round_num == 1:
        # 第一轮：保守
        config = OptimizerConfig(
            max_steps=1,
            gradient_accumulation_steps=5,
            conservative=True,
        )
        temp_desc = "保守优化（学习基本模式）"
    elif round_num == 2:
        # 第二轮：平衡
        config = OptimizerConfig(
            max_steps=1,
            gradient_accumulation_steps=8,
            conservative=False,
        )
        temp_desc = "平衡优化（探索改进空间）"
    else:
        # 第三轮：激进
        config = OptimizerConfig(
            max_steps=2,
            gradient_accumulation_steps=10,
            conservative=False,
        )
        temp_desc = "激进优化（最大化提升）"
    
    logger.info(f"策略: {temp_desc}")
    
    optimizer = TrainFreeOptimizer(
        adapter=adapter,
        config=config,
    )
    
    result = optimizer.optimize(
        prompt=prompt,
        experiences=experiences,
        validator=None,
    )
    
    new_prompt = result.optimized_prompt
    new_text = new_prompt.content if hasattr(new_prompt, 'content') else str(new_prompt.to_model_input())
    
    logger.info(f"\n优化完成!")
    logger.info(f"   优化步数: {result.steps_taken}")
    
    # 显示 prompt 变化
    old_text = prompt.content if hasattr(prompt, 'content') else str(prompt.to_model_input())
    if len(new_text) < len(old_text) * 0.8:
        logger.info(f"   Prompt 简化: {len(old_text)} → {len(new_text)} 字符")
    elif len(new_text) > len(old_text) * 1.2:
        logger.info(f"   Prompt 扩展: {len(old_text)} → {len(new_text)} 字符")
    else:
        logger.info(f"   Prompt 长度相近: {len(old_text)} → {len(new_text)} 字符")
    
    return new_prompt, new_text


def main():
    logger.info("\n" + "="*60)
    logger.info("🚀 增强版论文分类优化 Demo")
    logger.info("="*60)
    logger.info("特性:")
    logger.info("  ✅ 50条训练数据")
    logger.info("  ✅ 3轮渐进式优化")
    logger.info("  ✅ 温度调节策略")
    logger.info("  ✅ 实时效果追踪")
    
    # 追踪器
    tracker = OptimizationTracker()
    
    # 加载数据
    train_data, test_data = load_data(
        "demo/data/intern_camp5.csv",
        train_size=50,
        test_size=20,
    )
    
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
    
    logger.info(f"\n✅ 适配器创建完成")
    logger.info(f"   主模型: Qwen/Qwen2.5-14B-Instruct")
    logger.info(f"   Judge模型: Qwen/Qwen2.5-72B-Instruct")
    
    # 初始 prompt（更详细的版本）
    initial_prompt_text = """You are an expert scientific paper classifier. Your task is to categorize research papers into 26 scientific domains (A-Z).

For each paper:
1. Read the title and abstract carefully
2. Identify the main research area and methodology
3. Match it to the most appropriate category (A-Z)

Categories cover: AI/ML, Biology, Chemistry, Physics, Mathematics, Engineering, Medicine, and 19 other scientific domains.

Important:
- Return ONLY a single letter (A-Z)
- No explanations or additional text
- Be precise and consistent

Return your answer as a single letter."""
    
    initial_prompt = TextPrompt(content=initial_prompt_text)
    
    # 评估初始准确率
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 初始评估")
    logger.info(f"{'='*60}")
    
    initial_acc, _ = evaluate(main_adapter, initial_prompt_text, test_data, temperature=0.3)
    tracker.record(0, initial_acc, initial_prompt_text, 0.3, 0)
    
    best_acc = initial_acc
    best_prompt = initial_prompt
    best_prompt_text = initial_prompt_text
    
    # 多轮优化
    current_prompt = initial_prompt
    current_prompt_text = initial_prompt_text
    
    for round_num in range(1, 4):  # 3轮
        logger.info(f"\n\n{'='*60}")
        logger.info(f"🔄 第 {round_num}/3 轮优化")
        logger.info(f"{'='*60}")
        
        # 根据轮次调整温度
        if round_num == 1:
            temperature = 0.3  # 低温度，确定性收集
        elif round_num == 2:
            temperature = 0.5  # 中等温度，探索
        else:
            temperature = 0.6  # 高温度，更多探索
        
        # 收集经验
        # 每轮使用不同子集
        start_idx = (round_num - 1) * 15
        end_idx = start_idx + 20
        subset = train_data[start_idx:end_idx]
        
        experiences = collect_experiences(
            main_adapter,
            current_prompt_text,
            subset,
            temperature=temperature,
        )
        
        if len(experiences) == 0:
            logger.warning(f"⚠️  第 {round_num} 轮跳过（没有经验）")
            continue
        
        # 优化
        new_prompt, new_text = optimize_round(
            judge_adapter,
            current_prompt,
            experiences,
            round_num,
        )
        
        # 评估
        new_acc, _ = evaluate(main_adapter, new_text, test_data, temperature=0.3, show_details=False)
        
        # 记录
        tracker.record(round_num, new_acc, new_text, temperature, len(experiences))
        
        # 更新最佳
        if new_acc > best_acc:
            best_acc = new_acc
            best_prompt = new_prompt
            best_prompt_text = new_text
            logger.info(f"✅ 新的最佳准确率: {best_acc*100:.1f}%")
        
        # 早停条件
        if new_acc >= 0.8:  # 达到 80% 准确率
            logger.info(f"\n🎉 达到目标准确率（80%），提前结束！")
            break
        
        # 如果连续下降，也停止
        if round_num > 1:
            if (new_acc < tracker.history[-2]["accuracy"] and 
                tracker.history[-2]["accuracy"] < tracker.history[-3]["accuracy"] if len(tracker.history) >= 3 else False):
                logger.warning(f"\n⚠️  准确率连续下降，停止优化")
                break
        
        # 更新当前 prompt
        current_prompt = new_prompt
        current_prompt_text = new_text
    
    # 最终结果
    logger.info(f"\n\n{'='*60}")
    logger.info(f"📊 最终结果")
    logger.info(f"{'='*60}")
    logger.info(f"初始准确率: {initial_acc*100:.1f}%")
    logger.info(f"最终准确率: {best_acc*100:.1f}%")
    logger.info(f"总提升: {(best_acc-initial_acc)*100:+.1f}%")
    
    # 优化历史
    logger.info(f"\n📈 优化轨迹:")
    for record in tracker.history:
        delta = ""
        if record["round"] > 0:
            prev = tracker.history[record["round"]-1]["accuracy"]
            delta = f" ({(record['accuracy']-prev)*100:+.1f}%)"
        logger.info(
            f"   Round {record['round']}: {record['accuracy']*100:.1f}% "
            f"(temp={record['temperature']}, exps={record['experiences']}){delta}"
        )
    
    # 保存最佳 prompt
    logger.info(f"\n📝 最佳 Prompt:")
    logger.info(f"{best_prompt_text}")
    
    # 保存历史
    tracker.save("demo_optimization_history.json")
    
    # 保存最佳 prompt
    with open("demo_best_prompt.txt", 'w', encoding='utf-8') as f:
        f.write(best_prompt_text)
    logger.info(f"💾 最佳 prompt 已保存到: demo_best_prompt.txt")
    
    # 判断成功
    if best_acc > initial_acc:
        logger.info(f"\n✨ 优化成功！准确率提升了 {(best_acc-initial_acc)*100:.1f}%")
    else:
        logger.info(f"\n⚠️  准确率未提升，但优化流程正常运行")
    
    logger.info(f"\n✨ Demo 完成!")


if __name__ == "__main__":
    main()
