"""修复版 Demo - 正确的 API 使用"""
import csv, logging, random, os
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

from tresskill import (
    OpenAIAdapter, TrainFreeOptimizer, OptimizerConfig, TextPrompt,
    ConversationExperience, CompositeFeedback
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path, train=10, test=5):
    with open(path, 'r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    random.shuffle(data)
    logger.info(f"数据: {len(data)}, 训练: {train}, 测试: {test}")
    return data[:train], data[train:train+test]

def evaluate(adapter, prompt_text, test_data):
    correct = 0
    for item in test_data:
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": item['question']},
        ]
        pred = adapter._call_api(messages=messages, system=None, temperature=0.3).strip()
        if pred.upper() == item['answer'].upper():
            correct += 1
            logger.info(f"✅ {pred} == {item['answer']}")
        else:
            logger.info(f"❌ {pred} != {item['answer']}")
    acc = correct / len(test_data)
    logger.info(f"准确率: {correct}/{len(test_data)} = {acc*100:.1f}%")
    return acc

def collect(adapter, prompt_text, data):
    exps = []
    for item in data:
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": item['question']},
        ]
        pred = adapter._call_api(messages=messages, system=None, temperature=0.5).strip()
        is_correct = pred.upper() == item['answer'].upper()
        
        exp = ConversationExperience(
            messages=[{"role": "user", "content": item['question']}],
            response=pred,
        )
        
        if is_correct:
            exp.feedback = CompositeFeedback(score=0.9, critique="Correct")
        else:
            exp.feedback = CompositeFeedback(
                score=0.1,
                critique=f"Wrong. Should be {item['answer']}, not {pred}",
                correction=item['answer']
            )
        
        exps.append(exp)
    
    logger.info(f"收集: {len(exps)} 条")
    return exps

def main():
    logger.info("🚀 论文分类优化 Demo")
    
    train, test = load_data("demo/data/intern_camp5.csv")
    
    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    main_adapter = OpenAIAdapter(model="Qwen/Qwen2.5-14B-Instruct", api_key=api_key, base_url=base_url)
    judge_adapter = OpenAIAdapter(model="Qwen/Qwen2.5-72B-Instruct", api_key=api_key, base_url=base_url)
    
    prompt_text = """Classify papers into categories A-Z.
Return ONLY a single letter (A-Z), nothing else."""
    
    logger.info("\n📊 初始评估")
    initial = evaluate(main_adapter, prompt_text, test)
    
    logger.info("\n📚 收集经验")
    exps = collect(main_adapter, prompt_text, train)
    
    logger.info("\n🔧 优化中...")
    opt = TrainFreeOptimizer(
        adapter=judge_adapter,
        config=OptimizerConfig(max_steps=1, gradient_accumulation_steps=5),
    )
    result = opt.optimize(
        prompt=TextPrompt(content=prompt_text),
        experiences=exps,
        validator=None,
    )
    
    new_prompt = result.optimized_prompt.content if hasattr(result.optimized_prompt, 'content') else str(result.optimized_prompt.to_model_input())
    
    logger.info(f"\n优化后 prompt:\n{new_prompt}")
    
    logger.info("\n📊 最终评估")
    final = evaluate(main_adapter, new_prompt, test)
    
    logger.info(f"\n📊 结果: 初始 {initial*100:.1f}% → 最终 {final*100:.1f}% (提升 {(final-initial)*100:+.1f}%)")
    logger.info("✨ 完成!")

if __name__ == "__main__":
    main()
