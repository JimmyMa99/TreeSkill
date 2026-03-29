"""
论文分类真实 API 优化 Demo - 使用硅流动 API

使用硅流动 API (OpenAI 格式) 进行论文分类和优化

训练集: 30 条
验证集: 10 条
测试集: 10 条

优化流程:
1. 加载数据
2. 创建初始 skill (简单 prompt)
3. 在训练集上运行，收集失败案例
4. 使用 TrainFreeOptimizer 优化 prompt
5. 在验证集和测试集上评估
6. 监控优化效果
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
    TrainFreeOptimizer,
    OptimizerConfig,
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
)

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Step 1: 数据加载
# ============================================================================

def load_data(csv_path: str, train_size: int = 100, eval_size: int = 20, test_size: int = 20, seed: int = 42):
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

    # 统计分割后的类别分布
    train_labels = [item['answer'] for item in train]
    logger.info(f"\n训练集类别分布: {Counter(train_labels)}")

    return train, eval_set, test


# ============================================================================
# Step 2: 创建分类器
# ============================================================================

def create_initial_prompt() -> str:
    """创建改进的分类 prompt"""
    prompt = """You are an expert scientific paper classifier. Your task is to classify papers into 26 categories based on their main contribution, methodology, and research domain.

## Categories (A-Z):

**A: Quantum Physics** (quant-ph)
- Keywords: quantum, qubit, entanglement, superposition, quantum computing, quantum error correction
- Focus: Quantum mechanics applications and quantum information

**B: Chemical Physics** (physics.chem-ph)
- Keywords: molecular dynamics, chemical reactions, spectroscopy, computational chemistry
- Focus: Physics applied to chemical systems

**C: Atomic Physics** (physics.atom-ph)
- Keywords: atomic structure, laser cooling, atomic clocks, Bose-Einstein condensate
- Focus: Properties and behavior of atoms

**D: Soft Condensed Matter** (cond-mat.soft)
- Keywords: polymers, liquid crystals, foams, gels, granular materials
- Focus: Soft materials and complex fluids

**E: Robotics** (cs.RO)
- Keywords: robot, manipulation, navigation, autonomous systems, SLAM
- Focus: Robot hardware, control, and autonomy

**F: Computation and Language** (cs.CL)
- Keywords: NLP, text mining, language model, machine translation, sentiment
- Focus: Natural language processing

**G: Software Engineering** (cs.SE)
- Keywords: software testing, debugging, refactoring, version control, CI/CD
- Focus: Software development practices

**H: Information Retrieval** (cs.IR)
- Keywords: search engine, ranking, recommendation, document retrieval
- Focus: Finding and ranking information

**I: High Energy Physics - Theory** (hep-th)
- Keywords: string theory, particle physics, quantum field theory, cosmology
- Focus: Theoretical particle physics

**J: High Energy Physics - Phenomenology** (hep-ph)
- Keywords: particle collisions, detector physics, LHC, particle phenomenology
- Focus: Experimental particle physics predictions

**K: Optics** (physics.optics)
- Keywords: laser, photonics, optical fibers, metamaterials, light
- Focus: Light and optical systems

**L: Artificial Intelligence** (cs.AI)
- Keywords: machine learning, neural networks, reinforcement learning, reasoning
- Focus: AI algorithms and systems (general)

**M: Computer Vision** (cs.CV)
- Keywords: image recognition, object detection, segmentation, visual
- Focus: Understanding images and video

**N: Nuclear Theory** (nucl-th)
- Keywords: nuclear structure, nuclear reactions, nuclear forces
- Focus: Theory of atomic nuclei

**O: Astrophysics** (astro-ph)
- Keywords: stars, galaxies, black holes, cosmology, dark matter
- Focus: Physics of celestial objects

**P: Probability** (math.PR)
- Keywords: probability theory, random processes, stochastic
- Focus: Mathematical probability

**Q: Operating Systems** (cs.OS)
- Keywords: kernel, scheduling, memory management, file systems
- Focus: OS design and implementation

**R: Signal Processing** (eess.SP)
- Keywords: filter, Fourier transform, audio processing, image processing
- Focus: Processing signals (audio, video, etc.)

**S: Optimization and Control** (math.OC)
- Keywords: convex optimization, linear programming, optimal control
- Focus: Mathematical optimization methods

**T: Dynamical Systems** (math.DS)
- Keywords: chaos, bifurcation, nonlinear dynamics, attractor
- Focus: Systems that change over time

**U: Differential Geometry** (math.DG)
- Keywords: manifolds, curvature, differential forms, Riemannian
- Focus: Geometry using calculus

**V: Mathematical Physics** (math.MP)
- Keywords: mathematical methods, quantum mechanics math, statistical mechanics
- Focus: Physics problems requiring advanced math

**W: Multimedia** (cs.MM)
- Keywords: video, audio, compression, streaming, codecs
- Focus: Multimedia data processing

**X: Methodology** (stat.ME)
- Keywords: statistical methods, inference, Bayesian, estimation
- Focus: Statistical methodology

**Y: Combinatorics** (math.CO)
- Keywords: graph theory, discrete math, counting, permutations
- Focus: Discrete structures and counting

**Z: Neural and Evolutionary Computing** (cs.NE)
- Keywords: evolutionary algorithms, genetic algorithms, neural networks, bio-inspired
- Focus: Bio-inspired computing methods

## Classification Strategy:

1. **Identify keywords** - What technical terms appear?
2. **Identify methodology** - What methods are used? (theoretical, experimental, computational)
3. **Identify main contribution** - What's the primary advance?
4. **Choose the BEST fit** - Some papers span multiple areas; choose the PRIMARY one

## Examples:

Example 1:
Paper: "Quantum Error Correction Exploiting Degeneracy"
- Keywords: quantum, error correction
- Method: Theoretical analysis
- Main contribution: Quantum error correction codes
→ **Answer: A** (Quantum Physics)

Example 2:
Paper: "Deep Learning for Image Recognition"
- Keywords: deep learning, neural networks, image recognition
- Method: Computational/ML
- Main contribution: Computer vision application
→ **Answer: M** (Computer Vision)

Example 3:
Paper: "Optimization Algorithms for Neural Networks"
- Keywords: optimization, neural networks
- Method: Mathematical optimization
- Main contribution: Optimization methods
→ **Answer: S** (Optimization and Control)

Example 4:
Paper: "Natural Language Understanding with Transformers"
- Keywords: NLP, transformers, language model
- Method: Machine learning
- Main contribution: Language processing
→ **Answer: F** (Computation and Language)

Example 5:
Paper: "String Theory and Black Hole Thermodynamics"
- Keywords: string theory, black holes, thermodynamics
- Method: Theoretical physics
- Main contribution: High energy physics theory
→ **Answer: I** (High Energy Physics - Theory)

Now classify the following paper. Return ONLY the category letter (A-Z), no explanation.
"""
    return prompt


def classify_with_llm(
    adapter: OpenAIAdapter,
    prompt: str,
    question: str
) -> str:
    """使用 LLM 进行分类"""
    # 构建消息
    user_message = f"{prompt}\n\nPaper to classify:\n{question}\n\nCategory:"

    try:
        # 调用 OpenAI 格式的 API
        response = adapter.client.chat.completions.create(
            model=adapter.model_name,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=10,
            temperature=0.3,
        )

        # 提取答案
        answer = response.choices[0].message.content.strip().upper()

        # 提取字母（如果有多余字符）
        if len(answer) > 0 and answer[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return answer[0]
        else:
            # 默认返回 A
            logger.warning(f"无法解析答案: {answer}, 默认返回 A")
            return 'A'

    except Exception as e:
        logger.error(f"API 调用失败: {e}")
        return 'A'


# ============================================================================
# Step 3: 评估函数
# ============================================================================

def evaluate(
    adapter: OpenAIAdapter,
    prompt: str,
    data: List[Dict],
    dataset_name: str = "Dataset",
    max_samples: int = None
) -> Dict:
    """评估分类器性能"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 评估 {dataset_name}")
    logger.info(f"{'='*60}")

    if max_samples:
        data = data[:max_samples]

    correct = 0
    errors = []

    for i, item in enumerate(data):
        question = item['question']
        true_label = item['answer']

        # 预测
        pred_label = classify_with_llm(adapter, prompt, question)

        # 判断
        if pred_label == true_label:
            correct += 1
            logger.info(f"✅ {i+1}. 正确! 预测={pred_label}, 真实={true_label}")
        else:
            logger.info(f"❌ {i+1}. 错误! 预测={pred_label}, 真实={true_label}")
            errors.append({
                'question': question[:150],
                'true_label': true_label,
                'pred_label': pred_label,
            })

    accuracy = correct / len(data) if data else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"准确率: {accuracy:.2%} ({correct}/{len(data)})")
    logger.info(f"错误数: {len(errors)}/{len(data)}")
    logger.info(f"{'='*60}")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(data),
        'errors': errors,
    }


# ============================================================================
# Step 4: 收集失败案例（用于优化）
# ============================================================================

def collect_experiences(
    adapter: OpenAIAdapter,
    prompt: str,
    data: List[Dict]
) -> List[ConversationExperience]:
    """收集失败案例并转为 Experience 格式"""
    logger.info(f"\n收集训练集上的失败案例...")

    experiences = []

    for i, item in enumerate(data):
        question = item['question']
        true_label = item['answer']

        # 预测
        pred_label = classify_with_llm(adapter, prompt, question)

        # 创建 experience
        exp = ConversationExperience(
            messages=[{"role": "user", "content": f"Classify this paper:\n{question}"}],
            response=f"Category: {pred_label}",
        )

        # 添加反馈
        if pred_label != true_label:
            # 失败案例 - 添加负面反馈
            exp.feedback = CompositeFeedback(
                feedback_type=FeedbackType.CRITIQUE,
                critique=f"Wrong classification. Predicted {pred_label}, but correct answer is {true_label}. "
                         f"Need to better identify the main research domain.",
                score=0.0,
            )
            experiences.append(exp)
            logger.info(f"  {i+1}/{len(data)}: ❌ 失败案例 (预测={pred_label}, 真实={true_label})")
        else:
            # 成功案例 - 添加正面反馈
            exp.feedback = CompositeFeedback(
                feedback_type=FeedbackType.SCORE,
                score=1.0,
            )
            experiences.append(exp)
            logger.info(f"  {i+1}/{len(data)}: ✅ 成功案例 (预测={pred_label})")

    # 统计
    failures = [e for e in experiences if e.feedback and e.feedback.to_score() < 0.5]
    logger.info(f"\n✅ 收集到 {len(experiences)} 个 experiences")
    logger.info(f"   其中失败案例: {len(failures)} 条")
    logger.info(f"   成功案例: {len(experiences) - len(failures)} 条")

    return experiences


# ============================================================================
# Step 5: 优化循环
# ============================================================================

def optimize_prompt(
    base_adapter: OpenAIAdapter,
    judge_adapter: OpenAIAdapter,
    initial_prompt: str,
    train_data: List[Dict],
    eval_data: List[Dict],
    num_iterations: int = 5
) -> Tuple[str, Dict]:
    """优化 prompt"""

    logger.info(f"\n{'='*60}")
    logger.info(f"🔄 开始优化循环")
    logger.info(f"{'='*60}")

    # 创建优化器（使用 judge adapter）
    config = OptimizerConfig(
        max_steps=2,  # 每次迭代优化2步
        gradient_accumulation_steps=20,  # 使用20个失败案例
        early_stopping_patience=3,
        target="Improve classification accuracy for scientific papers by learning from misclassifications",
    )

    optimizer = TrainFreeOptimizer(adapter=judge_adapter, config=config)

    current_prompt = initial_prompt
    best_prompt = initial_prompt
    best_accuracy = 0.0

    history = []

    for iteration in range(num_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 优化迭代 {iteration + 1}/{num_iterations}")
        logger.info(f"{'='*60}")

        # 评估当前 prompt（只评估前5条以节省 API 调用）
        eval_result = evaluate(
            base_adapter, current_prompt, eval_data[:5], f"验证集 (迭代 {iteration+1}, 前5条)"
        )

        # 记录
        history.append({
            'iteration': iteration + 1,
            'accuracy': eval_result['accuracy'],
        })

        # 更新最佳 prompt
        if eval_result['accuracy'] >= best_accuracy:
            best_accuracy = eval_result['accuracy']
            best_prompt = current_prompt
            logger.info(f"✨ 最佳准确率: {best_accuracy:.2%}")

        # 收集失败案例（只收集部分以节省 API 调用）
        logger.info(f"\n收集失败案例用于优化...")
        experiences = collect_experiences(
            base_adapter, current_prompt, train_data[:20]  # 只用前20条训练数据
        )

        # 检查是否有失败案例
        failures = [e for e in experiences if e.feedback and e.feedback.to_score() < 0.5]
        if len(failures) == 0:
            logger.info(f"✅ 没有失败案例，停止优化")
            break

        # 创建 OptimizablePrompt
        prompt_obj = TextPrompt(content=current_prompt)

        # 优化
        logger.info(f"\n⚡ 正在优化 prompt...")
        logger.info(f"   使用 {len(failures)} 个失败案例计算梯度")

        result = optimizer.optimize(
            prompt=prompt_obj,
            experiences=experiences,
            validator=None,
        )

        # 更新 prompt
        current_prompt = result.optimized_prompt.content
        logger.info(f"✅ Prompt 已更新 (版本: {result.optimized_prompt.version})")

        # 显示改进
        if result.improvement is not None:
            logger.info(f"   改进: {result.improvement:+.3f}")

    # 最终评估
    logger.info(f"\n{'='*60}")
    logger.info(f"最终评估")
    logger.info(f"{'='*60}")

    final_result = evaluate(base_adapter, best_prompt, eval_data[:5], "验证集 (最终, 前5条)")

    optimization_result = {
        'initial_accuracy': history[0]['accuracy'] if history else 0,
        'final_accuracy': final_result['accuracy'],
        'improvement': final_result['accuracy'] - history[0]['accuracy'] if history else 0,
        'num_iterations': num_iterations,
        'history': history,
    }

    return best_prompt, optimization_result


# ============================================================================
# 主流程
# ============================================================================

def main():
    """主流程"""
    logger.info("🚀 论文分类真实 API 优化 Demo - 硅流动 API")
    logger.info("="*60)

    # Step 1: 加载数据
    train_data, eval_data, test_data = load_data(
        'demo/data/intern_camp5.csv',
        train_size=30,
        eval_size=10,
        test_size=10,
        seed=42
    )

    # Step 2: 创建 adapter (使用硅流动 API)
    logger.info(f"\n创建 OpenAI Adapter (硅流动 API)...")

    # 从环境变量读取配置
    api_key = os.getenv('TRES_LLM_API_KEY')
    base_url = os.getenv('TRES_LLM_BASE_URL')
    base_model = "Qwen/Qwen2.5-7B-Instruct"  # 基础模型：7B
    judge_model = "Qwen/Qwen2.5-72B-Instruct"  # Judge模型：72B

    if not api_key:
        raise ValueError("未找到 TRES_LLM_API_KEY 环境变量，请检查 .env 文件")

    # 创建基础模型 adapter (用于分类)
    base_adapter = OpenAIAdapter(
        model=base_model,
        api_key=api_key,
        base_url=base_url
    )

    # 创建 judge 模型 adapter (用于优化)
    judge_adapter = OpenAIAdapter(
        model=judge_model,
        api_key=api_key,
        base_url=base_url
    )

    logger.info(f"✅ Adapter 创建成功")
    logger.info(f"   Base URL: {base_url}")
    logger.info(f"   Base Model: {base_model}")
    logger.info(f"   Judge Model: {judge_model}")

    # 为了兼容现有代码，使用 base_adapter 作为主 adapter
    adapter = base_adapter

    # Step 3: 创建初始 prompt
    initial_prompt = create_initial_prompt()
    logger.info(f"\n✅ 初始 prompt 创建完成 ({len(initial_prompt)} 字符)")

    # Step 4: 评估初始模型（只评估前3条以节省 API 调用）
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 评估初始模型")
    logger.info(f"{'='*60}")

    initial_test_result = evaluate(
        base_adapter, initial_prompt, test_data[:10], "测试集 (初始, 前10条)"
    )

    # Step 5: 优化
    optimized_prompt, optimization_result = optimize_prompt(
        base_adapter=adapter,  # 用于预测
        judge_adapter=judge_adapter,  # 用于优化
        initial_prompt=initial_prompt,
        train_data=train_data,
        eval_data=eval_data,
        num_iterations=5  # 增加到5轮优化
    )

    # Step 6: 在测试集上评估优化后效果
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 评估优化后模型")
    logger.info(f"{'='*60}")

    final_test_result = evaluate(
        base_adapter, optimized_prompt, test_data[:10], "测试集 (优化后, 前10条)"
    )

    # Step 7: 总结
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 最终总结")
    logger.info(f"{'='*60}")
    logger.info(f"训练集大小: {len(train_data)}")
    logger.info(f"验证集大小: {len(eval_data)}")
    logger.info(f"测试集大小: {len(test_data)}")
    logger.info(f"\n优化前测试准确率: {initial_test_result['accuracy']:.2%}")
    logger.info(f"优化后测试准确率: {final_test_result['accuracy']:.2%}")
    logger.info(f"准确率变化: {(final_test_result['accuracy'] - initial_test_result['accuracy']):+.2%}")
    logger.info(f"\n验证集最佳准确率: {optimization_result['final_accuracy']:.2%}")
    logger.info(f"验证集提升: {optimization_result['improvement']:+.2%}")

    # 保存结果
    results = {
        'train_size': len(train_data),
        'eval_size': len(eval_data),
        'test_size': len(test_data),
        'initial_test_accuracy': initial_test_result['accuracy'],
        'final_test_accuracy': final_test_result['accuracy'],
        'test_improvement': final_test_result['accuracy'] - initial_test_result['accuracy'],
        'optimization': optimization_result,
        'config': {
            'base_url': base_url,
            'model': model,
        }
    }

    output_path = Path('demo/data/api_optimization_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ 结果已保存到: {output_path}")

    # 保存优化后的 prompt
    prompt_path = Path('demo/data/optimized_prompt.txt')
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(optimized_prompt)
    logger.info(f"✅ 优化后的 prompt 已保存到: {prompt_path}")

    logger.info(f"\n🎉 Demo 完成!")


if __name__ == "__main__":
    main()
