"""论文分类 APO 优化测试 — Qwen3-8B 执行 + Qwen2.5-72B Judge

使用升级后的 APOEngine（多候选 + 并行 + 重试）。
"""

import csv
import os
import random
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import openai

from tresskill.config import GlobalConfig
from tresskill.llm import LLMClient
from tresskill.optimizer import APOEngine
from tresskill.schema import Feedback, Message, Skill, Trace
from tresskill.skill import save as save_skill
from tresskill.skill_tree import SkillTree

# ── Config ──────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen3-8B"
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DATA_PATH = "demo/data/intern_camp5.csv"
TRAIN_SIZE = 30
EVAL_SIZE = 15
NUM_OPTIMIZE_ROUNDS = 3
NUM_CANDIDATES = 2


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    random.seed(42)
    random.shuffle(data)
    train = data[:TRAIN_SIZE]
    eval_set = data[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]
    print(f"数据集: {len(data)} 总量, 训练={len(train)}, 评估={len(eval_set)}")
    labels = Counter(item["answer"] for item in train)
    print(f"训练集类别分布: {dict(labels)}")
    return train, eval_set


INITIAL_PROMPT = """You are a scientific paper classifier. Classify the paper into one of 26 categories (A-Z).

Categories: A=Quantum Physics, B=Chemical Physics, C=Atomic Physics, D=Soft Condensed Matter, E=Robotics, F=Computation & Language, G=Software Engineering, H=Information Retrieval, I=HEP Theory, J=HEP Phenomenology, K=Optics, L=AI, M=Computer Vision, N=Nuclear Theory, O=Astrophysics, P=Probability, Q=Operating Systems, R=Signal Processing, S=Optimization & Control, T=Dynamical Systems, U=Differential Geometry, V=Mathematical Physics, W=Multimedia, X=Statistics Methodology, Y=Combinatorics, Z=Neural & Evolutionary Computing.

Return ONLY the single letter (A-Z). No explanation."""


def classify(client: openai.OpenAI, prompt: str, question: str) -> str:
    """Classify a paper using BASE_MODEL."""
    try:
        r = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Classify:\n{question}"},
            ],
            max_tokens=10,
            temperature=0.3,
        )
        ans = r.choices[0].message.content.strip().upper()
        for ch in ans:
            if ch.isalpha():
                return ch
        return "A"
    except Exception as e:
        print(f"  [WARN] classify error: {e}")
        return "A"


def evaluate(client: openai.OpenAI, prompt: str, data: list, label: str) -> tuple:
    """Evaluate prompt on data, return (accuracy, errors_list)."""
    correct = 0
    errors = []
    for item in data:
        pred = classify(client, prompt, item["question"])
        true = item["answer"]
        if pred == true:
            correct += 1
        else:
            errors.append({"q": item["question"][:100], "pred": pred, "true": true})
    acc = correct / len(data) if data else 0
    print(f"  [{label}] 准确率: {acc:.1%} ({correct}/{len(data)})")
    return acc, errors


def collect_traces(client: openai.OpenAI, prompt: str, data: list) -> list:
    """Run classification on data, return Trace objects with feedback."""
    traces = []
    for item in data:
        pred = classify(client, prompt, item["question"])
        true = item["answer"]
        t = Trace(
            inputs=[Message(role="user", content=f"Classify:\n{item['question']}")],
            prediction=Message(role="assistant", content=pred),
        )
        if pred != true:
            t.feedback = Feedback(
                score=0.0,
                critique=f"Predicted {pred}, correct is {true}.",
                correction=true,
            )
        else:
            t.feedback = Feedback(score=1.0)
        traces.append(t)
    return traces


def main():
    print(f"{'='*60}")
    print(f"论文分类 APO 优化 — Qwen3-8B + APOEngine")
    print(f"{'='*60}\n")

    train, eval_set = load_data()

    # Raw OpenAI client for classification (Qwen3-8B)
    client = openai.OpenAI(
        api_key=os.getenv("TREE_LLM_API_KEY"),
        base_url=os.getenv("TREE_LLM_BASE_URL"),
    )

    # APOEngine with judge model (Qwen2.5-72B)
    config = GlobalConfig()
    config = config.model_copy(update={
        "llm": config.llm.model_copy(update={"judge_model": JUDGE_MODEL}),
        "apo": config.apo.model_copy(update={
            "num_candidates": NUM_CANDIDATES,
            "gradient_accumulation_steps": 10,
        }),
    })
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    # ── Baseline ──
    print(f"\n{'─'*40}")
    print("Step 1: 基线评估（初始 prompt）")
    baseline_acc, _ = evaluate(client, INITIAL_PROMPT, eval_set, "基线")

    # ── Optimization loop ──
    current_prompt = INITIAL_PROMPT
    best_prompt = INITIAL_PROMPT
    best_acc = baseline_acc

    for round_num in range(1, NUM_OPTIMIZE_ROUNDS + 1):
        print(f"\n{'─'*40}")
        print(f"Step 2.{round_num}: 优化轮次 {round_num}/{NUM_OPTIMIZE_ROUNDS}")

        # Collect traces on training set
        t0 = time.time()
        traces = collect_traces(client, current_prompt, train)
        failures = [t for t in traces if t.feedback and t.feedback.score < 0.5]
        print(f"  训练集: {len(failures)}/{len(traces)} 失败 ({time.time()-t0:.1f}s)")

        if not failures:
            print("  零失败，提前结束")
            break

        # Create skill for APO
        skill = Skill(
            name="paper-classifier",
            description="Classify scientific papers into 26 categories.",
            system_prompt=current_prompt,
            target="Improve classification accuracy by learning from misclassifications.",
            version=f"v{round_num}.0",
        )

        # Optimize
        t0 = time.time()
        new_skill = engine.optimize(skill, traces)
        opt_time = time.time() - t0
        print(f"  优化耗时: {opt_time:.1f}s")

        if new_skill.version != skill.version:
            current_prompt = new_skill.system_prompt
            print(f"  ✓ Prompt 已更新 → {new_skill.version}")
            print(f"  新 prompt 前 150 字: {current_prompt[:150]}...")

            # Evaluate
            acc, _ = evaluate(client, current_prompt, eval_set, f"轮次{round_num}")
            if acc > best_acc:
                best_acc = acc
                best_prompt = current_prompt
                print(f"  ★ 新最佳: {best_acc:.1%}")
        else:
            print(f"  ✗ 候选均未超过原 prompt，保持不变")

    # ── Final ──
    print(f"\n{'='*60}")
    print(f"最终结果")
    print(f"{'='*60}")
    print(f"  基线准确率:   {baseline_acc:.1%}")
    print(f"  最佳准确率:   {best_acc:.1%}")
    print(f"  提升:         {best_acc - baseline_acc:+.1%}")
    print(f"\n最佳 prompt 前 200 字:")
    print(f"  {best_prompt[:200]}...")


if __name__ == "__main__":
    main()
