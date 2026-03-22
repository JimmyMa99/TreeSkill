#!/usr/bin/env python3
"""
Qwen3-8B 树优化 Demo — 使用升级后的 APOEngine

展示complete Tree 功能:
1. Auto-Split: 根据矛盾Feedback自动拆分子 skill
2. 多候选Score: 每轮生成 N 个候选并选最佳
3. 并行 API 调用 + 重试
4. 断点续跑 (ResumeState)
5. 多轮迭代优化

主模型: Qwen/Qwen3-8B (弱模型, 有优化空间)
Judge模型: Qwen/Qwen2.5-72B-Instruct (强模型, 生成梯度+Score)
"""

import csv
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict

import openai
from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.resume import ResumeState
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import save as save_skill
from treeskill.skill_tree import SkillTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────

MAIN_MODEL = "Qwen/Qwen3-8B"
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DATA_PATH = "demo/data/intern_camp5.csv"
CATEGORIES = ["A", "E", "G", "K", "M"]  # 5 类子集
SAMPLES_PER_CAT = 20
NUM_ROUNDS = 3
NUM_CANDIDATES = 2
OUTPUT_DIR = Path("demo/outputs/qwen3-8b-tree-new")

INITIAL_PROMPT = """You are a scientific paper classifier. Classify each paper into exactly one category.

Categories:
A = Quantum Physics (quantum, qubit, entanglement, quantum computing)
E = Robotics (robot, manipulation, navigation, autonomous, SLAM)
G = Software Engineering (testing, debugging, refactoring, CI/CD, code review)
K = Optics (laser, photonics, optical fibers, metamaterials, light)
M = Computer Vision (image recognition, object detection, segmentation, visual)

Rules:
- Return ONLY a single letter (A, E, G, K, or M)
- Choose the category that best matches the paper's PRIMARY contribution
- No explanation needed"""


# ── Data ────────────────────────────────────────────────

def load_data() -> tuple:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = list(csv.DictReader(f))

    cat_data = {c: [] for c in CATEGORIES}
    for item in all_data:
        label = item["answer"]
        if label in cat_data and len(cat_data[label]) < SAMPLES_PER_CAT:
            cat_data[label].append(item)

    balanced = []
    for cat, items in cat_data.items():
        balanced.extend(items)
        logger.info(f"  {cat}: {len(items)} 条")

    random.seed(42)
    random.shuffle(balanced)

    split = int(len(balanced) * 0.7)
    train, test = balanced[:split], balanced[split:]
    logger.info(f"总计 {len(balanced)}, 训练={len(train)}, test={len(test)}")
    return train, test


# ── Classification ──────────────────────────────────────

def classify(client: openai.OpenAI, prompt: str, question: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Classify:\n{question[:400]}\n\nCategory:"},
            ],
            max_tokens=10,
            temperature=0.3,
        )
        ans = r.choices[0].message.content.strip().upper()
        for ch in ans:
            if ch in "AEGKM":
                return ch
        return "A"
    except Exception as e:
        logger.warning(f"classify error: {e}")
        return "A"


def evaluate(client: openai.OpenAI, prompt: str, data: list, label: str = "") -> float:
    correct = 0
    for item in data:
        pred = classify(client, prompt, item["question"])
        if pred == item["answer"]:
            correct += 1
    acc = correct / len(data) if data else 0
    logger.info(f"  [{label}] 准确率: {acc:.1%} ({correct}/{len(data)})")
    return acc


def collect_traces(client: openai.OpenAI, prompt: str, data: list) -> List[Trace]:
    traces = []
    for item in data:
        pred = classify(client, prompt, item["question"])
        true = item["answer"]
        t = Trace(
            inputs=[Message(role="user", content=f"Classify:\n{item['question'][:400]}")],
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


# ── Main ────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Qwen3-8B 树优化 Demo (新架构 APOEngine)")
    logger.info("=" * 60)
    logger.info(f"主模型: {MAIN_MODEL}")
    logger.info(f"Judge: {JUDGE_MODEL}")
    logger.info(f"类别: {CATEGORIES}")

    train_data, test_data = load_data()

    # OpenAI client for classification
    client = openai.OpenAI(
        api_key=os.getenv("TREE_LLM_API_KEY"),
        base_url=os.getenv("TREE_LLM_BASE_URL"),
    )

    # APOEngine with judge model
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

    # ── Create初始 skill 树 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root_skill = Skill(
        name="paper-classifier",
        description="Classify scientific papers into 5 categories (A/E/G/K/M).",
        system_prompt=INITIAL_PROMPT,
        target="Improve classification accuracy by learning from misclassifications",
        version="v1.0",
    )
    save_skill(root_skill, OUTPUT_DIR)
    tree = SkillTree.load(OUTPUT_DIR)

    # ── 基线 ──
    logger.info(f"\n{'─'*40}")
    logger.info("基线评估")
    baseline_acc = evaluate(client, INITIAL_PROMPT, test_data, "基线")
    accuracy_history = [baseline_acc]
    best_acc = baseline_acc
    best_prompt = INITIAL_PROMPT

    # ── 多轮优化 ──
    samples_per_round = len(train_data) // NUM_ROUNDS

    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"第 {round_num}/{NUM_ROUNDS} 轮")
        logger.info(f"{'='*60}")

        # 每轮用不同的训练子集
        start = (round_num - 1) * samples_per_round
        end = start + samples_per_round
        round_data = train_data[start:end]

        # 收集 traces
        t0 = time.time()
        traces = collect_traces(client, tree.root.skill.system_prompt, round_data)
        failures = [t for t in traces if t.feedback and t.feedback.score < 0.5]
        logger.info(f"  收集: {len(failures)}/{len(traces)} failed ({time.time()-t0:.1f}s)")

        if not failures:
            logger.info("  零failed, Skip")
            accuracy_history.append(accuracy_history[-1])
            continue

        # evolve_tree (含 auto-split + resume)
        resume = ResumeState.create(
            OUTPUT_DIR,
            total_rounds=NUM_ROUNDS,
            metadata={"round": round_num, "failures": len(failures)},
        )
        resume.round_num = round_num

        def on_node_done(dotpath, node):
            logger.info(f"  ✓ {dotpath} → {node.skill.version}")

        t0 = time.time()
        try:
            engine.evolve_tree(
                tree, traces,
                auto_split=True,
                resume=resume,
                on_node_done=on_node_done,
            )
            tree.save()
            resume.clear()
        except KeyboardInterrupt:
            logger.warning("中断! 进度已Save, 下次可Continue")
            return
        except Exception as e:
            logger.error(f"优化出错: {e}")
            logger.info("进度已Save, 下次可Continue")
            raise

        opt_time = time.time() - t0
        logger.info(f"  优化耗时: {opt_time:.1f}s")

        # 显示树结构
        logger.info(f"\n  树结构:")
        logger.info(tree.list_tree())

        # 评估
        acc = evaluate(client, tree.root.skill.system_prompt, test_data, f"第{round_num}轮")
        accuracy_history.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_prompt = tree.root.skill.system_prompt
            logger.info(f"  ★ 新最佳: {best_acc:.1%}")

        # Save checkpoint
        ckpt_path = OUTPUT_DIR / f"round{round_num}"
        tree.save(ckpt_path)

    # ── 总结 ──
    logger.info(f"\n{'='*60}")
    logger.info("最终results")
    logger.info(f"{'='*60}")

    logger.info("\n准确率变化:")
    for i, acc in enumerate(accuracy_history):
        if i == 0:
            logger.info(f"  基线:    {acc:.1%}")
        else:
            delta = acc - accuracy_history[i - 1]
            logger.info(f"  第{i}轮:  {acc:.1%} ({delta:+.1%})")

    total_delta = best_acc - baseline_acc
    logger.info(f"\n最佳准确率: {best_acc:.1%} (提升 {total_delta:+.1%})")
    logger.info(f"\n最终树结构:")
    logger.info(tree.list_tree())

    def count_nodes(node):
        return 1 + sum(count_nodes(c) for c in node.children.values())

    logger.info(f"  总节点: {count_nodes(tree.root)}")
    logger.info(f"  叶子数: {tree.root.leaf_count()}")
    logger.info(f"\n最佳 prompt 前 200 字:")
    logger.info(f"  {best_prompt[:200]}...")
    logger.info(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
