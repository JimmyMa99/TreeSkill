#!/usr/bin/env python3
"""
10 分钟最小化树优化 Demo — 论文分类

用一个较弱的 baseline prompt（有类别名但缺关键词提示），3 类、少量数据、2 轮优化。
预期: baseline ~50-60% → After optimization ~70%+，并触发 auto-split 产生子节点。

用法:
    conda activate pr
    python demo/demo_tree_minimal_10min.py
"""

import csv
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────
MAIN_MODEL = os.getenv("TREE_LLM_MODEL", "Qwen/Qwen3-8B")
JUDGE_MODEL = os.getenv("TREE_LLM_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
DATA_PATH = "demo/data/intern_camp5.csv"

# 只用 3 类，降低复杂度
CATEGORIES = ["A", "E", "M"]  # 量子物理 / 机器人 / 计算机视觉
TRAIN_PER_CAT = 8
TEST_PER_CAT = 4
NUM_ROUNDS = 2
NUM_CANDIDATES = 2
MAX_WORKERS = 8
OUTPUT_DIR = Path("demo/outputs/tree-minimal-10min")

# 较弱的 baseline — 有类别名但缺关键词提示，留出优化空间
BAD_BASELINE = """You are a paper classifier. Given a scientific paper, classify it.

A - physics
E - engineering
M - computers

Output only the letter."""


# ── Data ───────────────────────────────────────────────

def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = list(csv.DictReader(f))

    cat_data = {c: [] for c in CATEGORIES}
    for item in all_data:
        label = item["answer"]
        if label in cat_data:
            cat_data[label].append(item)

    train, test = [], []
    random.seed(42)
    for cat in CATEGORIES:
        pool = cat_data[cat]
        random.shuffle(pool)
        train.extend(pool[:TRAIN_PER_CAT])
        test.extend(pool[TRAIN_PER_CAT : TRAIN_PER_CAT + TEST_PER_CAT])
        logger.info(f"  {cat}: train={min(len(pool), TRAIN_PER_CAT)}, test={TEST_PER_CAT}")

    random.shuffle(train)
    random.shuffle(test)
    logger.info(f"总计: train={len(train)}, test={len(test)}")
    return train, test


# ── Classification ─────────────────────────────────────

def classify(client: openai.OpenAI, prompt: str, question: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Paper:\n{question[:400]}"},
            ],
            max_tokens=10,
            temperature=0.3,
        )
        ans = r.choices[0].message.content.strip().upper()
        for ch in ans:
            if ch in CATEGORIES:
                return ch
        return CATEGORIES[0]
    except Exception as e:
        logger.warning(f"classify error: {e}")
        return CATEGORIES[0]


def evaluate(client: openai.OpenAI, prompt: str, data: list, label: str = "") -> float:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(classify, client, prompt, item["question"]): item for item in data}
        correct = sum(
            1 for f in as_completed(futures) if f.result() == futures[f]["answer"]
        )
    acc = correct / len(data) if data else 0
    logger.info(f"  [{label}] accuracy: {acc:.1%} ({correct}/{len(data)})")
    return acc


def collect_traces(
    client: openai.OpenAI,
    prompt: str,
    data: list,
    node_path: str = "paper-classifier",
) -> List[Trace]:
    def _classify_item(item):
        pred = classify(client, prompt, item["question"])
        true_label = item["answer"]
        t = Trace(
            inputs=[Message(role="user", content=f"Paper:\n{item['question'][:400]}")],
            prediction=Message(role="assistant", content=pred),
            node_path=node_path,
        )
        if pred != true_label:
            t.feedback = Feedback(
                score=0.0,
                critique=f"Predicted {pred}, correct is {true_label}.",
                correction=true_label,
            )
        else:
            t.feedback = Feedback(score=1.0)
        return t

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        traces = list(pool.map(_classify_item, data))
    return traces


# ── Main ───────────────────────────────────────────────

def main():
    t_start = time.time()

    logger.info("=" * 50)
    logger.info("10min Tree Demo — 论文分类 (3 类)")
    logger.info("=" * 50)

    train_data, test_data = load_data()

    client = openai.OpenAI(
        api_key=os.getenv("TREE_LLM_API_KEY"),
        base_url=os.getenv("TREE_LLM_BASE_URL"),
    )

    config = GlobalConfig()
    config = config.model_copy(update={
        "llm": config.llm.model_copy(update={"judge_model": JUDGE_MODEL}),
        "apo": config.apo.model_copy(update={
            "num_candidates": NUM_CANDIDATES,
            "gradient_accumulation_steps": 8,
            # Beam search (aligned with Agent-Lightning APO)
            "beam_width": 2,
            "branch_factor": 2,
            "beam_rounds": 2,
        }),
    })
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    # Create初始 skill 树
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root_skill = Skill(
        name="paper-classifier",
        description="Classify papers into A (Quantum Physics), E (Robotics), M (Computer Vision).",
        system_prompt=BAD_BASELINE,
        target="Improve accuracy by adding category descriptions and classification hints",
        version="v1.0",
    )
    save_skill(root_skill, OUTPUT_DIR)
    tree = SkillTree.load(OUTPUT_DIR)

    # ── 基线 ──
    logger.info("\n--- 基线评估 (故意很烂的 prompt) ---")
    logger.info(f"Prompt: {BAD_BASELINE.strip()}")
    baseline_acc = evaluate(client, BAD_BASELINE, test_data, "baseline")
    best_acc = baseline_acc

    # ── 多轮优化 ──
    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"第 {round_num}/{NUM_ROUNDS} 轮优化")
        logger.info(f"{'='*50}")

        t0 = time.time()
        traces = collect_traces(client, tree.root.skill.system_prompt, train_data)
        failures = [t for t in traces if t.feedback and t.feedback.score < 0.5]
        logger.info(f"  traces: {len(failures)}/{len(traces)} failures ({time.time()-t0:.1f}s)")

        if not failures:
            logger.info("  零failed, Skip")
            continue

        resume = ResumeState.create(
            OUTPUT_DIR,
            total_rounds=NUM_ROUNDS,
            metadata={"round": round_num, "failures": len(failures)},
        )
        resume.round_num = round_num

        t0 = time.time()
        try:
            engine.evolve_tree(tree, traces, auto_split=True, resume=resume)
            tree.save()
            resume.clear()
        except KeyboardInterrupt:
            logger.warning("中断! 进度已Save")
            return

        logger.info(f"  优化耗时: {time.time()-t0:.1f}s")
        logger.info(f"\n  树结构:\n{tree.list_tree()}")

        acc = evaluate(client, tree.root.skill.system_prompt, test_data, f"round {round_num}")
        if acc > best_acc:
            best_acc = acc
            logger.info(f"  ★ 新最佳: {best_acc:.1%}")

    # ── 总结 ──
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*50}")
    logger.info(f"完成! 总耗时 {elapsed/60:.1f} 分钟")
    logger.info(f"  baseline: {baseline_acc:.1%}")
    logger.info(f"  best:     {best_acc:.1%} ({best_acc - baseline_acc:+.1%})")
    logger.info(f"  nodes:    {tree.root.leaf_count()} leaves")
    logger.info(f"\n最终 prompt:\n{tree.root.skill.system_prompt[:300]}...")
    logger.info(f"\n输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
