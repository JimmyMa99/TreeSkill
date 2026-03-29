#!/usr/bin/env python3
"""
SealQA Benchmark — 对比有/无 TreeSkill 优化的效果

流程:
  1. 无优化 baseline: Qwen3.5-9B + 基础 prompt → 跑 SealQA 111 题
  2. TreeSkill APO 优化 prompt（judge/rewrite 用 doubao）
  3. 优化后: Qwen3.5-9B + 优化 prompt → 跑 SealQA 111 题

用法:
    # 先在服务器上启动 vLLM:
    # vllm serve /root/models/Qwen3.5-9B --port 8000
    # 然后本地跑:
    python demo/demo_sealqa.py
"""

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from treeskill.config import GlobalConfig
from treeskill.llm import LLMClient
from treeskill.optimizer import APOEngine
from treeskill.schema import Feedback, Message, Skill, Trace
from treeskill.skill import save as save_skill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────
# Actor: Qwen3.5-9B on vLLM (local server)
ACTOR_BASE_URL = os.getenv("ACTOR_BASE_URL", "http://localhost:8000/v1")
ACTOR_MODEL = os.getenv("ACTOR_MODEL", "Qwen3.5-9B")
ACTOR_API_KEY = "dummy"  # vLLM doesn't need real key

# Judge/Rewrite: via OneAPI
JUDGE_BASE_URL = os.getenv(
    "TREE_LLM_BASE_URL", "https://oneapi.liuyanxing.site:8443/v1"
)
JUDGE_MODEL = os.getenv("TREE_LLM_JUDGE_MODEL", "doubao/seed-2-0-lite")
JUDGE_API_KEY = os.getenv("TREE_LLM_API_KEY")
assert JUDGE_API_KEY, "Set TREE_LLM_API_KEY env var"

DATA_PATH = os.getenv("SEALQA_PATH", "/root/sealqa/seal-0.parquet")
OUTPUT_DIR = Path("demo/outputs/sealqa")

# Split
TRAIN_RATIO = 0.18
VAL_RATIO = 0.12
NUM_ROUNDS = 3

# Baseline prompt
BASELINE_PROMPT = """You are a knowledgeable assistant. Answer the question accurately and concisely.
Provide only the answer, no explanation."""


# ── Data ───────────────────────────────────────────────

def load_sealqa(path: str):
    """Load SealQA and split by topic (stratified)."""
    df = pd.read_parquet(path)
    logger.info(f"SealQA loaded: {len(df)} questions, topics: {df['topic'].nunique()}")

    train, val, test = [], [], []
    random.seed(42)

    for topic, group in df.groupby("topic"):
        items = group.to_dict("records")
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ── Scoring ────────────────────────────────────────────

def ask_model(
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    question: str,
    max_tokens: int = 500,
) -> str:
    """Ask the actor model a question."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"ask_model error: {e}")
        return ""


def judge_answer(
    judge_client: openai.OpenAI,
    question: str,
    predicted: str,
    ground_truth: str,
) -> float:
    """Use judge to score predicted vs ground truth (0 or 1)."""
    prompt = (
        f"Question: {question}\n"
        f"Ground truth answer: {ground_truth}\n"
        f"Predicted answer: {predicted}\n\n"
        "Is the predicted answer correct? Consider semantic equivalence, "
        "not just exact match. Return ONLY 1 (correct) or 0 (wrong)."
    )
    try:
        r = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict answer judge. Return only 0 or 1."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        raw = (r.choices[0].message.content or "").strip()
        return 1.0 if "1" in raw else 0.0
    except Exception as e:
        logger.warning(f"judge error: {e}")
        return 0.0


def evaluate_batch(
    actor: openai.OpenAI,
    judge: openai.OpenAI,
    system_prompt: str,
    data: List[Dict],
    label: str = "",
) -> float:
    """Evaluate on a batch, return accuracy."""
    correct = 0
    for i, item in enumerate(data):
        pred = ask_model(actor, ACTOR_MODEL, system_prompt, item["question"])
        score = judge_answer(judge, item["question"], pred, item["answer"])
        correct += score
        if (i + 1) % 10 == 0 or i == len(data) - 1:
            logger.info(f"  [{label}] {i+1}/{len(data)}: acc={correct/(i+1):.1%}")
    acc = correct / len(data) if data else 0
    logger.info(f"  [{label}] final: {acc:.1%} ({int(correct)}/{len(data)})")
    return acc


# ── Main ───────────────────────────────────────────────

def main():
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("SealQA Benchmark — TreeSkill APO")
    logger.info(f"  Actor: {ACTOR_MODEL} @ {ACTOR_BASE_URL}")
    logger.info(f"  Judge: {JUDGE_MODEL}")
    logger.info("=" * 60)

    train, val, test = load_sealqa(DATA_PATH)

    actor = openai.OpenAI(api_key=ACTOR_API_KEY, base_url=ACTOR_BASE_URL)
    judge = openai.OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Baseline (no optimization) ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Baseline (no TreeSkill)")
    logger.info("=" * 60)
    baseline_acc = evaluate_batch(actor, judge, BASELINE_PROMPT, test, "baseline")

    # ── Phase 2: APO Optimization ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: TreeSkill APO optimization")
    logger.info("=" * 60)

    skill = Skill(
        name="sealqa-answerer",
        description="Answer factual questions accurately",
        system_prompt=BASELINE_PROMPT,
        target="Improve answer accuracy on fact-seeking questions with noisy search results",
        version="v1.0",
    )

    # Setup framework
    os.environ["TREE_LLM_API_KEY"] = ACTOR_API_KEY
    os.environ["TREE_LLM_BASE_URL"] = ACTOR_BASE_URL
    os.environ["TREE_LLM_MODEL"] = ACTOR_MODEL
    os.environ["TREE_LLM_PROTOCOL"] = "openai"
    os.environ["TREE_LLM_JUDGE_API_KEY"] = JUDGE_API_KEY
    os.environ["TREE_LLM_JUDGE_BASE_URL"] = JUDGE_BASE_URL
    os.environ["TREE_LLM_JUDGE_MODEL"] = JUDGE_MODEL
    os.environ["TREE_LLM_JUDGE_PROTOCOL"] = "openai"

    from treeskill.config import LLMConfig, APOConfig
    config = GlobalConfig(
        llm=LLMConfig(),
        apo=APOConfig(
            gradient_accumulation_steps=8,
            beam_width=2,
            branch_factor=2,
            beam_rounds=2,
        ),
    )
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    # Score function: real accuracy
    def score_fn(prompt: str, traces: List[Trace]) -> float:
        correct = 0
        for t in traces:
            pred = ask_model(actor, ACTOR_MODEL, prompt, t.inputs[-1].content)
            expected = t.feedback.correction if t.feedback and t.feedback.correction else ""
            s = judge_answer(judge, t.inputs[-1].content, pred, expected)
            correct += s
        return correct / len(traces) if traces else 0.0

    engine._score_fn = score_fn

    best_skill = skill
    best_val = -1.0

    for rnd in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- Round {rnd}/{NUM_ROUNDS} ---")

        # Collect traces on train
        traces = []
        for item in train:
            pred = ask_model(actor, ACTOR_MODEL, best_skill.system_prompt, item["question"])
            s = judge_answer(judge, item["question"], pred, item["answer"])
            t = Trace(
                inputs=[Message(role="user", content=item["question"])],
                prediction=Message(role="assistant", content=pred),
            )
            t.feedback = Feedback(
                score=s,
                critique=f"Predicted: {pred[:100]}. Correct: {item['answer']}",
                correction=item["answer"],
            )
            traces.append(t)

        failures = [t for t in traces if t.feedback.score < 0.5]
        total = len(traces)
        logger.info(f"  Train: {len(failures)}/{total} failures")

        if not failures:
            logger.info("  All correct, skip")
            continue

        t0 = time.time()
        candidate = engine.optimize(best_skill, traces)
        logger.info(f"  Optimize: {time.time()-t0:.1f}s → {candidate.version}")

        # Validate
        val_acc = evaluate_batch(actor, judge, candidate.system_prompt, val, f"val r{rnd}")

        if val_acc > best_val:
            best_val = val_acc
            best_skill = candidate
            save_skill(best_skill, OUTPUT_DIR)
            logger.info(f"  ★ Accepted {best_skill.version} (val={val_acc:.1%})")
        else:
            logger.info(f"  Rejected (val={val_acc:.1%} <= {best_val:.1%})")

    # ── Phase 3: Final test ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Final test with optimized prompt")
    logger.info("=" * 60)
    final_acc = evaluate_batch(actor, judge, best_skill.system_prompt, test, "final")

    # ── Summary ──
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"SealQA Results ({elapsed/60:.1f} min)")
    logger.info(f"  Baseline (no TreeSkill): {baseline_acc:.1%}")
    logger.info(f"  Final (with TreeSkill):  {final_acc:.1%}")
    logger.info(f"  Delta:                   {final_acc - baseline_acc:+.1%}")
    logger.info(f"  Prompt version:          {best_skill.version}")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "baseline": baseline_acc,
            "final": final_acc,
            "delta": final_acc - baseline_acc,
            "version": best_skill.version,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "elapsed_min": elapsed / 60,
        }, f, indent=2)

    # Save optimized prompt
    logger.info(f"\nOptimized prompt:\n{best_skill.system_prompt[:500]}...")
    logger.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
