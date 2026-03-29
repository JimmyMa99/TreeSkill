#!/usr/bin/env python3
"""
树分裂 + 剪枝 Demo — 论文分类 (6 类 → 自动分裂为子技能)

策略:
1. 使用 6 个跨学科类别，分属 3 大领域:
   - 物理: A (quant-ph), J (hep-ph)
   - 计算机: L (cs.AI), M (cs.CV)
   - 数学: U (math.DG), X (stat.ME)
2. 故意给一个极烂的初始 prompt（不含领域知识）
3. 多轮优化 → 触发 auto-split（不同领域各自成子节点）
4. 如果子节点表现太差 → 触发 prune

预期:
  baseline ~20-30% → split 后子节点各自优化 → 总体 60%+

用法:
    conda activate pr
    python demo/demo_tree_split.py
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
MAIN_MODEL = os.getenv("TREE_LLM_MODEL", "SilliconCloud/Qwen3.5-4B")
JUDGE_MODEL = os.getenv("TREE_LLM_JUDGE_MODEL", "bailian/glm-5")
DATA_PATH = "demo/data/intern_camp5.csv"

# 6 类，跨 3 大领域 — 故意制造领域间混淆
CATEGORIES = ["A", "J", "L", "M", "U", "X"]
# A = quant-ph (量子物理)
# J = hep-ph (高能物理 - 现象学)
# L = cs.AI (人工智能)
# M = cs.CV (计算机视觉)
# U = math.DG (微分几何)
# X = stat.ME (统计方法)

CATEGORY_NAMES = {
    "A": "quantum physics",
    "J": "particle physics",
    "L": "artificial intelligence",
    "M": "computer vision",
    "U": "differential geometry",
    "X": "statistics",
}

TRAIN_PER_CAT = 10
TEST_PER_CAT = 5
NUM_ROUNDS = 4
NUM_CANDIDATES = 2
MAX_WORKERS = 8
OUTPUT_DIR = Path("demo/outputs/tree-split")

# 极烂的初始 prompt — 不含任何领域信息，6 选 1 随机猜约 16.7%
BAD_BASELINE = """Classify the paper. Pick one letter.

A, J, L, M, U, or X.

Output only the letter."""

# Extra body params for task model
EXTRA_BODY = {"enable_thinking": False}


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
        avail = len(pool)
        tr = min(TRAIN_PER_CAT, avail - TEST_PER_CAT)
        train.extend(pool[:tr])
        test.extend(pool[tr : tr + TEST_PER_CAT])
        logger.info(
            f"  {cat} ({CATEGORY_NAMES[cat]}): "
            f"total={avail}, train={tr}, test={TEST_PER_CAT}"
        )

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
                {"role": "user", "content": f"Paper:\n{question[:500]}"},
            ],
            max_tokens=10,
            temperature=0.3,
            extra_body=EXTRA_BODY,
        )
        ans = (r.choices[0].message.content or "").strip().upper()
        for ch in ans:
            if ch in CATEGORIES:
                return ch
        return CATEGORIES[0]
    except Exception as e:
        logger.warning(f"classify error: {e}")
        return CATEGORIES[0]


def evaluate(client: openai.OpenAI, prompt: str, data: list, label: str = "") -> float:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(classify, client, prompt, item["question"]): item
            for item in data
        }
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
            inputs=[Message(role="user", content=f"Paper:\n{item['question'][:500]}")],
            prediction=Message(role="assistant", content=pred),
            node_path=node_path,
        )
        if pred != true_label:
            t.feedback = Feedback(
                score=0.0,
                critique=f"Predicted {pred} ({CATEGORY_NAMES.get(pred, '?')}), "
                         f"correct is {true_label} ({CATEGORY_NAMES.get(true_label, '?')}).",
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

    logger.info("=" * 60)
    logger.info("Tree Split Demo — 论文分类 (6 类, 跨 3 大领域)")
    logger.info("=" * 60)

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
            "beam_width": 2,
            "branch_factor": 2,
            "beam_rounds": 2,
        }),
    })
    llm = LLMClient(config)

    engine = APOEngine(config, llm)

    # Real-model scoring (Agent-Lightning style)
    def real_score_fn(prompt: str, traces: List[Trace]) -> float:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for t in traces:
                user_text = t.inputs[-1].content if t.inputs else ""
                paper = (
                    user_text.replace("Paper:\n", "", 1)
                    if user_text.startswith("Paper:\n")
                    else user_text
                )
                futures[pool.submit(classify, client, prompt, paper)] = t
            preds = {}
            for f in as_completed(futures):
                preds[futures[f].id] = f.result()

        grade_batches = []
        for t in traces:
            expected = (
                t.feedback.correction
                if t.feedback and t.feedback.correction
                else None
            )
            if expected is None:
                expected = (
                    t.prediction.content.strip().upper()
                    if t.prediction.content
                    else ""
                )
            grade_batches.append(
                engine._build_grade_messages(preds[t.id], expected)
            )

        responses = llm.generate_batch(grade_batches, model=JUDGE_MODEL)

        total_reward = sum(
            engine._parse_score(
                r.content if isinstance(r.content, str) else str(r.content)
            )
            for r in responses
        )
        return total_reward / len(responses) if responses else 0.0

    engine._score_fn = real_score_fn

    # ── 初始化 skill 树（已有则续跑） ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skill_file = OUTPUT_DIR / "SKILL.md"

    if skill_file.exists():
        logger.info("检测到已有 skill 树，续跑模式")
        tree = SkillTree.load(OUTPUT_DIR)
        logger.info(f"  根节点版本: {tree.root.skill.version}")
        logger.info(f"  子节点: {list(tree.root.children.keys())}")
    else:
        cat_desc = ", ".join(f"{k} ({v})" for k, v in CATEGORY_NAMES.items())
        root_skill = Skill(
            name="paper-classifier",
            description=f"Classify papers into: {cat_desc}",
            system_prompt=BAD_BASELINE,
            target=(
                "Improve classification accuracy. The 6 categories span 3 domains "
                "(physics, CS, math). Consider splitting into domain-specific "
                "sub-skills if a single prompt cannot handle all domains well."
            ),
            version="v1.0",
        )
        save_skill(root_skill, OUTPUT_DIR)
        tree = SkillTree.load(OUTPUT_DIR)

    # ── 基线 ──
    logger.info("\n--- 基线评估 (极烂 prompt, 无领域信息) ---")
    logger.info(f"Prompt: {BAD_BASELINE.strip()}")
    baseline_acc = evaluate(client, BAD_BASELINE, test_data, "baseline")
    best_acc = baseline_acc

    # ── 多轮优化 ──
    for round_num in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"第 {round_num}/{NUM_ROUNDS} 轮优化")
        logger.info(f"{'='*60}")

        t0 = time.time()
        traces = collect_traces(
            client, tree.root.skill.system_prompt, train_data
        )
        failures = [t for t in traces if t.feedback and t.feedback.score < 0.5]
        logger.info(
            f"  traces: {len(failures)}/{len(traces)} failures "
            f"({time.time()-t0:.1f}s)"
        )

        if not failures:
            logger.info("  零 failures, skip")
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
            logger.warning("中断! 进度已保存")
            return

        logger.info(f"  优化耗时: {time.time()-t0:.1f}s")
        logger.info(f"\n  树结构:\n{tree.list_tree()}")

        acc = evaluate(
            client, tree.root.skill.system_prompt, test_data,
            f"round {round_num}",
        )
        if acc > best_acc:
            best_acc = acc
            logger.info(f"  ★ 新最佳: {best_acc:.1%}")

    # ── 总结 ──
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"完成! 总耗时 {elapsed/60:.1f} 分钟")
    logger.info(f"  baseline:  {baseline_acc:.1%}")
    logger.info(f"  best:      {best_acc:.1%} ({best_acc - baseline_acc:+.1%})")
    logger.info(f"  leaves:    {tree.root.leaf_count()}")
    logger.info(f"\n  最终树结构:\n{tree.list_tree()}")
    logger.info(f"\n  根 prompt:\n{tree.root.skill.system_prompt[:300]}...")
    logger.info(f"\n输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
