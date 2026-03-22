"""End-to-end test: upgraded APO with multi-candidate scoring + template diversity."""

import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()

from tresskill.config import GlobalConfig
from tresskill.llm import LLMClient
from tresskill.optimizer import APOEngine
from tresskill.resume import ResumeState
from tresskill.schema import Feedback, Message, Skill, Trace
from tresskill.skill import save as save_skill
from tresskill.skill_tree import SkillTree


def main():
    config = GlobalConfig()
    # Use 2 candidates for testing
    config = config.model_copy(
        update={"apo": config.apo.model_copy(update={"num_candidates": 2})}
    )
    llm = LLMClient(config)
    engine = APOEngine(config, llm)

    tmpdir = tempfile.mkdtemp(prefix="evo_test_")
    print(f"Temp dir: {tmpdir}")

    try:
        _run_tests(tmpdir, engine)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n✅ 端到端测试全部通过！")


def _run_tests(tmpdir, engine):
    # === Step 1: 创建 Skill ===
    print("\n=== Step 1: 创建测试 Skill ===")
    root = Skill(
        name="qa-assistant",
        description="简洁问答助手",
        system_prompt="你是一个问答助手。回答用户的问题。",
        target="更简洁准确，不超过两句话，禁止套话",
        version="v1.0",
    )
    save_skill(root, tmpdir)
    tree = SkillTree.load(tmpdir)
    print(f"  Skill: {tree.root.name} ({tree.root.skill.version})")

    # === Step 2: 构造 traces ===
    print("\n=== Step 2: 构造 feedback traces ===")
    traces = [
        Trace(
            inputs=[Message(role="user", content="什么是量子计算？")],
            prediction=Message(
                role="assistant",
                content=(
                    "量子计算是一种利用量子力学原理进行计算的技术。它使用量子比特而非经典比特，"
                    "能够在某些问题上实现指数级加速。量子计算机利用叠加态和纠缠态来处理信息，"
                    "目前仍处于早期阶段，但在密码学、药物发现等领域有巨大潜力。"
                ),
            ),
            feedback=Feedback(score=0.3, critique="回答太长了，应该两句话以内"),
        ),
        Trace(
            inputs=[Message(role="user", content="Python的GIL是什么？")],
            prediction=Message(
                role="assistant",
                content="作为AI助手，我很乐意为您解答。GIL是Python的全局解释器锁。",
            ),
            feedback=Feedback(score=0.2, critique="有AI套话，不够直接"),
        ),
        Trace(
            inputs=[Message(role="user", content="HTTP和HTTPS的区别？")],
            prediction=Message(
                role="assistant",
                content="这是一个好问题！HTTPS是HTTP的安全版本，使用SSL/TLS加密。",
            ),
            feedback=Feedback(score=0.3, critique="'这是一个好问题'是废话"),
        ),
    ]
    print(f"  {len(traces)} 条 feedback traces")

    # === Step 3: 单节点优化（多候选 + 评分） ===
    print("\n=== Step 3: 多候选 APO 优化 ===")
    new_skill = engine.optimize(tree.root.skill, traces)
    if new_skill.version != tree.root.skill.version:
        print(f"  ✓ 版本: {tree.root.skill.version} → {new_skill.version}")
        print(f"  新 prompt:\n    {new_skill.system_prompt[:200]}...")
    else:
        print("  ⚠ 所有候选都未超过原 prompt（这也是正确行为）")
    tree.root.skill = new_skill

    # === Step 4: Save roundtrip ===
    print("\n=== Step 4: Save/Load roundtrip ===")
    tree.save()
    tree2 = SkillTree.load(tmpdir)
    assert tree2.root.skill.version == new_skill.version
    assert tree2.root.skill.target == "更简洁准确，不超过两句话，禁止套话"
    print("  ✓ roundtrip 正确")

    # === Step 5: evolve_tree ===
    print("\n=== Step 5: evolve_tree + resume ===")
    resume = ResumeState.create(tmpdir, metadata={"test": True})
    done = []

    def on_done(dotpath, node):
        done.append(dotpath)
        print(f"  ✓ {dotpath} → {node.skill.version}")

    engine.evolve_tree(tree2, traces, resume=resume, on_node_done=on_done)
    print(f"  优化了 {len(done)} 个节点")
    resume.clear()

    # === Step 6: 最终状态 ===
    print("\n=== Step 6: 最终状态 ===")
    final = SkillTree.load(tmpdir)
    print(f"  版本: {final.root.skill.version}")
    print(f"  树结构:")
    print(final.list_tree())


if __name__ == "__main__":
    main()
