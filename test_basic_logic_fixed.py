#!/usr/bin/env python3
"""
基础逻辑单元测试 - 验证核心功能
"""
import os
import sys
from pathlib import Path

# 测试1: 基础导入
print("="*60)
print("测试1: 基础导入")
print("="*60)
try:
    from tresskill import (
        OpenAIAdapter,
        TreeAwareOptimizer,
        TreeOptimizerConfig,
        OptimizerConfig,
        SkillTree,
        TextPrompt,
        ConversationExperience,
        CompositeFeedback,
    )
    from tresskill.schema import Skill, SkillMeta
    from tresskill.skill_tree import SkillNode
    print("✅ 所有模块导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: 创建基本对象
print("\n" + "="*60)
print("测试2: 创建基本对象")
print("="*60)
try:
    # 创建TextPrompt
    prompt = TextPrompt(content="Test prompt")
    print(f"✅ TextPrompt创建成功: {prompt.content[:20]}...")
    
    # 创建Skill
    skill = Skill(
        name="test-skill",
        version="v1.0",
        system_prompt="Test system prompt",
        meta=SkillMeta(name="test", description="Test skill")
    )
    print(f"✅ Skill创建成功: {skill.name}")
    
    # 创建SkillNode
    node = SkillNode(name="test-node", skill=skill)
    print(f"✅ SkillNode创建成功: {node.name}")
    
    # 创建SkillTree
    tree = SkillTree(root=node, base_path=Path("test-tree/"))
    print(f"✅ SkillTree创建成功: {tree.root.name}")
    
except Exception as e:
    print(f"❌ 对象创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 创建Experience和Feedback
print("\n" + "="*60)
print("测试3: 创建Experience和Feedback")
print("="*60)
try:
    exp = ConversationExperience(
        messages=[{"role": "user", "content": "test question"}],
        response="test answer",
        metadata={"test": "data"}
    )
    
    exp.feedback = CompositeFeedback(
        critique="Test feedback",
        score=0.8,
    )
    print(f"✅ Experience创建成功: score={exp.feedback.to_score()}")
    
except Exception as e:
    print(f"❌ Experience创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: TreeOptimizerConfig
print("\n" + "="*60)
print("测试4: TreeOptimizerConfig")
print("="*60)
try:
    config = TreeOptimizerConfig(
        auto_split=True,
        auto_prune=False,
        max_tree_depth=3,
        min_samples_for_split=5,
        prune_threshold=0.3,
        optimization_order="bottom_up",
    )
    print(f"✅ TreeOptimizerConfig创建成功")
    print(f"   auto_split: {config.auto_split}")
    print(f"   auto_prune: {config.auto_prune}")
    print(f"   max_tree_depth: {config.max_tree_depth}")
    
except Exception as e:
    print(f"❌ TreeOptimizerConfig创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: _extract_prompt_text方法
print("\n" + "="*60)
print("测试5: _extract_prompt_text方法")
print("="*60)
try:
    # 创建一个mock adapter
    class MockAdapter:
        pass
    
    optimizer = TreeAwareOptimizer(
        adapter=MockAdapter(),
        config=config,
    )
    
    # 测试字符串
    text = optimizer._extract_prompt_text("simple string")
    print(f"✅ 字符串提取: '{text}'")
    assert text == "simple string", f"字符串提取失败: got '{text}'"
    
    # 测试TextPrompt
    prompt_obj = TextPrompt(content="test content")
    text = optimizer._extract_prompt_text(prompt_obj)
    print(f"✅ TextPrompt提取: '{text}'")
    assert text == "test content", f"TextPrompt提取失败: got '{text}'"
    
    print("✅ _extract_prompt_text方法测试通过")
    
except Exception as e:
    print(f"❌ _extract_prompt_text方法测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: SkillTree保存和加载
print("\n" + "="*60)
print("测试6: SkillTree保存和加载")
print("="*60)
try:
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    
    # 保存
    tree.save(temp_dir)
    print(f"✅ SkillTree保存成功: {temp_dir}")
    
    # 加载
    loaded_tree = SkillTree.load(temp_dir)
    print(f"✅ SkillTree加载成功: {loaded_tree.root.name}")
    
    # 验证
    assert loaded_tree.root.skill.name == skill.name, "加载的skill不匹配"
    assert loaded_tree.root.skill.system_prompt == skill.system_prompt, "加载的prompt不匹配"
    print("✅ SkillTree保存/加载测试通过")
    
    # 清理
    shutil.rmtree(temp_dir)
    
except Exception as e:
    print(f"❌ SkillTree保存/加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ 所有基础测试通过!")
print("="*60)
