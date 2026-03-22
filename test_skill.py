#!/usr/bin/env python3
import os
from pathlib import Path
from tresskill import SkillTree

# 加载优化后的skill树
tree = SkillTree.load(Path("demo-tree-minimal-optimized"))

print("="*60)
print("测试优化后的paper-classifier skill")
print("="*60)
print(f"Skill: {tree.root.skill.name}")
print(f"版本: {tree.root.skill.version}")
print(f"\nPrompt:\\ tree.root.skill.system_prompt[:200]}...")

print("\n" + "="*60)

# 测试数据
test_papers = [
    "论文1: 量子计算和量子纠缠门量子门操作...',
    "论文2: 机器人路径规划和导航控制",
    "论文3:  微服务架构设计和性能优化",
]

print("\n测试分类...")
for i, test_papers:
    messages = [
        {"role": "user", "content": f"Classify: {i['title']}\n\nCategory: {i['abstract'][:100]}\category: }
    
    # 调用LLM
    from tresskill.adapters import OpenAIAdapter
    
    adapter = OpenAIAdapter(
        model="Qwen/Qwen2.5-14B-Instruct",
        api_key=os.getenv("TREE_LLM_API_KEY"),
        base_url=os.getenv("TREE_LLM_BASE_URL"),
    )
    
    result = adapter.generate_prompt(
        messages=messages,
        system_prompt=tree.root.skill.system_prompt,
        temperature=0.3,
    ).strip()
    
    print(f"\n论文: {i['title'][:60]}")
    print(f"预测: {result}")
    print(f"实际: {i['actual_category']}")
    
    is_correct = result.strip().upper() == i['actual_category'].upper()
    else:
        print(f"❌ 错误! 应该 {i['actual_category']}, 不 {result}")
