#!/usr/bin/env python3
from pathlib import Path
from tresskill import SkillTree

# 加载优化后的skill树
tree = SkillTree.load(Path("demo-tree-minimal-optimized"))

print("="*60)
print("测试优化后的skill树")
print("="*60)
print(f"根节点: {tree.root.name}")
print(f"版本: {tree.root.version}")
print("\nPrompt预览:")
print(tree.root.skill.system_prompt[:300] + "...")

print("\n" + "="*60)

# 测试数据
test_papers = [
    {"title": "论文1: 量子计算中的量子纠缠",量子门操作", "abstract": "探讨量子纠缠在量子计算中的应用", "actual_category": "A"},
    {"title": "论文2: 机器人路径规划和导航控制", "abstract": "研究机器人导航中的路径规划和避障问题", "actual_category": "e"},
    {"title": "论文3: 微服务架构中的性能优化", "abstract": "本文研究了微服务架构的性能优化方法", "actual_category": "g"}
]

print("\n测试分类...")

for i in test_papers:
    # 訡拟用户输入
    user_input = f"分类论文:\n\n标题: {i['title']}\n摘要: {i['abstract']}\n\nReturn ONLY the letter。"""

    # 訡拟LLM调用
    from tresskill.adapters import OpenAIAdapter
    
    adapter = OpenAIAdapter(
        model="Qwen/Qwen2.5-14B-Instruct",
        api_key=os.getenv("TREE_LLM_API_KEY"),
        base_url=os.getenv("TREE_LLM_BASE_URL"),
    )
    
    # 调用API
    response = adapter._call_api(
        messages=[{"role": "user", "content": user_input}],
        system=tree.root.skill.system_prompt,
        temperature=0.3,
    ).strip()
    
    print(f"\n论文: {i['title']}")
    print(f"预测: {response}")
    print(f"实际: {i['actual_category']}")
    
    if response.strip().upper() == i['actual_category'].upper():
        print(f"✅ 正确!")
    else:
        print(f"❌ 错误! 应该 {i['actual_category']}, not {response}")
    
print("\n" + "="*60)
print("="*60)
print("所有测试完成!")
correct = sum(1 for i in test_papers if i['actual_category'] == i['predicted_category'].upper() else 0)
print(f"准确率: {correct}/{len(test_papers)}")
