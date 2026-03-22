#!/usr/bin/env python3
"""快速测试 API 调用"""

import os
from dotenv import load_dotenv
from tresskill import OpenAIAdapter

load_dotenv()

# 创建 adapter
api_key = os.getenv('TREE_LLM_API_KEY')
base_url = os.getenv('TREE_LLM_BASE_URL')

print(f"API Key: {api_key[:20]}...")
print(f"Base URL: {base_url}")

adapter = OpenAIAdapter(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key=api_key,
    base_url=base_url
)

print(f"\n✅ Adapter created: {adapter.model_name}")

# 测试简单调用
print("\n测试 API 调用...")
try:
    response = adapter.client.chat.completions.create(
        model=adapter.model_name,
        messages=[{"role": "user", "content": "Classify this paper into A or B:\nTitle: Quantum Computing\n\nCategory:"}],
        max_tokens=10,
        temperature=0.3,
    )
    answer = response.choices[0].message.content.strip()
    print(f"✅ API 响应成功: {answer}")
except Exception as e:
    print(f"❌ API 调用失败: {e}")
    import traceback
    traceback.print_exc()
