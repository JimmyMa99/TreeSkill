#!/usr/bin/env python3
"""
Qwen3-8B 快速测试 - 验证模型是否可用
"""
import os
from dotenv import load_dotenv

load_dotenv()

from tresskill import OpenAIAdapter

def test_qwen3_8b():
    print("🧪 测试 Qwen3-8B 是否可用...")

    api_key = os.getenv("TRES_LLM_API_KEY")
    base_url = os.getenv("TRES_LLM_BASE_URL", "https://api.siliconflow.cn/v1")

    # 尝试Qwen3-8B
    adapter = OpenAIAdapter(model="Qwen/Qwen3-8B", api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": "You are a classifier. Classify into A or B."},
        {"role": "user", "content": "Test. Return A or B."},
    ]

    try:
        result = adapter._call_api(messages=messages, system=None, temperature=0.3)
        print(f"✅ Qwen3-8B 响应成功: {result}")
        return True
    except Exception as e:
        print(f"❌ Qwen3-8B 失败: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen3_8b()
    print(f"\n{'✅ 测试通过' if success else '❌ 测试失败'}")
