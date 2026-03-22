# OpenAI Adapter 使用指南

## 概述

`OpenAIAdapter` 是用于OpenAI API（及其兼容API）的适配器实现，支持：
- ✅ 所有OpenAI模型（GPT-4o, GPT-4o-mini, GPT-4-turbo等）
- ✅ Vision模型（图像理解）
- ✅ 推理模型（o1-preview, o1-mini）
- ✅ OpenAI兼容API（如SiliconFlow、Azure OpenAI等）
- ✅ 精确的token计数（tiktoken）
- ✅ 上下文管理
- ✅ TGD梯度计算和应用

---

## 快速开始

### 1. 安装依赖

```bash
conda activate pr
pip install openai tiktoken
```

### 2. 基本使用

```python
from treeskill.core import TextPrompt
from treeskill.adapters.openai import OpenAIAdapter

# 创建适配器
adapter = OpenAIAdapter(
    model="gpt-4o-mini",
    api_key="your-api-key",  # 或设置 OPENAI_API_KEY 环境变量
)

# 创建Prompt
prompt = TextPrompt(
    content="你是一个友好的助手。",
    name="assistant",
)

# 生成响应
response = adapter.generate(prompt)
print(response)
```

### 3. 使用对话上下文

```python
from treeskill.core import ConversationExperience

# 创建对话历史
experiences = [
    ConversationExperience(
        messages=[{"role": "user", "content": "我叫张三"}],
        response="你好，张三！",
    ),
]

# 生成时带上下文
response = adapter.generate(prompt, context=experiences)
```

### 4. Prompt优化

```python
from treeskill.core import CompositeFeedback

# 创建失败经验
bad_experience = ConversationExperience(
    messages=[{"role": "user", "content": "你好"}],
    response="你好。",
    feedback=CompositeFeedback(critique="太冷淡"),
)

# 计算梯度
gradient = adapter.compute_gradient(
    prompt=prompt,
    failures=[bad_experience],
    target="更友好"
)

# 应用梯度
new_prompt = adapter.apply_gradient(prompt, gradient)
print(f"优化后: {new_prompt.version}")  # v1.0 → v1.1
```

---

## 支持的模型

| 模型 | Vision | Context | 推荐场景 |
|------|--------|---------|----------|
| `gpt-4o` | ✅ | 128K | 旗舰模型，多模态任务 |
| `gpt-4o-mini` | ✅ | 128K | 快速、便宜，日常使用 |
| `gpt-4-turbo` | ✅ | 128K | 旧版本，不推荐 |
| `gpt-4` | ❌ | 8K | 旧版本，不推荐 |
| `gpt-3.5-turbo` | ❌ | 16K | 便宜，简单任务 |
| `o1-preview` | ❌ | 128K | 复杂推理任务 |
| `o1-mini` | ❌ | 128K | 快速推理 |

**推荐**：
- 日常使用：`gpt-4o-mini`
- 复杂任务：`gpt-4o`
- 优化梯度计算（Judge）：`gpt-4o` 或 `gpt-4o-mini`

---

## OpenAI兼容API

### SiliconFlow（已测试 ✅）

```python
adapter = OpenAIAdapter(
    model="Qwen/Qwen2.5-14B-Instruct",
    api_key="sk-...",
    base_url="https://api.siliconflow.cn/v1",
)
```

### Azure OpenAI

```python
adapter = OpenAIAdapter(
    model="gpt-4o",
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/",
)
```

---

## 高级功能

### 1. Token计数

```python
# 精确计数
prompt = TextPrompt(content="你好，这是一个测试。")
tokens = adapter.count_tokens(prompt)
print(f"Tokens: {tokens}")

# 对话格式计数
messages = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "你好"},
]
total_tokens = adapter.count_messages_tokens(messages)
```

### 2. Prompt验证

```python
from treeskill.core import MultimodalPrompt

# 多模态Prompt
mm_prompt = MultimodalPrompt(
    text="分析这张图片",
    images=["photo.jpg"],
)

# 验证兼容性
issues = adapter.validate_prompt(mm_prompt)
if issues:
    print(f"问题: {issues}")
else:
    print("✓ 兼容")
```

### 3. 推理模型（o1系列）

```python
# o1模型有特殊限制（无system message）
adapter = OpenAIAdapter(model="o1-preview")

# System prompt会自动转为第一条user message
prompt = TextPrompt(content="请仔细分析...")
response = adapter.generate(prompt)
```

### 4. 生成参数

```python
response = adapter.generate(
    prompt,
    temperature=0.7,      # 温度
    max_tokens=500,       # 最大生成token数
    top_p=0.9,           # nucleus sampling
    presence_penalty=0.6, # 存在惩罚
    frequency_penalty=0.5, # 频率惩罚
)
```

---

## 测试结果

### SiliconFlow API 测试 ✅

```bash
$ python tests/test_openai_siliconflow.py

Test 1: Basic Generation
✓ Response: 好的，我会尽力简洁地回答您的问题...

Test 2: Gradient Computation
✓ Gradient computed successfully
✓ Applied gradient: v1.0 → v1.1
  New content: 你是一个友好、热情的助手...

Test 3: Context
✓ Generation with context successful

All tests completed! ✓
```

---

## 最佳实践

### 1. 模型选择

```python
# 聊天/生成：快速
chat_adapter = OpenAIAdapter(model="gpt-4o-mini")

# 梯度计算（Judge）：准确
judge_adapter = OpenAIAdapter(model="gpt-4o")
gradient = judge_adapter.compute_gradient(prompt, failures)

# 复杂推理：深度
reasoning_adapter = OpenAIAdapter(model="o1-preview")
```

### 2. 成本控制

```python
# 使用mini模型进行快速迭代
adapter = OpenAIAdapter(model="gpt-4o-mini")

# 验证prompt长度，避免超出限制
tokens = adapter.count_tokens(prompt)
if tokens > adapter.max_context_tokens * 0.8:
    print("⚠️ Prompt接近上下文限制，考虑精简")
```

### 3. 错误处理

```python
from openai import APIError, RateLimitError

try:
    response = adapter.generate(prompt)
except RateLimitError:
    print("API速率限制，等待后重试...")
    time.sleep(60)
    response = adapter.generate(prompt)
except APIError as e:
    print(f"API错误: {e}")
```

### 4. 批量处理

```python
# 批量生成（注意速率限制）
responses = []
for prompt in prompts:
    try:
        response = adapter.generate(prompt)
        responses.append(response)
    except RateLimitError:
        time.sleep(1)  # 等待1秒
        responses.append(adapter.generate(prompt))
```

---

## 环境变量配置

```bash
# .env 文件
OPENAI_API_KEY=sk-...

# 或使用OpenAI兼容API
TREE_LLM_API_KEY=sk-...
TREE_LLM_BASE_URL=https://api.siliconflow.cn/v1
TREE_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
TREE_LLM_JUDGE_MODEL=Qwen/Qwen2.5-72B-Instruct
```

```python
# 自动读取环境变量
import os
from dotenv import load_dotenv

load_dotenv()

adapter = OpenAIAdapter(
    model=os.getenv("TREE_LLM_MODEL"),
    api_key=os.getenv("TREE_LLM_API_KEY"),
    base_url=os.getenv("TREE_LLM_BASE_URL"),
)
```

---

## 常见问题

### Q: 如何处理vision任务？

```python
from treeskill.core import MultimodalPrompt

# 创建多模态prompt
prompt = MultimodalPrompt(
    text="分析这张图片中的产品缺陷",
    images=["defect_photo.jpg"],
)

# 确保使用vision模型
adapter = OpenAIAdapter(model="gpt-4o")  # 或 gpt-4o-mini

# 验证
issues = adapter.validate_prompt(prompt)
if issues:
    print(f"不兼容: {issues}")
else:
    response = adapter.generate(prompt)
```

### Q: 为什么o1模型没有system prompt？

A: o1系列是推理模型，OpenAI限制它不能使用system message。Adapter会自动将system prompt转换为第一条user message。

### Q: 如何处理速率限制？

```python
import time
from openai import RateLimitError

def generate_with_retry(adapter, prompt, max_retries=3):
    for i in range(max_retries):
        try:
            return adapter.generate(prompt)
        except RateLimitError:
            if i < max_retries - 1:
                wait_time = 60  # 等待60秒
                print(f"速率限制，等待{wait_time}秒...")
                time.sleep(wait_time)
            else:
                raise
```

### Q: 如何减少token使用？

```python
# 1. 精简prompt
prompt = TextPrompt(content="助手。")  # 而不是"你是一个非常专业的助手..."

# 2. 使用mini模型
adapter = OpenAIAdapter(model="gpt-4o-mini")

# 3. 限制生成长度
response = adapter.generate(prompt, max_tokens=100)

# 4. 缓存结果（对于相同的输入）
```

---

## 示例代码

完整的示例代码请参考：
- `tests/test_openai_adapter.py` - 基础测试（不需要API）
- `tests/test_openai_siliconflow.py` - 完整测试（使用SiliconFlow API）
- `examples/mock_adapter.py` - Mock adapter示例

---

## 下一步

- [ ] 实现 AnthropicAdapter（Claude API）

- [ ] 构建优化引擎
- [ ] 集成到CLI

---

## 参考资料

- [OpenAI API文档](https://platform.openai.com/docs/)
- [tiktoken文档](https://github.com/openai/tiktoken)
- [SiliconFlow文档](https://siliconflow.cn/docs)
