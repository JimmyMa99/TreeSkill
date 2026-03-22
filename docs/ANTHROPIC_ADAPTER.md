# Anthropic Adapter 使用指南

## 概述

`AnthropicAdapter` 是用于Anthropic Claude API的适配器实现，支持：
- ✅ Claude 4.5 系列模型（Opus、Sonnet、Haiku）
- ✅ Claude 3.5 / 3 系列模型（向下兼容）
- ✅ Vision 支持（所有 Claude 3+ 模型）
- ✅ 最大 1M 上下文（Claude 4.5 Opus）
- ✅ 独特的system prompt处理
- ✅ TGD梯度计算和应用

---

## 快速开始

### 1. 安装依赖

```bash
conda activate pr
pip install anthropic
```

### 2. 设置API密钥

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

### 3. 基本使用

```python
from treeskill.core import TextPrompt
from treeskill.adapters.anthropic import AnthropicAdapter

# 创建适配器
adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")

# 创建Prompt
prompt = TextPrompt(
    content="你是一个友好的助手。",
    name="assistant",
)

# 生成响应
response = adapter.generate(prompt)
print(response)
```

---

## 支持的模型

| 模型 | 上下文 | Vision | 推荐场景 | 成本 |
|------|--------|--------|----------|------|
| `claude-3-5-sonnet-20241022` | 200K | ✅ | 日常使用，平衡性能和成本 | $$ |
| `claude-3-5-haiku-20241022` | 200K | ✅ | 快速响应，成本敏感 | $ |
| `claude-3-opus-20240229` | 200K | ✅ | 复杂任务，最高质量 | $$$ |
| `claude-3-sonnet-20240229` | 200K | ✅ | 旧版本Sonnet | $$ |
| `claude-3-haiku-20240307` | 200K | ✅ | 旧版本Haiku | $ |

**推荐**：
- **日常使用**：`claude-3-5-sonnet-20241022`（最新Sonnet）
- **快速迭代**：`claude-3-5-haiku-20241022`（最快最便宜）
- **复杂推理**：`claude-3-opus-20240229`（最强）

---

## 核心特性

### 1. System Prompt处理

Claude的system prompt与OpenAI不同，它是**单独的参数**，不在messages里：

```python
# OpenAI风格（混在messages里）
messages = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "你好"},
]

# Claude风格（system单独参数）
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system="你是助手",  # 单独参数
    messages=[
        {"role": "user", "content": "你好"},
    ],
)
```

**Adapter自动处理**：
```python
# 使用Adapter时，无需关心底层差异
prompt = TextPrompt(content="你是助手")
response = adapter.generate(prompt, context=experiences)
# Adapter会自动将prompt.content作为system参数
```

### 2. 超长上下文（200K tokens）

```python
# Claude支持200K上下文，相当于~500页文本
long_document = load_large_document()  # 假设有500页
prompt = TextPrompt(
    content="总结这个文档",
    # 可以传入超长上下文
)

# 自动验证
issues = adapter.validate_prompt(prompt)
if issues:
    print(f"问题: {issues}")
else:
    response = adapter.generate(prompt, context=long_context)
```

### 3. Vision支持

```python
from treeskill.core import MultimodalPrompt

# 创建多模态Prompt
prompt = MultimodalPrompt(
    text="分析这张图片中的产品缺陷",
    images=["defect_photo.jpg"],
)

# 所有Claude 3+都支持vision
adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")

# 验证（总是通过，因为所有Claude 3+都支持vision）
issues = adapter.validate_prompt(prompt)  # []

# 生成
response = adapter.generate(prompt)
```

### 4. 对话上下文

```python
from treeskill.core import ConversationExperience

# 创建对话历史
experiences = [
    ConversationExperience(
        messages=[{"role": "user", "content": "我叫张三"}],
        response="你好，张三！",
    ),
    ConversationExperience(
        messages=[{"role": "user", "content": "我喜欢编程"}],
        response="很棒的爱好！",
    ),
]

# 生成时带上下文
prompt = TextPrompt(content="你是一个助手。")
response = adapter.generate(prompt, context=experiences)
# Claude会记住之前的对话
```

### 5. Prompt优化

```python
from treeskill.core import CompositeFeedback

# 创建失败经验
bad_exp = ConversationExperience(
    messages=[{"role": "user", "content": "你好"}],
    response="你好。",
    feedback=CompositeFeedback(critique="太冷淡"),
)

# 计算梯度
gradient = adapter.compute_gradient(
    prompt=prompt,
    failures=[bad_exp],
    target="更友好"
)

# 应用梯度
new_prompt = adapter.apply_gradient(prompt, gradient)
print(f"优化后: {new_prompt.version}")  # v1.0 → v1.1
```

---

## 高级功能

### 1. 工厂函数

```python
from treeskill.adapters.anthropic import (
    create_claude_35_sonnet,
    create_claude_35_haiku,
    create_claude_3_opus,
)

# 推荐方式
sonnet = create_claude_35_sonnet()  # 最新Sonnet
haiku = create_claude_35_haiku()    # 最新Haiku
opus = create_claude_3_opus()        # Opus
```

### 2. 自定义参数

```python
response = adapter.generate(
    prompt,
    temperature=0.7,     # 温度
    max_tokens=4096,     # 最大生成token（Claude必需）
    top_p=0.9,           # nucleus sampling
    top_k=50,            # top-k sampling
)
```

### 3. Token计数

```python
# 文本token计数（近似）
prompt = TextPrompt(content="你好，世界")
tokens = adapter.count_tokens(prompt)
print(f"Tokens: {tokens}")

# 消息token计数（包含格式开销）
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
]
total = adapter.count_messages_tokens(messages, system="你是助手")
```

**注意**：Anthropic未提供官方tokenizer，使用近似算法（~4字符/token）。

### 4. 错误处理

```python
from anthropic import APIError, RateLimitError

try:
    response = adapter.generate(prompt)
except RateLimitError:
    print("速率限制，等待后重试...")
    time.sleep(60)
except APIError as e:
    print(f"API错误: {e}")
```

---

## 测试结果

### 基础功能测试 ✅

```bash
$ python tests/test_anthropic_adapter.py

Test 4: Token Counting
✓ Text token counting
✓ Message token counting

Test 5: Model Variants
✓ claude-3-5-sonnet-20241022
✓ claude-3-5-haiku-20241022
✓ claude-3-opus-20240229
✓ claude-3-sonnet-20240229
✓ claude-3-haiku-20240307

Test 6: Vision Capability
✓ Vision capability supported

Test 7: Factory Functions
✓ Created Claude 3.5 Sonnet
✓ Created Claude 3.5 Haiku

All tests completed! ✓
```

---

## 与OpenAI的区别

| 特性 | OpenAI | Anthropic |
|------|--------|-----------|
| System Prompt | 在messages中 | 单独参数 |
| 上下文长度 | 128K（GPT-4o） | 200K（所有模型） |
| Vision支持 | 部分模型 | 所有Claude 3+ |
| Tokenizer | tiktoken（精确） | 近似算法 |
| 消息格式 | Open | 严格user/assistant交替 |

---

## 最佳实践

### 1. 模型选择

```python
# 日常使用：Sonnet（平衡）
sonnet = create_claude_35_sonnet()

# 快速迭代：Haiku（便宜快速）
haiku = create_claude_35_haiku()

# 复杂任务：Opus（最强）
opus = create_claude_3_opus()

# 梯度计算（Judge）：Sonnet
judge = create_claude_35_sonnet()
gradient = judge.compute_gradient(prompt, failures)
```

### 2. 成本控制

```python
# 1. 使用Haiku进行快速迭代
adapter = create_claude_35_haiku()

# 2. 限制生成长度
response = adapter.generate(prompt, max_tokens=500)

# 3. 利用200K上下文（减少多轮对话成本）
# 可以一次性传入完整历史，而不是分多轮

# 4. 精简prompt
prompt = TextPrompt(content="助手。")  # 而不是"你是一个非常专业的..."
```

### 3. 上下文管理

```python
# Claude的200K上下文允许传完整文档
long_doc = load_book()  # 加载整本书

prompt = TextPrompt(
    content="总结这本书的主要观点",
)

# 验证是否超出限制
tokens = adapter.count_messages_tokens(
    messages=[{"role": "user", "content": long_doc}],
    system=prompt.content,
)

if tokens > adapter.max_context_tokens * 0.9:
    print("⚠️ 文档接近上下文限制")
else:
    response = adapter.generate(prompt, context=[...])
```

### 4. Vision任务

```python
# Vision任务推荐Sonnet（比Haiku质量高）
adapter = create_claude_35_sonnet()

prompt = MultimodalPrompt(
    text="详细分析这张产品图片",
    images=["product.jpg"],
)

response = adapter.generate(prompt, max_tokens=2000)
```

---

## 常见问题

### Q: Claude的system prompt与OpenAI有何不同？

**A**: Claude的system prompt是**单独的参数**，不在messages列表里：

```python
# ❌ 错误（会跳过system消息）
messages = [
    {"role": "system", "content": "你是助手"},  # 会被忽略
    {"role": "user", "content": "你好"},
]

# ✅ 正确（Adapter自动处理）
prompt = TextPrompt(content="你是助手")
response = adapter.generate(prompt)  # content会作为system参数
```

### Q: 为什么需要max_tokens参数？

**A**: Claude API**必需**指定max_tokens，不像OpenAI有默认值。Adapter默认使用4096。

```python
# Claude必需指定
response = adapter.generate(prompt, max_tokens=1000)
```

### Q: 如何处理速率限制？

```python
import time
from anthropic import RateLimitError

def generate_with_retry(adapter, prompt, max_retries=3):
    for i in range(max_retries):
        try:
            return adapter.generate(prompt)
        except RateLimitError:
            if i < max_retries - 1:
                wait_time = 60
                print(f"速率限制，等待{wait_time}秒...")
                time.sleep(wait_time)
            else:
                raise
```

### Q: Token计数准确吗？

**A**: 不精确。Anthropic未提供官方tokenizer，使用近似算法（~4字符/token）。对于精确计费，请参考API返回的usage字段。

```python
# 近似计数
tokens = adapter.count_tokens(prompt)

# 精确计数（需要API调用）
response = adapter.client.messages.create(...)
actual_tokens = response.usage.input_tokens
```

---

## 示例代码

### 完整示例：多轮对话

```python
from treeskill.core import TextPrompt, ConversationExperience
from treeskill.adapters.anthropic import create_claude_35_sonnet

# 创建适配器
adapter = create_claude_35_sonnet()

# System prompt
prompt = TextPrompt(
    content="你是一个专业的Python编程助手。",
)

# 对话历史
experiences = []

# 第一轮
user_msg = "什么是装饰器？"
response = adapter.generate(
    prompt,
    context=experiences,
    max_tokens=1000,
)
print(f"Assistant: {response}")

# 保存历史
experiences.append(ConversationExperience(
    messages=[{"role": "user", "content": user_msg}],
    response=response,
))

# 第二轮（有上下文）
user_msg = "给我一个例子"
response = adapter.generate(
    prompt,
    context=experiences,
    max_tokens=1000,
)
print(f"Assistant: {response}")
```

---

## 环境变量

```bash
# .env 文件
ANTHROPIC_API_KEY=sk-ant-...

# 可选：自定义base URL
ANTHROPIC_BASE_URL=https://custom-anthropic-endpoint.com
```

```python
# 自动读取
import os
from dotenv import load_dotenv

load_dotenv()

adapter = AnthropicAdapter(
    model="claude-3-5-sonnet-20241022",
    # 自动从环境变量读取
)
```

---

## 下一步

- [ ] 构建优化引擎（`optimizer/engine.py`）
- [ ] 集成到CLI
- [ ] 实现流式输出
- [ ] 添加工具调用支持

---

## 参考资料

- [Anthropic API文档](https://docs.anthropic.com/)
- [Claude模型对比](https://www.anthropic.com/claude)
- [Vision能力](https://docs.anthropic.com/claude/docs/vision)

---

## 与OpenAIAdapter对比

| 特性 | OpenAIAdapter | AnthropicAdapter |
|------|---------------|------------------|
| 模型数量 | 10+ | 5 |
| Vision支持 | 部分模型 | 所有Claude 3+ ✅ |
| 上下文长度 | 128K | 200K ✅ |
| Tokenizer | tiktoken（精确）✅ | 近似算法 |
| System prompt | messages中 | 单独参数 |
| 推理模型 | o1系列 | 无 |
| 成本 | 中等 | 较高 |

**推荐使用场景**：
- **OpenAI**：快速迭代、成本敏感、需要推理模型
- **Anthropic**：长文档处理、Vision任务、需要最高质量

---

*Generated on 2026-03-17*
