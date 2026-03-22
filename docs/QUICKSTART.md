# 核心抽象层快速开始

## 安装

```bash
conda activate pr
pip install -e .
```

## 5分钟快速开始

### 1. 创建Prompt

```python
from treeskill.core import TextPrompt

prompt = TextPrompt(
    content="你是一个AI助手。",
    name="my-assistant",
    version="v1.0",
    target="更友好、更自然"
)
```

### 2. 收集经验

```python
from treeskill.core import ConversationExperience, CompositeFeedback

# 创建经验
exp = ConversationExperience(
    messages=[{"role": "user", "content": "你好"}],
    response="你好。",
)

# 添加反馈
exp_with_feedback = exp.attach_feedback(
    CompositeFeedback(critique="太冷淡，不够友好")
)

print(exp_with_feedback.is_failure)  # True
```

### 3. 使用MockAdapter测试

```python
from treeskill import MockAdapter

adapter = MockAdapter()

# 验证Prompt
issues = adapter.validate_prompt(prompt)
print(f"兼容性: {len(issues)} 个问题")

# 生成响应
response = adapter.generate(prompt)
print(f"响应: {response}")
```

### 4. 优化Prompt

```python
# 计算梯度
gradient = adapter.compute_gradient(
    prompt=prompt,
    failures=[exp_with_feedback],
    target="更友好"
)

# 应用梯度
new_prompt = adapter.apply_gradient(prompt, gradient)

print(f"版本: {prompt.version} → {new_prompt.version}")
print(f"新内容:\n{new_prompt.content}")
```

## 运行示例

```bash
# 运行MockAdapter示例
python examples/mock_adapter.py

# 运行单元测试
python tests/test_core_abstractions.py
```

## 多模态示例

```python
from treeskill.core import MultimodalPrompt, MultimodalExperience

# 创建多模态Prompt
prompt = MultimodalPrompt(
    text="分析这张图片中的产品缺陷",
    images=["defect_photo.jpg"],
    name="defect-analyzer",
)

# 创建多模态经验
exp = MultimodalExperience(
    input_text="这个产品有什么问题？",
    input_images=["product.jpg"],
    output_text="看起来有个划痕",
    feedback=CompositeFeedback(
        score=0.5,
        critique="应该更具体地描述缺陷位置"
    ),
)
```

## 下一步

1. **实现真实适配器**：`adapters/openai.py`, `adapters/anthropic.py`
2. **构建优化引擎**：`optimizer/engine.py`
3. **集成到CLI**：更新`cli.py`使用新抽象层

## 文档

- [核心抽象层使用指南](docs/CORE_ABSTRACTION.md)
- [实现总结](SUMMARY.md)
- [架构设计](CORE_IMPLEMENTATION.md)
