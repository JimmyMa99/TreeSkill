# Core Abstraction Layer 使用指南

## 概述

核心抽象层（`treeskill.core`）提供了让框架**模型无关**、**多模态支持**的基础接口。

### 设计原则

1. **抽象优于实现**：所有核心概念都有抽象基类（ABC）
2. **组合优于继承**：通过Protocol和ABC定义接口
3. **序列化友好**：所有对象都可以序列化/反序列化
4. **类型安全**：完整的类型提示

---

## 核心概念

### 1. OptimizablePrompt

所有可优化的Prompt的基类。

```python
from treeskill.core import TextPrompt

# 创建文本Prompt
prompt = TextPrompt(
    content="你是一个写作助手。",
    name="writing-assistant",
    version="v1.0",
    target="更像人类"
)

# 序列化
data = prompt.serialize()

# 反序列化
prompt2 = TextPrompt.deserialize(data)

# 版本升级
new_prompt = prompt.bump_version()  # v1.0 -> v1.1
```

### 2. TextualGradient

文本梯度，描述如何改进Prompt。

```python
from treeskill.core import SimpleGradient

gradient = SimpleGradient(
    text="Prompt太正式了，应该更口语化。",
    metadata={"source": "user_feedback", "sample_count": 5}
)

print(str(gradient))  # Prompt太正式了，应该更口语化。
```

### 3. Experience

单次交互经验。

```python
from treeskill.core import (
    ConversationExperience,
    CompositeFeedback,
)

# 创建经验
exp = ConversationExperience(
    messages=[{"role": "user", "content": "写一首诗"}],
    response="春眠不觉晓...",
)

# 添加反馈
feedback = CompositeFeedback(
    score=0.3,
    critique="太老套了"
)
exp_with_feedback = exp.attach_feedback(feedback)

# 检查是否失败
if exp_with_feedback.is_failure:
    print("这条经验需要优化")
```

### 4. ModelAdapter

模型适配器，核心抽象。

```python
from treeskill.core import BaseModelAdapter

class MyModelAdapter(BaseModelAdapter):
    @property
    def model_name(self) -> str:
        return "my-model"

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def max_context_tokens(self) -> int:
        return 128000

    def generate(self, prompt, context=None, temperature=0.7, **kwargs):
        # 实现生成逻辑
        messages = self._build_messages(prompt, context)
        return self._call_api(messages, temperature=temperature)

    def _call_api(self, messages, system=None, temperature=0.7, **kwargs):
        # 调用你的API
        pass

    def _count_tokens_impl(self, text: str) -> int:
        # 实现token计数
        return len(text.split())  # 简化版本
```

---

## 完整工作流示例

```python
from treeskill.core import (
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    SimpleGradient,
)

# 1. 创建初始Prompt
prompt = TextPrompt(
    content="你是一个AI助手。",
    name="assistant",
    version="v1.0",
)

# 2. 收集经验
experiences = [
    ConversationExperience(
        messages=[{"role": "user", "content": "你好"}],
        response="你好，我是AI助手。",
        feedback=CompositeFeedback(score=0.3, critique="太机械化"),
    ),
    # ... 更多经验
]

# 3. 计算梯度（需要ModelAdapter）
adapter = MyModelAdapter(api_key="...")
failures = [exp for exp in experiences if exp.is_failure]
gradient = adapter.compute_gradient(prompt, failures, target="更自然")

# 4. 应用梯度
new_prompt = adapter.apply_gradient(prompt, gradient)

print(f"旧版本: {prompt.version}")  # v1.0
print(f"新版本: {new_prompt.version}")  # v1.1
print(f"新内容: {new_prompt.content}")
```

---

## 多模态支持

```python
from treeskill.core import MultimodalPrompt, MultimodalExperience

# 创建多模态Prompt
prompt = MultimodalPrompt(
    text="分析这张图片",
    images=["/path/to/image.jpg"],
    name="image-analyzer",
)

# 验证兼容性
adapter = MyModelAdapter()
issues = adapter.validate_prompt(prompt)
if issues:
    print(f"兼容性问题: {issues}")
```

---

## 序列化和存储

所有对象都支持序列化：

```python
import json
from pathlib import Path

# 序列化Prompt
data = prompt.serialize()
Path("prompt.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))

# 反序列化
data = json.loads(Path("prompt.json").read_text())
prompt = TextPrompt.deserialize(data)

# 序列化Experience
exp_data = experience.to_training_sample()
Path("experience.jsonl").write_text(json.dumps(exp_data) + "\n")
```

---

## 扩展指南

### 添加新的Prompt类型

```python
from treeskill.core.abc import OptimizablePrompt

class MyCustomPrompt(OptimizablePrompt):
    def __init__(self, custom_field: str, **kwargs):
        self.custom_field = custom_field
        # ...

    def to_model_input(self):
        return {"custom": self.custom_field}

    def apply_gradient(self, gradient):
        # 通常委托给ModelAdapter
        return self

    def serialize(self):
        return {"custom_field": self.custom_field, ...}

    @classmethod
    def deserialize(cls, data):
        return cls(custom_field=data["custom_field"], ...)

    @property
    def version(self):
        return self._version

    def bump_version(self):
        # 返回新版本
        pass
```

### 添加新的Feedback类型

```python
from treeskill.core.abc import Feedback

class MyCustomFeedback(Feedback):
    def __init__(self, custom_metric: float):
        self.custom_metric = custom_metric

    def to_score(self) -> float:
        # 转换为0-1分数
        return self.custom_metric / 100.0

    def to_dict(self):
        return {"custom_metric": self.custom_metric}

    @property
    def is_negative(self) -> bool:
        return self.to_score() < 0.5
```

---

## 设计决策

### 为什么用ABC而不是Protocol？

- 需要共享实现逻辑（如`BaseModelAdapter`）
- 更清晰的继承关系
- 更好的IDE支持

### 为什么Feedback有to_score()？

- 统一不同类型的反馈（评分、文字、示例）
- 支持自动筛选失败案例
- 兼容TGD论文的数值梯度概念

### 为什么Experience是可变的？

- 先生成输出，后添加反馈（符合实际使用流程）
- 使用`attach_feedback()`返回新对象，保持不可变语义

---

## 常见问题

**Q: 如何处理大型图像/音频？**
A: 存储为文件引用（路径）而非base64嵌入，仅在需要时加载。

**Q: 如何支持流式输出？**
A: 在`ModelAdapter.generate()`中返回生成器，上层处理流式响应。

**Q: 如何做Prompt的diff？**
A: 比较两个Prompt的序列化结果，或使用专门的diff工具。

**Q: 版本号规则是什么？**
A: 默认`vX.Y`格式，Y递增。可以在子类中实现语义化版本控制。

---

## 下一步

1. 实现`OpenAIAdapter`（见`adapters/openai.py`）
2. 实现`AnthropicAdapter`（见`adapters/anthropic.py`）
3. 编写优化引擎（`optimizer/engine.py`）
4. 集成到现有CLI（`cli.py`）

---

## 参考

- [Textual Gradient Descent 论文](https://arxiv.org/pdf/2502.16923)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [OpenAI API 文档](https://platform.openai.com/docs/)
