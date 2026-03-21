# 核心抽象层实现完成 ✅

## 已实现的模块

### 1. `core/abc.py` - 抽象基类定义
- ✅ `OptimizablePrompt` - 可优化Prompt基类
- ✅ `TextualGradient` - 文本梯度基类
- ✅ `Experience` - 交互经验基类
- ✅ `Feedback` - 反馈基类
- ✅ `ModelAdapter` - 模型适配器基类
- ✅ `PromptSerializer` - 序列化协议

### 2. `core/prompts.py` - Prompt类型实现
- ✅ `TextPrompt` - 文本Prompt
- ✅ `MultimodalPrompt` - 多模态Prompt（文本+图像+音频）
- ✅ `StructuredPrompt` - 结构化Prompt（JSON Schema）
- ✅ 版本管理（bump_version）
- ✅ 序列化/反序列化

### 3. `core/gradient.py` - 梯度类型实现
- ✅ `SimpleGradient` - 简单文本梯度
- ✅ `MultimodalGradient` - 多模态梯度
- ✅ `GradientHistory` - 梯度历史（支持动量）

### 4. `core/experience.py` - 经验类型实现
- ✅ `ConversationExperience` - 对话经验
- ✅ `MultimodalExperience` - 多模态经验
- ✅ `CompositeFeedback` - 复合反馈（支持score/critique/correction）
- ✅ `FeedbackType` - 反馈类型枚举

### 5. `core/base_adapter.py` - 模型适配器基类
- ✅ `BaseModelAdapter` - 提供通用实现
- ✅ Prompt验证逻辑
- ✅ 梯度计算框架
- ✅ 梯度应用框架

---

## 测试结果

```bash
$ python tests/test_core_abstractions.py

============================================================
Core Abstraction Layer Tests
============================================================
Testing TextPrompt... ✓
Testing MultimodalPrompt... ✓
Testing SimpleGradient... ✓
Testing CompositeFeedback... ✓
Testing ConversationExperience... ✓

All tests passed! ✓
============================================================
```

---

## 架构特点

### 1. **模型无关**
```python
# 同样的代码可以用于不同模型
openai_adapter = OpenAIAdapter(model="gpt-4o")
claude_adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")

# 使用相同的接口
gradient1 = openai_adapter.compute_gradient(prompt, failures)
gradient2 = claude_adapter.compute_gradient(prompt, failures)
```

### 2. **多模态支持**
```python
# 文本Prompt
text_prompt = TextPrompt(content="你是助手")

# 多模态Prompt
mm_prompt = MultimodalPrompt(
    text="分析图片",
    images=["image.jpg"]
)

# 自动验证
adapter.validate_prompt(mm_prompt)  # 检查模型是否支持vision
```

### 3. **Train-Free优化**
```python
# 计算文本梯度
gradient = adapter.compute_gradient(
    prompt,
    failures,
    target="更自然"
)

# 应用梯度更新
new_prompt = adapter.apply_gradient(prompt, gradient)
```

---

## 使用示例

### 示例1：基本使用

```python
from evo_framework.core import (
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
)

# 创建Prompt
prompt = TextPrompt(
    content="你是AI助手",
    name="assistant",
    target="更像人类"
)

# 收集经验
exp = ConversationExperience(
    messages=[{"role": "user", "content": "你好"}],
    response="你好，我是AI。",
    feedback=CompositeFeedback(critique="太机械"),
)

# 优化（需要adapter）
if exp.is_failure:
    gradient = adapter.compute_gradient(prompt, [exp])
    optimized = adapter.apply_gradient(prompt, gradient)
```

### 示例2：多模态

```python
from evo_framework.core import MultimodalPrompt, MultimodalExperience

# Vision任务
prompt = MultimodalPrompt(
    text="识别图片中的物体",
    images=["photo.jpg"],
)

# 多模态经验
exp = MultimodalExperience(
    input_text="这是什么？",
    input_images=["object.jpg"],
    output_text="这是一个杯子",
    feedback=CompositeFeedback(score=0.8),
)
```

---

## 下一步计划

### Phase 1: 实现具体适配器
- [ ] `adapters/openai.py` - OpenAI API适配器
- [ ] `adapters/anthropic.py` - Claude API适配器


### Phase 2: 优化引擎
- [ ] `optimizer/engine.py` - TGD优化引擎
- [ ] `optimizer/strategies.py` - 优化策略（保守/激进）
- [ ] `optimizer/validator.py` - 自动验证器

### Phase 3: 存储层
- [ ] `storage/experience_store.py` - 经验存储（改进现有storage.py）
- [ ] `storage/prompt_registry.py` - Prompt版本管理
- [ ] `storage/checkpoint.py` - Checkpoint（改进现有checkpoint.py）

### Phase 4: 集成
- [ ] 更新`cli.py`使用新抽象层
- [ ] 更新`main.py`支持多模型
- [ ] 迁移现有Schema（兼容旧格式）

---

## 文档

- [核心抽象层使用指南](docs/CORE_ABSTRACTION.md)
- [TGD论文](https://arxiv.org/pdf/2502.16923)

---

## 设计亮点

1. **类型安全**：完整的类型提示，IDE友好
2. **可序列化**：所有对象都支持JSON/YAML存储
3. **不可变语义**：使用`attach_feedback()`等返回新对象
4. **可扩展**：易于添加新的Prompt类型、Feedback类型
5. **验证机制**：自动检查Prompt与模型的兼容性

---

## 与现有代码的关系

### 兼容性

新的核心抽象层**不破坏**现有代码：

```python
# 旧代码（继续工作）
from evo_framework.schema import Skill, Trace
skill = Skill(name="test", system_prompt="...")

# 新代码（推荐）
from evo_framework.core import TextPrompt
prompt = TextPrompt(content="...", name="test")
```

### 迁移路径

1. **Phase 1**: 新代码使用新抽象层，旧代码保持不变
2. **Phase 2**: 提供`Skill` → `TextPrompt`转换器
3. **Phase 3**: 逐步迁移核心模块
4. **Phase 4**: 废弃旧的Schema（v2.0）

---

## 贡献指南

### 添加新适配器

```python
from evo_framework.core import BaseModelAdapter

class MyAdapter(BaseModelAdapter):
    def generate(self, prompt, context=None, temperature=0.7, **kwargs):
        # 1. 转换prompt为模型输入
        messages = self._build_messages(prompt, context)

        # 2. 调用API
        response = self._call_api(messages, temperature=temperature)

        # 3. 返回结果
        return response
```

### 添加新Prompt类型

```python
from evo_framework.core.abc import OptimizablePrompt

class MyPrompt(OptimizablePrompt):
    def to_model_input(self):
        return {"custom": self.data}

    def serialize(self):
        return {"data": self.data}

    @classmethod
    def deserialize(cls, data):
        return cls(data["data"])
```

---

## License

MIT
