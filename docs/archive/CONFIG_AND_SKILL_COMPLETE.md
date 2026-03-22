# Config 和 Skill 存储功能完成总结

> 更新时间: 2026-03-17
> 版本: v0.2.0

---

## 完成内容

### 1. Config 配置系统 ✅

框架已有完整的配置系统，支持三种配置方式：

#### 优先级（从高到低）

```
环境变量 (TREE_*) > .env 文件 > YAML 配置文件 > Pydantic 默认值
```

#### 配置文件结构

**config.yaml**:
```yaml
llm:
  api_key: "your-api-key"
  base_url: "https://api.siliconflow.cn/v1"
  model: "Qwen/Qwen2.5-14B-Instruct"
  judge_model: "Qwen/Qwen2.5-14B-Instruct"
  temperature: 0.7

storage:
  trace_path: "./data/traces.jsonl"
  skill_path: "./skills"

apo:
  max_steps: 3
  gradient_accumulation_steps: 5

reward:
  enabled: false
  auto_judge: false
  model: null
  base_url: null
  api_key: null
  default_rubric: |

verbose: false
```

#### 在代码中使用

```python
from treeskill import GlobalConfig

# 方式 1: 从 YAML 加载
config = GlobalConfig.from_yaml("config.yaml")

# 方式 2: 自动从 .env 和环境变量加载
config = GlobalConfig()

# 访问配置
print(config.llm.model)  # "Qwen/Qwen2.5-14B-Instruct"
print(config.llm.api_key.get_secret_value())  # "your-api-key"
```

---

### 2. Skill 存储和加载 ✅

#### Skill 文件格式

**skill.yaml**:
```yaml
name: my-writing-assistant
version: v1.0
system_prompt: |
  你是一位专业的中文写作助手。
  帮助用户撰写、润色各种类型的文本。
  语言要自然流畅，避免 AI 腔。

target: "更像真人说话，少套话，有温度"
few_shot_messages: []
config:
  temperature: 0.7
```

#### 使用方法

```python
from treeskill import load_skill, save_skill, compile_messages

# 加载 skill
skill = load_skill("skills/my-skill.yaml")

# 修改并保存
skill.version = "v1.1"
skill.system_prompt = skill.system_prompt + "\n\n新增规则：避免过于正式的客套话。"
save_skill(skill, "skills/my-skill-v1.1.yaml")

# 编译消息
user_input = [{"role": "user", "content": "帮我写一段关于春天的短文"}]
messages = compile_messages(skill, user_input)
```

---

### 3. 完整示例代码 ✅

**文件**: `examples/example_load_skill_and_config.py`

**运行**:
```bash
python examples/example_load_skill_and_config.py
```

**示例内容**:
1. **示例 1**: 加载配置和 Skill
2. **示例 2**: 使用适配器调用 LLM
3. **示例 3**: 保存 Skill
4. **示例 4**: 手动反馈 + 计算梯度
5. **示例 5**: 加载并继续优化

**输出示例**:
```
============================================================
TreeSkill - 加载已存储 Skill 并使用 Config 示例
============================================================

✓ 从 demo/example/config.yaml 加载配置
  - LLM 模型: Qwen/Qwen2.5-14B-Instruct
  - API 地址: https://api.siliconflow.cn/v1

✓ 从 demo/example/skill.yaml 加载 Skill
  - 名称: my-writing-assistant
  - 版本: v1.0

✓ Skill 已保存到 skills/my-skill-v1.1.yaml

✓ 梯度计算完成:
  - 梯度内容: Mock API response

✓ 新 Prompt:
  你是一个专业且友好的助手...

所有示例完成！
```

---

### 4. 完整使用指南 ✅

**文件**: `USAGE_GUIDE.md`

**内容**:
- 配置系统详解（3 种方式）
- Skill 存储和加载详解
- 4 个完整代码示例
- 常见问题解答（7 个 Q&A）

---

### 5. 代码改进 ✅

#### 添加 MockAdapter 导出

**文件**: `treeskill/__init__.py`

```python
elif name == "MockAdapter":
    try:
        from examples.mock_adapter import MockAdapter as _MockAdapter
        return _MockAdapter
    except ImportError:
        raise ImportError(
            f"\n\n❌ MockAdapter 导入失败\n\n"
            f"MockAdapter 在 examples/mock_adapter.py 中\n"
        ) from None
```

#### 添加 Skill 管理函数导出

**文件**: `treeskill/__init__.py`

```python
from treeskill.skill import (
    load as load_skill,
    save as save_skill,
    compile_messages,
)

__all__ = [
    ...
    "load_skill",
    "save_skill",
    "compile_messages",
    ...
]
```

---

## 更新后的文件

### 新增文件

1. **`examples/example_load_skill_and_config.py`** - 完整示例代码（295 行）
2. **`USAGE_GUIDE.md`** - 完整使用指南（~400 行）

### 修改文件

1. **`treeskill/__init__.py`**
   - 添加 `MockAdapter` 延迟导入
   - 添加 `load_skill`, `save_skill`, `compile_messages` 导出
   - 更新 `__all__` 列表

2. **`README.md`**
   - 在"完整示例"部分添加"示例 4: 加载已存储的 Skill 和配置"
   - 添加 `USAGE_GUIDE.md` 引用

---

## 使用流程

### 标准流程

```bash
# 1. 配置 API
export TREE_LLM_API_KEY="your-key"
export TREE_LLM_BASE_URL="https://api.siliconflow.cn/v1"
export TREE_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"

# 2. 创建 config.yaml（可选）
cp demo/example/config.yaml my-config.yaml
# 编辑 my-config.yaml

# 3. 创建 skill
cp demo/example/skill.yaml skills/my-skill.yaml
# 编辑 skills/my-skill.yaml

# 4. 加载并使用
python examples/example_load_skill_and_config.py
```

### 在代码中使用

```python
#!/usr/bin/env python3
from treeskill import (
    GlobalConfig,
    load_skill,
    save_skill,
    compile_messages,
    OpenAIAdapter,
)

# 加载配置和 skill
config = GlobalConfig.from_yaml("config.yaml")
skill = load_skill("skills/my-skill.yaml")

# 创建适配器
adapter = OpenAIAdapter(
    api_key=config.llm.api_key.get_secret_value(),
    base_url=config.llm.base_url,
    model=config.llm.model,
)

# 调用 LLM
user_input = [{"role": "user", "content": "帮我写一段关于春天的短文"}]
messages = compile_messages(skill, user_input)
response = adapter.generate(messages)

# 保存
skill.version = "v1.1"
save_skill(skill, "skills/my-skill-v1.1.yaml")
```

---

## 常见问题

### Q1: 如何设置 API Key？

**推荐**: 环境变量
```bash
export TREE_LLM_API_KEY="your-api-key"
```

### Q2: 如何切换不同的 LLM？

修改配置即可：
```yaml
llm:
  base_url: "https://api.anthropic.com"
  model: "claude-3-5-sonnet-20241022"
```

### Q3: 如何查看已存储的 Skill？

```bash
ls skills/
```

### Q4: 如何恢复旧版本的 Skill？

```python
skill_v1 = load_skill("skills/my-skill-v1.0.yaml")
```

---

## 验证结果

### 测试通过 ✅

```bash
$ python examples/example_load_skill_and_config.py

✓ 从 demo/example/config.yaml 加载配置
✓ 从 demo/example/skill.yaml 加载 Skill
✓ Skill 已保存到 skills/my-skill-v1.1.yaml
✓ 梯度计算完成
✓ 新 Prompt 生成成功
✓ 加载并继续优化成功

所有示例完成！
```

### 文档完整 ✅

- `README.md` - 更新了完整示例部分
- `USAGE_GUIDE.md` - 新增完整使用指南
- `examples/example_load_skill_and_config.py` - 可运行的完整示例

---

## 相关文档

- `/README.md` - 项目概览（已更新）
- `/docs/USAGE_GUIDE.md` - 完整使用指南（新增）
- `/docs/OPTIMIZER_COMPLETE.md` - 优化器详细文档
- `/docs/TOOLS_COMPLETE.md` - 工具系统文档
- `/demo/example/config.yaml` - 配置文件模版
- `/demo/example/skill.yaml` - Skill 文件模版

---

## 总结

**完成目标**：
1. ✅ 揭示了框架已有完整的 Config 机制
2. ✅ 揭示了框架已有完整的 Skill 存储和加载机制
3. ✅ 创建了完整的使用指南（`USAGE_GUIDE.md`）
4. ✅ 创建了可运行的完整示例（`examples/example_load_skill_and_config.py`）
5. ✅ 更新了 README 添加新示例
6. ✅ 修复了 MockAdapter 导入问题
7. ✅ 添加了 Skill 管理函数的导出

**用户现在可以**：
- 使用 3 种方式配置框架（环境变量、.env、YAML）
- 轻松加载和保存 Skill
- 使用完整的示例代码作为起点
- 查看详细的使用指南解决常见问题

**框架的 Config 和 Skill 管理功能已经完全就绪！** 🎉
