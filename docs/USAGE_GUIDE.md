# TreeSkill 使用指南

> 如何加载已存储的 Skill 并使用 Config
> 版本: v0.2.0
> 更新时间: 2026-03-17

---

## 目录

1. [配置系统](#配置系统)
2. [Skill 存储和加载](#skill-存储和加载)
3. [完整示例](#完整示例)
4. [常见问题](#常见问题)

---

## 配置系统

TreeSkill 有完整的配置系统，支持三种配置方式，优先级从高到低：

### 优先级

```
环境变量 (TREE_*) > .env 文件 > YAML 配置文件 > Pydantic 默认值
```

### 方式 1: YAML 配置文件（推荐）

创建 `config.yaml`:

```yaml
# LLM 连接配置
llm:
  api_key: "your-api-key-here"
  base_url: "https://api.siliconflow.cn/v1"
  model: "Qwen/Qwen2.5-14B-Instruct"
  judge_model: "Qwen/Qwen2.5-14B-Instruct"
  temperature: 0.7

# 存储配置
storage:
  trace_path: "./data/traces.jsonl"
  skill_path: "./skills"

# APO 优化参数
apo:
  max_steps: 3
  gradient_accumulation_steps: 5

# Reward / 自动 Judge 配置
reward:
  enabled: false
  auto_judge: false
  model: null
  base_url: null
  api_key: null
  default_rubric: |
    你是一个写作质量评审专家...

# 全局选项
verbose: false
```

`trace_path` 指向的 JSONL 文件中，每行都是一条 `Trace`。`trace.id` 仍然标识单条交互记录，`session_id` 用来把同一次运行/会话里的多条 Trace 关联起来；旧文件如果没有 `session_id` 也能正常读取。

使用配置文件：

```bash
python your_script.py --config config.yaml
```

### 方式 2: .env 文件

在项目根目录创建 `.env`:

```env
TREE_LLM_API_KEY=your-api-key
TREE_LLM_BASE_URL=https://api.siliconflow.cn/v1
TREE_LLM_MODEL=Qwen/Qwen2.5-14B-Instruct
TREE_LLM_JUDGE_MODEL=Qwen/Qwen2.5-14B-Instruct
TREE_LLM_TEMPERATURE=0.7

TREE_STORAGE_TRACE_PATH=./data/traces.jsonl
TREE_STORAGE_SKILL_PATH=./skills

TREE_APO_MAX_STEPS=3
TREE_APO_GRADIENT_ACCUMULATION_STEPS=5

TREE_REWARD_ENABLED=false
TREE_REWARD_AUTO_JUDGE=false
```

`.env` 文件会自动加载，无需额外配置。

### 方式 3: 环境变量（最高优先级）

```bash
export TREE_LLM_API_KEY="your-api-key"
export TREE_LLM_BASE_URL="https://api.siliconflow.cn/v1"
export TREE_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"

python your_script.py
```

### 在代码中加载配置

```python
from treeskill import GlobalConfig

# 方式 1: 从 YAML 加载
config = GlobalConfig.from_yaml("config.yaml")

# 方式 2: 自动从 .env 和环境变量加载
config = GlobalConfig()

# 访问配置
print(config.llm.model)  # "Qwen/Qwen2.5-14B-Instruct"
print(config.llm.api_key.get_secret_value())  # "your-api-key"
print(config.storage.trace_path)  # Path("./data/traces.jsonl")
```

---

## Skill 存储和加载

### Skill 文件格式

Skill 以 YAML 格式存储：

```yaml
# Skill 名称（用于标识和文件命名）
name: my-writing-assistant

# 当前版本（每次 /optimize 后自动递增）
version: v1.0

# 核心：System Prompt（框架优化的"权重"）
system_prompt: |
  你是一位专业的中文写作助手。
  帮助用户撰写、润色各种类型的文本。
  语言要自然流畅，避免 AI 腔。

# 优化方向（可选）
target: "更像真人说话，少套话，有温度"

# Few-shot 示例（可选）
few_shot_messages: []

# Skill 级配置（可选）
config:
  # 覆盖全局 temperature
  temperature: 0.7

  # Judge 评分标准（可选）
  judge_rubric: |
    评分标准：
    - 是否像真人写的 (40%)
    - 语言是否流畅 (30%)
    - 是否切合场景 (30%)
```

### 加载 Skill

```python
from treeskill import load as load_skill

# 加载 skill
skill = load_skill("skills/my-skill.yaml")

print(skill.name)            # "my-writing-assistant"
print(skill.version)         # "v1.0"
print(skill.system_prompt)   # "你是一位专业的中文写作助手..."
print(skill.target)          # "更像真人说话，少套话，有温度"
```

### 保存 Skill

```python
from treeskill import save as save_skill

# 修改 skill
skill.version = "v1.1"
skill.system_prompt = skill.system_prompt + "\n\n新增规则：避免过于正式的客套话。"

# 保存
save_skill(skill, "skills/my-skill-v1.1.yaml")
```

### 编译消息

```python
from treeskill import compile_messages

# 准备用户输入
user_input = [{"role": "user", "content": "帮我写一段关于春天的短文"}]

# 编译消息（system prompt + few-shot + user input）
messages = compile_messages(skill, user_input)

# messages 结构:
# [
#   {"role": "system", "content": "你是一位专业的中文写作助手..."},
#   {"role": "user", "content": "帮我写一段关于春天的短文"}
# ]
```

---

## 完整示例

### 示例 1: 基础使用 - 加载配置和 Skill

```python
#!/usr/bin/env python3
from pathlib import Path
from treeskill import GlobalConfig, load as load_skill

# 1. 加载配置
config = GlobalConfig.from_yaml("config.yaml")

print(f"LLM 模型: {config.llm.model}")
print(f"API 地址: {config.llm.base_url}")

# 2. 加载 Skill
skill = load_skill("skills/my-skill.yaml")

print(f"Skill 名称: {skill.name}")
print(f"版本: {skill.version}")
```

### 示例 2: 使用适配器调用 LLM

```python
#!/usr/bin/env python3
from treeskill import (
    GlobalConfig,
    load as load_skill,
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

# 准备输入
user_input = [{"role": "user", "content": "帮我写一段关于春天的短文"}]
messages = compile_messages(skill, user_input)

# 调用 LLM
response = adapter.generate(messages, temperature=config.llm.temperature)

print(f"助手回复:\n{response.content}\n")

# 保存（可选）
skill.version = "v1.1"
save_skill(skill, "skills/my-skill-v1.1.yaml")
```

### 示例 3: 优化 Prompt（使用 MockAdapter）

```python
#!/usr/bin/env python3
from treeskill import (
    TextPrompt,
    ConversationExperience,
    CompositeFeedback,
    FeedbackType,
    MockAdapter,
)

# 创建初始 Prompt
prompt = TextPrompt(
    content="你是一个写作助手，帮助用户撰写各种文本。"
)

# 创建失败经验（模拟 /bad 和 /rewrite）
experience = ConversationExperience(
    conversation=[
        {"role": "user", "content": "写一段关于春天的短文"},
        {"role": "assistant", "content": "春天来了，万物复苏..."},
    ],
    feedback=CompositeFeedback(
        feedback_type=FeedbackType.CORRECTION,
        critique="回答太简单，缺乏细节和感情",
        correction="春天来了，小区的玉兰花开了，空气中弥漫着淡淡的花香...",
    ),
)

# 使用 Mock 适配器计算梯度
adapter = MockAdapter()

# 计算梯度
gradient = adapter.compute_gradient(
    prompt=prompt,
    experiences=[experience],
)

print(f"梯度分析:\n{gradient.content}\n")

# 应用梯度更新 Prompt
new_prompt = adapter.apply_gradient(
    prompt=prompt,
    gradient=gradient,
)

print(f"优化后的 Prompt:\n{new_prompt.content}\n")
```

### 示例 4: 完整工作流程

```bash
# 运行完整示例
python examples/example_load_skill_and_config.py
```

完整示例包含：
1. 加载配置和 Skill
2. 使用适配器调用 LLM
3. 保存 Skill
4. 手动反馈 + 计算梯度
5. 加载并继续优化

---

## 常见问题

### Q1: 如何设置 API Key？

**方式 A**: 环境变量（推荐）

```bash
export TREE_LLM_API_KEY="your-api-key"
```

**方式 B**: config.yaml

```yaml
llm:
  api_key: "your-api-key"
```

**方式 C**: .env 文件

```env
TREE_LLM_API_KEY=your-api-key
```

### Q2: 如何切换不同的 LLM？

修改配置即可：

```yaml
llm:
  base_url: "https://api.anthropic.com"  # Anthropic
  model: "claude-3-5-sonnet-20241022"
```

或者使用不同的适配器：

```python
from treeskill import AnthropicAdapter

adapter = AnthropicAdapter(
    api_key=config.llm.api_key.get_secret_value(),
    model="claude-3-5-sonnet-20241022",
)
```

### Q3: 如何查看已存储的 Skill？

```bash
# 列出所有 skill
ls skills/

# 或在代码中
from pathlib import Path
skill_files = list(Path("skills").glob("*.yaml"))
print(skill_files)
```

### Q4: 如何恢复旧版本的 Skill？

每个优化版本都会保存，直接加载即可：

```python
# 加载旧版本
skill_v1 = load_skill("skills/my-skill-v1.0.yaml")
skill_v1_1 = load_skill("skills/my-skill-v1.1.yaml")
```

### Q5: 配置优先级是什么？

```
环境变量 (TREE_*) > .env 文件 > YAML 配置文件 > Pydantic 默认值
```

示例：
- `.env`: `TREE_LLM_MODEL=gpt-3.5`
- `config.yaml`: `llm.model: gpt-4o`
- 环境变量: `export TREE_LLM_MODEL=claude-3`

**最终使用**: `claude-3`（环境变量优先级最高）

### Q6: 如何使用不同的 Judge 模型？

```yaml
llm:
  model: "Qwen/Qwen2.5-14B-Instruct"  # 聊天模型
  judge_model: "Qwen/Qwen2.5-72B-Instruct"  # Judge 模型（能力更强）
```

或单独配置：

```yaml
reward:
  model: "Qwen/Qwen2.5-72B-Instruct"
  base_url: "https://api.another-provider.com/v1"
  api_key: "another-api-key"
```

### Q7: Skill 文件可以放在任何位置吗？

可以，但推荐统一放在 `skills/` 目录：

```python
# 方式 1: 指定完整路径
skill = load_skill("/path/to/your/skill.yaml")

# 方式 2: 配置 skill_path
config = GlobalConfig.from_yaml("config.yaml")
# config.storage.skill_path = Path("./skills")
skill = load_skill("skills/my-skill.yaml")  # 会自动在 skill_path 中查找
```

---

## 相关文档

- `/README.md` - 项目概览
- `/docs/OPTIMIZER_COMPLETE.md` - 优化器详细文档
- `/docs/TOOLS_COMPLETE.md` - 工具系统文档
- `/docs/MIGRATION_GUIDE.md` - 迁移指南
- `/demo/example/config.yaml` - 完整配置模版
- `/demo/example/skill.yaml` - Skill 文件模版

---

## 下一步

1. 查看完整示例: `python examples/example_load_skill_and_config.py`
2. 创建你的第一个 Skill: 复制 `demo/example/skill.yaml` 并修改
3. 配置 API: 复制 `demo/example/config.yaml` 并填入你的 API Key
4. 开始优化: 运行 `examples/example_optimizer.py` 或 `examples/example_fully_automatic.py`

**祝你使用愉快！** 🎉
