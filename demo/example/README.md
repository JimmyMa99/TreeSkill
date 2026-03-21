# Example: complete使用Example

这个目录包含一套complete的Config模版，帮你从零跑起 evoskill。

## file说明

| file | 说明 |
|------|------|
| `SKILL.md` | Example Skill file，遵循 [Agent Skills 标准](https://agentskills.io/specification) |
| `config.yaml` | **所有参数**的complete框架Config模版，每个字段都有中文注释 |

## Skill file格式 (Agent Skills 标准)

```markdown
---
name: my-writing-assistant          # 必填，kebab-case，≤64字符
description: 专业中文写作助手...     # 必填，≤1024字符
metadata:
  version: "1.0"                    # evoskill 扩展：Version号
  target: "更像真人说话"             # evoskill 扩展：优化方向
---

你是一位专业的中文写作助手。
（这里是 system prompt，也是 APO 优化的目标）
```

## 快速开始

```bash
cd /path/to/evo_agent

# 1. 复制 config 并填入你的 API Key
cp demo/example/config.yaml my-config.yaml
# 编辑 my-config.yaml, 把 api_key 改成你的

# 2. 用 config 启动（skill 参数传目录path）
python -m evoskill.main --config my-config.yaml --skill demo/example

# 3. 在聊天中体验completeFlow
#    发消息 → /bad Feedback → /target 设方向 → /optimize 优化
```

## 参数说明

### config.yaml 主要分区

```yaml
llm:           # LLM 连接（API key、模型、地址）
storage:       # 存储path（traces、skills）
apo:           # APO 优化参数（Steps、样本数）
reward:        # 自动 Judge Set（模型、rubric、开关）
verbose:       # 调试日志开关
```

### SKILL.md 字段映射

| SKILL.md 位置 | evoskill 字段 | 说明 |
|---|---|---|
| frontmatter `name` | `Skill.name` | 技能名 (kebab-case) |
| frontmatter `description` | `Skill.description` | 技能描述 |
| frontmatter `metadata.version` | `Skill.version` | Version号 |
| frontmatter `metadata.target` | `Skill.target` | 优化方向 |
| Markdown body | `Skill.system_prompt` | 核心 prompt (被优化的"权重") |
| `config.yaml` | `Skill.config` | temperature, judge_rubric 等 |

### 命令行参数

```bash
python -m evoskill.main \
  --config config.yaml \        # 框架Configfile
  --skill ./my-skill \          # Skill 目录（含 SKILL.md）
  --ckpt ckpt/xxx \             # 从 checkpoint 恢复
  --ckpt-dir ./ckpt \           # checkpoint 存储目录
  --optimize \                  # 批量优化模式（不进入聊天）
  -v                            # 调试日志
```

### 聊天命令

```
/bad <原因>         标记上条回复不好
/rewrite <文本>     提供Ideal reply
/target <方向>      Set优化方向
/optimize           触发 APO 优化 + 自动存 checkpoint
/image <path>       附加图片
/save               手动Save skill
/tree               查看技能树
/select <path>      切换子技能（如 social.moments）
/split              分析并拆分技能
/ckpt               列出 checkpoint
/restore <名称>     恢复 checkpoint
/quit               退出
```
