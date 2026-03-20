# Example: 完整使用示例

这个目录包含一套完整的配置模版，帮你从零跑起 evoskill。

## 文件说明

| 文件 | 说明 |
|------|------|
| `SKILL.md` | 示例 Skill 文件，遵循 [Agent Skills 标准](https://agentskills.io/specification) |
| `config.yaml` | **所有参数**的完整框架配置模版，每个字段都有中文注释 |

## Skill 文件格式 (Agent Skills 标准)

```markdown
---
name: my-writing-assistant          # 必填，kebab-case，≤64字符
description: 专业中文写作助手...     # 必填，≤1024字符
metadata:
  version: "1.0"                    # evoskill 扩展：版本号
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

# 2. 用 config 启动（skill 参数传目录路径）
python -m evoskill.main --config my-config.yaml --skill demo/example

# 3. 在聊天中体验完整流程
#    发消息 → /bad 反馈 → /target 设方向 → /optimize 优化
```

## 参数说明

### config.yaml 主要分区

```yaml
llm:           # LLM 连接（API key、模型、地址）
storage:       # 存储路径（traces、skills）
apo:           # APO 优化参数（步数、样本数）
reward:        # 自动 Judge 设置（模型、rubric、开关）
verbose:       # 调试日志开关
```

### SKILL.md 字段映射

| SKILL.md 位置 | evoskill 字段 | 说明 |
|---|---|---|
| frontmatter `name` | `Skill.name` | 技能名 (kebab-case) |
| frontmatter `description` | `Skill.description` | 技能描述 |
| frontmatter `metadata.version` | `Skill.version` | 版本号 |
| frontmatter `metadata.target` | `Skill.target` | 优化方向 |
| Markdown body | `Skill.system_prompt` | 核心 prompt (被优化的"权重") |
| `config.yaml` | `Skill.config` | temperature, judge_rubric 等 |

### 命令行参数

```bash
python -m evoskill.main \
  --config config.yaml \        # 框架配置文件
  --skill ./my-skill \          # Skill 目录（含 SKILL.md）
  --ckpt ckpt/xxx \             # 从 checkpoint 恢复
  --ckpt-dir ./ckpt \           # checkpoint 存储目录
  --optimize \                  # 批量优化模式（不进入聊天）
  -v                            # 调试日志
```

### 聊天命令

```
/bad <原因>         标记上条回复不好
/rewrite <文本>     提供理想回复
/target <方向>      设置优化方向
/optimize           触发 APO 优化 + 自动存 checkpoint
/image <路径>       附加图片
/save               手动保存 skill
/tree               查看技能树
/select <路径>      切换子技能（如 social.moments）
/split              分析并拆分技能
/ckpt               列出 checkpoint
/restore <名称>     恢复 checkpoint
/quit               退出
```
