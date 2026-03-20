# Skill 文件格式 (Agent Skills 标准)

evoskill 的 Skill 文件遵循 [Agent Skills 开放标准](https://agentskills.io/specification)，与 Claude Code、Cursor、Gemini CLI 等 30+ 工具互通。

## 目录结构

一个 Skill 是一个包含 `SKILL.md` 的目录：

```
my-skill/
├── SKILL.md          # 必须：YAML frontmatter + Markdown body
├── config.yaml       # 可选：few-shot 示例 + 模型参数
└── docs/             # 可选：参考文档（渐进式加载）
```

---

## SKILL.md 格式

```markdown
---
name: my-writing-assistant
description: 专业中文写作助手，帮助撰写和润色各类文本。当用户需要写作帮助时触发。
metadata:
  version: "1.0"
  target: "更像真人说话，少套话，有温度"
---

你是一位专业的中文写作助手。
帮助用户撰写、润色各种类型的文本。
语言要自然流畅，避免 AI 腔。
```

### Frontmatter 字段

| 字段 | 必填 | 约束 | evoskill 映射 |
|------|------|------|---------------|
| `name` | 是 | kebab-case，≤64字符，需与目录名一致 | `Skill.name` |
| `description` | 是 | ≤1024字符，描述用途和触发条件 | `Skill.description` |
| `metadata` | 否 | 任意 key-value | — |
| `metadata.version` | — | 语义版本号 | `Skill.version` |
| `metadata.target` | — | 一句话优化方向 | `Skill.target` |
| `license` | 否 | 许可证 | — |
| `compatibility` | 否 | ≤500字符，环境要求 | — |

### Markdown Body

Frontmatter 之后的正文就是 **System Prompt**——也是 APO 优化的目标"权重"。

---

## config.yaml（可选）

存放 evoskill 专属的运行时配置：

```yaml
# 模型参数
temperature: 0.7

# Judge 评分标准（覆盖全局 rubric）
judge_rubric: |
  评分标准：
  - 自然度 (40%)
  - 准确性 (30%)
  - 实用性 (30%)

# Few-shot 示例
few_shot_messages:
  - role: user
    content: "什么是 Python 的 GIL？"
  - role: assistant
    content: "GIL 是全局解释器锁，限制了多线程并行执行 Python 字节码。"
```

---

## Skill 树

多个 Skill 通过目录嵌套形成树形层级：

```
writing-skills/
├── SKILL.md              # 根：通用写作
├── social/
│   ├── SKILL.md          # 社交媒体写作
│   └── moments/
│       └── SKILL.md      # 朋友圈专精
└── business/
    ├── SKILL.md          # 商务写作
    └── email/
        └── SKILL.md      # 邮件专精
```

用 `SkillTree.load("writing-skills/")` 加载整棵树。

---

## 渐进式披露

Agent Skills 标准的三层加载策略：

| 层 | 加载时机 | 内容 | Token 开销 |
|----|---------|------|-----------|
| 1. 目录 | 会话启动 | name + description | ~50-100 / skill |
| 2. 指令 | 激活 skill | 完整 SKILL.md body | < 5000（推荐） |
| 3. 资源 | 按需引用 | docs/、scripts/ | 视文件大小 |

**最佳实践**：SKILL.md 正文控制在 500 行以内，详细参考资料放 `docs/` 子目录。
