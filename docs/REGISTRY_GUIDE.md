# 插件注册机制

TreeSkill 提供了一套装饰器驱动的插件注册系统，让用户可以自定义优化流程中的关键组件。

## 四种注册器

| 注册器 | 装饰器 | 用途 | 内置实现 |
|--------|--------|------|---------|
| **Scorer** | `@scorer("name")` | 评分函数：评估候选 prompt 的好坏 | `exact-match`, `judge-grade` |
| **Gradient** | `@gradient("name")` | 梯度模板：分析失败原因的 system prompt | `simple`, `root-cause`, `comprehensive` |
| **Rewriter** | `@rewriter("name")` | 重写策略：修改 prompt 的 system prompt | `full-rewrite`, `conservative`, `distill` |
| **Skill Format** | `@skill_format("name")` | 格式适配：导入/导出不同 Skill 格式 | 待扩展 |

## Scorer — 评分函数

Scorer 决定"怎么评估一个 prompt 的好坏"。APO beam search 用它选择最佳候选，评估阶段用它计算最终分数。

### 签名

```python
def scorer_fn(output: str, expected: str, context: dict) -> float:
    """
    Args:
        output: 模型生成的内容
        expected: 期望的正确答案 / gold standard
        context: 额外上下文（judge client、task metadata 等）
    Returns:
        0.0 - 1.0 的分数
    """
```

### 内置 Scorer

```python
from treeskill.registry import registry

# 精确匹配（分类任务）
scorer = registry.get_scorer("exact-match")
scorer("A", "a", {})  # → 1.0（不区分大小写）
scorer("A", "B", {})  # → 0.0

# Judge 评分（默认）
scorer = registry.get_scorer("judge-grade")
scorer("hello", "hello", {})  # → 1.0（无 judge_fn 时 fallback 到精确匹配）
```

### 自定义 Scorer 示例

```python
from treeskill import scorer

@scorer("code-review", set_default=True)
def code_review(output: str, expected: str, context: dict) -> float:
    """用 judge 模型对比代码质量"""
    judge = context.get("judge_client")
    if not judge:
        return 0.5

    resp = judge.messages.create(
        model="MiniMax-M2.7",
        system="Score code quality 0-1. Return ONLY a number.",
        messages=[{"role": "user", "content": f"Gold:\n{expected}\n\nStudent:\n{output}\n\nScore:"}],
        max_tokens=500,
    )
    # 解析分数...
    return score

@scorer("bleu-score")
def bleu_scorer(output: str, expected: str, context: dict) -> float:
    """用 BLEU 评估翻译质量"""
    from nltk.translate.bleu_score import sentence_bleu
    ref = expected.split()
    hyp = output.split()
    return sentence_bleu([ref], hyp)
```

### 在 Skill 中声明 Scorer

```yaml
# SKILL.md frontmatter
metadata:
  scorer: code-review  # 使用注册的 scorer
```

## Gradient — 梯度分析模板

Gradient 模板决定"怎么分析失败原因"。它是一段 system prompt，告诉 judge 模型如何从失败案例中提取有用的梯度信息。

### 内置模板

- `simple` — 直接列出问题清单
- `root-cause` — 深入分析根因，引用 prompt 中的具体问题段落
- `comprehensive` — 多维度审计（清晰度、风格、边界、边缘情况）

### 自定义 Gradient 示例

```python
from treeskill import gradient

@gradient("code-debug")
def code_debug_gradient():
    return (
        "You are a code debugging expert. For each failure:\n"
        "1. Identify the bug type (logic error, missing import, wrong API usage)\n"
        "2. Quote the specific instruction in the prompt that caused it\n"
        "3. Suggest a concrete fix\n"
        "Return 3-5 actionable bullets."
    )

@gradient("copy-review")
def copy_review_gradient():
    return (
        "You are a copywriting coach. Analyze why the copy failed:\n"
        "1. Did it follow the specified framework (AIDA/PAS/FAB)?\n"
        "2. Is the headline compelling enough?\n"
        "3. Are there placeholder text or generic phrases?\n"
        "4. Does it address the target audience's pain points?\n"
        "Return specific, actionable improvements."
    )
```

## Rewriter — 重写策略

Rewriter 模板决定"怎么修改 prompt"。APO 引擎根据梯度分析的结果，用 rewriter 生成候选 prompt。

### 内置策略

- `full-rewrite` — 全量重写，可以重新组织结构
- `conservative` — 保守修改，只修一个最关键的问题
- `distill` — 蒸馏模式，精简 prompt 适配小模型

### 自定义 Rewriter 示例

```python
from treeskill import rewriter

@rewriter("expand")
def expand_rewriter():
    return (
        "You are a prompt expansion expert. Based on the failure analysis:\n"
        "ADD more detailed examples and explanations to the prompt.\n"
        "Do NOT remove any existing content.\n"
        "Focus on making implicit rules explicit.\n"
        "Return ONLY the new prompt — no commentary."
    )

@rewriter("translate-zh")
def translate_rewriter():
    return (
        "Based on the failure analysis, rewrite the System Prompt in Chinese.\n"
        "Preserve all rules and structure, but translate to natural Chinese.\n"
        "Return ONLY the new prompt — no commentary."
    )
```

## Skill Format — 格式适配

Skill Format 用于导入/导出不同生态的 Skill 文件。

### 自定义格式示例

```python
from treeskill import skill_format
from treeskill.schema import Skill

@skill_format("minimax")
class MiniMaxFormat:
    """导入/导出 MiniMax Skills 格式"""

    @staticmethod
    def load(path) -> Skill:
        """从 MiniMax SKILL.md 导入"""
        text = Path(path).read_text()
        # 解析 MiniMax 的 frontmatter 格式...
        return Skill(name=name, system_prompt=body, ...)

    @staticmethod
    def save(skill: Skill, path):
        """导出为 MiniMax 格式"""
        # 转换为 MiniMax 的 YAML frontmatter...
        Path(path).write_text(content)
```

## 工作原理

注册器采用**有注册用注册，没注册用默认**的策略：

```
APO 需要梯度模板
    ↓
Registry 里有注册的 gradient？
    ├── 有 → 随机选一个注册的
    └── 没有 → 用硬编码的 _GRADIENT_TEMPLATES
```

所有内置模板在首次访问时自动注册（懒加载），用户注册的组件会和内置的并存。

### 查看已注册组件

```python
from treeskill.registry import registry

summary = registry.summary()
print(summary)
# {
#   'scorers': {'count': 2, 'names': ['exact-match', 'judge-grade'], 'default': 'judge-grade'},
#   'gradients': {'count': 3, 'names': ['simple', 'root-cause', 'comprehensive']},
#   'rewriters': {'count': 3, 'names': ['full-rewrite', 'conservative', 'distill']},
#   'skill_formats': {'count': 0, 'names': []},
#   ...
# }
```

### 从配置文件加载

```yaml
# config.yaml
plugins:
  scorers:
    - my_module.custom_scorer
  gradients:
    - my_module.custom_gradient
```
