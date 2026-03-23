# Plugin Registry

TreeSkill provides a decorator-driven plugin registry that lets you customize key components of the optimization pipeline.

## Four Registry Types

| Registry | Decorator | Purpose | Built-in |
|----------|-----------|---------|----------|
| **Scorer** | `@scorer("name")` | Scoring function: evaluate candidate prompts | `exact-match`, `judge-grade` |
| **Gradient** | `@gradient("name")` | Gradient template: system prompt for failure analysis | `simple`, `root-cause`, `comprehensive` |
| **Rewriter** | `@rewriter("name")` | Rewrite strategy: system prompt for prompt modification | `full-rewrite`, `conservative`, `distill` |
| **Skill Format** | `@skill_format("name")` | Format adapter: import/export different Skill formats | extensible |

## Scorer — Scoring Functions

Scorers determine "how good is this prompt." APO beam search uses them to select the best candidate; evaluation uses them for final scores.

### Signature

```python
def scorer_fn(output: str, expected: str, context: dict) -> float:
    """
    Args:
        output: Model-generated content
        expected: Ground truth / gold standard
        context: Extra context (judge client, task metadata, etc.)
    Returns:
        Score between 0.0 and 1.0
    """
```

### Built-in Scorers

```python
from treeskill.registry import registry

# Exact match (classification tasks)
scorer = registry.get_scorer("exact-match")
scorer("A", "a", {})  # → 1.0 (case-insensitive)
scorer("A", "B", {})  # → 0.0

# Judge grading (default)
scorer = registry.get_scorer("judge-grade")
```

### Custom Scorer Example

```python
from treeskill import scorer

@scorer("bleu-score")
def bleu_scorer(output: str, expected: str, context: dict) -> float:
    """Evaluate translation quality with BLEU"""
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu([expected.split()], output.split())

@scorer("code-review", set_default=True)
def code_review(output: str, expected: str, context: dict) -> float:
    """Use a judge model to compare code quality"""
    judge = context.get("judge_client")
    if not judge:
        return 0.5
    # Call judge, parse score...
    return score
```

## Gradient — Failure Analysis Templates

Gradient templates determine "how to analyze failures." They are system prompts that tell the judge model how to extract useful gradient information from failure cases.

### Built-in Templates

- `simple` — Direct issue checklist
- `root-cause` — Deep root cause analysis, quoting specific prompt sections
- `comprehensive` — Multi-dimensional audit (clarity, style, scope, edge cases)

### Custom Gradient Example

```python
from treeskill import gradient

@gradient("code-debug")
def code_debug_gradient():
    return (
        "You are a code debugging expert. For each failure:\n"
        "1. Identify the bug type (logic error, missing import, wrong API)\n"
        "2. Quote the specific instruction that caused it\n"
        "3. Suggest a concrete fix\n"
        "Return 3-5 actionable bullets."
    )
```

## Rewriter — Prompt Modification Strategies

Rewriter templates determine "how to modify the prompt." The APO engine uses them to generate candidate prompts based on gradient analysis.

### Built-in Strategies

- `full-rewrite` — Complete restructure allowed
- `conservative` — Minimal change, fix only the most critical issue
- `distill` — Distillation mode: simplify for smaller models

### Custom Rewriter Example

```python
from treeskill import rewriter

@rewriter("expand")
def expand_rewriter():
    return (
        "You are a prompt expansion expert. Based on the failure analysis:\n"
        "ADD more detailed examples and explanations.\n"
        "Do NOT remove any existing content.\n"
        "Return ONLY the new prompt — no commentary."
    )
```

## Skill Format — Format Adapters

Skill Format adapters enable importing/exporting Skills across different ecosystems.

### Custom Format Example

```python
from treeskill import skill_format
from treeskill.schema import Skill

@skill_format("minimax")
class MiniMaxFormat:
    @staticmethod
    def load(path) -> Skill:
        # Parse MiniMax SKILL.md format...
        return Skill(name=name, system_prompt=body, ...)

    @staticmethod
    def save(skill: Skill, path):
        # Convert to MiniMax format...
        Path(path).write_text(content)
```

## How It Works

The registry uses a **registered-first, fallback-to-default** strategy:

```
APO needs a gradient template
    ↓
Registry has registered gradients?
    ├── Yes → randomly pick one
    └── No  → use hardcoded _GRADIENT_TEMPLATES
```

Built-in templates are lazily registered on first access. User-registered components coexist with built-ins.

### Inspect Registered Components

```python
from treeskill.registry import registry

print(registry.summary())
# {
#   'scorers': {'count': 2, 'names': ['exact-match', 'judge-grade'], 'default': 'judge-grade'},
#   'gradients': {'count': 3, 'names': ['simple', 'root-cause', 'comprehensive']},
#   'rewriters': {'count': 3, 'names': ['full-rewrite', 'conservative', 'distill']},
#   'skill_formats': {'count': 0, 'names': []},
# }
```
