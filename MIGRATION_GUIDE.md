# 🔄 改名 + 插件化迁移指南

## 改名：evo-framework → evoskill

### 动机

| 维度 | evo-framework | evoskill |
|------|--------------|----------|
| **记忆度** | ⭐⭐ 一般 | ⭐⭐⭐⭐⭐ 很好 |
| **具体性** | ⭐⭐ 太泛 | ⭐⭐⭐⭐⭐ 明确 |
| **专业性** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 专业 |
| **品牌** | evo-framework | `evoskill` ✅ |

---

## Phase 1: 项目改名（已完成）

### 1.1 包结构

```
旧结构：
evo_framework/
├── __init__.py
├── core/
├── adapters/
├── ...

新结构：
evoskill/              # ✅ 改名
├── __init__.py
├── core/              # ✅ 核心抽象层
├── adapters/          # ✅ 模型适配器
├── registry.py        # ✅ 插件注册表（新增）
├── ...

# 保留向后兼容
evo_framework/         # ⚠️ 保留，但弃用
├── __init__.py        # 导入并重导出 evoskill
```

### 1.2 导入迁移

#### 旧代码（v0.1）

```python
from evo_framework import Skill, Trace
from evo_framework.llm import LLMClient
```

#### 新代码（v0.2+）

```python
# 推荐：新API
from evoskill import TextPrompt, OpenAIAdapter

# 向后兼容：旧API继续工作
from evo_framework import Skill, Trace  # ⚠️ 弃用但可用
```

---

## Phase 2: 插件化架构（新增）

### 2.1 核心概念

```python
from evoskill import registry, adapter, optimizer, hook

# ------------------------------------------------
# 1. 注册适配器
# ------------------------------------------------

# 方法1: 装饰器
@adapter("my-custom", set_default=True)
class MyAdapter(BaseModelAdapter):
    def generate(self, prompt, **kwargs):
        return "Hello"

# 方法2: 直接注册
registry.register_adapter("openai", OpenAIAdapter)

# 方法3: 配置文件
registry.load_from_config("config.yaml")

# ------------------------------------------------
# 2. 使用适配器
# ------------------------------------------------

# 获取默认适配器
adapter = registry.get_adapter()

# 获取指定适配器
adapter = registry.get_adapter("my-custom", model="gpt-4o")

# 列出所有适配器
print(registry.list_adapters())

# ------------------------------------------------
# 3. 注册钩子
# ------------------------------------------------

@hook('after_optimize')
def log_to_wandb(old_skill, new_skill, gradient):
    wandb.log({'version': new_skill.version})

@hook('on_skill_saved')
def backup_to_s3(skill, path):
    s3.upload(path, f'backups/{skill.name}.yaml')

# ------------------------------------------------
# 4. 配置文件驱动
# ------------------------------------------------

# config.yaml
"""
adapters:
  openai:
    class: evoskill.adapters.openai.OpenAIAdapter
    default: true
    config:
      model: gpt-4o-mini

  claude:
    class: evoskill.adapters.anthropic.AnthropicAdapter
    config:
      model: claude-3-5-sonnet-20241022

optimizers:
  default:
    class: evoskill.optimizer.TrainFreeOptimizer
    config:
      max_steps: 3

hooks:
  after_optimize:
    - my_hooks.log_to_wandb
  on_skill_saved:
    - my_hooks.backup_to_s3
"""

# 使用
from evoskill import EvoSkill

app = EvoSkill.from_config("config.yaml")
adapter = app.get_adapter()
optimizer = app.get_optimizer()
```

### 2.2 生命周期钩子

| 事件 | 触发时机 | 回调参数 |
|------|---------|----------|
| `before_generate` | 生成前 | `prompt, context` |
| `after_generate` | 生成后 | `prompt, context, response` |
| `before_optimize` | 优化前 | `skill, experiences` |
| `after_optimize` | 优化后 | `old_skill, new_skill, gradient` |
| `on_gradient_computed` | 梯度计算完成 | `skill, failures, gradient` |
| `on_skill_saved` | skill保存时 | `skill, path` |
| `on_error` | 发生错误 | `error, event, callback` |

### 2.3 用户客制化示例

#### 示例 1: 自定义优化策略

```python
# my_optimizer.py

from evoskill import optimizer, TrainFreeOptimizer

@optimizer("aggressive")
class AggressiveOptimizer(TrainFreeOptimizer):
    """激进优化 - 每次大幅修改prompt"""

    def apply_gradient(self, prompt, gradient):
        # 不保守，大改
        return super().apply_gradient(
            prompt,
            gradient,
            conservative=False  # 激进模式
        )

@optimizer("conservative")
class ConservativeOptimizer(TrainFreeOptimizer):
    """保守优化 - 只微调"""

    def apply_gradient(self, prompt, gradient):
        # 保守，小改
        return super().apply_gradient(
            prompt,
            gradient,
            conservative=True  # 保守模式
        )

# 使用
optimizer = registry.get_optimizer("aggressive")
```

#### 示例 2: 监控和日志钩子

```python
# hooks.py

from evoskill import hook
import wandb
import slack_sdk

@hook('after_optimize')
def log_to_wandb(old_skill, new_skill, gradient):
    """优化后记录到WandB"""
    wandb.init(project='evoskill')
    wandb.log({
        'version': new_skill.version,
        'gradient_length': len(str(gradient)),
        'timestamp': datetime.now(),
    })

@hook('on_skill_saved')
def notify_slack(skill, path):
    """保存时通知Slack"""
    client = slack_sdk.WebClient(token=os.environ['SLACK_TOKEN'])
    client.chat_postMessage(
        channel='#ml-updates',
        text=f"✅ Skill saved: {skill.name} v{skill.version}",
    )

@hook('on_error')
def log_error(error, event, callback):
    """错误时记录到Sentry"""
    import sentry_sdk
    sentry_sdk.capture_exception(error)
```

---

## Phase 3: 完整迁移步骤

### Step 1: 重命名包

```bash
# 1. 重命名目录
mv evo_framework evoskill

# 2. 更新 pyproject.toml
[project]
name = "evoskill"
version = "0.2.0"

[project.scripts]
evoskill = "evoskill.cli:main"

# 3. 创建向后兼容层
mkdir evo_framework
cat > evo_framework/__init__.py << 'EOF'
"""
Evo-Framework (Legacy)

⚠️ DEPRECATED: Use 'evoskill' instead.
This module is kept for backward compatibility only.
"""

# Re-export everything from evoskill
from evoskill import *
EOF
```

### Step 2: 更新导入

```python
# 全局替换
find . -type f -name "*.py" -exec sed -i '' 's/from evo_framework/from evoskill/g' {} \;
find . -type f -name "*.py" -exec sed -i '' 's/import evo_framework/import evoskill/g' {} \;

# 更新文档
find . -type f -name "*.md" -exec sed -i '' 's/evo-framework/evoskill/g' {} \;
find . -type f -name "*.md" -exec sed -i '' 's/evo_framework/evoskill/g' {} \;
```

### Step 3: 添加Registry

```python
# evoskill/__init__.py

# Core
from evoskill.core import (
    TextPrompt,
    MultimodalPrompt,
    # ...
)

# Adapters
from evoskill.adapters.openai import OpenAIAdapter
from evoskill.adapters.anthropic import AnthropicAdapter

# Registry (新增)
from evoskill.registry import (
    registry,
    adapter,
    optimizer,
    hook,
    ComponentMeta,
)

# Main API
from evoskill.api import EvoSkill

__all__ = [
    # Core
    "TextPrompt",
    # Adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    # Registry
    "registry",
    "adapter",
    "optimizer",
    "hook",
    "ComponentMeta",
    # Main API
    "EvoSkill",
]
```

### Step 4: 更新文档

```markdown
# README.md

# EvoSkill: Train-free Prompt Evolution Framework

## 安装

\`\`\`bash
pip install evoskill
\`\`\`

## 快速开始

\`\`\`python
from evoskill import TextPrompt, OpenAIAdapter, registry

# 创建适配器
adapter = OpenAIAdapter(model="gpt-4o-mini")

# 或使用Registry
adapter = registry.get_adapter("openai")

# 生成
prompt = TextPrompt(content="你是助手。")
response = adapter.generate(prompt)
\`\`\`

## 自定义组件

\`\`\`python
from evoskill import adapter, hook

@adapter("my-custom")
class MyAdapter(BaseModelAdapter):
    pass

@hook('after_optimize')
def log_to_wandb(old, new, gradient):
    wandb.log({'version': new.version})
\`\`\`
```

---

## Phase 4: 发布

### 4.1 PyPI发布

```bash
# 构建
python -m build

# 上传
twine upload dist/*

# 用户安装
pip install evoskill
```

### 4.2 版本管理

```python
# evoskill/__init__.py

__version__ = "0.2.0"
__author__ = "Your Name"
__email__ = "your@email.com"

VERSION_INFO = {
    'major': 0,
    'minor': 2,
    'patch': 0,
}
```

---

## 迁移检查清单

### 改名 ✅
- [ ] 重命名 `evo_framework/` → `evoskill/`
- [ ] 更新 `pyproject.toml`
- [ ] 创建向后兼容层 `evo_framework/__init__.py`
- [ ] 全局替换导入语句
- [ ] 更新所有文档
- [ ] 更新测试文件

### 插件化 ✅
- [ ] 实现 `Registry` 类
- [ ] 实现装饰器 `@adapter`, `@optimizer`, `@hook`
- [ ] 实现配置文件加载
- [ ] 添加生命周期钩子系统
- [ ] 更新所有适配器使用Registry
- [ ] 添加示例代码
- [ ] 更新文档

### 测试 ✅
- [ ] 测试向后兼容性
- [ ] 测试装饰器注册
- [ ] 测试配置文件加载
- [ ] 测试钩子触发
- [ ] 测试动态导入

### 文档 ✅
- [ ] 更新 README
- [ ] 添加插件开发指南
- [ ] 添加迁移指南
- [ ] 更新API文档

---

## 时间估算

| 任务 | 时间 | 优先级 |
|------|------|--------|
| 改名 | 2小时 | P0 |
| 实现Registry | 4小时 | P0 |
| 更新适配器 | 2小时 | P0 |
| 测试 | 3小时 | P0 |
| 文档 | 3小时 | P1 |
| **总计** | **14小时（~2天）** | |

---

## 下一步

1. ✅ **改名** → evoskill
2. ✅ **实现Registry** → 插件化
3. ⏳ **继续优化引擎** → 核心功能
4. ⏳ **发布PyPI** → 社区使用

---

*Migration Guide v0.2.0*
