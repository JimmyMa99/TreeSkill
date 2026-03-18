# 渐进式剪枝（Progressive Pruning）功能说明

## 🎯 核心概念

**渐进式剪枝** 是一种智能的技能树管理策略，通过多层次的剪枝控制，避免误删有潜力的新节点，同时实现"渐进式披露"（Progressive Disclosure）的上下文管理。

---

## 📊 问题背景

### 传统剪枝的问题

```python
# 传统方法：立即剪枝低使用率节点
if usage_count < 2:
    prune(node)  # ❌ 新创建的节点usage_count=0，立即被删除！
```

**问题**：
1. 新节点刚创建时`usage_count=0`，立即被剪枝
2. 没有给新节点足够时间积累经验
3. 可能删除有潜力的技能分支

### Demo中的实际案例

```
第1轮: 拆分出5个子节点
       Pruning 'astronomy': very low usage count (0)  ❌
       Pruning 'engineering': very low usage count (0)  ❌
       ...
       所有子节点被删除！

结果: 拆分27次，剪枝27次，最终只剩root节点
```

---

## 🔧 解决方案：渐进式剪枝

### 1. 节点年龄追踪（Node Age Tracking）

```python
@dataclass
class SkillNode:
    name: str
    skill: Skill
    children: Dict[str, "SkillNode"]
    age: int = 0  # 🎯 节点年龄：经历了几轮优化
    usage_count: int = 0  # 🎯 使用次数
    collapsed: bool = False  # 🎯 折叠状态（渐进式披露）
```

**工作原理**：
- 每次优化开始时，所有节点`age += 1`
- 新创建的节点`age = 0`
- 剪枝决策考虑节点年龄

---

### 2. 保护期机制（Protection Period）

```python
config = TreeOptimizerConfig(
    prune_protection_rounds=3,  # 🔧 新节点保护3轮
)

def analyze_prune_need(node, metrics):
    # 新节点保护期
    if node.age < config.prune_protection_rounds:
        logger.info(f"🛡️ 保护 '{node.name}': age {node.age} < protection period {config.prune_protection_rounds}")
        return False  # 不剪枝

    # 其他剪枝条件...
```

**效果**：
- 第0-2轮：新节点被保护，不剪枝
- 第3轮+：有足够数据后，才考虑剪枝
- 给新节点充分的"成长"时间

---

### 3. 多级剪枝策略（Pruning Strategies）

```python
config = TreeOptimizerConfig(
    prune_strategy="conservative",  # 🔧 选择策略
)
```

#### 可用策略

| 策略 | 描述 | Usage阈值 | Performance阈值 | Success阈值 | 适用场景 |
|------|------|-----------|----------------|-------------|---------|
| `disabled` | 完全禁用 | - | - | - | 只拆分不剪枝 |
| `conservative` | 保守 | 5次 | 0.2 | 0.2 | 保护创新 |
| `moderate` | 适中 | 2次 | 0.3 | 0.3 | **默认推荐** |
| `aggressive` | 激进 | 1次 | 0.4 | 0.4 | 快速收敛 |

**推荐**：
- **探索阶段**：使用`conservative`或`moderate`
- **生产环境**：使用`moderate`
- **快速实验**：使用`aggressive`（但风险高）

---

### 4. 渐进式披露（Progressive Disclosure）

```python
config = TreeOptimizerConfig(
    collapse_instead_of_prune=True,  # 🔧 折叠而非删除
)
```

**工作原理**：
- 不直接删除节点，而是标记为`collapsed=True`
- 折叠的节点：
  - 不参与路由（节省推理开销）
  - 保留在树中（知识不丢失）
  - 可在需要时"展开"（知识复用）

**示例**：
```python
# 传统剪枝
prune(node)  # ❌ 永久删除，知识丢失

# 渐进式披露
collapse(node)  # ✅ 隐藏但保留，未来可恢复
```

---

## 🚀 完整配置示例

### 推荐配置（保守策略）

```python
config = TreeOptimizerConfig(
    # 启用功能
    auto_split=True,
    auto_prune=True,

    # 渐进式剪枝
    prune_strategy="conservative",  # 保守策略
    prune_protection_rounds=3,  # 保护3轮
    prune_usage_threshold=5,  # 至少使用5次
    prune_threshold=0.2,  # performance < 0.2才剪枝
    collapse_instead_of_prune=True,  # 折叠而非删除

    # 其他参数
    max_tree_depth=3,
    min_samples_for_split=3,
)
```

### 推荐配置（适中策略）- **默认推荐**

```python
config = TreeOptimizerConfig(
    auto_split=True,
    auto_prune=True,

    prune_strategy="moderate",  # 适中策略
    prune_protection_rounds=2,  # 保护2轮
    prune_usage_threshold=2,  # 至少使用2次
    prune_threshold=0.3,
    collapse_instead_of_prune=True,

    max_tree_depth=3,
    min_samples_for_split=5,
)
```

### 禁用剪枝（只拆分）

```python
config = TreeOptimizerConfig(
    auto_split=True,
    auto_prune=False,  # 或 prune_strategy="disabled"
)
```

---

## 📈 效果对比

### 传统剪枝（无保护）

```
第1轮: 拆分5个节点
       剪枝5个节点（usage_count=0）
       最终: 1个节点（只有root）

准确率: 23.3% → 33.3% (+42.9%)
```

### 渐进式剪枝（保守策略）

```
第1轮: 拆分5个节点
       保护所有新节点（age=0 < 3）
       最终: 6个节点（root + 5个子节点）

第2轮: 继续优化
       仍然保护（age=1 < 3）
       最终: 6个节点（可能拆分更多）

第3轮: 开始考虑剪枝
       仅删除真正低效的节点
       最终: 保留高效节点

准确率: 预期更高（需要实验验证）
```

---

## 🎯 使用场景

### 场景1：探索性任务

```python
# 使用保守策略，鼓励探索
config = TreeOptimizerConfig(
    prune_strategy="conservative",
    prune_protection_rounds=5,
    collapse_instead_of_prune=True,
)
```

**适用于**：
- 新领域探索
- 创新性任务
- 不确定最优结构

### 场景2：生产环境

```python
# 使用适中策略，平衡效率和质量
config = TreeOptimizerConfig(
    prune_strategy="moderate",
    prune_protection_rounds=2,
    collapse_instead_of_prune=True,
)
```

**适用于**：
- 稳定的任务类型
- 已知的技能结构
- 平衡性能和维护成本

### 场景3：快速迭代

```python
# 使用激进策略或禁用
config = TreeOptimizerConfig(
    prune_strategy="aggressive",
    # 或
    auto_prune=False,
)
```

**适用于**：
- 原型开发
- 快速实验
- 数据充足的情况

---

## ⚠️ 注意事项

### 1. 保护期不宜过长

```python
# ❌ 太保守
prune_protection_rounds=10  # 10轮后才剪枝，树可能过于庞大

# ✅ 合理
prune_protection_rounds=2-3  # 给新节点2-3轮时间足够
```

### 2. Collapse vs Delete

```python
# 优点：知识保留，可恢复
collapse_instead_of_prune=True

# 缺点：占用内存，树结构复杂
# 建议：定期清理完全无用的折叠节点
```

### 3. 策略选择

- **初期**：使用`conservative`，鼓励探索
- **中期**：切换到`moderate`，平衡优化
- **后期**：可考虑`aggressive`，精简结构

---

## 🔬 实验验证

### 测试脚本

```bash
# 运行带渐进式剪枝的demo
python demo/demo_qwen3_8b_tree.py
```

### 预期结果

1. **第1轮**：
   - 拆分出多个子节点
   - 所有新节点被保护（age=0）
   - 无剪枝发生

2. **第2轮**：
   - 继续保护（age=1）
   - 可能拆分更多
   - 无剪枝

3. **第3轮**：
   - 保护期结束（age=2）
   - 开始考虑剪枝
   - 只删除真正低效的节点

### 观察指标

```bash
# 查看日志中的保护信息
grep "🛡️ Protecting" demo-qwen3-8b-tree/*.log

# 查看最终树结构
cat demo-qwen3-8b-tree/TREE_VISUALIZATION.txt

# 统计节点数
grep "子skill数" demo-qwen3-8b-tree/*.log
```

---

## 📚 参考资料

### 相关文件

- `evoskill/skill_tree.py` - SkillNode定义（age, usage_count, collapsed字段）
- `evoskill/core/tree_optimizer.py` - 剪枝逻辑实现
- `demo/demo_qwen3_8b_tree.py` - 完整Demo

### 配置参数文档

```python
@dataclass
class TreeOptimizerConfig:
    prune_strategy: str = "moderate"
    # 可选: "disabled", "conservative", "moderate", "aggressive"

    prune_protection_rounds: int = 2
    # 新节点保护轮数

    prune_usage_threshold: int = 2
    # 最小使用次数要求

    prune_threshold: float = 0.3
    # 性能阈值（0.0-1.0）

    collapse_instead_of_prune: bool = True
    # True=折叠，False=删除
```

---

## 🎉 总结

**渐进式剪枝的核心价值**：

1. **保护创新** - 给新节点足够的成长时间
2. **渐进披露** - 折叠而非删除，保留知识
3. **可控性** - 多种策略适应不同场景
4. **鲁棒性** - 避免误删，提高系统稳定性

**推荐使用方式**：
- 默认使用`moderate`策略
- 探索任务用`conservative`
- 启用`collapse_instead_of_prune`
- 定期评估和调整策略

---

*Generated with 🤖 by Claude Code*
*Date: 2026-03-18*
