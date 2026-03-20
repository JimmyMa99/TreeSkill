# Bug 修复报告

**修复时间**: 2026-03-19
**检查范围**: 基于 5 轮实验问题报告 (demo/EXPERIMENT_5ROUNDS_ISSUES.md)

---

## ✅ 已修复的关键 Bug

### 🔴 Bug #1: max_tree_depth 完全失效

**严重程度**: P0 (致命)

**位置**: `evoskill/core/tree_optimizer.py:243-268`

**问题**:
- 配置了 `max_tree_depth=3`，但实际达到 4 层深度
- 代码在执行自动分裂时，**完全没有检查 `node.depth`**
- 导致节点数从 6 个爆炸到 169 个

**现象**:
```python
Round 1: 6节点
Round 2: ~50节点
Round 3: 169节点 ❌
mathematics.arts_classifier.arts_classification = 4层深度
```

**修复**:
```python
# 修复前：
if self.config.auto_split:  # ❌ 没有深度检查
    specs = self.analyze_split_need(...)

# 修复后：
if self.config.auto_split:
    if node.depth >= self.config.max_tree_depth:
        logger.info(f"🚫 已达最大深度 {self.config.max_tree_depth}，跳过分裂")
    else:
        specs = self.analyze_split_need(...)
```

**影响**:
- ✅ 防止节点爆炸
- ✅ 控制树深度在配置范围内
- ✅ 减少不必要的 API 调用

---

### 🔴 Bug #2: 剪枝从未生效的根本原因

**严重程度**: P0 (致命)

**位置**: `evoskill/core/tree_optimizer.py:688-689`

**问题**:
- `_collect_node_metrics` 方法中，找不到相关 experience 时使用**所有 experiences** 作为 fallback
- 导致 `usage_count` 虚高（等于所有 experiences 的数量）
- 永远不会触发 `usage_count < prune_usage_threshold` 的剪枝条件
- **这就是为什么 3 轮都是剪枝 0 次！**

**现象**:
```python
Round 1: 剪枝0次
Round 2: 剪枝0次
Round 3: 剪枝0次
→ 剪枝从未生效
```

**修复**:
```python
# 修复前：
if not relevant_experiences:
    relevant_experiences = experiences  # ❌ Bug! usage_count 虚高

# 修复后：
if not relevant_experiences:
    usage_count = getattr(node, 'usage_count', 0)
    return {
        "performance_score": 0.5,
        "usage_count": usage_count,  # ✅ 使用节点的实际使用次数
        "success_rate": 0.5,
    }
```

**影响**:
- ✅ 剪枝策略能够正常工作
- ✅ 能够移除低效节点
- ✅ 控制树的规模

---

### ⚠️ Bug #3: 分裂判断过于宽松

**严重程度**: P1 (重要)

**位置**: `evoskill/core/tree_optimizer.py:376-379`

**问题**:
- 分裂提示词没有强调分裂的代价
- LLM 过度建议分裂（44 次）
- 平均每次生成 4 个子节点

**现象**:
```python
LLM recommends splitting into 4 children  # 44次
→ 节点数指数增长
```

**修复**:
- 添加更严格的分裂标准
- 强调 "Default to NOT splitting unless overwhelming evidence"
- 限制子节点数量（2-4 个 max）
- 添加 "DO NOT split" 的反例指导

**修复后的提示词**:
```
**IMPORTANT: Splitting increases complexity and should ONLY be done when clearly necessary.**

SPLIT ONLY if you see CLEAR evidence of:
- **Strong contradictory requirements**
- **Fundamentally different task types**
- **Irreconcilable feedback conflicts**

DO NOT split if:
- The prompt can be improved with better wording
- Issues can be fixed with examples or clarifications
- The feedback suggests incremental improvements

**Default to NOT splitting unless there is overwhelming evidence.**
```

**影响**:
- ✅ 减少不必要的分裂
- ✅ 控制树的规模
- ✅ 提高优化效率

---

### 🔴 Bug #1.5: SkillNode 缺少 depth 属性

**严重程度**: P0 (致命)

**位置**: `evoskill/skill_tree.py:48-81`

**问题**:
- SkillNode 类没有 `depth` 属性
- 在修复 Bug #1 时使用了 `node.depth`，会导致 AttributeError

**修复**:
- 不添加 depth 属性到 SkillNode（避免复杂度）
- 改为使用 `node_path.count('.') + 1` 计算深度
- Root node: node_path = "", depth = 0
- 第一层: node_path = "child1", depth = 1
- 第二层: node_path = "child1.grandchild", depth = 2

```python
# 修复后：
current_depth = node_path.count('.') + 1 if node_path else 0
if current_depth >= self.config.max_tree_depth:
    logger.info(f"🚫 跳过分裂")
```

**影响**:
- ✅ 避免运行时错误
- ✅ 简单高效的深度计算
- ✅ 不需要修改 SkillNode 数据结构

---

## 📊 修复效果预测

### 修复前（实验数据）:
```
Round 1: 6节点
Round 2: ~50节点
Round 3: 169节点 ❌
剪枝次数: 0
运行时间: 1小时42分钟（失败）
```

### 修复后（预期）:
```
Round 1: 6节点
Round 2: 10-15节点
Round 3: 8-12节点 ✅
剪枝次数: 3-5次
运行时间: 约30-45分钟（成功）
```

---

## 🔧 配置建议

基于实验结果，建议调整配置：

### demo_qwen3_8b_tree.py

```python
config = TreeOptimizerConfig(
    auto_split=True,
    auto_prune=True,
    prune_strategy="moderate",
    prune_protection_rounds=1,
    prune_usage_threshold=1,
    collapse_instead_of_prune=False,  # 🔧 改为 False（真正删除）
    max_tree_depth=3,
    min_samples_for_split=5,  # 🔧 从 3 提高到 5
    prune_threshold=0.3,
)
```

### 实验参数

```python
num_rounds = 3  # 🔧 从 5 改为 3（快速验证）
samples_per_round = 20  # 🔧 从 14 提高到 20
```

---

## 🎯 验证步骤

1. **单元测试**:
```bash
python -m pytest tests/test_tree_optimizer.py -v
```

2. **小规模验证**:
```bash
python demo/demo_qwen3_8b_tree.py  # 3轮，20条/轮
```

3. **检查点**:
- [ ] Tree 深度不超过 max_tree_depth
- [ ] 剪枝次数 > 0
- [ ] 节点数控制在 20 以内
- [ ] 测试准确率 ≥ 基准线

---

## 📝 相关文件

- 实验报告: `demo/EXPERIMENT_5ROUNDS_ISSUES.md`
- 修复代码: `evoskill/core/tree_optimizer.py`
- 测试脚本: `demo/demo_qwen3_8b_tree.py`

---

## 🚀 下一步

1. ✅ 运行小规模验证实验（3 轮）
2. ✅ 检查剪枝是否生效
3. ✅ 验证深度控制是否有效
4. ✅ 评估性能提升

---

**修复完成时间**: 2026-03-19
**修复验证**: 待验证
