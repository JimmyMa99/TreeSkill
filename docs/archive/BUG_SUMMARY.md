# 🎯 EvoSkill Bug 检查总结

**检查时间**: 2026-03-19
**检查范围**: 基于 5 轮实验失败分析
**发现问题**: 4 个 P0 级 bug
**修复状态**: ✅ 全部修复

---

## 📋 发现的 Bug 列表

| # | 严重程度 | 问题 | 影响 | 状态 |
|---|---------|------|------|------|
| 1 | 🔴 P0 | max_tree_depth 完全失效 | 节点从 6 个爆炸到 169 个 | ✅ 已修复 |
| 1.5 | 🔴 P0 | SkillNode 缺少 depth 属性 | 运行时 AttributeError | ✅ 已修复 |
| 2 | 🔴 P0 | 剪枝从未生效 | usage_count 虚高，3 轮剪枝 0 次 | ✅ 已修复 |
| 3 | ⚠️ P1 | 分裂判断过于宽松 | 44 次过度分裂 | ✅ 已修复 |

---

## 🔍 Bug 详细分析

### Bug #1: max_tree_depth 完全失效

**根因**:
```python
# 代码在分裂前没有检查深度
if self.config.auto_split:  # ❌ 缺少深度检查
    specs = self.analyze_split_need(...)
```

**修复**:
```python
current_depth = node_path.count('.') + 1 if node_path else 0
if current_depth >= self.config.max_tree_depth:
    logger.info("🚫 跳过分裂")
else:
    specs = self.analyze_split_need(...)
```

**验证**:
- ✅ 深度计算正确（基于 node_path）
- ✅ 不超过 max_tree_depth 配置
- ✅ 不需要修改 SkillNode 数据结构

---

### Bug #1.5: SkillNode 缺少 depth 属性

**问题**:
- Bug #1 的修复中使用了 `node.depth`
- SkillNode 没有 depth 属性，会导致 AttributeError

**修复**:
- 使用 `node_path.count('.')` 计算深度
- Root: "" → depth=0
- Level 1: "child1" → depth=1
- Level 2: "child1.grandchild" → depth=2

---

### Bug #2: 剪枝从未生效

**根因**:
```python
# 找不到相关 experience 时，使用所有 experiences
if not relevant_experiences:
    relevant_experiences = experiences  # ❌ Bug!
    usage_count = len(relevant_experiences)  # 虚高
```

**影响**:
- usage_count = 所有 experiences 的数量
- 永远不会触发 `usage_count < threshold`
- 3 轮都是剪枝 0 次

**修复**:
```python
if not relevant_experiences:
    usage_count = getattr(node, 'usage_count', 0)  # 使用实际值
    return {
        "performance_score": 0.5,
        "usage_count": usage_count,  # ✅ 正确的 usage_count
        "success_rate": 0.5,
    }
```

---

### Bug #3: 分裂判断过于宽松

**根因**:
- 提示词没有强调分裂的代价
- LLM 倾向于过度分裂（44 次）

**修复**:
- 添加更严格的分裂标准
- 强调 "Default to NOT splitting"
- 添加 "DO NOT split" 的反例指导
- 限制子节点数量（2-4 个 max）

**新的提示词**:
```
**IMPORTANT: Splitting increases complexity and should ONLY be done when clearly necessary.**

SPLIT ONLY if you see CLEAR evidence of:
- **Strong contradictory requirements**
- **Fundamentally different task types**
- **Irreconcilable feedback conflicts**

DO NOT split if:
- The prompt can be improved with better wording
- Issues can be fixed with examples or clarifications

**Default to NOT splitting unless there is overwhelming evidence.**
```

---

## 📊 修复前后对比

### 修复前（实验数据）:
```
Round 1: 6 节点
Round 2: ~50 节点
Round 3: 169 节点 ❌
深度: 4 层（超过配置的 3 层）
剪枝次数: 0
运行时间: 1小时42分钟（失败）
```

### 修复后（预期）:
```
Round 1: 6 节点
Round 2: 10-15 节点
Round 3: 8-12 节点 ✅
深度: ≤ 3 层（符合配置）
剪枝次数: 3-5 次
运行时间: 约30-45分钟（成功）
```

---

## 🎯 修复验证步骤

### 1. 语法检查
```bash
python3 -m py_compile evoskill/core/tree_optimizer.py
python3 -m py_compile evoskill/skill_tree.py
```
**状态**: ✅ 通过

### 2. 单元测试
```bash
python -m pytest tests/test_tree_optimizer.py -v
```
**状态**: ⏳ 待运行

### 3. 小规模验证实验
```bash
python demo/demo_qwen3_8b_tree.py
```
**配置**:
- num_rounds = 3
- samples_per_round = 20
- max_tree_depth = 3

**检查点**:
- [ ] 深度不超过 3 层
- [ ] 节点数控制在 20 以内
- [ ] 剪枝次数 > 0
- [ ] 测试准确率 ≥ 基准线

---

## 🔧 配置建议

### TreeOptimizerConfig
```python
config = TreeOptimizerConfig(
    auto_split=True,
    auto_prune=True,
    prune_strategy="moderate",
    prune_protection_rounds=1,
    prune_usage_threshold=1,
    collapse_instead_of_prune=False,  # 改为 False（真正删除）
    max_tree_depth=3,
    min_samples_for_split=5,  # 从 3 提高到 5
    prune_threshold=0.3,
)
```

### 实验参数
```python
num_rounds = 3  # 从 5 改为 3
samples_per_round = 20  # 从 14 提高到 20
```

---

## 📁 修改的文件

1. ✅ `evoskill/core/tree_optimizer.py`
   - 添加深度检查
   - 修复 usage_count 计算
   - 改进分裂提示词

2. ✅ `evoskill/skill_tree.py`
   - 保持原样（不需要添加 depth 属性）

3. ✅ `BUGFIX_REPORT_20260319.md`
   - 详细修复报告

4. ✅ `BUG_SUMMARY.md`
   - 本总结文档

---

## 🎓 经验教训

### 1. 深度控制至关重要
- **问题**: 缺少深度检查导致节点爆炸
- **教训**: 树结构的深度限制必须在分裂前检查

### 2. Fallback 逻辑要谨慎
- **问题**: "找不到数据就用全部数据" 导致虚高
- **教训**: Fallback 应该返回保守值，而不是全部数据

### 3. LLM 提示词需要明确的指导
- **问题**: 提示词过于宽松导致过度分裂
- **教训**: 提供 "DO NOT" 的反例和更严格的限制

### 4. 节点属性设计要考虑实际使用
- **问题**: 使用了不存在的 depth 属性
- **教训**: 在使用属性前先验证其存在性

---

## 🚀 下一步行动

1. ✅ **代码修复** - 已完成
2. ⏳ **单元测试** - 待运行
3. ⏳ **小规模验证** - 待运行（3 轮，20 条/轮）
4. ⏳ **性能评估** - 待验证

---

## 📞 联系方式

如有问题，请查看：
- 详细报告: `BUGFIX_REPORT_20260319.md`
- 实验记录: `demo/EXPERIMENT_5ROUNDS_ISSUES.md`
- 测试脚本: `demo/demo_qwen3_8b_tree.py`

---

**最后更新**: 2026-03-19
**修复完成**: ✅
**验证状态**: ⏳ 待验证
