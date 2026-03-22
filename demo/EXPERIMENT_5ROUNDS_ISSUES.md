# 5轮Tree优化实验问题总结

**实验时间**: 2026-03-19
**实验目标**: 运行5轮tree-aware优化，验证渐进式剪枝策略
**最终结果**: ❌ 失败（Round 4因API错误中断）

---

## 📊 实验结果

### 完成进度

- ✅ Round 1: 完成
- ✅ Round 2: 完成
- ✅ Round 3: 完成
- ❌ Round 4: 失败（SSL错误）
- ❌ Round 5: 未执行

**运行时间**: 09:35 - 11:17 (约1小时42分钟)

### 测试准确率变化

| 轮次 | 训练集 | 训练准确率 | 测试准确率 | 变化 | 累计变化 |
|------|-------|-----------|-----------|------|---------|
| 基准 | - | - | **23.3%** | - | - |
| Round 1 | 14条 | 21.4% | **23.3%** | +0.0% | +0.0% |
| Round 2 | 14条 | 35.7% | **16.7%** | **-6.7%** | **-6.7%** |
| Round 3 | 14条 | - | **20.0%** | **+3.3%** | **-3.3%** |

**结论**: 3轮后性能下降3.3%

### Tree规模演化

```
Round 1: 6节点 (root + 5个子节点)
├── astronomy
├── environmental_science
├── computer_science
├── physics
└── mathematics

Round 2: ~50节点
└── 每个子节点分裂3-4个专门分类器

Round 3: 169节点 ❌
└── 深层嵌套：mathematics.arts_classifier.arts_classification (4+层)
```

**问题**: 节点数爆炸增长（28倍），深度失控

---

## 🔴 P0级问题（致命）

### 1. Tree节点爆炸

**现象**:
- Round 1: 6节点 ✅
- Round 2: ~50节点 ⚠️
- Round 3: 169节点 ❌

**根因**:
```python
# LLM几乎总是建议分裂
LLM recommends splitting into 4 children  # 44次
→ 平均每次生成4个子节点
→ 节点数指数增长
```

**影响**:
- 单轮需要3-6小时优化所有节点
- 无法完成5轮实验
- API调用次数爆炸

**数据**:
```
分裂次数: 44次（Round 2未完成时）
生成子节点: 169个
平均每次分裂: 4个子节点
```

### 2. max_tree_depth 完全失效

**配置**:
```python
max_tree_depth=3
```

**实际**:
```
mathematics.arts_classifier.arts_classification
root → mathematics → arts_classifier → arts_classification
= 4层深度 ❌
```

**根因**:
- 代码中未检查 `node.depth >= max_tree_depth`
- 分裂前未验证深度限制

**影响**: 无法控制树的规模和深度

### 3. 剪枝策略无效

**配置**:
```python
prune_strategy="moderate"
prune_protection_rounds=1
prune_usage_threshold=1
collapse_instead_of_prune=True  # 只折叠，不删除
```

**问题**:
1. 新节点保护期内不剪枝
2. 保护期后继续分裂
3. 节点被折叠但仍需优化
4. 节点数只增不减

**数据**:
```
Round 1: 剪枝0次
Round 2: 剪枝0次
Round 3: 剪枝0次
→ 剪枝从未生效
```

---

## 🟡 P1级问题（重要）

### 4. 性能下降

**时间线**:
```
基准 → Round 1: 23.3% → 23.3% (+0.0%) ✅ 正常
Round 1 → Round 2: 23.3% → 16.7% (-6.7%) ❌ 崩溃
Round 2 → Round 3: 16.7% → 20.0% (+3.3%) ⚠️ 部分恢复
```

**Round 2 性能崩溃分析**:
- 启用剪枝后性能大幅下降
- 训练准确率35.7% vs 测试准确率16.7% → 过拟合？
- 可能原因：过早剪枝导致有效节点不足

### 5. API不稳定

**错误**:
```
httpcore.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: self-signed certificate in certificate chain
```

**影响**:
- Round 4开始时失败
- 无法完成5轮实验
- 浪费1小时42分钟计算

### 6. 训练数据量不足

**配置**:
```python
num_rounds = 5
samples_per_round = 14  # 每轮仅14条数据
```

**问题**:
- 经验积累不足
- 节点使用次数低
- 剪枝判断不准确

---

## 🟢 P2级问题（次要）

### 7. 温度参数递增

```python
Round 1: temp=0.35
Round 2: temp=0.40
Round 3: temp=0.45
Round 4: temp=0.50  # 过于随机
```

### 8. Checkpoint目录过大

```bash
demo/outputs/demo-qwen3-8b-tree-5rounds/
├── round1/  # 6个节点目录
├── round2/  # ~50个节点目录
└── round3/  # 169个节点目录 ❌
```

---

## 💡 解决方案建议

### 最小可行修复（MVP）

#### 1. 修复深度控制

```python
# treeskill/core/tree_optimizer.py

def _should_split_node(self, node: SkillTreeNode) -> bool:
    # 检查深度限制
    if node.depth >= self.config.max_tree_depth:
        logger.info(f"🚫 已达最大深度 {self.config.max_tree_depth}，跳过分裂")
        return False

    # 原有逻辑...
```

#### 2. 提高分裂门槛

```python
TreeOptimizerConfig(
    min_samples_for_split=5,  # 从3提高到5
    min_performance_gap=0.2,  # 性能差距<20%不分裂
)
```

#### 3. 真正删除节点

```python
TreeOptimizerConfig(
    collapse_instead_of_prune=False,  # 删除而非折叠
    prune_strategy="aggressive",  # 激进剪枝
    prune_protection_rounds=0,  # 无保护期
)
```

#### 4. 降低轮次验证

```python
num_rounds = 3  # 从5改为3
samples_per_round = 20  # 从14提高到20
```

### 预期效果

```
限制前:
Round 1: 6节点 → Round 2: 50节点 → Round 3: 169节点 ❌
运行时间: 1小时42分钟（失败）

限制后:
Round 1: 6节点 → Round 2: 10-15节点 → Round 3: 8-12节点 ✅
运行时间: 约30-45分钟（成功完成）
```

---

## 📚 关键代码位置

### 1. 深度控制缺失

**文件**: `treeskill/core/tree_optimizer.py`
**方法**: `_auto_split_node()`
**问题**: 未检查 `node.depth`

### 2. 分裂门槛过低

**文件**: `demo/demo_qwen3_8b_tree.py`
**配置**: `min_samples_for_split=3`

### 3. 剪枝策略问题

**文件**: `treeskill/core/tree_optimizer.py`
**方法**: `_should_prune_node()`, `_prune_node()`
**问题**:
- 保护期机制
- collapse而非删除

---

## 🎯 下一步行动

### 立即修复（优先级排序）

1. ✅ **修复max_tree_depth** - 添加深度检查
2. ✅ **提高分裂门槛** - min_samples=5, 性能差距>20%
3. ✅ **真正删除节点** - collapse_instead_of_prune=False
4. ✅ **降低轮次** - 3轮，每轮20条数据

### 验证实验

```bash
# 修改配置后运行3轮验证
python demo/demo_qwen3_8b_tree.py

# 预期:
# - 30-45分钟完成
# - 节点数控制在20以内
# - 测试准确率≥23.3%
```

---

## 📖 相关文档

- [渐进式剪枝策略](./PROGRESSIVE_PRUNING.md)
- [Tree优化Demo说明](./TREE_OPTIMIZATION_DEMO.md)
- [Bug修复总结](./BUGFIX_SUMMARY.md)

---

## 📝 实验日志

### Round 1 (09:35-09:42, 7分钟)

**配置**:
- 拆分: ✅
- 剪枝: ❌
- 温度: 0.35

**结果**:
- 训练: 21.4%
- 测试: 23.3% (+0.0%)
- 节点: 6个
- 拆分: 5次
- 剪枝: 0次

### Round 2 (09:42-10:29, 47分钟)

**配置**:
- 拆分: ✅
- 剪枝: ✅
- 温度: 0.40

**结果**:
- 训练: 35.7%
- 测试: 16.7% (-6.7%) ❌
- 节点: ~50个
- 拆分: ~44次
- 剪枝: 0次

**观察**: 性能崩溃，但训练准确率提升，疑似过拟合

### Round 3 (10:29-10:44, 15分钟)

**配置**:
- 拆分: ✅
- 剪枝: ✅
- 温度: 0.45

**结果**:
- 测试: 20.0% (+3.3%) ⚠️
- 节点: 169个 ❌
- 拆分: ~100次

**观察**: 部分恢复，但节点爆炸

### Round 4 (10:44-11:17, 33分钟)

**配置**:
- 拆分: ✅
- 剪枝: ✅
- 温度: 0.50

**结果**:
- 训练: 21.4%
- 测试: ❌ 失败
- 错误: SSL证书验证失败

---

**总结**: 本次实验暴露了tree-aware优化的严重缺陷，主要问题是节点爆炸和深度控制失效。修复后需要重新验证。
