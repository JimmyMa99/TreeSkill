# 论文分类树感知优化 Demo - 使用指南

## 📋 Demo 简介

这个 demo 展示了 **TreeAwareOptimizer** 的完整能力，通过多轮渐进式优化来自动改进论文分类任务。

### 核心功能展示

1. **自动拆分** - 检测到矛盾反馈时自动拆分 skill 为子 skills
2. **自动剪枝** - 根据性能指标自动移除低效子 skill
3. **部分优化** - 支持只修改 prompt 的某一部分
4. **树感知优化** - 递归优化整棵 skill 树（bottom-up）
5. **温度调节** - 通过调整温度来控制探索/利用的平衡

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate pr

# 安装依赖（如果还没装）
pip install -e .
```

### 2. 配置 API

创建 `.env` 文件：

```bash
# 主模型（用于论文分类）
OPENAI_API_KEY=your-siliconflow-api-key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen2.5-14B-Instruct

# Judge 模型（用于优化，建议用更强的模型）
JUDGE_MODEL=Qwen/Qwen2.5-72B-Instruct
```

### 3. 运行 Demo

```bash
cd /Users/mzm/code/evo_agent
python demo/demo_tree_optimization_complete.py
```

## 📊 预期效果

### 优化流程

```
第1轮：大类分类学习 (温度 0.3)
├─ 目标：训练根节点学会大类分类
├─ 数据：20条训练样本
├─ 策略：低温度，确定性优化
└─ 目标准确率：50-60%

第2轮：细分类别拆分 (温度 0.5)
├─ 目标：检测矛盾，自动拆分
├─ 数据：50条训练样本
├─ 策略：高温度，探索多样性
├─ 自动拆分：检测到 CS vs AI vs Bio 等矛盾
└─ 目标准确率：55-65%

第3轮：细类优化 + 剪枝 (温度 0.2)
├─ 目标：精细优化，移除低性能节点
├─ 数据：50条训练样本
├─ 策略：低温度，保守优化
├─ 自动剪枝：移除准确率 < 30% 的节点
└─ 目标准确率：60-70%
```

### 预期准确率提升

- **初始准确率**: ~40-50% (简单 prompt)
- **第1轮后**: ~50-60% (+5-15%)
- **第2轮后**: ~55-65% (+5-15%)
- **最终准确率**: ~60-70% (+10-25%)

## 🎯 关键特性

### 1. 多轮渐进式优化

不是一次性优化，而是分阶段：
- 第1轮：学会基本分类（大类）
- 第2轮：细化分类（自动拆分）
- 第3轮：精调 + 清理（剪枝）

### 2. 温度调节策略

- **低温 (0.2-0.3)**: 确定性高，保守优化
- **中温 (0.4-0.5)**: 平衡探索和利用
- **高温 (0.6-0.7)**: 探索更多可能性

### 3. 双模型策略

- **主模型 (Qwen2.5-14B)**: 快速分类，成本低
- **Judge 模型 (Qwen2.5-72B)**: 精准优化，质量高

## 🔍 观察要点

运行 demo 时，关注以下日志：

### 1. 拆分检测

```
第2轮中，会看到：
✅ LLM recommends splitting into 3 children
   - child 1: cs-systems (系统类)
   - child 2: ai-ml (AI/机器学习)
   - child 3: bio-medical (生物医学)
```

### 2. 剪枝判断

```
第3轮中，会看到：
✂️  Pruning 'low-performer': performance 0.25 < threshold 0.30
   - 移除低性能节点
   - 职责回归到父节点
```

### 3. 准确率提升

```
📊 第1轮准确率: 55.00% (+12.50%)
📊 第2轮准确率: 62.50% (+7.50%)
📊 第3轮准确率: 67.50% (+5.00%)
```

## 📁 输出文件

优化完成后，会在 `demo-paper-tree-optimized/` 生成 skill 树：

```
demo-paper-tree-optimized/
├── _meta.yaml              # 包级元信息
├── root.yaml               # 根节点（优化后的）
├── cs-systems/             # 自动拆分的子节点
│   ├── _meta.yaml
│   ├── root.yaml
│   ├── distributed-systems.yaml
│   └── databases.yaml
├── ai-ml/
│   ├── _meta.yaml
│   ├── root.yaml
│   ├── nlp.yaml
│   └── cv.yaml
└── bio-medical/
    ├── _meta.yaml
    └── root.yaml
```

## ⚙️ 自定义配置

如果你想调整参数，修改 `run_tree_optimization()` 调用：

```python
# 更激进的拆分
tree_optimizer = TreeAwareOptimizer(
    adapter=adapter,
    config=TreeOptimizerConfig(
        auto_split=True,
        auto_prune=True,
        prune_threshold=0.4,      # 提高剪枝阈值
        min_samples_for_split=3,  # 降低拆分最小样本数
        max_tree_depth=5,         # 允许更深层次
    ),
)
```

## 🐛 常见问题

### Q1: API 调用失败？

```bash
# 检查 API Key
echo $OPENAI_API_KEY

# 测试连接
curl https://api.siliconflow.cn/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Q2: 准确率没有提升？

可能原因：
1. 训练数据太少（建议至少 30 条）
2. 温度设置不当（试试 0.3-0.5 之间）
3. Judge 模型不够强（建议用 72B 模型）

### Q3: 拆分没有发生？

检查条件：
- 至少 5 个有反馈的样本
- 反馈确实存在矛盾（不同任务要求冲突）
- `auto_split=True` 已设置

## 📈 效果验证

### 成功标志

✅ 准确率持续提升 3 轮
✅ 自动拆分检测到矛盾并生成子节点
✅ 剪枝移除了低性能节点
✅ 最终树结构清晰合理

### 提前结束条件

如果连续 3 轮都有显著提升（>15%），可以认为优化成功，提前结束。

## 🎓 学习要点

通过这个 demo，你会学到：

1. **如何使用 TreeAwareOptimizer** 进行树感知优化
2. **如何调整温度** 来控制优化策略
3. **如何解读拆分和剪枝** 的日志输出
4. **如何评估优化效果** 的好坏
5. **如何保存和加载** 优化后的 skill 树

## 📚 相关文档

- [TreeAwareOptimizer 完整文档](../tresskill/core/tree_optimizer.py)
- [Skill 树管理](../tresskill/skill_tree.py)
- [优化策略](../tresskill/core/strategies.py)
- [验证器](../tresskill/core/validators.py)

## 💡 提示

- 如果是第一次运行，建议使用默认参数
- 观察日志中的 "Split analysis" 和 "Prune analysis" 部分
- 注意第 2 轮的温度提升（0.3 → 0.5）
- 检查最终保存的树结构是否符合预期

祝优化成功！ 🎉
