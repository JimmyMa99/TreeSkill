# APO 自动提示优化原理

evoskill 的核心是 **APO（Automatic Prompt Optimization）**——把 System Prompt 当作"权重"，通过文本梯度下降（TGD）让 prompt 自动进化。

## 两种优化模式

| 模式 | 反馈来源 | 适用场景 | 示例 |
|------|---------|---------|------|
| **交互式** | 人工 `/bad`、`/rewrite` | 开发调试、快速迭代 | `examples/example_optimizer.py` |
| **全自动** | 测试集 + LLM Judge | 生产环境、批量优化 | `examples/example_fully_automatic.py` |

---

## 模式 1：交互式优化

用户在 CLI 中与 Agent 对话，标记不满意的回复，触发优化。

```mermaid
graph TD
    A[加载 Skill] --> B[用户提问]
    B --> C[Agent 回复]
    C --> D{满意?}

    D -->|是| B
    D -->|否| E[标记反馈]

    E --> F["/bad 原因"]
    E --> G["/rewrite 理想回答"]
    E --> H["/target 优化方向"]

    F --> I["触发 /optimize"]
    G --> I
    H --> I

    I --> J[TGD 优化]
    J --> K[保存新 Prompt + Checkpoint]
    K --> B

    style A fill:#e1f5ff
    style E fill:#fff9c4
    style J fill:#fff9c4
    style K fill:#c8e6c9
```

**步骤**：
1. **收集反馈** — `/bad`、`/rewrite` 标记不满意的交互
2. **诊断失败** — 筛选低分 Trace
3. **计算梯度** — Judge 模型分析 prompt 为何导致失败
4. **重写 Prompt** — Judge 模型据此改写 System Prompt
5. **保存** — 版本号 +1，写入 SKILL.md + 保存 checkpoint

---

## 模式 2：全自动优化

基于测试集和 LLM Judge，无需人工干预，适合生产环境持续优化。

```mermaid
graph TD
    A[初始 Prompt] --> B[准备测试集]
    B --> C[在测试集上运行]

    C --> D{LLM Judge 评估}
    D -->|失败| E[收集失败案例]
    D -->|全部通过| F["✅ 优化完成"]

    E --> G[计算文本梯度]
    G --> H[重写 Prompt]
    H --> I[验证新 Prompt]

    I --> J{达标?}
    J -->|否| C
    J -->|是| F

    style A fill:#e1f5ff
    style F fill:#c8e6c9
    style E fill:#fff9c4
    style G fill:#fff9c4
    style H fill:#fff9c4
```

---

## 核心 TGD 循环

无论哪种模式，内部都走同一套 TGD 循环：

```mermaid
graph LR
    A[失败案例] --> B[梯度计算]
    B --> C[分析失败原因]
    C --> D[改进建议]

    D --> E[梯度应用]
    E --> F[重写 Prompt]

    F --> G{验证}
    G -->|通过| H[返回新 Prompt]
    G -->|不通过| A

    H --> I["版本号 +1"]

    style A fill:#ffcdd2
    style B fill:#fff9c4
    style E fill:#fff9c4
    style H fill:#c8e6c9
```

**核心思想**：
- 失败案例 = 训练信号
- 文本梯度 = 自然语言的失败分析
- 梯度应用 = 据此重写 prompt
- 无需训练模型，仅靠 API 调用

---

## 断点续跑

优化过程支持中断恢复。每个节点优化完成后自动保存进度到 `.evo_resume.json`，中断后下次启动可从断点继续。

```mermaid
graph TD
    A[开始优化] --> B[创建 ResumeState]
    B --> C[优化 node-a]
    C --> D["保存进度 ✓ node-a"]
    D --> E[优化 node-b]
    E --> F["⚡ 中断"]

    F --> G[下次启动]
    G --> H{检测到 .evo_resume.json}
    H -->|resume| I["跳过 node-a"]
    I --> J[从 node-b 继续]
    J --> K[全部完成]
    K --> L[清除 resume 文件]

    style F fill:#ffcdd2
    style D fill:#c8e6c9
    style I fill:#e1f5ff
    style L fill:#c8e6c9
```

---

## 优化特性

| 特性 | 说明 |
|------|------|
| 目标导向 | `/target` 设置方向后，梯度分析和重写都以此为指导 |
| 层级优化 | Skill 树模式下 bottom-up：先叶子后父节点 |
| 自动拆分 | 反馈互相矛盾时建议拆分为子技能 |
| 自动剪枝 | 低性能子节点自动移除 |
| 策略选择 | 保守 / 激进 / 自适应三种策略 |
| 部分优化 | 可只优化 prompt 的指令、示例、或约束部分 |
| 断点续跑 | 中断后可从上次进度恢复 |
