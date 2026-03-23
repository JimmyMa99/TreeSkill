# TreeSkill 核心架构理解

> 本文档面向下一任开发者，记录框架的设计决策、核心数据流、各模块职责与扩展点。

---

## 1. 设计哲学

框架的核心类比是 **"训练神经网络"**，但操作对象是 System Prompt 而非模型权重：

| 深度学习概念 | TreeSkill 对应 |
|-------------|-------------------|
| 模型权重 | System Prompt（`Skill.system_prompt`） |
| 训练数据 | 交互历史（`Trace` 对象，存在 JSONL 中；`trace.id` 标识单条交互，`session_id` 标识同一运行会话） |
| Loss / 梯度 | 用户反馈（`Feedback.critique`）→ Judge 模型的失败分析 |
| 反向传播 | Judge 模型分析 → 重写 prompt |
| 学习率 / batch size | `APOConfig.gradient_accumulation_steps` |
| 优化目标 | `Skill.target`（用户一句话方向） |
| 模型结构 | **Skill 树**（层级目录，子技能可自动拆分/合并） |
| Checkpoint | `ckpt/xxx/skill/*` + `ckpt/xxx/mem/` |

## 2. 核心数据流

```
User Input
    │
    ▼
compile_messages(skill, history)    ← skill.py
    │  拼接: [system_prompt] + [few_shot] + [history]
    ▼
LLMClient.generate(messages)        ← llm.py
    │  调用 OpenAI 兼容 API
    ▼
Trace(id, session_id, inputs, prediction) ← schema.py
    │  记录本次交互；`session_id` 用来把同一运行会话的多条 Trace 关联起来
    ▼
TraceStorage.append(trace)          ← storage.py
    │  追加写入 JSONL
    ▼
[用户反馈: /bad, /rewrite, /target]
    │
    ▼
APOEngine.optimize(skill, traces)   ← optimizer.py
    │  Step 1: 筛选有 feedback 的 traces
    │  Step 2: _compute_gradient() — 让 judge 分析为什么 prompt 失败
    │  Step 3: _apply_update() — 让 judge 重写 prompt
    │  Step 4: 版本号 +1，返回新 Skill
    ▼
CheckpointManager.save()            ← checkpoint.py
    │  保存 ckpt/xxx/skill/* + ckpt/xxx/mem/
    ▼
skill_module.save(new_skill)        ← skill.py（allow_unicode=True）
```

### 2.1 数据集驱动模式数据流

```
Dataset (ChatML JSONL)                ← dataset.py
    │
    ├──► --optimize --dataset         ← evaluator.py (全自动)
    │        auto-judge 评分 → Traces → APO 优化
    │
    └──► --annotate --dataset         ← annotate.py (人机协作)
             逐条展示 → auto/human judge → Traces
                  │                         │
                  ├──► APO 优化             └──► export_dpo() → DPO 微调
```

### 2.2 树模式数据流（额外步骤）

```
SkillTree.load(directory)           ← skill_tree.py
    │  递归加载文件夹 → SkillNode 树
    ▼
APOEngine.evolve_tree(tree, traces)
    │  底向上优化：先叶子后父节点
    │  每个节点: optimize() + analyze_split_need()
    │  如需拆分: generate_child_prompts() → 新建子节点
    ▼
SkillTree.save(directory)
    │  递归保存到文件夹（同时清理已剪枝的子目录）
```

## 3. 各模块职责

### schema.py — 数据模型
- **关键设计**: `Message.content` 是 `Union[str, List[ContentPart]]`，从第一天就支持多模态
- `ContentPart` 是以 `type` 字段做鉴别的联合体（TextContent | ImageContent）
- `Message.to_api_dict()` 负责序列化为 OpenAI API 格式
- `Skill` — 版本化的 system-prompt 容器，含 `target`（优化方向）、`few_shot_messages`、`config`
- **`SkillMeta`** — Skill 树目录级元数据（`_meta.yaml`），含 `name`、`description`、`created_at`

### config.py — 配置系统
- 使用 `pydantic-settings`，支持 `.env` 文件和 `TREE_` 前缀环境变量
- `.env` / 环境变量使用扁平键（如 `TREE_LLM_MODEL`）
- **支持 YAML 配置文件**：`GlobalConfig.from_yaml(path)` 从 YAML 加载，`--config` 传入
- **优先级**：环境变量 > `.env` > YAML 文件 > 默认值
- `GlobalConfig` 聚合 `LLMConfig`、`StorageConfig`、`APOConfig`、**`RewardConfig`**
- **`RewardConfig`** — 自动 Judge 配置：
  - `enabled` / `auto_judge` — 开关
  - `model` / `base_url` / `api_key` — 独立配置，为空则回落到 `LLMConfig`
  - `default_rubric` — 默认评分标准 prompt（Skill 级 `config.judge_rubric` 可覆盖）
- **完整配置模版**: `demo/example/config.yaml`（每个字段都有中文注释）

### registry.py — 组件注册表
- 线程安全单例模式（`__new__` + threading.Lock）
- 预留给将来插件化（自定义优化器、存储后端等）

### llm.py — LLM 客户端
- 薄封装层，核心是 `generate()` 方法
- model 参数可按调用覆盖（APO 用 judge_model，聊天用 chat model）

### storage.py — Trace 存储
- 追加写入 JSONL，每行一个 JSON 对象
- `Trace.id` 仍然是单条交互的唯一标识；`session_id` 用来把同一次交互会话中的多条 Trace 归组
- `get_feedback_samples(min_score, max_score)` 筛选"差评"样本给 APO
- `get_dpo_pairs()` 提取有 correction 的 traces 为 DPO 偏好对
- `export_dpo(output_path)` 导出 DPO JSONL（prompt/chosen/rejected）

### dataset.py — 数据集加载
- 加载 ChatML JSONL 格式数据集（OpenAI fine-tuning 格式）
- 最后一条 assistant message 作为 ground truth

### evaluator.py — 自动评估
- 对数据集批量跑模型预测 + LM judge 评分
- 输出 `List[Trace]` 供 APO 使用

### annotate.py — 人机协作标注
- 数据集驱动的标注模式（`--annotate --dataset`）
- auto / manual 模式可运行时切换（`/auto`、`/manual`）
- 人工反馈作为偏好锚点，同时服务 APO 梯度计算和 DPO 数据导出

### skill.py — Skill 管理
- `load/save` 支持 YAML 和 JSON
- `save()` 使用 `allow_unicode=True`，中文直接可读（不再转义为 `\uXXXX`）
- `compile_messages()` 的拼接顺序: `[system] + [few_shot] + [user_input]`

### skill_tree.py — 层级 Skill 树
- **核心理念**: 文件夹即层级，目录结构表达树形关系
- `SkillNode` 数据类 — 树节点，包含 `skill`（root.yaml）+ `children` 字典 + `meta`（_meta.yaml）
- `SkillTree` 类 — 管理整棵树的生命周期：

| 方法 | 说明 |
|------|------|
| `load(path)` | 递归加载目录 → SkillNode 树。也兼容单个 YAML 文件（包装为单节点树） |
| `save(path)` | 递归保存到目录，自动清理已剪枝的子目录 |
| `get(dotpath)` | 用点分路径获取节点，如 `social.moments` |
| `list_tree()` | 返回 pretty-print 的树形字符串 |
| `add_child()` | 在指定父节点下添加子技能 |
| `split(path, specs)` | 将节点拆分为多个子节点 |
| `merge(paths, name, prompt)` | 合并多个同级节点 |
| `prune(path)` | 删除叶节点，职责归还父节点 |

**磁盘结构示例**:
```
my-skills/
├── _meta.yaml          # SkillMeta（包名、描述）
├── root.yaml           # 根 Skill
├── social/
│   ├── _meta.yaml
│   ├── root.yaml       # 社交类 Skill
│   └── moments.yaml
└── business/
    ├── root.yaml
    └── email.yaml
```

### checkpoint.py — Checkpoint 管理
- **核心理念**: 每次优化后自动快照，可随时恢复到任意中间状态
- `CheckpointManager` 类：

| 方法 | 说明 |
|------|------|
| `save(skill_source, trace_path, name)` | 保存 `ckpt/xxx/skill/*` + `ckpt/xxx/mem/`。skill_source 可以是 Skill 对象、单文件、或目录（树） |
| `load(ckpt_path)` | 读取 checkpoint 元信息，返回 skill_path + trace_path + meta |
| `restore_to(ckpt_path, dest)` | 复制 checkpoint 内容到工作目录 |
| `list_checkpoints()` | 列出所有 checkpoint（按时间倒序） |

**磁盘结构**:
```
ckpt/
└── writing-assistant_v1.2_20260306_140000/
    ├── skill/              # 完整 skill（单文件或整棵树）
    │   └── root.yaml
    └── mem/
        ├── traces.jsonl    # 交互历史（每行一个 Trace；同一 session_id 可能对应多行）
        └── meta.json       # 优化轮次、版本、时间等
```

### optimizer.py — APO 优化引擎（对齐 Agent-Lightning）
- **搜索策略**：支持单轨模式（beam_width=1）和 Beam Search（beam_width>1）
- **核心算法**: 多模板 × 多候选 × 评分选择
  1. `_compute_gradient()`: 随机选 3 种梯度模板之一，让 judge 分析失败原因
  2. `_build_edit_messages()`: 随机选 2 种编辑模板之一（激进重写 / 保守修复），生成候选
  3. `_score_prompts_batch()`: 并行评分所有候选，选最佳或保留 top-k beam
- **target 融入方式**: 如果 `skill.target` 不为空，在梯度和编辑环节的 system prompt 中追加方向提示
- **节点级 trace 路由**: `Trace.node_path` 确保每个节点只用属于自己的数据优化
- 版本号自动递增: `v1.0` → `v1.1` → `v1.2`

**树感知扩展**：
- `analyze_split_need(skill, traces)` → 让 LLM 判断反馈是否跨领域，建议拆分方案
- `generate_child_prompts(parent, specs)` → 为每个子技能生成专属 System Prompt
- `evolve_tree(tree, traces)` → 递归底向上优化整棵树（含 rich 进度条 + ETA）
- `_evolve_node()` → 内部递归辅助，按 node_path 过滤 traces

### cli.py — 终端界面
- 基于 Rich 库，支持 Markdown 渲染
- 构造时可选传入 `skill_tree` 和 `ckpt_dir`
- 命令分发在 `_handle_command()` 中

**完整命令表**:

| 命令 | 处理方法 | 说明 |
|------|----------|------|
| `/image <path>` | `_cmd_image` | 附加图片 |
| `/bad <reason>` | `_cmd_bad` | 标记不好 |
| `/rewrite <text>` | `_cmd_rewrite` | 提供理想回答（同时积累 DPO 数据） |
| `/export-dpo <path>` | `_cmd_export_dpo` | 导出 DPO 偏好数据 |
| `/target <text>` | `_cmd_target` | 设置优化方向 |
| `/save` | `_cmd_save` | 保存 skill |
| `/optimize` | `_cmd_optimize` | APO 优化 + 自动 checkpoint |
| `/tree` | `_cmd_tree` | 显示技能树 |
| `/select <path>` | `_cmd_select` | 切换活动技能节点 |
| `/split` | `_cmd_split` | 分析并拆分（需用户确认） |
| `/ckpt` | `_cmd_ckpt` | 列出 checkpoint |
| `/restore <name>` | `_cmd_restore` | 恢复 checkpoint |
| `/quit` | — | 退出 |

### main.py — 入口点
- `--config <path>` — YAML 配置文件路径（见 `demo/example/config.yaml`）
- `--skill <name-or-path>` — 支持 YAML 文件**和**目录（自动判断是否加载为树）
- `--optimize` — 批量优化模式（支持树）
- `--annotate` — 数据集驱动的人机协作标注模式（需配合 `--dataset`）
- `--manual` — 标注模式下使用纯手动 judge（默认 auto）
- `--dataset <path>` — ChatML JSONL 数据集路径，配合 `--optimize` 或 `--annotate` 使用
- `--no-resume` — 跳过断点续跑提示，直接重新开始（适用于非交互环境）
- `--ckpt <path>` — 从 checkpoint 恢复并继续
- `--ckpt-dir <dir>` — 指定 checkpoint 存储目录（默认 `./ckpt`）
- `-v / --verbose` — 启用 DEBUG 日志

## 4. 扩展点

### 添加新命令
在 `cli.py` 的 `_handle_command()` 中加分支 + 实现 `_cmd_xxx()` 方法。

### 自定义优化策略
1. 实现新的优化器类（参考 `APOEngine` 的接口）
2. 用 `@registry.register("optimizer", "your_name")` 注册
3. 在 main.py 中根据配置选择优化器

### 自定义拆分/合并策略
修改 `optimizer.py` 中 `analyze_split_need()` 的 prompt 来改变拆分判定逻辑。

### 多模态优化
当前 APO `_compute_gradient()` 中对多模态响应用 `[multimodal response]` 占位。如需支持图片反馈，需在此处展开 ContentPart 的具体内容。

### 存储后端替换
当前是简单 JSONL 文件。如需换为数据库：
1. 实现与 `TraceStorage` 相同接口的新类
2. 在 registry 中注册
3. 通过配置切换

## 5. 待办事项

详见 [TODO.md](../TODO.md)。
- **Skill 树路由**: 当前需用户手动 `/select` 切换子技能。可考虑根据用户输入自动匹配最合适的叶子节点。
- **拆分/合并门槛**: 自动拆分/合并的判断完全依赖 LLM，可考虑加入基于反馈统计的硬性门槛。
