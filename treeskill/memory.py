"""Memory Module — 用户偏好记忆的持久化存储与训练信号转换。

用户在交互过程中表达的偏好、纠正、行为模式和上下文信息会被记录为
MemoryEntry，持久化到 ``memory.jsonl`` 文件。这些记忆既为实时对话提供
个性化上下文，也可作为 APO 优化器的训练信号——每条记忆都能转换为一条
合成 Trace，供 TGD 梯度计算使用。

存储格式::

    my-skill/
    ├── SKILL.md
    ├── config.yaml
    ├── script.py
    └── memory.jsonl    ← 用户记忆

每行一条 JSON，与 TraceStorage 格式对齐。

记忆类型:

- **preference**  — 用户风格偏好（如"不要用 emoji"、"回复要简短"）
- **correction**  — 用户对输出的显式纠正（附带 before/after 对比）
- **pattern**     — 从多次交互中提取的行为模式
- **context**     — 影响回复的用户背景信息（如"我是数据科学家"）

训练数据转换:

- preference  → 添加到 system prompt 约束
- correction  → 转为 Trace (feedback.correction)，作为 TGD 梯度
- pattern     → 生成合成 few-shot 示例
- context     → 注入 system prompt 上下文
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tresskill.schema import Feedback, Message, Trace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_FILE = "memory.jsonl"


# ---------------------------------------------------------------------------
# Memory Schema
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    """记忆类型枚举。"""

    PREFERENCE = "preference"
    CORRECTION = "correction"
    PATTERN = "pattern"
    CONTEXT = "context"


class MemoryEntry(BaseModel):
    """一条用户记忆。

    每条记忆都是结构化的，包含类型、内容、可选的来源追溯，
    以及权重（表示该记忆的重要程度）。
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType = Field(
        ..., description="记忆类型: preference, correction, pattern, context",
    )
    content: str = Field(
        ..., description="记忆内容，自然语言描述。",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="分类标签，用于检索和过滤（如 'style', 'tone', 'format'）。",
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="重要性权重。1.0=普通，>1=更重要，<1=次要。",
    )
    source_trace_id: Optional[str] = Field(
        default=None,
        description="来源 Trace ID，便于追溯记忆的产生上下文。",
    )
    # correction 类型专用字段
    before: Optional[str] = Field(
        default=None,
        description="纠正前的内容（correction 类型使用）。",
    )
    after: Optional[str] = Field(
        default=None,
        description="纠正后的内容（correction 类型使用）。",
    )
    # 元数据
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    active: bool = Field(
        default=True,
        description="是否生效。设为 False 表示软删除。",
    )


# ---------------------------------------------------------------------------
# Memory Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """用户记忆的持久化存储。

    基于 append-only JSONL 格式，与 TraceStorage 设计一致。

    Parameters
    ----------
    path : Path | str
        JSONL 文件路径。如果传入目录，自动使用 ``memory.jsonl``。
    """

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        if path.is_dir() or not path.suffix:
            path = path / MEMORY_FILE
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    # -- Write --

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        """追加一条记忆。返回同一对象（可能 id 已填充）。"""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        logger.info("记忆已添加: [%s] %s", entry.type.value, entry.content[:60])
        return entry

    def add_preference(
        self,
        content: str,
        *,
        tags: Optional[List[str]] = None,
        weight: float = 1.0,
        source_trace_id: Optional[str] = None,
    ) -> MemoryEntry:
        """快捷方法：添加一条偏好记忆。"""
        entry = MemoryEntry(
            type=MemoryType.PREFERENCE,
            content=content,
            tags=tags or [],
            weight=weight,
            source_trace_id=source_trace_id,
        )
        return self.add(entry)

    def add_correction(
        self,
        content: str,
        *,
        before: Optional[str] = None,
        after: Optional[str] = None,
        tags: Optional[List[str]] = None,
        weight: float = 1.5,
        source_trace_id: Optional[str] = None,
    ) -> MemoryEntry:
        """快捷方法：添加一条纠正记忆。"""
        entry = MemoryEntry(
            type=MemoryType.CORRECTION,
            content=content,
            before=before,
            after=after,
            tags=tags or [],
            weight=weight,
            source_trace_id=source_trace_id,
        )
        return self.add(entry)

    def add_pattern(
        self,
        content: str,
        *,
        tags: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> MemoryEntry:
        """快捷方法：添加一条行为模式记忆。"""
        entry = MemoryEntry(
            type=MemoryType.PATTERN,
            content=content,
            tags=tags or [],
            weight=weight,
        )
        return self.add(entry)

    def add_context(
        self,
        content: str,
        *,
        tags: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> MemoryEntry:
        """快捷方法：添加一条用户上下文记忆。"""
        entry = MemoryEntry(
            type=MemoryType.CONTEXT,
            content=content,
            tags=tags or [],
            weight=weight,
        )
        return self.add(entry)

    # -- Read --

    def load_all(self) -> List[MemoryEntry]:
        """加载所有记忆（包括 inactive）。"""
        if not self._path.exists():
            return []
        entries: List[MemoryEntry] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(MemoryEntry.model_validate_json(line))
        return entries

    def load_active(self) -> List[MemoryEntry]:
        """加载所有生效的记忆。"""
        return [e for e in self.load_all() if e.active]

    def query(
        self,
        *,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        min_weight: float = 0.0,
        active_only: bool = True,
    ) -> List[MemoryEntry]:
        """按条件查询记忆。

        Parameters
        ----------
        memory_type : MemoryType | None
            按类型过滤。
        tags : list[str] | None
            按标签过滤（任一匹配即可）。
        min_weight : float
            最小权重阈值。
        active_only : bool
            是否只返回生效的记忆。
        """
        entries = self.load_active() if active_only else self.load_all()

        if memory_type is not None:
            entries = [e for e in entries if e.type == memory_type]

        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if tag_set & set(e.tags)]

        if min_weight > 0:
            entries = [e for e in entries if e.weight >= min_weight]

        return entries

    # -- Update / Delete --

    def deactivate(self, entry_id: str) -> bool:
        """软删除：将指定记忆标记为 inactive。

        通过重写整个文件实现（JSONL 不支持原地更新）。
        """
        entries = self.load_all()
        found = False
        for e in entries:
            if e.id == entry_id:
                e.active = False
                e.updated_at = datetime.now(timezone.utc)
                found = True
                break

        if found:
            self._rewrite(entries)
            logger.info("记忆已停用: %s", entry_id)
        return found

    def update_weight(self, entry_id: str, new_weight: float) -> bool:
        """更新记忆权重。"""
        entries = self.load_all()
        found = False
        for e in entries:
            if e.id == entry_id:
                e.weight = max(0.0, min(10.0, new_weight))
                e.updated_at = datetime.now(timezone.utc)
                found = True
                break

        if found:
            self._rewrite(entries)
        return found

    def compact(self) -> int:
        """压缩：移除所有 inactive 记忆。返回移除数量。"""
        entries = self.load_all()
        active = [e for e in entries if e.active]
        removed = len(entries) - len(active)
        if removed > 0:
            self._rewrite(active)
            logger.info("压缩完成: 移除 %d 条 inactive 记忆", removed)
        return removed

    def _rewrite(self, entries: List[MemoryEntry]) -> None:
        """将完整的记忆列表重写到文件。"""
        with self._path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(entry.model_dump_json() + "\n")

    # -- Stats --

    def stats(self) -> Dict[str, Any]:
        """返回记忆统计信息。"""
        entries = self.load_all()
        active = [e for e in entries if e.active]
        by_type: Dict[str, int] = {}
        for e in active:
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1

        return {
            "total": len(entries),
            "active": len(active),
            "inactive": len(entries) - len(active),
            "by_type": by_type,
        }


# ---------------------------------------------------------------------------
# Training Data Conversion — 记忆 → 训练信号
# ---------------------------------------------------------------------------

class MemoryCompiler:
    """将用户记忆编译为各种训练信号格式。

    这是记忆模块与 APO 优化器的桥梁：
    - preference  → system prompt 约束文本
    - correction  → 合成 Trace (with feedback)
    - pattern     → few-shot Message 对
    - context     → system prompt 上下文段落
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    # -- 生成 system prompt 片段 --

    def compile_prompt_constraints(
        self,
        *,
        include_preferences: bool = True,
        include_context: bool = True,
        max_entries: int = 20,
    ) -> str:
        """将 preference 和 context 记忆编译为 system prompt 约束段落。

        输出格式::

            ## 用户偏好
            - 回复要简短，不超过3句
            - 不要使用emoji

            ## 用户背景
            - 用户是数据科学家，熟悉 Python
        """
        sections: List[str] = []

        if include_preferences:
            prefs = self._store.query(memory_type=MemoryType.PREFERENCE)
            # 按权重降序，取 top N
            prefs.sort(key=lambda e: e.weight, reverse=True)
            prefs = prefs[:max_entries]
            if prefs:
                lines = [f"- {p.content}" for p in prefs]
                sections.append("## 用户偏好\n" + "\n".join(lines))

        if include_context:
            ctxs = self._store.query(memory_type=MemoryType.CONTEXT)
            ctxs.sort(key=lambda e: e.weight, reverse=True)
            ctxs = ctxs[:max_entries]
            if ctxs:
                lines = [f"- {c.content}" for c in ctxs]
                sections.append("## 用户背景\n" + "\n".join(lines))

        return "\n\n".join(sections)

    # -- 生成合成 Trace --

    def compile_traces(self) -> List[Trace]:
        """将 correction 记忆转换为合成 Trace 列表，可直接喂给 APO 优化器。

        每条 correction 记忆变成一条 Trace:
        - inputs = [user message: correction.content]
        - prediction = assistant message: correction.before
        - feedback = Feedback(score=0.1, correction=correction.after)
        """
        corrections = self._store.query(memory_type=MemoryType.CORRECTION)
        traces: List[Trace] = []

        for mem in corrections:
            if not mem.before:
                continue

            user_msg = Message(role="user", content=mem.content)
            pred_msg = Message(role="assistant", content=mem.before)
            feedback = Feedback(
                score=0.1,
                critique=f"用户纠正 (记忆 {mem.id[:8]})",
                correction=mem.after,
            )

            trace = Trace(
                id=f"mem-{mem.id}",
                timestamp=mem.created_at,
                inputs=[user_msg],
                prediction=pred_msg,
                feedback=feedback,
            )
            traces.append(trace)

        return traces

    # -- 生成 few-shot 示例 --

    def compile_few_shots(self, max_pairs: int = 5) -> List[Message]:
        """将 pattern 记忆编译为 few-shot message 对。

        每条 pattern 记忆生成一个 user→assistant 示例对。
        pattern.content 格式约定::

            Q: 用户说了什么
            A: 应该怎么回

        如果不符合 Q/A 格式，则跳过。
        """
        patterns = self._store.query(memory_type=MemoryType.PATTERN)
        patterns.sort(key=lambda e: e.weight, reverse=True)
        patterns = patterns[:max_pairs]

        messages: List[Message] = []
        for pat in patterns:
            q, a = _parse_qa_pattern(pat.content)
            if q and a:
                messages.append(Message(role="user", content=q))
                messages.append(Message(role="assistant", content=a))

        return messages

    # -- 全部编译 --

    def compile_all(self) -> Dict[str, Any]:
        """一次性编译所有记忆为训练信号。

        Returns
        -------
        dict
            - ``prompt_constraints``: str — 可追加到 system prompt
            - ``synthetic_traces``: List[Trace] — 可喂给 APO
            - ``few_shot_messages``: List[Message] — 可追加到 few-shot
            - ``stats``: dict — 编译统计
        """
        prompt_constraints = self.compile_prompt_constraints()
        traces = self.compile_traces()
        few_shots = self.compile_few_shots()

        return {
            "prompt_constraints": prompt_constraints,
            "synthetic_traces": traces,
            "few_shot_messages": few_shots,
            "stats": {
                "constraint_chars": len(prompt_constraints),
                "synthetic_traces": len(traces),
                "few_shot_pairs": len(few_shots) // 2,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_qa_pattern(content: str) -> tuple[str, str]:
    """解析 Q/A 格式的 pattern 内容。

    支持格式::

        Q: question text
        A: answer text

    或::

        用户: question
        助手: answer
    """
    lines = content.strip().splitlines()
    q_lines: List[str] = []
    a_lines: List[str] = []
    current = None

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith(("q:", "用户:", "user:")):
            current = "q"
            # 去掉前缀
            for prefix in ("Q:", "q:", "用户:", "user:"):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):].strip()
                    break
            q_lines.append(stripped)
        elif lower.startswith(("a:", "助手:", "assistant:")):
            current = "a"
            for prefix in ("A:", "a:", "助手:", "assistant:"):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):].strip()
                    break
            a_lines.append(stripped)
        elif current == "q":
            q_lines.append(stripped)
        elif current == "a":
            a_lines.append(stripped)

    return "\n".join(q_lines).strip(), "\n".join(a_lines).strip()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "MEMORY_FILE",
    "MemoryType",
    "MemoryEntry",
    "MemoryStore",
    "MemoryCompiler",
]
