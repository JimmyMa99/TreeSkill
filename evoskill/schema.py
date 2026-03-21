"""Unified Data Schema — the source of truth for all I/O types.

Supports OpenAI's "Content Parts" format (Text + Image) from day one,
ensuring multimodal compatibility across the entire framework.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content Parts — Discriminated Union for Multimodality
# ---------------------------------------------------------------------------

class TextContent(BaseModel):
    """A plain-text content part."""

    type: Literal["text"] = "text"
    text: str


class ImageURL(BaseModel):
    """Inner payload for an image content part."""

    url: str  # Regular URL or base64 data-URL


class ImageContent(BaseModel):
    """An image_url content part (for vision models)."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class InputAudio(BaseModel):
    """Inner payload for an audio content part."""

    data: str  # base64-encoded audio
    format: str = "wav"  # wav, mp3, etc.


class AudioContent(BaseModel):
    """An input_audio content part (for speech models).

    Follows OpenAI's ``input_audio`` content part format.
    Requires audio-capable models (e.g. gpt-4o-audio-preview).
    """

    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudio


# Discriminated union keyed on `type`
ContentPart = Union[TextContent, ImageContent, AudioContent]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single chat message, compatible with the OpenAI Chat API.

    `content` may be a simple string *or* a list of ContentPart objects
    to support multimodal conversations (text + images).
    """

    role: Literal["system", "user", "assistant", "function"]
    content: Union[str, List[ContentPart]]

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize to the dict format expected by the OpenAI Python SDK."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {
            "role": self.role,
            "content": [part.model_dump() for part in self.content],
        }


# ---------------------------------------------------------------------------
# Agenda — long-horizon event entries stored inside config.yaml
# ---------------------------------------------------------------------------

class AgendaType(str, Enum):
    """事件类型。"""
    REMINDER = "reminder"
    RECURRING = "recurring"
    DEADLINE = "deadline"
    MILESTONE = "milestone"


class Recurrence(str, Enum):
    """周期规则。"""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class AgendaEntry(BaseModel):
    """一条日程事件，存储在 config.yaml 的 ``agenda`` 列表中。

    Skill 加载时自动读取，对话开始时注入上下文。

    config.yaml 格式::

        agenda:
          - type: recurring
            title: 结婚纪念日
            recurrence: yearly
            month: 3
            day: 14
            origin_year: 2021
          - type: reminder
            title: 提醒开会
            due: "2026-03-21T15:30:00"
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: AgendaType = Field(..., description="事件类型")
    title: str = Field(..., description="事件标题")
    description: str = Field(default="", description="详细描述")
    tags: List[str] = Field(default_factory=list)

    # -- 时间 --
    due: Optional[datetime] = Field(
        default=None,
        description="到期时间（reminder/deadline 使用）",
    )
    month: Optional[int] = Field(default=None, ge=1, le=12)
    day: Optional[int] = Field(default=None, ge=1, le=31)
    weekday: Optional[int] = Field(
        default=None, ge=0, le=6,
        description="星期几 (0=Mon, 6=Sun)",
    )
    time_of_day: Optional[str] = Field(
        default=None, description="HH:MM 格式",
    )

    # -- 周期 --
    recurrence: Recurrence = Field(default=Recurrence.ONCE)

    # -- 状态 --
    done: bool = Field(default=False)
    active: bool = Field(default=True)

    # -- 元信息 --
    origin_year: Optional[int] = Field(
        default=None,
        description="起始年份（用于计算第 N 周年）",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def matches_date(self, target: date) -> bool:
        """判断此事件是否匹配给定日期。"""
        if not self.active:
            return False
        if self.done and self.recurrence == Recurrence.ONCE:
            return False

        if self.recurrence == Recurrence.ONCE:
            return self.due is not None and self.due.date() == target
        if self.recurrence == Recurrence.DAILY:
            return True
        if self.recurrence == Recurrence.WEEKLY:
            return self.weekday is not None and target.weekday() == self.weekday
        if self.recurrence == Recurrence.MONTHLY:
            return self.day is not None and target.day == self.day
        if self.recurrence == Recurrence.YEARLY:
            return (
                self.month is not None
                and self.day is not None
                and target.month == self.month
                and target.day == self.day
            )
        return False

    def display_info(self, reference_date: Optional[date] = None) -> str:
        """生成人类可读的事件描述。"""
        parts = [self.title]
        if self.recurrence == Recurrence.YEARLY and self.origin_year and reference_date:
            years = reference_date.year - self.origin_year
            if years > 0:
                parts.append(f"(第 {years} 年)")
        if self.due and self.recurrence == Recurrence.ONCE:
            parts.append(f"[{self.due.strftime('%Y-%m-%d %H:%M')}]")
        elif self.time_of_day:
            parts.append(f"[{self.time_of_day}]")
        if self.description:
            parts.append(f"— {self.description}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Tool Reference — 声明 Skill 使用的外部工具
# ---------------------------------------------------------------------------

class ToolRef(BaseModel):
    """Skill 级别的工具声明，存储在 config.yaml 的 ``tools`` 列表中。

    工具可以是三种来源之一:
    - **script** — 来自同目录的 script.py（自动推导，无需声明）
    - **http**   — 远程 HTTP API
    - **mcp**    — MCP 服务器上的工具

    config.yaml 格式::

        tools:
          - name: weather
            type: http
            endpoint: https://api.weather.com/current
            method: GET
            description: 获取天气信息

          - name: database
            type: mcp
            mcp_server: localhost:5000
            tool_name: query
            description: 查询数据库
    """

    name: str = Field(..., description="工具名称")
    type: str = Field(
        default="http",
        description="工具类型: http, mcp",
    )
    description: str = Field(default="", description="工具描述（供 LLM 理解）")

    # HTTP 工具字段
    endpoint: Optional[str] = Field(default=None, description="HTTP API 端点")
    method: str = Field(default="GET", description="HTTP 方法")
    headers: Dict[str, str] = Field(default_factory=dict)

    # MCP 工具字段
    mcp_server: Optional[str] = Field(default=None, description="MCP 服务器地址")
    tool_name: Optional[str] = Field(default=None, description="MCP 工具名")
    auth_token: Optional[str] = Field(default=None, description="MCP 认证 token")


# ---------------------------------------------------------------------------
# Skill — a versioned system-prompt container
# ---------------------------------------------------------------------------

class Skill(BaseModel):
    """A Skill following the Agent Skills standard (https://agentskills.io).

    On disk, a Skill is a directory containing a ``SKILL.md`` with YAML
    frontmatter (name, description, metadata) and a Markdown body that
    serves as the system prompt.  An optional ``config.yaml`` stores
    few-shot examples and model-level configuration.

    Field mapping::

        SKILL.md frontmatter.name        → name
        SKILL.md frontmatter.description → description
        SKILL.md frontmatter.metadata    → metadata (version, target, …)
        SKILL.md body                    → system_prompt
        config.yaml few_shot_messages    → few_shot_messages
        config.yaml (rest)               → config
    """

    name: str = Field(
        ...,
        description="Skill 名称，kebab-case，≤64字符，需与目录名一致。",
    )
    description: str = Field(
        default="",
        description="Skill 描述，≤1024字符。说明 skill 的用途和触发条件。",
    )
    version: str = "v1.0"
    system_prompt: str = Field(
        default="",
        description="SKILL.md Markdown body — 即 LLM 的 system prompt，也是 APO 优化目标。",
    )
    target: Optional[str] = Field(
        default=None,
        description="用户一句话优化方向，如'更像人'、'更简洁'。APO 优化时会参考此目标。",
    )
    few_shot_messages: List[Message] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    script: Optional[str] = Field(
        default=None,
        description="可选的 Python 脚本源码（来自 script.py），定义自定义工具函数。",
    )
    agenda: List[AgendaEntry] = Field(
        default_factory=list,
        description="长程日程列表，存储在 config.yaml 的 agenda 字段，加载时自动读取。",
    )
    tools: List[ToolRef] = Field(
        default_factory=list,
        description="Skill 声明的外部工具列表，存储在 config.yaml 的 tools 字段。",
    )


# ---------------------------------------------------------------------------
# Skill Meta — metadata for skill-tree directories
# ---------------------------------------------------------------------------

class SkillMeta(BaseModel):
    """Metadata for a skill-tree node (directory-level ``_meta.yaml``).

    Each sub-directory in a skill tree may contain a ``_meta.yaml`` that
    stores the group name, a human-readable description, and the creation
    timestamp.
    """

    name: str
    description: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class Feedback(BaseModel):
    """Human (or auto-judge) feedback on a single interaction."""

    score: float = Field(..., ge=0.0, le=1.0)
    critique: Optional[str] = None
    correction: Optional[str] = None


# ---------------------------------------------------------------------------
# Trace — the atomic unit of storage
# ---------------------------------------------------------------------------

class Trace(BaseModel):
    """An immutable record of one agent interaction.

    Stores the full conversation context (`inputs`), the agent's
    `prediction`, and optional `feedback` used by the APO optimizer.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    inputs: List[Message]
    prediction: Message
    feedback: Optional[Feedback] = None
    node_path: Optional[str] = Field(
        default=None,
        description="Dot-separated path of the SkillNode that handled this trace "
        "(e.g. 'social.moments'). Used by APO to route traces to the correct node.",
    )
