"""Agenda Module — 基于 Skill 标准的长程日程管理。

Agenda 数据存储在 Skill 的 ``config.yaml`` 中，作为 Skill 定义的一部分。
加载 Skill 时自动读取 agenda，无需额外文件。

config.yaml 格式::

    few_shot_messages: [...]

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

使用方式::

    from tresskill.skill import load, save
    from tresskill.agenda import AgendaManager, compile_agenda_context

    skill = load("./my-skill")          # agenda 自动加载到 skill.agenda
    mgr = AgendaManager(skill)          # 包装 skill.agenda 提供便捷操作
    mgr.add_recurring("生日", month=6, day=15, recurrence="yearly")
    save(mgr.skill, "./my-skill")       # 保存回 config.yaml

    # 编译上下文注入对话
    ctx = compile_agenda_context(skill.agenda)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from tresskill.schema import (
    AgendaEntry,
    AgendaType,
    Recurrence,
    Skill,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOOKAHEAD_DAYS = 3


# ---------------------------------------------------------------------------
# AgendaManager — 操作 Skill.agenda 的便捷包装
# ---------------------------------------------------------------------------

class AgendaManager:
    """操作 ``Skill.agenda`` 的便捷管理器。

    直接修改 Skill 对象的 agenda 列表，调用方通过
    ``skill.save()`` 持久化到 ``config.yaml``。

    Parameters
    ----------
    skill : Skill
        要管理日程的 Skill 对象。
    """

    def __init__(self, skill: Skill) -> None:
        self._skill = skill

    @property
    def skill(self) -> Skill:
        """返回底层 Skill 对象（agenda 已就地修改）。"""
        return self._skill

    @property
    def entries(self) -> List[AgendaEntry]:
        """所有日程条目。"""
        return self._skill.agenda

    # -- Add --

    def add(self, entry: AgendaEntry) -> AgendaEntry:
        """添加一条日程。"""
        self._skill.agenda.append(entry)
        logger.info("日程已添加: [%s] %s", entry.type.value, entry.title)
        return entry

    def add_reminder(
        self,
        title: str,
        due: Union[str, datetime],
        *,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> AgendaEntry:
        """添加一次性提醒。

        Parameters
        ----------
        due : str | datetime
            到期时间。支持 ISO 格式或相对时间 ("+30m", "+2h", "+1d")。
        """
        due_dt = parse_due(due)
        entry = AgendaEntry(
            type=AgendaType.REMINDER,
            title=title,
            description=description,
            tags=tags or [],
            due=due_dt,
            recurrence=Recurrence.ONCE,
            time_of_day=due_dt.strftime("%H:%M"),
        )
        return self.add(entry)

    def add_recurring(
        self,
        title: str,
        *,
        recurrence: Union[str, Recurrence] = Recurrence.YEARLY,
        month: Optional[int] = None,
        day: Optional[int] = None,
        weekday: Optional[int] = None,
        origin_year: Optional[int] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        time_of_day: Optional[str] = None,
    ) -> AgendaEntry:
        """添加周期性事件（纪念日、生日等）。"""
        if isinstance(recurrence, str):
            recurrence = Recurrence(recurrence)
        entry = AgendaEntry(
            type=AgendaType.RECURRING,
            title=title,
            description=description,
            tags=tags or [],
            recurrence=recurrence,
            month=month,
            day=day,
            weekday=weekday,
            origin_year=origin_year,
            time_of_day=time_of_day,
        )
        return self.add(entry)

    def add_deadline(
        self,
        title: str,
        due: Union[str, datetime],
        *,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> AgendaEntry:
        """添加截止日期。"""
        due_dt = parse_due(due)
        entry = AgendaEntry(
            type=AgendaType.DEADLINE,
            title=title,
            description=description,
            tags=tags or [],
            due=due_dt,
            recurrence=Recurrence.ONCE,
        )
        return self.add(entry)

    def add_milestone(
        self,
        title: str,
        *,
        achieved_date: Optional[Union[str, datetime]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> AgendaEntry:
        """添加里程碑。"""
        due_dt = (
            parse_due(achieved_date) if achieved_date
            else datetime.now(timezone.utc)
        )
        entry = AgendaEntry(
            type=AgendaType.MILESTONE,
            title=title,
            description=description,
            tags=tags or [],
            due=due_dt,
            recurrence=Recurrence.ONCE,
            done=True,
        )
        return self.add(entry)

    # -- Query --

    def active(self) -> List[AgendaEntry]:
        """生效的日程。"""
        return [e for e in self.entries if e.active]

    def due_today(self, *, today: Optional[date] = None) -> List[AgendaEntry]:
        """今天到期/匹配的事件。"""
        today = today or date.today()
        return [e for e in self.active() if e.matches_date(today)]

    def due_range(self, start: date, end: date) -> List[AgendaEntry]:
        """日期范围内的事件。"""
        results = []
        for e in self.active():
            current = start
            while current <= end:
                if e.matches_date(current):
                    results.append(e)
                    break
                current += timedelta(days=1)
        return results

    def upcoming(
        self,
        *,
        today: Optional[date] = None,
        lookahead_days: int = _DEFAULT_LOOKAHEAD_DAYS,
    ) -> List[AgendaEntry]:
        """即将到来的事件（今天 + 未来 N 天）。"""
        today = today or date.today()
        end = today + timedelta(days=lookahead_days)
        return self.due_range(today, end)

    def overdue(self, *, today: Optional[date] = None) -> List[AgendaEntry]:
        """已过期但未完成的一次性事件。"""
        today = today or date.today()
        return [
            e for e in self.active()
            if e.recurrence == Recurrence.ONCE
            and not e.done
            and e.due is not None
            and e.due.date() < today
        ]

    def query(
        self,
        *,
        agenda_type: Optional[AgendaType] = None,
        tags: Optional[List[str]] = None,
        include_done: bool = False,
    ) -> List[AgendaEntry]:
        """按条件查询。"""
        entries = self.active()

        if not include_done:
            entries = [
                e for e in entries
                if not e.done or e.recurrence != Recurrence.ONCE
            ]
        if agenda_type is not None:
            entries = [e for e in entries if e.type == agenda_type]
        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if tag_set & set(e.tags)]
        return entries

    # -- Update --

    def mark_done(self, entry_id: str) -> bool:
        """标记事件为已完成。"""
        for e in self.entries:
            if e.id == entry_id:
                e.done = True
                return True
        return False

    def deactivate(self, entry_id: str) -> bool:
        """软删除。"""
        for e in self.entries:
            if e.id == entry_id:
                e.active = False
                return True
        return False

    def compact(self) -> int:
        """移除 inactive 条目。"""
        before = len(self.entries)
        self._skill.agenda = [e for e in self.entries if e.active]
        return before - len(self.entries)

    # -- Stats --

    def stats(self, *, today: Optional[date] = None) -> Dict[str, Any]:
        today = today or date.today()
        active = self.active()
        by_type: Dict[str, int] = {}
        for e in active:
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
        return {
            "total": len(self.entries),
            "active": len(active),
            "due_today": len(self.due_today(today=today)),
            "upcoming_3d": len(self.upcoming(today=today, lookahead_days=3)),
            "overdue": len(self.overdue(today=today)),
            "by_type": by_type,
        }


# ---------------------------------------------------------------------------
# Context Compilation — 将 agenda 列表编译为 LLM 上下文
# ---------------------------------------------------------------------------

def compile_agenda_context(
    agenda: List[AgendaEntry],
    *,
    today: Optional[date] = None,
    lookahead_days: int = _DEFAULT_LOOKAHEAD_DAYS,
    include_overdue: bool = True,
    include_milestones: bool = False,
) -> str:
    """将 agenda 列表编译为可注入 system prompt 的上下文文本。

    这是纯函数，不依赖 Store/Manager，直接操作 ``Skill.agenda``。

    输出格式::

        ## 今日日程 (2026-03-21)
        - 结婚纪念日 (第 5 年)
        - 提醒开会 [15:30]

        ## 即将到来 (3天内)
        - 2026-03-23: 项目截止日期

        ## 已过期
        - 2026-03-19: 交报告 [已过期2天]
    """
    today = today or date.today()
    end = today + timedelta(days=lookahead_days)
    active = [e for e in agenda if e.active]
    sections: List[str] = []

    # 今日
    today_events = [e for e in active if e.matches_date(today)]
    if today_events:
        lines = [f"- {e.display_info(reference_date=today)}" for e in today_events]
        sections.append(f"## 今日日程 ({today.isoformat()})\n" + "\n".join(lines))

    # 即将到来（排除今天）
    tomorrow = today + timedelta(days=1)
    upcoming: List[str] = []
    for e in active:
        match_day = _find_next_match(e, tomorrow, end)
        if match_day:
            prefix = match_day.isoformat()
            upcoming.append(f"- {prefix}: {e.display_info(reference_date=today)}")
    if upcoming:
        sections.append(f"## 即将到来 ({lookahead_days}天内)\n" + "\n".join(upcoming))

    # 已过期
    if include_overdue:
        overdue_lines: List[str] = []
        for e in active:
            if (
                e.recurrence == Recurrence.ONCE
                and not e.done
                and e.due is not None
                and e.due.date() < today
            ):
                days_late = (today - e.due.date()).days
                overdue_lines.append(
                    f"- {e.due.date().isoformat()}: {e.title} [已过期{days_late}天]"
                )
        if overdue_lines:
            sections.append("## 已过期\n" + "\n".join(overdue_lines))

    # 里程碑
    if include_milestones:
        milestones = [
            e for e in active
            if e.type == AgendaType.MILESTONE and e.done
        ]
        if milestones:
            lines = [
                f"- {e.due.date().isoformat() if e.due else '?'}: {e.title}"
                for e in milestones[-5:]
            ]
            sections.append("## 里程碑\n" + "\n".join(lines))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_due(value: Union[str, datetime]) -> datetime:
    """解析到期时间。

    支持:
    - datetime 对象
    - ISO 格式: "2026-03-21T15:30:00"
    - 纯日期: "2026-03-21"
    - 相对时间: "+30m", "+2h", "+1d", "+1w"
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    value = value.strip()
    if value.startswith("+"):
        return _parse_relative(value)

    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    raise ValueError(
        f"无法解析时间: {value!r}。"
        f"支持: ISO 8601, YYYY-MM-DD, +30m/+2h/+1d/+1w"
    )


def _parse_relative(value: str) -> datetime:
    """解析相对时间: +30m, +2h, +1d, +1w。"""
    now = datetime.now(timezone.utc)
    raw = value.lstrip("+").strip()
    if not raw:
        raise ValueError("空的相对时间表达式")

    unit = raw[-1].lower()
    try:
        amount = int(raw[:-1])
    except ValueError:
        raise ValueError(f"无法解析相对时间: {value}")

    deltas = {"m": timedelta(minutes=amount), "h": timedelta(hours=amount),
              "d": timedelta(days=amount), "w": timedelta(weeks=amount)}
    if unit not in deltas:
        raise ValueError(f"未知时间单位: {unit}。支持: m, h, d, w")
    return now + deltas[unit]


def _find_next_match(
    entry: AgendaEntry, start: date, end: date,
) -> Optional[date]:
    """在 [start, end] 中找第一个匹配日。"""
    current = start
    while current <= end:
        if entry.matches_date(current):
            return current
        current += timedelta(days=1)
    return None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AgendaManager",
    "compile_agenda_context",
    "parse_due",
]
