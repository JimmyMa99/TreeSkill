"""Tests for tresskill.agenda — 基于 Skill 标准的长程日程模块。"""

from datetime import date, datetime, timedelta, timezone

import pytest

from tresskill.agenda import (
    AgendaManager,
    compile_agenda_context,
    parse_due,
)
from tresskill.schema import (
    AgendaEntry,
    AgendaType,
    Recurrence,
    Skill,
)


# ---------------------------------------------------------------------------
# AgendaEntry (schema) tests
# ---------------------------------------------------------------------------

class TestAgendaEntry:

    def test_create_reminder(self):
        entry = AgendaEntry(
            type=AgendaType.REMINDER,
            title="开会",
            due=datetime(2026, 3, 21, 15, 30, tzinfo=timezone.utc),
        )
        assert entry.type == AgendaType.REMINDER
        assert entry.recurrence == Recurrence.ONCE
        assert entry.active is True
        assert entry.done is False

    def test_create_recurring_yearly(self):
        entry = AgendaEntry(
            type=AgendaType.RECURRING,
            title="结婚纪念日",
            recurrence=Recurrence.YEARLY,
            month=3,
            day=14,
            origin_year=2021,
        )
        assert entry.month == 3
        assert entry.day == 14

    def test_matches_date_once(self):
        entry = AgendaEntry(
            type=AgendaType.REMINDER,
            title="提醒",
            due=datetime(2026, 3, 21, tzinfo=timezone.utc),
        )
        assert entry.matches_date(date(2026, 3, 21)) is True
        assert entry.matches_date(date(2026, 3, 22)) is False

    def test_matches_date_done_excluded(self):
        entry = AgendaEntry(
            type=AgendaType.REMINDER,
            title="已完成",
            due=datetime(2026, 3, 21, tzinfo=timezone.utc),
            done=True,
        )
        assert entry.matches_date(date(2026, 3, 21)) is False

    def test_matches_date_yearly(self):
        entry = AgendaEntry(
            type=AgendaType.RECURRING,
            title="生日",
            recurrence=Recurrence.YEARLY,
            month=6,
            day=15,
        )
        assert entry.matches_date(date(2026, 6, 15)) is True
        assert entry.matches_date(date(2027, 6, 15)) is True
        assert entry.matches_date(date(2026, 6, 16)) is False

    def test_matches_date_monthly(self):
        entry = AgendaEntry(
            type=AgendaType.RECURRING,
            title="月度总结",
            recurrence=Recurrence.MONTHLY,
            day=1,
        )
        assert entry.matches_date(date(2026, 3, 1)) is True
        assert entry.matches_date(date(2026, 3, 2)) is False

    def test_matches_date_weekly(self):
        entry = AgendaEntry(
            type=AgendaType.RECURRING,
            title="周会",
            recurrence=Recurrence.WEEKLY,
            weekday=0,  # Monday
        )
        # 2026-03-23 is Monday
        assert entry.matches_date(date(2026, 3, 23)) is True
        assert entry.matches_date(date(2026, 3, 24)) is False

    def test_matches_date_daily(self):
        entry = AgendaEntry(
            type=AgendaType.RECURRING,
            title="日报",
            recurrence=Recurrence.DAILY,
        )
        assert entry.matches_date(date(2026, 3, 21)) is True

    def test_matches_date_inactive(self):
        entry = AgendaEntry(
            type=AgendaType.RECURRING,
            title="已停用",
            recurrence=Recurrence.DAILY,
            active=False,
        )
        assert entry.matches_date(date(2026, 3, 21)) is False

    def test_display_info_anniversary(self):
        entry = AgendaEntry(
            type=AgendaType.RECURRING,
            title="结婚纪念日",
            recurrence=Recurrence.YEARLY,
            month=3,
            day=14,
            origin_year=2021,
        )
        info = entry.display_info(reference_date=date(2026, 3, 14))
        assert "第 5 年" in info

    def test_serialization_roundtrip(self):
        entry = AgendaEntry(
            type=AgendaType.DEADLINE,
            title="项目交付",
            due=datetime(2026, 4, 1, tzinfo=timezone.utc),
            tags=["work"],
        )
        json_str = entry.model_dump_json()
        restored = AgendaEntry.model_validate_json(json_str)
        assert restored.id == entry.id
        assert restored.title == entry.title


# ---------------------------------------------------------------------------
# Skill integration — agenda 存在 config.yaml 中
# ---------------------------------------------------------------------------

class TestSkillAgendaIntegration:

    def test_skill_with_agenda(self):
        """Skill 直接携带 agenda 列表。"""
        skill = Skill(
            name="test",
            system_prompt="Hello",
            agenda=[
                AgendaEntry(
                    type=AgendaType.RECURRING,
                    title="生日",
                    recurrence=Recurrence.YEARLY,
                    month=6,
                    day=15,
                ),
            ],
        )
        assert len(skill.agenda) == 1
        assert skill.agenda[0].title == "生日"

    def test_skill_save_load_with_agenda(self, tmp_path):
        """agenda 通过 config.yaml 持久化。"""
        from tresskill.skill import load, save

        skill = Skill(
            name="agenda-test",
            system_prompt="You are helpful.",
            agenda=[
                AgendaEntry(
                    type=AgendaType.RECURRING,
                    title="结婚纪念日",
                    recurrence=Recurrence.YEARLY,
                    month=3,
                    day=14,
                    origin_year=2021,
                ),
                AgendaEntry(
                    type=AgendaType.REMINDER,
                    title="开会",
                    due=datetime(2026, 3, 21, 15, 30, tzinfo=timezone.utc),
                ),
            ],
        )
        save(skill, tmp_path)

        # 验证 config.yaml 包含 agenda
        config_path = tmp_path / "config.yaml"
        assert config_path.is_file()
        config_text = config_path.read_text()
        assert "agenda" in config_text
        assert "结婚纪念日" in config_text

        # 重新加载
        loaded = load(tmp_path)
        assert len(loaded.agenda) == 2
        assert loaded.agenda[0].title == "结婚纪念日"
        assert loaded.agenda[0].recurrence == Recurrence.YEARLY
        assert loaded.agenda[1].title == "开会"

    def test_skill_without_agenda(self, tmp_path):
        """无 agenda 时 config.yaml 不含 agenda 字段。"""
        from tresskill.skill import load, save

        skill = Skill(name="no-agenda", system_prompt="Hello.")
        save(skill, tmp_path)

        loaded = load(tmp_path)
        assert loaded.agenda == []


# ---------------------------------------------------------------------------
# AgendaManager tests
# ---------------------------------------------------------------------------

class TestAgendaManager:

    def _make_skill(self) -> Skill:
        return Skill(name="test", system_prompt="Hello.")

    def test_add_reminder(self):
        mgr = AgendaManager(self._make_skill())
        entry = mgr.add_reminder("开会", due="2026-03-21T15:30:00")
        assert entry.type == AgendaType.REMINDER
        assert len(mgr.entries) == 1

    def test_add_reminder_relative(self):
        mgr = AgendaManager(self._make_skill())
        entry = mgr.add_reminder("30分钟后", due="+30m")
        now = datetime.now(timezone.utc)
        delta = entry.due - now
        assert 29 * 60 <= delta.total_seconds() <= 31 * 60

    def test_add_recurring(self):
        mgr = AgendaManager(self._make_skill())
        entry = mgr.add_recurring(
            "生日", recurrence="yearly", month=6, day=15,
        )
        assert entry.recurrence == Recurrence.YEARLY

    def test_add_deadline(self):
        mgr = AgendaManager(self._make_skill())
        entry = mgr.add_deadline("项目交付", due="2026-04-01")
        assert entry.type == AgendaType.DEADLINE

    def test_add_milestone(self):
        mgr = AgendaManager(self._make_skill())
        entry = mgr.add_milestone("v2.0 发布", achieved_date="2026-03-17")
        assert entry.done is True

    def test_due_today(self):
        mgr = AgendaManager(self._make_skill())
        target = date(2026, 3, 21)
        mgr.add_reminder("今天的", due="2026-03-21T10:00:00")
        mgr.add_reminder("明天的", due="2026-03-22T10:00:00")
        mgr.add_recurring("每天", recurrence="daily")

        events = mgr.due_today(today=target)
        assert len(events) == 2  # 今天的 + 每天

    def test_upcoming(self):
        mgr = AgendaManager(self._make_skill())
        target = date(2026, 3, 21)
        mgr.add_reminder("今天", due="2026-03-21T10:00:00")
        mgr.add_reminder("2天后", due="2026-03-23T10:00:00")
        mgr.add_reminder("10天后", due="2026-03-31T10:00:00")

        events = mgr.upcoming(today=target, lookahead_days=3)
        assert len(events) == 2

    def test_overdue(self):
        mgr = AgendaManager(self._make_skill())
        mgr.add_reminder("过期的", due="2026-03-19T10:00:00")
        mgr.add_reminder("未来的", due="2026-03-25T10:00:00")

        overdue = mgr.overdue(today=date(2026, 3, 21))
        assert len(overdue) == 1
        assert overdue[0].title == "过期的"

    def test_mark_done(self):
        mgr = AgendaManager(self._make_skill())
        entry = mgr.add_reminder("完成它", due="2026-03-21T10:00:00")

        assert mgr.mark_done(entry.id) is True
        assert mgr.mark_done("nonexistent") is False
        assert mgr.entries[0].done is True

    def test_deactivate(self):
        mgr = AgendaManager(self._make_skill())
        entry = mgr.add_reminder("停用", due="2026-03-21")
        mgr.deactivate(entry.id)
        assert len(mgr.active()) == 0

    def test_compact(self):
        mgr = AgendaManager(self._make_skill())
        mgr.add_reminder("保留", due="2026-03-21")
        e2 = mgr.add_reminder("删除", due="2026-03-22")
        mgr.deactivate(e2.id)

        removed = mgr.compact()
        assert removed == 1
        assert len(mgr.entries) == 1

    def test_query_by_type(self):
        mgr = AgendaManager(self._make_skill())
        mgr.add_reminder("提醒", due="2026-03-21")
        mgr.add_recurring("生日", recurrence="yearly", month=6, day=15)

        reminders = mgr.query(agenda_type=AgendaType.REMINDER)
        assert len(reminders) == 1

    def test_query_by_tags(self):
        mgr = AgendaManager(self._make_skill())
        mgr.add_reminder("工作会", due="2026-03-21", tags=["work"])
        mgr.add_reminder("朋友聚餐", due="2026-03-22", tags=["personal"])

        work = mgr.query(tags=["work"])
        assert len(work) == 1

    def test_stats(self):
        mgr = AgendaManager(self._make_skill())
        target = date(2026, 3, 21)
        mgr.add_reminder("今天", due="2026-03-21T10:00:00")
        mgr.add_recurring("每天", recurrence="daily")

        stats = mgr.stats(today=target)
        assert stats["total"] == 2
        assert stats["due_today"] == 2

    def test_skill_reference_updated(self):
        """Manager 直接修改 Skill.agenda，无需额外同步。"""
        skill = self._make_skill()
        mgr = AgendaManager(skill)
        mgr.add_reminder("测试", due="2026-03-21")

        # skill 对象本身已更新
        assert len(skill.agenda) == 1
        assert skill.agenda[0].title == "测试"


# ---------------------------------------------------------------------------
# compile_agenda_context tests
# ---------------------------------------------------------------------------

class TestCompileContext:

    def test_basic(self):
        agenda = [
            AgendaEntry(
                type=AgendaType.REMINDER,
                title="开会",
                due=datetime(2026, 3, 21, 15, 30, tzinfo=timezone.utc),
            ),
        ]
        ctx = compile_agenda_context(agenda, today=date(2026, 3, 21))
        assert "今日日程" in ctx
        assert "开会" in ctx

    def test_upcoming(self):
        agenda = [
            AgendaEntry(
                type=AgendaType.DEADLINE,
                title="项目截止",
                due=datetime(2026, 3, 23, tzinfo=timezone.utc),
            ),
        ]
        ctx = compile_agenda_context(agenda, today=date(2026, 3, 21))
        assert "即将到来" in ctx
        assert "项目截止" in ctx

    def test_overdue(self):
        agenda = [
            AgendaEntry(
                type=AgendaType.REMINDER,
                title="过期任务",
                due=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
            ),
        ]
        ctx = compile_agenda_context(agenda, today=date(2026, 3, 21))
        assert "已过期" in ctx
        assert "过期2天" in ctx

    def test_anniversary(self):
        agenda = [
            AgendaEntry(
                type=AgendaType.RECURRING,
                title="结婚纪念日",
                recurrence=Recurrence.YEARLY,
                month=3,
                day=14,
                origin_year=2021,
            ),
        ]
        ctx = compile_agenda_context(agenda, today=date(2026, 3, 14))
        assert "第 5 年" in ctx

    def test_milestones(self):
        agenda = [
            AgendaEntry(
                type=AgendaType.MILESTONE,
                title="v2.0 发布",
                due=datetime(2026, 3, 17, tzinfo=timezone.utc),
                done=True,
            ),
        ]
        ctx = compile_agenda_context(
            agenda, today=date(2026, 3, 21), include_milestones=True,
        )
        assert "里程碑" in ctx
        assert "v2.0" in ctx

    def test_empty(self):
        ctx = compile_agenda_context([], today=date(2026, 3, 21))
        assert ctx == ""

    def test_no_overdue_flag(self):
        agenda = [
            AgendaEntry(
                type=AgendaType.REMINDER,
                title="过期",
                due=datetime(2026, 3, 19, tzinfo=timezone.utc),
            ),
        ]
        ctx = compile_agenda_context(
            agenda, today=date(2026, 3, 21), include_overdue=False,
        )
        assert "已过期" not in ctx


# ---------------------------------------------------------------------------
# compile_messages + agenda integration
# ---------------------------------------------------------------------------

class TestCompileMessagesWithAgenda:

    def test_agenda_injected(self):
        from tresskill.schema import Message
        from tresskill.skill import compile_messages

        skill = Skill(
            name="test",
            system_prompt="You are helpful.",
            agenda=[
                AgendaEntry(
                    type=AgendaType.REMINDER,
                    title="开会",
                    due=datetime(2026, 3, 21, 15, 30, tzinfo=timezone.utc),
                ),
            ],
        )
        user_msgs = [Message(role="user", content="你好")]

        # 无 agenda 注入
        msgs_no_agenda = compile_messages(skill, user_msgs)
        assert "开会" not in msgs_no_agenda[0].content

        # 有 agenda 注入
        ctx = compile_agenda_context(skill.agenda, today=date(2026, 3, 21))
        msgs = compile_messages(skill, user_msgs, agenda_context=ctx)
        assert "开会" in msgs[0].content
        assert "You are helpful." in msgs[0].content


# ---------------------------------------------------------------------------
# parse_due tests
# ---------------------------------------------------------------------------

class TestParseDue:

    def test_iso_datetime(self):
        dt = parse_due("2026-03-21T15:30:00")
        assert dt.year == 2026
        assert dt.hour == 15

    def test_date_only(self):
        dt = parse_due("2026-03-21")
        assert dt.year == 2026
        assert dt.hour == 0

    def test_datetime_object(self):
        original = datetime(2026, 3, 21, tzinfo=timezone.utc)
        assert parse_due(original) == original

    def test_relative_minutes(self):
        now = datetime.now(timezone.utc)
        dt = parse_due("+30m")
        assert (dt - now).total_seconds() >= 29 * 60

    def test_relative_hours(self):
        now = datetime.now(timezone.utc)
        dt = parse_due("+2h")
        assert (dt - now).total_seconds() >= 119 * 60

    def test_relative_days(self):
        now = datetime.now(timezone.utc)
        dt = parse_due("+1d")
        delta = dt - now
        assert 23 * 3600 <= delta.total_seconds() <= 25 * 3600

    def test_relative_weeks(self):
        now = datetime.now(timezone.utc)
        dt = parse_due("+1w")
        delta = dt - now
        assert 6 * 86400 <= delta.total_seconds() <= 8 * 86400

    def test_invalid(self):
        with pytest.raises(ValueError, match="无法解析"):
            parse_due("not-a-date")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="未知时间单位"):
            parse_due("+5x")
