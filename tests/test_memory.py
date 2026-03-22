"""Tests for tresskill.memory — 用户记忆模块。"""

from pathlib import Path

import pytest

from tresskill.memory import (
    MEMORY_FILE,
    MemoryCompiler,
    MemoryEntry,
    MemoryStore,
    MemoryType,
    _parse_qa_pattern,
)


# ---------------------------------------------------------------------------
# MemoryEntry tests
# ---------------------------------------------------------------------------

class TestMemoryEntry:

    def test_create_preference(self):
        entry = MemoryEntry(
            type=MemoryType.PREFERENCE,
            content="回复要简短，不超过3句",
            tags=["style", "length"],
        )
        assert entry.type == MemoryType.PREFERENCE
        assert entry.active is True
        assert entry.weight == 1.0
        assert len(entry.id) > 0

    def test_create_correction(self):
        entry = MemoryEntry(
            type=MemoryType.CORRECTION,
            content="语气太生硬",
            before="请按照要求完成。",
            after="好的，我来帮你处理这个～",
            weight=1.5,
        )
        assert entry.before is not None
        assert entry.after is not None

    def test_serialization_roundtrip(self):
        entry = MemoryEntry(
            type=MemoryType.CONTEXT,
            content="用户是 Python 开发者",
            tags=["role"],
        )
        json_str = entry.model_dump_json()
        restored = MemoryEntry.model_validate_json(json_str)
        assert restored.id == entry.id
        assert restored.type == entry.type
        assert restored.content == entry.content


# ---------------------------------------------------------------------------
# MemoryStore tests
# ---------------------------------------------------------------------------

class TestMemoryStore:

    def test_init_with_dir(self, tmp_path):
        store = MemoryStore(tmp_path)
        assert store.path == tmp_path / MEMORY_FILE

    def test_init_with_file(self, tmp_path):
        file_path = tmp_path / "custom.jsonl"
        store = MemoryStore(file_path)
        assert store.path == file_path

    def test_add_and_load(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.add_preference("不要用 emoji", tags=["style"])
        store.add_preference("回复要简短")

        entries = store.load_all()
        assert len(entries) == 2
        assert entries[0].type == MemoryType.PREFERENCE

    def test_add_correction(self, tmp_path):
        store = MemoryStore(tmp_path)
        entry = store.add_correction(
            "语气问题",
            before="做完了。",
            after="已经帮你完成啦！",
        )
        assert entry.weight == 1.5  # correction 默认权重更高

        loaded = store.load_all()
        assert loaded[0].before == "做完了。"
        assert loaded[0].after == "已经帮你完成啦！"

    def test_add_pattern(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.add_pattern(
            "Q: 你好\nA: 你好呀！有什么可以帮你的？",
            tags=["greeting"],
        )
        entries = store.load_all()
        assert entries[0].type == MemoryType.PATTERN

    def test_add_context(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.add_context("用户是数据科学家", tags=["role"])
        entries = store.load_all()
        assert entries[0].type == MemoryType.CONTEXT

    def test_load_active(self, tmp_path):
        store = MemoryStore(tmp_path)
        e1 = store.add_preference("偏好A")
        e2 = store.add_preference("偏好B")

        store.deactivate(e1.id)

        active = store.load_active()
        assert len(active) == 1
        assert active[0].id == e2.id

    def test_query_by_type(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.add_preference("偏好")
        store.add_context("上下文")
        store.add_correction("纠正", before="a", after="b")

        prefs = store.query(memory_type=MemoryType.PREFERENCE)
        assert len(prefs) == 1
        assert prefs[0].content == "偏好"

    def test_query_by_tags(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.add_preference("简短", tags=["style"])
        store.add_preference("正式", tags=["tone"])
        store.add_preference("用中文", tags=["language", "style"])

        results = store.query(tags=["style"])
        assert len(results) == 2

    def test_query_by_weight(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.add_preference("普通", weight=1.0)
        store.add_preference("重要", weight=3.0)
        store.add_preference("次要", weight=0.5)

        results = store.query(min_weight=2.0)
        assert len(results) == 1
        assert results[0].content == "重要"

    def test_deactivate(self, tmp_path):
        store = MemoryStore(tmp_path)
        entry = store.add_preference("要删除的")

        assert store.deactivate(entry.id) is True
        assert store.deactivate("nonexistent") is False

        all_entries = store.load_all()
        assert len(all_entries) == 1
        assert all_entries[0].active is False

    def test_update_weight(self, tmp_path):
        store = MemoryStore(tmp_path)
        entry = store.add_preference("测试权重")

        store.update_weight(entry.id, 5.0)

        loaded = store.load_all()
        assert loaded[0].weight == 5.0

    def test_compact(self, tmp_path):
        store = MemoryStore(tmp_path)
        e1 = store.add_preference("保留")
        e2 = store.add_preference("删除")
        store.deactivate(e2.id)

        removed = store.compact()
        assert removed == 1

        entries = store.load_all()
        assert len(entries) == 1
        assert entries[0].content == "保留"

    def test_stats(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.add_preference("偏好1")
        store.add_preference("偏好2")
        store.add_context("上下文")
        e = store.add_correction("纠正", before="a", after="b")
        store.deactivate(e.id)

        stats = store.stats()
        assert stats["total"] == 4
        assert stats["active"] == 3
        assert stats["inactive"] == 1
        assert stats["by_type"]["preference"] == 2
        assert stats["by_type"]["context"] == 1

    def test_empty_store(self, tmp_path):
        store = MemoryStore(tmp_path)
        assert store.load_all() == []
        assert store.load_active() == []
        assert store.stats()["total"] == 0


# ---------------------------------------------------------------------------
# MemoryCompiler tests
# ---------------------------------------------------------------------------

class TestMemoryCompiler:

    def _make_store(self, tmp_path) -> MemoryStore:
        store = MemoryStore(tmp_path)
        store.add_preference("回复要简短，不超过3句", tags=["style"])
        store.add_preference("不要使用 emoji", tags=["style"])
        store.add_context("用户是数据科学家", tags=["role"])
        store.add_context("熟悉 Python 和机器学习", tags=["skill"])
        store.add_correction(
            "语气太生硬",
            before="完成。",
            after="已经帮你完成啦！还有其他需要吗？",
        )
        store.add_pattern(
            "Q: 你好\nA: 你好呀！有什么可以帮你的？",
            tags=["greeting"],
        )
        return store

    def test_compile_prompt_constraints(self, tmp_path):
        store = self._make_store(tmp_path)
        compiler = MemoryCompiler(store)

        text = compiler.compile_prompt_constraints()
        assert "用户偏好" in text
        assert "不要使用 emoji" in text
        assert "用户背景" in text
        assert "数据科学家" in text

    def test_compile_traces(self, tmp_path):
        store = self._make_store(tmp_path)
        compiler = MemoryCompiler(store)

        traces = compiler.compile_traces()
        assert len(traces) == 1
        assert traces[0].feedback is not None
        assert traces[0].feedback.correction == "已经帮你完成啦！还有其他需要吗？"
        assert traces[0].prediction.content == "完成。"
        assert traces[0].id.startswith("mem-")

    def test_compile_few_shots(self, tmp_path):
        store = self._make_store(tmp_path)
        compiler = MemoryCompiler(store)

        messages = compiler.compile_few_shots()
        assert len(messages) == 2  # one Q/A pair
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert "你好" in messages[0].content

    def test_compile_all(self, tmp_path):
        store = self._make_store(tmp_path)
        compiler = MemoryCompiler(store)

        result = compiler.compile_all()
        assert len(result["prompt_constraints"]) > 0
        assert len(result["synthetic_traces"]) == 1
        assert len(result["few_shot_messages"]) == 2
        assert result["stats"]["synthetic_traces"] == 1
        assert result["stats"]["few_shot_pairs"] == 1

    def test_compile_empty_store(self, tmp_path):
        store = MemoryStore(tmp_path)
        compiler = MemoryCompiler(store)

        result = compiler.compile_all()
        assert result["prompt_constraints"] == ""
        assert result["synthetic_traces"] == []
        assert result["few_shot_messages"] == []

    def test_correction_without_before_skipped(self, tmp_path):
        store = MemoryStore(tmp_path)
        # 没有 before 的 correction 不生成 trace
        store.add_correction("语气问题", after="好的！")

        compiler = MemoryCompiler(store)
        traces = compiler.compile_traces()
        assert len(traces) == 0

    def test_preference_only(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.add_preference("回复用中文")

        compiler = MemoryCompiler(store)
        text = compiler.compile_prompt_constraints(
            include_context=False,
        )
        assert "用户偏好" in text
        assert "用户背景" not in text


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestParseQAPattern:

    def test_standard_format(self):
        q, a = _parse_qa_pattern("Q: 你好\nA: 你好呀！")
        assert q == "你好"
        assert a == "你好呀！"

    def test_chinese_format(self):
        q, a = _parse_qa_pattern("用户: 帮我写代码\n助手: 好的，写什么？")
        assert q == "帮我写代码"
        assert a == "好的，写什么？"

    def test_multiline(self):
        text = "Q: 第一行\n第二行\nA: 回答第一行\n回答第二行"
        q, a = _parse_qa_pattern(text)
        assert "第一行" in q
        assert "第二行" in q
        assert "回答第一行" in a

    def test_no_qa_format(self):
        q, a = _parse_qa_pattern("这不是QA格式")
        assert q == ""
        assert a == ""

    def test_english_format(self):
        q, a = _parse_qa_pattern("user: hello\nassistant: hi there!")
        assert q == "hello"
        assert a == "hi there!"
