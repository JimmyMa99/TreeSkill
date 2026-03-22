"""Tests for skill grafting and skill-scoped tool registration."""

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tresskill.schema import Skill, ToolRef, AgendaEntry, AgendaType, Recurrence
from tresskill.skill import load, save
from tresskill.skill_tree import SkillTree, SkillNode, resolve_skill_tools


# ---------------------------------------------------------------------------
# ToolRef schema tests
# ---------------------------------------------------------------------------

class TestToolRef:

    def test_http_tool_ref(self):
        ref = ToolRef(
            name="weather",
            type="http",
            endpoint="https://api.weather.com/current",
            method="GET",
            description="获取天气",
        )
        assert ref.name == "weather"
        assert ref.type == "http"

    def test_mcp_tool_ref(self):
        ref = ToolRef(
            name="database",
            type="mcp",
            mcp_server="localhost:5000",
            tool_name="query",
            description="查询数据库",
        )
        assert ref.mcp_server == "localhost:5000"

    def test_serialization_roundtrip(self):
        ref = ToolRef(
            name="test",
            type="http",
            endpoint="https://example.com",
        )
        json_str = ref.model_dump_json()
        restored = ToolRef.model_validate_json(json_str)
        assert restored.name == ref.name


# ---------------------------------------------------------------------------
# Skill with tools — config.yaml integration
# ---------------------------------------------------------------------------

class TestSkillTools:

    def test_skill_with_tools(self):
        skill = Skill(
            name="with-tools",
            system_prompt="Hello.",
            tools=[
                ToolRef(
                    name="weather",
                    type="http",
                    endpoint="https://api.weather.com",
                    description="天气查询",
                ),
            ],
        )
        assert len(skill.tools) == 1

    def test_skill_save_load_with_tools(self, tmp_path):
        skill = Skill(
            name="tool-test",
            system_prompt="You are helpful.",
            tools=[
                ToolRef(
                    name="weather",
                    type="http",
                    endpoint="https://api.weather.com",
                    description="天气查询",
                ),
                ToolRef(
                    name="db",
                    type="mcp",
                    mcp_server="localhost:5000",
                    tool_name="query",
                    description="数据库",
                ),
            ],
        )
        save(skill, tmp_path)

        config_text = (tmp_path / "config.yaml").read_text()
        assert "tools" in config_text
        assert "weather" in config_text

        loaded = load(tmp_path)
        assert len(loaded.tools) == 2
        assert loaded.tools[0].name == "weather"
        assert loaded.tools[1].name == "db"

    def test_skill_without_tools(self, tmp_path):
        skill = Skill(name="no-tools", system_prompt="Hello.")
        save(skill, tmp_path)
        loaded = load(tmp_path)
        assert loaded.tools == []


# ---------------------------------------------------------------------------
# resolve_skill_tools — ToolRef → BaseTool
# ---------------------------------------------------------------------------

class TestResolveSkillTools:

    def test_resolve_http_tool(self):
        skill = Skill(
            name="test",
            system_prompt="Hello.",
            tools=[
                ToolRef(
                    name="weather",
                    type="http",
                    endpoint="https://api.weather.com",
                    description="天气",
                ),
            ],
        )
        tools = resolve_skill_tools(skill)
        assert "weather" in tools
        assert tools["weather"].name == "weather"
        assert tools["weather"].description == "天气"

    def test_resolve_mcp_tool(self):
        skill = Skill(
            name="test",
            system_prompt="Hello.",
            tools=[
                ToolRef(
                    name="db",
                    type="mcp",
                    mcp_server="localhost:5000",
                    tool_name="query",
                    description="DB",
                ),
            ],
        )
        tools = resolve_skill_tools(skill)
        assert "db" in tools

    def test_resolve_with_script(self, tmp_path):
        """script.py 中的函数也作为工具。"""
        script_src = textwrap.dedent("""\
            def greet(name: str) -> str:
                \"\"\"向用户打招呼。\"\"\"
                return f"Hello, {name}!"
        """)
        skill = Skill(
            name="script-tools",
            system_prompt="Hello.",
            script=script_src,
        )
        save(skill, tmp_path)

        tools = resolve_skill_tools(skill, skill_dir=tmp_path)
        assert "greet" in tools
        assert tools["greet"].execute(name="World") == "Hello, World!"

    def test_script_overrides_toolref(self, tmp_path):
        """script.py 中同名函数覆盖 ToolRef。"""
        script_src = textwrap.dedent("""\
            def weather(city: str) -> str:
                \"\"\"本地天气实现。\"\"\"
                return f"Weather in {city}: sunny"
        """)
        skill = Skill(
            name="override-test",
            system_prompt="Hello.",
            script=script_src,
            tools=[
                ToolRef(
                    name="weather",
                    type="http",
                    endpoint="https://api.weather.com",
                    description="远程天气",
                ),
            ],
        )
        save(skill, tmp_path)

        tools = resolve_skill_tools(skill, skill_dir=tmp_path)
        # script.py 的版本覆盖了 http 版本
        result = tools["weather"].execute(city="Beijing")
        assert "Beijing" in result

    def test_empty_skill(self):
        skill = Skill(name="empty", system_prompt="Hello.")
        tools = resolve_skill_tools(skill)
        assert tools == {}


# ---------------------------------------------------------------------------
# SkillTree.collect_tools — 工具继承
# ---------------------------------------------------------------------------

class TestCollectTools:

    def _make_tree(self) -> SkillTree:
        root_skill = Skill(
            name="root",
            system_prompt="Root.",
            tools=[
                ToolRef(name="global-search", type="http",
                        endpoint="https://search.example.com",
                        description="全局搜索"),
            ],
        )
        social_skill = Skill(
            name="social",
            system_prompt="Social.",
            tools=[
                ToolRef(name="weibo-api", type="http",
                        endpoint="https://api.weibo.com",
                        description="微博API"),
            ],
        )
        moments_skill = Skill(
            name="moments",
            system_prompt="Moments.",
            tools=[
                ToolRef(name="image-gen", type="http",
                        endpoint="https://image.example.com",
                        description="图片生成"),
            ],
        )

        root = SkillNode(name="root", skill=root_skill)
        social = SkillNode(name="social", skill=social_skill)
        moments = SkillNode(name="moments", skill=moments_skill)
        social.children["moments"] = moments
        root.children["social"] = social

        return SkillTree(root=root, base_path=Path("/tmp/test-tree"))

    def test_root_tools(self):
        tree = self._make_tree()
        tools = tree.collect_tools("")
        assert len(tools) == 1
        assert tools[0].name == "global-search"

    def test_child_inherits_parent(self):
        tree = self._make_tree()
        tools = tree.collect_tools("social")
        names = {t.name for t in tools}
        assert "global-search" in names  # 继承自 root
        assert "weibo-api" in names       # 自己的

    def test_deep_inheritance(self):
        tree = self._make_tree()
        tools = tree.collect_tools("social.moments")
        names = {t.name for t in tools}
        assert "global-search" in names   # 继承自 root
        assert "weibo-api" in names       # 继承自 social
        assert "image-gen" in names       # 自己的

    def test_child_overrides_parent(self):
        """子节点的同名工具覆盖父节点。"""
        root_skill = Skill(
            name="root",
            system_prompt="Root.",
            tools=[
                ToolRef(name="search", type="http",
                        endpoint="https://old-search.com",
                        description="旧搜索"),
            ],
        )
        child_skill = Skill(
            name="child",
            system_prompt="Child.",
            tools=[
                ToolRef(name="search", type="http",
                        endpoint="https://new-search.com",
                        description="新搜索"),
            ],
        )
        root = SkillNode(name="root", skill=root_skill)
        child = SkillNode(name="child", skill=child_skill)
        root.children["child"] = child

        tree = SkillTree(root=root, base_path=Path("/tmp"))
        tools = tree.collect_tools("child")
        assert len(tools) == 1
        assert tools[0].endpoint == "https://new-search.com"


# ---------------------------------------------------------------------------
# SkillTree.graft — 嫁接
# ---------------------------------------------------------------------------

class TestGraft:

    def _make_tree(self) -> SkillTree:
        root = SkillNode(
            name="root",
            skill=Skill(name="root", system_prompt="I am root."),
        )
        social = SkillNode(
            name="social",
            skill=Skill(name="social", system_prompt="Social skill."),
        )
        root.children["social"] = social
        return SkillTree(root=root, base_path=Path("/tmp/test"))

    def test_graft_skill(self):
        """嫁接一个 Skill 对象。"""
        tree = self._make_tree()
        external = Skill(
            name="email",
            system_prompt="Email assistant.",
            description="邮件助手",
        )
        node = tree.graft("social", external)
        assert node.name == "email"
        assert "email" in tree.root.children["social"].children
        assert tree.get("social.email").skill.system_prompt == "Email assistant."

    def test_graft_skill_node(self):
        """嫁接一个 SkillNode（含子节点）。"""
        tree = self._make_tree()

        # 准备一个带子节点的 SkillNode
        parent_node = SkillNode(
            name="writing",
            skill=Skill(name="writing", system_prompt="Writing."),
        )
        child_node = SkillNode(
            name="blog",
            skill=Skill(name="blog", system_prompt="Blog writing."),
        )
        parent_node.children["blog"] = child_node

        tree.graft("", parent_node)  # 嫁接到根
        assert "writing" in tree.root.children
        assert "blog" in tree.get("writing").children

    def test_graft_tree(self):
        """嫁接整棵 SkillTree。"""
        tree = self._make_tree()

        other_root = SkillNode(
            name="business",
            skill=Skill(name="business", system_prompt="Business."),
        )
        email_node = SkillNode(
            name="email",
            skill=Skill(name="email", system_prompt="Email."),
        )
        other_root.children["email"] = email_node
        other_tree = SkillTree(root=other_root, base_path=Path("/tmp/other"))

        tree.graft("", other_tree)
        assert "business" in tree.root.children
        assert "email" in tree.get("business").children

    def test_graft_with_rename(self):
        """嫁接时重命名。"""
        tree = self._make_tree()
        skill = Skill(name="old-name", system_prompt="Hello.")
        node = tree.graft("social", skill, name="new-name")

        assert node.name == "new-name"
        assert "new-name" in tree.root.children["social"].children

    def test_graft_duplicate_raises(self):
        """嫁接到已存在的名字报错。"""
        tree = self._make_tree()
        skill = Skill(name="social", system_prompt="Dup.")

        with pytest.raises(ValueError, match="已存在"):
            tree.graft("", skill)

    def test_graft_preserves_tools(self):
        """嫁接保留工具声明。"""
        tree = self._make_tree()
        skill = Skill(
            name="tooled",
            system_prompt="Tooled skill.",
            tools=[
                ToolRef(name="api", type="http",
                        endpoint="https://api.example.com",
                        description="外部API"),
            ],
        )
        tree.graft("social", skill)

        tools = tree.collect_tools("social.tooled")
        names = {t.name for t in tools}
        assert "api" in names

    def test_graft_preserves_agenda(self):
        """嫁接保留 agenda。"""
        tree = self._make_tree()
        skill = Skill(
            name="scheduled",
            system_prompt="Scheduled skill.",
            agenda=[
                AgendaEntry(
                    type=AgendaType.RECURRING,
                    title="周报",
                    recurrence=Recurrence.WEEKLY,
                    weekday=4,
                ),
            ],
        )
        tree.graft("social", skill)
        grafted = tree.get("social.scheduled")
        assert len(grafted.skill.agenda) == 1

    def test_graft_deep_copy(self):
        """嫁接后修改源不影响树。"""
        tree = self._make_tree()
        source = SkillNode(
            name="external",
            skill=Skill(name="external", system_prompt="Original."),
        )
        tree.graft("", source)

        # 修改源不影响树
        source.skill = Skill(name="external", system_prompt="Modified.")
        assert tree.get("external").skill.system_prompt == "Original."

    def test_graft_save_load(self, tmp_path):
        """嫁接后保存再加载，结构完整。"""
        # 创建主树
        root_dir = tmp_path / "main"
        root_skill = Skill(name="main", system_prompt="Main root.")
        save(root_skill, root_dir)
        tree = SkillTree.load(root_dir)

        # 创建外部 skill
        ext_dir = tmp_path / "external"
        ext_skill = Skill(
            name="imported",
            system_prompt="Imported skill.",
            tools=[
                ToolRef(name="ext-api", type="http",
                        endpoint="https://ext.example.com",
                        description="外部"),
            ],
        )
        save(ext_skill, ext_dir)

        # 嫁接
        loaded_ext = load(ext_dir)
        tree.graft("", loaded_ext)
        tree.save()

        # 重新加载验证
        reloaded = SkillTree.load(root_dir)
        assert "imported" in reloaded.root.children
        imported = reloaded.get("imported")
        assert imported.skill.system_prompt == "Imported skill."
        assert len(imported.skill.tools) == 1
