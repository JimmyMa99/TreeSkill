"""Tests for tresskill.script — Python 脚本验证和存储。"""

import textwrap
from pathlib import Path

import pytest

from tresskill.script import (
    SCRIPT_FILE,
    ScriptValidator,
    load_script,
    load_script_as_tools,
    save_script,
    validate_script,
    validate_script_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_SCRIPT = textwrap.dedent("""\
    def greet(name: str) -> str:
        \"\"\"向用户打招呼。\"\"\"
        return f"Hello, {name}!"

    def add(a: int, b: int) -> int:
        \"\"\"计算两个数的和。\"\"\"
        return a + b
""")

SCRIPT_NO_DOCSTRING = textwrap.dedent("""\
    def greet(name: str) -> str:
        return f"Hello, {name}!"
""")

SCRIPT_SYNTAX_ERROR = textwrap.dedent("""\
    def broken(
        return "oops"
""")

SCRIPT_DANGEROUS_IMPORT = textwrap.dedent("""\
    import os

    def list_files(path: str) -> str:
        \"\"\"列出目录文件。\"\"\"
        return str(os.listdir(path))
""")

SCRIPT_DANGEROUS_CALL = textwrap.dedent("""\
    def run_code(code: str) -> str:
        \"\"\"执行代码。\"\"\"
        return str(eval(code))
""")

SCRIPT_NO_PUBLIC_FUNC = textwrap.dedent("""\
    _INTERNAL = 42

    def _helper():
        pass
""")

SCRIPT_WITH_DEFAULTS = textwrap.dedent("""\
    def search(query: str, limit: int = 10, verbose: bool = False) -> str:
        \"\"\"搜索内容。\"\"\"
        return f"Searching {query} (limit={limit})"
""")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestScriptValidator:

    def test_valid_script(self):
        result = validate_script(VALID_SCRIPT)
        assert result.valid is True
        assert len(result.errors) == 0
        assert "greet" in result.functions
        assert "add" in result.functions
        assert result.source_lines > 0

    def test_syntax_error(self):
        result = validate_script(SCRIPT_SYNTAX_ERROR)
        assert result.valid is False
        assert any("语法错误" in e.message for e in result.errors)

    def test_dangerous_import(self):
        result = validate_script(SCRIPT_DANGEROUS_IMPORT)
        assert result.valid is False
        assert any("os" in e.message for e in result.errors)

    def test_dangerous_call(self):
        result = validate_script(SCRIPT_DANGEROUS_CALL)
        assert result.valid is False
        assert any("eval" in e.message for e in result.errors)

    def test_no_public_function(self):
        result = validate_script(SCRIPT_NO_PUBLIC_FUNC)
        assert result.valid is False
        assert any("公开函数" in e.message for e in result.errors)

    def test_missing_docstring_warning(self):
        result = validate_script(SCRIPT_NO_DOCSTRING)
        assert result.valid is True  # warning 不阻止通过
        assert len(result.warnings) > 0
        assert any("docstring" in w.message for w in result.warnings)

    def test_subprocess_blocked(self):
        source = textwrap.dedent("""\
            import subprocess
            def run(cmd: str) -> str:
                \"\"\"Run cmd.\"\"\"
                return subprocess.check_output(cmd, shell=True).decode()
        """)
        result = validate_script(source)
        assert result.valid is False

    def test_from_import_blocked(self):
        source = textwrap.dedent("""\
            from os.path import join
            def concat(a: str, b: str) -> str:
                \"\"\"Join paths.\"\"\"
                return join(a, b)
        """)
        result = validate_script(source)
        assert result.valid is False

    def test_safe_imports_allowed(self):
        source = textwrap.dedent("""\
            import json
            import re
            from datetime import datetime

            def parse(data: str) -> str:
                \"\"\"Parse JSON data.\"\"\"
                return json.dumps(json.loads(data))
        """)
        result = validate_script(source)
        assert result.valid is True

    def test_strict_blocks_open(self):
        source = textwrap.dedent("""\
            def read(path: str) -> str:
                \"\"\"Read file.\"\"\"
                return open(path).read()
        """)
        # 非严格模式：允许
        result_normal = ScriptValidator(strict=False).validate(source)
        assert result_normal.valid is True

        # 严格模式：禁止
        result_strict = ScriptValidator(strict=True).validate(source)
        assert result_strict.valid is False

    def test_size_limit(self):
        huge = "x = 1\n" * 3000
        result = validate_script(huge)
        assert result.valid is False
        assert any("行数限制" in e.message for e in result.errors)

    def test_summary(self):
        result = validate_script(VALID_SCRIPT)
        assert "✓" in result.summary()

        result = validate_script(SCRIPT_SYNTAX_ERROR)
        assert "✗" in result.summary()


# ---------------------------------------------------------------------------
# Disk I/O tests
# ---------------------------------------------------------------------------

class TestScriptDiskIO:

    def test_load_script_not_exists(self, tmp_path):
        assert load_script(tmp_path) is None

    def test_save_and_load(self, tmp_path):
        save_script(tmp_path, VALID_SCRIPT)
        loaded = load_script(tmp_path)
        assert loaded == VALID_SCRIPT
        assert (tmp_path / SCRIPT_FILE).is_file()

    def test_save_invalid_raises(self, tmp_path):
        with pytest.raises(ValueError, match="验证失败"):
            save_script(tmp_path, SCRIPT_SYNTAX_ERROR)

    def test_validate_script_file(self, tmp_path):
        script_path = tmp_path / "test.py"
        script_path.write_text(VALID_SCRIPT)
        result = validate_script_file(script_path)
        assert result.valid is True

    def test_validate_script_file_not_exists(self, tmp_path):
        result = validate_script_file(tmp_path / "nope.py")
        assert result.valid is False


# ---------------------------------------------------------------------------
# Tool conversion tests
# ---------------------------------------------------------------------------

class TestScriptToTool:

    def test_load_as_tools(self, tmp_path):
        (tmp_path / SCRIPT_FILE).write_text(VALID_SCRIPT)
        tools = load_script_as_tools(tmp_path)
        assert "greet" in tools
        assert "add" in tools

        # 验证工具可执行
        result = tools["greet"].execute(name="World")
        assert result == "Hello, World!"

        result = tools["add"].execute(a=3, b=5)
        assert result == 8

    def test_tool_schema(self, tmp_path):
        (tmp_path / SCRIPT_FILE).write_text(VALID_SCRIPT)
        tools = load_script_as_tools(tmp_path)

        schema = tools["greet"].to_schema()
        assert schema["name"] == "greet"
        assert "name" in schema["parameters"]["properties"]
        assert schema["parameters"]["properties"]["name"]["type"] == "string"

    def test_tool_with_defaults(self, tmp_path):
        (tmp_path / SCRIPT_FILE).write_text(SCRIPT_WITH_DEFAULTS)
        tools = load_script_as_tools(tmp_path)

        schema = tools["search"].to_schema()
        # query is required, limit and verbose have defaults
        assert "query" in schema["parameters"]["required"]
        assert "limit" not in schema["parameters"]["required"]
        assert "verbose" not in schema["parameters"]["required"]

    def test_no_script_returns_empty(self, tmp_path):
        tools = load_script_as_tools(tmp_path)
        assert tools == {}

    def test_invalid_script_raises(self, tmp_path):
        (tmp_path / SCRIPT_FILE).write_text(SCRIPT_DANGEROUS_IMPORT)
        with pytest.raises(ValueError, match="验证失败"):
            load_script_as_tools(tmp_path)

    def test_skip_validation(self, tmp_path):
        """validate=False 跳过验证（用于信任的脚本）。"""
        (tmp_path / SCRIPT_FILE).write_text(SCRIPT_DANGEROUS_IMPORT)
        # 不验证不报错（但不建议在生产中使用）
        tools = load_script_as_tools(tmp_path, validate=False)
        assert "list_files" in tools


# ---------------------------------------------------------------------------
# Integration with Skill load/save
# ---------------------------------------------------------------------------

class TestSkillIntegration:

    def test_skill_load_with_script(self, tmp_path):
        """Skill 目录包含 script.py 时，自动加载到 skill.script。"""
        from tresskill.skill import load, save
        from tresskill.schema import Skill

        # 创建完整的 skill 目录
        skill = Skill(
            name="test-skill",
            description="A test skill with script",
            system_prompt="You are a helpful assistant.",
            script=VALID_SCRIPT,
        )
        save(skill, tmp_path)

        # 验证文件存在
        assert (tmp_path / "SKILL.md").is_file()
        assert (tmp_path / "script.py").is_file()

        # 重新加载
        loaded = load(tmp_path)
        assert loaded.script == VALID_SCRIPT
        assert loaded.name == "test-skill"

    def test_skill_save_without_script(self, tmp_path):
        """没有脚本时不创建 script.py。"""
        from tresskill.skill import save
        from tresskill.schema import Skill

        skill = Skill(
            name="no-script",
            system_prompt="Just a prompt.",
        )
        save(skill, tmp_path)

        assert not (tmp_path / "script.py").exists()

    def test_skill_remove_script(self, tmp_path):
        """移除脚本后，保存时删除 script.py。"""
        from tresskill.skill import load, save
        from tresskill.schema import Skill

        # 先创建带脚本的
        skill = Skill(
            name="with-script",
            system_prompt="Prompt.",
            script=VALID_SCRIPT,
        )
        save(skill, tmp_path)
        assert (tmp_path / "script.py").is_file()

        # 移除脚本后保存
        skill_no_script = skill.model_copy(update={"script": None})
        save(skill_no_script, tmp_path)
        assert not (tmp_path / "script.py").exists()
