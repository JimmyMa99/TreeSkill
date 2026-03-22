"""Python Script Validation & Storage — 验证和持久化 Skill 中的 Python 脚本。

每个 Skill 目录可以包含一个可选的 ``script.py``，用于定义自定义工具函数。
该模块负责：

1. **语法验证** — 确保脚本可被 Python 解析
2. **安全检查** — 检测危险导入和函数调用
3. **格式验证** — 确保脚本符合 Skill 规范（包含入口函数）
4. **磁盘读写** — 从 Skill 目录加载/保存 script.py
5. **工具转换** — 将脚本中的函数注册为 PythonFunctionTool

Skill 目录结构::

    my-skill/
    ├── SKILL.md        # Required
    ├── config.yaml     # Optional
    └── script.py       # Optional: Python 脚本

script.py 格式要求::

    # 至少包含一个公开函数（不以 _ 开头）
    # 函数需要有 docstring 用于描述功能
    # 支持类型注解

    def my_tool(arg: str) -> str:
        \"\"\"工具描述，会作为 LLM 的工具描述。\"\"\"
        return result
"""

from __future__ import annotations

import ast
import logging
import textwrap
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_FILE = "script.py"

# 高危模块 — 默认禁止导入
_BLOCKED_MODULES: frozenset[str] = frozenset({
    "os",
    "subprocess",
    "shutil",
    "sys",
    "ctypes",
    "importlib",
    "pickle",
    "shelve",
    "socket",
    "http.server",
    "xmlrpc",
    "multiprocessing",
    "signal",
    "resource",
    "pty",
    "fcntl",
    "termios",
})

# 高危函数调用
_BLOCKED_CALLS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "open",   # 仅在安全模式下阻止
})

# script.py 中必须存在至少一个公开函数的最大行数限制
_MAX_SCRIPT_LINES = 2000
_MAX_SCRIPT_BYTES = 100_000  # 100 KB


# ---------------------------------------------------------------------------
# Validation Result
# ---------------------------------------------------------------------------

class ScriptIssue(BaseModel):
    """脚本验证中发现的单个问题。"""

    level: str = Field(
        ..., description="严重级别: error, warning, info"
    )
    line: Optional[int] = Field(
        default=None, description="问题所在行号"
    )
    message: str = Field(
        ..., description="问题描述"
    )


class ScriptValidationResult(BaseModel):
    """脚本验证的完整结果。"""

    valid: bool = Field(
        ..., description="脚本是否通过验证"
    )
    issues: List[ScriptIssue] = Field(
        default_factory=list, description="发现的所有问题"
    )
    functions: List[str] = Field(
        default_factory=list, description="脚本中发现的公开函数名"
    )
    source_lines: int = Field(
        default=0, description="脚本总行数"
    )

    @property
    def errors(self) -> List[ScriptIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ScriptIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def summary(self) -> str:
        """返回人类可读的摘要。"""
        if self.valid:
            return (
                f"✓ 验证通过 ({self.source_lines} 行, "
                f"{len(self.functions)} 个公开函数)"
            )
        error_msgs = "; ".join(e.message for e in self.errors)
        return f"✗ 验证失败: {error_msgs}"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ScriptValidator:
    """Python 脚本验证器。

    Parameters
    ----------
    blocked_modules : frozenset[str] | None
        禁止导入的模块集合。传 None 使用默认列表。
    blocked_calls : frozenset[str] | None
        禁止的函数调用集合。传 None 使用默认列表。
    strict : bool
        严格模式：额外检查 open() 等调用。
    """

    def __init__(
        self,
        *,
        blocked_modules: Optional[frozenset[str]] = None,
        blocked_calls: Optional[frozenset[str]] = None,
        strict: bool = False,
    ) -> None:
        self.blocked_modules = (
            blocked_modules if blocked_modules is not None else _BLOCKED_MODULES
        )
        base_calls = (
            blocked_calls if blocked_calls is not None else _BLOCKED_CALLS
        )
        if not strict:
            # 非严格模式允许 open
            base_calls = base_calls - {"open"}
        self.blocked_calls = base_calls

    def validate(self, source: str) -> ScriptValidationResult:
        """执行完整验证流程。"""
        issues: List[ScriptIssue] = []
        functions: List[str] = []

        # 1) 大小检查
        if len(source.encode("utf-8")) > _MAX_SCRIPT_BYTES:
            issues.append(ScriptIssue(
                level="error",
                message=f"脚本超过大小限制 ({_MAX_SCRIPT_BYTES} bytes)",
            ))
            return ScriptValidationResult(
                valid=False, issues=issues, functions=[], source_lines=0,
            )

        lines = source.splitlines()
        if len(lines) > _MAX_SCRIPT_LINES:
            issues.append(ScriptIssue(
                level="error",
                message=f"脚本超过行数限制 ({_MAX_SCRIPT_LINES} 行)",
            ))
            return ScriptValidationResult(
                valid=False, issues=issues, functions=[], source_lines=len(lines),
            )

        # 2) 语法检查
        tree = self._check_syntax(source, issues)
        if tree is None:
            return ScriptValidationResult(
                valid=False, issues=issues, functions=[], source_lines=len(lines),
            )

        # 3) 安全检查
        self._check_imports(tree, issues)
        self._check_calls(tree, issues)

        # 4) 格式检查 — 提取公开函数
        functions = self._extract_public_functions(tree, issues)

        has_errors = any(i.level == "error" for i in issues)
        return ScriptValidationResult(
            valid=not has_errors,
            issues=issues,
            functions=functions,
            source_lines=len(lines),
        )

    # -- internal checks --

    def _check_syntax(
        self, source: str, issues: List[ScriptIssue]
    ) -> Optional[ast.Module]:
        """解析脚本 AST，返回 None 表示语法错误。"""
        try:
            return ast.parse(source, filename="script.py")
        except SyntaxError as exc:
            issues.append(ScriptIssue(
                level="error",
                line=exc.lineno,
                message=f"语法错误: {exc.msg}",
            ))
            return None

    def _check_imports(
        self, tree: ast.Module, issues: List[ScriptIssue]
    ) -> None:
        """检查是否导入了禁止的模块。"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in self.blocked_modules:
                        issues.append(ScriptIssue(
                            level="error",
                            line=node.lineno,
                            message=f"禁止导入模块: {alias.name}",
                        ))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module in self.blocked_modules:
                        issues.append(ScriptIssue(
                            level="error",
                            line=node.lineno,
                            message=f"禁止导入模块: {node.module}",
                        ))

    def _check_calls(
        self, tree: ast.Module, issues: List[ScriptIssue]
    ) -> None:
        """检查是否调用了禁止的函数。"""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = self._resolve_call_name(node)
            if func_name and func_name in self.blocked_calls:
                issues.append(ScriptIssue(
                    level="error",
                    line=node.lineno,
                    message=f"禁止调用函数: {func_name}()",
                ))

    @staticmethod
    def _resolve_call_name(node: ast.Call) -> Optional[str]:
        """从 AST Call 节点提取函数名。"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _extract_public_functions(
        self, tree: ast.Module, issues: List[ScriptIssue]
    ) -> List[str]:
        """提取所有公开函数（不以 _ 开头），并检查 docstring。"""
        functions: List[str] = []
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            functions.append(node.name)
            # 检查 docstring
            if not ast.get_docstring(node):
                issues.append(ScriptIssue(
                    level="warning",
                    line=node.lineno,
                    message=f"函数 {node.name}() 缺少 docstring（建议添加，用作工具描述）",
                ))
        if not functions:
            issues.append(ScriptIssue(
                level="error",
                message="脚本中没有公开函数（至少需要一个不以 _ 开头的函数）",
            ))
        return functions


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def load_script(skill_dir: Path) -> Optional[str]:
    """从 Skill 目录加载 script.py 源码，不存在则返回 None。"""
    script_path = Path(skill_dir) / SCRIPT_FILE
    if not script_path.is_file():
        return None
    return script_path.read_text(encoding="utf-8")


def save_script(skill_dir: Path, source: str) -> Path:
    """将脚本源码保存到 Skill 目录的 script.py。

    保存前会自动执行验证，验证失败抛出 ValueError。

    Returns
    -------
    Path
        写入文件的路径。
    """
    skill_dir = Path(skill_dir)
    skill_dir.mkdir(parents=True, exist_ok=True)

    validator = ScriptValidator()
    result = validator.validate(source)
    if not result.valid:
        raise ValueError(
            f"脚本验证失败，无法保存:\n{result.summary()}\n"
            + "\n".join(
                f"  L{i.line or '?'}: [{i.level}] {i.message}"
                for i in result.errors
            )
        )

    script_path = skill_dir / SCRIPT_FILE
    script_path.write_text(source, encoding="utf-8")
    logger.info(f"脚本已保存: {script_path} ({result.source_lines} 行)")
    return script_path


# ---------------------------------------------------------------------------
# Script → Tool conversion
# ---------------------------------------------------------------------------

def _extract_function_schema(
    func_node: ast.FunctionDef,
) -> Dict[str, Any]:
    """从 AST 函数定义提取 OpenAI function calling 参数 schema。"""
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for arg in func_node.args.args:
        param_name = arg.arg
        if param_name == "self":
            continue

        param_schema: Dict[str, Any] = {}

        # 尝试从类型注解推断类型
        if arg.annotation:
            param_schema["type"] = _annotation_to_json_type(arg.annotation)
        else:
            param_schema["type"] = "string"

        properties[param_name] = param_schema

    # 有默认值的参数不是 required
    n_defaults = len(func_node.args.defaults)
    n_args = len(func_node.args.args)
    for i, arg in enumerate(func_node.args.args):
        if arg.arg == "self":
            continue
        # 没有默认值 → required
        if i < n_args - n_defaults:
            required.append(arg.arg)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _annotation_to_json_type(annotation: ast.expr) -> str:
    """将 Python 类型注解转换为 JSON Schema 类型。"""
    if isinstance(annotation, ast.Name):
        mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        return mapping.get(annotation.id, "string")
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }
        return mapping.get(annotation.value, "string")
    return "string"


def load_script_as_tools(
    skill_dir: Path,
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """从 Skill 目录加载 script.py 并转换为工具字典。

    Parameters
    ----------
    skill_dir : Path
        Skill 目录路径。
    validate : bool
        是否在加载前验证脚本。

    Returns
    -------
    Dict[str, BaseTool]
        函数名 → PythonFunctionTool 的映射。空字典表示无脚本。

    Raises
    ------
    ValueError
        验证失败时。
    """
    from tresskill.tools import PythonFunctionTool

    source = load_script(skill_dir)
    if source is None:
        return {}

    if validate:
        validator = ScriptValidator()
        result = validator.validate(source)
        if not result.valid:
            raise ValueError(
                f"脚本验证失败:\n{result.summary()}\n"
                + "\n".join(
                    f"  L{i.line or '?'}: [{i.level}] {i.message}"
                    for i in result.errors
                )
            )

    # 解析 AST 提取函数信息
    tree = ast.parse(source, filename="script.py")
    func_nodes: Dict[str, ast.FunctionDef] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            func_nodes[node.name] = node

    if not func_nodes:
        return {}

    # 在隔离的命名空间中执行脚本
    namespace: Dict[str, Any] = {}
    exec(compile(tree, filename="script.py", mode="exec"), namespace)  # noqa: S102

    tools: Dict[str, Any] = {}
    for func_name, func_node in func_nodes.items():
        func = namespace.get(func_name)
        if func is None or not callable(func):
            continue

        docstring = ast.get_docstring(func_node) or f"Tool: {func_name}"
        schema = _extract_function_schema(func_node)

        tools[func_name] = PythonFunctionTool(
            _name=func_name,
            _description=docstring,
            func=func,
            parameters_schema=schema,
        )

    return tools


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def validate_script(source: str, *, strict: bool = False) -> ScriptValidationResult:
    """便捷函数：验证脚本源码。"""
    return ScriptValidator(strict=strict).validate(source)


def validate_script_file(path: Path, *, strict: bool = False) -> ScriptValidationResult:
    """便捷函数：验证脚本文件。"""
    path = Path(path)
    if not path.is_file():
        return ScriptValidationResult(
            valid=False,
            issues=[ScriptIssue(level="error", message=f"文件不存在: {path}")],
        )
    source = path.read_text(encoding="utf-8")
    return ScriptValidator(strict=strict).validate(source)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "SCRIPT_FILE",
    "ScriptIssue",
    "ScriptValidationResult",
    "ScriptValidator",
    "load_script",
    "save_script",
    "load_script_as_tools",
    "validate_script",
    "validate_script_file",
]
