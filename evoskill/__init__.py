"""
evoskill: Train-free Prompt Evolution Framework

A plugin-based framework for evolving LLM prompts and skills
using Textual Gradient Descent (TGD).

Core Components:
- Core Abstraction Layer: Model-agnostic interfaces
- Model Adapters: OpenAI, Anthropic, and more
- Optimizer: TGD-based prompt optimization
- Registry: Plugin system for extensibility
- Legacy: Backward compatible with v0.1
"""

# Core abstraction layer (new)
try:
    from evoskill.core import (
        # Abstract base classes
        OptimizablePrompt,
        TextualGradient,
        Experience,
        Feedback,
        ModelAdapter,

        # Concrete implementations
        TextPrompt,
        MultimodalPrompt,
        StructuredPrompt,
        SimpleGradient,
        MultimodalGradient,
        GradientHistory,
        ConversationExperience,
        MultimodalExperience,
        CompositeFeedback,
        FeedbackType,
        BaseModelAdapter,

        # Optimizer
        TrainFreeOptimizer,
        OptimizerConfig,
        OptimizationResult,
        OptimizationStep,
        Validator,

        # Strategies
        OptimizationStrategy,
        ConservativeStrategy,
        AggressiveStrategy,
        AdaptiveStrategy,
        get_strategy,

        # Validators
        AutoValidator,
        MetricValidator,
        CompositeValidator,
        create_simple_validator,
        create_metric_validator,

        # Tree Optimizer
        TreeAwareOptimizer,
        TreeOptimizerConfig,
        TreeOptimizationResult,
    )
except ImportError as e:
    missing_module = str(e).split("'")[-2] if "'" in str(e) else "unknown"
    raise ImportError(
        f"\n\n❌ 导入失败: 缺少必需的依赖 '{missing_module}'\n\n"
        f"解决方法:\n"
        f"  1. 激活conda环境:\n"
        f"     conda activate pr\n\n"
        f"  2. 安装依赖:\n"
        f"     pip install {missing_module}\n\n"
        f"  或者安装所有依赖:\n"
        f"     pip install -e .\n\n"
        f"详细依赖列表请查看 pyproject.toml\n"
    ) from None

# Model adapters (new) - lazy import to avoid dependency issues
# from evoskill.adapters.openai import OpenAIAdapter
# from evoskill.adapters.anthropic import AnthropicAdapter

# For backward compatibility, provide lazy-loading with helpful errors
def __getattr__(name):
    """Lazy import for adapters with helpful error messages."""
    if name == "OpenAIAdapter":
        try:
            from evoskill.adapters.openai import OpenAIAdapter as _OpenAIAdapter
            return _OpenAIAdapter
        except ImportError:
            raise ImportError(
                f"\n\n❌ OpenAIAdapter 导入失败\n\n"
                f"需要安装 OpenAI SDK 和 tiktoken:\n"
                f"  pip install openai tiktoken\n\n"
                f"或者安装所有依赖:\n"
                f"  pip install -e .\n"
            ) from None
    elif name == "AnthropicAdapter":
        try:
            from evoskill.adapters.anthropic import AnthropicAdapter as _AnthropicAdapter
            return _AnthropicAdapter
        except ImportError:
            raise ImportError(
                f"\n\n❌ AnthropicAdapter 导入失败\n\n"
                f"需要安装 Anthropic SDK:\n"
                f"  pip install anthropic\n\n"
                f"或者安装所有依赖:\n"
                f"  pip install -e .\n"
            ) from None
    elif name == "MockAdapter":
        try:
            from examples.mock_adapter import MockAdapter as _MockAdapter
            return _MockAdapter
        except ImportError:
            raise ImportError(
                f"\n\n❌ MockAdapter 导入失败\n\n"
                f"MockAdapter 在 examples/mock_adapter.py 中\n"
                f"确保 examples/ 目录在 Python 路径中\n"
            ) from None
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}\n"
        f"可用的适配器: OpenAIAdapter, AnthropicAdapter, MockAdapter (需要安装依赖)"
    )

# Registry system (new)
from evoskill.registry import (
    EvoSkillRegistry,
    registry,
    adapter,
    optimizer,
    hook,
    ComponentMeta,
)

# Tool registry system (new)
from evoskill.tools import (
    BaseTool,
    PythonFunctionTool,
    HTTPTool,
    MCPTool,
    ToolRegistry,
    tool_registry,
    tool,
    create_http_tool,
    create_mcp_tool,
)

# Schema imports
from evoskill.schema import (
    ContentPart,
    Feedback as LegacyFeedback,
    ImageContent,
    Message,
    Skill,
    SkillMeta,
    TextContent,
    Trace,
)
from evoskill.config import GlobalConfig
from evoskill.skill_tree import SkillTree, SkillNode
from evoskill.checkpoint import CheckpointManager
from evoskill.resume import ResumeState

# Skill management functions (Agent Skills format)
from evoskill.skill import (
    load as load_skill,
    save as save_skill,
    compile_messages,
    SKILL_FILE,
    CONFIG_FILE,
    SCRIPT_FILE,
)

# Script validation & storage
from evoskill.script import (
    ScriptValidator,
    ScriptValidationResult,
    ScriptIssue,
    validate_script,
    validate_script_file,
    load_script,
    save_script,
    load_script_as_tools,
)

__version__ = "0.2.0"
__author__ = "EvoSkill Team"
__email__ = "evoskill@example.com"

__all__ = [
    # Core abstraction layer (new)
    "OptimizablePrompt",
    "TextualGradient",
    "Experience",
    "Feedback",
    "ModelAdapter",
    "TextPrompt",
    "MultimodalPrompt",
    "StructuredPrompt",
    "SimpleGradient",
    "MultimodalGradient",
    "GradientHistory",
    "ConversationExperience",
    "MultimodalExperience",
    "CompositeFeedback",
    "FeedbackType",
    "BaseModelAdapter",

    # Optimizer
    "TrainFreeOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "OptimizationStep",
    "Validator",

    # Strategies
    "OptimizationStrategy",
    "ConservativeStrategy",
    "AggressiveStrategy",
    "AdaptiveStrategy",
    "get_strategy",

    # Validators
    "AutoValidator",
    "MetricValidator",
    "CompositeValidator",
    "create_simple_validator",
    "create_metric_validator",

    # Tree Optimizer
    "TreeAwareOptimizer",
    "TreeOptimizerConfig",
    "TreeOptimizationResult",

    # Model adapters (new)
    "OpenAIAdapter",
    "AnthropicAdapter",

    # Registry system (new)
    "EvoSkillRegistry",
    "registry",
    "adapter",
    "optimizer",
    "hook",
    "ComponentMeta",

    # Tool registry system (new)
    "BaseTool",
    "PythonFunctionTool",
    "HTTPTool",
    "MCPTool",
    "ToolRegistry",
    "tool_registry",
    "tool",
    "create_http_tool",
    "create_mcp_tool",

    # Schema & config
    "CheckpointManager",
    "ContentPart",
    "LegacyFeedback",
    "GlobalConfig",
    "ImageContent",
    "Message",
    "Skill",
    "SkillMeta",
    "SkillNode",
    "SkillTree",
    "TextContent",
    "Trace",

    # Skill management functions (Agent Skills format)
    "load_skill",
    "save_skill",
    "compile_messages",
    "SKILL_FILE",
    "CONFIG_FILE",
    "SCRIPT_FILE",

    # Script validation & storage
    "ScriptValidator",
    "ScriptValidationResult",
    "ScriptIssue",
    "validate_script",
    "validate_script_file",
    "load_script",
    "save_script",
    "load_script_as_tools",

    # Version
    "__version__",
    "__author__",
    "__email__",
]
