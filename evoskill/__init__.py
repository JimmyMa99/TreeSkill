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


def _missing_optional(module_name, feature_name):
    """Return a callable placeholder that raises a helpful import error."""

    def _raiser(*args, **kwargs):
        raise ImportError(
            f"\n\n❌ {feature_name} is unavailable\n\n"
            f"The current codebase is missing the optional module: {module_name}\n"
            f"Add the corresponding implementation files before using this feature.\n"
        )

    return _raiser

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
        f"\n\n❌ Import failed: missing required dependency '{missing_module}'\n\n"
        f"How to fix it:\n"
        f"  1. Activate the conda environment:\n"
        f"     conda activate pr\n\n"
        f"  2. Install the dependency:\n"
        f"     pip install {missing_module}\n\n"
        f"  Or install all project dependencies:\n"
        f"     pip install -e .\n\n"
        f"See pyproject.toml for the full dependency list.\n"
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
                f"\n\n❌ Failed to import OpenAIAdapter\n\n"
                f"Install the OpenAI SDK and tiktoken:\n"
                f"  pip install openai tiktoken\n\n"
                f"Or install all project dependencies:\n"
                f"  pip install -e .\n"
            ) from None
    elif name == "AnthropicAdapter":
        try:
            from evoskill.adapters.anthropic import AnthropicAdapter as _AnthropicAdapter
            return _AnthropicAdapter
        except ImportError:
            raise ImportError(
                f"\n\n❌ Failed to import AnthropicAdapter\n\n"
                f"Install the Anthropic SDK:\n"
                f"  pip install anthropic\n\n"
                f"Or install all project dependencies:\n"
                f"  pip install -e .\n"
            ) from None
    elif name == "MockAdapter":
        try:
            from examples.mock_adapter import MockAdapter as _MockAdapter
            return _MockAdapter
        except ImportError:
            raise ImportError(
                f"\n\n❌ Failed to import MockAdapter\n\n"
                f"MockAdapter lives in examples/mock_adapter.py\n"
                f"Make sure examples/ is available on the Python path.\n"
            ) from None
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}\n"
        f"Available adapters: OpenAIAdapter, AnthropicAdapter, MockAdapter (dependencies required)"
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
from evoskill.skill_tree import SkillTree, SkillNode, resolve_skill_tools
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
try:
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
except ImportError:
    ScriptValidator = None
    ScriptValidationResult = None
    ScriptIssue = None
    validate_script = _missing_optional("evoskill.script", "script validation")
    validate_script_file = _missing_optional("evoskill.script", "script validation")
    load_script = _missing_optional("evoskill.script", "script loading")
    save_script = _missing_optional("evoskill.script", "script saving")
    load_script_as_tools = _missing_optional("evoskill.script", "script tool loading")

# Memory module
try:
    from evoskill.memory import (
        MEMORY_FILE,
        MemoryType,
        MemoryEntry,
        MemoryStore,
        MemoryCompiler,
    )
except ImportError:
    MEMORY_FILE = None
    MemoryType = None
    MemoryEntry = None
    MemoryStore = None
    MemoryCompiler = None

# Schema: Agenda & ToolRef
from evoskill.schema import AgendaEntry, AgendaType, Recurrence, ToolRef
try:
    from evoskill.agenda import (
        AgendaManager,
        compile_agenda_context,
        parse_due,
    )
except ImportError:
    AgendaManager = None
    compile_agenda_context = _missing_optional("evoskill.agenda", "agenda context compilation")
    parse_due = _missing_optional("evoskill.agenda", "agenda time parsing")

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
    "resolve_skill_tools",
    "ToolRef",
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

    # Memory module
    "MEMORY_FILE",
    "MemoryType",
    "MemoryEntry",
    "MemoryStore",
    "MemoryCompiler",

    # Agenda module
    "AgendaEntry",
    "AgendaType",
    "Recurrence",
    "AgendaManager",
    "compile_agenda_context",
    "parse_due",

    # Version
    "__version__",
    "__author__",
    "__email__",
]
