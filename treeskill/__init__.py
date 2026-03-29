"""
treeskill: Kode-forwarded AS(skill)O framework

Primary pipeline:
- Kode performs the forward pass
- ASO evolves full programs (root prompt + skills + selection policy)
- SealQA lifecycle demo is the current recommended end-to-end example

Core Components:
- Core Abstraction Layer: Model-agnostic interfaces
- Model Adapters: OpenAI, Anthropic, and more
- Optimizer: TGD-based prompt and skill optimization
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
    from treeskill.core import (
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
# from treeskill.adapters.openai import OpenAIAdapter
# from treeskill.adapters.anthropic import AnthropicAdapter

# For backward compatibility, provide lazy-loading with helpful errors
def __getattr__(name):
    """Lazy import for adapters with helpful error messages."""
    if name == "OpenAIAdapter":
        try:
            from treeskill.adapters.openai import OpenAIAdapter as _OpenAIAdapter
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
            from treeskill.adapters.anthropic import AnthropicAdapter as _AnthropicAdapter
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
from treeskill.registry import (
    TreeSkillRegistry,
    registry,
    adapter,
    optimizer,
    hook,
    ComponentMeta,
)

# Tool registry system (new)
from treeskill.tools import (
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
from treeskill.schema import (
    ContentPart,
    Feedback as LegacyFeedback,
    ImageContent,
    Message,
    Skill,
    SkillMeta,
    TextContent,
    Trace,
)
from treeskill.config import GlobalConfig
from treeskill.skill_tree import SkillTree, SkillNode, resolve_skill_tools
from treeskill.checkpoint import CheckpointManager
from treeskill.resume import ResumeState
from treeskill.aso_program import ASOProgram, ASOSkill
from treeskill.aso_optimizer import ASOOptimizer, ASOSkillAction, ASOResult, ASOIterationResult
from treeskill.tasks import SealQAExample, SealQATaskAdapter

# Skill management functions (Agent Skills format)
from treeskill.skill import (
    load as load_skill,
    save as save_skill,
    compile_messages,
    SKILL_FILE,
    CONFIG_FILE,
    SCRIPT_FILE,
)

# Script validation & storage
try:
    from treeskill.script import (
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
    validate_script = _missing_optional("treeskill.script", "script validation")
    validate_script_file = _missing_optional("treeskill.script", "script validation")
    load_script = _missing_optional("treeskill.script", "script loading")
    save_script = _missing_optional("treeskill.script", "script saving")
    load_script_as_tools = _missing_optional("treeskill.script", "script tool loading")

# Memory module
try:
    from treeskill.memory import (
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
from treeskill.schema import AgendaEntry, AgendaType, Recurrence, ToolRef
try:
    from treeskill.agenda import (
        AgendaManager,
        compile_agenda_context,
        parse_due,
    )
except ImportError:
    AgendaManager = None
    compile_agenda_context = _missing_optional("treeskill.agenda", "agenda context compilation")
    parse_due = _missing_optional("treeskill.agenda", "agenda time parsing")

__version__ = "0.2.0"
__author__ = "TreeSkill Team"
__email__ = "treeskill@example.com"

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
    "TreeSkillRegistry",
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
