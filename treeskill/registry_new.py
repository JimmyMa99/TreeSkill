"""TreeSkill Registry - Plugin-based component management.

This module provides a flexible plugin system that allows users to:
1. Register custom adapters, optimizers, validators, etc.
2. Define lifecycle hooks for extensibility
3. Load components from configuration files
4. Use decorator syntax for easy registration

Examples
--------
>>> from treeskill import registry, adapter
>>>
>>> # Method 1: Decorator
>>> @adapter("my-custom")
... class MyAdapter(BaseModelAdapter):
...     pass
>>>
>>> # Method 2: Direct registration
>>> registry.register_adapter("my-custom", MyAdapter)
>>>
>>> # Method 3: Config file
>>> registry.load_from_config("config.yaml")
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

# Import base classes
from treeskill.core.abc import ModelAdapter
from treeskill.core.base_adapter import BaseModelAdapter


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component Metadata
# ---------------------------------------------------------------------------

@dataclass
class ComponentMeta:
    """Metadata for registered components.

    Parameters
    ----------
    name : str
        Component name.
    component_type : str
        Type: 'adapter', 'optimizer', 'validator', 'storage', 'hook'.
    version : str
        Version string (e.g., '1.0.0').
    description : str
        Brief description.
    author : str
        Author name or organization.
    tags : List[str]
        Tags for categorization.
    config : Dict[str, Any]
        Default configuration.
    """
    name: str
    component_type: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global Registry
# ---------------------------------------------------------------------------

class TreeSkillRegistry:
    """Central registry for all pluggable components.

    This registry manages:
    - Adapters: Model API adapters (OpenAI, Anthropic, etc.)
    - Optimizers: Optimization strategies (TGD, etc.)
    - Validators: Prompt validation and testing
    - Storages: Storage backends (JSONL, SQLite, etc.)
    - Hooks: Lifecycle callbacks

    Thread-safe singleton pattern.
    """

    _instance = None
    _lock = __import__('threading').Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Storage for different component types
        self._adapters: Dict[str, Type[ModelAdapter]] = {}
        self._optimizers: Dict[str, Type] = {}
        self._validators: Dict[str, Type] = {}
        self._storages: Dict[str, Type] = {}
        self._hooks: Dict[str, List[Callable]] = {}

        # Metadata storage
        self._adapter_meta: Dict[str, ComponentMeta] = {}
        self._optimizer_meta: Dict[str, ComponentMeta] = {}
        self._validator_meta: Dict[str, ComponentMeta] = {}
        self._storage_meta: Dict[str, ComponentMeta] = {}

        # Default components
        self._default_adapter: Optional[str] = None
        self._default_optimizer: Optional[str] = None

        # Configuration
        self._config: Dict[str, Any] = {}

        self._initialized = True

        logger.info("TreeSkillRegistry initialized")

    # ------------------------------------------------------------------
    # Adapter Registration
    # ------------------------------------------------------------------

    def register_adapter(
        self,
        name: str,
        adapter_class: Type[ModelAdapter],
        meta: Optional[ComponentMeta] = None,
        set_default: bool = False,
    ) -> None:
        """Register a model adapter.

        Parameters
        ----------
        name : str
            Unique name for this adapter.
        adapter_class : Type[ModelAdapter]
            Adapter class (must inherit from ModelAdapter).
        meta : Optional[ComponentMeta]
            Component metadata.
        set_default : bool
            Set as default adapter.

        Examples
        --------
        >>> registry.register_adapter("openai", OpenAIAdapter, set_default=True)
        >>> adapter = registry.get_adapter("openai", model="gpt-4o-mini")
        """
        if name in self._adapters:
            logger.warning(f"Overwriting existing adapter: {name}")

        self._adapters[name] = adapter_class

        if meta:
            self._adapter_meta[name] = meta

        if set_default or self._default_adapter is None:
            self._default_adapter = name

        logger.info(f"Registered adapter: {name} (default={set_default})")

    def get_adapter(
        self,
        name: Optional[str] = None,
        **kwargs,
    ) -> ModelAdapter:
        """Get an adapter instance.

        Parameters
        ----------
        name : Optional[str]
            Adapter name. If None, uses default.
        **kwargs
            Arguments passed to adapter constructor.

        Returns
        -------
        ModelAdapter
            Adapter instance.

        Raises
        ------
        KeyError
            If adapter not found.
        """
        adapter_name = name or self._default_adapter

        if not adapter_name:
            raise ValueError(
                "No adapter registered. Register an adapter first or specify a name."
            )

        if adapter_name not in self._adapters:
            available = list(self._adapters.keys())
            raise KeyError(
                f"Adapter '{adapter_name}' not found. "
                f"Available adapters: {available}"
            )

        adapter_class = self._adapters[adapter_name]

        # Merge with default config if available
        if adapter_name in self._adapter_meta:
            default_config = self._adapter_meta[adapter_name].config
            merged_kwargs = {**default_config, **kwargs}
        else:
            merged_kwargs = kwargs

        return adapter_class(**merged_kwargs)

    def list_adapters(self) -> List[str]:
        """List all registered adapters."""
        return list(self._adapters.keys())

    def get_adapter_meta(self, name: str) -> Optional[ComponentMeta]:
        """Get metadata for an adapter."""
        return self._adapter_meta.get(name)

    # ------------------------------------------------------------------
    # Optimizer Registration
    # ------------------------------------------------------------------

    def register_optimizer(
        self,
        name: str,
        optimizer_class: Type,
        meta: Optional[ComponentMeta] = None,
        set_default: bool = False,
    ) -> None:
        """Register an optimizer."""
        self._optimizers[name] = optimizer_class

        if meta:
            self._optimizer_meta[name] = meta

        if set_default or self._default_optimizer is None:
            self._default_optimizer = name

        logger.info(f"Registered optimizer: {name}")

    def get_optimizer(
        self,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Get an optimizer instance."""
        optimizer_name = name or self._default_optimizer

        if not optimizer_name:
            raise ValueError("No optimizer registered.")

        if optimizer_name not in self._optimizers:
            available = list(self._optimizers.keys())
            raise KeyError(
                f"Optimizer '{optimizer_name}' not found. "
                f"Available: {available}"
            )

        optimizer_class = self._optimizers[optimizer_name]

        if optimizer_name in self._optimizer_meta:
            default_config = self._optimizer_meta[optimizer_name].config
            merged_kwargs = {**default_config, **kwargs}
        else:
            merged_kwargs = kwargs

        return optimizer_class(**merged_kwargs)

    def list_optimizers(self) -> List[str]:
        """List all registered optimizers."""
        return list(self._optimizers.keys())

    # ------------------------------------------------------------------
    # Lifecycle Hooks
    # ------------------------------------------------------------------

    def register_hook(
        self,
        event: str,
        callback: Callable,
        priority: int = 100,
    ) -> None:
        """Register a lifecycle hook.

        Parameters
        ----------
        event : str
            Event name. Supported events:
            - 'before_generate': Before model generation
            - 'after_generate': After generation
            - 'before_optimize': Before optimization
            - 'after_optimize': After optimization
            - 'on_gradient_computed': When gradient is computed
            - 'on_skill_saved': When skill is saved
            - 'on_error': On any error
        callback : Callable
            Callback function.
        priority : int
            Priority (lower = earlier). Default 100.

        Examples
        --------
        >>> def log_optimization(old_skill, new_skill, gradient):
        ...     print(f"Optimized: {old_skill.version} → {new_skill.version}")
        >>>
        >>> registry.register_hook('after_optimize', log_optimization)
        """
        if event not in self._hooks:
            self._hooks[event] = []

        self._hooks[event].append((priority, callback))
        # Sort by priority
        self._hooks[event].sort(key=lambda x: x[0])

        logger.debug(f"Registered hook for event '{event}'")

    def trigger_hook(
        self,
        event: str,
        *args,
        **kwargs,
    ) -> None:
        """Trigger all hooks for an event.

        Parameters
        ----------
        event : str
            Event name.
        *args, **kwargs
            Arguments passed to callbacks.
        """
        callbacks = self._hooks.get(event, [])

        for priority, callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Hook callback failed for event '{event}': {e}",
                    exc_info=True,
                )
                # Trigger error hook
                if event != 'on_error':
                    self.trigger_hook('on_error', e, event, callback)

    # ------------------------------------------------------------------
    # Configuration Loading
    # ------------------------------------------------------------------

    def load_from_config(
        self,
        config_path: Union[str, Path],
    ) -> None:
        """Load all components from a YAML config file.

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to config.yaml.

        Examples
        --------
        config.yaml:
        ```yaml
        adapters:
          openai:
            class: treeskill.adapters.openai.OpenAIAdapter
            default: true
            config:
              model: gpt-4o-mini
              temperature: 0.7

          claude:
            class: treeskill.adapters.anthropic.AnthropicAdapter
            config:
              model: claude-3-5-sonnet-20241022

        optimizers:
          default:
            class: treeskill.optimizer.TrainFreeOptimizer
            config:
              max_steps: 3
              gradient_accumulation_steps: 5

        hooks:
          after_optimize:
            - my_module.log_to_wandb
            - my_module.notify_slack
        ```
        """
        import yaml

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._config = config

        # Load adapters
        for name, cfg in config.get('adapters', {}).items():
            cls = self._import_class(cfg['class'])

            meta = ComponentMeta(
                name=name,
                component_type='adapter',
                config=cfg.get('config', {}),
            )

            self.register_adapter(
                name=name,
                adapter_class=cls,
                meta=meta,
                set_default=cfg.get('default', False),
            )

        # Load optimizers
        for name, cfg in config.get('optimizers', {}).items():
            cls = self._import_class(cfg['class'])

            meta = ComponentMeta(
                name=name,
                component_type='optimizer',
                config=cfg.get('config', {}),
            )

            self.register_optimizer(
                name=name,
                optimizer_class=cls,
                meta=meta,
                set_default=cfg.get('default', False),
            )

        # Load hooks
        for event, callbacks in config.get('hooks', {}).items():
            for callback_path in callbacks:
                callback = self._import_class(callback_path)
                self.register_hook(event, callback)

        logger.info(f"Loaded config from {config_path}")

    def _import_class(self, class_path: str) -> Type:
        """Dynamically import a class by path.

        Parameters
        ----------
        class_path : str
            Full path like 'treeskill.adapters.openai.OpenAIAdapter'

        Returns
        -------
        Type
            The imported class.
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset registry (mainly for testing)."""
        self._adapters.clear()
        self._optimizers.clear()
        self._validators.clear()
        self._storages.clear()
        self._hooks.clear()
        self._adapter_meta.clear()
        self._optimizer_meta.clear()
        self._default_adapter = None
        self._default_optimizer = None
        self._config.clear()
        logger.info("Registry reset")

    def summary(self) -> Dict[str, Any]:
        """Get a summary of registered components."""
        return {
            'adapters': {
                'count': len(self._adapters),
                'names': list(self._adapters.keys()),
                'default': self._default_adapter,
            },
            'optimizers': {
                'count': len(self._optimizers),
                'names': list(self._optimizers.keys()),
                'default': self._default_optimizer,
            },
            'hooks': {
                event: len(callbacks)
                for event, callbacks in self._hooks.items()
            },
        }


# ---------------------------------------------------------------------------
# Global Registry Instance
# ---------------------------------------------------------------------------

registry = TreeSkillRegistry()


# ---------------------------------------------------------------------------
# Decorator Functions
# ---------------------------------------------------------------------------

def adapter(name: str, set_default: bool = False, meta: Optional[ComponentMeta] = None):
    """Decorator to register an adapter.

    Parameters
    ----------
    name : str
        Adapter name.
    set_default : bool
        Set as default adapter.
    meta : Optional[ComponentMeta]
        Component metadata.

    Examples
    --------
    >>> from treeskill import adapter, BaseModelAdapter
    >>>
    >>> @adapter("my-custom")
    ... class MyAdapter(BaseModelAdapter):
    ...     def generate(self, prompt, **kwargs):
    ...         return "Hello"
    >>>
    >>> # Use it
    >>> from treeskill import registry
    >>> adapter = registry.get_adapter("my-custom")
    """
    def decorator(cls):
        registry.register_adapter(name, cls, meta=meta, set_default=set_default)
        return cls
    return decorator


def optimizer(name: str, set_default: bool = False, meta: Optional[ComponentMeta] = None):
    """Decorator to register an optimizer.

    Examples
    --------
    >>> from treeskill import optimizer
    >>>
    >>> @optimizer("aggressive")
    ... class AggressiveOptimizer:
    ...     pass
    """
    def decorator(cls):
        registry.register_optimizer(name, cls, meta=meta, set_default=set_default)
        return cls
    return decorator


def hook(event: str, priority: int = 100):
    """Decorator to register a lifecycle hook.

    Parameters
    ----------
    event : str
        Event name.
    priority : int
        Priority (lower = earlier).

    Examples
    --------
    >>> from treeskill import hook
    >>>
    >>> @hook('after_optimize')
    ... def log_optimization(old_skill, new_skill, gradient):
    ...     print(f"Optimized: {old_skill.version} → {new_skill.version}")
    """
    def decorator(func):
        registry.register_hook(event, func, priority=priority)
        return func
    return decorator
