"""Tool Registry - user-defined tool registration system.

This module provides a flexible registry for:
1. Python function tools (direct invocation)
2. HTTP API tools (remote invocation)
3. MCP tools (Model Context Protocol)
4. User-defined tools

You can register tools with decorators or configuration files.
"""

from __future__ import annotations

import json
import logging
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


# ===========================================================================
# Abstract tool base class
# ===========================================================================

class BaseTool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description used by the LLM."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        pass

    def to_schema(self) -> Dict[str, Any]:
        """Generate an OpenAI function-calling schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }


# ===========================================================================
# Concrete tool implementations
# ===========================================================================

@dataclass
class PythonFunctionTool(BaseTool):
    """Python function tool with direct invocation.

    Example:
        @tool("laplace_transform")
        def laplace_transform(expr: str) -> str:
            '''Compute a Laplace transform.'''
            from sympy import laplace_transform, sympify
            expr = sympify(expr)
            result = laplace_transform(expr, t, s)
            return str(result)
    """

    _name: str
    _description: str
    func: Callable
    parameters_schema: Optional[Dict] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, *args, **kwargs) -> Any:
        """Call the wrapped Python function directly."""
        logger.info(f"Executing tool [{self.name}]: args={args}, kwargs={kwargs}")
        try:
            result = self.func(*args, **kwargs)
            logger.info(f"Tool [{self.name}] executed successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool [{self.name}] execution failed: {e}")
            raise

    def to_schema(self) -> Dict[str, Any]:
        """Generate the schema."""
        if self.parameters_schema:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }

        # Default schema.
        return super().to_schema()


@dataclass
class HTTPTool(BaseTool):
    """HTTP API tool for remote requests.

    Example:
        tool = HTTPTool(
            name="weather_api",
            description="Fetch weather information",
            endpoint="https://api.weather.com/current",
            method="GET",
            headers={"Authorization": "Bearer xxx"},
        )

        result = tool.execute(city="Beijing")
    """

    _name: str
    _description: str
    endpoint: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, **kwargs) -> Any:
        """Send the HTTP request."""
        logger.info(f"Executing HTTP tool [{self.name}]: endpoint={self.endpoint}, params={kwargs}")

        try:
            if self.method.upper() == "GET":
                response = requests.get(
                    self.endpoint,
                    params=kwargs,
                    headers=self.headers,
                    timeout=self.timeout
                )
            else:  # POST
                response = requests.post(
                    self.endpoint,
                    json=kwargs,
                    headers=self.headers,
                    timeout=self.timeout
                )

            response.raise_for_status()
            result = response.json()
            logger.info(f"HTTP tool [{self.name}] executed successfully")
            return result

        except Exception as e:
            logger.error(f"HTTP tool [{self.name}] execution failed: {e}")
            raise


@dataclass
class MCPTool(BaseTool):
    """MCP (Model Context Protocol) tool.

    Example:
        tool = MCPTool(
            name="database_query",
            description="Query the database",
            mcp_server="localhost:5000",
            tool_name="query",
        )

        result = tool.execute(sql="SELECT * FROM users")
    """

    _name: str
    _description: str
    mcp_server: str
    tool_name: str
    auth_token: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, **kwargs) -> Any:
        """Invoke the tool through the MCP protocol."""
        logger.info(f"Executing MCP tool [{self.name}]: server={self.mcp_server}, params={kwargs}")

        try:
            # Simplified MCP protocol call example.
            payload = {
                "tool": self.tool_name,
                "parameters": kwargs
            }

            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            response = requests.post(
                f"http://{self.mcp_server}/invoke",
                json=payload,
                headers=headers,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()
            logger.info(f"MCP tool [{self.name}] executed successfully")
            return result

        except Exception as e:
            logger.error(f"MCP tool [{self.name}] execution failed: {e}")
            raise


# ===========================================================================
# Tool registry
# ===========================================================================

class ToolRegistry:
    """Registry that manages all registered tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(
        self,
        name: str,
        tool: BaseTool,
        description: Optional[str] = None,
        override: bool = False
    ):
        """Register a tool.

        Parameters:
            name: Tool name
            tool: Tool instance
            description: Optional tool description
            override: Whether to overwrite an existing tool
        """
        if name in self._tools and not override:
            raise ValueError(f"Tool '{name}' already exists; use override=True to replace it")

        self._tools[name] = tool
        logger.info(f"✓ Registered tool: {name}")

    def get(self, name: str) -> BaseTool:
        """Get a registered tool."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def execute(self, name: str, *args, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        return tool.execute(*args, **kwargs)

    def load_from_config(self, config_path: Union[str, Path]):
        """Load tools from a configuration file.

        Example configuration format:
        ```yaml
        tools:
          - name: weather
            type: http
            endpoint: https://api.weather.com/current
            method: GET
            headers:
              Authorization: Bearer xxx

          - name: database
            type: mcp
            mcp_server: localhost:5000
            tool_name: query
        ```
        """
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for tool_config in config.get('tools', []):
            tool = self._create_tool_from_config(tool_config)
            self.register(tool_config['name'], tool)

    def _create_tool_from_config(self, config: Dict) -> BaseTool:
        """Create a tool from a configuration dictionary."""
        tool_type = config.get('type', 'function')

        if tool_type == 'http':
            return HTTPTool(
                _name=config['name'],
                _description=config.get('description', ''),
                endpoint=config['endpoint'],
                method=config.get('method', 'GET'),
                headers=config.get('headers', {}),
            )
        elif tool_type == 'mcp':
            return MCPTool(
                _name=config['name'],
                _description=config.get('description', ''),
                mcp_server=config['mcp_server'],
                tool_name=config.get('tool_name', config['name']),
                auth_token=config.get('auth_token'),
            )
        else:
            raise ValueError(f"Unsupported tool type: {tool_type}")


# Global tool registry
tool_registry = ToolRegistry()


# ===========================================================================
# Decorator-based user-friendly registration
# ===========================================================================

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    schema: Optional[Dict] = None,
):
    """Decorator that registers a Python function as a tool.

    Example 1: simple usage
        ```python
        @tool()
        def calculate_laplace(expr: str) -> str:
            '''Compute a Laplace transform.'''
            from sympy import laplace_transform, symbols, sympify
            t, s = symbols('t s')
            expr = sympify(expr)
            result = laplace_transform(expr, t, s)
            return str(result[0])
        ```

    Example 2: custom name and description
        ```python
        @tool(name="math_laplace", description="Compute the Laplace transform of a mathematical expression")
        def my_laplace_function(expr: str) -> str:
            ...
        ```

    Example 3: with a schema
        ```python
        @tool(
            name="weather",
            schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        )
        def get_weather(city: str, unit: str = "celsius") -> dict:
            '''Fetch weather information.'''
            ...
        ```
    """
    def decorator(func: Callable):
        # Tool name.
        tool_name = name or func.__name__

        # Tool description.
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

        # Create the tool instance.
        tool_instance = PythonFunctionTool(
            _name=tool_name,
            _description=tool_desc,
            func=func,
            parameters_schema=schema,
        )

        # Register it in the global registry.
        tool_registry.register(tool_name, tool_instance)

        # Return the original function so it remains callable.
        return func

    return decorator


# ===========================================================================
# Convenience helpers for quick tool creation
# ===========================================================================

def create_http_tool(
    name: str,
    endpoint: str,
    description: str = "",
    method: str = "GET",
    headers: Optional[Dict] = None,
) -> HTTPTool:
    """Create an HTTP tool quickly.

    Example:
        ```python
        weather_tool = create_http_tool(
            name="weather",
            endpoint="https://api.weather.com/current",
            description="Fetch weather information",
            headers={"Authorization": "Bearer xxx"}
        )

        result = weather_tool.execute(city="Beijing")
        ```
    """
    tool = HTTPTool(
        _name=name,
        _description=description,
        endpoint=endpoint,
        method=method,
        headers=headers or {},
    )

    tool_registry.register(name, tool)
    return tool


def create_mcp_tool(
    name: str,
    mcp_server: str,
    tool_name: str,
    description: str = "",
    auth_token: Optional[str] = None,
) -> MCPTool:
    """Create an MCP tool quickly.

    Example:
        ```python
        db_tool = create_mcp_tool(
            name="database",
            mcp_server="localhost:5000",
            tool_name="query",
            description="Query the database"
        )

        result = db_tool.execute(sql="SELECT * FROM users")
        ```
    """
    tool = MCPTool(
        _name=name,
        _description=description,
        mcp_server=mcp_server,
        tool_name=tool_name,
        auth_token=auth_token,
    )

    tool_registry.register(name, tool)
    return tool


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    # Base class
    "BaseTool",

    # Tool types
    "PythonFunctionTool",
    "HTTPTool",
    "MCPTool",

    # Registry
    "ToolRegistry",
    "tool_registry",

    # Decorator
    "tool",

    # Convenience helpers
    "create_http_tool",
    "create_mcp_tool",
]
