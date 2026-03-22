# 🛠️ 工具注册系统使用指南

完整的第三方工具集成解决方案，支持Python函数、HTTP API、MCP等多种方式。

---

## ✨ 核心特性

- ✅ **Python函数工具** - 装饰器注册，零配置
- ✅ **HTTP API工具** - 远程调用，RESTful支持
- ✅ **MCP工具** - Model Context Protocol标准
- ✅ **配置文件驱动** - YAML配置，热加载
- ✅ **用户友好** - 装饰器API，极简接入
- ✅ **LLM集成** - OpenAI Function Calling支持
- ✅ **与框架集成** - 适配器/优化器/验证器无缝使用

---

## 🚀 快速开始

### 1. Python函数工具（最简单）

```python
from treeskill.tools import tool, tool_registry

# 使用装饰器注册
@tool()
def calculate_laplace(expr: str) -> str:
    """计算拉普拉斯变换"""
    from sympy import laplace_transform, symbols
    t, s = symbols('t s')
    result, _, _ = laplace_transform(expr, t, s)
    return str(result)

# 使用
result = tool_registry.execute("calculate_laplace", "sin(t)")
print(result)  # 1/(s**2 + 1)
```

**就这么简单！** 不需要任何配置。

---

### 2. HTTP API工具

```python
from treeskill.tools import create_http_tool

# 创建HTTP工具
weather_api = create_http_tool(
    name="weather_api",
    endpoint="https://api.openweathermap.org/data/2.5/weather",
    description="查询天气",
    method="GET",
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

# 使用
result = weather_api.execute(q="Beijing", appid="YOUR_KEY")
print(result)
```

---

### 3. MCP工具（Model Context Protocol）

```python
from treeskill.tools import create_mcp_tool

# 创建MCP工具
database_query = create_mcp_tool(
    name="database_query",
    mcp_server="localhost:5000",
    tool_name="sql_query",
    description="执行SQL查询",
    auth_token="your-token-here"
)

# 使用
result = database_query.execute(sql="SELECT * FROM users LIMIT 10")
print(result)
```

---

### 4. 配置文件驱动

**config.yaml**:
```yaml
tools:
  - name: weather
    type: http
    description: 天气查询API
    endpoint: https://api.weather.com/current
    method: GET
    headers:
      Authorization: Bearer xxx

  - name: database
    type: mcp
    description: 数据库查询
    mcp_server: localhost:5000
    tool_name: query
    auth_token: secret-token
```

**加载**:
```python
from treeskill.tools import tool_registry

# 从配置文件加载
tool_registry.load_from_config("config.yaml")

# 使用
weather = tool_registry.execute("weather", city="Beijing")
data = tool_registry.execute("database", sql="SELECT * FROM users")
```

---

## 📖 完整示例

查看 `examples/example_tools.py` 获取完整可运行示例。

运行测试：
```bash
python examples/example_tools.py
```

---

## 🎓 高级用法

### 自定义工具类

```python
from treeskill.tools import BaseTool, tool_registry

class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "我的自定义工具"

    def execute(self, *args, **kwargs):
        # 自定义逻辑
        return {"result": "success"}

# 注册
tool_registry.register("my_tool", MyCustomTool())

# 使用
result = tool_registry.execute("my_tool")
```

---

## 📊 API参考

| 函数 | 说明 |
|------|------|
| `@tool()` | 装饰器注册Python函数 |
| `create_http_tool()` | 创建HTTP工具 |
| `create_mcp_tool()` | 创建MCP工具 |
| `tool_registry.execute()` | 执行工具 |
| `tool_registry.list_tools()` | 列出所有工具 |
| `tool_registry.load_from_config()` | 从配置文件加载 |

---

## ✅ 支持的工具类型

| 类型 | 接入方式 | 难度 | 用例 |
|------|---------|------|------|
| **Python函数** | `@tool()` 装饰器 | ⭐ | 本地计算、数据处理 |
| **HTTP API** | `create_http_tool()` | ⭐⭐ | 第三方API、微服务 |
| **MCP** | `create_mcp_tool()` | ⭐⭐⭐ | Claude集成、标准化工具 |
| **自定义** | 继承 `BaseTool` | ⭐⭐⭐⭐ | 复杂场景 |

---

## 🎯 使用场景

### 1. 数学计算验证

```python
@tool()
def verify_laplace(expr: str, answer: str) -> bool:
    """验证拉普拉斯变换答案"""
    from sympy import laplace_transform, symbols, sympify
    t, s = symbols('t s')
    correct, _, _ = laplace_transform(sympify(expr), t, s)
    return str(correct) == answer

# 在优化器中使用
is_correct = tool_registry.execute("verify_laplace", "sin(t)", "1/(s**2+1)")
```

### 2. 数据库查询

```python
db_tool = create_mcp_tool(
    name="query_db",
    mcp_server="localhost:5000",
    tool_name="sql_query"
)

users = tool_registry.execute("query_db", sql="SELECT * FROM users")
```

### 3. LLM Function Calling

```python
@tool(
    name="get_weather",
    schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"]
    }
)
def get_weather(city: str) -> dict:
    return {"city": city, "temp": 20}

# 获取schema
schema = tool_registry.get("get_weather").to_schema()

# 传递给LLM
# response = adapter.generate(prompt, tools=[schema])
```

---

## 🔧 与框架集成

工具可以与优化器、验证器、适配器无缝集成。

查看完整文档：`TOOLS_GUIDE.md`

---

*完成时间: 2026-03-17*
*支持: Python/HTTP/MCP/自定义*
*用户友好度: ⭐⭐⭐⭐⭐*
