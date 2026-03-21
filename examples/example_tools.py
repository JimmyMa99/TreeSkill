"""
工具注册系统 - completeExample

演示如何使用装饰器、HTTP、MCP等多种Method注册和使用自定义工具。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoskill.tools import (
    tool,
    tool_registry,
    create_http_tool,
    create_mcp_tool,
)


# ===========================================================================
# Example 1: 使用装饰器注册Python函数（最简单）
# ===========================================================================

@tool()
def calculate_laplace(expr: str) -> str:
    """计算拉普拉斯变换

    参数:
        expr: 数学表达式，如 "sin(t)", "exp(-t)"

    返回:
        拉普拉斯变换results
    """
    try:
        from sympy import laplace_transform, symbols, sympify

        t, s = symbols('t s')
        expr_sym = sympify(expr)
        result, _, _ = laplace_transform(expr_sym, t, s)
        return str(result)

    except Exception as e:
        return f"错误: {e}"


@tool(description="计算两个数的和")
def add(a: int, b: int) -> int:
    """简单的加法函数"""
    return a + b


@tool(
    name="weather_query",
    description="查询城市天气（模拟）",
    schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
    }
)
def get_weather_mock(city: str, unit: str = "celsius") -> dict:
    """模拟天气查询API"""
    # 实际中会调用真实API
    weather_data = {
        "beijing": {"temp": 20, "condition": "晴"},
        "shanghai": {"temp": 25, "condition": "多云"},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return {
            "city": city,
            "temperature": data["temp"],
            "unit": unit,
            "condition": data["condition"]
        }
    else:
        return {"error": f"未找到城市: {city}"}


# ===========================================================================
# Example 2: HTTP工具（远程API）
# ===========================================================================

def example_http_tool():
    """CreateHTTP工具Example"""

    # 方法1: 使用便捷函数
    weather_api = create_http_tool(
        name="weather_api",
        endpoint="https://api.openweathermap.org/data/2.5/weather",
        description="OpenWeatherMap API",
        headers={"Authorization": "Bearer YOUR_API_KEY"}
    )

    # 使用
    # result = weather_api.execute(q="Beijing", appid="YOUR_KEY")


# ===========================================================================
# Example 3: MCP工具（Model Context Protocol）
# ===========================================================================

def example_mcp_tool():
    """CreateMCP工具Example"""

    # 方法1: 使用便捷函数
    database_query = create_mcp_tool(
        name="database_query",
        mcp_server="localhost:5000",
        tool_name="sql_query",
        description="执行SQL查询",
        auth_token="your-token-here"
    )

    # 使用
    # result = database_query.execute(sql="SELECT * FROM users LIMIT 10")


# ===========================================================================
# Example 4: Configfile驱动
# ===========================================================================

def example_config_driven():
    """从ConfigfileLoad工具"""

    # CreateConfigfile
    config_yaml = """
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
"""

    # 写入临时file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_path = f.name

    # Load
    # tool_registry.load_from_config(config_path)

    # 清理
    import os
    os.unlink(config_path)


# ===========================================================================
# Example 5: 在优化器中使用工具
# ===========================================================================

def example_use_in_optimizer():
    """在优化Flow中使用工具"""

    from evoskill.core import TextPrompt, ConversationExperience, CompositeFeedback
    from evoskill.core.experience import FeedbackType

    # 假设我们有一个数学助手，需要使用拉普拉斯变换工具
    prompt = TextPrompt(
        content="你是一个数学助手，帮助用户解决微积分问题。",
        version="v1.0"
    )

    # Failure case：助手没有正确计算拉普拉斯变换
    failure = ConversationExperience(
        messages=[{"role": "user", "content": "计算 sin(t) 的拉普拉斯变换"}],
        response="让我想想...应该是 1/(s^2+1)...",
        feedback=CompositeFeedback(
            feedback_type=FeedbackType.CORRECTION,
            critique="计算results不complete",
            correction="正确答案是 1/(s^2+1)",
        )
    )

    # 在优化过程中，我们可以调用工具来验证答案
    # 优化器可以使用 tool_registry.execute("calculate_laplace", "sin(t)")
    # 来获得正确答案，然后生成更好的提示词


# ===========================================================================
# 主函数 - Run所有Example
# ===========================================================================

def main():
    print("=" * 80)
    print("工具注册系统Example")
    print("=" * 80)

    # 查看已注册的工具
    print("\n【已注册的工具】")
    print(f"工具列表: {tool_registry.list_tools()}")

    # 执行工具
    print("\n【执行工具】")

    # 1. Python函数工具
    print("\n1. 计算拉普拉斯变换:")
    result = tool_registry.execute("calculate_laplace", "sin(t)")
    print(f"   L[sin(t)] = {result}")

    result = tool_registry.execute("calculate_laplace", "exp(-t)")
    print(f"   L[exp(-t)] = {result}")

    # 2. 简单加法
    print("\n2. 加法:")
    result = tool_registry.execute("add", 5, 3)
    print(f"   5 + 3 = {result}")

    # 3. 天气查询（模拟）
    print("\n3. 天气查询:")
    result = tool_registry.execute("weather_query", city="Beijing")
    print(f"   北京天气: {result}")

    result = tool_registry.execute("weather_query", city="Shanghai", unit="fahrenheit")
    print(f"   上海天气: {result}")

    # 获取工具schema
    print("\n【工具Schema（用于LLM function calling）】")
    tool = tool_registry.get("weather_query")
    print(f"Schema: {tool.to_schema()}")

    # Example：HTTP和MCP
    print("\n【其他工具类型】")
    print("✓ HTTP工具: 使用 create_http_tool() Create")
    print("✓ MCP工具: 使用 create_mcp_tool() Create")
    print("✓ Configfile: 使用 tool_registry.load_from_config() Load")

    print("\n" + "=" * 80)
    print("✅ ExampleRun完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
