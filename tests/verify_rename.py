#!/usr/bin/env python3
"""
验证脚本：测试 evoskill 包是否正确安装和导入

使用方法：
1. 激活 conda 环境：conda activate pr
2. 安装依赖：pip install -e .
3. 运行验证：python tests/verify_rename.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_core_imports():
    """测试核心抽象层导入"""
    print("=" * 60)
    print("测试 1: 核心抽象层导入")
    print("=" * 60)

    try:
        from evoskill.core import (
            OptimizablePrompt,
            TextPrompt,
            MultimodalPrompt,
            SimpleGradient,
            ConversationExperience,
            BaseModelAdapter,
        )
        print("✅ 核心抽象层导入成功")

        # 测试创建对象
        prompt = TextPrompt(content="测试提示词")
        print(f"✅ 创建 TextPrompt: {prompt.content[:20]}...")

        gradient = SimpleGradient(text="测试梯度")
        print(f"✅ 创建 SimpleGradient: {gradient.text[:20]}...")

        return True
    except Exception as e:
        print(f"❌ 核心导入失败:")
        traceback.print_exc()
        return False


def test_registry():
    """测试插件注册表"""
    print("\n" + "=" * 60)
    print("测试 2: 插件注册表")
    print("=" * 60)

    try:
        from evoskill import registry, adapter, hook, ComponentMeta

        print("✅ Registry 导入成功")
        print(f"   - 已注册适配器: {list(registry.list_adapters().keys())}")
        print(f"   - 已注册优化器: {list(registry.list_optimizers().keys())}")

        # 测试装饰器
        @adapter("test-adapter")
        class TestAdapter:
            pass

        print("✅ @adapter 装饰器工作正常")

        @hook('after_optimize')
        def test_hook(old, new, gradient):
            pass

        print("✅ @hook 装饰器工作正常")

        return True
    except Exception as e:
        print(f"❌ Registry 测试失败:")
        traceback.print_exc()
        return False


def test_adapter_imports():
    """测试适配器导入（可能需要依赖）"""
    print("\n" + "=" * 60)
    print("测试 3: 适配器导入（需要 tiktoken 和 anthropic）")
    print("=" * 60)

    try:
        # 尝试导入适配器
        try:
            from evoskill import OpenAIAdapter
            print("✅ OpenAIAdapter 导入成功")
        except ImportError as e:
            if "tiktoken" in str(e):
                print("⚠️  OpenAIAdapter 需要 tiktoken: pip install tiktoken")
            else:
                raise

        try:
            from evoskill import AnthropicAdapter
            print("✅ AnthropicAdapter 导入成功")
        except ImportError as e:
            if "anthropic" in str(e):
                print("⚠️  AnthropicAdapter 需要 anthropic: pip install anthropic")
            else:
                raise

        return True
    except Exception as e:
        print(f"❌ 适配器导入失败:")
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """测试向后兼容（evo_framework 已移除，此测试验证 evoskill 直接导入）"""
    print("\n" + "=" * 60)
    print("测试 4: evoskill 直接导入")
    print("=" * 60)

    try:
        from evoskill import TextPrompt, registry

        print("✅ evoskill 导入成功")

        prompt = TextPrompt(content="导入测试")
        print(f"✅ 创建对象成功: {prompt.content[:20]}...")

        return True
    except Exception as e:
        print(f"❌ 导入测试失败:")
        traceback.print_exc()
        return False


def test_package_info():
    """测试包信息"""
    print("\n" + "=" * 60)
    print("测试 5: 包信息")
    print("=" * 60)

    try:
        import evoskill

        print(f"✅ 包名: {evoskill.__name__}")
        print(f"✅ 版本: {evoskill.__version__}")
        print(f"✅ 作者: {evoskill.__author__}")

        # 检查 __all__
        print(f"✅ 导出符号数量: {len(evoskill.__all__)}")
        print(f"   主要符号: {', '.join(evoskill.__all__[:10])}...")

        return True
    except Exception as e:
        print(f"❌ 包信息测试失败:")
        traceback.print_exc()
        return False


def test_legacy_imports():
    """测试遗留API导入"""
    print("\n" + "=" * 60)
    print("测试 6: 遗留 API（v0.1）")
    print("=" * 60)

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from evoskill import Skill, Trace, Message, SkillTree

            print("✅ 遗留 API 导入成功")
            print("   - Skill")
            print("   - Trace")
            print("   - Message")
            print("   - SkillTree")

        return True
    except Exception as e:
        print(f"❌ 遗留 API 测试失败:")
        traceback.print_exc()
        return False


def main():
    print("\n" + "🔍 " * 20)
    print("evoskill 包验证脚本")
    print("🔍 " * 20 + "\n")

    results = []

    # 运行所有测试
    results.append(("核心导入", test_core_imports()))
    results.append(("插件注册表", test_registry()))
    results.append(("适配器导入", test_adapter_imports()))
    results.append(("向后兼容", test_backward_compatibility()))
    results.append(("包信息", test_package_info()))
    results.append(("遗留API", test_legacy_imports()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 60)

    if passed == total:
        print("\n🎉 所有测试通过！evoskill 包已正确安装和配置。")
        print("\n下一步:")
        print("  1. 运行测试: pytest tests/test_*.py")
        print("  2. 查看文档: cat docs/RENAME_COMPLETE.md")
        print("  3. 开始使用: from evoskill import TextPrompt")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查依赖安装:")
        print("  pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
