#!/usr/bin/env python3
"""
Verification script for checking whether the evoskill package is installed
and imported correctly.

Usage:
1. Activate the conda environment: conda activate pr
2. Install dependencies: pip install -e .
3. Run verification: python tests/verify_rename.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_core_imports():
    """Test importing the core abstraction layer."""
    print("=" * 60)
    print("Test 1: Core abstraction imports")
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
        print("✅ Core abstraction imports succeeded")

        # Test object construction.
        prompt = TextPrompt(content="Test prompt")
        print(f"✅ Created TextPrompt: {prompt.content[:20]}...")

        gradient = SimpleGradient(text="Test gradient")
        print(f"✅ Created SimpleGradient: {gradient.text[:20]}...")

        return True
    except Exception as e:
        print(f"❌ Core import test failed:")
        traceback.print_exc()
        return False


def test_registry():
    """Test the plugin registry."""
    print("\n" + "=" * 60)
    print("Test 2: Plugin registry")
    print("=" * 60)

    try:
        from evoskill import registry, adapter, hook, ComponentMeta

        print("✅ Registry imported successfully")
        print(f"   - Registered adapters: {list(registry.list_adapters().keys())}")
        print(f"   - Registered optimizers: {list(registry.list_optimizers().keys())}")

        # Test decorators.
        @adapter("test-adapter")
        class TestAdapter:
            pass

        print("✅ @adapter decorator works")

        @hook('after_optimize')
        def test_hook(old, new, gradient):
            pass

        print("✅ @hook decorator works")

        return True
    except Exception as e:
        print(f"❌ Registry test failed:")
        traceback.print_exc()
        return False


def test_adapter_imports():
    """Test adapter imports, which may require optional dependencies."""
    print("\n" + "=" * 60)
    print("Test 3: Adapter imports (requires tiktoken and anthropic)")
    print("=" * 60)

    try:
        # Try importing adapters.
        try:
            from evoskill import OpenAIAdapter
            print("✅ OpenAIAdapter imported successfully")
        except ImportError as e:
            if "tiktoken" in str(e):
                print("⚠️  OpenAIAdapter requires tiktoken: pip install tiktoken")
            else:
                raise

        try:
            from evoskill import AnthropicAdapter
            print("✅ AnthropicAdapter imported successfully")
        except ImportError as e:
            if "anthropic" in str(e):
                print("⚠️  AnthropicAdapter requires anthropic: pip install anthropic")
            else:
                raise

        return True
    except Exception as e:
        print(f"❌ Adapter import test failed:")
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility through direct evoskill imports."""
    print("\n" + "=" * 60)
    print("Test 4: Direct evoskill import")
    print("=" * 60)

    try:
        from evoskill import TextPrompt, registry

        print("✅ evoskill imported successfully")

        prompt = TextPrompt(content="Import test")
        print(f"✅ Object creation succeeded: {prompt.content[:20]}...")

        return True
    except Exception as e:
        print(f"❌ Direct import test failed:")
        traceback.print_exc()
        return False


def test_package_info():
    """Test package metadata."""
    print("\n" + "=" * 60)
    print("Test 5: Package metadata")
    print("=" * 60)

    try:
        import evoskill

        print(f"✅ Package name: {evoskill.__name__}")
        print(f"✅ Version: {evoskill.__version__}")
        print(f"✅ Author: {evoskill.__author__}")

        # Check __all__.
        print(f"✅ Exported symbol count: {len(evoskill.__all__)}")
        print(f"   Main symbols: {', '.join(evoskill.__all__[:10])}...")

        return True
    except Exception as e:
        print(f"❌ Package metadata test failed:")
        traceback.print_exc()
        return False


def test_legacy_imports():
    """Test legacy API imports."""
    print("\n" + "=" * 60)
    print("Test 6: Legacy API (v0.1)")
    print("=" * 60)

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from evoskill import Skill, Trace, Message, SkillTree

            print("✅ Legacy API imports succeeded")
            print("   - Skill")
            print("   - Trace")
            print("   - Message")
            print("   - SkillTree")

        return True
    except Exception as e:
        print(f"❌ Legacy API test failed:")
        traceback.print_exc()
        return False


def main():
    print("\n" + "🔍 " * 20)
    print("evoskill package verification script")
    print("🔍 " * 20 + "\n")

    results = []

    # Run all tests.
    results.append(("Core imports", test_core_imports()))
    results.append(("Plugin registry", test_registry()))
    results.append(("Adapter imports", test_adapter_imports()))
    results.append(("Backward compatibility", test_backward_compatibility()))
    results.append(("Package metadata", test_package_info()))
    results.append(("Legacy API", test_legacy_imports()))

    # Summarize results.
    print("\n" + "=" * 60)
    print("📊 Test summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("\n" + "=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All tests passed! The evoskill package is installed and configured correctly.")
        print("\nNext steps:")
        print("  1. Run tests: pytest tests/test_*.py")
        print("  2. View docs: cat docs/RENAME_COMPLETE.md")
        print("  3. Start using it: from evoskill import TextPrompt")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check dependency installation:")
        print("  pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
