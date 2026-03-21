# Slash Command Dropdown Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a slash-command dropdown in the chat CLI so typing `/` shows all available slash commands and typing a prefix filters them.

**Architecture:** Replace the main chat input call in `ChatCLI.run()` with a `prompt_toolkit` prompt session backed by a small completer that reads from `_COMMAND_SPECS`. Keep existing command parsing and execution paths unchanged so the feature is isolated to input handling.

**Tech Stack:** Python 3.12, Rich, prompt_toolkit, pytest

---

### Task 1: Add a failing completer test for full slash listing

**Files:**
- Modify: `tests/test_cli_audio.py`
- Test: `tests/test_cli_audio.py`

**Step 1: Write the failing test**

Add a test that builds the slash completer and asserts that input `/` returns command candidates including `/help`, `/audio`, `/select`, and `/quit`.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -k slash -v`
Expected: FAIL because the completer helper does not exist yet.

**Step 3: Write minimal implementation**

Add the slash completer helper in `evoskill/cli.py` using `_COMMAND_SPECS`.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -k slash -v`
Expected: PASS

### Task 2: Add a failing completer test for prefix filtering

**Files:**
- Modify: `tests/test_cli_audio.py`
- Test: `tests/test_cli_audio.py`

**Step 1: Write the failing test**

Add a test asserting that `/h` returns `/help` and excludes unrelated commands like `/audio`.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -k slash -v`
Expected: FAIL because filtering behavior is not fully implemented yet.

**Step 3: Write minimal implementation**

Implement prefix filtering in the completer so only matching slash commands are returned.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -k slash -v`
Expected: PASS

### Task 3: Add a failing test for non-slash input

**Files:**
- Modify: `tests/test_cli_audio.py`
- Test: `tests/test_cli_audio.py`

**Step 1: Write the failing test**

Add a test asserting that normal text like `hello` returns no slash completions.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -k slash -v`
Expected: FAIL because the completer currently suggests regardless of prefix.

**Step 3: Write minimal implementation**

Guard the completer so it only yields candidates when the input line starts with `/`.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -k slash -v`
Expected: PASS

### Task 4: Wire prompt_toolkit into the chat loop

**Files:**
- Modify: `evoskill/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli_audio.py`

**Step 1: Write the failing test**

Add a light construction-level test that the chat prompt session helper can be created without executing the interactive loop.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -k prompt_session -v`
Expected: FAIL because the helper or dependency wiring does not exist yet.

**Step 3: Write minimal implementation**

- Add `prompt_toolkit` to runtime dependencies.
- Create a prompt-session helper in `evoskill/cli.py`.
- Use that helper inside `ChatCLI.run()` in place of the current `Prompt.ask()` call.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -k prompt_session -v`
Expected: PASS

### Task 5: Verify focused behavior and regression safety

**Files:**
- Modify: `README.md`
- Test: `tests/test_cli_audio.py`

**Step 1: Update docs**

Mention that slash commands now support interactive dropdown suggestions in the CLI.

**Step 2: Run focused tests**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -v`
Expected: PASS

**Step 3: Run one broader regression check**

Run: `source .venv/bin/activate && pytest tests/test_package_import.py -v`
Expected: PASS

**Step 4: Manual sanity note**

Record that true dropdown rendering is best verified by launching the CLI in a real terminal because pytest will only cover completion logic, not terminal UI visuals.
