# Audio Input Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add minimal OpenAI-compatible audio input support to EvoSkill so users can attach a local audio file in the CLI and send it as part of a chat request while keeping text-only output.

**Architecture:** Extend the message schema with an audio content part that mirrors the existing image content part, then teach the CLI to stage audio attachments and serialize them through the existing `LLMClient.generate()` path. Keep the change narrow: no audio output, no new high-level API surface beyond the existing chat loop.

**Tech Stack:** Python 3.10, Pydantic v2, OpenAI Python SDK, Rich, pytest

---

### Task 1: Define audio content behavior in tests

**Files:**
- Modify: `tests/test_core_abstractions.py`
- Modify: `tests/test_openai_adapter.py`

**Step 1: Write the failing test**

Add tests that assert:
- message content can contain an `audio_url` content part
- multimodal prompt audio is preserved in serialized model input

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_core_abstractions.py -q`
Expected: FAIL because schema currently lacks an audio content part.

**Step 3: Write minimal implementation**

Add the schema/model changes needed to represent audio content.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_core_abstractions.py -q`
Expected: PASS

### Task 2: Define CLI audio attachment behavior in tests

**Files:**
- Create: `tests/test_cli_audio.py`
- Modify: `evoskill/cli.py`

**Step 1: Write the failing test**

Add tests that assert:
- `/audio <path>` stages an audio attachment
- the next user message includes both text and the encoded audio part
- pending attachments are cleared after message creation

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -q`
Expected: FAIL because the CLI has no `/audio` command.

**Step 3: Write minimal implementation**

Add `_cmd_audio`, shared media encoding, and pending audio state in the CLI.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_cli_audio.py -q`
Expected: PASS

### Task 3: Verify the focused audio-input path

**Files:**
- Modify: `evoskill/schema.py`
- Modify: `evoskill/cli.py`
- Modify: `README.md` (optional, if command table needs update)

**Step 1: Run focused verification**

Run: `source .venv/bin/activate && pytest tests/test_core_abstractions.py tests/test_cli_audio.py -q`

**Step 2: Run one broader safety check**

Run: `source .venv/bin/activate && pytest tests/test_core_abstractions.py -q`

**Step 3: Summarize scope**

Note explicitly that this milestone supports audio input only and does not add audio output or realtime behavior.
