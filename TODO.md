# TODO

This file tracks the confirmed unfinished tasks in the current EvoSkill codebase.

**Last Updated**: March 21, 2026

## ✅ Completed Items

The following items have been implemented and are no longer tracked:

- ✅ **AnthropicAdapter** - Implemented in `evoskill/adapters/anthropic.py`
- ✅ **AutoValidator** - Implemented in `evoskill/core/validators.py`
- ✅ **Tree-aware optimization** - Implemented in `evoskill/core/tree_optimizer.py` with auto-split/prune
- ✅ **Skill tree management** - Full implementation in `evoskill/skill_tree.py` (add/split/merge/prune/graft)
- ✅ **Multimodal support** - Image and audio attachment in CLI
- ✅ **Checkpoint management** - Rollback and state management in `evoskill/checkpoint.py`
- ✅ **Resume state** - Optimization state persistence in `evoskill/resume.py`
- ✅ **Built-in tools** - Python function, HTTP, and MCP tool support in `evoskill/tools.py`
- ✅ **Fix duplicate trace writes when feedback is attached** - `/bad` and `/rewrite` now update traces by ID via `TraceStorage.upsert()` with load-time deduplication in `evoskill/storage.py`
- ✅ **Fix SiliconFlow pytest return-value warning** - `tests/test_openai_siliconflow.py` now separates reusable script logic from the pytest test function
- ✅ **DPO preference data export** - `/export-dpo` command in CLI, `TraceStorage.export_dpo()` and `get_dpo_pairs()` in `evoskill/storage.py`
- ✅ **Human-in-the-loop annotation mode** - `--annotate --dataset` with auto/manual judge toggle (`/auto`, `/manual`), implemented in `evoskill/annotate.py`
- ✅ **Non-interactive resume skip** - `--no-resume` flag to skip interactive resume prompt, fixes hang in CI/script environments
- ✅ **Dataset-driven evaluation pipeline** - `DataLoader` in `evoskill/dataset.py` and `Evaluator` in `evoskill/evaluator.py` for ChatML JSONL datasets
- ✅ **English documentation** - Full English README at `docs_en/README.md` with language switcher

## P0 - Critical

- [ ] **Unify the main optimization path onto the newer multi-step optimizer**
  
  The CLI still uses `APOEngine`, which performs a single optimization cycle.
  The newer `TrainFreeOptimizer` in `evoskill/core/optimizer.py` supports multi-step optimization.
  
  **Files to update**:
  - `evoskill/cli.py` - migrate from `APOEngine` to `TrainFreeOptimizer`
  - `evoskill/optimizer.py` - consider deprecating or integrating with core optimizer
  - `evoskill/core/optimizer.py` - ensure it's production-ready
  
  **Notes**: `APOEngine` is still functional but doesn't leverage the full TGD loop with `max_steps`.

- [ ] **Implement automatic few-shot example construction from high-quality traces**
  
  `few_shot_messages` exists in the schema, but there is no pipeline that promotes strong traces into reusable examples.
  
  **Proposed workflow**:
  1. Identify high-scoring traces (score > 0.8)
  2. Extract user input + ideal response as Message pair
  3. Add to `few_shot_messages` with version tracking
  4. Limit to top-K examples to avoid context bloat
  
  **Files to update**:
  - `evoskill/schema.py` - add example extraction metadata
  - `evoskill/skill.py` - add `promote_to_few_shot()` method
  - `evoskill/core/optimizer.py` - integrate into optimization loop

## P1 - Important

- [ ] **Add automatic routing for skill trees**
  
  Users currently have to manually switch sub-skills with `/select`; there is no automatic leaf selection based on user input.
  
  **Proposed solution**: Implement a router that analyzes user input and selects the most appropriate leaf skill automatically.
  
  **Files to update**:
  - `evoskill/cli.py` - add auto-routing before message compilation
  - `evoskill/skill_tree.py` - add `route(input_text: str) -> SkillNode` method
  - `evoskill/core/tree_optimizer.py` - potentially use LLM for routing decision

- [ ] **Add hard thresholds for auto split and auto prune decisions**
  
  The current structure evolution relies heavily on LLM judgment and needs statistical guardrails.
  
  **Proposed thresholds**:
  - Split: minimum 5 samples with contradictory feedback patterns
  - Prune: performance < 0.3 for 3+ consecutive rounds, usage_count < 2
  - Protection: newly created nodes protected for 2 rounds (already implemented)
  
  **Files to update**:
  - `evoskill/core/tree_optimizer.py` - add threshold checks in `analyze_split_need()` and `analyze_prune_need()`
  - `evoskill/core/optimizer_config.py` - add threshold configuration options

- [ ] **Complete the automatic merge workflow for skill trees**
  
  `SkillTree.merge()` exists, but there is no full merge analysis, trigger path, or CLI/product flow around it.
  
  **Missing pieces**:
  - Merge detection logic (when should merge be suggested?)
  - CLI command `/merge` with interactive flow
  - Merge preview showing combined prompt
  
  **Files to update**:
  - `evoskill/skill_tree.py` - enhance `merge()` with analysis
  - `evoskill/core/tree_optimizer.py` - add `analyze_merge_need()`
  - `evoskill/cli.py` - add `/merge` command

## P2 - Nice to Have

- [ ] **Improve storage concurrency safety**
  
  The current JSONL append model may fail under multi-process usage and likely needs file locking or a different backend.
  
  **Proposed solutions**:
  - Add file locking with `fcntl` (Unix) or `msvcrt` (Windows)
  - Consider SQLite backend for high-concurrency scenarios
  - Add atomic write operations
  
  **Files to update**:
  - `evoskill/storage.py` - add locking mechanism
  - `evoskill/config.py` - add storage backend configuration

- [ ] **Expand multimodal failure analysis beyond placeholder handling**
  
  Multimodal optimization support exists at a basic level, but detailed content-aware feedback analysis is still incomplete.
  
  **Missing features**:
  - Image content analysis in gradient computation
  - Audio transcription integration for feedback analysis
  - Multimodal few-shot example extraction
  
  **Files to update**:
  - `evoskill/core/optimizer.py` - enhance `compute_gradient()` for multimodal
  - `evoskill/core/abc.py` - add multimodal-specific methods to `ModelAdapter`

## Notes

- **v0.2 architecture**: The new `evoskill/core/` package provides clean abstractions (`ModelAdapter`, `OptimizablePrompt`, `TrainFreeOptimizer`)
- **Backward compatibility**: Legacy `evo_framework/` still works with deprecation warnings
- **YAML encoding**: `skill.save()` uses `allow_unicode=True` for readable Chinese
- **Configuration priority**: Environment variables > `.env` > YAML config > defaults
