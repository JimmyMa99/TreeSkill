# TODO

This file tracks the confirmed unfinished tasks in the current EvoSkill codebase.

## P0

- [ ] Unify the main optimization path onto the newer multi-step optimizer.
  The CLI still uses `APOEngine`, which performs a single optimization cycle.
  Relevant files:
  - `evoskill/cli.py`
  - `evoskill/optimizer.py`
  - `evoskill/core/optimizer.py`

- [ ] Fix duplicate trace writes when feedback is attached.
  `/bad` and `/rewrite` currently append the same trace again after adding feedback, which can pollute optimization data.
  Relevant files:
  - `evoskill/cli.py`
  - `evoskill/storage.py`
  - `ARCHITECTURE.md`

- [ ] Implement automatic few-shot example construction from high-quality traces.
  `few_shot_messages` exists in the schema, but there is no pipeline that promotes strong traces into reusable examples.
  Relevant files:
  - `evoskill/schema.py`
  - `evoskill/skill.py`
  - `ARCHITECTURE.md`

## P1

- [ ] Add automatic routing for skill trees.
  Users currently have to manually switch sub-skills with `/select`; there is no automatic leaf selection based on user input.
  Relevant files:
  - `evoskill/cli.py`
  - `evoskill/skill_tree.py`
  - `ARCHITECTURE.md`

- [ ] Add hard thresholds for auto split and auto prune decisions.
  The current structure evolution relies heavily on LLM judgment and needs statistical guardrails.
  Relevant files:
  - `evoskill/optimizer.py`
  - `evoskill/core/tree_optimizer.py`
  - `ARCHITECTURE.md`

- [ ] Complete the automatic merge workflow for skill trees.
  `SkillTree.merge()` exists, but there is no full merge analysis, trigger path, or CLI/product flow around it.
  Relevant files:
  - `evoskill/skill_tree.py`
  - `evoskill/core/tree_optimizer.py`
  - `evoskill/cli.py`

## P2

- [ ] Improve storage concurrency safety.
  The current JSONL append model may fail under multi-process usage and likely needs file locking or a different backend.
  Relevant files:
  - `evoskill/storage.py`
  - `ARCHITECTURE.md`

- [ ] Expand multimodal failure analysis beyond placeholder handling.
  Multimodal optimization support exists at a basic level, but detailed content-aware feedback analysis is still incomplete.
  Relevant files:
  - `evoskill/optimizer.py`
  - `ARCHITECTURE.md`

## Notes

- Some older summary documents in the repo mention unfinished items that are already implemented now, such as `AnthropicAdapter` and `AutoValidator`.
- This TODO list only includes items that still appear unfinished after checking both code and architecture notes.
