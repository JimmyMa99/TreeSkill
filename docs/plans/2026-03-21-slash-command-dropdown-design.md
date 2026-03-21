# Slash Command Dropdown Design

**Goal:** Add an in-input dropdown for slash commands so typing `/` in the chat CLI shows all available slash commands for selection.

## Context

The current chat loop in `evoskill/cli.py` reads input with `rich.prompt.Prompt.ask()`. That works for plain line input, but it only sees the full string after the user presses Enter. Because of that, slash commands are handled after input submission and there is no hook for interactive suggestions while the user is typing.

## Requirements

- Typing `/` should show all available slash commands in a dropdown-like completion menu.
- Typing a slash prefix such as `/he` should filter the command list.
- Selecting a completion should insert the command into the input line.
- Existing command behavior in `_handle_command()` and `_cmd_*()` should remain unchanged.
- This change should only target command-name suggestions, not argument completion.

## Approaches Considered

### 1. `readline` Tab completion

This is the smallest dependency footprint, but it mainly supports shell-style completion on demand. It does not reliably provide the always-visible dropdown experience requested here, and it is harder to make the interaction consistent across terminals.

### 2. Render a fake menu with Rich

We could intercept `/` after submit and redraw a panel with matching commands. This avoids a new dependency, but it is not true inline completion and would feel clunky compared with a real dropdown under the cursor.

### 3. Replace the chat input with `prompt_toolkit`

This adds one dependency, but it directly supports inline completions, dropdown menus, prefix filtering, and future extension to argument completion. The rest of the CLI can stay mostly the same because only the input gathering step changes.

**Recommendation:** Approach 3.

## Proposed Design

### Input layer

- Replace the main chat `Prompt.ask()` call in `ChatCLI.run()` with a `prompt_toolkit.PromptSession`.
- Keep `rich` for rendering panels, markdown, tool events, and status output.
- Use a dedicated prompt-toolkit completer for the main chat line only. Other prompts such as resume/restart and split confirmation can keep `Prompt.ask()` because they do not need dropdown completion.

### Completion source

- Reuse `_COMMAND_SPECS` as the single source of truth.
- Add a small helper that converts command specs into completion candidates with:
  - completion text: the slash command name like `/help`
  - display text: command plus usage like `/help`
  - metadata: the short description shown in the menu

### Matching behavior

- Show completions only when the current line starts with `/`.
- `/` with no suffix returns every command except the placeholder root alias duplication can be handled intentionally.
- A partial prefix like `/re` returns matching commands such as `/restore` and `/rewrite`.
- Completion inserts the command name only. Usage text remains visual help in the dropdown.

### Test strategy

- Add focused unit tests around the completer logic instead of terminal-level integration tests.
- Verify:
  - `/` returns the full slash command set
  - `/h` returns `/help`
  - non-slash input returns no completions
- Keep existing CLI tests passing to ensure command execution behavior is unchanged.

## Risks and Mitigations

- `prompt_toolkit` is a new runtime dependency.
  - Mitigation: keep the integration localized to the chat prompt and avoid rewiring the rest of the CLI.
- Rich and prompt-toolkit both manage terminal output.
  - Mitigation: use prompt-toolkit only for line input and continue using Rich for output rendering between prompts.
- Duplicate command entries for `/` and `/help` could clutter the menu.
  - Mitigation: keep both if desired for discoverability, or tune display ordering during implementation if it feels noisy.

## Out of Scope

- File path completion for `/image` or `/audio`
- Skill path completion for `/select`
- Fuzzy matching beyond slash-prefix filtering
- Reworking non-chat prompts into prompt-toolkit widgets
