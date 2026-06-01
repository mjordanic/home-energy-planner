## Tooling

Use `uv` for all Python tooling — `uv run`, `uv add`, `uv sync`. Never `pip`, `python -m`, or `poetry`.

## Git commits

Do not sign commits with `Co-Authored-By: Claude …` (or any other co-author trailer attributing the commit to Claude/Anthropic or OpenAI or Cursor or any other coding agent). Leave the commit message unsigned. Applies to the main thread and any subagent or skill that creates commits.

## Agent skills

### Issue tracker

Issues and PRDs live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, default strings, recorded as `Status:` lines in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Implementation orchestrator (project-local, not vendored)

`/implement-issues` drives dependency-ordered, unattended implementation of
`ready-for-agent` issues in a `.scratch/<feature>/` folder. It builds a wave plan, then
dispatches subagents:

- `wave-runner` — runs one wave; owns git worktrees, cherry-pick integration, and report
  updates (a context firewall — only a small summary returns).
- `issue-implementer` — implements one issue via `/tdd`, one commit prefixed `<id>:`, then
  moves the issue file to `issues/done/`.

These are kept tool-agnostic the same way the skills are: the canonical files live under
`.agents/commands/implement-issues.md` and `.agents/agents/{wave-runner,issue-implementer}.md`,
and are symlinked into `.claude/commands/` and `.claude/agents/` so Claude Code discovers them
as a native `/`-command and subagents. Other coding assistants read the `.agents/` copies
directly. (Unlike the skills, these are project-local — not part of the `mattpocock/skills`
collection and not tracked in `skills-lock.json`.)

Questions are allowed only in Phases 0–1 (preflight + plan); from wave dispatch onward
the run is fully unattended and resumable. Relies on the issue-tracker layout, the five
triage labels, and `/tdd` above. First run audits `.claude/settings.local.json` against
its Required Bash patterns appendix and will prompt once to add any missing `permissions.allow`
entries (local git/filesystem only; the remote-touching deny list stays intact).

## Agent skills (vendored from mattpocock/skills)

This repo vendors the `mattpocock/skills` collection under `.agents/skills/`
(tracked in `skills-lock.json`) and symlinks each into `.claude/skills/` so
Claude Code discovers them as native `/`-invocable skills. 

To use one outside of Claude Code, read `.agents/skills/<name>/SKILL.md` and follow it.

Available: diagnose, git-guardrails-claude-code, grill-me, grill-with-docs,
handoff, improve-codebase-architecture, prototype, request-refactor-plan,
setup-matt-pocock-skills, tdd, teach, to-issues, to-prd, triage,
write-a-skill, writing-fragments, writing-shape, zoom-out.

Note: `teach`, `zoom-out`, and `setup-matt-pocock-skills` are manual-only
(`disable-model-invocation: true`) — invoke them explicitly.
