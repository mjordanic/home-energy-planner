# Implement Issues

Orchestrate dependency-ordered implementation of `ready-for-agent` issues in a `.scratch/<feature>/` folder. Builds a wave plan, dispatches one fresh `wave-runner` subagent per wave (each runs the wave end-to-end and returns a small summary), and maintains a resumable `implementation_report.md` so a credit-out / killed session can pick up cleanly on the next invocation.

**Question window**: the user is available only during Phases 0–1 (preflight + planning). The moment Phase 2 starts (resume reconcile + wave dispatch), the run is fully unattended — anomalies are recorded in the report and the run continues; nothing blocks for human input. Use the planning phase to surface anything ambiguous up front so dispatch can run cleanly.

**Permissions contract**: this skill and its `wave-runner` / `issue-implementer` subagents issue a fixed vocabulary of *local* git and filesystem commands (full list in the **Required Bash patterns** appendix at the bottom of this file). The deny list (anything that touches a remote — `git push/pull/fetch/remote/clone/ls-remote/submodule`, `gh`, `hub`) is enforced separately and is not affected by this skill. Phase 0's permission audit surfaces any missing allows once, before dispatch — the orchestration phase itself never prompts for tool permissions. If the audit can't bring the allow list into alignment (e.g., the harness hard-blocks edits to `.claude/settings.local.json`), it prints a copy-pasteable JSON block and asks the user to apply it manually, then aborts. Re-run the skill once settings are in place.

The wave-runner indirection is a context firewall: per-wave git activity, worktree management, and per-issue subagent transcripts stay inside the wave-runner's context. The orchestrator only sees a small return summary per wave, so its context stays roughly constant regardless of how many issues or waves are involved.

## Inputs

- **Argument (optional)**: a feature folder path (e.g., `.scratch/<feature>/`). If omitted, infer the most recently modified `.scratch/<feature>/` directory. If two or more candidates were modified within 60s of each other, record an "ambiguous feature folder" preflight failure and exit.
- **Parallelism cap (optional)**: default `3`. `cap == 1` ⇒ in-place sequential; `cap > 1` ⇒ worktree-per-issue parallel within each wave. Wave size is computed from the dependency graph, not assumed.
- **Model selection (optional)**: choose the model backing each subagent role. Two independent knobs plus a shorthand:
  - `--runner-model <model>` — model for the `wave-runner` subagent (git plumbing: worktrees, cherry-picks, report writes).
  - `--implementer-model <model>` — model for the `issue-implementer` subagents (the TDD coding loop).
  - `--model <model>` — shorthand that sets BOTH. An explicit `--runner-model` / `--implementer-model` wins over `--model` when both are supplied.
  - Valid values: `opus`, `sonnet`, `haiku`. When a flag is given it overrides the subagent's `model:` frontmatter via the dispatching tool's per-call model parameter. **When a flag is omitted, pass no model parameter at all** — the subagent's own frontmatter default applies. Do not hardcode a default model here; the frontmatter is the single source of truth for it.
  - **Resume**: on a resume run, an omitted flag inherits the value recorded in the existing report header (Phase 4 §1); a supplied flag overrides it; with neither, the role stays unpinned and its frontmatter default applies. The resolved selection (explicit model, or "default") is always (re)written to the header so a later resume stays consistent.
  - **Per-issue complexity (automatic, implementer only).** When `--implementer-model` is *not* set, the implementer model is chosen **per issue** from a `Complexity:` line in the issue file (read in Phase 1). The runner model is never complexity-driven — coordinating git plumbing is mechanical regardless of issue size — so this layer applies only to `issue-implementer`. Per-issue resolution, first match wins:
    1. a `Model:` line in the issue file (hard per-issue override),
    2. the global `--implementer-model` / `--model` flag,
    3. the issue's `Complexity:` mapping (Phase 1 policy: `high → opus`; `standard` / `low` / absent → unpinned),
    4. the issue-implementer frontmatter default (no model parameter passed).

## Phase 0 — Preflight

Run FIRST, every invocation. **This is one of the two phases where you may ask the user.** Surface anything ambiguous; do not paper over it just to keep moving.

1. **Integration branch** — `BASE_BRANCH = $(git symbolic-ref --short HEAD)` from the repo root. Refuse if HEAD is detached or `BASE_BRANCH` ∈ {`main`, `master`} (ask the user to switch to a feature branch).
2. **Working tree clean** — `git status --porcelain` empty *except* for `<feature>/implementation_report.md` (skill-owned). If dirty, ask the user to stash/commit; do not run a partial dispatch over uncommitted work.
3. **`uv` available** — `uv --version` succeeds. If not, ask.
4. **Feature folder** — if no argument was given and inference returns multiple candidates modified within 60s of each other, ask the user which one. If the user-supplied argument doesn't exist, ask.
5. **PRD exists** — `<feature>/PRD.md` is present. If not, ask.
6. **Permission audit** — read `.claude/settings.local.json` and compute which entries from the **Required Bash patterns** appendix are missing from the `permissions.allow` list. If any are missing, bundle the diff into Phase 1's wave-plan question as a single "apply these permission additions to `.claude/settings.local.json`?" prompt. If the user approves and the harness allows the edit, write the additions back atomically. If the harness hard-blocks the edit (settings files are commonly self-modification-protected), print the JSON patch as a copy-paste block, ask the user to apply it manually, and abort preflight; re-running the skill after the user updates settings will pass the audit. The orchestration phase never re-checks — it assumes the contract is in place once Phase 1 ends.

7. **Model resolution (global tier).** Resolve the two *global* model knobs from the flags (see Inputs), falling back to the existing report header on a resume run: `runner-model` is final here (the runner is never complexity-driven), while `implementer-model` resolves only the global override tier — the per-issue refinement from each issue's `Complexity:`/`Model:` line happens in Phase 1 once the issue files are read. If neither flag nor header pins a role, leave it unpinned — the subagent's frontmatter default applies and you pass no model parameter for it. Validate any explicit value ∈ {`opus`, `sonnet`, `haiku`}. An unknown value is a preflight failure: ask the user to correct it (Phase 2+ is unattended, so bad model input must fail loud and early, never silently downgrade).

Record `BASE_BRANCH`, started-at, parallelism cap, and the resolved `runner-model` / `implementer-model` selections (an explicit model, or "default") in the report header so resume runs can verify the same branch and reuse the same models.

## Phase 1 — Plan

1. List every `*.md` under `<feature>/issues/` (skip `done/`).
2. For each file, extract: ID (filename minus `.md`), Title (first H1), Status (`Status:` line), Blocked-by (`## Blocked by` section), and — for model selection — an optional `Complexity:` line (`high` / `standard` / `low`) and an optional `Model:` line (`opus` / `sonnet` / `haiku`, a hard per-issue override). Both are optional; absent ⇒ that issue falls through to the next precedence tier. Treat IDs already in `done/` as satisfied.
3. Keep only issues with `Status: ready-for-agent`. Skip the rest. **Exception**: if every file in the feature lacks a `Status:` line, surface this to the user and ask whether to treat them all as `ready-for-agent`. Record the answer in the report header.
4. Build the dependency graph. **Cycles → ask the user.** Cycles indicate `/to-issues` produced non-vertical slices and need human review; do not invent a tie-breaker.
5. Compute waves: wave N is every issue whose blockers are all done or in waves `< N`. If a wave is larger than the parallelism cap, split it into consecutive sub-waves of cap-sized chunks.
6. **Resolve each issue's implementer model** using the precedence in Inputs (`Model:` line → global `--implementer-model`/`--model` → `Complexity:` mapping → frontmatter default). Validate any `Model:` line and any mapped value ∈ {`opus`, `sonnet`, `haiku`}; an unknown value is a planning failure — ask the user. The mapping (`high → opus`; everything else → unpinned) is a small fixed policy block here, not a runtime flag; edit this file to retune it. Record the resolved model (explicit model, or "default") against each issue for the wave plan.
7. **Present the wave plan to the user** — annotate each issue with its resolved implementer model so a complexity-driven `opus` pick is visible — and let them abort, reorder waves, drop issues, adjust the parallelism cap, or override any per-issue model. This is the **last point at which questions are allowed** — once you start Phase 2, the run is unattended.
8. Write the initial report (schema in Phase 4) with every issue as `pending`.

## Phase 2 — Resume reconcile

Run EVERY invocation, even fresh. Cheap, idempotent, no external state.

1. If a prior `implementation_report.md` exists, read its status table.
2. Run `git log --oneline -n 100` on `BASE_BRANCH` and `git status --porcelain`.
3. Cross-reference, in priority order:
   - **Issue file in `done/`** → `committed` (strongest signal — implementer moves the file as its last step).
   - **Commit on `BASE_BRANCH` with subject prefix `<id>:`** → `committed`. Capture the SHA.
   - **Report says `in-progress` but neither of the above** → prior subagent died. Mark `pending` again.
4. **Stale worktree sweep**: `git worktree list`; prune any `.claude/worktrees/issue-*` whose branch has no unique commits ahead of `BASE_BRANCH`. `git worktree remove --force` and delete the orphan branch.
5. **Salvage sweep**: for any `.claude/worktrees/issue-<id>/` whose branch HAS a `<id>:`-prefixed commit not yet on `BASE_BRANCH`, cherry-pick it now (apply Phase 3's cherry-pick rules: clean → `committed`; conflict → abort + mark `failed` + log files). Clean up the worktree afterward.
6. Rewrite the report's status table with this reconciled view **before** spawning anything.
7. Drop any `committed` issue from the dispatch queue.

## Phase 3 — Dispatch waves

For each wave with at least one pending issue, sequentially:

1. Dispatch ONE fresh `wave-runner` subagent. If `runner-model` is pinned, **set the Agent tool's `model` parameter to it** (overriding the wave-runner's frontmatter); if it is unpinned ("default"), omit the `model` parameter so the frontmatter applies. Pass in the prompt:
   - Repo root (absolute).
   - Feature path (absolute).
   - Report path (absolute).
   - `BASE_BRANCH`.
   - Wave number.
   - Parallelism cap.
   - List of pending issue IDs in this wave.
   - Feature slug (for branch naming).
   - **Per-issue implementer models** — a `{issue-id → model}` map covering this wave's issues, each value an explicit model (`opus`/`sonnet`/`haiku`) or "default". The wave-runner can see neither your flags nor the issue files' `Complexity:`/`Model:` lines, so hand it the **already-resolved** model per issue (you computed these in Phase 1 step 6). "default" tells the wave-runner to omit the model parameter for that issue and let the issue-implementer frontmatter apply.
2. Wait for the subagent's return summary (small `summary` block listing committed / failed / blocked / conflicts).
3. Merge its counts into your running totals.
4. **Continue to the next wave regardless of failures inside this one.** A failed wave does not cascade.

The wave-runner owns: worktree creation, dispatching `issue-implementer` subagents, cherry-pick integration, per-wave report updates, worktree cleanup. You only do dispatch + summary aggregation.

## Phase 4 — Report schema

The report file is the contract between this skill, its wave-runners, and the human reviewer. It must contain, in this order:

1. **Header** — feature name, link to `PRD.md`, started-at, last-updated, parallelism cap, integration branch (`BASE_BRANCH`), resolved `runner-model` / `implementer-model`, and any preflight assumptions (e.g., `Status:` lines missing across the board).
2. **Status table** — one row per scheduled issue: `ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes`. Statuses: `pending`, `in-progress`, `committed`, `failed`, `blocked`. Issue file at `done/` ⇔ report status `committed`. For in-place dispatches, Worktree SHA == Integrated SHA.
3. **Dependency graph** — fenced ASCII or mermaid block.
4. **Wave plan** — ordered list, member issues per wave, each issue annotated with its resolved implementer model (explicit, complexity-derived, or "default"). On resume this is re-derived deterministically from the issue files + Phase 1 policy, so an unedited issue keeps its model; the annotation is the human-visible record of what each wave will dispatch with.
5. **Activity log** — append-only, timestamped one-liners for every state transition (commit SHAs, worktree paths, cherry-pick outcomes).
6. **Outstanding follow-ups** — aggregated from subagent reports plus any cherry-pick conflicts.
7. **Resume instructions** — re-run the skill with the same feature path; reconcile catches up.

The orchestrator owns sections 1, 3, 4, 7 plus the *initial pending* rows of section 2. Each wave-runner owns the rows in section 2 for its wave and appends to sections 5 and 6. Atomic writes only (build full content in memory, single `Write` call).

## Phase 5 — Final summary

When all waves are dispatched (or an early-exit preflight happened):

1. Read the final report; recompute counts from the status table.
2. **Final cleanup**: confirm no `.claude/worktrees/issue-*` directories or `<feature-slug>/issue-*` branches remain. Force-remove stragglers; append the cleanup to the activity log.
3. Print to chat: shipped, failed, blocked, total commits, top follow-ups, path to the report file.

## Hard rules

- **No user prompts after Phase 1 ends.** Phases 0 and 1 are the only window for questions; everything from Phase 2 onward must be unattended. Mid-run anomalies are recorded in the report and the run continues with whatever is still actionable.
- Do **NOT** trust the `Agent` tool's `isolation: "worktree"` flag. It branches from `origin/main`, not the integration branch. Always use `git worktree add` explicitly from `BASE_BRANCH`. (Wave-runner enforces the same rule.)
- Do **NOT** dispatch parallel `issue-implementer` subagents into the repo root. Concurrent edits to the same working tree race on the index and corrupt commits. (Wave-runner enforces this — the orchestrator never dispatches `issue-implementer` directly.)
- Do **NOT** write feature code yourself. You orchestrate; subagents implement.
- Do **NOT** auto-resolve cherry-pick conflicts (in the salvage sweep). Abort, mark `failed`, log files.
- Do **NOT** cascade-fail. A failed issue does not stop the wave; a failed wave does not stop the run.
- A killed session is recoverable: re-run the skill with the same feature path. Phase 2 reconciles from disk + git.

## Required Bash patterns (appendix)

The Phase 0 permission audit checks `.claude/settings.local.json`'s `permissions.allow` list against this set. Add any missing entries to silence runtime prompts during waves. The deny list at the bottom of this appendix is enforced independently — keep it intact so that no remote-touching command can slip through even if the allow list grows.

```jsonc
// permissions.allow — minimum set for unattended dispatch
"Bash(uv run *)",
"Bash(MPLBACKEND=* uv run *)",

// Local git verbs (no remote interaction)
"Bash(git status*)",
"Bash(git log*)",
"Bash(git diff*)",
"Bash(git show*)",
"Bash(git symbolic-ref*)",
"Bash(git rev-parse*)",
"Bash(git branch*)",
"Bash(git checkout*)",
"Bash(git switch*)",
"Bash(git restore*)",
"Bash(git add*)",
"Bash(git rm*)",
"Bash(git mv*)",
"Bash(git commit*)",
"Bash(git cherry-pick*)",
"Bash(git apply*)",
"Bash(git worktree*)",
"Bash(git reflog*)",
"Bash(git cat-file*)",
"Bash(git ls-tree*)",
"Bash(git ls-files*)",
"Bash(git check-ignore*)",
"Bash(git -C *)",

// Worktree filesystem (both standard locations the wave-runner may pick)
"Bash(mkdir -p /tmp/*)",
"Bash(mkdir -p .claude/worktrees/*)",
"Bash(rm -rf /tmp/worktrees/*)",
"Bash(rm -rf .claude/worktrees/*)",
"Bash(ls /tmp/worktrees*)",
"Bash(ls .claude/worktrees*)",
"Bash(find /tmp/worktrees *)",
"Bash(find .claude/worktrees *)",
"Bash(cd /tmp/worktrees/*)",
"Bash(cd .claude/worktrees/*)",

// Small utilities used by report writes and reconcile
"Bash(date*)",
"Bash(pwd)"
```

```jsonc
// permissions.deny — keep these as the safety floor regardless of allow-list growth
"Bash(git push*)",
"Bash(git pull*)",
"Bash(git fetch*)",
"Bash(git remote*)",
"Bash(git clone*)",
"Bash(git ls-remote*)",
"Bash(git submodule*)",
"Bash(gh *)", "Bash(gh)",
"Bash(hub *)", "Bash(hub)"
```

Notes on scope:
- `git worktree*` covers `add`, `remove --force`, `list`, `prune` — used by the wave-runner.
- `rm -rf /tmp/worktrees/*` and `rm -rf .claude/worktrees/*` are scoped to worktree paths only, not a general `rm -rf *` grant.
- `MPLBACKEND=* uv run *` is needed for headless notebook re-execution (`jupyter nbconvert`); without it, env-var-prefixed `uv run` calls trip a separate permission check than plain `uv run *`.
- The wave-runner may stage worktrees under either `.claude/worktrees/<feature-slug>-issue-*` or `/tmp/worktrees/issue-*` depending on its agent definition. Both are covered above; the audit treats them as one set.
