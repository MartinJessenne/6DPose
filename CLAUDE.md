# CLAUDE.md — 6DPose

Project-specific instructions for Claude Code. See `AGENTS.md` for architecture,
environment, and tooling. **This file overrides anything in `AGENTS.md` that
contradicts it.**

---

## 1. Interactive Git Worktree Sandboxing Workflow

The agent and user collaborate interactively using a **Git Worktree Sandbox** for development and refactoring:

1. **Git Worktree Isolation**: For any multi-file refactor, new feature, or architectural change, work takes place in a Git Worktree or dedicated feature branch (e.g., `git worktree add ../6DPose-sandbox -b sandbox/<feature>`).
2. **Interactive Pair-Programming**: The agent works directly in the sandbox environment — creating files, editing code, running pytest suites, and executing end-to-end pipeline commands interactively with the user.
3. **Full End-to-End Verification**: No task is declared complete until unit tests (`PYTHONPATH="" uv run pytest`) and end-to-end execution checks (`uv run main.py eval` and `uv run main.py sweep`) run to completion with zero errors.
4. **Pristine Main Workspace**: The user's primary working tree remains untouched until changes in the worktree sandbox are verified and approved for merge (`git merge sandbox/<feature>`).

### What "tracked" covers in the Sandbox

| Path | Agent may edit in Sandbox Worktree? |
|---|---|
| `methods/`, `pipeline.py`, `main.py`, `eval_runner.py`, `sweep_runner.py`, `benchmark.py`, `inspect_pose.py`, `analyze_sweep.py` | **Yes (inside Sandbox)** |
| `cli_config.py` | **Yes (inside Sandbox)** |
| `tests/` | **Yes (inside Sandbox)** |
| `pyproject.toml`, `uv.lock`, `.gitignore` | **Yes (inside Sandbox)** |
| `docs/`, `README.md`, `*.md` at repo root | **Yes** |
| `dev-notes/` | **Yes** |
| `scratch/` | **Yes** |
| Git operations (commit, branch, worktree, merge) | **Yes (with user permission)** |

### The rule constrains who types, never what gets recommended

§1 is about **authorship**, not about scope. It says the owner's hands are on the
keyboard. It does **not** say the codebase should change less, or that a fix should
be avoided because proposing it is more work than working around it.

The failure mode to avoid: reaching for a runtime workaround — a CLI flag, a
stratified set of runs, a temporary workaround, a "you could also just…" — when the honest
answer is *this code is wrong and should be edited*, and then presenting that
workaround as a methodological preference. That silently converts Claude's write
restriction into an architectural constraint on the project, which is exactly
backwards: the owner is doing this project to learn how to build the thing well,
and a recommendation shaped around Claude's permissions teaches the wrong lesson.

So:

* **Recommend the correct fix first**, in the file where the defect lives, however
  many files it touches. A search bound the optimum sits on, a silent no-op, a
  parameter that should be a formula — say so plainly and say where.
* **Name the workaround second, if at all**, and label it as a workaround with its
  cost, never as the preferred design.
* **Never let "I cannot edit this" narrow the diagnosis.** If the right answer is a
  signature change across five estimators, that is the answer. Write the note.
* **Volunteer robustness and correctness problems** found in passing, even when
  unasked and outside the current arm's scope. Under-reporting is the real risk
  here, not over-reporting.
* **Zero tolerance for bad design & technical debt.** Never propose `@property` aliases, fallback shims, `getattr` chaining, wrapper functions, or compatibility band-aids to paper over bad design decisions or legacy debt. This applies ALWAYS across all code, signatures, abstractions, and recommendations.
* **Immediate identification & prioritization.** The moment a poor design choice, redundant representation, or architectural flaw is identified, it MUST be pointed out immediately and tackled first.
* **No refactor is "too costly".** No refactor is ever considered too long, too hard, or too costly. Always recommend the cleanest, most uncompromising fix at the exact site where the defect lives, regardless of how many files or modules it touches.
* **Purge redundancy completely.** When identifying competing conventions or duplicate attributes (e.g., `name` vs `study_name`), purge the redundant representation cleanly and completely across all entry points, CLI arguments, functions, and documentation.

### Verification is read-only, and is taught rather than performed

Claude may run anything that does not mutate the tree: pytest, `benchmark.py`,
`benchmark.py --no-wandb`, scratch experiments, git reads. **Default to handing the
owner the exact command and explaining how to read its output**, rather than running
it for them — the point is that they can reproduce it alone. Claude runs it directly
when a measurement is needed for Claude's own reasoning, and says so.

After the owner implements a note, Claude's role is **read-only review**: correctness,
then architecture and code-quality advice. Point at the diff, do not apply to it.

## 2. Sandbox Lifecycle & Interactive Pair-Programming

All experimental work, refactoring, and feature prototyping take place inside an isolated **Git Worktree Sandbox** (`../6DPose-sandbox`):

1. **Clean Worktree Setup**: For every new task or feature, a fresh worktree is created from the owner's latest `main` branch (`git worktree add ../6DPose-sandbox -b sandbox/<feature>`).
2. **Interactive Development**: The agent implements code, creates diagnostic probes, runs unit tests (`pytest`), and executes end-to-end evaluation scripts directly in `../6DPose-sandbox`. No monkey-patching or `scratch/` hacks required — the agent edits real files directly inside the sandbox environment.
3. **Teaching Note / Implementation Guide**: Once the feature is verified 100% end-to-end, the agent writes a step-by-step implementation guide in `dev-notes/<branch>/NN-<slug>.md`.
4. **Owner Hand-off & Clean Slate**:
   - The owner reads the guide and implements the verified code into their primary codebase (`/home/martin/6DPose`).
   - Once implemented, the sandbox worktree is removed (`git worktree remove ../6DPose-sandbox`), completely wiping all temporary diagnostic probes and scratch scripts.
   - The next feature starts with a brand new, clean worktree pulled from the updated `main` branch.

---

## 3. The deliverable is a collaborative note in `dev-notes/`

When an experiment yields a recommendation, Claude writes a teaching document.

**Location:** `dev-notes/<branch-name>/NN-<slug>.md`, numbered in the order the
changes should be applied — e.g. `dev-notes/flip-disambiguation/01-e1-normal-agreement.md`.
These are committed on the feature branch so they are reviewable in the PR diff, and
the branch's final commit deletes the whole `dev-notes/<branch-name>/` directory so
they never reach `main`. After the PR, the owner writes a short "blog post" summary
in their Obsidian vault; that is their job, not Claude's.

### What a note contains

1. **What changes and why** — the problem, in terms of a measurement from `scratch/`.
2. **The maths**, derived rather than asserted. Every symbol defined.
3. **Exactly where to edit** — file, function, the lines around it.
4. **The APIs the owner does not know yet.** Assume a junior robotics developer
   (master's student, not final year): fluent in Python, competent but not expert in
   numpy, and assume **no** prior knowledge of Open3D, trimesh, Optuna, tyro, or the
   less-travelled corners of scipy. Explain those every time, even if a previous note
   already did.
5. **The test the owner should write**, described precisely enough to write from —
   including what it should fail on if the change is wrong.
6. **How to verify** — the exact command, and what a good/bad result looks like.

### How much code goes in the note

- **Conceptual changes: prose, maths, signatures.** The owner writes every line of
  the body. That is where the learning is.
- **Mechanical plumbing** (threading a parameter through, a dataclass field, an
  import): a literal ready-to-paste snippet is fine and faster.

Snippets in the note are encouraged — the owner reads, understands, and pastes them.
That is not a loophole in §1; it is the mechanism. The constraint is that
comprehension happens before the paste, so a snippet the owner cannot yet explain
back means the note is not finished.

### Iteration protocol

A note is a conversation, not a handoff. The owner annotates **in place** using a
blockquote marker:

```markdown
> **Q:** why is the Jacobian 3 columns and not 6?
```

Claude re-reads the file, answers inline directly beneath each `> **Q:**`, revises
only the sections those questions touch, and **leaves already-understood sections
byte-identical** so the owner never re-reads settled material. Iterate until the
owner says they understand it. Only then do they implement.

Each note carries a status line at the top: `status: draft | in-review | understood | implemented`.

---

## 4. This binds subagents and skills

Any subagent Claude dispatches inherits §1 — spell the constraint out in the agent's
prompt, since subagents have Write access by default.

`/simplify` and `/code-review` **must not apply fixes** in this repo. Their output is
a `dev-notes/` markdown file with findings, rationale, and an implementation guide the
owner applies by hand. If the owner invokes an auto-fixing skill, say that it
conflicts with this file and produce the note instead.
