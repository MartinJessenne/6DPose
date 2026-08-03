# 01 — Harness fixes before the confirmation batch

`status: draft`

**Scope.** Three edits to `benchmark.py` and `cli_config.py` that protect the integrity
of any run you launch, plus the W&B reading of the arms that ran on `f4f83c9`. None of
them changes estimator behaviour.

Apply this note before [02](02-deriving-the-fixed-parameters.md), which changes what
the parameters *mean*. This one only makes sure the harness reports honestly what it ran.

---

## 0. Where you are — checklist as of 2026-08-03

Edits 1 and 2 are landed. Edit 3 is half-landed and **currently breaks every sweep**.
Full diagnosis in §9; this is the ordered to-do list.

**Step 1 replaces six whole functions.** That is deliberate: the three blocking bugs are
entangled (a deletion in one file, a signature in another, a call site in a third), and
surgical line edits have an ordering hazard where the tree is broken in between. Replacing
each `suggest_params` wholesale means every file is consistent the moment you finish it.

### How to drive a refactor like this without missing a site

The failure mode is not difficulty, it is **bookkeeping** — six near-identical edits, and
the one you forget fails at runtime, three hours into a sweep. The senior habit is to stop
relying on memory and make the *tooling* hold the list.

**1. Commit first.** A clean tree turns every experiment into free undo: `git diff` shows
exactly your blast radius, and `git checkout -- methods/ppf.py` reverts one file without
touching the others. Do not start a multi-file refactor on a dirty tree.

**2. Build the worklist mechanically, not from the note.** `Ctrl+Shift+F`, search
`def suggest_params`. Then click **"Open in editor"** in the results header — that turns the
search panel into a real, editable buffer you can keep in a split and annotate as you go.
Six hits, six edits; if the note and the search disagree, the search is right.

**3. Better still, make the worklist a failing test.** Write test 9
(`test_all_estimators_accept_fixed`, §6.3) **before** touching any estimator. It enumerates
the classes and fails on every unmigrated one, so `uv run pytest -q` becomes a live progress
bar: six failures, then five, then zero. This is the difference between "I think I got them
all" and knowing. A checklist that runs itself never goes stale.

**4. Work one file at a time, `git add` after each.** Staging per file means `git diff`
always shows only what you have not yet reviewed. Some people commit WIP per file and squash
at the end; either works, the point is that finished work stops competing for attention.

**5. Define a completion check that must return *nothing*.** This is the part most people
skip, and it is the whole trick. Do not ask "did I get them all?" — ask a question whose
correct answer is empty output:

A search that returns nothing is a proof; a search that returns something is your next edit.
Run these before you run anything expensive. There are four:

| # | look for | expect |
| :-: | :--- | :--- |
| 1 | `def suggest_params` | 6 hits, every one carrying `fixed` |
| 2 | `suggest_params(trial)` | **0** — an unforwarded call site |
| 3 | `"ransac_max_iterations"` | **0** |
| 4 | `if TYPE_CHECKING:` followed by `pass` | **0** |

**Scope every one of them to `methods/` and `benchmark.py`.** An unscoped search matches this
note, `scratch/`, and `tests/` — about fifteen lines of noise that drown the two real hits,
and a check you have to squint at is a check you will stop running. (`tests/` is excluded for
a second reason: its calls pass one argument and stay valid via the default, so they are not
defects — but they *do* need attention, see the warning below.)

#### Doing it in the GUI

`Ctrl+Shift+F`, then use the two boxes under the search field (click the **`…`** to reveal
them if they are hidden):

```
Search:            suggest_params(trial)
files to include:  methods/,benchmark.py
files to exclude:  (leave empty — the include box already scopes it)
```

Leave the **`.*` regex toggle OFF** (`Alt+R`). With it off, VS Code searches for your text
*literally*, so `suggest_params(trial)` needs no escaping — the parentheses are just
parentheses. Every check above works as plain text; none of them needs a regex.

**The result header is the assertion.** It reads `12 results in 3 files`, or `No results` —
and `No results` is exactly the green light checks 2, 3 and 4 are looking for.

**The GUI has one advantage the CLI cannot match: the list updates live as you edit.** Leave
the search panel open on `"ransac_max_iterations"` while you work through step 1, and watch
it count 7 → 4 → 0 as you save each file. That is a progress bar for the refactor, free, and
it is the single best reason to prefer the GUI while you are *doing* the work.

Two more GUI details worth knowing:

* **"Open in editor"** in the results header turns the search into a **Search Editor** — a
  real, saveable buffer with context lines that does *not* refresh under you. Use it when you
  want a stable worklist to tick through, as opposed to the live panel.
* Check 4 is two-stage ("`if TYPE_CHECKING:` where the *next* line is `pass`"), which the
  panel cannot express. Just search `if TYPE_CHECKING:` and eyeball two hits — or use the CLI
  form below.

#### Doing it on the command line

Worth learning **only for the second job**: the GUI is for finding your way around while you
work, the CLI is for the repeatable pass/fail check you run at the end, paste into a commit
message, or drop into a script. It answers a different question, and it answers it in one
line instead of four sets of box-filling.

```bash
rg -F 'def suggest_params' methods/                 # expect 6
rg -F 'suggest_params(trial)' methods/ benchmark.py # expect ZERO
rg -F '"ransac_max_iterations"' methods/            # expect ZERO
rg -A1 'if TYPE_CHECKING:' methods/ | rg -F pass    # expect ZERO
```

**`-F` means "fixed string" — regex off.** That is the flag that makes ripgrep pleasant if
you do not want to think about escaping: `(`, `)`, `.` and `"` are all just characters. It is
the exact equivalent of leaving the `.*` toggle off in the GUI. You genuinely do not need
regex for any check in this note.

The whole ripgrep vocabulary worth memorising is five flags:

| flag | does |
| :--- | :--- |
| `-F` | literal string, no regex |
| `-n` | show line numbers (on by default when printing to a terminal) |
| `-t py` | only Python files, without writing a glob |
| `-g '!tests'` | exclude a path (note the `!`) |
| `-c` | print only the count per file |

Everything else you can look up the day you need it. Ripgrep also skips `.gitignore`d paths
automatically, which is why `.venv/` never appears in your results without you asking.

One reason the CLI wins for the *final* check: **exit status**. `rg` exits non-zero when it
finds nothing, so the four checks collapse into one command that either prints `ALL CLEAR` or
does not:

```bash
! rg -qF 'suggest_params(trial)' methods/ benchmark.py \
  && ! rg -qF '"ransac_max_iterations"' methods/ \
  && echo "ALL CLEAR"
```

You do not need to write that today. But it is the shape of thing that later becomes a
pre-commit hook, and it is why "search from the terminal" is a habit worth having alongside
the GUI rather than instead of it.

### ⚠️ Step 1 breaks an existing test — budget for it

`tests/test_estimators.py:46-64`, `test_ransac3dof_suggest_params_construct_params`, pins
`"ransac_max_iterations": 20000` and asserts it survives into the params object. Once step 1b
removes that suggestion, `suggest_params` no longer returns the key,
`Ransac3DoFParams(**suggested)` falls back to the class default, and the assertion loop fails.

This is the test doing its job — it is telling you a swept parameter stopped being swept. Fix
it by deleting `ransac_max_iterations` from that test's `fixed_values` dict, and leave a
comment saying it is now a per-profile budget. **Do not** "fix" it by adding the suggestion
back.

Note 02 will do the same to `"front_crop_depth": 0.35` in the same dict when the field is
renamed. Expect it, and treat it as confirmation rather than breakage.

**6. Then let the machine check style and behaviour**, in this order — cheapest first:

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

### The shortcuts worth having in your fingers

| what | VS Code (Linux) | in VSCodeVim |
| :--- | :--- | :--- |
| Search across files | `Ctrl+Shift+F` | — |
| Search → editable buffer | "Open in editor" in results header | — |
| **Rename a symbol everywhere** | `F2` | — |
| Find all references | `Shift+Alt+F12` | — |
| Go to definition / back | `F12` / `Ctrl+Alt+-` | `gd` / `Ctrl+o`, forward `Ctrl+i` |
| Peek definition (no jump) | `Alt+F12` | — |
| Symbol in file / in project | `Ctrl+Shift+O` / `Ctrl+T` | — |
| Fold / unfold this block | `Ctrl+Shift+[` / `Ctrl+Shift+]` | `zc` / `zo`, toggle `za` |
| Problems panel | `Ctrl+Shift+M` | — |
| Repeat last edit | — | `.` |
| Record / replay macro | — | `qa` … `q`, then `@a`, then `@@` |

Three of these actually matter here:

**Folding is how you replace a whole function safely.** Put the cursor in
`suggest_params`, `zc` to collapse it to a single line, `V` to select that line (a folded
region selects as one unit), then paste. No counting braces, no accidentally eating the next
method's decorator. `zo` to reopen. This is the cleanest way to execute "replace the complete
final body" six times.

**`.` and macros are for step 4 of §0.1c**, where you wrap three suggestions in identical
`if "..." not in fixed:` guards. Do the first by hand, then `q a` to start recording, do the
second, `q` to stop, and `@a` for the third. If you find yourself typing the same five
keystrokes a third time, you have already paid for the macro.

**`F2` (Rename Symbol) is the right tool for note 02, not for this note.** Here you are
changing *signatures*, which rename cannot do. But note 02 renames `front_crop_depth` →
`front_crop_fraction`, and `F2` on the dataclass field will update every reference Pylance
can see — including `_get_prep_params_key`, which is exactly the site §4 of that note warns
you about. Use `F2` there and the trap disappears. Do check the preview (`Ctrl+Shift+Enter`
opens the rename preview pane) — Pylance cannot see names built as strings, and
`--param-overrides front_crop_depth …` in your shell history is one of those.

> **VSCodeVim gotcha.** VSCodeVim claims `Ctrl+D` for half-page-scroll, so VS Code's
> add-selection-to-next-match does not reach it. If you want the multi-cursor version,
> release the key in `settings.json`:
> ```json
> "vim.handleKeys": { "<C-d>": false }
> ```
> The same applies to `<C-f>`. Decide once; a half-working keybinding costs more attention
> than either choice.

---

Each block below is the **complete final body** — read it, then paste it over the existing
function. The canonical signature is the same everywhere:

```python
    @classmethod
    def suggest_params(
        cls, trial: "optuna.Trial", fixed: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
```

Three things are happening in it at once: `fixed` is added, the `"optuna.Trial"` annotation
is restored (§9.5 — it was dropped, and it is what makes the `TYPE_CHECKING` import
meaningful), and the whole thing is wrapped because the one-line form is 105 columns against
your 100-column ruff config.

### ☐ 1a. `methods/base.py`

Restore the import at the top — currently `if TYPE_CHECKING: pass`:

```python
if TYPE_CHECKING:
    import optuna
```

Then the abstract declaration:

```python
    @classmethod
    def suggest_params(
        cls, trial: "optuna.Trial", fixed: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """
        Suggests hyperparameters for this matching method using an Optuna trial.

        Args:
            trial: The active Optuna trial.
            fixed: Parameter names already pinned for this arm (via --param-overrides)
                and which must therefore NOT be suggested. Suggesting one anyway costs
                a TPE dimension that cannot affect the objective, and makes Optuna's
                parameter-importance output meaningless for that field.

        Returns:
            dict[str, Any]: Suggested parameter dictionary.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError(
            f"Estimator class '{cls.__name__}' does not implement 'suggest_params' for parameter sweeps."
        )
```

Note the blank line before `Returns:` — without it the Google-style `Args:` block swallows
the rest of the docstring.

### ☐ 1b. `methods/ransac.py`

Same `if TYPE_CHECKING: import optuna` restore, then:

```python
    @classmethod
    def suggest_params(
        cls, trial: "optuna.Trial", fixed: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Suggests parameters for RANSAC + ICP registration."""
        params = {}

        if "voxel_size" not in fixed:
            params["voxel_size"] = trial.suggest_float("voxel_size", 0.02, 0.10, step=0.01)

        if "icp_max_correspondence_distance" not in fixed:
            params["icp_max_correspondence_distance"] = trial.suggest_float(
                "icp_max_correspondence_distance", 0.05, 0.25
            )

        if "icp_max_iterations" not in fixed:
            params["icp_max_iterations"] = trial.suggest_int("icp_max_iterations", 10, 100, step=10)

        # ransac_max_iterations is deliberately NOT suggested here. A controlled ladder
        # (W&B 74n9avkc / xviophc7 / ifbe9gvt / rouf33ov, single variable) is monotone in
        # accuracy from 1k to 47k, so it is a latency budget to be set per profile, not a
        # dimension to search. It was also suggested by Ransac3DoFEstimator, and two
        # suggestions of one name with different distributions abort every trial.
        return params
```

**This is the deletion that unblocks `--sweep`.** From here on a sweep that cares about the
budget sets it explicitly: `--param-overrides ransac_max_iterations 46940`.

### ☐ 1c. `methods/ransac3dof.py` — `Ransac3DoFEstimator`

```python
    @classmethod
    def suggest_params(
        cls, trial: "optuna.Trial", fixed: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Suggests parameters for the SE(2)-constrained RANSAC + ICP registration."""
        params = super().suggest_params(trial, fixed=fixed)

        if "edge_length_threshold" not in fixed:
            params["edge_length_threshold"] = trial.suggest_float(
                "edge_length_threshold", 0.8, 0.95
            )

        # First z-gate sweep pressed against the old 0.20 ceiling (19/20 top
        # trials above 0.15): the optimum lies higher, so give it headroom.
        if "z_gate_threshold" not in fixed:
            params["z_gate_threshold"] = trial.suggest_float("z_gate_threshold", 0.05, 0.35)

        # Registering the asymmetric front slab instead of the full cart
        # halved the flip rate and doubled AR in A/B benchmarks; the slab
        # depth trades feature support against re-imported symmetry. Upper
        # bound is the longest cart in the fleet (colruyt, ~2.57 m x-extent):
        # beyond that the crop no longer removes any mesh, silently
        # re-introducing the front/back symmetry this parameter exists to break.
        if "front_crop_depth" not in fixed:
            params["front_crop_depth"] = trial.suggest_float("front_crop_depth", 0.1, 2.5)

        # front_face_max_angle_deg is NOT suggested: it is the arm's independent
        # variable, set per-arm via --param-overrides. A parameter Optuna sweeps
        # but nothing contrasts is a tuned nuisance, not a controlled comparison.
        #
        # icp_visibility_cull / icp_refine_ladder / icp_yaw_guard_deg are left
        # out for the same reason -- they are the T0 arm's independent variables
        # and are selected by profile (tuned-vis) so control and treatment differ
        # by the profile token alone. icp_max_correspondence_distance IS swept
        # (inherited from RansacEstimator) and should be re-swept once the cull
        # lands: its current optimum was found against a biased objective.
        return params
```

`ransac_max_iterations` is gone; the other three gain guards. The trailing comment block is
unchanged and still worth keeping — it is the record of *why* four parameters are absent.

### ☐ 1d. `methods/ransac3dof.py` — `Ransac3DoFFullMeshEstimator`

```python
    @classmethod
    def suggest_params(
        cls, trial: "optuna.Trial", fixed: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        params = RansacEstimator.suggest_params(trial, fixed=fixed)

        if "edge_length_threshold" not in fixed:
            params["edge_length_threshold"] = trial.suggest_float(
                "edge_length_threshold", 0.8, 0.95
            )

        if "z_gate_threshold" not in fixed:
            params["z_gate_threshold"] = trial.suggest_float("z_gate_threshold", 0.05, 0.35)

        return params
```

Watch the `RansacEstimator.suggest_params(trial, fixed=fixed)` line: this class calls the
grandparent **explicitly by name** rather than via `super()`, so `fixed` has to be forwarded
by hand. Miss it and the base guard is silently bypassed for this estimator only.

### ☐ 1e. `methods/vsac_se2.py` — the estimator you actually run

```python
    @classmethod
    def suggest_params(
        cls, trial: "optuna.Trial", fixed: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Suggests parameters for the VSAC-based SE(2) RANSAC + ICP registration."""
        params = super().suggest_params(trial, fixed=fixed)
        # rho ~ a few voxels to a few tens of cm: too small and every inlier
        # looks independent (the tiebreak never fires); too large and the
        # whole cart collapses to ~1 independent cluster (same failure mode).
        if "rho" not in fixed:
            params["rho"] = trial.suggest_float("rho", 0.05, 0.6)
        return params
```

### ☐ 1f. `methods/ppf.py`

Not on the current arm's path, but it must accept the new signature or step 2 crashes it:

```python
    @classmethod
    def suggest_params(
        cls, trial: "optuna.Trial", fixed: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Suggests parameters for PPF + ICP registration."""
        params = {}

        if "ppf_sampling_step" not in fixed:
            params["ppf_sampling_step"] = trial.suggest_float(
                "ppf_sampling_step", 0.02, 0.10, step=0.01
            )

        if "ppf_distance_step" not in fixed:
            params["ppf_distance_step"] = trial.suggest_float(
                "ppf_distance_step", 0.02, 0.10, step=0.01
            )

        if "ppf_match_threshold" not in fixed:
            params["ppf_match_threshold"] = trial.suggest_float(
                "ppf_match_threshold", 0.02, 0.10, step=0.01
            )

        if "ppf_match_tolerance" not in fixed:
            params["ppf_match_tolerance"] = trial.suggest_float(
                "ppf_match_tolerance", 0.01, 0.08, step=0.01
            )

        if "icp_max_correspondence_distance" not in fixed:
            params["icp_max_correspondence_distance"] = trial.suggest_float(
                "icp_max_correspondence_distance", 0.02, 0.20
            )

        if "icp_max_iterations" not in fixed:
            params["icp_max_iterations"] = trial.suggest_int("icp_max_iterations", 10, 100, step=10)

        return params
```

### ☐ 2. Connect the wire — `benchmark.py:874`

Currently `fixed` defaults to empty on every call, so everything above does nothing until
this lands:

```python
            suggested_params = {
                **estimator_cls.suggest_params(trial, fixed=frozenset(resolved_overrides)),
                **resolved_overrides,
            }
```

Keep the merge **as well as** the argument. They do different jobs: `fixed` suppresses the
*suggestion*, the merge supplies the *value* — including for the parameters
(`icp_visibility_cull`, `front_face_max_angle_deg`, …) that are never suggested at all.

`frozenset(resolved_overrides)` iterates a dict, which yields its keys — the parameter
names, which is exactly what `fixed` expects.

### ☐ 3. Extract and move the Edit 1 guard

Add near `resolve_param_overrides` in `benchmark.py`:

```python
def reject_param_overrides_outside_sweep(args) -> None:
    """
    --param-overrides reaches only the sweep path; refuse it anywhere else.

    On the plain-benchmark path parameters come from args.model.profile.params and
    args.param_overrides is never read, so accepting it silently would run the control
    while the run's --name, its W&B config and the lab notebook all claim a treatment.
    Same reasoning as resolve_param_overrides' unknown-name error: raise, don't warn.
    """
    if args.param_overrides and not args.sweep:
        raise ValueError(
            "--param-overrides applies only to --sweep; it has no effect on the plain "
            "benchmark path. Set profile params directly instead, e.g. "
            "--model.profile.params.icp-visibility-cull\n"
            f"Received: {dict(args.param_overrides)}"
        )
```

Call it as the second line of `main()`:

```python
def main():
    args = tyro.cli(BenchmarkArgs)
    reject_param_overrides_outside_sweep(args)
```

and **delete** the inline block currently at `benchmark.py:1138-1142`. This buys two things
at once: the check now runs before `load_hf_model` and `load_parquet_dataset` instead of
after them, and it becomes callable from a test without a dataset (§6.2).

### ☐ 4. `benchmark.py:714` — the unbalanced quote

```
`icp_refine_ladder "(0.05,0.02,0.01)`     ->     `icp_refine_ladder "(0.05,0.02,0.01)"`
```

### ☐ 5. Write `tests/test_param_overrides.py`

Full walkthrough in §6. Write **test 8** (`test_no_parameter_suggested_twice`) first — it is
three lines and it is the regression guard for step 1b.

### ☐ 6. Verify

```bash
uv run pytest tests/ -q                       # nothing else broke
uv run ruff check . && uv run ruff format --check .
uv run benchmark.py --eval-size 2 --param-overrides icp_visibility_cull true \
  model:vsac3dof model.profile:tuned          # must fail instantly, before asset loading
```

Then a 3-trial smoke sweep, which is what actually proves step 1b:

```bash
uv run benchmark.py --sweep --trials 3 --eval-size 6 --name SmokeSweep \
  --param-overrides icp_visibility_cull true \
  model:vsac3dof model.profile:tuned
```

Good: three trials complete. Bad: `ValueError: Cannot set different log configuration to the
same parameter name` — that means a `ransac_max_iterations` suggestion survived somewhere.

### ☐ 7. Only then, note 02

[02 — Deriving the fixed parameters](02-deriving-the-fixed-parameters.md).

---

## 0. What the W&B runs showed

Four runs carry `commit = f4f83c9`, all 2026-07-29. Three form a clean single-variable
ladder over 1481 frames × 3 internal seeds = 4443 evaluations:

| run | arm | `pose_ar` | `good_rate` | gross yaw | abstain | trans p50 | yaw p50 | p95 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ifprheyj` | baseline (`tuned`) | 0.1754 | 0.5540 | 0.4453 | 0.0007 | 2.78 cm | 0.386° | 5.55 s |
| `om1fcwqn` | + cull + ladder + guard | 0.3508 | 0.5559 | 0.4441 | 0.0000 | **0.81 cm** | 0.145° | 5.70 s |
| `s71rtlvr` | + front-face gate @ 60° | **0.5174** | **0.9293** | 0.0687 | 0.0020 | 0.93 cm | 0.135° | **3.08 s** |

**The cull replicated.** 2.00× `pose_ar` at n=4443 against 2.08× at n=18; translation
2.78 → 0.81 cm; `good_rate` flat to 0.2 pp, exactly the guarantee the yaw guard provides.

**The gate overperformed its fixture estimate by 2.2×.** Fixtures predicted `good_rate`
0.611 → 0.778 (+0.167); the full set gives 0.554 → 0.929 (**+0.375**), with 9 abstentions
out of 4413. Note 30.06 says 18 fixtures overstate effects ~2.5×; here they *understated*.
**The fixture set is a smoke test, not a calibrated proxy** — the caveat in 30.06 should be
strengthened from "only the ranking transfers" to "not even the sign of the error is
predictable".

Two defects in how the batch was run:

* **Three arms, three different seeds** (2144065271 / 1629433846 / 2072881601). Nothing is
  at risk at n=4443 against a +0.375 effect, but the arms are not **paired**, so
  `scripts/mcnemar_arms.py` cannot be applied. §4 fixes this for free on the next batch.
* **The 100-trial sweep `0891mgm4` ran with the gate off**, because
  `front_face_max_angle_deg` is absent from `suggest_params` and no override supplied it.
  Best trial 0.396 on 60 frames against 0.517 for the hand-set gate arm on the full set.
  Roughly 60 GPU-hours in a subspace with the strongest known lever disabled.

---

## 1. Vocabulary

| term | meaning here |
| :--- | :--- |
| **arm** | one evaluation run in a controlled comparison, every parameter fixed except the one under test. From clinical trials: control arm vs treatment arm. |
| **paired comparison** | two arms on *the same frames in the same order with the same internal RNG*, so per-frame outcomes match 1:1. Required by McNemar. |
| **search space** | exactly those parameters on which `trial.suggest_*` is called. A parameter set by other means is not in it, however much it varies between runs. |

---

## 2. Edit 1 — `--param-overrides` on the plain path must be a hard error

**Do this one first.** Cheapest edit here, most likely to save a multi-hour run.

`main()` branches at roughly `benchmark.py:1091`, and the two branches read parameters
from entirely different places:

| | plain benchmark | sweep |
| :--- | :--- | :--- |
| params from | `args.model.profile.params` | `{**suggest_params(trial), **resolved_overrides}` |
| `--model.profile.params.*` | ✅ authoritative | ignored |
| `--param-overrides` | **silently ignored** | ✅ the only mechanism |

`args.param_overrides` appears exactly once outside its own definition, at
`benchmark.py:1108`, inside the `run_parameter_sweep(...)` call. The `else:` branch never
references it. A plain benchmark carrying `--param-overrides` therefore runs the
**control** while its W&B config, its `--name`, and your notebook all claim otherwise.

This is precisely the failure `resolve_param_overrides` exists to prevent — its own
docstring says replacing one silent no-op with another "would leave the same failure
available — an arm that reports it is testing something while running the control". The
mechanism exists; it is not wired to this path.

**The edit.** In the `else:` branch, **before** `wandb.init` and before any compute, raise
if `args.param_overrides` is non-empty. Placement matters: a mistake should cost a second,
not an hour plus an orphaned W&B run. Let the message teach — the person hitting it
believes they used the right flag:

```
--param-overrides is ignored on the plain benchmark path (it applies only to
--sweep). Set profile params directly instead, e.g.
  --model.profile.params.icp-visibility-cull
Received: {...}
```

Raise, do not warn, for the reason the existing docstring gives: nobody reads a warning
in a long log.

---

## 3. Edit 2 — two docstrings state the wrong CLI syntax

`BenchmarkArgs.param_overrides` (`cli_config.py:664-678`) and `resolve_param_overrides`
(`benchmark.py:712-714`) both document:

```
--param-overrides front_face_max_angle_deg=60.0
```

**Wrong.** `param_overrides` is `dict[str, str]`, and tyro parses a dict flag as an
even-length run of alternating keys and values. Verified against your installed tyro:

```
['--param-overrides', 'a=true', 'b=60.0']        -> {'a=true': 'b=60.0'}
['--param-overrides', 'a', 'true', 'b', '60.0']  -> {'a': 'true', 'b': '60.0'}
```

The `=` form does not pass silently — the bogus key fails the unknown-name check — but it
fails with a message about an unknown *parameter*, sending you hunting for a typo in the
name rather than in the separator.

**The edit.** Correct both to the space-separated form; add one sentence to
`BenchmarkArgs.param_overrides` saying it applies **only to `--sweep`** (the current text
says "this is how an A/B arm is declared" with no qualifier, which is what makes Edit 1's
no-op so easy to walk into); and spell out the tuple case, which is genuinely non-obvious:

```bash
--param-overrides icp_refine_ladder "(0.05,0.02,0.01)"
```

Parentheses and commas because the value goes through `ast.literal_eval`; no internal
spaces because the shell would otherwise split it into separate dict tokens.

---

## 4. Edit 3 — teach `suggest_params` what is already fixed

> **Status: half-landed, and currently breaking every sweep.** The signatures exist on
> `base.py` and `ransac.py`; the call site is not wired and three estimators still have the
> old signature. This section explains *why* the change is shaped the way it is — for the
> ordered list of what is left to type, see §0 steps 1–4, and for the diagnosis see §9.

Lower priority than it was, because note 02 removes six parameters from the search space
outright, which addresses most of the waste. Recorded because the mechanism is still wrong.

`benchmark.py:873` builds each trial as:

```python
suggested_params = {**estimator_cls.suggest_params(trial), **resolved_overrides}
```

When an override names a parameter `suggest_params` *does* propose, the
`trial.suggest_*` call has **already executed** when the merge overwrites its result.
Optuna records the parameter, TPE models it, and the trial ignores the sampled value. The
run is correct — the pinned value executes — but a dimension with zero influence on the
objective is being modelled, wasting trials and making parameter-importance meaningless
for that field.

Overrides for parameters never proposed (`icp_visibility_cull`, `front_face_max_angle_deg`, …)
are unaffected; they were never in the search space.

**The edit.** Pass the fixed names down and guard each suggestion:

```python
@classmethod
def suggest_params(cls, trial, fixed: frozenset[str] = frozenset()) -> dict[str, Any]:
    params = {}
    if "icp_max_correspondence_distance" not in fixed:
        params["icp_max_correspondence_distance"] = trial.suggest_float(...)
    ...
```

with `estimator_cls.suggest_params(trial, frozenset(resolved_overrides))` at the call site.
Touches every implementing class — `RansacEstimator`, `Ransac3DoFEstimator`,
`Ransac3DoFFullMeshEstimator`, `VSACSe2Estimator`, `PPFEstimator` — plus subclasses that
`super()`-call. A `frozenset()` default keeps every existing call working, so it can land
one estimator at a time.

**The alternative I would not take.** `optuna.samplers.PartialFixedSampler(fixed_params=…,
base_sampler=TPESampler(…))` does this in one line at study creation, but only for keys
*in* the search space — you would keep the dict merge for the rest and maintain two
mechanisms each covering half the cases.

---

## 5. Pairing: reuse the seed

`draw_eval_indices(total, eval_size, seed)` and `derive_internal_seeds(seed, n_seeds)` are
both pure functions of `--seed`. Passing **`--seed 2144065271`** — `s71rtlvr`'s seed —
reproduces the same frames in the same order with the same internal RNG, making every new
arm **paired** against the existing gate arm. `scripts/mcnemar_arms.py` then applies, and
you do not need to re-run the control.

Do this on every arm from now on. It costs nothing and it is the difference between "the
rate went up" and "the rate went up, p = 0.003".

---

## 6. Tests to write

There is currently **no test in `tests/` referencing `param_overrides`**. Everything below
is pure-function and fast — no dataset, no YOLO weights, no W&B, no Optuna study. The whole
file should run in well under a second.

Put it in `tests/test_param_overrides.py`, `unittest` style to match the rest of the suite.

### 6.0 The two tools you need

**`assertRaises` as a context manager.** To test that something *fails*, you cannot just
call it — the exception would fail the test. You wrap it:

```python
with self.assertRaises(ValueError) as ctx:
    resolve_param_overrides(VSACSe2Estimator, extrinsic, {"typo_name": "true"})
self.assertIn("typo_name", str(ctx.exception))
```

The `with` block passes only if a `ValueError` (or a subclass) is raised inside it. `ctx`
then holds the exception, so `str(ctx.exception)` is the message — letting you assert the
message is *useful*, not just that something blew up. That second assertion is what stops
a future refactor from replacing a helpful error with a bare `raise ValueError()`.

**A stub (or "fake").** An object that looks like a real dependency to the code under test,
but is simple enough to inspect. Here the real dependency is `optuna.Trial`, which needs a
live study and a sampler.

Your suite already has one convention for this: `tests/test_estimators.py` uses
**`optuna.trial.FixedTrial`**, Optuna's own stub, constructed from a dict of pre-chosen
values. It has a property that makes it useful beyond convenience: **asking it for a name
that is not in its dict raises**. So test 7 can be written in the house style with no new
machinery —

```python
    def test_fixed_params_are_not_suggested(self):
        # voxel_size is deliberately absent from the dict: if the guard works it is
        # never requested, and if it is broken FixedTrial raises on the lookup.
        trial = optuna.trial.FixedTrial({"rho": 0.1, ...})
        VSACSe2Estimator.suggest_params(trial, fixed=frozenset({"voxel_size"}))
```

— and passing *is* the assertion. Prefer this where it fits; matching the existing suite is
worth more than a marginally prettier test.

**But `FixedTrial` cannot detect a duplicate suggestion**, which is test 8's whole job: ask
it for the same name twice and it cheerfully answers twice. For that you need a stub that
records, because the question is "how many times was this asked?", not "what came back":

```python
class RecordingTrial:
    """Stands in for optuna.Trial and records which parameters were asked for."""

    def __init__(self):
        self.asked: list[str] = []

    def suggest_float(self, name, low, high, step=None, log=False):
        self.asked.append(name)
        return low

    def suggest_int(self, name, low, high, step=1, log=False):
        self.asked.append(name)
        return low
```

```python
class RecordingTrial:
    """Stands in for optuna.Trial and records which parameters were asked for."""

    def __init__(self):
        self.asked: list[str] = []

    def suggest_float(self, name, low, high, step=None, log=False):
        self.asked.append(name)
        return low

    def suggest_int(self, name, low, high, step=1, log=False):
        self.asked.append(name)
        return low
```

Returning `low` is arbitrary — the tests never look at the values. **The stub's whole
purpose is `self.asked`**: it turns "which parameters did the code ask Optuna for?" into a
plain list you can assert on. That question is invisible from the outside otherwise, and
§6.3 is entirely about questions of that shape.

So: `FixedTrial` for tests about *values*, `RecordingTrial` for tests about *which questions
were asked*. Both belong in the file.

### 6.1 Tests against `resolve_param_overrides`

```python
import unittest
import numpy as np
from benchmark import resolve_param_overrides
from cli_config import CameraConfig
from methods.vsac_se2 import VSACSe2Estimator


class ResolveParamOverridesTest(unittest.TestCase):
    def setUp(self):
        self.extrinsic = np.array(CameraConfig().extrinsic, dtype=np.float64)

    def resolve(self, overrides):
        return resolve_param_overrides(VSACSe2Estimator, self.extrinsic, overrides)
```

`setUp` runs before each test method; the `resolve` helper just saves repeating the first
two arguments. Now one test per guarantee:

**1. An unknown name raises, and says which name.** The anti-typo guarantee — the reason
the function rejects rather than warns. Note the deliberately misspelled key:

```python
    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.resolve({"icp_visibility_culll": "true"})
        self.assertIn("icp_visibility_culll", str(ctx.exception))
```

*Fails if:* someone softens the error to a warning, or drops the offending name from the
message. Either regression reinstates the phantom-arm failure — a run that reports it is
testing something while executing the control.

**2. Booleans become real booleans.** This is the subtle one:

```python
    def test_bools_become_real_bools(self):
        out = self.resolve({"icp_visibility_cull": "true",
                            "hoppe_normal_orientation": "false"})
        self.assertIs(out["icp_visibility_cull"], True)
        self.assertIs(out["hoppe_normal_orientation"], False)
```

**Use `assertIs`, never `assertTrue`.** `assertTrue(out["icp_visibility_cull"])` passes when
the value is the *string* `"true"` — because every non-empty string is truthy in Python —
and that string is exactly the bug. It would also pass on the string `"false"`, which is the
same bug pointing the other way. `assertIs(x, True)` demands the actual `True` singleton, so
it fails on both. This test exists specifically to pin the `raw.capitalize()` behaviour in
`resolve_param_overrides`; delete that call and this test goes red while a truthiness
assertion would stay green.

**3. `"none"` becomes `None`.**

```python
    def test_none_becomes_none(self):
        out = self.resolve({"front_face_max_angle_deg": "none"})
        self.assertIsNone(out["front_face_max_angle_deg"])
```

Same trap as #2: the string `"none"` is truthy, so a gate written as
`if params.front_face_max_angle_deg:` would silently switch the front-face gate **on** at
whatever the string coerces to, in a run whose command line says to turn it off.

**4. Tuples parse, with float members.**

```python
    def test_tuple_literal(self):
        out = self.resolve({"icp_refine_ladder": "(0.05,0.02,0.01)"})
        self.assertEqual(out["icp_refine_ladder"], (0.05, 0.02, 0.01))
        self.assertTrue(all(isinstance(v, float) for v in out["icp_refine_ladder"]))
```

The second assertion matters because `assertEqual` is happy comparing `(0.05, 0.02, 0.01)`
to a tuple of numpy scalars or of `Decimal`s; the ICP ladder needs plain floats.

**5. An unparseable value falls through to the raw string.**

```python
    def test_unparseable_falls_through(self):
        out = self.resolve({"z_offset": "not-a-number"})
        self.assertEqual(out["z_offset"], "not-a-number")
```

This pins the `except (ValueError, SyntaxError)` branch. It is documented behaviour, not an
accident — a parameter could legitimately be a string — so it deserves a test saying so.

**6. Non-string values pass through untouched.**

```python
    def test_non_string_passes_through(self):
        out = self.resolve({"icp_yaw_guard_deg": 5.0})
        self.assertEqual(out["icp_yaw_guard_deg"], 5.0)
```

Pins the `else` branch. Cheap, and it stops someone "simplifying" the `isinstance(raw, str)`
check away — which would crash `ast.literal_eval` on a float.

### 6.2 The test for Edit 1, and the refactor it needs

Edit 1's guard currently lives *inside* `main()`, which makes it untestable without running
the whole program. Extract it into a module-level function in `benchmark.py`:

```python
def reject_param_overrides_outside_sweep(args) -> None:
    """--param-overrides only reaches the sweep path; refuse it anywhere else."""
    if args.param_overrides and not args.sweep:
        raise ValueError(...)
```

and call it immediately after `args = tyro.cli(BenchmarkArgs)`. That is worth doing for two
independent reasons: it becomes testable, **and** it moves the check ahead of
`load_hf_model` and `load_parquet_dataset`, so a bad command line fails instantly instead of
after the YOLO weights and parquet shards have loaded. See §7.

The test then needs no dataset at all, because the function only touches two attributes:

```python
from types import SimpleNamespace
from benchmark import reject_param_overrides_outside_sweep


class RejectOverridesOutsideSweepTest(unittest.TestCase):
    def test_plain_path_with_overrides_raises(self):
        args = SimpleNamespace(sweep=False, param_overrides={"voxel_size": "0.02"})
        with self.assertRaises(ValueError) as ctx:
            reject_param_overrides_outside_sweep(args)
        self.assertIn("--model.profile.params", str(ctx.exception))

    def test_sweep_path_with_overrides_is_fine(self):
        args = SimpleNamespace(sweep=True, param_overrides={"voxel_size": "0.02"})
        reject_param_overrides_outside_sweep(args)   # must not raise

    def test_plain_path_without_overrides_is_fine(self):
        args = SimpleNamespace(sweep=False, param_overrides={})
        reject_param_overrides_outside_sweep(args)
```

`SimpleNamespace(sweep=False, ...)` builds a throwaway object whose attributes are whatever
you pass — a one-line stand-in for the real `BenchmarkArgs`, which would otherwise drag in
a full tyro parse. The function reads `.sweep` and `.param_overrides` and nothing else, so
the stand-in is complete.

Three tests, not one, because the guard has to be right in **both** directions: a guard that
raises unconditionally would pass a single "it raises" test while breaking every sweep you
own. The second test is the one that catches that, and a test with no assertion in it is
correct here — "this call completes without raising" is the entire claim.

The `assertIn("--model.profile.params", ...)` pins the *helpful* part of the message. The
person who hits this error believes they used the right flag; an error that does not name
the right one sends them to the source.

### 6.3 Tests for Edit 3 — and the one that would have caught the bug

**7. A fixed parameter is not suggested.**

```python
    def test_fixed_params_are_not_suggested(self):
        trial = RecordingTrial()
        VSACSe2Estimator.suggest_params(trial, fixed=frozenset({"voxel_size"}))
        self.assertNotIn("voxel_size", trial.asked)
        self.assertIn("rho", trial.asked)
```

**Why assert on `trial.asked` and not on the returned dict.** The returned dict is the wrong
place to look: `benchmark.py` merges `{**suggest_params(trial), **resolved_overrides}`, and
the override wins either way. So the dict holds the correct value **whether or not the bug
is present** — a test asserting on it passes in both worlds and tells you nothing. The
defect is not a wrong value; it is *a question asked of Optuna that should never have been
asked*, which inflates the search space by a dimension that cannot affect the objective. The
only place that question is visible is the call record. Hence the stub.

The second assertion (`"rho"` still asked) is not filler: it catches a guard written too
broadly, e.g. one that early-returns and stops suggesting anything once `fixed` is non-empty.

**8. No parameter is suggested twice.** Add this one — it is three lines and it would have
caught the bug currently in your working tree before you ever launched a sweep:

```python
    def test_no_parameter_suggested_twice(self):
        for cls in (VSACSe2Estimator, Ransac3DoFEstimator, Ransac3DoFFullMeshEstimator):
            with self.subTest(estimator=cls.__name__):
                trial = RecordingTrial()
                cls.suggest_params(trial)
                duplicates = {n for n in trial.asked if trial.asked.count(n) > 1}
                self.assertEqual(duplicates, set(), f"suggested twice: {duplicates}")
```

`suggest_params` builds a dict, so a name suggested by both a base class and its subclass
leaves **no trace in the output** — the second write just overwrites the first. But Optuna
sees two calls, and depending on how the two distributions differ it either raises on every
trial or silently returns the first call's value while the code reads as though the second
range applied. Both outcomes are bad; only one is loud. See §9.

`self.subTest(...)` runs the body once per estimator and reports each independently, so a
failure names the offending class instead of stopping at the first one.

**9. The signature is uniform across estimators.** Guards against the `TypeError` in §9:

```python
    def test_all_estimators_accept_fixed(self):
        for cls in (RansacEstimator, Ransac3DoFEstimator, Ransac3DoFFullMeshEstimator,
                    VSACSe2Estimator, PPFEstimator):
            with self.subTest(estimator=cls.__name__):
                trial = RecordingTrial()
                cls.suggest_params(trial, fixed=frozenset())   # must not TypeError
```

An abstract-base default that raises `NotImplementedError` is fine to exclude; every
concrete estimator reachable from `ModelPreset` must be in this list.

### 6.4 Running them

```bash
uv run pytest tests/test_param_overrides.py -v
```

`pytest` discovers and runs `unittest.TestCase` classes natively, so the `unittest` style
used across this suite needs no adaptation. `-v` prints one line per test, which is what you
want while writing them — the names become a readable list of the guarantees you have
pinned.

While writing each test, **make it fail first**: revert the behaviour it targets (comment out
the `raw.capitalize()`, or the `if ... not in fixed`), confirm red, then restore and confirm
green. A test you have never seen fail is a test you have not verified tests anything — and
for tests 2, 3 and 7 in particular, the plausible-but-wrong version passes on broken code.

---

## 7. How to verify

**Edits 1–2, seconds, local.** Run a plain benchmark with an override and confirm it dies
immediately, before W&B initialises:

```bash
uv run benchmark.py --eval-size 2 --param-overrides icp_visibility_cull true \
  model:vsac3dof model.profile:tuned
```

Good: immediate `ValueError` naming `--model.profile.params`. Bad: the run starts and
reports `icp_visibility_cull: false` in its W&B config — current behaviour, and the reason
for the edit.

**Edit 3.** Test 7 is the verification; there is no run-level signal, which is exactly why
the test asserts on the stub trial's call record rather than the output.

---

## 8. APIs and details

**`ast.literal_eval`** (`benchmark.py:731`). Parses a string containing a *Python literal* —
number, string, tuple, list, dict, `True`/`False`/`None` — and returns the value. Unlike
`eval` it never executes code, so it is safe on untrusted input; it raises `ValueError` or
`SyntaxError` on anything else. It is why the ladder override needs Python tuple syntax.
The `raw.capitalize() if raw in _BOOLS` on that line is shell ergonomics: `literal_eval("true")`
raises, `literal_eval("True")` works. Without it, `"true"` would fall through to the raw
string — which is truthy, so the bug would be invisible.

**`dict[str, str]` in tyro** (`cli_config.py:678`). tyro builds parsers from type
annotations, so it normally converts for you. `dict[str, str]` deliberately erases that
information: one flag must accept values for fields of a dozen types, and tyro cannot know
at parser-construction time which you will name. Hence the hand-rolled coercion. The flag
consumes an even-length run of tokens as alternating keys and values; an odd count is a
parse error.

**Nested tyro flags.** Names dash-convert at every level; subcommand-scoped flags must
appear *after* their subcommand token; booleans generate a `--x` / `--no-x` pair. So
`hoppe_normal_orientation` is `--model.profile.params.hoppe-normal-orientation`, valid only
after `model:vsac3dof model.profile:tuned-vis-gate`. Union types print as alternatives:
`tuple[float, ...] | None` renders as `{None}|{[FLOAT [FLOAT ...]]}`, meaning the literal
`None` or a space-separated run of floats.

**McNemar's test** (`scripts/mcnemar_arms.py`). For two binary classifiers on the *same*
items, it discards the cases where both agree and tests only the disagreements: of the
frames where exactly one arm succeeded, is the split significantly off 50/50? That is why
pairing is mandatory — without identical frames and seeds there are no matched pairs, and
the test does not apply.

---

## 9. Review of the first implementation pass

Read against the working tree of 2026-07-31. Edits 1 and 2 are substantially right; Edit 3
is half-landed and currently **breaks every sweep**.

### 9.1 Blocking — `ransac_max_iterations` is suggested twice

`RansacEstimator.suggest_params` now adds

```python
params["ransac_max_iterations"] = trial.suggest_int(
    "ransac_max_iterations", 10000, 200000, step=10000)
```

while `Ransac3DoFEstimator.suggest_params` (`methods/ransac3dof.py:698`) and
`Ransac3DoFFullMeshEstimator.suggest_params` still call

```python
trial.suggest_int("ransac_max_iterations", 2000, 100000, log=True)
```

Same parameter name, two distributions, one trial. Measured against your installed Optuna:

```
ValueError: Cannot set different log configuration to the same parameter name.
[W] Trial 0 failed with value None.
```

**Every trial of every sweep dies on trial 0.**

**The loudness is luck, not design.** `optuna.distributions.check_distribution_compatibility`
raises on exactly three mismatches: distribution *class*, the *log* flag, and categorical
*choices*. Different `low`/`high`/`step` with the same `log` setting **passes silently**, and
the second call then returns the value the first call sampled. Had the base been written
`log=True`, this bug would have been invisible: the effective range would be the base's
`[10000, 200000]` while the subclass line, and every code reader, said `[2000, 100000]`.
That is the same phantom-arm failure mode as Edit 1, one layer down.

**The fix, and it is not "pick one".** Note
[02 §7](02-deriving-the-fixed-parameters.md) establishes from your own measured ladder that
`ransac_max_iterations` should leave the search space entirely and become a per-profile
budget. So delete it from **both** `suggest_params` implementations. Two further reasons the
new range is wrong on its own terms: the floor of 10 000 discards the cheap end of a curve
you measured as monotone from 1 000, and the ceiling of 200 000 is 4× beyond the largest
value ever run on this project, at a p95 already 5.6 s at 46 940.

### 9.2 Blocking — the `fixed` mechanism is never invoked

`benchmark.py:873` is unchanged:

```python
suggested_params = {**estimator_cls.suggest_params(trial), **resolved_overrides}
```

`fixed` therefore defaults to `frozenset()` on every call, and no parameter is ever skipped.
The signatures are plumbed; the wire is not connected. It needs:

```python
suggested_params = {
    **estimator_cls.suggest_params(trial, fixed=frozenset(resolved_overrides)),
    **resolved_overrides,
}
```

Keep the merge as well as the `fixed` argument — they do different jobs. `fixed` stops the
*suggestion*; the merge supplies the *value*, including for parameters `suggest_params` never
proposes.

### 9.3 Blocking — two estimators will `TypeError` the moment 9.2 lands

Still on the old signature:

* `VSACSe2Estimator.suggest_params(cls, trial)` — `methods/vsac_se2.py:598`. **This is the
  estimator you actually run.**
* `Ransac3DoFFullMeshEstimator.suggest_params(cls, trial)` — `methods/ransac3dof.py:732`,
  which additionally calls `RansacEstimator.suggest_params(trial)` without forwarding
  `fixed`, so the base guard would be bypassed even after the signature is fixed.

Check `PPFEstimator` too. Test 9 in §6.3 is the guard against this whole class of omission.

### 9.4 Important — the subclass's own parameters are unguarded

In `Ransac3DoFEstimator.suggest_params`, `edge_length_threshold`, `z_gate_threshold` and
`front_crop_depth` are suggested unconditionally. These are precisely the parameters note 02
is about, so pinning `z_gate_threshold` via `--param-overrides` would still waste its
dimension. Each needs the same `if ... not in fixed` guard.

### 9.5 Minor

* **Unbalanced quote**, `benchmark.py:714`: `` `icp_refine_ladder "(0.05,0.02,0.01)` `` is
  missing its closing `"`. Harmless at runtime — it is a docstring — but it teaches the
  wrong syntax, which is what the edit was for.
* **Inconsistent examples.** `cli_config.py` writes `"(0.05, 0.02, 0.01)"` with spaces,
  `benchmark.py` without. Both work *when quoted*; unify them, and prefer the space-free
  form so the example survives being pasted without quotes.
* **Guard placement.** The raise sits inside the `else:` branch, after `load_hf_model` and
  `load_parquet_dataset` have already run at the top of `main()`. It is correctly before
  `wandb.init`, but a bad command line still costs the full asset load. §6.2's extraction
  fixes this and makes it testable in one move.
* **The message drops the offending dict.** Appending `Received: {args.param_overrides}`
  costs nothing and turns "which one did I get wrong?" into a glance. Compare
  `resolve_param_overrides`, which names the bad key and lists the valid ones.
* **`if TYPE_CHECKING: pass`** left behind in `methods/base.py` and `methods/ransac.py`,
  and `trial` lost its `"optuna.Trial"` annotation. The original pattern was deliberate:
  the string annotation plus a `TYPE_CHECKING`-only import gives type checkers the real type
  without importing Optuna at runtime, which matters because `pipeline.py` imports these
  modules and Optuna is a heavy import. Restore `trial: "optuna.Trial"` and the import,
  rather than dropping both.
* **`methods/base.py` docstring**: the blank line before `Returns:` was removed, which ends
  the Google-style `Args:` block early. Also worth splitting the `fixed` description across
  lines — `E501` is in your ruff ignore list so nothing will complain, but it is a 130-column
  line in an 100-column file.

### 9.6 Not a mistake

Several one-line reformattings in the diff (`gross_yaw_rate_per_seed.append(...)`, the
`gross_yaw_rate` print, `orient_normals_hoppe`'s signature, the `trimesh.Trimesh` call) are
`ruff format` output at your configured 100 columns, not edits. The `AGENTS.md` rewrite
correctly collapses the old §6 into a pointer at `CLAUDE.md`, which is what §4 of that file
asks for.

---

## 10. Related

[02 — Deriving the fixed parameters](02-deriving-the-fixed-parameters.md) · vault
`30.06 - T0 Translation Error and the Visibility Cull` ·
`Local Evaluation Harness - Worklog` (the o3d RNG trap).

W&B: `ifprheyj`, `om1fcwqn`, `s71rtlvr` (the arms), `0891mgm4` (the gate-off sweep).
