# 6DPose — Remediation Roadmap

## Context

A full review of the codebase (~1,970 LOC) found the architecture sound — the Strategy pattern in
`methods/`, `MaskedImageFrame` as a coordinate-safety type, and `COORDINATE_TRANSFORMS_MEMO.md` are
all above the bar. The problems live in the layer *around* that core:

1. **It cannot start.** `config/config.yaml:3` requires a `config/dataset/` group that does not exist
   and that `.gitignore:16` (`dataset/`) silently prevents from ever being committed.
2. **The benchmark's numbers are not trustworthy.** For a project whose deliverable *is* a method
   comparison (Objective 5) and a latency guarantee (Objective 1), this is what matters. Several
   defects corrupt results *silently* rather than crashing.

Intended outcome: a codebase where `main` runs from a fresh clone, benchmark numbers are reproducible
and mean what they claim, and measured latency predicts Jetson deployment.

**Sequencing rule driving this plan:** Milestone B (trustworthy metric) must land *before* the feature
work from the 2026-07-09 Keshav call (3-DOF constraint, front-face prim). Otherwise there is no way to
tell whether those changes helped.

## Scope of this task

**Deliverable is the document only — no code changes.** Publish this roadmap as a polished, shareable
document (suitable for sharing with Keshav). Implementation is deferred to later sessions.

## Decisions taken

| Fork | Decision |
|---|---|
| Metric rewrite | **BOP-style Average Recall** — `1 − AR` over a threshold grid + p95 latency |
| Rotation error | **Yaw-first**, anticipating the 3-DOF constraint; report tilt drift as a diagnostic |
| Symmetry-aware metric | **Rejected** — carts are asymmetric; it would hide the worst failure mode |

## Dependency spine

```
Milestone A (Unblock) ──gates──▶ Milestone B (Trustworthy metric) ──gates──▶ Downstream
      │                                                                  (3-DOF, front-face prim, tyro)
      └── C (Honest latency)  ·  D (Correctness/tests/CI)  ·  E (Housekeeping)  ·  F (ROS2 packaging)
                              (hardening tracks — run after A, alongside/after B)
```

A gates everything (nothing is verifiable until it lands). B gates the downstream feature work.
C/D/E/F are hardening tracks that run once A is in.

---

## Milestone A — Unblock (nothing else is verifiable until this lands)

| # | Task | Files |
|---|---|---|
| A1 | Create `config/dataset/default.yaml` with `path` + `test_glob` (mirror `main.Config.DATASET_PATH` / `TEST_PARQUET_GLOB`) | `config/dataset/default.yaml` |
| A2 | Scope the gitignore pattern `dataset/` → `/dataset/` so it stops matching `config/dataset/` at any depth | `.gitignore:16` |
| A3 | Delete `main.Config`; make Hydra the single source of truth | `main.py:134-161`, `inspect_pose.py`, `benchmark.py` |

**A3 detail** — `main.Config` is a shadow config system that silently *beats* Hydra in three places:

- `inspect_pose.py:161` uses `Config.OUTPUT_DIR`, ignoring `cfg.output_dir`
- `inspect_pose.py:192,283` omit `depth_trunc`, silently using the `main.Config` default of **3.0** and
  ignoring `cfg.depth_trunc`. So `ransac_pareto1.yaml`'s `depth_trunc: 6.2` is honoured by benchmark
  but **not** by inspect. The two runners therefore reconstruct *different clouds from the same
  sample, silently*: at 3.0m the floor and the cart's far extent get clipped that 6.2m would keep.
  (The cart center is not truncated away — the reconstructed cloud center sits at camera-frame
  Z ≈ 2.53m per memo §4, which survives a 3.0m cutoff. The `3.2055m` figure in the memo is the
  robot-frame **X** coordinate, not camera depth.)
- `inspect_pose.py:212,300` omit the extrinsic, falling back to hardcoded `Config.T_ROBOT_CAMERA` —
  the exact bug `benchmark.py:171-175`'s own comment warns about

Collapse the extrinsic's **four** copies (`main.py:148`, `ppf.py:84-89`, `ransac.py:61-66`,
`config/camera/default.yaml`) to one, in config. (The `config/model/*.yaml` files already reference
`${camera.extrinsic}`, so they are not additional copies.)

**Acceptance:** fresh clone + `uv run inspect_pose.py mode=random random_samples=2 model=ransac` runs;
a `depth_trunc=6.2` override demonstrably changes the exported cloud (floor / far extent);
`git status` shows the dataset config tracked.

---

## Milestone B — Make the benchmark trustworthy (blocks every comparison claim)

| # | Task | Files |
|---|---|---|
| B1 | Seed-per-sweep, recorded ("record, don't fix") | `benchmark.py:213` |
| B2 | Study integrity guard | `benchmark.py:249-254` |
| B3 | Yaw-first metrics + BOP-style Average Recall objective | `benchmark.py:65-103, 217-244` |
| B4 | Count failures as misses; log swallowed exceptions | `benchmark.py:146-167` |

### B1 — seed

Draw a fresh seed per sweep, hold it for the run, persist it:

```python
seed = int(np.random.SeedSequence().entropy % (2**32))
rng  = np.random.default_rng(seed)
sweep_indices = rng.choice(total_samples, size, replace=False)
study.set_user_attr("seed", seed)
study.set_user_attr("sweep_indices", sweep_indices.tolist())
```

Gives internal consistency (all trials see identical data), no long-run seed-overfitting, and
replayability. Also seed Open3D's global RNG — `o3d.utility.random.seed()` (**verify** it exists in
0.19 before relying on it).

### B2 — study guard

On `load_if_exists=True`, compare stored `seed`/`eval_size`; **refuse** on mismatch. Direct fix for
incomparable trials merging into one Pareto front.

### B3 — the metric

The current `mean_trans + mean_rot/180 + 5.0*failed` mixes metres with normalized degrees, averages
only *successful* samples (selection bias), uses an unnormalized failure count, and encodes a **flip**
(a safety-critical discrete failure) as continuous rotation error. It also weights failures by an
arbitrary `5.0` that nobody chose — an implicit accuracy/speed exchange rate baked into a magic number.

Decompose the pose error along physically meaningful axes:

```python
yaw_err   = wrap180(yaw(R_est) - yaw(R_gt))          # atan2(R[1,0], R[0,0])
tilt_err  = angle(R_est @ [0,0,1], R_gt @ [0,0,1])   # roll/pitch drift
trans_xy  = norm(t_est[:2] - t_gt[:2])
trans_z   = abs(t_est[2] - t_gt[2])                  # cart is floor-bound; should be ~0

success   = (trans_xy < τ_t) and (abs(yaw_err) < τ_r)
flip      = abs(yaw_err) > 90
```

Objectives:

- **Obj 1** = `1 − AverageRecall`, averaging recall over a *grid* of `(τ_t, τ_r)` (keeps it smooth for
  the TPE sampler; a single threshold at `eval_size=20` gives only 21 discrete values)
- **Obj 2** = **p95** online latency (5Hz is a guarantee, not an average)
- Anchor `τ_t = 0.01m` to the project's stated "sub-centimeter" spec

Bounded `[0,1]`, normalized (comparable across `eval_size`), no unit mixing, failures counted as
misses automatically — no magic `5.0`.

**Keep geodesic rotation error as a reported diagnostic** — it is standard and correct. Do **not** make
it symmetry-aware: the carts are asymmetric (handles), so a flip is a genuine error and
min-over-symmetry-group would hide the most important failure mode.

**Diagnostics (reported, not optimized):** flip rate, `tilt_err` distribution, median error on
non-flipped samples, detection-failure vs pose-failure split.

Two things this buys beyond a better objective:

- **Flip rate** empirically settles the open Martin/Keshav disagreement — the carts are asymmetric in
  CAD, but whether that asymmetry is *observable* through a noisy simulated D455 (Keshav: 3-4mm grills
  barely image) is unknown. If the handles aren't resolved in depth, `refine_pose_dual_hypothesis` is
  choosing on noise, i.e. a coin flip. Nothing currently measures this.
- **`tilt_err`** quantifies how far the unconstrained roll/pitch actually drift — i.e. it measures the
  value of the 3-DOF branch *before* you build it.

**Acceptance:** a sweep is replayable from its recorded seed; flip rate and tilt drift are reported;
re-running with a different `eval_size` refuses to pollute the existing study.

---

## Milestone C — Honest latency & deployment-shaped lifecycle

Model-side work is currently redone **per sample**, inside the timed region:

- `ppf.py:138` `detector.trainModel()` — enumerates ~10⁶ CAD point pairs into a hash table
- `ppf.py:125`, `ransac.py:99` `sample_points_uniformly()` (also unseeded → nondeterministic)
- `ransac.py:106,118` model voxel downsample + model FPFH
- `benchmark.py:156`, `inspect_pose.py:200,295` `read_triangle_mesh()` — **disk I/O per sample**

All of it is a pure function of (CAD mesh, params) — independent of scene, camera, and viewpoint — so
in deployment it happens **once at startup**, like the YOLO weights. PPF's "training" is a misnomer:
it builds a hash table keyed on quantized 4D point-pair descriptors `(‖d‖, ∠(n₁,d), ∠(n₂,d), ∠(n₁,n₂))`.
No learning, deterministic, seconds — needs only the `.ply`.

**The harm is not merely inflated latency:** because training cost is charged per-sample in the sweep
but is free in deployment, **the Pareto front traded accuracy away to buy speed that deployment gets
for free.** Chosen params may be needlessly coarse. (Corollary: the ">10 Hz" reported to Keshav is
*pessimistic* — real online latency is better.)

| # | Task | Files |
|---|---|---|
| C1 | Add a `prepare(cad_mesh)` lifecycle method to `BasePoseEstimator`; cache per `(cart_type, params)` | `methods/base.py`, `ppf.py`, `ransac.py` |
| C2 | Time only the online path | `benchmark.py:159-168` |
| C3 | Hoist mesh loading out of the loop | `benchmark.py:156`, `inspect_pose.py` |
| C4 | Fix O(N²) dataset access | `benchmark.py:134-135`, `inspect_pose.py:181-182`, `main.py:501-502` |

**C4** — `dataset["rgb"][idx]` materializes the **entire column** (~1,482 decoded images), keeps one,
discards the rest, then repeats. `compute_ground_truth_pose` does it twice more per sample. Use
`row = dataset[int(idx)]`, or `dataset.select(indices)` and iterate sequentially.

**C1 caution:** cached model objects must not be mutated — `cad_mesh.compute_vertex_normals()` mutates
in place. Caching also incidentally removes the `sample_points_uniformly` nondeterminism.

**Acceptance:** report online-only p95 latency; confirm it is lower than the current figure; assert
`prepare()` runs once per cart type; a re-sweep produces a materially different Pareto front.

---

## Milestone D — Correctness hardening, tests, CI

| # | Task | Files |
|---|---|---|
| D1 | Refactor `compute_ground_truth_pose(dataset, idx)` → pure fn taking the two matrices | `main.py:484-509` |
| D2 | Regression-test the transform chain against the memo's verified `[3.2055, 0.0, 0.0100]` | `tests/` |
| D3 | Unit-test `MaskedImageFrame.get_o3d_intrinsics` (principal-point shift) and the ICP tie-break | `tests/` |
| D4 | Run `pytest` in CI | `.github/workflows/` |
| D5 | Document/remove in-place mutation in `prepare_scene_point_cloud` | `methods/base.py:73-84` |
| D6 | Orthonormal extrinsic (`0.866` → `√3/2`); pin `optuna` | `config/`, `pyproject.toml:19` |

**D1 is the keystone**: the same refactor fixes C4's perf problem *and* makes the most bug-prone logic
in the repo (the transform chain) unit-testable. Memo §4 already contains hand-verified numbers — they
should be a test, not prose.

**Current state:** `tests/test_estimators.py` only asserts that Optuna's `FixedTrial` returns the values
it was handed — **it tests Optuna, not this codebase**. And `.github/workflows/docs.yml` builds docs
only, so `pytest` never runs. There is CI guarding the prose and none guarding the maths.

**D5** — `prepare_scene_point_cloud` mutates the caller's `pcd` *and* returns it. Calling two estimators
on one cloud would **double-apply the extrinsic** — a live trap for the method-comparison work this
project exists to do.

**D6** — `0.866` vs `√3/2` gives `det(R) = 0.999956`, so the extrinsic is not strictly SO(3). Measured
impact ≈ 0.13mm at 3m — *not* an accuracy problem today, but free to fix under a sub-cm target.

---

## Milestone E — Housekeeping

- **Dead code:** `get_estimator` (never called — Hydra uses `_target_`; yet `docs/how-to/add_estimator.md:68`
  *instructs contributors to register there*, where it has zero effect), `Camera.get_o3d_intrinsics`
  (`main.py:429`), `width_orig`/`height_orig` (`main.py:335-336`).
- **Stale docs:** both module docstrings document an **argparse** CLI (`--random`, `--method ppf_icp`)
  that no longer exists; `ppf_icp` names nothing; `AGENTS.md:39` repeats it; `main.py:36` says `ppf_icp.py`.
- **BGR/RGB swap:** `main.py:548` takes colours from `result[0].orig_img` (ultralytics stores **BGR**)
  while `img` is RGB PIL — point-cloud colours are channel-swapped in every `.ply`/`.glb`. Cosmetic only.
- **Debug-artifact frame inconsistency:** `inspect_pose.py:289` writes the `.ply` *before* `estimate_pose`
  (camera frame); `:303` writes the `.glb` *after* (robot frame). Same index, two frames, undocumented.
- **Fragile RANSAC failure check:** `ransac.py:147` uses `np.allclose(T_init, np.eye(4))`; prefer
  `result_ransac.fitness == 0`.
- **`print` → `logging`** — `refine_pose_dual_hypothesis` prints per call (600 lines/sweep); ROS2 needs `logging`.
- **`best.pt` in git** — redundant, `load_hf_model` already downloads it. `.gitignore` covers `*.pth`, not `*.pt`.
- **Adopt Hydra's run dir** as the single artifact sink (GLBs, optuna DB, reports) instead of scattering
  outputs and gitignoring `outputs/`/`multirun/`.
- **Provenance for magic numbers** — `0.17528702727791115` (`ppf.py:24`), `0.1200659534`
  (`ransac_pareto1.yaml`): record which study/trial produced them.
- **Rename `main.py`** — it is a library with no `__main__`; `pipeline.py` or a package.
- **Design musings as code** — `main.py:168-180` (the presets proposal) belongs in an issue.

---

## Milestone F — Package for ROS2 (future-proofing, Objective 4)

No `[tool.setuptools]`/packages; `from main import ...` only resolves from the repo root. Objective 4
puts this **inside a ROS2 node**, which imports from a foreign working directory. Must become an
installable package. This also sharpens the Hydra question: `@hydra.main` hijacks CWD and owns global
state, which is hostile to library use.

---

## Downstream — separate branches/PRs (gated on Milestone B)

| Branch | Source | Note |
|---|---|---|
| **3-DOF constraint** | Keshav call 229-233 | Cart is always floor-bound → solve `(x, y, yaw)`, not 6-DOF. B3's yaw-first metric is already designed for this. |
| **Front-face / vertical-tube prim** | Keshav call 239-299, 387-463 | Crop CAD to the towable face (Keshav: approaching from the rear *"will never happen"*); possibly down to the two outermost vertical tubes (Adrian's prior cylinder-fit approach). Would let `refine_pose_dual_hypothesis` be deleted. **Risk to evaluate:** less geometry may constrain the pose more weakly (sliding along the face) — precisely why B must land first. |
| **tyro migration** | user question | **Defer.** Milestone A3 + dataclass params deliver most of the benefit and make the swap mechanical. Re-decide after F, when the ROS2 library-use constraint is concrete. Consider pydantic validators for SE(3) invariants (`R·Rᵀ ≈ I`) — would have caught the non-orthonormal extrinsic automatically. |

---

## Verification (for when implementation begins)

- **A:** fresh clone → `uv run inspect_pose.py mode=random random_samples=2 model=ransac` completes;
  a `depth_trunc` override demonstrably changes the exported cloud.
- **B:** two sweeps with the same recorded seed → identical `sweep_indices`; mismatched `eval_size` is
  refused; flip rate and tilt drift appear in the report.
- **C:** online p95 latency < current mean; `prepare()` called once per cart type (assert via counter).
- **D:** `pytest` green in CI; transform test reproduces the memo's `[3.2055, 0.0, 0.0100]`.

> ⚠️ **Environment:** the venv is unsynced and disk space is constrained. `uv sync` pulls
> torch/open3d/ultralytics — confirm headroom before running it. Do **not** run `initialize_project.py`
> on this machine (dataset download).
