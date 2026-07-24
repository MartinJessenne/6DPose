# VSAC (SE(2)) — Implementation Plan

A step-by-step build order for a VSAC-style robust estimator in SE(2), as a drop-in
alternative to `constrained_ransac_se2`. Pure numpy/scipy core (`methods/vsac_se2.py`),
a thin estimator subclass, CLI/Optuna wiring, and tests. Follows the same
algorithm-core / estimator-glue split as `constrained_ransac.py` ↔ `ransac3dof.py`.

Reference: Ivashechkin, Barath, Matas, *"VSAC: Efficient and Accurate Estimator for
H and F"*, ICCV 2021. F/H-specific parts (DEGENSAC, epipolar independent-inlier
rules, Gaussian-elimination null-space solvers) are dropped — the SE(2) minimal
solver is already an exact 2-point closed form.

**Primary payoff for this branch:** the independent-inlier signal (Steps 5–7) lets
the loop prefer spatially-spread support over the *clustered* support that a
180°-flipped pose accumulates — attacking the flip *inside* RANSAC, before the
free-space/ICP guard.

---

## Module layout

- `methods/vsac_se2.py` — new numpy/scipy algorithm core. Public entry
  `vsac_ransac_se2(...)`, superset signature of `constrained_ransac_se2`, returns
  `RansacResult` (imported from `constrained_ransac`).
- `methods/ransac3dof.py` (or a new `methods/vsac_se2_estimator.py`) —
  `VSACSe2Params(Ransac3DoFParams)` + `VSACSe2Estimator(Ransac3DoFEstimator)`.
- `cli_config.py` — new `model:vsac3dof` preset arm.
- `tests/test_vsac_se2.py` — mirrors `tests/test_se2.py`.

Reused verbatim (do **not** reimplement): `minimal_solver_se2`, `so2_exp`,
`se2_exp/log` (`methods/se2_lie_utils.py`); `RansacResult`, `se2_to_se3`,
`project_to_se2`, `match_correspondences_fpfh`, `_sample_and_validate_pair`, the
z-gate block (`methods/constrained_ransac.py`); `icp_point_to_plane_se2`
(`methods/se2_icp.py`); the whole `Ransac3DoFEstimator` pipeline.

---

## Build order

### Step 0 — Scaffold `methods/vsac_se2.py`
- Copy the header/imports and the z-gate + correspondence setup of
  `constrained_ransac_se2` (`constrained_ransac.py:228-262`) into a new
  `vsac_ransac_se2(...)`. Keep the exact same positional args; add keyword-only VSAC
  knobs (all defaulted so it stays drop-in): `use_prosac`, `use_sprt`, `msac`,
  `rho`, `poisson_confidence`, `lo_iterations`, `use_magsac_score`.
- Get a trivial pass-through working first (call into the existing binary-fitness
  loop) so the estimator + tests wire up before the algorithm is finished.

### Step 1 — PROSAC sampling  ← *first deep dive, see chat*
- Extend `match_correspondences_fpfh` with a `return_distance=False` flag (or add
  `match_correspondences_fpfh_scored`) that also returns the mutual-NN **descriptor
  distance** per correspondence — the quality signal PROSAC orders on.
- Sort correspondences ascending by that distance (best match first).
- Implement the PROSAC growth schedule (pool starts at `m=2`, grows toward
  `n_corr`), drawing the pair from the current top-`n` prefix and always including
  the newest point at a growth step. Reuse `_sample_and_validate_pair`'s
  edge-length / min-baseline validation on the drawn indices.
- Fallback: once the pool reaches `n_corr`, PROSAC degenerates to uniform RANSAC.

### Step 2 — MSAC scoring
- `score_msac(T, model_points, scene_tree, tau)` → truncated-quadratic quality.
  Per model point residual `r = ‖T·p − nn_scene‖`; cost `min(r², τ²)`; quality
  `Q = Σ(τ² − cost)` (still a *maximization*, like today's `fitness`, so the
  best-update logic is unchanged).
- Keep the vectorized kd-tree query and the random-subsample **pre-score gate**
  (`constrained_ransac.py:293-301`) — in numpy this is the practical stand-in for
  full per-point SPRT. Return `(quality, inlier_mask, dists)`.

### Step 3 — Adaptive SPRT verification
- `verify_sprt(...)`, **chunked** to stay vectorized: evaluate model points in
  blocks; accumulate Wald's likelihood ratio
  `λ_W *= (ε/δ)` for inliers, `(1−ε)/(1−δ)` for outliers, per block; bail as soon
  as `λ_W > (1−α)/β`. Parameters from Step 5's calibration: `δ0 = λ̂/T`,
  `ε0 = max(I_δ, I*_best)/T`.
- Guard: only enable when the expected speedup beats misverification cost (skip
  when `ε0 ≈ δ0`). Toggle via `use_sprt`; default on but must never be *slower*
  than the vectorized pre-score path.

### Step 4 — Independent inliers *(flip-disambiguation core)*
- `count_independent_inliers(inlier_mask, model_points, rho)`: build a `cKDTree`
  over the **inlier model points in model-frame XY** (pose-invariant); greedily
  walk inliers, marking any within `rho` of an already-accepted independent inlier
  as *dependent*; return `n_independent` (and optionally the independent index set
  for Jaccard).
- Use as a **secondary score**: when two hypotheses tie on MSAC quality within a
  small band, prefer the higher independent-inlier count. This is what suppresses
  the clustered flip in-loop.
- Default `rho ≈ 3·voxel_size` (= `min_sample_distance`).

### Step 5 — Null calibration
- `calibrate_null(independent_counts, best_inliers, all_inlier_sets)` after the
  first `n_warmup ≈ 50` accepted models: drop the current best + anything with
  `Jaccard(inliers) > 0.95` to it; `λ̃ = median` of remaining independent counts;
  trim above the 95th Poisson percentile; `λ̂ = mean` of the rest.
- Derive `δ0`, `ε0`, and `I_δ = λ̂ + 3.719·√(λ̂(1−δ0))` (≈4σ acceptance threshold).
  Feeds Steps 3, 6, 7.

### Step 6 — Random-model rejection
- `random_model_ok(n_independent_best, lam_hat, N, confidence)`: accept only if
  `C_Poisson(n_independent_best; λ̂)^N ≥ confidence`, i.e. `n_independent_best ≥ I_δ`.
  On failure return `RansacResult(np.eye(4), 0.0, np.inf)` — "no object present"
  (near-zero false positives on empty/occluded frames).

### Step 7 — Gated local optimization
- After a new best, run LO only if (a) its independent-inlier count ≥ `I_δ` **and**
  (b) `Jaccard(new inliers, prev-best inliers) < 0.95`, so LO fires ~once per run.
- `local_optimize`: 2D weighted least-squares (Umeyama/Kabsch on the XY inlier set:
  centroid-subtract, `H = Σ Δp Δqᵀ`, `U,_,Vt = svd(H)`, `R = V·diag(1,det(VUᵀ))·Uᵀ`,
  `t = q̄ − R p̄`) for a few reseeded iterations; keep the increment only if MSAC
  quality improves. Cheap and self-contained; `icp_point_to_plane_se2` stays the
  *final* refinement in `_refine_pose`, untouched.

### Step 8 — MAGSAC++-style final polish *(stretch, default off)*
- `marginalized_refine`: IRLS on inliers with weights = inlier probability
  marginalized over σ ∈ [σ_min, σ_max], removing hard dependence on a single τ.
- Behind `use_magsac_score`; **default off** for the first benchmark pass so any
  flip-rate improvement is attributable to the independent-inlier machinery alone.

### Step 9 — Assemble `vsac_ransac_se2`
- Same skeleton as `constrained_ransac_se2` (z-gate → sample → solve → pre-score
  gate → score → update best → adaptive termination via
  `log(1−conf)/log(1−w²)`), now with: PROSAC (1), MSAC (2), SPRT (3),
  independent-inlier tiebreak + collection (4/5), gated LO (7).
- Termination additionally requires the best to clear `I_δ`. Final: random-model
  check (6), optional polish (8), return via `se2_to_se3`.

### Step 10 — Estimator subclass
- `VSACSe2Params(Ransac3DoFParams)` adds: `rho`, `poisson_confidence` (0.99),
  `use_sprt`, `use_prosac`, `use_magsac_score`, `lo_iterations` (~10), `msac` (True).
- `VSACSe2Estimator(Ransac3DoFEstimator)` overrides **only** `_global_registration`
  to call `vsac_ransac_se2(...)` (mirror the arg-marshalling at
  `ransac3dof.py:252-270`). Everything else — prepare/crop/z-offset, dual-hypothesis
  `_refine_pose`, `_project_pose` — inherited unchanged.
- Extend `suggest_params` (pattern at `ransac3dof.py:319`) to sweep `rho`,
  `poisson_confidence`, and the boolean toggles.

### Step 11 — CLI / Optuna wiring
- `cli_config.py`: add `VSACSe2Profile` (mirror `Ransac3DoFProfile`, line 205),
  a `VSACSe2ProfileSelect` Union (`default`/`acc_opt`/`rt_opt`), a `VSACSe2Preset`
  with `ESTIMATOR_CLS = VSACSe2Estimator`, and a `model:vsac3dof` arm in
  `ModelPreset` (line 362). Import the new estimator/params (lines 20-24).

### Step 12 — Tests (`tests/test_vsac_se2.py`)
- Reuse `_make_scene` / `_make_corner_scene` from `tests/test_se2.py`.
- `vsac_ransac_se2` exact recovery with 30% outliers + z-offset (parity).
- `count_independent_inliers`: tight cluster of N → ≈1; N spread → ≈N.
- **Flip case**: near-symmetric scene where the flipped pose has high raw-inlier but
  clustered support; assert the independent-inlier tiebreak selects the correct
  pose where binary fitness ties. *(Branch payoff — anchor with a dedicated test.)*
- `random_model_ok`: pure-outlier / empty scene → `RansacResult(eye(4), 0, inf)`.
- SPRT on/off and PROSAC on/off reach the same optimum (behavioral equivalence).
- Seeded reproducibility.

---

## Verification

1. `uv run pytest tests/test_vsac_se2.py tests/test_se2.py tests/test_flip_disambiguation.py tests/test_cli_config.py -q`
2. A/B on real data (headline metric = **flip rate**):
   `uv run benchmark.py --eval-size 30 model:ransac3dof model.profile:acc-opt`
   vs `uv run benchmark.py --eval-size 30 model:vsac3dof model.profile:default`
   (accuracy, flip rate, AR, p95 latency are already reported by `benchmark.py`).
3. Tune new knobs:
   `uv run benchmark.py --sweep --trials 30 --name VSAC3DoF_Sweep model:vsac3dof model.profile:default`
   (writes `sweeps/optuna_VSAC3DoF_Sweep.db`).

## Notes / decisions
- In vectorized numpy the random-subsample pre-score already delivers much of
  SPRT's benefit; SPRT is chunked so it never regresses below the vectorized path,
  and stays toggleable so the benchmark can prove whether it helps.
- Neither PROSAC nor the descriptor tweaks break yaw symmetry — the flip is broken
  by the crop + independent-inlier + free-space signals.
- Companion near-term win: `FPFH_PLANAR_MATCHING_PLAN.md` Option A raises the
  correspondence inlier ratio `w` that SPRT/termination/PROSAC all feed on.
```
Build sequence: 0 → 1 → 2 → (4,5,6 together) → 3 → 7 → 9 → 10 → 11 → 12, then 8.
```
