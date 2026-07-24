# Plan: Baking the planar (SE(2) / gravity) assumption into feature matching

## Context

The 3DoF pipeline knows something FPFH does not: **the object only rotates about the vertical (+Z) axis of the robot base frame, and every point has a meaningful absolute height.** Today that knowledge is used only as a *post-hoc filter* — `match_correspondences_fpfh` (`methods/constrained_ransac.py:56`) forms mutual-NN correspondences in raw 33-dim FPFH space, and only afterwards does `constrained_ransac_se2` drop wrong-height pairs via `dz = |scene_z − model_z − z_offset| < z_gate_threshold` (`constrained_ransac.py:241-245`).

Two structural weaknesses follow:

1. **Lost inliers (Option A target).** A model point's global mutual-NN can be a spurious scene point at the *wrong* height that merely happens to be closer in descriptor space. The correct same-height match is never formed — it's discarded, not redirected. The gate removes inliers, not just outliers, depressing the correspondence inlier ratio `w` that the RANSAC termination bound `log(1−conf)/log(1−w²)` depends on.

2. **Over-invariance (Option C target).** FPFH is invariant to the **full SO(3)** rotation group. Our nuisance group is only **SO(2)** (yaw). FPFH therefore discards discriminative information for free: it assigns the same descriptor to two surface patches related by a rotation about a *horizontal* axis — a motion that can never occur for a ground-bounded cart. A descriptor that quotients out only yaw, using the known gravity direction, is strictly more discriminative on this problem.

This document develops both improvements. **Option A** is a near-term, low-risk change that complements the VSAC work (it raises the inlier ratio that SPRT and adaptive termination both consume). **Option C** is a higher-effort research direction that replaces/augments the descriptor itself.

> **Scope caveat (both options).** These changes sharpen *vertical / planar* discrimination. Neither breaks the **front/back yaw symmetry** that produces the 180° flips, because both are (correctly) yaw-invariant. The flip remains the job of the front-slab crop, the VSAC independent-inlier signal, and the free-space guard. Descriptor work and flip work are complementary, not substitutes.

---

## Option A — Height-conditioned matching (near-term)

Redirect matching so a point's nearest neighbor is chosen *within* the z-compatible band, instead of globally then filtered.

### A.1 Pragmatic implementation: weighted height augmentation

Append the (offset-corrected) absolute height as an extra, heavily-weighted descriptor dimension, then run the existing mutual-NN. Cross-height candidates are pushed out of the neighborhood *before* the mutual filter, so genuine same-height matches survive.

```python
def match_correspondences_fpfh(
    feat_model,
    feat_scene,
    model_z=None,
    scene_z=None,
    z_offset=0.0,
    z_weight=0.0,  # new, defaulted → old behavior
):
    if z_weight > 0.0 and model_z is not None and scene_z is not None:
        # Put model heights into the SAME frame as scene heights before comparing.
        fm = np.column_stack([feat_model, z_weight * (model_z + z_offset)])
        fs = np.column_stack([feat_scene, z_weight * scene_z])
    else:
        fm, fs = feat_model, feat_scene
    # ... existing mutual-NN over (fm, fs) unchanged ...
```

- **Choosing `z_weight`.** Pick it so that a height mismatch of one `z_gate_threshold` adds a descriptor-space distance comparable to the typical FPFH separation between true and false matches. A safe, self-tuning rule: `z_weight ≈ median_FPFH_NN_gap / z_gate_threshold`, measured on the first prepared model. Expose it as a swept hyperparameter rather than hard-coding.
- **Backward compatible.** `z_weight=0.0` reproduces today's behavior exactly, so this is a pure superset change.
- The downstream hard z-gate in `constrained_ransac_se2` **stays** as a final guarantee; augmentation only changes *which* correspondences form.

### A.2 Exact alternative: banded k-NN

If soft weighting proves too blunt (heights are voxel-quantized, so soft distances can be noisy), do exact banded matching: query `k>1` descriptor neighbors, keep only those inside the z-band, take the closest survivor. Implement by querying `tree.query(feat, k=k)` and masking by `|scene_z[idx] − model_z − z_offset| < z_gate`, with `k` grown adaptively when all neighbors are rejected. More correct, slightly more code; keep it as the fallback if A.1's swept `z_weight` doesn't cleanly separate.

### A.3 Call-site changes

`Ransac3DoFEstimator._global_registration` (`methods/ransac3dof.py:252`) already has `model_down`, `pcd_down`, and `self._active_z_offset` in scope — pass `model_z = model_points[:,2]`, `scene_z = scene_points[:,2]`, `z_offset=self._active_z_offset`, and a new `z_weight` param. `z_offset` is resolved in `estimate_pose` before this hook runs, so the height frames line up.

### A.4 Why it helps the VSAC iteration

Higher inlier ratio `w` → the `log(1−conf)/log(1−w²)` bound terminates sooner, SPRT's `ε0` estimate is cleaner, and PROSAC ordering starts from a higher-quality correspondence pool. Cheap multiplier on everything else.

### A.5 Effort & risk

Small (one function + one call site + one swept param). Low risk (backward-compatible default). **Recommended to land alongside or just before VSAC.**

---

## Option C — Gravity-aware descriptor (research)

Replace FPFH's SO(3) invariance with **SO(2)-only (yaw) invariance**, exploiting the known base-frame up-axis `u = (0,0,1)`.

### C.1 Rationale

FPFH encodes, per point, a histogram of Darboux angles `(α, φ, θ)` between the point's normal and its neighbors' normals — all *pairwise-relative*, hence fully rotation-invariant. In our setting, anchoring one axis to gravity lets us keep **absolute** vertical quantities that FPFH deliberately discards, while remaining invariant to the only real nuisance (yaw). More signal, same invariance budget.

### C.2 Proposed descriptor (gravity-aligned cylindrical signature)

For each point `p` with normal `n`, over neighbors `q` within radius `r`:

- **Absolute per-point terms** (new information vs FPFH):
  - height `z_p` (relative to object base / `z_offset`),
  - normal elevation `ψ = angle(n, u) ∈ [0, π]` — absolute tilt from vertical.
- **Yaw-invariant pairwise terms** over neighbors (all invariant to rotation about `u`):
  - height difference `Δz = z_q − z_p`,
  - horizontal (XY-projected) radial distance `ρ_xy = ‖(q − p)_xy‖`,
  - neighbor normal elevation `ψ_q`,
  - the angle between the two normals *projected onto the horizontal plane relative to the connecting vector's azimuth* — i.e. keep the relative azimuth `Δazimuth`, never the absolute one.

Bin these into a joint (or factored) histogram over `(Δz, ρ_xy, ψ, ψ_q, Δazimuth)`. Absolute azimuth is never encoded ⇒ yaw invariance by construction; `z`, `ψ`, `Δz` are kept ⇒ vertical discrimination FPFH throws away.

Conceptually this is "FPFH computed in a gravity-aligned cylindrical frame" — or equivalently a **SHOT-style descriptor with the local reference frame's z pinned to gravity** instead of estimated from the local surface (which removes SHOT's sign ambiguity too).

### C.3 Implementation approach

- Compute in the **base frame** (points already there after `prepare_scene_point_cloud`, `methods/base.py:68`, which also orients normals toward the camera). The CAD model is Z-up, so both sides share `u`.
- Pure-numpy/scipy custom descriptor (open3d exposes no gravity-aware option): cKDTree radius query per point, vectorized angle/height binning. Keep dimensionality modest (comparable to FPFH's 33) so matching cost is unchanged.
- Two integration modes, behind a flag:
  1. **Augment**: concatenate the gravity-aware histogram to FPFH (safest — strictly adds signal).
  2. **Replace**: use it instead of FPFH (cleaner test of the idea; more risk).
- Touch points: descriptor computation in `Ransac3DoFEstimator.prepare` (`ransac3dof.py:220`, currently `compute_fpfh_feature`) and the scene-side equivalent in the shared prep. Matching (`match_correspondences_fpfh`) is descriptor-agnostic and needs no change; Option A's height weighting still applies on top.

### C.4 Risks & open questions

- **Normal orientation consistency.** Absolute elevation `ψ` is only meaningful if normals are consistently oriented; scene normals are camera-oriented, model normals are outward — verify the two conventions agree, or use `|cos ψ|` to be sign-robust.
- **Voxel quantization** of height/normals at coarse `voxel_size` may wash out fine vertical bins — tie bin widths to `voxel_size`.
- **Tuning surface** grows (bin counts, radius, augment-vs-replace). Sweep via Optuna like the other estimators.
- Won't fix flips (see scope caveat) — measure it on planar-discrimination metrics, not flip rate alone.

### C.5 Effort & risk

Moderate–large (new descriptor module + prep integration + sweep). Higher risk. **Do after VSAC lands**, as an isolated A/B (augment mode first).

---

## Verification (both options)

1. **Correspondence quality (direct).** Instrument `match_correspondences_fpfh` to log `n_corr` and correspondence inlier ratio `w` against ground-truth pose on the synthetic scenes in `tests/test_se2.py` (`_make_scene`, `_make_corner_scene`). Expect A to raise `n_corr` and `w`; expect C to raise the precision of the top-`k` matches.
2. **Unit tests** in `tests/` mirroring `test_se2.py`: A — a scene with a decoy same-descriptor point at the wrong height, assert the correct same-height match is now formed; C — two patches related by a horizontal-axis rotation get *different* gravity-aware descriptors (whereas FPFH gives identical ones), and a yaw rotation leaves them *unchanged*.
3. **End-to-end A/B** on the real dataset via the existing harness (compare against `model:ransac3dof model.profile:acc-opt`), tracking accuracy, AR, p95 latency, and — as a guardrail, not a target — flip rate.
4. **Optuna** sweeps for the new knobs (`z_weight` for A; bin/radius/augment-mode for C), writing to `sweeps/optuna_*.db`, same as the other estimators.

## Sequencing

1. **Option A** — land near-term; it multiplies the inlier ratio the VSAC loop feeds on. Backward-compatible default (`z_weight=0`).
2. **VSAC SE(2)** — per `.claude/plans/we-re-going-to-implement-memoized-wreath.md`.
3. **Option C** — research follow-up, augment mode first, isolated A/B.
