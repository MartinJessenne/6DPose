# Roadmap: SE(2) Pose Estimation — From Offline Benchmark to Real-Time ROS 2 Proof of Concept

> **Status**: draft roadmap, written 2026-07-19, branch `3DoF`.
> **Scope**: everything between the current `Ransac3DoFEstimator` benchmark results and a first
> real-time proof of concept running against recorded rosbags on the Jetson Orin Nano 8GB.
> **Companion docs**: `AGENTS.md` (architecture), `COORDINATE_TRANSFORMS_MEMO.md` (frames),
> vault objectives `10.02`–`10.05` (targets).

---

## 0. Where we are, and how far the targets are

### 0.1 Current baseline (benchmark.py, RANSAC 3DoF, 1481 frames)

| Metric | Real-time config | Accuracy config | Target (Obj 2/5) |
|---|---|---|---|
| Success rate | 79.9 % (288 pose failures) | 99.3 % (0 pose failures) | — (implicit ~100 %) |
| BOP-style AR | 0.223 | 0.331 | — |
| Median XY translation error | 22.4 mm | 21.1 mm (non-flipped) | **< 10 mm** |
| Median yaw error | 0.61° | 0.34° (0.25° non-flipped) | **< 2°** ✅ |
| Mean yaw error | 33.4° | 28.9° | — |
| Flip rate | 16.9 % | 15.0 % | ~0 % |
| p95 latency | 0.72 s | 2.10 s | **≥ 5 Hz** on Orin Nano |

Hardware caveat: these latencies come from a server-grade CPU and an RTX PRO 6000. On the
Orin Nano 8GB (6× Cortex-A78AE @ ~1.5 GHz, 1024-core Ampere GPU, shared 8 GB LPDDR5) expect
roughly **3–10× slower** for the CPU-bound geometric backend and a similar factor for
un-optimized PyTorch inference. The realistic reading of "p95 = 0.72 s" is therefore
"several seconds per frame on target hardware".

### 0.2 The three diagnoses that shape this roadmap

1. **The estimator is precise but bimodal.** Median errors (2.2 cm / 0.3°) are close to target;
   mean errors are wrecked by the 15–17 % of frames that converge to the 180°-flipped pose.
   Fixing flips is the single highest-leverage accuracy work, and it fixes AR "for free".
2. **The benchmark measures the wrong regime for deployment.** Every frame is solved cold
   (global registration from scratch). In deployment, consecutive frames are ~33–200 ms apart
   and the cart barely moves between them. A **track-then-reinitialize** architecture makes the
   expensive global stage a rare event and turns the per-frame cost into a cheap local
   refinement. This — not micro-optimizing RANSAC — is what makes 5 Hz on the Orin plausible.
3. **The current dataset cannot validate deployment behavior.** Frames are independent
   (no temporal continuity), so tracking, temporal gating, and EKF smoothing are
   unbenchmarkable with what exists. A continuous-sequence dataset (rosbags with ground
   truth) is a prerequisite for the second half of the roadmap, not an afterthought.

### 0.3 Phase overview

| # | Phase | Depends on | Primary outcome | Rough effort |
|---|---|---|---|---|
| 1 | Flip disambiguation (free-space check + front-slab scoring) | — | Flip rate 15 % → < 3 %; mean yaw ≈ median yaw; AR roughly doubles | days |
| 2 | Failure & error-budget analysis | — (parallel with 1) | Understand the 288 RT failures and the 21 mm floor | days |
| 3 | BEV correlative matching (deterministic global stage) | 1 (for fair A/B) | Optional replacement of constrained RANSAC: deterministic, faster, flip-aware | 1–2 weeks |
| 4 | Sequence dataset: rosbag recording + ground truth | — (can start anytime; blocks 5, 7, 8) | Replayable real-sensor sequences with per-frame GT | days of setup + recording sessions |
| 5 | Tracking architecture (SE(2) ICP tracker + state machine) | 4 | Per-frame cost 10–50 ms; global stage only on init/loss | ~1 week |
| 6 | C++ port of the geometric backend (pybind11, parity-tested) | 1, 5 frozen | 10–50× on the hand-written loops; ROS 2-ready library | 1–2 weeks |
| 7 | ROS 2 node on Orin Nano + TensorRT YOLO + rosbag HIL + rviz | 4, 5, 6 | **The proof of concept**: ≥ 5 Hz pose stream on `/tf` from replayed bags | 1–2 weeks |
| 8 | Temporal filter (SE(2) EKF / gating) | 5, 7 | Smooth, dropout-tolerant pose stream; residual flips eliminated | ~1 week |

Phases 1–3 are pure Python work on the existing benchmark. Phase 4 is fieldwork + tooling.
Phases 5–8 are the deployment track. Phases 2 and 4 can run in parallel with everything.

---

## Phase 1 — Flip disambiguation

### Description

The carts are near-symmetric under a 180° yaw rotation, so for every frame two poses explain
the point cloud almost equally well. The pipeline already runs a **dual-hypothesis ICP**
(`refine_pose_dual_hypothesis` in `methods/base.py`, SE(2) variant in `methods/se2_icp.py`):
it refines both the raw registration pose and its 180°-flipped twin, then keeps the one with
the higher ICP fitness. The problem: fitness is computed over the **whole cart**, and the
whole cart is (nearly) symmetric — so both hypotheses score within noise of each other, and
in ~1 frame out of 6 the wrong one wins.

Two independent fixes, both benchmarkable on the existing dataset:

**(a) Front-slab hypothesis scoring.** Compare the two hypotheses only where they *disagree*:
the asymmetric front (towing) face. `front_crop_depth` already crops the model for
registration; reuse the same crop for the *selection* step.

**(b) Free-space (visibility) consistency check.** A depth pixel measuring distance *d* asserts
two facts: there is a surface at *d*, and the ray from the camera to *d* passes through empty
space. Registration fitness only uses the first fact. The flipped pose typically places cart
geometry (drawbar, front-face structure) inside the observed free space — physically
impossible. Project candidate model points into the depth image and count these violations;
prefer the hypothesis with fewer.

### Objectives

- Reduce flip rate from 15–17 % to **< 3 %** on the existing benchmark.
- No regression in non-flipped median errors or success rate.
- Added selection cost **< 10 ms** per frame (Python; it is a few matrix multiplies).

### Expected results

- Mean yaw error collapses toward the median (33° → low single digits).
- Mean XY error collapses toward the median (0.127 m → ~0.03 m).
- BOP AR roughly doubles (flipped frames currently score ~0 on every threshold).
- Downstream: temporal gating (Phase 8) only has to handle rare residual flips.

### Technical implementation

#### (a) Front-slab scoring

Seam: the hypothesis-selection block at the end of `refine_pose_dual_hypothesis`
(`methods/base.py:174-188`) and its SE(2) counterpart in
`refine_pose_dual_hypothesis_se2` (`methods/se2_icp.py`).

1. Precompute a **front-slab model point cloud** alongside the full model cloud during
   `prepare()` — `crop_front_face` (`methods/ransac3dof.py:15`) already produces the mesh;
   sample it the same way the full model is sampled, cache it under the existing
   `_PREPARATION_CACHE` key (extend `_get_prep_params_key` if the slab depth becomes a
   separate parameter).
2. After both ICPs converge, score each refined pose `T` by *slab fitness*: transform the
   slab points by `T`, count the fraction with a scene neighbor within
   `icp_max_correspondence_distance` (one `o3d.geometry.KDTreeFlann` query per point, or
   `scene_pcd.compute_point_cloud_distance`). Select on slab fitness; keep full-cloud
   fitness only as a tiebreaker.
3. Careful with degenerate views: when the front face is barely visible (cart seen from
   behind), the slab has few supporting scene points for *both* hypotheses. Guard with a
   minimum-support threshold (e.g. require ≥ 30 slab points with any neighbor); below it,
   fall back to combined scoring (slab + free-space check below).

#### (b) Free-space check

The check runs in the **camera frame** against the depth crop, so it needs data the estimator
currently does not receive: the `MaskedImageFrame` (crop-adjusted intrinsics + depth crop,
`pipeline.py:261`) and the camera→robot extrinsic (already on the estimator as
`self.extrinsic` via `RansacEstimator.__init__`). `estimate_pose(**kwargs)` was designed for
exactly this — pass `frame=frame` through from the benchmark/inspection call sites.

Algorithm, per hypothesis pose `T` (model→base_link):

```python
def free_space_violations(model_points, T, T_robot_camera, frame, margin=0.03):
    """Count model points the depth image says are floating in observed free space."""
    # 1. model -> base_link -> camera frame
    P_base = (T[:3, :3] @ model_points.T).T + T[:3, 3]
    T_cam_base = np.linalg.inv(T_robot_camera)
    P_cam = (T_cam_base[:3, :3] @ P_base.T).T + T_cam_base[:3, 3]

    z = P_cam[:, 2]
    valid = z > 0.05  # in front of the camera
    # 2. project with crop-shifted pinhole intrinsics (same math as get_o3d_intrinsics)
    u = np.round(P_cam[:, 0] / z * frame.camera.fx + (frame.camera.cx - frame.xmin)).astype(int)
    v = np.round(P_cam[:, 1] / z * frame.camera.fy + (frame.camera.cy - frame.ymin)).astype(int)
    inside = valid & (u >= 0) & (u < frame.width) & (v >= 0) & (v < frame.height)

    depth = frame.depth.numpy()
    d_measured = depth[v[inside], u[inside]]
    d_predicted = z[inside]
    observed = d_measured > 0  # 0 = masked-out / hole: no evidence, skip
    # 3. violation: pose claims solid cart >margin *in front of* a measured surface
    violations = observed & (d_predicted < d_measured - margin)
    return int(violations.sum()), int(observed.sum())
```

Notes that matter in this codebase:

- **Use the full-model sampled cloud** for the check, not the front slab — the violations of
  the flipped pose mostly come from full-body geometry protruding into free space.
- **Masked-out pixels carry no evidence.** `crop_and_mask_inputs` zeroes background depth;
  a zero pixel must be skipped, not treated as "free to infinity". Same for D455 depth holes.
- `margin` should sit above sensor noise at working range (D455: ~2 % of distance at 2 m
  → 3–4 cm is a safe start). Expose it as a `Ransac3DoFParams` field and let a short Optuna
  sweep place it (`suggest_params` already exists for exactly this).
- **Decision rule**: normalize to a violation *ratio* (violations / observed projections) and
  combine with slab fitness, e.g. select hypothesis 2 iff
  `slab_fitness_2 - slab_fitness_1 > τ_f` **or** `viol_ratio_1 - viol_ratio_2 > τ_v`.
  Start simple: pick the hypothesis with the lower violation ratio unless the ratios are
  within 2 % of each other, then fall back to slab fitness.
- **Benchmark instrumentation**: `benchmark.py` already computes a per-frame flip flag; log
  the four scores (fitness ×2, violation ratio ×2) per frame so mis-selections can be
  audited offline before touching thresholds.

### Validation

Full benchmark run with both configs (RT + accuracy). Acceptance: flip rate < 3 %, success
rate and non-flipped medians unchanged (±5 %), per-frame overhead < 10 ms. Keep an ablation:
(a) only, (b) only, (a)+(b).

---

## Phase 2 — Failure and error-budget analysis

### Description

Two open questions decide where later effort goes, and both are answerable with the current
benchmark and a few hours of instrumentation:

1. **Why does the real-time config lose 288 frames (20 %) that the accuracy config solves?**
2. **What is behind the 21 mm median XY floor** (target: < 10 mm) — estimator noise,
   voxel-size quantization, or a systematic bias (extrinsics / ground truth / mesh scale)?

### Objectives

- A per-failure categorization of the 288 pose failures (correlated with: cart distance,
  visible point count after masking, cart type, yaw relative to camera).
- A decomposition of the 21 mm median: random scatter vs. constant offset, per cart type.

### Expected results

- If failures correlate with low point counts / distance → adaptive RANSAC budget or a
  single retry-with-bigger-budget fallback recovers most of them cheaply. (With Phase 5's
  architecture, the global stage runs rarely, so it can simply *use the accuracy config* —
  possibly making this moot for deployment while still improving the benchmark.)
- If the XY error has a consistent direction per cart type or per viewpoint → it is a
  calibration/GT bias, and no amount of estimator work will fix it; fix the extrinsic or the
  GT export instead. If it is isotropic scatter → multi-scale ICP (final pass at fine voxel
  size on the already-converged pose) is the cheap win; a final ICP pass at half the current
  voxel size typically costs a few tens of ms and halves the residual.

### Technical implementation

- Extend `benchmark.py`'s per-frame record with: masked point count (pre/post downsample),
  distance to cart (from GT), signed XY error components **in the cart's own frame**
  (rotate the error vector by the GT yaw — a bias along the cart's viewing axis screams
  depth/extrinsic bias; a bias along +x of the cart screams mesh-origin mismatch),
  RANSAC iteration count at exit, and ICP fitness/RMSE of the winning hypothesis.
- Dump to parquet/CSV; analyze in a notebook. Plot error vs. distance, error-vector rose
  plots per cart type, failure histogram vs. point count.
- For the multi-scale ICP experiment: add an optional `fine_voxel_size` to
  `Ransac3DoFParams`; after `_refine_pose`, run one more `refine_pose_dual_hypothesis_se2`
  round (single hypothesis — the winner) with model/scene downsampled at the fine size and a
  correspondence distance ≈ 2× fine voxel.

---

## Phase 3 — BEV correlative matching (deterministic global stage)

*Optional but recommended experiment. It competes with `constrained_ransac_se2` for the
"global initialization" slot; whichever wins the A/B goes into the C++ port (Phase 6).*

### Description

The problem has exactly 3 DoF, so the global search can be run **exhaustively in 2D** instead
of sampled in 3D. Project the scene cloud (already in Z-up `base_link`) onto the ground
plane as a 2D grid; precompute the cart's top-down footprint template from the CAD mesh;
score the template over all (x, y, yaw) candidates via cross-correlation against a blurred
("likelihood field") version of the scene grid. Multi-resolution search (Olson 2009,
"Real-Time Correlative Scan Matching"; Cartographer's branch-and-bound variant,
`fast_correlative_scan_matcher_2d.cc`, is the canonical open-source reference) makes it fast
and *globally optimal within the window* — deterministic, seedless, no iteration budget to
tune.

### Objectives

- Match or beat constrained-RANSAC success rate and coarse-pose quality (final accuracy is
  identical by construction — the same SE(2) ICP refines both).
- Zero run-to-run variance (kills the `seed=0` benchmarking crutch and Optuna noise).
- Python prototype < 200 ms/frame; projected C++ < 20 ms on Orin-class CPU.

### Expected results

- A deterministic global stage whose failures are *explainable* (score map inspectable as an
  image — you can literally look at the correlation surface to see why a frame failed).
- Flip resistance built into the global stage: the height-map template includes the
  asymmetric front-face structure, so the flipped yaw scores strictly worse whenever the
  front face is visible.
- If it wins the A/B: simpler C++ port than the RANSAC (dense array ops, no KD-trees, no
  FPFH — note it removes the FPFH computation entirely from the deployment hot path).

### Technical implementation

New estimator `methods/bev_correlative.py`, subclassing `Ransac3DoFEstimator` and overriding
only `_global_registration` (the hook already isolates the global stage; FPFH inputs are
simply ignored — longer-term, add a flag to skip FPFH computation when this estimator is
active, since it is pure wasted latency).

**1. Scene grid (per frame).**

```python
res = 0.01  # 1 cm cells
pts = np.asarray(pcd_down.points)  # base_link, Z-up
pts = pts[pts[:, 2] > 0.02]  # drop floor points if any survive masking
# grid extents from the YOLO-derived cloud bbox + search margin
origin = pts[:, :2].min(axis=0) - margin
ij = ((pts[:, :2] - origin) / res).astype(int)
H = np.zeros(grid_shape, np.float32)
np.maximum.at(H, (ij[:, 0], ij[:, 1]), pts[:, 2])  # height map: max z per cell
occ = H > 0
# likelihood field: smooth score falloff with distance to nearest occupied cell
D = scipy.ndimage.distance_transform_edt(~occ) * res
L = np.exp(-((D / sigma) ** 2))  # sigma ≈ 0.02–0.03 m (D455 noise)
```

Prefer the **height map** variant for scoring when possible: score a template cell against
`exp(-(Δheight/σ_h)²)·L` so that tall posts must land on tall cells. Start with pure
occupancy `L` (simpler), add the height term only if front/back discrimination needs it.

**2. Model template (once per cart type, cached in `_PREPARATION_CACHE`).**
Sample the CAD mesh surface (the sampled cloud from `prepare()` works), project to the
model's own xy plane at the same resolution, store as a sparse list of occupied cell offsets
`(dx, dy, h)` relative to the CAD origin. Sparse-list scoring (sum of `L` lookups at
transformed template cells) beats dense 2D convolution for a template this size.

**3. Search.**

- Yaw: coarse pass every 5° over 360° (72 rotated templates — precompute the rotated
  sparse offset lists once per resolution), fine pass every 1° in ±5° around the coarse peak.
- (x, y): the scene grid is already crop-sized (YOLO bbox + margin ≈ ±0.5 m), so brute
  force at coarse resolution (4 cm) then refine at 1 cm around the peak is enough — the
  full Olson/Cartographer branch-and-bound machinery is an optimization to add in C++ only
  if profiling demands it. Score = mean of `L` over template cells that land in-grid.
- Vectorize in NumPy: for one yaw, scoring all offsets is a stack of shifted gathers —
  or use `scipy.signal.fftconvolve(L, template_mask[::-1, ::-1])` per yaw for the coarse pass.
- Output: best `(x, y, yaw)` → 4×4 SE(2) matrix with `z = self._active_z_offset` (same
  contract as `constrained_ransac_se2`); hand to the unchanged SE(2) ICP refinement. Also
  return the score *and* the runner-up peak — a small best/second-best margin is a
  usable ambiguity signal (feeds Phase 5's confidence gating).

**4. A/B against constrained RANSAC** on the same benchmark: success rate, AR, flip rate,
latency, plus a determinism check (two runs, identical outputs).

Pitfalls: the mask sometimes includes floor pixels around the wheels (hence the z-cut);
resolution below ~1 cm buys nothing (sensor noise dominates); make sure template and scene
use the same convention for "cell center vs corner" or you inject a half-cell bias.

---

## Phase 4 — Sequence dataset: rosbag recording and ground truth

*Blocks Phases 5, 7, 8. Start early — it involves physical setup iteration.*

### Description

Create the missing dataset class: **continuous RGB-D sequences from the real D455(i), with
per-frame or per-segment ground truth, stored as replayable ROS 2 bags.** These bags are both
the validation set for tracking/filtering and the hardware-in-the-loop stimulus for the
Orin Nano PoC (Obj 5, physical track).

### Objectives

- ≥ 10 sequences × 30–120 s covering: static cart (multiple distances 1–3 m and yaws,
  including front-hidden views), robot approaching the cart, lateral arcs, cart partially
  occluded, cart entering/leaving frame (tests re-initialization), lighting variations,
  and at least one worst-case (fast motion + specular highlights).
- Ground truth: **< 5 mm / < 0.5°** reference accuracy for static segments (this must be
  meaningfully better than the < 10 mm system target, or the evaluation proves nothing);
  continuous reference for at least a few dynamic sequences.
- Bags replay deterministically on both the dev machine and the Orin.

### Technical implementation

#### 4.1 Recording rig and driver

- ROS 2 (Humble or Jazzy — **pick the same distro the Orin will run**, see Phase 7) with
  `realsense2_camera` (`ros-$DISTRO-realsense2-camera` + `librealsense2`).
- Launch:

```bash
ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    enable_color:=true enable_depth:=true \
    rgb_camera.color_profile:=848x480x30 \
    depth_module.depth_profile:=848x480x30 \
    enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
    pointcloud.enable:=false
```

  - `align_depth.enable:=true` gives `/camera/camera/aligned_depth_to_color/image_raw` —
    depth registered to the color frame, which is what the pipeline assumes (one intrinsics
    set for both). Record the *aligned* depth.
  - `pointcloud.enable:=false`: never record point clouds; they are 10× the bandwidth and
    the pipeline rebuilds them from depth anyway.
  - Pick the profile once and keep it for every bag; changing resolution changes intrinsics.
  - IMU topics cost almost nothing and Obj 3 (EKF) will want them — record them now.

#### 4.2 What to record

```bash
ros2 bag record -s mcap \
    --max-cache-size 1073741824 \
    /camera/camera/color/image_raw \
    /camera/camera/color/camera_info \
    /camera/camera/aligned_depth_to_color/image_raw \
    /camera/camera/aligned_depth_to_color/camera_info \
    /camera/camera/imu \
    /tf /tf_static \
    -o bags/seq_<NN>_<description>
```

- **`-s mcap`**: use the MCAP storage plugin (`ros-$DISTRO-rosbag2-storage-mcap`) — faster,
  self-contained, plays nicely with Foxglove for visual inspection.
- **Bandwidth reality check**: 848×480 RGB8 + 16UC1 depth at 30 fps ≈ 60 MB/s raw. Record to
  an SSD, not an SD card. If that is too much, either record at 15 fps (still above the 5 Hz
  target with margin) or record compressed color (`image_transport` republish to
  `.../compressed`) — but keep **depth lossless** (PNG-compressed 16UC1 via
  `compressedDepth` is acceptable; lossy depth is not).
- **`camera_info` is non-negotiable**: it carries the intrinsics that replace the current
  hardcoded `config/camera` values at replay time.
- Record `/tf_static` so the camera's mount transform (once published, see 4.4) travels
  with the bag.
- Write a `metadata.yaml` sidecar per bag: cart type, GT method, measured GT values,
  lighting notes, anomalies. Discipline here is what makes the dataset usable in 3 months.

#### 4.3 Ground truth strategy

No motion-capture assumption. Three tiers, cheapest first — most sequences only need tier 1:

**Tier 1 — Static-cart segments (primary).** Cart parked, robot/camera static or moving.
Measure the cart pose *once* per segment with high confidence:
  - Print a large **AprilTag** (tag36h11, ≥ 15 cm — bigger is better; pose accuracy scales
    with apparent size) on rigid board, mount it flat on the cart at a jig-measured offset
    `T_cart_tag` (drill/locating pins on the mount so it is repeatable across sessions;
    measure the offset once with calipers + CAD, it becomes a constant).
  - At recording time, capture a few seconds where the tag is clearly visible; solve
    `T_camera_tag` with any AprilTag detector (e.g. `apriltag_ros` live, or offline on the
    bag), average over ~100 frames → `T_base_cart = T_base_camera · T_camera_tag · T_tag_cart⁻¹`.
  - **Remove or occlude the tag for the actual evaluation segment** if you worry the
    detector-facing geometry biases the pipeline (it should not — the tag is invisible to
    depth — but the RGB mask from YOLO could react to it; do one A/B early to check).
  - For static segments the GT is a constant; every frame in the segment is labeled.
**Tier 2 — Dynamic sequences with tag-in-frame.** Keep the tag visible during motion and run
  the AprilTag detector on the recorded RGB offline → continuous (noisier, ~1 cm / 1°) GT
  trajectory. Good enough to validate tracking continuity and EKF behavior, not for the
  final <10 mm claim (that claim rests on tier 1).
**Tier 3 — Self-consistency (no GT).** For sequences with neither: evaluate *precision*
  (pose jitter over static camera+cart) and *closure* (pose at end of an out-and-back camera
  motion matches pose at start). Catches drift and instability without any reference.

Store GT per sequence in the sidecar (tier 1) or as a CSV of timestamped poses (tier 2),
in `base_link`, same 4×4 convention as `compute_ground_truth_pose` output.

#### 4.4 Extrinsics at record time

The SE(2) constraint lives or dies on `T_robot_camera` being right (Z truly up). Before the
first real session:

- Mount the camera rigidly; publish the mount transform as a static TF
  (`ros2 run tf2_ros static_transform_publisher ... base_link camera_link`).
- **Verify the ground plane**: back-project a depth frame of empty floor, fit a plane
  (Open3D `segment_plane`), check its normal in `base_link` is `[0,0,1]` within ~0.3° and
  its offset is 0 within a few mm. If not, correct the extrinsic *now* — a 1° pitch error
  puts a distance-proportional bias on XY (17 mm/m of range: at 2 m that alone can exceed
  the whole 10 mm budget, and it may already explain part of Phase 2's findings).
- Re-run this check at the start of every recording session (mounts drift; 30 seconds of
  floor footage per session is cheap insurance and can be validated offline from the bag).

#### 4.5 Replay and offline consumption

Two consumption modes, both required:

- **HIL replay** (Phase 7): `ros2 bag play bags/seq_03 --clock`, all nodes with
  `use_sim_time:=true`. `--rate 2.0` for stress tests, `--rate 0.5` to isolate accuracy
  from latency. Replay is the *stimulus*; the pipeline must keep up or drop frames exactly
  as it would live.
- **Offline evaluation** (Phase 5): a `rosbag2_py`-based reader that iterates
  synchronized (color, aligned-depth, camera_info) triplets and feeds them through the
  existing Python pipeline **without ROS running** — i.e. a `load_rosbag_dataset()` sibling
  to `load_parquet_dataset()` in `pipeline.py`, yielding the same fields the benchmark
  expects plus a timestamp. Message pairing: exact-match `header.stamp` on color vs.
  aligned depth (the realsense driver stamps them identically when aligned); tolerate
  ±1 ms otherwise. This is what makes tracking benchmarkable in the existing harness —
  extend `benchmark.py` with a `dataset=rosbag` mode and sequence-aware metrics
  (per-sequence AR, jitter, time-to-first-lock, recovery-after-occlusion).

---

## Phase 5 — Tracking architecture

### Description

Restructure the pipeline from "global registration every frame" to a **tracker with a
re-initialization fallback**:

```
                    ┌────────────────────────────────────────────┐
                    │                  LOST                      │
                    │  run global stage (RANSAC-3DoF or BEV)     │
                    │  + dual-hypothesis ICP + flip checks       │
                    └────────────┬───────────────────────────────┘
                    success      │      ▲ fitness < τ_lost for N frames,
                    + confidence │      │ or YOLO lost the cart,
                                 ▼      │ or innovation gate rejects
                    ┌────────────────────────────────────────────┐
                    │                 TRACKING                   │
                    │  seed = previous pose (+ const-velocity    │
                    │  extrapolation); SE(2) ICP only            │
                    │  single hypothesis + free-space sanity     │
                    └────────────────────────────────────────────┘
```

In TRACKING, the per-frame geometric cost is **one SE(2) ICP from a near-correct seed**
(a handful of Gauss-Newton iterations) — no FPFH, no RANSAC, no dual hypothesis. This is
the step that changes the latency class of the whole system.

### Objectives

- Per-frame geometric cost in TRACKING: **< 50 ms in Python** on the dev machine
  (the Phase 6 C++ port then takes it well under 10 ms on the Orin).
- Track continuity on Phase 4 dynamic sequences: no flip ever *during* a track (flips can
  only enter at initialization, which Phases 1/3 defend).
- Re-initialization: time-to-relock after full occlusion < 2 s (a few global-stage frames).

### Expected results

- Deployment latency budget becomes: YOLO (GPU) + cloud prep + one small ICP ≈ real-time.
  The global stage runs only at startup and after track loss, so it can afford the
  *accuracy* config (99.3 % success) — the RT-config failure problem dissolves.
- Sequence-level metrics (jitter, availability, relock time) exist for the first time.

### Technical implementation

- New class `Se2Tracker` (e.g. `methods/tracker.py`) *composing* (not subclassing) a global
  estimator and the SE(2) ICP:
  - State: `T_last`, `v_last` (finite-difference SE(2) velocity via `se2_lie_utils` log/exp),
    `fitness_history`, `mode`.
  - `update(pcd, cad_mesh, cart_type, frame, dt)`:
    - TRACKING: seed `T_pred = T_last · exp(dt · v_last)` (constant-velocity in the Lie
      algebra; with dt ≈ 0.1–0.2 s this keeps the seed within ICP's basin even at cart-
      pushing speeds). Run single-hypothesis `refine_pose_dual_hypothesis_se2`'s inner ICP
      (factor the single-ICP path out of the dual wrapper) with a *tighter* correspondence
      distance than cold-start (e.g. 0.05 m) — tight correspondences reject clutter that the
      cold-start setting must tolerate.
    - Health checks per frame: ICP fitness > τ (calibrate on sequences: τ ≈ 0.6–0.7 of
      typical locked fitness), free-space violation ratio below threshold (Phase 1 code,
      reused — it is the cheap guard against latching onto a wrong local minimum), and
      pose increment within physical limits (`|Δv|` gate).
    - On failure → LOST; on K consecutive failures also drop `v_last`.
  - LOST: run the full existing `Ransac3DoFEstimator.estimate_pose` (accuracy config) +
    Phase 1 flip checks; require *confidence* to enter TRACKING (fitness above τ_init and,
    if BEV is in play, a clear best/second-best score margin).
- **Downsampling asymmetry**: in TRACKING the scene cloud can be downsampled more
  aggressively (ICP from a good seed needs far fewer points than FPFH matching does) —
  a separate `tracking_voxel_size` knob, worth its own micro-sweep.
- **Evaluation** (needs Phase 4 bags): extend `benchmark.py` with sequence mode — feed
  frames in timestamp order, report per-sequence: AR, median errors, availability
  (% frames with a pose), flips-per-track, mean relock time, per-mode latency split.
- **Keep the cold-start benchmark** as a separate CI-style check — it validates the LOST
  path and remains the regression test for Phases 1–3.

---

## Phase 6 — C++ port of the geometric backend (pybind11 first, ROS 2 second)

### Description

Port the *hot, hand-written* geometry to C++ — but wrap it with **pybind11 and validate it
inside the existing Python benchmark before any ROS 2 code exists**. The Python harness
(dataset, YOLO, metrics, sweeps) is an asset to keep, not legacy to replace.

Port scope (the code where interpreter overhead actually lives):
- `constrained_ransac_se2` (`methods/constrained_ransac.py`) — per-iteration Python loop; **or** the BEV matcher if Phase 3 wins the A/B (dense array ops — an easier and faster port).
- `refine_pose_dual_hypothesis_se2` + Gauss-Newton inner loop (`methods/se2_icp.py`).
- `se2_lie_utils` (trivial, header-only).
- The Phase 1 flip checks and the Phase 5 tracker logic (small, but they belong with the backend).

**Not** ported: YOLO (TensorRT handles it, Phase 7), cloud back-projection / normals /
downsampling / FPFH (already C++ inside Open3D — Python only orchestrates).

### Objectives

- **Bit-comparable parity** with Python on the full benchmark: same seeds → same poses
  (tolerance 1e-9 on deterministic paths; identical RANSAC sampling given the same RNG —
  simplest is to port the sampling to draw from an identical PCG64 sequence, or inject the
  sample indices during parity tests).
- ≥ 10× on `se2_icp`, ≥ 10× on the global stage, measured on x86 *and* on the Orin.
- One CMake project producing: a plain C++ static/shared lib (`libcartpose`), the pybind11
  module (`cartpose_py`), and (Phase 7) the ROS 2 node linking `libcartpose`.

### Expected results

- The same `benchmark.py` runs with `backend=cpp` as a config switch — every future
  algorithm change is validated once and deployed twice (Python for research, C++ for the
  robot) without divergence.
- TRACKING-mode geometric cost on the Orin drops to low single-digit ms.

### Technical implementation

- Layout: `cpp/` with `CMakeLists.txt`, `src/`, `include/cartpose/`, `bindings/`,
  `tests/` (Catch2/GTest unit tests mirroring `tests/`).
- Dependencies: **Eigen** (all pose/Jacobian math) + **nanoflann** (header-only KD-tree for
  ICP correspondences and slab fitness). Do *not* link C++ Open3D just for ICP — its
  dependency footprint on Jetson is pain you do not need; your SE(2) ICP is already
  self-written math that maps 1:1 to Eigen.
- pybind11 interface mirrors the Python function signatures (`Eigen::MatrixXd` ⇄ NumPy is
  zero-copy with `py::EigenDRef`); estimator classes in `methods/` grow a
  `backend="python"|"cpp"` switch that swaps the `_global_registration`/`_refine_pose`
  internals — nothing above that layer changes.
- Parity harness: a pytest that runs N benchmark frames through both backends and asserts
  pose agreement; keep it in CI (it is the contract that lets you keep improving the Python
  side later).
- Build for both x86 (dev) and aarch64 (Orin). Nothing here is exotic — plain CMake +
  `-O3 -march=native` (x86) / `-mcpu=native` (Orin); OpenMP `parallel for` over RANSAC
  iterations / BEV yaw candidates when profiling justifies it.

---

## Phase 7 — ROS 2 node on the Orin Nano + rosbag HIL: the proof of concept

### Description

Assemble the deployment stack on the Jetson Orin Nano 8GB and drive it with Phase 4 bags
replayed at true rate. Success looks like: **rviz showing the cart mesh glued to the point
cloud at ≥ 5 Hz, from a bag the Orin has never seen, with latency and error numbers logged.**

### Objectives (= Obj 4/5 completion conditions, PoC subset)

- Sustained pose output ≥ 5 Hz on the Orin under bag replay at 15–30 fps input.
- End-to-end latency (image stamp → `/tf` publish) < 200 ms, p95.
- Pose stream correct on tier-1 GT segments: median XY < 15 mm at this stage
  (< 10 mm is the post-EKF, post-Phase 2-calibration target).
- No memory growth over a 10-minute replay (8 GB is shared CPU+GPU — watch it).

### Technical implementation

#### 7.1 Platform

- JetPack 6.x (Ubuntu 22.04). ROS 2 **Humble** matches 22.04 natively.
- **YOLO via TensorRT**: `yolo export model=pt_model.pth format=engine half=True device=0`
  (Ultralytics wraps trtexec; run the export **on the Orin** — TensorRT engines are not
  portable across GPU architectures). FP16 first; INT8 later if segmentation quality holds
  (needs a calibration set — a few hundred frames from the bags). Expected: nano-seg at
  848×480 in the 10–25 ms range on Orin.
- Power/thermals: `sudo nvpmodel -m 0` (MAXN) + `jetson_clocks` for benchmarking; record
  which mode every number was measured in. Add a fan; the Orin Nano throttles hard without one.

#### 7.2 Node architecture (keep it boring)

One perception node (start Python `rclpy` — the heavy code is already native via TensorRT +
`libcartpose`; move the node shell to `rclcpp` only if the glue itself shows up in profiles):

- Subscribes: color + aligned depth + camera_info (`message_filters` exact-time sync).
- Pipeline per frame: TensorRT YOLO → `crop_and_mask_inputs` → back-projection →
  `Se2Tracker.update()` (C++ backend) → publish.
- Publishes:
  - `/tf`: `base_link → cart_<type>` (the deliverable),
  - `geometry_msgs/PoseWithCovarianceStamped` on `/cart_pose` (the EKF in Phase 8 wants
    covariance; approximate it from ICP fitness/RMSE for now),
  - a small diagnostics message: mode (TRACKING/LOST), fitness, violation ratio,
    per-stage latencies.
- QoS: `SENSOR_DATA` (best-effort, keep-last-1) on image subs; reliable, keep-last-1 on pose.
- **Frame-dropping policy, explicit**: process latest-only (queue depth 1, skip stale
  frames). A real-time pose stream that silently falls behind is worse than one that drops.

#### 7.3 The HIL loop

```bash
# terminal 1 (Orin or dev machine on same LAN):
ros2 bag play bags/seq_07_approach --clock --loop
# terminal 2 (Orin):
ros2 launch cartpose perception.launch.py use_sim_time:=true
# terminal 3 (dev machine):
rviz2   # PointCloud2 (debug topic, decimated), TF, cart mesh Marker at /cart_pose
```

- Publish a `visualization_msgs/Marker` (`mesh_resource` → the cart PLY) at the estimated
  pose: the "mesh glued to the cloud" view is the single most informative debug artifact.
- Measure: per-stage wall times (YOLO / prep / tracker / publish) via the diagnostics topic;
  end-to-end latency = `now() - image.header.stamp` at publish (valid under `--clock`);
  `tegrastats` logging in parallel for RAM/GPU/thermals.
- Evaluate against GT with the Phase 4/5 sequence metrics — same code, running on the
  recorded `/tf` output (`ros2 bag record /tf /cart_pose /diagnostics` during replay, then
  offline comparison).
- Stress: `--rate 2.0`, two bags back-to-back, 10-minute loop for the memory check.

#### 7.4 Order of bring-up (each step has a visible success state)

1. Bag replays on Orin, rviz shows raw cloud (no pipeline). Confirms transport + sim time.
2. TensorRT YOLO node alone: masks at ≥ 15 Hz. Confirms front-end budget.
3. Full pipeline, LOST-mode only (global stage every frame): correct but slow — this is
   your baseline number, and it validates the C++ parity on-target.
4. Enable tracker → watch rate jump to input-limited. **This is the PoC moment.**
5. Occlusion/relock tests (hand in front of camera during replayed sequences via bag
   editing, or use the recorded occlusion sequences).

---

## Phase 8 — Temporal filtering (SE(2) EKF) and gating

### Description

Obj 3, drastically simplified by the SE(2) reduction: the state is `(x, y, θ, vx, vy, ω)` —
a small, well-conditioned filter rather than a full SE(3)+IMU estimator.

### Objectives

- Smooth pose stream; jitter (std on static segments) reduced ≥ 3×.
- Extrapolation through ≥ 200 ms dropouts with < 50 mm drift (Obj 3 condition).
- Innovation gating: 180°-flip measurements and outlier jumps rejected — residual flip
  rate in the *output stream* ≈ 0.

### Technical implementation

- Try `robot_localization` (`ekf_node`) first: constant-velocity model, fuse `/cart_pose`
  as pose measurement; 2D mode (`two_d_mode: true`) matches SE(2) exactly. Caveat: it
  estimates the *cart's* motion in `base_link` while its motion model assumes a mostly
  smooth trajectory — true for a parked/towed cart, and it saves writing Jacobians. If its
  configuration fights back, a hand-rolled 6-state EKF is ~150 lines and you already have
  the covariance inputs.
- **Flip gate lives outside the EKF** (pre-filter): if the measurement's yaw innovation
  vs. the predicted state exceeds ~90°, un-flip it (subtract 180°) and re-test; if it then
  passes, feed the corrected measurement — the estimator's flip becomes a recoverable
  event, not a rejection. Log every gate activation; the rate should be ≈ the residual
  Phase 1 flip rate.
- Camera IMU fusion (D455i) is *not* needed for the PoC — the cart's motion, not the
  camera's, is the state. It becomes relevant when the robot itself moves fast (ego-motion
  compensation between image stamp and publish time); defer until the PoC shows the need.

---

## Appendix A — Acceptance summary (definition of "PoC done")

| Check | Threshold | Where measured |
|---|---|---|
| Flip rate, single-frame (cold start) | < 3 % | Phase 1, existing benchmark |
| Cold-start success rate (accuracy cfg) | ≥ 99 % | existing benchmark |
| Sequence availability (TRACKING) | ≥ 95 % of frames | Phase 5, bag benchmark |
| Flips during a locked track | 0 | Phase 5, bag benchmark |
| Relock after occlusion | < 2 s | Phase 5/7 |
| Pose rate on Orin, bag replay | ≥ 5 Hz sustained | Phase 7, diagnostics topic |
| End-to-end latency p95 | < 200 ms | Phase 7 |
| Median XY error, tier-1 GT segments | < 15 mm (PoC) → < 10 mm (post-EKF + Phase 2 calibration) | Phase 7/8 |
| Output-stream flip rate (post-gate) | ≈ 0 | Phase 8 |
| 10-min replay memory growth | ~0 | Phase 7, tegrastats |

## Appendix B — Risk register (short)

- **Extrinsic/GT bias explains the 21 mm floor** → Phase 2 finds it early; if so, the fix is
  calibration, and Phase 4.4's plane check prevents re-poisoning the new dataset. *Do Phase 2
  before trusting any accuracy conclusion from the bags.*
- **YOLO trained on synthetic/parquet imagery underperforms on real D455 RGB** → surfaces the
  moment Phase 4 bags exist; budget for a fine-tune on a few hundred labeled real frames
  (mask labeling is fast with SAM-assisted tooling). This is the most likely "surprise".
- **Orin memory pressure (8 GB shared)** → TensorRT engine + ROS + Open3D fits, but avoid
  loading all cart meshes' prep caches simultaneously; make the cache LRU or per-deployment.
- **BEV experiment inconclusive** → no harm: constrained RANSAC is already adequate as the
  rare re-init stage once Phase 5 lands; BEV is an upside bet, not a dependency.
- **Front face never visible in some approach geometry** → flip disambiguation degrades to
  free-space + temporal evidence only; verify Phase 1 metrics *per viewing angle* (Phase 2
  instrumentation gives the yaw-relative-to-camera breakdown) so this blind spot is
  quantified, not discovered in the field.
