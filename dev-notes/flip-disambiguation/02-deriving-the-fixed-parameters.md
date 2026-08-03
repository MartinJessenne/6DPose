# 02 — Deriving the fixed parameters

`status: draft`

**What changes.** Six of the nine swept parameters stop being search dimensions and
become constants or formulas derived from the dataset-generation geometry, the D455
datasheet, and the CAD. The search space drops from **9 free parameters to 3**.

**Why.** Two reasons, and the second is the important one.

*Statistically*, TPE needs roughly exponential trials in effective dimension; at nine
dimensions 100 trials is sparse enough that the argmax is substantially a lucky draw,
consistent with the +0.021 best-of-N inflation already measured on this project.
`n_seeds=3` controls seed luck and does nothing about sampling luck in parameter space.

*Structurally*, a parameter fitted to three cart types and one camera is a
generalization liability the moment a fourth cart or a new sensor arrives. Each of the
six below has a **physical or geometric determination** that Optuna has been
re-discovering approximately, at cost, from data.

Apply [01](01-harness-fixes-before-the-confirmation-batch.md) first — it protects the
runs that verify this note.

> **Parity with previous runs is deliberately broken.** Every `tuned*` profile changes
> meaning, and studies before and after are not comparable. That is the owner's explicit
> call and the right one: the current values are fits, not measurements.

---

## 1. Vocabulary

| term | meaning here |
| :--- | :--- |
| **image-plane depth** | the depth a camera actually reports: the component of the point's position **along the optical axis**, not the Euclidean range. Isaac's `distance_to_image_plane` annotator, and what `depth_trunc` compares against. |
| **disparity** | the pixel shift of a feature between the two stereo images. Depth is inversely proportional to it, which is why depth noise grows as $z^2$. |
| **subpixel disparity noise** ($\sigma_d$) | the std of the stereo matcher's disparity estimate, in pixels. The single number that sets a stereo camera's depth precision. |
| **slab / front slab** | what `crop_front_face` keeps: the part of the CAD within `front_crop_depth` of the towing face. |
| **aspect ratio** (of a slab) | width ÷ depth. At 1.0 the slab is square in plan and has a genuine 90° rotational twin. |
| **cart frame** | CAD frame: origin at the towing-face centre on the floor, $+x$ toward the towing face, body extending toward $-x$. |

---

## 2. `depth_trunc` — from the generation script

### The geometry

`/home/martin/yolo_dataset_creation/generate_dataset_6_0_1.py:886-900` places the camera:

```python
d     = rng.uniform(0.8, 3.0)     # horizontal distance to the cart's front-centre
alpha = rng.uniform(-45.0, 45.0)  # bearing, degrees
target_z = CAMERA_HEIGHT + d * tan(TILT_ANGLE)
```

with `CAMERA_HEIGHT = 0.304` m and `TILT_ANGLE = 30°`. In the cart frame the camera eye
sits at $e = (d\cos\alpha,\; d\sin\alpha,\; h)$ with $h = 0.304$, and looks at
$t = (0,\, 0,\, h + d\tan\theta)$ with $\theta = 30°$.

### The optical axis, in closed form

$$t - e = (-d\cos\alpha,\; -d\sin\alpha,\; d\tan\theta)$$

Its length is $d\sqrt{\cos^2\alpha + \sin^2\alpha + \tan^2\theta} = d\sqrt{1+\tan^2\theta} = d/\cos\theta$.
The $d$ and the $\alpha$ both cancel, leaving a unit axis that depends only on the tilt:

$$\hat a = \cos\theta\,(-\cos\alpha,\; -\sin\alpha,\; \tan\theta) = (-\cos\theta\cos\alpha,\; -\cos\theta\sin\alpha,\; \sin\theta)$$

### Depth of an arbitrary cart point

For a cart point $p$, the image-plane depth is $z(p) = (p - e)\cdot\hat a$. Expanding and
using $\cos^2\alpha + \sin^2\alpha = 1$:

$$z(p) = d\cos\theta \;-\; \cos\theta\,(p_x\cos\alpha + p_y\sin\alpha) \;+\; (p_z - h)\sin\theta$$

The first term is the camera's own standoff; the second is how much further back into the
scene the point sits; the third is the tilt's contribution from height.

### The worst case

The cart occupies $x\in[-L_x, 0]$, $y\in[-L_y/2, L_y/2]$, $z\in[0, L_z]$ in its own frame
(origin at the towing face, body toward $-x$). Maximising:

* $p_x = -L_x$ — the rear — contributes $+\cos\theta\,L_x\cos\alpha$
* $p_y = -\operatorname{sign}(\sin\alpha)\,L_y/2$ contributes $+\cos\theta\,(L_y/2)|\sin\alpha|$
* $p_z = L_z$ contributes $(L_z - h)\sin\theta$

Since $L_x \gg L_y/2$ for every cart, the $\cos\alpha$ term dominates and the maximum is at
$\alpha = 0$ — the head-on approach:

$$\boxed{\;z_{\max} = \cos\theta\,(d_{\max} + L_x) \;+\; (L_z - h)\sin\theta\;}$$

Measured extents (from `meshes/*.ply`, axis-aligned bounding boxes):

| cart | $L_x$ | $L_y$ | $L_z$ | $z_{\max}$ at $d=3.0$ |
| :--- | ---: | ---: | ---: | ---: |
| colruyt | 2.575 | 0.837 | 0.757 | **5.06 m** |
| picanol | 1.522 | 0.795 | 2.018 | 4.77 m |
| leanflow | 0.972 | 0.704 | 0.617 | 3.60 m |

The $\pm10°$ cart-yaw jitter (`generate_dataset:883`) widens the effective bearing range
but does not move the maximum, which is attained at effective $\alpha = 0$ either way.

### What the shipped value does

`tuned` carries `depth_trunc = 4.6`. Solving $z_{\max}(d) > 4.6$ for colruyt:

$$0.866\,(d + 2.575) + 0.227 > 4.6 \;\Longrightarrow\; d > 2.47\ \text{m}$$

and $d\sim U(0.8, 3.0)$, so **the rear of every colruyt is silently truncated on ~24% of
its frames.** That is a range-dependent change in scene-cloud content that the model knows
nothing about — the worst kind of hidden parameter, because it makes far frames a
different problem from near frames.

### The edit

Set **`depth_trunc = 5.5`** in every VSACSe2 profile (`cli_config.py`), and **remove
`depth_trunc` from the sweep** at `benchmark.py:868`
(`trial.suggest_float("depth_trunc", 2.0, 7.0, step=0.1)`).

Note the asymmetry, because it is easy to get wrong: on the plain path `depth_trunc` comes
from `VSACSe2Profile.depth_trunc`, a field on the *profile*, not on `params`. On the sweep
path it comes from that `suggest_float` in `benchmark.py`. **Both need changing**, and
neither is reachable via `--param-overrides`, which only validates against the params
dataclass.

In `benchmark.py`, inside `run_parameter_sweep`'s objective, replace the suggestion with a
constant:

```python
            # 1. Suggest global parameters
            # depth_trunc is NOT suggested: it is workcell geometry, not a tunable.
            # The generation script places the camera at d ~ U(0.8, 3.0) m from the
            # cart's front face with a 30 deg upward tilt, so the farthest point of the
            # longest cart (colruyt, 2.575 m) sits at cos(30)*(3.0 + 2.575) + (0.757 -
            # 0.304)*sin(30) = 5.06 m of image-plane depth. 5.5 clears that with margin
            # and stays inside the D455's 0.6-6 m ideal range.
            depth_trunc = DEPTH_TRUNC_M
```

with a module-level constant next to the other tuning constants:

```python
# Image-plane depth beyond which returns are discarded. Derived in
# dev-notes/flip-disambiguation/02-deriving-the-fixed-parameters.md §2 from the
# dataset-generation envelope; NOT a tuned value. Lowering it below 5.06 silently
# truncates the rear of the longest cart on far frames.
DEPTH_TRUNC_M = 5.5
```

and in `cli_config.py`, `depth_trunc=4.6` becomes `depth_trunc=5.5` in each of the four
VSACSe2 profiles (`tuned`, `tuned-cheap`, `tuned-vis`, `tuned-vis-gate`). Leave `default`
and `bare` alone — they are ablation baselines whose job is to reproduce historical numbers.

5.5 clears the 5.06 worst case with margin and sits inside the D455's 0.6–6 m ideal range.
The point is not that 5.5 is optimal — it is that `depth_trunc` should bound the *working
volume* and never shape the object. At 5.5 it becomes inert for the cart, and its only
remaining job is discarding warehouse background that survives mask dilation.

---

## 3. `voxel_size` — from the D455 datasheet

### The stereo noise model

Depth from stereo comes from disparity $D$ via $z = fB/D$, where $f$ is the focal length
in pixels and $B$ the baseline. Differentiating, $\left|\frac{\partial z}{\partial D}\right| = \frac{fB}{D^2} = \frac{z^2}{fB}$, so:

$$\sigma_z = \frac{z^2}{fB}\,\sigma_d$$

The $z^2$ is the whole story of stereo depth: precision degrades quadratically with range.

For your configuration, both numbers are pinned:

* $B = 0.095$ m — the D455's baseline, [confirmed by Intel](https://www.intelrealsense.com/depth-camera-d455/)
* $f = 639.99768$ px — `generate_dataset_6_0_1.py:132`, and the same value in `CameraConfig.fx`

$$fB = 639.99768 \times 0.095 = 60.80\ \text{px·m}$$

### Pinning $\sigma_d$

Intel specifies **depth error < 2% at 4 m**, i.e. $\sigma_z(4) < 0.08$ m. Inverting:

$$\sigma_d = \frac{\sigma_z\,fB}{z^2} = \frac{0.08 \times 60.80}{16} = 0.304\ \text{px}$$

That is the *conservative* figure — the datasheet number is an accuracy budget including
calibration bias, not pure matching noise. A well-textured scene with good subpixel
interpolation achieves $\sigma_d \approx 0.1$ px. So bracket $\sigma_d \in [0.1, 0.3]$ px
and carry both through.

### Evaluating over the working range

The front slab — the only geometry that survives cropping — sits at image-plane depth
$z \approx d\cos\theta$, so $z \in [0.69, 2.60]$ m for $d \in [0.8, 3.0]$, median $\approx 1.65$ m.

| $z$ | $\sigma_z$ at $\sigma_d{=}0.1$ | $\sigma_z$ at $\sigma_d{=}0.3$ |
| ---: | ---: | ---: |
| 0.69 m (nearest) | 0.08 cm | 0.24 cm |
| 1.65 m (median) | 0.45 cm | 1.34 cm |
| 2.60 m (farthest) | 1.11 cm | 3.34 cm |

### The bracket

**From below — sensor noise.** A voxel smaller than $\sigma_z$ stops averaging noise and
starts preserving it. Over the far half of the range $\sigma_z$ is 1.1–3.3 cm, so anything
below 0.02 m is sub-noise there. This is why the sweep pinned at the 0.02 boundary and why
going lower would not have helped: the boundary was physics, not truncation.

**From above — two geometric constraints.** The slab must survive quantization: at 25%
crop (§4) leanflow's slab is 0.243 m deep, only ~12 voxels at 0.02 m. And
`min_sample_distance = 3.0 * voxel_size` (`vsac_se2.py:566`) becomes 0.06 m at 0.02 —
already a quarter of leanflow's slab depth. At voxel 0.04 it would be 0.12 m, half the
slab, and the sampler would have almost no admissible pairs.

Sensor noise from below, slab geometry from above, and the sweep's argmax in between —
three independent lines converging.

### The edit

**Fix `voxel_size = 0.02`** as the `VSACSe2Params` default, and **delete it from
`RansacEstimator.suggest_params`**. Not pinned by override — removed from the search space,
so nothing suggests it and nothing wastes a TPE dimension.

In `methods/ransac.py`, the `if "voxel_size" not in fixed:` block (which note 01 §0.1b has
you write) comes straight back out, replaced by the record of why:

```python
        # voxel_size is NOT suggested: it is set by sensor noise, not by search.
        # Stereo depth noise is sigma_z = z^2 * sigma_d / (f*B); for the D455,
        # f*B = 639.99768 px * 0.095 m = 60.80, and sigma_d is 0.1-0.3 px (the upper
        # end being Intel's "<2% at 4 m" spec, which includes calibration bias). Over
        # the front slab's working range (0.69-2.60 m image-plane depth) that is
        # 0.08-3.3 cm, so a voxel below 0.02 m is sub-noise over the far half of the
        # range: it preserves noise instead of averaging it, and FPFH gets less
        # repeatable. From above, min_sample_distance = 3*voxel is already a quarter of
        # leanflow's 0.243 m slab at 0.02. Bracketed on both sides -- see
        # dev-notes/flip-disambiguation/02-deriving-the-fixed-parameters.md §3.
```

and in `RansacParams` (or `VSACSe2Params`, wherever the field is declared), the default
becomes `voxel_size: float = 0.02` with a one-line pointer to that comment.

The next person to widen this should have to argue against $\sigma_z$, not against a number.

---

## 4. `front_crop_depth` → `front_crop_fraction`

### The defect

`front_crop_depth` is in **metres**, applied to a fleet whose body lengths span 0.972 m to
2.575 m — a factor of 2.6. One scalar therefore means something different on every cart.

Note 30.06's Trap 1 diagnosed this as a leanflow problem: its slab at 0.735 m is
0.735 × 0.704, aspect 0.96, square in plan, with a genuine 90° twin that a tight ICP stage
falls into. The yaw guard exists to block it.

**Measuring all three carts shows it is not a leanflow problem.** With the shipped
`front_crop_depth = 0.735`:

| cart | $L_x$ | $L_y$ | slab (depth × width) | aspect $L_y/\text{depth}$ |
| :--- | ---: | ---: | :--- | ---: |
| colruyt | 2.575 | 0.837 | 0.735 × 0.837 | **1.14** |
| leanflow | 0.972 | 0.704 | 0.735 × 0.704 | **0.96** |
| picanol | 1.522 | 0.795 | 0.735 × 0.795 | **1.08** |

**All three slabs are within 15% of square.** The 90° ambiguity the yaw guard defends
against is fleet-wide, and the guard has been masking it rather than removing it.

### The formula

With $\text{depth} = f \cdot L_x$, the aspect ratio becomes

$$R_{\text{cart}}(f) = \frac{L_y}{f\,L_x}$$

so each cart contributes a constant $L_y/L_x$ divided by $f$:

| cart | $L_y/L_x$ | $R$ at $f{=}0.25$ | $R$ at $f{=}0.35$ |
| :--- | ---: | ---: | ---: |
| colruyt | 0.325 | 1.30 | **0.93** |
| picanol | 0.522 | 2.09 | 1.49 |
| leanflow | 0.724 | 2.90 | 2.07 |

**colruyt is the binding constraint at every $f$** — it has the longest body and so the
squarest slab. At $f = 0.35$ colruyt returns to 0.93, square again. So $f = 0.25$ sits near
the top of the safe range, not in the middle of it: there is headroom downward, none upward.

Resulting slab depths: colruyt 0.644 m, picanol 0.380 m, leanflow 0.243 m.

### The edit

Replace the field. In `Ransac3DoFParams`:

```python
front_crop_fraction: float | None = 0.25
```

`crop_front_face(mesh, depth, min_height)` stays exactly as it is — it is a correct pure
geometric function and should keep taking metres, so it remains testable in isolation. Do
the conversion in `Ransac3DoFEstimator.prepare`, where the mesh is in hand:

```python
x_extent = float(np.ptp(np.asarray(mesh.vertices)[:, 0]))
depth = self.params.front_crop_fraction * x_extent
slab_mesh = crop_front_face(mesh, depth)
```

Then **remove `front_crop_depth` from `Ransac3DoFEstimator.suggest_params`** — the
`if "front_crop_depth" not in fixed:` block from note 01 §0.1c comes out entirely — and
update every profile in `cli_config.py`:

```python
                    # was: front_crop_depth=0.7352383501440559,
                    front_crop_fraction=0.25,
```

`np.ptp` is "peak to peak" — `max - min` along an axis. Note it must be the **mesh's own**
extent, computed before cropping, not the cropped slab's.

**Do not miss the cache key.** `methods/ransac3dof.py:401-408`:

```python
    def _get_prep_params_key(self) -> tuple:
        # front_crop_depth changes the prepared model representation, so it
        # must be part of the cache key alongside voxel_size.
        return (
            self.params.voxel_size,
            self.params.front_crop_depth,      # <- becomes front_crop_fraction
            self.params.hoppe_normal_orientation,
        )
```

`prepare()` skips its work entirely when this tuple matches a cached entry. Rename the field
without touching this and you get an `AttributeError` — which is the *good* outcome. The bad
one is if you add `front_crop_fraction` as a new field while leaving `front_crop_depth` in
place as dead code: the key would then track a parameter nothing reads, two arms with
different fractions would share one cached slab, and the second arm would silently register
against the first's geometry. That is the same phantom comparison as note 01 §2, one layer
down, and it would look exactly like "the crop fraction makes no difference".

### A strictly better formula, if you want it

The 25% rule improves things a lot but still leaves colruyt at 1.30 while leanflow gets
2.90 — it equalises the *wrong* quantity. What actually matters is the aspect ratio, so
target it directly:

$$\text{depth} = \frac{L_y}{R_{\text{target}}}$$

At $R_{\text{target}} = 2.0$: colruyt 0.419 m, picanol 0.398 m, leanflow 0.352 m.
**Every cart gets aspect exactly 2.0, and the absolute slab depths come out nearly
uniform** (0.35–0.42 m) — which also makes `min_sample_distance`, voxel quantization and
FPFH support behave comparably across the fleet, instead of leanflow's 0.243 m being 2.6×
thinner than colruyt's 0.644 m.

I would take this one. It is the same amount of code — `depth = y_extent / R` instead of
`depth = f * x_extent` — and it controls the failure mode directly rather than by proxy.
Your call; the fraction version is a large improvement either way.

### The risk to watch

Both versions cut leanflow's slab substantially (0.735 → 0.243 m at 25%, → 0.352 m at
$R{=}2$). Less geometry means fewer FPFH keypoints and weaker retrieval. If `good_rate`
drops specifically on leanflow, that is the mechanism — the per-frame CSV from
`--dump-frames` will show it by cart type.

---

## 5. `z_gate_threshold` — floor derivable, ceiling measurable, optimum neither

Your instinct — "it should be floored by sensor noise like `voxel_size`" — is right about
the floor and does not settle the value. Both bounds are derivable; the shipped value
violates the upper one.

### What it does

`z_gate_threshold` is the half-width of a consistency gate on FPFH correspondences: a
model point and a scene point may only match if their heights in the **robot base frame**
agree to within it.

### The floor, from sensor noise

Your extrinsic (`CameraConfig.extrinsic`) is a rotation of 60° about $y$:

$$z_{\text{base}} = -0.866\,x_{\text{cam}} + 0.5\,z_{\text{cam}} + 0.304$$

The dominant error is along the camera's depth axis, entering with coefficient 0.5, so
$\sigma_{z,\text{base}} \approx 0.5\,\sigma_z \in [0.2, 1.7]$ cm over the working range
(§3). A gate must not reject true correspondences, so it needs $\approx 3\sigma$:

$$z_{\text{gate}} \gtrsim 3 \times 1.7\ \text{cm} \approx 0.05\ \text{m}$$

This is where the "same as `voxel_size`" intuition lands: both are floored by the same
$\sigma_z$, but with different multipliers. A voxel quantizes at $\approx 1\sigma$; a
*tolerance* must admit at $\approx 3\sigma$. So $z_{\text{gate}} \approx 3\times$ voxel is
the defensible coupling — **0.06 m**, not 0.02. (The docstring records a historical
`voxel_size * 1.5` coupling; the multiplier was too small.)

### The ceiling, from the CAD

The gate's *purpose* is to stop a model point on one horizontal structure matching a scene
point on a different one. Its ceiling is therefore half the minimum vertical spacing
between distinct horizontal structures on the cart — deck, rails, top frame. If two rails
sit 0.30 m apart, a half-width above 0.15 m lets them swap.

**The shipped `tuned` value is 0.349 m.** On a picanol that is 2.018 m tall this is
plausibly survivable; on a leanflow 0.617 m tall it is **more than half the cart's entire
height**, which means the gate is very nearly not gating at all.

### Why it is not cleanly derivable, and what to do

A tuned value 7× the noise floor and (probably) above the CAD ceiling is a symptom, not a
setting: the sweep pushed it wide because tightening it *starves the correspondence set*.
That points back at retrieval quality — the standing S0 finding — not at the gate.

So do not derive this one blind. **Measure the ceiling first**, in `scratch/`:

> Load each cart's `model_down` at its ground-truth pose, transform to the base frame, take
> the $z$ coordinates, and histogram them at 1 cm bins. Read off the modes — these are the
> cart's horizontal structures — and record the minimum gap between adjacent modes. The
> ceiling is half that gap.

Then set `z_gate_threshold = min(ceiling, 0.06)` if the ceiling exceeds the floor, and run
it as an arm against 0.349. If the ceiling comes out *below* 0.05 m the two bounds have
crossed, which would be a real finding: it would mean the gate cannot simultaneously admit
true matches and reject structural confusions at this sensor's precision, and the mechanism
needs rethinking rather than retuning.

Until that measurement exists, **keep `z_gate_threshold` in the sweep**. It is the one
parameter here I cannot honestly hand you a number for.

---

## 6. `edge_length_threshold` — the parameterisation is wrong

`methods/vsac_se2.py:40` carries the comment `# still need to understand what that's used
for`. Here is what it does, and why a single ratio cannot be correct.

### What it does

At `methods/vsac_se2.py:335` (and identically at `methods/constrained_ransac.py:228`):

```python
if min(len_p, len_q) / max(len_p, len_q) < params.edge_length_threshold:
    continue   # reject this sample pair
```

`len_p` is the distance between the two sampled **model** points, `len_q` between their
two corresponding **scene** points. A rigid transform preserves distances, so a correct
pair of correspondences must have $len_p \approx len_q$. The check rejects pairs whose two
edge lengths disagree by more than a multiplicative factor — a cheap pre-filter that kills
bad samples before the expensive pose solve.

### Why a ratio is the wrong quantity

Model points are noise-free — they come from the CAD. Scene points carry position noise
$\sigma$. For an edge of length $L$ between two scene points with independent isotropic
noise, the error in the *length* is the difference of the two endpoint errors projected on
the edge direction, so $\sigma_L \approx \sigma\sqrt{2}$ — **independent of $L$**.

But the check is on the *ratio*, so the tolerance it implies is relative:

$$\text{ratio} \approx 1 - \frac{k\,\sigma\sqrt 2}{L}$$

for a $k\sigma$ acceptance. The required ratio therefore depends on edge length. At
$\sigma = 0.01$ m (mid-range, §3) and $k = 3$:

| $L$ | required ratio |
| ---: | ---: |
| 0.06 m (= `min_sample_distance`) | **0.29** |
| 0.15 m | 0.72 |
| 0.42 m | 0.90 |
| 1.00 m | 0.96 |

The shipped 0.9 is correct at exactly one edge length. Setting $1 - 3\sigma\sqrt2/L = 0.9$
gives the break-even:

$$L^* = 30\,\sigma\sqrt 2 \approx 42\,\sigma \approx 0.42\ \text{m}$$

**Every sample pair with a baseline below 0.42 m is being over-rejected**, and
`min_sample_distance` is only $3 \times 0.02 = 0.06$ m — so the admissible range is
0.06–0.42 m of over-rejection against a thin sliver above it. On a leanflow slab 0.243 m
deep, *every* pair is in the over-rejected regime. The sweep's preference for
`edge_length_threshold` near the bottom of its 0.8–0.95 range is this effect showing
through.

### The edit

Replace the ratio with an absolute tolerance, in both files:

```python
if abs(len_p - len_q) > params.edge_length_tolerance:
    continue
```

with `edge_length_tolerance = k * sigma * sqrt(2)`, $k = 3$, $\sigma$ the scene-point
position noise. Since $\sigma$ is itself range-dependent (§3), the honest constant is the
worst case over the working range, $\sigma = 0.033$ m at $\sigma_d = 0.3$:

$$\text{tolerance} = 3 \times 0.033 \times 1.414 \approx 0.14\ \text{m}$$

That is generous. If you want it tighter, the principled version computes $\sigma$
per-correspondence from the scene point's own depth — you have $z$ for every point, and
$\sigma_z = z^2\sigma_d/(fB)$ is one multiply. That is the version I would build eventually;
the constant is fine to start.

**The cheap alternative**, if you do not want to touch the sampler: raise
`min_sample_distance` above $L^*$ so only long edges are sampled, where 0.9 is approximately
right. It costs nothing but discards short-baseline pairs entirely — and on leanflow's
0.243 m slab there are no long baselines to be had, so this fails exactly where it is most
needed. Prefer the absolute tolerance.

Either way, **remove `edge_length_threshold` from `suggest_params`**
(`methods/ransac3dof.py:692`).

---

## 7. `ransac_max_iterations` — I was wrong, do not fix it at 10k

**Correction to note 01's earlier framing and to my answer in conversation.** I said this
parameter "buys latency and nothing else", citing Spearman $-0.006$ against `good_rate` and
$+0.930$ against p95. That correlation was computed **within a sweep bucket where iterations
co-varied with eight other parameters**. Your own dedicated ladder refutes it.

All four runs below are on `8c6893c`, seed 5829, `eval_size 70`, `n_seeds 3`,
`hoppe_normal_orientation: true`, `normal_consistency: false` — a clean single-variable
ladder:

| W&B run | iterations | `good_rate` | `pose_ar` | p95 |
| :--- | ---: | ---: | ---: | ---: |
| `74n9avkc` | 1 000 | 0.6184 | 0.2138 | 1.93 s |
| `xviophc7` | 3 000 | 0.6522 | 0.2350 | 2.02 s |
| `ifbe9gvt` | 10 000 | 0.7488 | 0.2825 | 2.56 s |
| `rouf33ov` | 46 940 | **0.8164** | **0.3137** | 5.59 s |

**Monotone in both accuracy metrics, across a 47× range.** Capping at 10 000 costs 0.068
`good_rate` and 0.031 `pose_ar` to save 3.0 s of p95. That is a real trade, not a free win.

### So: no fixed constant, but stop searching it

It *is* a latency budget — that part was right. What was wrong was concluding the accuracy
curve is flat. The budget has to be chosen against the curve, and **the curve has moved**:
the front-face gate cut p95 from 5.55 s to 3.08 s by pruning hypotheses, so the
iterations-to-accuracy relationship on the gated pipeline is not the one measured above.

**The edit.** Remove `ransac_max_iterations` from `suggest_params`
(`methods/ransac3dof.py:698`) and set it per profile as an explicit budget. Then re-measure
the ladder on top of the gate — `{2 000, 10 000, 46 940, 100 000}` on the full test set,
paired seed — and pick the knee for a `tuned-fast` profile and the plateau for
`tuned-accurate`. Two profiles, one honest trade-off, no search.

Nothing here reaches the 0.2 s / 5 Hz Orin Nano budget: the fastest arm above is 1.93 s,
still 10× over, and that stays true across every configuration measured on this project.
The iteration knob is choosing where on a bad curve to sit, not getting onto a good one.

---

## 8. What the search space becomes

| parameter | after this note |
| :--- | :--- |
| `depth_trunc` | **fixed 5.5** — workcell geometry (§2) |
| `voxel_size` | **fixed 0.02** — sensor noise (§3) |
| `front_crop_depth` | **formula** `front_crop_fraction × L_x`, or `L_y / R` (§4) |
| `edge_length_threshold` | **replaced** by an absolute tolerance ≈ 0.14 m (§6) |
| `ransac_max_iterations` | **per-profile budget**, chosen from a measured ladder (§7) |
| `z_gate_threshold` | **still swept** pending the CAD measurement (§5) |
| `rho` | swept |
| `icp_max_correspondence_distance` | swept — and its optimum has moved, see below |
| `icp_max_iterations` | swept |

**Four free parameters, and one of those is temporary.** At four dimensions, 100 TPE trials
is a genuine search rather than a lottery.

`icp_max_correspondence_distance` deserves a note: its job changed when the visibility cull
landed. It used to be capture range *and* final precision; the ladder now supplies precision,
so it is capture range only. Its optimum should move upward, and the first post-refactor
sweep is where that shows up.

---

## 9. Tests to write

**§2 — `depth_trunc`.** A test that the value covers the generation envelope: assert
$\cos\theta\,(3.0 + L_x) + (L_z - h)\sin\theta < \texttt{depth\_trunc}$ for all three meshes,
with $\theta = 30°$, $h = 0.304$ hard-coded from the generation script. It should fail if
someone lowers `depth_trunc` back toward 4.6, and the failure message should name the cart
that no longer fits.

**§3 — `voxel_size`.** Assert `voxel_size` is no longer a key returned by
`RansacEstimator.suggest_params(stub_trial)`. Guards against a re-added search dimension.

**§4 — the crop formula.** Three tests:
1. The slab's actual x-extent after `prepare()` is within one voxel of
   `fraction × L_x` — this catches the mesh-extent-vs-slab-extent confusion directly.
2. The resulting aspect ratio $L_y/\text{depth} \ge 1.25$ **for all three carts**. This is
   the regression test for Trap 1, and it should fail at `fraction = 0.35` (colruyt drops
   to 0.93). Assert on all three, not just leanflow — the whole point of §4 is that this
   was never a leanflow-only problem.
3. The cropped mesh is non-empty and retains at least some minimum vertex count, so a
   too-small fraction fails loudly rather than producing an empty registration target.

**§6 — the edge-length check.** Construct two model points at a known separation $L$ and two
scene points at $L + \delta$. Assert the pair is accepted for $\delta$ just under the
tolerance and rejected just over, **at two very different $L$ values** (say 0.08 m and
0.80 m). Under the old ratio check the 0.08 m case rejects and the 0.80 m case accepts for
the same $\delta$ — that asymmetry is the bug, and this test is what pins the fix.

---

## 10. The arms, and how to verify

Every arm at **`--seed 2144065271`** so it pairs against `s71rtlvr` (see 01 §5).

Because this note changes several parameters at once, **do not launch them as one arm.**
The point of the whole refactor is that each value is derived; if the stack regresses you
need to know which derivation was wrong.

```bash
# B1 — depth_trunc alone (expect: neutral-to-positive, biggest effect on far colruyt frames)
uv run benchmark.py --eval-size 1500 --n-seeds 3 --seed 2144065271 \
  --name B1_depthtrunc55 \
  model:vsac3dof model.profile:tuned-vis-gate \
  --model.profile.depth-trunc 5.5

# B2 — B1 + the crop formula (expect: the yaw guard should stop firing)
#      add --model.profile.params.front-crop-fraction 0.25 once the field exists

# B3 — B2 + the edge-length tolerance
# B4 — the iteration ladder, four runs at 2000/10000/46940/100000
```

**Verification, per arm.**

*B1.* Good: `good_rate` flat or up, and the per-frame CSV shows the improvement concentrated
in colruyt frames at large range. Bad: a drop — that would mean the truncation was
accidentally helping by hiding scene points that steal ICP correspondences, which would be
worth knowing and would point at the correspondence radius rather than at `depth_trunc`.

*B2.* The sharp test: **how often does the yaw guard fire?** It exists to catch the 90° twin,
and §4 removes the twin's cause on all three carts. Good: guard trips fall toward zero and
`good_rate` holds. Bad: `good_rate` drops on leanflow specifically — that is the thin-slab
retrieval risk from §4, and the answer is $R_{\text{target}} = 2$ instead of a fixed
fraction, not a retreat to metres.

*B3.* Good: `good_rate` up, latency roughly flat — a better pre-filter admits more true
pairs without changing the cost per pair. Bad: latency up sharply, meaning the tolerance is
so loose that bad samples now reach the pose solve; tighten $k$ from 3 toward 2.

*B4.* Read `good_rate` against p95 and pick two points. There is no "pass" here — it is a
trade-off curve, and the deliverable is two profiles.

**Across all of them**, run `scripts/mcnemar_arms.py` against `s71rtlvr` rather than
eyeballing rates, and always read `good_rate` and `abstention_rate` **as a pair** — a rate
that improves while abstention climbs is a denominator being emptied, which is the failure
mode note 30.06 checked for explicitly.

---

## 11. APIs and details

**`np.ptp`** — "peak to peak", i.e. `arr.max() - arr.min()`. `np.ptp(vertices[:, 0])` is the
mesh's x-extent. Takes an `axis=` argument like other reductions.

**Axis-aligned bounding box, via trimesh.** `trimesh.load(path, force="mesh").bounds` returns
a `(2, 3)` array of `[[xmin, ymin, zmin], [xmax, ymax, zmax]]`; the extents are `hi - lo`.
`force="mesh"` collapses a multi-part scene into a single mesh, which matters because some
of these files load as `Scene` objects otherwise. The `.ply` files are the ones the pipeline
uses; the `.usdc` siblings are the Isaac assets and are not loadable this way.

**Why `distance_to_image_plane`, not range.** Isaac's annotator (and the `DepthSensorDistance`
one the generation script actually writes) reports the component along the optical axis, not
Euclidean distance from the camera centre. This is what every depth camera reports and what
`depth_trunc` compares against, which is why §2 projects onto $\hat a$ rather than taking a
norm. Getting this wrong would overestimate $z_{\max}$ by about 10% at the worst-case
geometry — not fatal here, but the derivation should be right.

**Why depth noise grows as $z^2$.** Disparity is what the sensor measures, and it is
*inversely* proportional to depth: $D = fB/z$. A fixed matching error in disparity therefore
maps to a depth error that grows quadratically. It is the single most important fact about
stereo depth and it is why the far half of your working range dominates every noise budget
in this note.

**Intel's "2% at 4 m" is an accuracy spec, not a noise spec.** It bundles calibration bias
with random error. Random error alone is smaller — hence the $\sigma_d \in [0.1, 0.3]$ px
bracket in §3 rather than a single number. Where a decision depends on which end you take,
§3 says so.

**A caveat on all of §3.** Your data is *synthetic*. The depth comes from Isaac's
`SingleViewDepthCameraSensor` with the noise model embedded in `rsd455.usd`
(`generate_dataset_6_0_1.py:783-801`), and the script's own comments flag that wiring as
unverified. So the D455 datasheet bounds the *real* sensor, while the actual noise in your
parquet files is whatever Isaac simulates. If a `voxel_size` arm behaves oddly, measuring
the empirical noise directly — fit a plane to a flat region of a scene cloud and take the
residual std — is the check that settles it.

---

## 12. Related

[01 — Harness fixes](01-harness-fixes-before-the-confirmation-batch.md) · vault
`30.06 - T0 Translation Error and the Visibility Cull` (Trap 1, generalised here to the
whole fleet) · `30.04.5 - P0 Normal Orientation and E01 Normal Agreement` ·
`30.05 - Global Registration Replacement Plan` (§5's starved-correspondence argument bears
on its premise).

Source: `/home/martin/yolo_dataset_creation/generate_dataset_6_0_1.py` ·
[Intel RealSense D455](https://www.intelrealsense.com/depth-camera-d455/) ·
[D455 product brief](https://www.mouser.com/pdfDocs/D455ProductBriefv90.pdf)

W&B: `74n9avkc`, `xviophc7`, `ifbe9gvt`, `rouf33ov` (the iteration ladder) · `s71rtlvr`
(the pairing baseline).
