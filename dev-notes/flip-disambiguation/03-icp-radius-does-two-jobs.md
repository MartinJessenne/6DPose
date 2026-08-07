---
status: draft
---

# The ICP search radius is still doing two jobs

## What is wrong

`icp_max_correspondence_distance` is the radius the KD-tree searches in. It does
two things at once:

1. **Find matches.** A model point only gets a scene match if one sits inside the
   radius. Too small, and points far from their true match get nothing, so the
   solver cannot pull them in.
2. **Throw out bad matches.** Anything outside the radius is dropped completely.
   Too big, and points with no real match still get paired with something far
   away, and that drags the pose off.

GNC was added to take job 2 away from the radius. The kernel starts wide and
shrinks, so bad matches lose weight smoothly instead of being cut off. The radius
was supposed to be left alone as job 1.

**It only half worked.** The Geman-McClure kernel makes a bad match small, never
zero. At the final scale of 0.0148 m, a match that is 5 cm off still carries about
0.7% weight. One of those does not matter. Thousands do, and the sum pulls the
pose off.

So the radius is still needed for job 2. Deleting `icp_refine_ladder`, which was
shrinking the radius in steps, made the pose worse.

## The numbers

Distance from ground truth, in cm. No randomness: ICP starts at the true pose and
we measure how far it walks away.

| cart | with the old ladder | now (fixed wide radius) |
| --- | --- | --- |
| colruyt | 0.087 | 0.155 |
| leanflow | 0.092 | 0.103 |
| picanol | 0.152 | 0.249 |

On the 6 local fixture frames: `pose_ar` 0.76 -> 0.69, median translation error
0.0079 m -> 0.0083 m. Every single frame got worse.

I checked whether the ladder helped for a different reason — because it re-ran the
visibility cull three times instead of once. It did not. Running the wide stage
three times with re-culling gives 0.155, the same as running it once. The
shrinking radius is the thing that mattered.

## Why we cannot just pick a smaller radius

Best fixed radius, per cart:

| cart | best radius | error there | error at 0.010 |
| --- | --- | --- | --- |
| colruyt | 0.010 | 0.089 | 0.089 |
| leanflow | 0.044 | 0.101 | 0.206 |
| picanol | 0.015 | 0.145 | 0.152 |

Leanflow is twice as bad at colruyt's best value. There is no single number that
works for all three carts. There is already a test for this in
`tests/test_gnc_icp.py` (`TestOptimalFixedRadiusIsNotKnowable`); this is the same
result on the real meshes.

## Options

### A. Shrink the radius along with the kernel, inside `icp_se2`

Right now the radius is fixed for the whole run. Instead, set it each iteration
from the current kernel scale:

    radius = min(icp_max_correspondence_distance, c * scale)

Early on the scale is wide, so the radius is the full capture radius and nothing
changes. Late on the scale is 0.0148 m, so the radius closes in and cuts off the
far matches the kernel cannot fully kill.

One new number, `c`. I tested c = 3, 2, 1 and 0.7. Only 0.7 starts to win
(colruyt 0.082, picanol 0.133), and leanflow is flat throughout. So `c` needs to
be swept, not guessed.

- Cost: about 5 lines in the ICP loop, one new parameter.
- Keeps the two jobs separate: one schedule drives both, no hand-picked steps.
- The radius stays tied to the sensor noise floor, not to a cart.

### B. Put the ladder back

Restore `icp_refine_ladder = (0.05, 0.02, 0.01)` and the extra stages.

- It works today: it is the best column in the table above.
- But the values are hand-picked and cart-independent by luck, not by design. The
  last step, 0.01 m, is below the sensor noise floor of 0.0148 m, so that stage
  finds almost no matches at all. It works despite that, which nobody can explain.
- It also runs 3 extra full ICP passes, and latency is already 10x over budget.

### C. Use a kernel that actually reaches zero

Swap Geman-McClure for a kernel with hard cut-off at the end, such as Tukey. At
the final scale a match past the cut-off gets exactly zero weight, so job 2 is
fully done by the kernel and the radius can stay wide.

- Cleanest in theory: one mechanism, no new parameter.
- Bigger change: the GNC schedule and its convergence story are written around
  Geman-McClure, and Tukey needs its own schedule.
- Untested here. I do not know if it recovers the ladder's numbers.

## What I would do

**Option A.** It keeps the split GNC was added for, adds one parameter instead of
three hand-picked ones, and `c` is a normal thing to sweep next to the anneal
start size you already want to sweep.

Option C is the better end state if it works, but it is a bigger rewrite and
nothing has been measured yet. Option B is the only one that is fast to do and
known to work today, but it puts back the coupling we just removed and nobody can
say why 0.01 m is a good number.

## How to check it worked

    PYTHONPATH="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_gnc_icp.py -q

Then the RNG-free check, which is the one that matters: ICP starts at ground truth
and we measure the walk-away. Good means the three carts land at or under the
ladder column: 0.087, 0.092, 0.152 cm. Bad means they sit at the current 0.155,
0.103, 0.249.
