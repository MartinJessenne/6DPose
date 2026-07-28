"""
Paired significance test between two A/B arms, on their per-frame CSVs.

Why this exists
---------------
The E1 factorial moved good_rate 0.6957 -> 0.7633 over 207 evaluations. An
UNPAIRED two-proportion test on that gives z = 1.55, p ~ 0.12 -- not significant
-- but an unpaired test is the wrong test and throws away most of the
information. Both arms ran the SAME 70 frames under the SAME internal seeds
(derive_internal_seeds is a pure function of --seed), so the comparison is
paired, and the only frames carrying any signal are the ones where the two arms
DISAGREE. That is McNemar's test:

           chi2 = (|b - c| - 1)^2 / (b + c)          (continuity-corrected)

where b = frames the control got right and the treatment got wrong, c = the
reverse. Frames both arms agree on cancel and contribute nothing, which is
exactly why pairing is more powerful here: at good_rate ~0.7 most frames agree.

Two granularities, because they answer different questions:

  per-frame (majority vote over the 3 internal seeds), n = 70
      The honest unit. Frames are independent; the 3 seeds of one frame are not.
      This is the number to quote.

  per-evaluation (frame x seed), n = 210
      Secondary. Higher apparent power, but the observations are correlated
      within a frame, so its p-value is anti-conservative. Reported so the gap
      between the two is visible rather than hidden by choosing one.

Also reports the disagreement breakdown by cart type and by outcome, because
"which frames did it fix, and did it break any" is more actionable than a
p-value.

Usage
-----
    uv run scripts/mcnemar_arms.py \
        benchmark_runs/E1_00_control_frames.csv \
        benchmark_runs/E1_11_both_frames.csv
"""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_arm(path: Path) -> tuple[dict[int, list[str]], dict[int, str]]:
    """
    Returns {sample_idx: [outcome per seed, in run order]} and {sample_idx: cart}.

    Rows repeat sample_idx once per internal RANSAC seed with no seed column, and
    evaluate_pipeline visits frames in the same order for every seed, so the k-th
    occurrence of a sample_idx is its k-th seed in BOTH arms. That correspondence
    is what makes the per-evaluation pairing below meaningful; it holds only
    because both arms ran the same --seed and --n-seeds.
    """
    outcomes: dict[int, list[str]] = defaultdict(list)
    carts: dict[int, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["sample_idx"])
            outcomes[idx].append(row["outcome"])
            if row.get("cart_type"):
                carts[idx] = row["cart_type"]
    return dict(outcomes), carts


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Continuity-corrected chi2 and its two-sided p-value (1 dof)."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # Survival function of chi2 with 1 dof = erfc(sqrt(chi2/2)).
    return chi2, math.erfc(math.sqrt(chi2 / 2.0))


def report(label: str, b: int, c: int, n: int) -> None:
    chi2, p = mcnemar(b, c)
    verdict = "SIGNIFICANT at 0.05" if p < 0.05 else "not significant at 0.05"
    print(f"\n{label}  (n = {n})")
    print(f"  control right, treatment wrong   b = {b}")
    print(f"  control wrong, treatment right   c = {c}")
    print(f"  agree                            {n - b - c}")
    print(f"  net                              {c - b:+d}")
    print(f"  McNemar chi2 = {chi2:.3f}   p = {p:.4f}   -> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("control_csv", type=Path)
    ap.add_argument("treatment_csv", type=Path)
    ap.add_argument(
        "--success",
        default="good",
        help="FrameRecord.outcome value counted as a success (default: good)",
    )
    args = ap.parse_args()

    ctrl, carts = load_arm(args.control_csv)
    treat, carts_t = load_arm(args.treatment_csv)
    carts = {**carts_t, **carts}

    shared = sorted(set(ctrl) & set(treat))
    if not shared:
        print("No shared sample_idx between the two CSVs.", file=sys.stderr)
        return 1
    only_c, only_t = set(ctrl) - set(treat), set(treat) - set(ctrl)
    if only_c or only_t:
        print(f"WARNING: {len(only_c)} frames only in control, {len(only_t)} only in "
              f"treatment -- dropped. The arms did not evaluate the same set.")

    def rate(d):
        tot = sum(len(v) for k, v in d.items() if k in shared)
        ok = sum(o == args.success for k, v in d.items() if k in shared for o in v)
        return ok, tot

    ok_c, n_c = rate(ctrl)
    ok_t, n_t = rate(treat)
    print(f"control   {args.control_csv.name}   {args.success} {ok_c}/{n_c} = {ok_c / n_c:.4f}")
    print(f"treatment {args.treatment_csv.name}   {args.success} {ok_t}/{n_t} = {ok_t / n_t:.4f}")

    # --- per frame, majority vote over seeds (the quotable unit) ---
    def majority(seq):
        return sum(o == args.success for o in seq) * 2 > len(seq)

    b = c = 0
    flipped_to_good, flipped_to_bad = [], []
    for idx in shared:
        mc, mt = majority(ctrl[idx]), majority(treat[idx])
        if mc and not mt:
            b += 1
            flipped_to_bad.append(idx)
        elif mt and not mc:
            c += 1
            flipped_to_good.append(idx)
    report("PER FRAME (majority over seeds)", b, c, len(shared))

    # --- per evaluation, frame x seed (anti-conservative, reported for contrast) ---
    be = ce = 0
    n_eval = 0
    for idx in shared:
        for oc, ot in zip(ctrl[idx], treat[idx], strict=False):
            n_eval += 1
            if oc == args.success and ot != args.success:
                be += 1
            elif ot == args.success and oc != args.success:
                ce += 1
    report("PER EVALUATION (frame x seed, correlated -- p is optimistic)", be, ce, n_eval)

    # --- what actually changed ---
    print(f"\nfixed   ({len(flipped_to_good)}): "
          f"{Counter(carts.get(i, '?') for i in flipped_to_good).most_common()}")
    print(f"broken  ({len(flipped_to_bad)}): "
          f"{Counter(carts.get(i, '?') for i in flipped_to_bad).most_common()}")

    if flipped_to_bad:
        print("\nregressed frames (sample_idx, cart, control -> treatment outcomes):")
        for i in flipped_to_bad:
            print(f"  {i:5d} {carts.get(i, '?'):9} {ctrl[i]} -> {treat[i]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
