"""
Renders the 18 committed fixtures as a single contact-sheet PNG.

Why this exists
---------------
`tests/fixtures/manifest.json` says what each frame *is* (cart type, bearing,
range) but not what it *looks like*. Two things need eyeballing before any dense
registration work: whether the leanflow frames carry cargo (boxes present in the
point cloud and absent from the CAD), and how much of the cart is actually in
frame at the near ranges.

Each panel is labelled with split, dataset index, cart type, bearing and range,
so a panel can be traced straight back to a manifest entry.

Usage
-----
    uv run scripts/inspect_fixtures.py
    uv run scripts/inspect_fixtures.py --out /tmp/sheet.png --cols 6
"""

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SPLITS = ("test", "validation", "train")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=str(REPO_ROOT / "tests" / "fixtures"))
    ap.add_argument("--out", default=str(REPO_ROOT / "fixtures_contact_sheet.png"))
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--panel-width", type=int, default=520)
    args = ap.parse_args()

    import pyarrow.parquet as pq  # noqa: PLC0415

    fixtures = Path(args.fixtures)
    manifest = json.loads((fixtures / "manifest.json").read_text())

    panels = []
    for split in SPLITS:
        table = pq.read_table(fixtures / "data" / f"{split}-00000-of-00001.parquet")
        rgb_col = table.column("rgb").to_pylist()
        entries = manifest[split]["frames"]
        # The parquet holds exactly the manifest's frames, in manifest order --
        # the dataset index in the manifest refers to the ORIGINAL shard, not to
        # this row, so zip rather than index.
        for row, entry in zip(rgb_col, entries, strict=True):
            img = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
            label = (
                f"{split}[{entry['index']}]  {entry['cart_type']}  "
                f"bearing {entry['bearing_deg']:+.1f}deg  range {entry['range_m']:.2f}m"
            )
            panels.append((img, label))

    if not panels:
        print("no fixtures found", file=sys.stderr)
        return 1

    pw = args.panel_width
    ph = int(pw * panels[0][0].height / panels[0][0].width)
    bar = 22
    cols = args.cols
    rows = (len(panels) + cols - 1) // cols

    sheet = Image.new("RGB", (cols * pw, rows * (ph + bar)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (img, label) in enumerate(panels):
        x = (i % cols) * pw
        y = (i // cols) * (ph + bar)
        sheet.paste(img.resize((pw, ph), Image.LANCZOS), (x, y))
        draw.text((x + 6, y + ph + 5), label, fill=(235, 235, 235))

    sheet.save(args.out)
    print(f"{len(panels)} panels -> {args.out}  ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
