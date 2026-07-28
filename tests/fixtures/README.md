# Test fixtures

18 frames sampled from [`UItraviolet/industrial_cart`](https://huggingface.co/datasets/UItraviolet/industrial_cart),
committed so the full pipeline can be run and scored without network or credentials.

Regenerate with `uv run scripts/fetch_test_samples.py`.

## Why these frames

6 per split, 2 per cart type, chosen to span the **bearing** range -- the angle
between the cart's outward front-face arrow and the direction to the camera.
A front-face gate wrongly written against a fixed base_link axis (instead of the
direction to the camera) passes on head-on carts and fails only on angled ones,
so both extremes must be present.

Ground-truth bearings in this dataset span roughly ±45°; flipped poses sit at
≥135°. See the plan's threshold discussion.

## Provenance

| split | source shard | row | cart | bearing | range |
|---|---|---|---|---|---|
| test | `test-00000-of-00016.parquet` | 45 | picanol | 0.89° | 2.81 m |
| test | `test-00000-of-00016.parquet` | 60 | colruyt | 43.97° | 2.90 m |
| test | `test-00000-of-00016.parquet` | 61 | colruyt | -0.13° | 1.35 m |
| test | `test-00000-of-00016.parquet` | 63 | leanflow | 1.99° | 1.90 m |
| test | `test-00000-of-00016.parquet` | 66 | picanol | 44.61° | 1.86 m |
| test | `test-00000-of-00016.parquet` | 77 | leanflow | 42.32° | 2.02 m |
| validation | `validation-00000-of-00016.parquet` | 23 | colruyt | 44.88° | 0.91 m |
| validation | `validation-00000-of-00016.parquet` | 37 | picanol | 42.16° | 1.57 m |
| validation | `validation-00000-of-00016.parquet` | 65 | leanflow | 0.15° | 1.66 m |
| validation | `validation-00000-of-00016.parquet` | 81 | colruyt | 3.02° | 1.74 m |
| validation | `validation-00000-of-00016.parquet` | 84 | picanol | 2.94° | 2.11 m |
| validation | `validation-00000-of-00016.parquet` | 90 | leanflow | 43.96° | 2.42 m |
| train | `train-00000-of-00127.parquet` | 28 | picanol | -1.17° | 2.56 m |
| train | `train-00000-of-00127.parquet` | 33 | colruyt | 44.82° | 2.31 m |
| train | `train-00000-of-00127.parquet` | 40 | leanflow | 0.58° | 1.01 m |
| train | `train-00000-of-00127.parquet` | 48 | colruyt | 1.59° | 1.67 m |
| train | `train-00000-of-00127.parquet` | 61 | leanflow | 43.66° | 1.70 m |
| train | `train-00000-of-00127.parquet` | 90 | picanol | 44.86° | 0.99 m |
