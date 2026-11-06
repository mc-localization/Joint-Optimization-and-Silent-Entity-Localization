"""Generate the independent four-receiver dataset used by the new experiment.

The four receivers are placed in the four quadrants around the transmitter.  A
checkerboard split of the spatial grid makes train and validation positions
disjoint; repetitions at a position are Monte-Carlo replicates, not new spatial
units.  The test set contains continuous random positions.

The defaults are deliberately smaller than the original 10,000-molecule,
1,000-step generator so generation remains practical on a laptop.  They can be
overridden from the command line without editing the experiment code.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


TX = (0.0, 0.0)
# Four receiver centers at (+/-20,+/-20) um; the SE region remains [-20,20]^2.
RX_LIST = [(-20.0, -20.0, 4.0), (-20.0, 20.0, 4.0),
           (20.0, -20.0, 4.0), (20.0, 20.0, 4.0)]
R_E = 4.0
# SE centers must be strictly more than 4 um from Tx and 8 um from each Rx.
X_RANGE = (-20.0, 20.0)
Y_RANGE = (-20.0, 20.0)


def valid_point(x: float, y: float) -> bool:
    if (x - TX[0]) ** 2 + (y - TX[1]) ** 2 <= R_E ** 2:
        return False
    return all((x - rx) ** 2 + (y - ry) ** 2 > (R_E + rr) ** 2
               for rx, ry, rr in RX_LIST)


def make_grid(step: float) -> tuple[np.ndarray, np.ndarray]:
    from geometry import make_grid as exact_grid
    return exact_grid(step)


def continuous_points(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result: list[tuple[float, float]] = []
    while len(result) < n:
        count = max(1000, 3 * (n - len(result)))
        xx = rng.uniform(*X_RANGE, size=count)
        yy = rng.uniform(*Y_RANGE, size=count)
        result.extend((float(x), float(y)) for x, y in zip(xx, yy)
                      if valid_point(float(x), float(y)))
    return np.asarray(result[:n], dtype=np.float64)


def simulate(x_e: float, y_e: float, seed: int, molecules: int,
             n_steps: int, dt: float, diffusion: float,
             e_radius: float, n_features: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = float(np.sqrt(2.0 * diffusion * dt))
    xx = np.full(molecules, TX[0], dtype=np.float32)
    yy = np.full(molecules, TX[1], dtype=np.float32)
    alive = np.ones(molecules, dtype=bool)
    absorbed = np.zeros((len(RX_LIST), n_steps), dtype=np.int32)
    for t in range(n_steps):
        active = np.flatnonzero(alive)
        if active.size == 0:
            break
        xx[active] += rng.normal(0.0, sigma, size=active.size).astype(np.float32)
        yy[active] += rng.normal(0.0, sigma, size=active.size).astype(np.float32)
        in_e = ((xx[active] - x_e) ** 2 + (yy[active] - y_e) ** 2) <= e_radius ** 2
        alive[active[in_e]] = False
        remaining = active[~in_e]
        for r, (rx, ry, rr) in enumerate(RX_LIST):
            if remaining.size == 0:
                break
            hit = ((xx[remaining] - rx) ** 2 + (yy[remaining] - ry) ** 2) <= rr ** 2
            if hit.any():
                selected = remaining[hit]
                absorbed[r, t] += int(selected.size)
                alive[selected] = False
                remaining = remaining[~hit]
    cumulative = np.cumsum(absorbed, axis=1)
    sample_idx = np.rint(np.linspace(0, n_steps - 1, n_features)).astype(np.int64)
    return cumulative[:, sample_idx].reshape(-1).astype(np.uint16)


def simulate_position(task: tuple[int, float, float, int, int, int, float, float, int, int]):
    position_id, x, y, repeats, seed_base, molecules, dt, diffusion, steps, features = task
    rows = []
    for rep in range(repeats):
        seed = seed_base + position_id * 100_003 + rep
        rows.append((simulate(x, y, seed, molecules, steps, dt, diffusion,
                              R_E, features), x, y))
    return rows


def write_shards(out_dir: Path, prefix: str, positions: np.ndarray, repeats: int,
                 seed_base: int, args: argparse.Namespace) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(positions)))
    tasks = [(i, float(p[0]), float(p[1]), repeats, seed_base, args.molecules,
              args.dt, args.diffusion, args.steps, args.features)
             for i, p in enumerate(positions)]
    shard_rows, shard_id, total = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(simulate_position, task) for task in tasks]
        for future in as_completed(futures):
            shard_rows.extend(future.result())
            while len(shard_rows) >= args.shard_size:
                chunk, shard_rows = shard_rows[:args.shard_size], shard_rows[args.shard_size:]
                X = np.stack([row[0] for row in chunk])
                np.savez_compressed(out_dir / f"{prefix}_{shard_id:03d}.npz",
                                    X=X,
                                    x=np.asarray([row[1] for row in chunk], dtype=np.float32),
                                    y=np.asarray([row[2] for row in chunk], dtype=np.float32))
                total += len(chunk); shard_id += 1
                print(f"[{prefix}] rows={total}", flush=True)
    if shard_rows:
        X = np.stack([row[0] for row in shard_rows])
        np.savez_compressed(out_dir / f"{prefix}_{shard_id:03d}.npz",
                            X=X,
                            x=np.asarray([row[1] for row in shard_rows], dtype=np.float32),
                            y=np.asarray([row[2] for row in shard_rows], dtype=np.float32))
        total += len(shard_rows)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data_4rx")
    ap.add_argument("--grid-step", type=float, default=0.8)
    ap.add_argument("--train-repeats", type=int, default=60)
    ap.add_argument("--valid-repeats", type=int, default=15)
    ap.add_argument("--test-points", type=int, default=500)
    ap.add_argument("--molecules", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--diffusion", type=float, default=79.4)
    ap.add_argument("--features", type=int, default=250)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    grid, parity = make_grid(args.grid_step)
    train_pos, valid_pos = grid[parity != 0], grid[parity == 0]
    print(f"grid={len(grid)}, train_positions={len(train_pos)}, valid_positions={len(valid_pos)}")
    for name in ("data_train", "data_valid", "data_test"):
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        # The loader reads every ``*.npz`` in a split directory.  Remove old
        # shards first so rerunning with different repeats or shard sizes
        # cannot silently append stale samples to the new dataset.
        for old_shard in out_dir.glob("*.npz"):
            old_shard.unlink()
    train_n = write_shards(root / "data_train", "train", train_pos, args.train_repeats,
                           args.seed, args)
    valid_n = write_shards(root / "data_valid", "valid", valid_pos, args.valid_repeats,
                           args.seed + 500_000, args)
    test_pos = continuous_points(args.test_points, args.seed + 999)
    test_n = write_shards(root / "data_test", "test", test_pos, 1,
                          args.seed + 1_000_000, args)
    metadata = vars(args) | {
        "tx": TX, "receivers": RX_LIST, "eve_radius": R_E,
        "x_range": X_RANGE, "y_range": Y_RANGE,
        "train_positions": int(len(train_pos)), "valid_positions": int(len(valid_pos)),
        "train_rows": train_n, "valid_rows": valid_n, "test_rows": test_n,
        "split": "checkerboard parity=(ix+iy)%5; validation parity=0",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"done: {root}")


if __name__ == "__main__":
    raise SystemExit("Use generate_data.py and the isolated configuration.")

