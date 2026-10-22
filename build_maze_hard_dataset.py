"""Generate the Maze-Hard dataset (HRM recipe) at an arbitrary grid size.

HRM's released Maze-Hard (sapientinc/maze-30x30-hard-1k) only exists at
30x30. This script reproduces the HRM recipe (HRM paper, App.: "we adopt the
instance generation procedure of Lehnert et al. (2024), but introduce an
additional filter to retain only those instances whose difficulty exceeds
110") so it can be run at other sizes:

  1. wall density p ~ U(wall_prob_min, wall_prob_max)   (Lehnert: 30-50%)
  2. every cell is a wall independently with prob. p (no border walls)
  3. start S and goal G drawn uniformly from distinct open cells
  4. keep the instance iff it is solvable and rating >= min_rating, where
     rating = number of cells on the shortest path *including* S and G
     (= BFS distance + 1). This matches the `rating` column of HRM's CSVs
     (min 110 there, i.e. the "exceeds 110" filter is rating >= 110).

Checked against the released 30x30 data: with p ~ U(0.3, 0.5) and
rating >= 110, the accepted mazes' wall fraction has median 0.371
(HRM train: 0.371, test: 0.372). HRM mazes do NOT have unique shortest
paths (0/1000 do); the label marks one of them. Here it is the BFS-parent
path with fixed neighbour order, so labels are deterministic.

Step 3+4 is plain rejection sampling over (grid, S, G). Instead of drawing
G and usually rejecting, we BFS from S once and accept with probability
k/(M-1) (k = open cells at rating >= min_rating, M = open cells), then pick
G uniformly among those k. That's the same distribution, far fewer grids.
Acceptance is ~2e-5 per (grid, S) at the HRM threshold, so the BFS runs
batched on bit-packed rows; accepted mazes are re-solved with a plain BFS
for the label.

Output: per split, .npy arrays (char2id "# SGo" -> 1..5, `test` = the
validation split) plus HRM-format CSVs (source,question,answer,rating) for
cross-checking with HRM tooling.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import csv
from collections import deque
import json
import os

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata

cli = ArgParser()

CHARSET = "# SGo"
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bfs(open_mask: np.ndarray, start: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    n = open_mask.shape[0]
    dist = np.full((n, n), -1, dtype=np.int32)
    parent = np.full((n, n, 2), -1, dtype=np.int32)
    q: deque[Tuple[int, int]] = deque([start])
    dist[start] = 0

    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and open_mask[ny, nx] and dist[ny, nx] == -1:
                dist[ny, nx] = d + 1
                parent[ny, nx] = (y, x)
                q.append((ny, nx))

    return dist, parent


def _reconstruct_path(
    parent: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    path = [goal]
    cur = goal
    while cur != start:
        py, px = parent[cur]
        if py < 0:
            return None
        cur = (int(py), int(px))
        path.append(cur)
    path.reverse()
    return path


class DataProcessConfig(BaseModel):
    output_dir: str = "data/maze-20x20-hard-1k"
    grid_size: int = 20
    train_samples: int = 1000
    test_samples: int = 1000

    wall_prob_min: float = 0.3
    wall_prob_max: float = 0.5
    # HRM uses 110 at 30x30, which keeps a (grid, S, G) draw with prob.
    # 1.95e-5. At 20x20, 71 gives the same rate (1.85e-5), so the filter
    # cuts the same tail of the generator; rescale this way for other sizes
    # (not by area: 110*400/900 = 49 keeps ~160x more of the tail).
    min_rating: int = 71

    trials_per_chunk: int = 50000
    num_workers: int = 8
    seed: int = 42


def _far_cells(open_bits: np.ndarray, start_bits: np.ndarray, n: int, min_dist: int) -> np.ndarray:
    """Batched BFS on bit-packed grids (one uint64 per row, bit x = column x).

    Returns, per grid, the bitmask of open cells whose shortest-path distance
    from S is >= min_dist.
    """
    row_mask = np.uint64((1 << n) - 1)
    one = np.uint64(1)
    visited = start_bits.copy()
    frontier = start_bits.copy()
    within = None
    level = 0
    while True:
        if level == min_dist - 1:
            within = visited.copy()  # cells at distance <= min_dist - 1
        if not frontier.any():
            break
        nxt = ((frontier << one) | (frontier >> one)) & row_mask
        nxt[:, 1:] |= frontier[:, :-1]
        nxt[:, :-1] |= frontier[:, 1:]
        frontier = nxt & open_bits & ~visited
        visited |= frontier
        level += 1
    if within is None:  # every BFS died before reaching min_dist
        return np.zeros_like(visited)
    return visited & ~within


def _run_chunk(args) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], float]]:
    """Run a fixed number of (grid, S) trials from one seed; return accepts.

    Chunks are keyed by index, not by worker, so the output is identical for
    any num_workers.
    """
    n, p_lo, p_hi, min_rating, trials, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    p = rng.uniform(p_lo, p_hi, size=trials)
    open_mask = rng.random((trials, n, n)) >= p[:, None, None]
    flat = open_mask.reshape(trials, -1)
    m = flat.sum(1)
    # S uniform over open cells: argmax of random priorities on open cells.
    s_idx = np.argmax(np.where(flat, rng.random(flat.shape), -1.0), axis=1)

    weights = np.uint64(1) << np.arange(n, dtype=np.uint64)
    open_bits = (open_mask.astype(np.uint64) * weights).sum(-1, dtype=np.uint64)
    start_bits = np.zeros_like(open_bits)
    sy, sx = s_idx // n, s_idx % n
    start_bits[np.arange(trials), sy] = weights[sx]

    far = _far_cells(open_bits, start_bits, n, min_rating - 1)  # rating = dist + 1
    far_mask = ((far[:, :, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(bool)
    k = far_mask.reshape(trials, -1).sum(1)
    # Keep (grid, S) with prob. k/(M-1): same as drawing G uniformly and
    # rejecting unless it's far enough.
    u = rng.random(trials)
    keep = np.flatnonzero((m >= 2) & (u < k / np.maximum(m - 1, 1)))

    out = []
    for i in keep:
        cand = np.argwhere(far_mask[i])
        goal = tuple(int(v) for v in cand[rng.integers(len(cand))])
        out.append((open_mask[i], (int(sy[i]), int(sx[i])), goal, 1.0 - float(m[i]) / (n * n)))
    return out


def _render(open_mask, start, goal, parent_path) -> Tuple[str, str]:
    grid = np.where(open_mask, " ", "#").astype("<U1")
    grid[start] = "S"
    grid[goal] = "G"
    label = grid.copy()
    for y, x in parent_path[1:-1]:
        label[y, x] = "o"
    return "".join(grid.reshape(-1)), "".join(label.reshape(-1))


def _save_split(split_name: str, rows: List[Tuple[str, str, int]], config: DataProcessConfig, char2id: np.ndarray):
    n = config.grid_size
    save_dir = os.path.join(config.output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    to_ids = lambda s: char2id[np.frombuffer(s.encode(), dtype=np.uint8)]
    num = len(rows)
    results_np = {
        "inputs": np.stack([to_ids(q) for q, _, _ in rows]).astype(np.uint8),
        "labels": np.stack([to_ids(a) for _, a, _ in rows]).astype(np.uint8),
        "group_indices": np.arange(num + 1, dtype=np.int32),
        "puzzle_indices": np.arange(num + 1, dtype=np.int32),
        "puzzle_identifiers": np.zeros(num, dtype=np.int32),
    }
    metadata = PuzzleDatasetMetadata(
        seq_len=n * n,
        vocab_size=len(CHARSET) + 1,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=num,
        mean_puzzle_examples=1,
        total_samples=num,
        sets=["all"],
    )
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results_np.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    with open(os.path.join(config.output_dir, f"{split_name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "question", "answer", "rating"])
        for q, a, r in rows:
            w.writerow(["", q, a, r])

    ratings = np.array([r for _, _, r in rows])
    print(f"{split_name}: {num} mazes, rating min={ratings.min()} "
          f"median={int(np.median(ratings))} max={ratings.max()}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    n = config.grid_size
    if n > 64:
        raise ValueError("grid_size must be <= 64 (one uint64 per row)")
    if not 0 <= config.wall_prob_min <= config.wall_prob_max < 1:
        raise ValueError("need 0 <= wall_prob_min <= wall_prob_max < 1")
    if not 2 <= config.min_rating <= n * n:
        raise ValueError(f"min_rating must be in [2, {n * n}]")

    total = config.train_samples + config.test_samples
    root = np.random.SeedSequence(config.seed)
    rows: List[Tuple[str, str, int]] = []
    seen: set[str] = set()
    wall_fracs: List[float] = []
    chunk_idx = 0

    with ProcessPoolExecutor(config.num_workers) as pool, tqdm(total=total, desc="mazes") as bar:
        while len(rows) < total:
            # One round = num_workers chunks; results are consumed in chunk
            # order so the dataset only depends on seed, not scheduling.
            seeds = root.spawn(config.num_workers)
            jobs = [(n, config.wall_prob_min, config.wall_prob_max, config.min_rating,
                     config.trials_per_chunk, s) for s in seeds]
            for accepted in pool.map(_run_chunk, jobs):
                for open_mask, start, goal, wf in accepted:
                    if len(rows) >= total:
                        break
                    dist, parent = _bfs(open_mask, start)
                    rating = int(dist[goal]) + 1
                    assert rating >= config.min_rating, "bitset BFS disagrees with _bfs"
                    path = _reconstruct_path(parent, start, goal)
                    assert path is not None and len(path) == rating
                    q, a = _render(open_mask, start, goal, path)
                    if q in seen:  # dedupe across train and test
                        continue
                    seen.add(q)
                    rows.append((q, a, rating))
                    wall_fracs.append(wf)
                    bar.update(1)
            chunk_idx += config.num_workers

    print(f"trials run: {chunk_idx * config.trials_per_chunk:,}; "
          f"wall fraction median={np.median(wall_fracs):.3f} "
          f"[5%={np.percentile(wall_fracs, 5):.3f}, 95%={np.percentile(wall_fracs, 95):.3f}]")

    char2id = np.zeros(256, dtype=np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET), dtype=np.uint8) + 1

    os.makedirs(config.output_dir, exist_ok=True)
    _save_split("train", rows[:config.train_samples], config, char2id)
    _save_split("test", rows[config.train_samples:], config, char2id)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    with open(os.path.join(config.output_dir, "generation_config.json"), "w") as f:
        json.dump(config.model_dump(), f, indent=2)


if __name__ == "__main__":
    cli()
