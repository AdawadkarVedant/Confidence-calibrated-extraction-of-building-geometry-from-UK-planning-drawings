"""NSGA-II multi-objective optimizer, written from scratch (Deb et al. 2002).

Steps: sort into fronts -> measure crowding -> pick parents by tournament
-> breed (crossover + mutation) -> keep the best pop_size survivors.
Repeat for N generations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- sorting

def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Partition objective values into fronts. Minimisation throughout."""
    n = F.shape[0]
    # dominates[i, j] is True when i dominates j
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dominates = le & lt

    domination_count = dominates.sum(axis=0)          # how many dominate i
    fronts: list[np.ndarray] = []
    current = np.where(domination_count == 0)[0]
    counts = domination_count.copy()

    while current.size:
        fronts.append(current)
        counts_next = counts.copy()
        for i in current:
            counts_next[dominates[i]] -= 1
        counts_next[current] = -1                     # retire assigned members
        current = np.where(counts_next == 0)[0]
        counts = counts_next

    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Density estimate within a single front."""
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(F[:, k])
        f = F[order, k]
        dist[order[0]] = dist[order[-1]] = np.inf
        span = f[-1] - f[0]
        if span <= 0:
            continue
        dist[order[1:-1]] += (f[2:] - f[:-2]) / span
    return dist


# ------------------------------------------------------------- variation

def sbx_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator,
    eta: float = 15.0, prob: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend two parents into two children (simulated binary crossover)."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    swap = rng.random(p1.shape) < 0.5
    c1 = np.where(swap, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), c1)
    c2 = np.where(swap, 0.5 * ((1 - beta) * p1 + (1 + beta) * p2), c2)
    return c1, c2


def polynomial_mutation(
    x: np.ndarray, rng: np.random.Generator,
    lower: np.ndarray, upper: np.ndarray, eta: float = 20.0,
    prob: float | None = None,
) -> np.ndarray:
    """Randomly nudge some values in x (polynomial mutation)."""
    n = x.size
    prob = 1.0 / n if prob is None else prob
    y = x.copy()
    mask = rng.random(n) < prob
    if not mask.any():
        return y
    span = upper - lower
    delta1 = (x - lower) / np.where(span > 0, span, 1)
    delta2 = (upper - x) / np.where(span > 0, span, 1)
    u = rng.random(n)
    mut_pow = 1.0 / (eta + 1.0)
    dq = np.where(
        u <= 0.5,
        (2 * u + (1 - 2 * u) * (1 - delta1) ** (eta + 1)) ** mut_pow - 1,
        1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta + 1)) ** mut_pow,
    )
    y[mask] = x[mask] + dq[mask] * span[mask]
    return np.clip(y, lower, upper)


def tournament(
    rank: np.ndarray, crowd: np.ndarray, rng: np.random.Generator
) -> int:
    """Pick 2 random individuals, return the better one (lower rank wins)."""
    a, b = rng.integers(0, rank.size, 2)
    if rank[a] != rank[b]:
        return int(a if rank[a] < rank[b] else b)
    return int(a if crowd[a] > crowd[b] else b)


# ------------------------------------------------------------------ loop

@dataclass
class Result:
    """Output of minimise(): the final Pareto front."""

    X: np.ndarray                 # threshold vectors on the front
    F: np.ndarray                 # their objective values (error, 1-coverage, cost)
    history: list[dict]           # stats per generation, for debugging


def minimise(
    evaluate,
    n_var: int,
    lower: np.ndarray | float = 0.0,
    upper: np.ndarray | float = 1.0,
    pop_size: int = 100,
    generations: int = 120,
    seed: int = 0,
    verbose: bool = False,
) -> Result:
    """Run NSGA-II. `evaluate` maps a (pop_size, n_var) array to (pop_size, n_obj)."""
    rng = np.random.default_rng(seed)
    lower = np.full(n_var, lower) if np.isscalar(lower) else np.asarray(lower, float)
    upper = np.full(n_var, upper) if np.isscalar(upper) else np.asarray(upper, float)

    X = rng.uniform(lower, upper, size=(pop_size, n_var))
    F = evaluate(X)
    history: list[dict] = []

    for gen in range(generations):
        fronts = fast_non_dominated_sort(F)
        rank = np.empty(len(X), dtype=int)
        crowd = np.empty(len(X))
        for r, idx in enumerate(fronts):
            rank[idx] = r
            crowd[idx] = crowding_distance(F[idx])

        # offspring
        children = []
        while len(children) < pop_size:
            p1 = X[tournament(rank, crowd, rng)]
            p2 = X[tournament(rank, crowd, rng)]
            c1, c2 = sbx_crossover(p1, p2, rng)
            children.append(polynomial_mutation(np.clip(c1, lower, upper), rng, lower, upper))
            if len(children) < pop_size:
                children.append(polynomial_mutation(np.clip(c2, lower, upper), rng, lower, upper))
        C = np.array(children)
        FC = evaluate(C)

        # elitist (mu + lambda) survival
        X_all = np.vstack([X, C])
        F_all = np.vstack([F, FC])
        fronts_all = fast_non_dominated_sort(F_all)

        survivors: list[int] = []
        for idx in fronts_all:
            if len(survivors) + len(idx) <= pop_size:
                survivors.extend(idx.tolist())
            else:
                d = crowding_distance(F_all[idx])
                keep = idx[np.argsort(-d)][: pop_size - len(survivors)]
                survivors.extend(keep.tolist())
                break
        survivors_arr = np.array(survivors)
        X, F = X_all[survivors_arr], F_all[survivors_arr]

        first = fast_non_dominated_sort(F)[0]
        history.append(
            {
                "generation": gen,
                "front_size": int(first.size),
                "best_error": float(F[first, 0].min()),
                "best_coverage": float(1 - F[first, 1].min()),
            }
        )
        if verbose and gen % 20 == 0:
            h = history[-1]
            print(
                f"gen {gen:3d}  front {h['front_size']:3d}  "
                f"min error {h['best_error']:.3f}  max coverage {h['best_coverage']:.3f}"
            )

    first = fast_non_dominated_sort(F)[0]
    order = np.argsort(F[first, 1])           # sort front by coverage for readability
    idx = first[order]
    return Result(X=X[idx], F=F[idx], history=history)


# ------------------------------------------------------------- indicator

def hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float:
    """Area between the front and a reference point. Bigger = better front."""
    pts = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return float(hv)
