#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

try:
    from pymoo.problems import get_problem
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing optional dependency `pymoo`. Install it with:\n"
        "    python -m pip install -r requirements_baselines.txt"
    ) from exc


PROBLEM_KWARGS: dict[str, dict[str, int]] = {
    "dtlz2": {"n_obj": 2, "n_var": 10},
    "wfg1": {"n_obj": 2, "n_var": 10},
    "wfg2": {"n_obj": 2, "n_var": 10},
    "wfg3": {"n_obj": 2, "n_var": 10},
}

PROBLEM_DESCRIPTIONS = {
    "zdt3": "discontinuous Pareto front",
    "zdt4": "multimodal decision space",
    "mw2": "constrained multi-objective benchmark",
    "welded_beam": "engineering design optimization problem",
    "dtlz2": "scalable DTLZ benchmark with a higher-dimensional decision space and a spherical Pareto front",
    "wfg1": "WFG benchmark with biased transformations and a mixed Pareto-front shape",
    "wfg2": "WFG benchmark with nonseparable transformations and a disconnected Pareto front",
    "wfg3": "WFG benchmark with nonseparable transformations and a degenerate Pareto front",
}


@dataclass(frozen=True)
class RunConfig:
    weights: int
    starts: int
    seed: int
    maxiter: int
    output_dir: Path
    pienn_iter: int
    nc: int
    r_factor: float


@dataclass
class PiennResult:
    subproblem_index: int
    epsilon_f2: float
    best_x: np.ndarray
    best_f: np.ndarray
    constraint_violation: float
    epsilon_violation: float
    scalar_value: float
    runtime_sec: float


def evaluate(problem, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x2d = np.atleast_2d(np.asarray(x, dtype=float))
    if problem.n_ieq_constr > 0:
        f, g = problem.evaluate(x2d, return_values_of=["F", "G"])
        return np.asarray(f, dtype=float), np.asarray(g, dtype=float)
    f = problem.evaluate(x2d, return_values_of=["F"])
    return np.asarray(f, dtype=float), np.zeros((x2d.shape[0], 0))


def nondominated(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    idx = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[idx]


def create_problem(name: str):
    problem_name = name.lower()
    return get_problem(problem_name, **PROBLEM_KWARGS.get(problem_name, {}))


def finite_reference_front(problem) -> np.ndarray:
    pf = np.asarray(problem.pareto_front(), dtype=float)
    if pf.ndim != 2 or pf.shape[1] != 2:
        raise ValueError(f"Only bi-objective reference fronts are supported, got {pf.shape}.")
    pf = pf[np.all(np.isfinite(pf), axis=1)]
    return nondominated(pf)


def select_reference_targets(reference: np.ndarray, count: int) -> np.ndarray:
    """Select reference-front targets with approximately uniform normalized arc length."""
    reference = reference[np.argsort(reference[:, 0])]
    if len(reference) <= count:
        return reference

    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    normalized = (reference - ideal) / span
    segment_lengths = np.sqrt(np.sum(np.diff(normalized, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 1e-12:
        indices = np.linspace(0, len(reference) - 1, count).round().astype(int)
        return reference[indices]

    target_lengths = np.linspace(0.0, cumulative[-1], count)
    indices = np.searchsorted(cumulative, target_lengths, side="left")
    indices = np.clip(indices, 0, len(reference) - 1)
    return reference[indices]


def generate_epsilon_values(reference: np.ndarray, count: int) -> np.ndarray:
    targets = select_reference_targets(reference, count)
    return targets[:, 1]


def problem_violation(g: np.ndarray) -> float:
    return float(np.sum(np.maximum(g[0], 0.0))) if g.size else 0.0


def epsilon_violation(f: np.ndarray, epsilon_f2: float) -> float:
    return max(float(f[0, 1] - epsilon_f2), 0.0)


def total_violation(f: np.ndarray, g: np.ndarray, epsilon_f2: float | None = None) -> float:
    violation = problem_violation(g)
    if epsilon_f2 is not None:
        violation += epsilon_violation(f, epsilon_f2)
    return violation


def epsilon_scalar_value(
    problem,
    x: np.ndarray,
    epsilon_f2: float,
    ideal: np.ndarray,
    span: np.ndarray,
    penalty: float = 1e5,
) -> float:
    f, g = evaluate(problem, x)
    f1n = (float(f[0, 0]) - ideal[0]) / span[0]
    violation = total_violation(f, g, epsilon_f2)
    return float(f1n + penalty * violation * violation)


def local_vprnn_subproblem_surrogate(
    problem,
    x0: np.ndarray,
    epsilon_f2: float,
    ideal: np.ndarray,
    span: np.ndarray,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Numerically solve one PIENN scalar subproblem from one particle.

    The original VPRNN ODE solver in `main.py` is problem-specific. For these
    generic pymoo instances, this routine plays the role of the local constrained
    subproblem solver inside the same PIENN+PIO outer framework.
    """
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    bounds = list(zip(xl, xu))

    def obj(x: np.ndarray) -> float:
        return epsilon_scalar_value(problem, x, epsilon_f2, ideal, span)

    constraints = [{"type": "ineq", "fun": lambda x: epsilon_f2 - evaluate(problem, x)[0][0, 1]}]
    if problem.n_ieq_constr > 0:
        for idx in range(problem.n_ieq_constr):
            constraints.append({"type": "ineq", "fun": lambda x, i=idx: -evaluate(problem, x)[1][0, i]})

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step, clipping to bounds",
            category=RuntimeWarning,
        )
        res = minimize(
            obj,
            np.clip(x0, xl, xu),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-9, "disp": False},
        )
    x = np.clip(res.x if res.x is not None else x0, xl, xu)
    f, g = evaluate(problem, x)
    problem_v = problem_violation(g)
    epsilon_v = epsilon_violation(f, epsilon_f2)
    return x, f[0], problem_v, epsilon_v, obj(x)


def initial_particles(problem, rng: np.random.Generator, swarm: int, subproblem_fraction: float) -> np.ndarray:
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    n_var = len(xl)
    problem_name = problem.__class__.__name__.lower()
    particles: list[np.ndarray] = []

    center = (xl + xu) / 2.0
    zero_inside = np.clip(np.zeros(n_var), xl, xu)
    lower = xl.copy()
    upper = xu.copy()
    structured = center.copy()

    if problem_name.startswith("zdt"):
        structured = zero_inside.copy()
    elif problem_name.startswith("mw") and n_var > 1:
        structured = zero_inside.copy()
        structured[1:] = np.arange(1, n_var, dtype=float) / n_var
    elif problem_name.startswith("wfg"):
        structured = 0.35 * xu

    if n_var > 0:
        x1 = xl[0] + subproblem_fraction * (xu[0] - xl[0])
        center[0] = x1
        zero_inside[0] = x1
        lower[0] = x1
        upper[0] = x1
        structured[0] = x1

    for candidate in (structured, center, zero_inside, lower, upper):
        particles.append(np.clip(candidate, xl, xu))
        if len(particles) >= swarm:
            return np.asarray(particles, dtype=float).T

    while len(particles) < swarm:
        x = rng.uniform(xl, xu)
        if n_var > 0 and rng.random() < 0.5:
            x[0] = xl[0] + subproblem_fraction * (xu[0] - xl[0])
        particles.append(np.clip(x, xl, xu))

    return np.asarray(particles, dtype=float).T


def run_pienn_subproblem(
    problem,
    epsilon_f2: float,
    subproblem_index: int,
    subproblem_count: int,
    cfg: RunConfig,
    rng: np.random.Generator,
    ideal: np.ndarray,
    span: np.ndarray,
) -> PiennResult:
    start_time = time.perf_counter()
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    swarm = cfg.starts
    fraction = subproblem_index / max(subproblem_count - 1, 1)

    y0 = initial_particles(problem, rng, swarm, fraction)
    velocity = np.zeros_like(y0)
    personal_best_x = y0.copy()
    personal_best_val = np.full(swarm, np.inf)
    personal_best_f = np.zeros((swarm, 2))
    personal_best_violation = np.full(swarm, np.inf)
    global_best_x: np.ndarray | None = None
    global_best_f: np.ndarray | None = None
    global_best_val = np.inf
    global_best_violation = np.inf
    global_best_epsilon_violation = np.inf

    for iteration in range(1, cfg.pienn_iter + 1):
        x_bar = np.zeros_like(y0)
        for particle in range(swarm):
            x, f, problem_v, epsilon_v, value = local_vprnn_subproblem_surrogate(
                problem, y0[:, particle], epsilon_f2, ideal, span, cfg.maxiter
            )
            x_bar[:, particle] = x
            if value < personal_best_val[particle]:
                personal_best_val[particle] = value
                personal_best_x[:, particle] = x
                personal_best_f[particle] = f
                personal_best_violation[particle] = problem_v
            if value < global_best_val:
                global_best_val = value
                global_best_x = x.copy()
                global_best_f = f.copy()
                global_best_violation = problem_v
                global_best_epsilon_violation = epsilon_v

        if global_best_x is None:
            raise RuntimeError("PIENN failed to obtain a global best for the current subproblem.")

        if iteration <= cfg.nc:
            for particle in range(swarm):
                velocity[:, particle] = velocity[:, particle] * np.exp(-cfg.r_factor * iteration) + rng.random() * (
                    global_best_x - x_bar[:, particle]
                )
                y0[:, particle] = np.clip(y0[:, particle] + velocity[:, particle], xl, xu)
        else:
            gaps = np.maximum(personal_best_val - global_best_val, 1e-12)
            weights = 1.0 / gaps
            weights[~np.isfinite(weights)] = 0.0
            if float(np.sum(weights)) <= 0.0:
                landmark = global_best_x
            else:
                landmark = (personal_best_x * weights.reshape(1, -1)).sum(axis=1) / float(np.sum(weights))
            for particle in range(swarm):
                y0[:, particle] = np.clip(y0[:, particle] + rng.random() * (landmark - x_bar[:, particle]), xl, xu)

        if global_best_violation <= 1e-8 and np.std(personal_best_val[np.isfinite(personal_best_val)]) < 1e-8:
            break

    assert global_best_x is not None and global_best_f is not None
    runtime = time.perf_counter() - start_time
    return PiennResult(
        subproblem_index=subproblem_index,
        epsilon_f2=float(epsilon_f2),
        best_x=global_best_x,
        best_f=global_best_f,
        constraint_violation=float(global_best_violation),
        epsilon_violation=float(global_best_epsilon_violation),
        scalar_value=float(global_best_val),
        runtime_sec=runtime,
    )


def run_instance(name: str, cfg: RunConfig) -> None:
    name = name.lower()
    problem = create_problem(name)
    if problem.n_obj != 2:
        raise ValueError(f"{name} is not bi-objective and is skipped.")

    description = PROBLEM_DESCRIPTIONS.get(name, "additional benchmark problem")
    reference = finite_reference_front(problem)
    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    epsilons = generate_epsilon_values(reference, cfg.weights)
    name_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))
    rng = np.random.default_rng(cfg.seed + name_seed)

    print(
        f"{name}: running {len(epsilons)} epsilon subproblems "
        f"(starts={cfg.starts}, local maxiter={cfg.maxiter}, pienn_iter={cfg.pienn_iter})",
        flush=True,
    )
    results: list[PiennResult] = []
    progress_step = max(1, len(epsilons) // 10)
    instance_start = time.perf_counter()
    for idx, epsilon in enumerate(epsilons):
        result = run_pienn_subproblem(problem, epsilon, idx, len(epsilons), cfg, rng, ideal, span)
        results.append(result)
        completed = idx + 1
        if completed == 1 or completed == len(epsilons) or completed % progress_step == 0:
            elapsed = time.perf_counter() - instance_start
            print(
                f"{name}: completed {completed}/{len(epsilons)} "
                f"(last={result.runtime_sec:.2f}s, elapsed={elapsed:.1f}s)",
                flush=True,
            )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.output_dir / f"{name}_pienn_solutions.csv"
    max_dim = max(len(item.best_x) for item in results)
    fieldnames = [
        "problem",
        "subproblem_index",
        "epsilon_f2",
        "f1",
        "f2",
        "constraint_violation",
        "epsilon_violation",
        "scalar_value",
        "runtime_sec",
        "source",
    ] + [f"x{j}" for j in range(1, max_dim + 1)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row: dict[str, float | str | int] = {
                "problem": name,
                "subproblem_index": item.subproblem_index,
                "epsilon_f2": item.epsilon_f2,
                "f1": float(item.best_f[0]),
                "f2": float(item.best_f[1]),
                "constraint_violation": item.constraint_violation,
                "epsilon_violation": item.epsilon_violation,
                "scalar_value": item.scalar_value,
                "runtime_sec": item.runtime_sec,
                "source": "PIENN_epsilon_PIO",
            }
            for j, value in enumerate(item.best_x, start=1):
                row[f"x{j}"] = float(value)
            writer.writerow(row)

    approx = nondominated(np.asarray([item.best_f for item in results], dtype=float))
    approx = approx[np.argsort(approx[:, 0])]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(reference[:, 0], reference[:, 1], s=10, color="0.72", label="Reference PF")
    ax.plot(approx[:, 0], approx[:, 1], "r*", markersize=7, label="PIENN solutions")
    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")
    ax.set_title(f"{name.replace('_', ' ').upper()} ({description})")
    ax.grid(True, color="0.85")
    ax.legend(framealpha=1.0)
    fig.tight_layout()
    fig_path = cfg.output_dir / f"{name}_pareto.png"
    eps_path = fig_path.with_suffix(".eps")
    fig.savefig(fig_path, dpi=300)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)

    feasible = sum(item.constraint_violation <= 1e-6 for item in results)
    print(
        f"{name}: {description}; saved {csv_path}, {fig_path}, and {eps_path} "
        f"(PIENN, feasible {feasible}/{len(results)})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIENN on additional benchmark and engineering test instances.")
    parser.add_argument(
        "--problems",
        default="zdt3,zdt4,mw2,welded_beam,dtlz2,wfg3",
        help="Comma-separated pymoo problem names.",
    )
    parser.add_argument("--weights", type=int, default=31, help="Number of epsilon subproblems / Pareto candidates.")
    parser.add_argument("--starts", type=int, default=4, help="PIENN swarm size / number of particles.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum local subproblem-solver iterations.")
    parser.add_argument("--pienn-iter", type=int, default=3, help="Outer PIENN-PIO iterations for each epsilon subproblem.")
    parser.add_argument("--nc", type=int, default=2, help="PIENN map/landmark operator switch iteration.")
    parser.add_argument("--r-factor", type=float, default=0.3, help="PIENN map-operator decay factor.")
    parser.add_argument("--output-dir", type=Path, default=Path("additional_outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(args.weights, args.starts, args.seed, args.maxiter, args.output_dir, args.pienn_iter, args.nc, args.r_factor)
    for name in [item.strip() for item in args.problems.split(",") if item.strip()]:
        run_instance(name, cfg)


if __name__ == "__main__":
    main()

