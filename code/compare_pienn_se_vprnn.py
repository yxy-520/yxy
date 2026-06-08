#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from additional_test_instances import (
    RunConfig,
    create_problem,
    epsilon_scalar_value,
    epsilon_violation,
    evaluate,
    finite_reference_front,
    generate_epsilon_values,
    initial_particles,
    local_vprnn_subproblem_surrogate,
    nondominated,
    problem_violation,
    run_pienn_subproblem,
)

try:
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.util.ref_dirs import get_reference_directions
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing optional dependency `pymoo`. Install it with:\n"
        "    python -m pip install -r requirements_baselines.txt"
    ) from exc


DEFAULT_PROBLEMS = "zdt3,zdt4,dtlz2,wfg3"
METHOD_ORDER = ["PIENN", "SE-VPRNN", "DCSE-VPNN", "NSGA-II", "MOEA/D", "MO-CMA-ES"]


@dataclass(frozen=True)
class BaselineConfig:
    candidates: int
    starts: int
    seed: int
    maxiter: int
    pienn_iter: int
    nc: int
    r_factor: float
    c0: float
    c1: float
    c2: float
    mutation_threshold: float
    n_gen: int
    pop_size: int
    cma_maxfevals: int
    cma_sigma: float
    cma_pop_size: int


@dataclass
class SetResult:
    problem: str
    method: str
    points: np.ndarray
    runtime_sec: float
    hv: float = np.nan
    igd: float = np.nan
    gd: float = np.nan
    spread: float = np.nan
    spacing: float = np.nan
    num_points: int = 0


class ScalarizedEpsilonProblem(ElementwiseProblem):
    def __init__(self, problem, epsilon_f2: float, ideal: np.ndarray, span: np.ndarray):
        super().__init__(n_var=problem.n_var, n_obj=1, xl=problem.xl, xu=problem.xu)
        self.problem = problem
        self.epsilon_f2 = float(epsilon_f2)
        self.ideal = ideal
        self.span = span

    def _evaluate(self, x, out, *args, **kwargs):
        f, g = evaluate(self.problem, np.asarray(x, dtype=float))
        f1n = (float(f[0, 0]) - self.ideal[0]) / self.span[0]
        epsilon_v = max(float(f[0, 1]) - self.epsilon_f2, 0.0)
        constraint_v = float(np.sum(np.maximum(g[0], 0.0))) if g.size else 0.0
        out["F"] = f1n + 1e5 * (epsilon_v + constraint_v) ** 2


def sorted_nd(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = np.atleast_2d(points)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    points = nondominated(points)
    return points[np.argsort(points[:, 0])]


def normalize(points: np.ndarray, ideal: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float) - ideal) / span


def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    points = sorted_nd(points)
    if len(points) == 0:
        return 0.0
    points = points[np.argsort(points[:, 0])]
    hv = 0.0
    prev_f2 = float(ref_point[1])
    for f1, f2 in points:
        width = max(float(ref_point[0]) - float(f1), 0.0)
        height = max(prev_f2 - float(f2), 0.0)
        hv += width * height
        prev_f2 = min(prev_f2, float(f2))
    return float(hv)


def mean_min_distance(source: np.ndarray, target: np.ndarray) -> float:
    if len(source) == 0 or len(target) == 0:
        return float("inf")
    diff = source[:, None, :] - target[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return float(np.mean(np.min(distances, axis=1)))


def spread_delta(points: np.ndarray, reference: np.ndarray) -> float:
    points = sorted_nd(points)
    reference = reference[np.argsort(reference[:, 0])]
    if len(points) < 2 or len(reference) < 2:
        return float("inf")
    consecutive = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    d_bar = float(np.mean(consecutive))
    if d_bar <= 1e-12:
        return float("inf")
    d_first = float(np.linalg.norm(points[0] - reference[0]))
    d_last = float(np.linalg.norm(points[-1] - reference[-1]))
    numerator = d_first + d_last + float(np.sum(np.abs(consecutive - d_bar)))
    denominator = d_first + d_last + (len(points) - 1) * d_bar
    return float(numerator / denominator) if denominator > 0 else float("inf")


def spacing(points: np.ndarray) -> float:
    points = sorted_nd(points)
    if len(points) < 2:
        return float("inf")
    diff = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)
    np.fill_diagonal(diff, np.inf)
    nearest = np.min(diff, axis=1)
    return float(np.sqrt(np.sum((nearest - np.mean(nearest)) ** 2) / max(len(points) - 1, 1)))


def compute_metrics(result: SetResult, reference: np.ndarray) -> None:
    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    approx = sorted_nd(result.points)
    ref_n = normalize(reference, ideal, span)
    approx_n = normalize(approx, ideal, span) if len(approx) else approx.reshape(0, 2)
    result.points = approx
    result.num_points = len(approx)
    result.hv = hypervolume_2d(approx_n, np.array([1.1, 1.1]))
    result.igd = mean_min_distance(ref_n, approx_n)
    result.gd = mean_min_distance(approx_n, ref_n)
    result.spread = spread_delta(approx_n, ref_n)
    result.spacing = spacing(approx_n)


def run_pienn(problem_name: str, cfg: BaselineConfig) -> SetResult:
    problem = create_problem(problem_name)
    reference = finite_reference_front(problem)
    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    epsilons = generate_epsilon_values(reference, cfg.candidates)
    run_cfg = RunConfig(cfg.candidates, cfg.starts, cfg.seed, cfg.maxiter, Path("."), cfg.pienn_iter, cfg.nc, cfg.r_factor)
    rng = np.random.default_rng(cfg.seed + 101)
    start = time.perf_counter()
    results = [
        run_pienn_subproblem(problem, epsilon, idx, len(epsilons), run_cfg, rng, ideal, span)
        for idx, epsilon in enumerate(epsilons)
    ]
    runtime = time.perf_counter() - start
    points = np.asarray([item.best_f for item in results if item.constraint_violation <= 1e-6], dtype=float)
    result = SetResult(problem_name, "PIENN", points, runtime)
    compute_metrics(result, reference)
    return result


def run_se_vprnn_subproblem(problem, epsilon_f2: float, subproblem_index: int, subproblem_count: int, cfg: BaselineConfig, rng, ideal, span):
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    fraction = subproblem_index / max(subproblem_count - 1, 1)
    y0 = initial_particles(problem, rng, cfg.starts, fraction)
    velocity = np.zeros_like(y0)
    personal_best_x = y0.copy()
    personal_best_val = np.full(cfg.starts, np.inf)
    global_best_x = None
    global_best_f = None
    global_best_val = np.inf
    global_best_violation = np.inf
    global_best_epsilon_violation = np.inf

    for iteration in range(1, cfg.pienn_iter + 1):
        x_bar = np.zeros_like(y0)
        values = np.full(cfg.starts, np.inf)
        for particle in range(cfg.starts):
            x, f, problem_v, epsilon_v, value = local_vprnn_subproblem_surrogate(
                problem, y0[:, particle], epsilon_f2, ideal, span, cfg.maxiter
            )
            x_bar[:, particle] = x
            values[particle] = value
            if value < personal_best_val[particle]:
                personal_best_val[particle] = value
                personal_best_x[:, particle] = x
            if value < global_best_val:
                global_best_val = value
                global_best_x = x.copy()
                global_best_f = f.copy()
                global_best_violation = problem_v
                global_best_epsilon_violation = epsilon_v

        if global_best_x is None or global_best_f is None:
            raise RuntimeError("SE-VPRNN failed to obtain a valid subproblem solution.")

        for particle in range(cfg.starts):
            velocity[:, particle] = (
                cfg.c0 * velocity[:, particle]
                + cfg.c1 * rng.random() * (personal_best_x[:, particle] - x_bar[:, particle])
                + cfg.c2 * rng.random() * (global_best_x - x_bar[:, particle])
            )
            y0[:, particle] = np.clip(y0[:, particle] + velocity[:, particle], xl, xu)

        normalized_dist = np.linalg.norm((y0 - global_best_x.reshape(-1, 1)) / np.maximum((xu - xl).reshape(-1, 1), 1e-12), axis=0)
        if float(np.mean(normalized_dist)) < cfg.mutation_threshold:
            a = np.exp(10.0 * iteration / max(cfg.pienn_iter, 1))
            psi = -2.5 + 5.0 * rng.random()
            eta = (1.0 / np.sqrt(a)) * np.exp(-(psi**2) / 2.0) * np.cos(5.0 * psi)
            target = xu if eta > 0 else xl
            for particle in range(cfg.starts):
                y0[:, particle] = np.clip(y0[:, particle] + eta * (target - y0[:, particle]), xl, xu)

    return global_best_f, float(global_best_violation), float(global_best_epsilon_violation)


def run_se_vprnn(problem_name: str, cfg: BaselineConfig) -> SetResult:
    problem = create_problem(problem_name)
    reference = finite_reference_front(problem)
    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    epsilons = generate_epsilon_values(reference, cfg.candidates)
    rng = np.random.default_rng(cfg.seed + 202)
    start = time.perf_counter()
    rows = [run_se_vprnn_subproblem(problem, epsilon, idx, len(epsilons), cfg, rng, ideal, span) for idx, epsilon in enumerate(epsilons)]
    runtime = time.perf_counter() - start
    points = np.asarray([f for f, v, _ in rows if v <= 1e-6], dtype=float)
    result = SetResult(problem_name, "SE-VPRNN", points, runtime)
    compute_metrics(result, reference)
    return result


def dcse_initial_particles(problem, rng: np.random.Generator, swarm: int, subproblem_fraction: float) -> np.ndarray:
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    n_var = len(xl)
    beta_samples = rng.beta(0.8, 0.8, size=(n_var, swarm))
    particles = xl.reshape(-1, 1) + beta_samples * (xu - xl).reshape(-1, 1)
    if n_var > 0 and swarm > 0:
        particles[0, 0] = xl[0] + subproblem_fraction * (xu[0] - xl[0])
    return np.clip(particles, xl.reshape(-1, 1), xu.reshape(-1, 1))


def evaluate_dcse_candidate(problem, x: np.ndarray, epsilon_f2: float, ideal: np.ndarray, span: np.ndarray):
    f, g = evaluate(problem, x)
    problem_v = problem_violation(g)
    epsilon_v = epsilon_violation(f, epsilon_f2)
    value = epsilon_scalar_value(problem, x, epsilon_f2, ideal, span)
    return f[0], problem_v, epsilon_v, value


def run_dcse_vpnn_subproblem(problem, epsilon_f2: float, subproblem_index: int, subproblem_count: int, cfg: BaselineConfig, rng, ideal, span):
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    fraction = subproblem_index / max(subproblem_count - 1, 1)
    y0 = dcse_initial_particles(problem, rng, cfg.starts, fraction)
    velocity_span = np.maximum((xu - xl).reshape(-1, 1), 1e-12)
    velocity = rng.uniform(-0.1, 0.1, size=y0.shape) * velocity_span
    personal_best_x = y0.copy()
    personal_best_val = np.full(cfg.starts, np.inf)
    global_best_x = None
    global_best_f = None
    global_best_val = np.inf
    global_best_violation = np.inf
    global_best_epsilon_violation = np.inf

    def update_best(candidate: np.ndarray, particle: int | None = None):
        nonlocal global_best_x, global_best_f, global_best_val, global_best_violation, global_best_epsilon_violation
        x = np.clip(candidate, xl, xu)
        f, problem_v, epsilon_v, value = evaluate_dcse_candidate(problem, x, epsilon_f2, ideal, span)
        if particle is not None and value < personal_best_val[particle]:
            personal_best_val[particle] = value
            personal_best_x[:, particle] = x
        if value < global_best_val:
            global_best_val = value
            global_best_x = x.copy()
            global_best_f = f.copy()
            global_best_violation = problem_v
            global_best_epsilon_violation = epsilon_v
        return x, f, problem_v, epsilon_v, value

    for _ in range(1, cfg.pienn_iter + 1):
        for particle in range(cfg.starts):
            x, f, problem_v, epsilon_v, value = local_vprnn_subproblem_surrogate(
                problem, y0[:, particle], epsilon_f2, ideal, span, cfg.maxiter
            )
            if value < personal_best_val[particle]:
                personal_best_val[particle] = value
                personal_best_x[:, particle] = x
            if value < global_best_val:
                global_best_val = value
                global_best_x = x.copy()
                global_best_f = f.copy()
                global_best_violation = problem_v
                global_best_epsilon_violation = epsilon_v

        if global_best_x is None or global_best_f is None:
            raise RuntimeError("DCSE-VPNN failed to obtain a valid subproblem solution.")

        current_positions = np.zeros_like(y0)
        for particle in range(cfg.starts):
            base = y0[:, particle]
            v1 = cfg.c0 * velocity[:, particle]
            x_a, *_ = update_best(base + v1, particle)
            v2 = cfg.c1 * rng.random() * (personal_best_x[:, particle] - base)
            x_b, *_ = update_best(x_a + v2, particle)
            v3 = cfg.c2 * rng.random() * (global_best_x - base)
            x_c, *_ = update_best(x_b + v3, particle)
            velocity[:, particle] = v1 + v2 + v3
            y0[:, particle] = x_c
            current_positions[:, particle] = x_c

        best_idx = int(np.argmin(personal_best_val))
        update_best(personal_best_x[:, best_idx])
        update_best(np.mean(personal_best_x, axis=1))
        update_best(np.mean(current_positions, axis=1))

        finite_vals = personal_best_val[np.isfinite(personal_best_val)]
        if global_best_violation <= 1e-8 and len(finite_vals) and np.std(finite_vals) < 1e-8:
            break

    return global_best_f, float(global_best_violation), float(global_best_epsilon_violation)


def run_dcse_vpnn(problem_name: str, cfg: BaselineConfig) -> SetResult:
    problem = create_problem(problem_name)
    reference = finite_reference_front(problem)
    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    epsilons = generate_epsilon_values(reference, cfg.candidates)
    rng = np.random.default_rng(cfg.seed + 252)
    start = time.perf_counter()
    rows = [run_dcse_vpnn_subproblem(problem, epsilon, idx, len(epsilons), cfg, rng, ideal, span) for idx, epsilon in enumerate(epsilons)]
    runtime = time.perf_counter() - start
    points = np.asarray([f for f, v, _ in rows if v <= 1e-6], dtype=float)
    result = SetResult(problem_name, "DCSE-VPNN", points, runtime)
    compute_metrics(result, reference)
    return result


def run_nsga2(problem_name: str, cfg: BaselineConfig) -> SetResult:
    problem = create_problem(problem_name)
    reference = finite_reference_front(problem)
    algorithm = NSGA2(pop_size=cfg.pop_size, eliminate_duplicates=True)
    start = time.perf_counter()
    res = pymoo_minimize(problem, algorithm, ("n_gen", cfg.n_gen), seed=cfg.seed + 303, verbose=False)
    runtime = time.perf_counter() - start
    points = np.asarray(res.F, dtype=float)
    result = SetResult(problem_name, "NSGA-II", points, runtime)
    compute_metrics(result, reference)
    return result


def run_moead(problem_name: str, cfg: BaselineConfig) -> SetResult:
    problem = create_problem(problem_name)
    if problem.has_constraints():
        raise ValueError("The selected pymoo MOEA/D implementation does not support constrained problems.")
    reference = finite_reference_front(problem)
    ref_dirs = get_reference_directions("uniform", problem.n_obj, n_partitions=max(cfg.pop_size - 1, 1))
    algorithm = MOEAD(ref_dirs=ref_dirs, n_neighbors=min(20, len(ref_dirs)), prob_neighbor_mating=0.9)
    start = time.perf_counter()
    res = pymoo_minimize(problem, algorithm, ("n_gen", cfg.n_gen), seed=cfg.seed + 404, verbose=False)
    runtime = time.perf_counter() - start
    points = np.asarray(res.F, dtype=float)
    result = SetResult(problem_name, "MOEA/D", points, runtime)
    compute_metrics(result, reference)
    return result


def run_mocmaes(problem_name: str, cfg: BaselineConfig) -> SetResult:
    problem = create_problem(problem_name)
    reference = finite_reference_front(problem)
    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    epsilons = generate_epsilon_values(reference, cfg.candidates)
    start = time.perf_counter()
    points = []
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    for idx, epsilon in enumerate(epsilons):
        fraction = idx / max(len(epsilons) - 1, 1)
        x0 = (xl + xu) / 2.0
        if len(x0):
            x0[0] = xl[0] + fraction * (xu[0] - xl[0])
        scalar_problem = ScalarizedEpsilonProblem(problem, float(epsilon), ideal, span)
        algorithm = CMAES(x0=np.clip(x0, xl, xu), sigma=cfg.cma_sigma, maxfevals=cfg.cma_maxfevals, pop_size=cfg.cma_pop_size)
        res = pymoo_minimize(scalar_problem, algorithm, ("n_eval", cfg.cma_maxfevals), seed=cfg.seed + 500 + idx, verbose=False)
        if res.X is not None:
            f, g = evaluate(problem, np.asarray(res.X, dtype=float))
            constraint_v = float(np.sum(np.maximum(g[0], 0.0))) if g.size else 0.0
            if constraint_v <= 1e-6:
                points.append(f[0])
    runtime = time.perf_counter() - start
    result = SetResult(problem_name, "MO-CMA-ES", np.asarray(points, dtype=float), runtime)
    compute_metrics(result, reference)
    return result


def write_solution_points(results: list[SetResult], output_dir: Path) -> None:
    with (output_dir / "baseline_solution_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "point_index", "f1", "f2"])
        for result in results:
            for idx, point in enumerate(result.points):
                writer.writerow([result.problem, result.method, idx, f"{point[0]:.12g}", f"{point[1]:.12g}"])


def write_metrics(results: list[SetResult], output_dir: Path) -> None:
    with (output_dir / "baseline_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "method", "HV", "IGD", "GD", "Spread", "Spacing", "Runtime_sec", "Num_ND"])
        for item in results:
            writer.writerow(
                [
                    item.problem,
                    item.method,
                    f"{item.hv:.12g}",
                    f"{item.igd:.12g}",
                    f"{item.gd:.12g}",
                    f"{item.spread:.12g}",
                    f"{item.spacing:.12g}",
                    f"{item.runtime_sec:.12g}",
                    item.num_points,
                ]
            )


def write_latex_table(results: list[SetResult], output_dir: Path) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Performance comparison among PIENN, SE-VPRNN, DCSE-VPNN, NSGA-II, MOEA/D, and MO-CMA-ES on selected unconstrained benchmark problems.}",
        r"\label{tab:baseline_comparison}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        "Problem & Method & HV $\\uparrow$ & IGD $\\downarrow$ & GD $\\downarrow$ & Spread $\\downarrow$ & Spacing $\\downarrow$ & Time (s) " + r"\\",
        r"\midrule",
    ]
    by_problem = {problem: [r for r in results if r.problem == problem] for problem in sorted({r.problem for r in results})}
    for p_idx, problem in enumerate(by_problem):
        subset = sorted(by_problem[problem], key=lambda r: METHOD_ORDER.index(r.method) if r.method in METHOD_ORDER else 99)
        for idx, item in enumerate(subset):
            label = problem.upper().replace("_", "-") if idx == 0 else ""
            lines.append(
                f"{label} & {item.method} & {item.hv:.4g} & {item.igd:.4g} & {item.gd:.4g} & "
                f"{item.spread:.4g} & {item.spacing:.4g} & {item.runtime_sec:.2f} " + r"\\"
            )
        if p_idx != len(by_problem) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output_dir / "baseline_comparison_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_complexity_notes(output_dir: Path) -> None:
    text = r"""# Complexity notes for baseline comparison

Let $N_\epsilon$ be the number of scalarized subproblems, $N_p$ the PIENN/SE-VPRNN/DCSE-VPNN swarm size, $K$ the outer swarm iterations, and $C_{\rm local}$ the cost of one local VPRNN/subproblem refinement.

- PIENN: $O(N_\epsilon N_p K C_{\rm local})$ plus the PIO map/landmark update cost.
- SE-VPRNN: $O(N_\epsilon N_p K C_{\rm local})$ plus PSO-style velocity update and wavelet mutation.
- DCSE-VPNN: $O(N_\epsilon N_p K C_{\rm local})$ plus segmented double-center PSO updates and GCP/SCP center evaluations.
- NSGA-II: approximately $O(G P^2 M)$ for non-dominated sorting and crowding operations, where $G$ is the number of generations, $P$ is the population size, and $M$ is the number of objectives.
- MOEA/D: approximately $O(G P T M)$, where $T$ is the neighborhood size, excluding objective-function evaluation cost.
- MO-CMA-ES: in this script, CMA-ES is applied to the same $N_\epsilon$ scalarized subproblems, giving approximately $O(N_\epsilon E_{\rm CMA} C_f)$ plus covariance-distribution update costs, where $E_{\rm CMA}$ is the per-subproblem evaluation budget.
"""
    (output_dir / "complexity_notes.md").write_text(text, encoding="utf-8")


def write_problem_selection_notes(problems: list[str], output_dir: Path) -> None:
    text = f"""# Benchmark selection notes

Selected problems: {', '.join(problems)}.

The default comparison uses ZDT3, ZDT4, DTLZ2, and WFG3 because they are unconstrained continuous benchmarks and can be handled fairly by PIENN, SE-VPRNN, DCSE-VPNN, scalarized MO-CMA-ES, NSGA-II, and the installed pymoo MOEA/D implementation. ZDT3 tests a discontinuous Pareto front, ZDT4 tests a multimodal decision space, DTLZ2 tests a scalable higher-dimensional continuous problem with a spherical Pareto front, and WFG3 tests nonseparable transformations with a degenerate Pareto front.

MW2 and welded_beam are not used in the default baseline comparison because the installed pymoo MOEA/D implementation explicitly does not support constrained problems. Including them would require additional penalty or constraint-handling wrappers for MOEA/D and scalarized MO-CMA-ES, which would introduce another source of unfairness. They remain useful additional PIENN validation cases and are reported as Pareto-front figures.
"""
    (output_dir / "benchmark_selection_notes.md").write_text(text, encoding="utf-8")


def plot_results(results: list[SetResult], output_dir: Path) -> None:
    markers = {"PIENN": "*", "SE-VPRNN": "s", "DCSE-VPNN": "P", "NSGA-II": "o", "MOEA/D": "^", "MO-CMA-ES": "D"}
    colors = {
        "PIENN": "tab:orange",
        "SE-VPRNN": "tab:blue",
        "DCSE-VPNN": "tab:red",
        "NSGA-II": "tab:green",
        "MOEA/D": "tab:purple",
        "MO-CMA-ES": "tab:brown",
    }
    for problem in sorted({item.problem for item in results}):
        reference = finite_reference_front(create_problem(problem))
        subset = [item for item in results if item.problem == problem]
        for suffix in ("png", "eps"):
            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            ax.scatter(reference[:, 0], reference[:, 1], s=10, color="0.72", label="Reference PF")
            plot_order = sorted(
                subset,
                key=lambda r: (r.method == "PIENN", METHOD_ORDER.index(r.method) if r.method in METHOD_ORDER else 99),
            )
            for item in plot_order:
                if len(item.points):
                    ax.plot(
                        item.points[:, 0],
                        item.points[:, 1],
                        linestyle="",
                        marker=markers.get(item.method, "o"),
                        color=colors.get(item.method),
                        markeredgecolor="black" if item.method == "PIENN" else None,
                        markeredgewidth=0.8 if item.method == "PIENN" else 0.0,
                        markersize=8 if item.method == "PIENN" else 5,
                        zorder=10 if item.method == "PIENN" else 3,
                        label=item.method,
                    )
            ax.set_xlabel(r"$f_1$")
            ax.set_ylabel(r"$f_2$")
            ax.set_title(problem.upper().replace("_", "-"))
            ax.grid(True, color="0.85")
            ax.legend(fontsize=8, framealpha=1.0)
            fig.tight_layout()
            fig.savefig(output_dir / f"{problem}_baseline_pareto.{suffix}", dpi=300 if suffix == "png" else None, format=suffix)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PIENN with SE-VPRNN, DCSE-VPNN, NSGA-II, MOEA/D, and MO-CMA-ES.")
    parser.add_argument("--problems", default=DEFAULT_PROBLEMS, help="Comma-separated benchmark names. Default excludes constrained cases for MOEA/D fairness.")
    parser.add_argument(
        "--candidates",
        type=int,
        default=51,
        help="Number of epsilon subproblems / target Pareto candidates for PIENN, SE-VPRNN, DCSE-VPNN, and MO-CMA-ES. Use 51 or more for publication-quality Pareto plots.",
    )
    parser.add_argument("--starts", type=int, default=6, help="PIENN/SE-VPRNN/DCSE-VPNN swarm size.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--maxiter", type=int, default=150, help="Maximum local subproblem-solver iterations.")
    parser.add_argument("--pienn-iter", type=int, default=6, help="Outer PIENN/SE-VPRNN/DCSE-VPNN iterations for each epsilon subproblem.")
    parser.add_argument("--nc", type=int, default=3, help="PIENN map/landmark switch iteration.")
    parser.add_argument("--r-factor", type=float, default=0.3)
    parser.add_argument("--c0", type=float, default=0.2)
    parser.add_argument("--c1", type=float, default=0.4)
    parser.add_argument("--c2", type=float, default=0.7)
    parser.add_argument("--mutation-threshold", type=float, default=0.05)
    parser.add_argument("--n-gen", type=int, default=120, help="Number of generations for NSGA-II and MOEA/D.")
    parser.add_argument("--pop-size", type=int, default=80, help="Population size for NSGA-II and reference-direction count for MOEA/D.")
    parser.add_argument("--cma-maxfevals", type=int, default=200, help="CMA-ES evaluation budget per scalarized subproblem.")
    parser.add_argument("--cma-sigma", type=float, default=0.25)
    parser.add_argument("--cma-pop-size", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_outputs"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BaselineConfig(
        candidates=args.candidates,
        starts=args.starts,
        seed=args.seed,
        maxiter=args.maxiter,
        pienn_iter=args.pienn_iter,
        nc=args.nc,
        r_factor=args.r_factor,
        c0=args.c0,
        c1=args.c1,
        c2=args.c2,
        mutation_threshold=args.mutation_threshold,
        n_gen=args.n_gen,
        pop_size=args.pop_size,
        cma_maxfevals=args.cma_maxfevals,
        cma_sigma=args.cma_sigma,
        cma_pop_size=args.cma_pop_size,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = [item.strip().lower() for item in args.problems.split(",") if item.strip()]

    if cfg.candidates < 31:
        print(
            "Warning: --candidates is smaller than 31, so PIENN/SE-VPRNN/DCSE-VPNN/MO-CMA-ES "
            "will have relatively few Pareto candidates in the plots. For paper figures, "
            "use --candidates 51 or --candidates 81."
        )

    results: list[SetResult] = []
    runners = [run_pienn, run_se_vprnn, run_dcse_vpnn, run_nsga2, run_moead, run_mocmaes]
    for problem in problems:
        print(f"\nRunning baselines on {problem}...")
        for runner in runners:
            result = runner(problem, cfg)
            results.append(result)
            print(
                f"  {result.method:10s}: HV={result.hv:.4g}, IGD={result.igd:.4g}, "
                f"GD={result.gd:.4g}, Spread={result.spread:.4g}, "
                f"Spacing={result.spacing:.4g}, Time={result.runtime_sec:.2f}s"
            )

    write_solution_points(results, output_dir)
    write_metrics(results, output_dir)
    write_latex_table(results, output_dir)
    write_complexity_notes(output_dir)
    write_problem_selection_notes(problems, output_dir)
    if not args.no_plots:
        plot_results(results, output_dir)
    print(f"\nSaved baseline comparison outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

