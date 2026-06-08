
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from additional_test_instances import RunConfig, finite_reference_front, generate_epsilon_values, run_pienn_subproblem


BASE_OUTPUT_DIR = Path("sensitivity_outputs")


def nondominated(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    idx = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[idx]


def mean_distance_to_reference(points: np.ndarray, reference: np.ndarray) -> float:
    if len(points) == 0:
        return float("inf")
    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    pn = (points - ideal) / span
    rn = (reference - ideal) / span
    distances = np.sqrt(np.sum((pn[:, None, :] - rn[None, :, :]) ** 2, axis=2))
    return float(np.mean(np.min(distances, axis=1)))


def hypervolume_2d(points: np.ndarray, reference_point: np.ndarray) -> float:
    points = nondominated(points)
    points = points[np.argsort(points[:, 0])]
    hv = 0.0
    prev_f2 = float(reference_point[1])
    for f1, f2 in points:
        width = max(float(reference_point[0] - f1), 0.0)
        height = max(prev_f2 - float(f2), 0.0)
        hv += width * height
        prev_f2 = min(prev_f2, float(f2))
    return float(hv)


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(float(item.strip())) for item in text.split(",") if item.strip()]


def run_case(problem_name: str, cfg: RunConfig) -> tuple[np.ndarray, dict[str, float | int]]:
    problem = get_problem(problem_name)
    reference = finite_reference_front(problem)
    ideal = np.min(reference, axis=0)
    span = np.maximum(np.max(reference, axis=0) - ideal, 1e-12)
    epsilons = generate_epsilon_values(reference, cfg.weights)
    name_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(problem_name))
    rng = np.random.default_rng(cfg.seed + name_seed)

    start = time.perf_counter()
    results = [
        run_pienn_subproblem(problem, epsilon, idx, len(epsilons), cfg, rng, ideal, span)
        for idx, epsilon in enumerate(epsilons)
    ]
    runtime = time.perf_counter() - start
    points = np.asarray([item.best_f for item in results], dtype=float)
    approx = nondominated(points)
    ref_point = np.max(reference, axis=0) + 0.1 * np.maximum(np.ptp(reference, axis=0), 1e-12)
    metrics: dict[str, float | int] = {
        "num_solutions": int(len(points)),
        "num_nd": int(len(approx)),
        "num_feasible": int(sum(item.constraint_violation <= 1e-6 for item in results)),
        "max_constraint_violation": float(max(item.constraint_violation for item in results)),
        "mean_distance_to_ref": mean_distance_to_reference(approx, reference),
        "hypervolume": hypervolume_2d(approx, ref_point),
        "runtime_sec": float(runtime),
    }
    return approx, metrics


def add_case(rows: list[dict[str, float | int | str]], problem: str, parameter: str, value: float | int, cfg: RunConfig) -> np.ndarray:
    approx, metrics = run_case(problem, cfg)
    row: dict[str, float | int | str] = {
        "problem": problem,
        "parameter": parameter,
        "value": value,
        "weights": cfg.weights,
        "starts": cfg.starts,
        "maxiter": cfg.maxiter,
        "pienn_iter": cfg.pienn_iter,
        "nc": cfg.nc,
        "r_factor": cfg.r_factor,
    }
    row.update(metrics)
    rows.append(row)
    print(
        f"{problem} {parameter}={value}: nd={metrics['num_nd']}, "
        f"feasible={metrics['num_feasible']}/{metrics['num_solutions']}, "
        f"dist={metrics['mean_distance_to_ref']:.4g}, hv={metrics['hypervolume']:.4g}, "
        f"time={metrics['runtime_sec']:.2f}s"
    )
    return approx


def format_value(value: float | int | str) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-12:
        return str(int(round(number)))
    return f"{number:.3g}"


def write_latex_table(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    parameter_order = ["N_p", "K_max", "R", "N_c"]
    parameter_latex = {
        "N_p": r"$N_p$",
        "K_max": r"$K_{\max}$",
        "R": r"$R$",
        "N_c": r"$N_c$",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Parameter sensitivity analysis of PIENN on ZDT4.}",
        r"\label{tab:parameter_sensitivity}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        "Parameter & Value & Mean distance & HV & Runtime (s) " + r"\\",
        r"\midrule",
    ]
    for p_idx, parameter in enumerate(parameter_order):
        subset = [row for row in rows if row["parameter"] == parameter]
        for idx, row in enumerate(subset):
            label = parameter_latex[parameter] if idx == 0 else ""
            lines.append(
                f"{label} & {format_value(row['value'])} & "
                f"{float(row['mean_distance_to_ref']):.4g} & "
                f"{float(row['hypervolume']):.4g} & "
                f"{float(row['runtime_sec']):.2f} \\\\" 
            )
        if p_idx != len(parameter_order) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(rows: list[dict[str, float | int | str]], parameter: str, metric: str, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = [row for row in rows if row["parameter"] == parameter]
    if not subset:
        return
    x = [float(row["value"]) for row in subset]
    y = [float(row[metric]) for row in subset]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(x, y, "o-", linewidth=1.6)
    ax.set_xlabel(parameter)
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{parameter}_{metric}.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-factor-at-a-time PIENN parameter sensitivity analysis.")
    parser.add_argument("--problem", default="zdt4", help="pymoo bi-objective problem name, e.g., zdt4, mw2, welded_beam")
    parser.add_argument("--weights", type=int, default=11, help="baseline number of epsilon subproblems")
    parser.add_argument("--starts", type=int, default=4, help="baseline PIENN swarm size")
    parser.add_argument("--maxiter", type=int, default=80, help="baseline local subproblem max iterations")
    parser.add_argument("--pienn-iter", type=int, default=3, help="baseline PIENN outer iterations")
    parser.add_argument("--nc", type=int, default=2, help="baseline map/landmark switch iteration")
    parser.add_argument("--r-factor", type=float, default=0.3, help="baseline map-operator decay factor")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=BASE_OUTPUT_DIR)
    parser.add_argument("--starts-values", default="3,4,6,8")
    parser.add_argument("--pienn-iter-values", default="2,3,4,6")
    parser.add_argument("--r-factor-values", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--nc-values", default="1,2,3,4")
    parser.add_argument("--plots", action="store_true", help="optionally generate metric curves; by default only CSV and LaTeX table are written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = RunConfig(
        weights=args.weights,
        starts=args.starts,
        seed=args.seed,
        maxiter=args.maxiter,
        output_dir=output_dir,
        pienn_iter=args.pienn_iter,
        nc=args.nc,
        r_factor=args.r_factor,
    )

    rows: list[dict[str, float | int | str]] = []
    for value in parse_int_list(args.starts_values):
        add_case(rows, args.problem, "N_p", value, replace(base_cfg, starts=value))
    for value in parse_int_list(args.pienn_iter_values):
        add_case(rows, args.problem, "K_max", value, replace(base_cfg, pienn_iter=value, nc=min(base_cfg.nc, value)))
    for value in parse_float_list(args.r_factor_values):
        add_case(rows, args.problem, "R", value, replace(base_cfg, r_factor=value))
    for value in parse_int_list(args.nc_values):
        add_case(rows, args.problem, "N_c", value, replace(base_cfg, nc=min(value, base_cfg.pienn_iter)))

    csv_path = output_dir / f"{args.problem}_sensitivity_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table_path = output_dir / f"{args.problem}_sensitivity_table.tex"
    write_latex_table(rows, table_path)

    if args.plots:
        for parameter in ("N_p", "K_max", "R", "N_c"):
            plot_metric(rows, parameter, "mean_distance_to_ref", output_dir)
            plot_metric(rows, parameter, "hypervolume", output_dir)
            plot_metric(rows, parameter, "runtime_sec", output_dir)

    print(f"Saved sensitivity summary to {csv_path}")
    print(f"Saved LaTeX table to {table_path}")


if __name__ == "__main__":
    main()
