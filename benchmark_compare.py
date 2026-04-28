#!/usr/bin/env python3
"""Запуск всех трёх MPC (IPOPT / OSQP / ActiveSet) на общих сценариях и построение
сводных графиков эффективности: время решения, качество слежения, длина пути,
успех задачи. Запуск без GUI.

Примеры::

    python benchmark_compare.py
    python benchmark_compare.py --scenario head_on --integrators euler,rk4
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# -----------------------------------------------------------------------------
# Загрузка симуляций как модулей (без пакета simulation/)
# -----------------------------------------------------------------------------


def _load_sim_script(filename: str):
    path = os.path.join(_ROOT, "simulation", filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _discover_scenarios() -> List[Tuple[str, str]]:
    base = os.path.join(_ROOT, "scenarios")
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        ypath = os.path.join(d, f"{name}.yaml")
        if os.path.isfile(ypath):
            out.append((name, ypath))
    return out


# -----------------------------------------------------------------------------
# Метрики по траектории и управлению
# -----------------------------------------------------------------------------


def point_to_polyline_distance(p: np.ndarray, poly: np.ndarray) -> float:
    """Минимальное расстояние от точки p до ломаной poly (N, 2)."""
    best = np.inf
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0))
        proj = a + t * ab
        best = min(best, float(np.linalg.norm(p - proj)))
    return best


def cross_track_series(traj_xy: np.ndarray, ref_path: np.ndarray) -> np.ndarray:
    """Поперечная ошибка для каждой точки траектории относительно опорной полилинии."""
    out = np.zeros(len(traj_xy))
    for k in range(len(traj_xy)):
        out[k] = point_to_polyline_distance(traj_xy[k], ref_path[:, :2])
    return out


def path_length(traj_xy: np.ndarray) -> float:
    if len(traj_xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(traj_xy, axis=0), axis=1)))


def control_smoothness(controls: np.ndarray) -> Dict[str, float]:
    """Прокси гладкости: сумма квадратов приращений v и omega."""
    if controls is None or len(controls) < 2:
        return {"jerk_proxy_v": 0.0, "jerk_proxy_w": 0.0}
    dv = np.diff(controls[:, 0])
    dw = np.diff(controls[:, 1])
    return {
        "jerk_proxy_v": float(np.sum(dv**2)),
        "jerk_proxy_w": float(np.sum(dw**2)),
    }


def enrich_metrics(
    result: Dict[str, Any],
    goal: np.ndarray,
    dt: float,
    solver: str,
    scenario: str,
    integrator: str = "euler",
) -> Dict[str, Any]:
    traj = result["trajectory"]
    ref = result.get("reference_path")
    times = np.asarray(result["solve_times"], dtype=float)
    ctrls = result.get("controls")

    traj_xy = traj[:, :2]
    fd = float(result.get("final_dist", np.inf))

    ig = str(integrator).lower()
    metrics: Dict[str, Any] = {
        "solver": solver,
        "integrator": ig,
        "run_label": f"{solver}_{ig}",
        "scenario": scenario,
        "steps": int(result.get("steps", 0)),
        "final_dist": fd,
        "success": fd < 0.2,
        "dt": dt,
        "wall_time_sim": float(result.get("steps", 0)) * dt,
        "path_length": path_length(traj_xy),
        "straight_dist": float(np.linalg.norm(traj_xy[0] - goal[:2]))
        if len(traj_xy)
        else 0.0,
        "solve_times_ms": (times * 1000).tolist() if len(times) else [],
        "solve_mean_ms": float(np.mean(times) * 1000) if len(times) else 0.0,
        "solve_median_ms": float(np.median(times) * 1000) if len(times) else 0.0,
        "solve_p95_ms": float(np.percentile(times * 1000, 95)) if len(times) else 0.0,
        "solve_p99_ms": float(np.percentile(times * 1000, 99)) if len(times) else 0.0,
        "solve_max_ms": float(np.max(times) * 1000) if len(times) else 0.0,
        "total_solve_s": float(np.sum(times)),
        "control_energy": float(np.sum(np.sum(ctrls**2, axis=1))) if ctrls is not None else 0.0,
    }
    js = control_smoothness(ctrls if ctrls is not None else np.zeros((0, 2)))
    metrics.update(js)

    if ref is not None and len(ref) >= 2:
        cte = cross_track_series(traj_xy, ref)
        metrics["cte_rmse"] = float(np.sqrt(np.mean(cte**2)))
        metrics["cte_max"] = float(np.max(cte))
    else:
        metrics["cte_rmse"] = float("nan")
        metrics["cte_max"] = float("nan")

    # Эффективность пути: отношение прямой длины к фактической (≤ 1 — идеально по кратчайшему)
    pl = metrics["path_length"]
    if pl > 1e-9:
        straight = float(np.linalg.norm(goal[:2] - traj_xy[0]))
        metrics["path_efficiency"] = straight / pl
    else:
        metrics["path_efficiency"] = 0.0

    return metrics


# -----------------------------------------------------------------------------
# Фабрики MPC (общие веса/горизонт/лимиты совпадают с simulation/UnicycleMPC_Tracking*.py)
# -----------------------------------------------------------------------------


def make_ipopt_mpc(env, integration_method: str = "euler"):
    from MPC.unicycle_mpc_ipopt_tracking import UnicycleMPC_Tracking

    return UnicycleMPC_Tracking(
        dt=env.step_time,
        horizon=25,
        q_pos=3.0,
        q_theta=2.0,
        r=0.1,
        r_yaw=0.1,
        integration_method=integration_method,
        v_max=1.0,
        w_max=1.0,
        dv_max=0.25,
        dw_max=0.5,
        safe_distance=0.2,
        n_obs=10,
    )


def make_osqp_mpc(env, integration_method: str = "euler"):
    from MPC.unicycle_mpc_osqp_tracking import UnicycleMPC_OSQP_Tracking

    return UnicycleMPC_OSQP_Tracking(
        dt=env.step_time,
        horizon=25,
        q_pos=3.0,
        q_theta=2.0,
        r=0.1,
        r_yaw=0.1,
        integration_method=integration_method,
        v_max=1.0,
        w_max=1.0,
        dv_max=0.25,
        dw_max=0.5,
        safe_distance=0.35,
        n_obs=10,
        slack_weight=300.0,
        max_active_obs=5,
        obs_horizon=8,
        activation_margin=3.5,
        progress_weight=4.5,
        ttc_threshold=4.0,
        tangent_bias_gain=0.3,
        obstacle_slowdown_margin=1.5,
        min_speed_factor_near_obstacle=0.8,
        dynamic_obstacle_speed_threshold=0.05,
    )


def make_activeset_mpc(env, integration_method: str = "euler"):
    from MPC.unicycle_mpc_active_set_tracking import UnicycleMPC_ActiveSet_Tracking

    return UnicycleMPC_ActiveSet_Tracking(
        dt=env.step_time,
        horizon=25,
        q_pos=3.0,
        q_theta=2.0,
        r=0.1,
        r_yaw=0.1,
        integration_method=integration_method,
        v_max=1.0,
        w_max=1.0,
        dv_max=0.25,
        dw_max=0.5,
        safe_distance=0.35,
        n_obs=10,
        slack_weight=300.0,
        max_active_obs=5,
        obs_horizon=8,
        activation_margin=3.5,
        progress_weight=4.5,
        ttc_threshold=4.0,
        tangent_bias_gain=0.3,
        obstacle_slowdown_margin=1.5,
        min_speed_factor_near_obstacle=0.8,
        dynamic_obstacle_speed_threshold=0.05,
    )


def run_one_solver(
    yaml_path: str,
    scenario_name: str,
    solver_name: str,
    mpc_factory: Callable,
    mod,
    max_steps: int = 500,
    integrator: str = "euler",
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    import irsim

    try:
        env = irsim.make(yaml_path)
        goal = np.array(env.robot.goal).flatten()
        mpc = mpc_factory(env, integrator)
        run = mod.run_simulation
        raw = run(env, mpc, goal, max_steps=max_steps, verbose=False, render=False)
        env.end()
        dt = float(env.step_time)
        metrics = enrich_metrics(raw, goal, dt, solver_name, scenario_name, integrator=integrator)
        return raw, metrics, None
    except Exception as exc:
        return None, None, str(exc)


# -----------------------------------------------------------------------------
# Визуализация
# -----------------------------------------------------------------------------

COLORS = {"IPOPT": "#d62728", "OSQP": "#1f77b4", "ActiveSet": "#2ca02c"}

SOLVER_ORDER = ["IPOPT", "OSQP", "ActiveSet"]


def _metric_run_label(m: Dict[str, Any]) -> str:
    return str(m.get("run_label") or f"{m['solver']}_{m.get('integrator', 'euler')}")


def _run_label_sort_key(rl: str) -> Tuple[int, int]:
    if "_" not in rl:
        return 99, 99
    solver, ig = rl.rsplit("_", 1)
    try:
        si = SOLVER_ORDER.index(solver)
    except ValueError:
        si = 99
    ii = 0 if ig == "euler" else 1 if ig == "rk4" else 2
    return si, ii


def ordered_run_labels_from_metrics(all_metrics: List[Dict[str, Any]]) -> List[str]:
    """Порядок серий на графиках: IPOPT/OSQP/ActiveSet × euler, затем rk4."""
    ints = sorted(set(m.get("integrator", "euler") for m in all_metrics))
    labels: List[str] = []
    for s in SOLVER_ORDER:
        for ig in ints:
            rl = f"{s}_{ig}"
            if any(_metric_run_label(m) == rl for m in all_metrics):
                labels.append(rl)
    return labels


def run_label_color_map(run_labels: List[str]) -> Dict[str, str]:
    import matplotlib.colors as mcolors

    n = len(run_labels)
    if n == 0:
        return {}
    return {
        rl: mcolors.to_hex(plt.cm.tab10(i / max(n - 1, 1)))
        for i, rl in enumerate(run_labels)
    }


def parse_integrators_arg(s: str) -> List[str]:
    out: List[str] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in ("euler", "rk4"):
            raise SystemExit(f"Неизвестный интегратор {p!r}; допустимы euler, rk4")
        if p not in out:
            out.append(p)
    return out if out else ["euler"]


def plot_scenario_dashboard(
    scenario_name: str,
    results: Dict[str, Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
    goal: np.ndarray,
    out_dir: str,
) -> None:
    """Один сценарий: большая фигура с несколькими панелями."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(results)
    if n == 0:
        return

    run_order = sorted(results.keys(), key=_run_label_sort_key)
    pal = run_label_color_map(run_order)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1) XY траектории
    ax_xy = fig.add_subplot(gs[0, :2])
    for name in run_order:
        r = results[name]
        tr = r["trajectory"]
        c = pal.get(name, "gray")
        ax_xy.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=2, label=name, alpha=0.9)
        ax_xy.scatter(tr[0, 0], tr[0, 1], c=c, s=40, zorder=5)
    ref = next(iter(results.values())).get("reference_path")
    if ref is not None:
        ax_xy.plot(ref[:, 0], ref[:, 1], "k--", lw=1, alpha=0.5, label="Прямой reference")
    ax_xy.scatter([goal[0]], [goal[1]], c="red", s=120, marker="*", zorder=6, label="Цель")
    ax_xy.set_title(f"Траектории: {scenario_name}")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best", fontsize=8)

    # 2) Таблица скаляров
    ax_tbl = fig.add_subplot(gs[0, 2])
    ax_tbl.axis("off")
    rows = []
    for name in run_order:
        if name not in metrics:
            continue
        m = metrics[name]
        rows.append(
            f"{name}: ok={m['success']}  steps={m['steps']}  "
            f"mean={m['solve_mean_ms']:.2f}ms  p95={m['solve_p95_ms']:.2f}ms  "
            f"CTE_rmse={m['cte_rmse']:.3f}"
        )
    ax_tbl.text(0, 0.95, "\n".join(rows), transform=ax_tbl.transAxes, fontsize=9, va="top", family="monospace")

    # 3) Расстояние до цели
    ax_d = fig.add_subplot(gs[1, 0])
    for name in run_order:
        r = results[name]
        ax_d.plot(r["distances"], color=pal.get(name, "gray"), label=name)
    ax_d.axhline(0.2, color="red", ls="--", alpha=0.5)
    ax_d.set_title("Дистанция до цели")
    ax_d.set_xlabel("Шаг")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    # 4) Время решателя (мс)
    ax_t = fig.add_subplot(gs[1, 1])
    for name in run_order:
        r = results[name]
        st = np.asarray(r["solve_times"]) * 1000
        ax_t.plot(st, color=pal.get(name, "gray"), lw=1, label=name, alpha=0.85)
    ax_t.set_title("Время решения MPC (мс)")
    ax_t.set_xlabel("Шаг")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)

    # 5) Кумулятивное время MPC
    ax_c = fig.add_subplot(gs[1, 2])
    for name in run_order:
        r = results[name]
        st = np.asarray(r["solve_times"])
        ax_c.plot(np.cumsum(st), color=pal.get(name, "gray"), label=name)
    ax_c.set_title("Накопленное время решателя (с)")
    ax_c.set_xlabel("Шаг")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    # 6) Поперечная ошибка
    ax_cte = fig.add_subplot(gs[2, 0])
    for name in run_order:
        r = results[name]
        refp = r.get("reference_path")
        if refp is None:
            continue
        tr = r["trajectory"]
        cte = cross_track_series(tr[:, :2], refp[:, :2])
        ax_cte.plot(cte, color=pal.get(name, "gray"), label=name)
    ax_cte.set_title("Поперечная ошибка до прямой")
    ax_cte.set_xlabel("Шаг")
    ax_cte.legend(fontsize=8)
    ax_cte.grid(True, alpha=0.3)

    # 7) Управление |u|
    ax_u = fig.add_subplot(gs[2, 1])
    for name in run_order:
        r = results[name]
        u = r.get("controls")
        if u is None:
            continue
        nu = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
        ax_u.plot(nu, color=pal.get(name, "gray"), label=name)
    ax_u.set_title("Норма управления √(v²+ω²)")
    ax_u.set_xlabel("Шаг")
    ax_u.legend(fontsize=8)
    ax_u.grid(True, alpha=0.3)

    # 8) ECDF времени решения
    ax_ecdf = fig.add_subplot(gs[2, 2])
    for name in run_order:
        r = results[name]
        st = np.sort(np.asarray(r["solve_times"]) * 1000)
        if len(st) == 0:
            continue
        y = np.linspace(0, 1, len(st), endpoint=False)
        ax_ecdf.plot(st, y, color=pal.get(name, "gray"), label=name)
    ax_ecdf.set_title("ECDF времени решения (мс)")
    ax_ecdf.set_xlabel("Время (мс)")
    ax_ecdf.set_ylabel("Доля")
    ax_ecdf.legend(fontsize=8)
    ax_ecdf.grid(True, alpha=0.3)

    fig.suptitle(f"MPC сравнение — {scenario_name}", fontsize=14)
    out_path = os.path.join(out_dir, f"dashboard_{scenario_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def plot_aggregate(
    all_metrics: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    """Сводка по всем сценариям: столбцы, тепловая карта, scatter, радар."""
    if not all_metrics:
        return
    os.makedirs(out_dir, exist_ok=True)

    scenarios = sorted(set(m["scenario"] for m in all_metrics))
    run_labels = ordered_run_labels_from_metrics(all_metrics)
    if not run_labels:
        run_labels = sorted({_metric_run_label(m) for m in all_metrics}, key=_run_label_sort_key)
    pal = run_label_color_map(run_labels)
    n_rl = len(run_labels)
    w_bar = min(0.8 / max(n_rl, 1), 0.22)

    def _row_for(sc: str, rl: str):
        return next(
            (m for m in all_metrics if m["scenario"] == sc and _metric_run_label(m) == rl),
            None,
        )

    # --- Бар: среднее время и p95 по сценариям (группы)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(len(scenarios))
    for i, rl in enumerate(run_labels):
        means = []
        p95s = []
        for sc in scenarios:
            row = _row_for(sc, rl)
            means.append(row["solve_mean_ms"] if row else 0)
            p95s.append(row["solve_p95_ms"] if row else 0)
        xpos = x + (i - (n_rl - 1) / 2.0) * w_bar
        axes[0].bar(xpos, means, width=w_bar * 0.92, label=rl, color=pal[rl])
        axes[1].bar(xpos, p95s, width=w_bar * 0.92, label=rl, color=pal[rl])
    axes[0].set_title("Среднее время решения MPC (мс) по сценариям")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios, rotation=25, ha="right")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].set_title("p95 времени решения (мс) по сценариям")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios, rotation=25, ha="right")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aggregate_timing_bars.png"), dpi=150)
    plt.close(fig)

    # --- Тепловая карта: успех
    success_mat = np.zeros((len(scenarios), len(run_labels)))
    for si, sc in enumerate(scenarios):
        for ji, rl in enumerate(run_labels):
            row = _row_for(sc, rl)
            if row:
                success_mat[si, ji] = 1.0 if row["success"] else 0.0
            else:
                success_mat[si, ji] = np.nan

    fig_w = max(8.0, 0.45 * len(run_labels))
    fig, ax = plt.subplots(figsize=(fig_w, max(4, 0.35 * len(scenarios))))
    im = ax.imshow(success_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(run_labels)))
    ax.set_xticklabels(run_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title("Успех достижения цели (dist < 0.2)")
    for i in range(success_mat.shape[0]):
        for j in range(success_mat.shape[1]):
            v = success_mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, "✓" if v > 0.5 else "✗", ha="center", va="center", fontsize=14)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap_success.png"), dpi=150)
    plt.close(fig)

    # --- Scatter: mean solve time vs CTE RMSE
    fig, ax = plt.subplots(figsize=(10, 7))
    for rl in run_labels:
        xs = [m["solve_mean_ms"] for m in all_metrics if _metric_run_label(m) == rl]
        ys = [m["cte_rmse"] for m in all_metrics if _metric_run_label(m) == rl]
        ax.scatter(
            xs,
            ys,
            c=pal[rl],
            label=rl,
            s=80,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.set_xlabel("Среднее время решения (мс)")
    ax.set_ylabel("RMSE поперечной ошибки")
    ax.set_title("Компромисс: скорость решателя vs точность слежения")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_time_vs_cte.png"), dpi=150)
    plt.close(fig)

    # --- Гистограммы: все покадровые времена решения (по всем сценариям), по сериям
    fig, ax = plt.subplots(figsize=(10, 6))
    for rl in run_labels:
        st_all: List[float] = []
        for m in all_metrics:
            if _metric_run_label(m) != rl:
                continue
            st_all.extend(m.get("solve_times_ms") or [])
        if st_all:
            ax.hist(st_all, bins=40, alpha=0.45, label=rl, color=pal[rl], density=True)
    ax.set_xlabel("Время решения MPC (мс)")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределение времени решателя (все шаги всех сценариев)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_solve_times_pooled.png"), dpi=150)
    plt.close(fig)

    # --- Столбцы: RMSE поперечной ошибки по сценариям
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenarios))
    for i, rl in enumerate(run_labels):
        heights = []
        for sc in scenarios:
            row = _row_for(sc, rl)
            v = row["cte_rmse"] if row and not np.isnan(row["cte_rmse"]) else 0.0
            heights.append(v)
        xpos = x + (i - (n_rl - 1) / 2.0) * w_bar
        ax.bar(xpos, heights, width=w_bar * 0.92, label=rl, color=pal[rl])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel("RMSE поперечной ошибки")
    ax.set_title("Точность удержания прямого reference")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bar_cte_rmse_by_scenario.png"), dpi=150)
    plt.close(fig)

    # --- Boxplot: распределение mean_ms по сценариям для каждой серии
    data_by_rl = {rl: [m["solve_mean_ms"] for m in all_metrics if _metric_run_label(m) == rl] for rl in run_labels}
    if any(len(data_by_rl[rl]) > 0 for rl in run_labels):
        fig_w = max(10, 1.2 * len(run_labels))
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        lists = [data_by_rl[rl] for rl in run_labels]
        positions = list(range(1, len(run_labels) + 1))
        bp = ax.boxplot(lists, positions=positions, widths=0.5, patch_artist=True)
        for patch, rl in zip(bp["boxes"], run_labels):
            patch.set_facecolor(pal[rl])
            patch.set_alpha(0.45)
        ax.set_xticks(positions)
        ax.set_xticklabels(run_labels, rotation=25, ha="right")
        ax.set_ylabel("Среднее время решения (мс), по сценариям")
        ax.set_title("Распределение среднего времени MPC по сценариям")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "box_mean_solve_time.png"), dpi=150)
        plt.close(fig)

    # --- Нормализованный радар (среднее по сценариям для каждой серии solver+integrator)
    def aggregate_run(rl: str) -> Dict[str, float]:
        ms = [m for m in all_metrics if _metric_run_label(m) == rl]
        if not ms:
            return {}
        return {
            "mean_time": np.mean([x["solve_mean_ms"] for x in ms]),
            "p95_time": np.mean([x["solve_p95_ms"] for x in ms]),
            "cte": np.nanmean([x["cte_rmse"] for x in ms]),
            "path_eff": np.nanmean([x["path_efficiency"] for x in ms]),
            "success_rate": np.mean([1.0 if x["success"] else 0.0 for x in ms]),
        }

    labels = ["Быстрота\n(inv mean ms)", "Низкий p95\n(inv)", "Точность\n(inv CTE)", "Эффект. пути", "Успех"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    raw_rows: List[List[float]] = []
    rl_order: List[str] = []
    for rl in run_labels:
        ag = aggregate_run(rl)
        if not ag:
            continue
        mt = ag["mean_time"] + 1e-6
        p9 = ag["p95_time"] + 1e-6
        ct = ag["cte"] if not np.isnan(ag["cte"]) else 1.0
        raw_rows.append([1.0 / mt, 1.0 / p9, 1.0 / (ct + 1e-6), ag["path_eff"], ag["success_rate"]])
        rl_order.append(rl)
    if raw_rows:
        mat = np.array(raw_rows)
        mat_n = mat / (np.max(mat, axis=0) + 1e-12)
        for idx, rl in enumerate(rl_order):
            row = mat_n[idx].tolist()
            row += row[:1]
            ax.plot(angles, row, "o-", color=pal[rl], label=rl)
            ax.fill(angles, row, color=pal[rl], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Нормализованный профиль эффективности (среднее по сценариям)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "radar_normalized.png"), dpi=150)
    plt.close(fig)

    # --- Столбцы: шаги до цели (или max_steps если не успех)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenarios))
    for i, rl in enumerate(run_labels):
        heights = []
        for sc in scenarios:
            row = _row_for(sc, rl)
            heights.append(row["steps"] if row else 0)
        xpos = x + (i - (n_rl - 1) / 2.0) * w_bar
        ax.bar(xpos, heights, width=w_bar * 0.92, label=rl, color=pal[rl])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel("Число шагов")
    ax.set_title("Длина эпизода (шаги симулятора)")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aggregate_steps.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Сравнение MPC солверов и графики")
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Имя сценария (папка в scenarios/). По умолчанию — все.",
    )
    parser.add_argument("--output", type=str, default=os.path.join(_ROOT, "benchmark_results"))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--integrators",
        type=str,
        default="euler",
        help="Интегратор(ы) дискретизации внутри MPC: euler, rk4 или euler,rk4 (через запятую).",
    )
    args = parser.parse_args()
    integrators = parse_integrators_arg(args.integrators)

    mod_ipopt = _load_sim_script("UnicycleMPC_Tracking.py")
    mod_osqp = _load_sim_script("UnicycleMPC_Tracking_OSQP.py")
    mod_as = _load_sim_script("UnicycleMPC_Tracking_ActiveSet.py")

    scenarios = _discover_scenarios()
    if args.scenario:
        scenarios = [(n, p) for n, p in scenarios if n == args.scenario]
        if not scenarios:
            print(f"Сценарий не найден: {args.scenario}")
            return

    os.makedirs(args.output, exist_ok=True)
    all_metrics: List[Dict[str, Any]] = []
    mapping = [
        ("IPOPT", make_ipopt_mpc, mod_ipopt),
        ("OSQP", make_osqp_mpc, mod_osqp),
        ("ActiveSet", make_activeset_mpc, mod_as),
    ]

    import irsim

    for scenario_name, yaml_path in scenarios:
        print(f"\n=== Сценарий: {scenario_name} ===")
        results: Dict[str, Dict[str, Any]] = {}
        metrics_run: Dict[str, Dict[str, Any]] = {}

        env_goal = irsim.make(yaml_path)
        goal_ref = np.array(env_goal.robot.goal).flatten()
        env_goal.end()

        for integrator in integrators:
            for solver_name, factory, mod in mapping:
                run_lbl = f"{solver_name}_{integrator}"
                raw, met, err = run_one_solver(
                    yaml_path,
                    scenario_name,
                    solver_name,
                    factory,
                    mod,
                    max_steps=args.max_steps,
                    integrator=integrator,
                )
                if err:
                    print(f"  [{run_lbl}] ОШИБКА: {err}")
                    continue
                assert raw is not None and met is not None
                results[run_lbl] = raw
                metrics_run[run_lbl] = met
                all_metrics.append(met)
                print(
                    f"  [{run_lbl}] success={met['success']} steps={met['steps']} "
                    f"mean_ms={met['solve_mean_ms']:.2f} cte_rmse={met['cte_rmse']:.4f}"
                )

        if results:
            plot_scenario_dashboard(scenario_name, results, metrics_run, goal_ref, args.output)

    # JSON Lines сводка
    jl_path = os.path.join(args.output, "metrics.jsonl")
    with open(jl_path, "w", encoding="utf-8") as f:
        for m in all_metrics:
            slim = {k: v for k, v in m.items() if k != "solve_times_ms"}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
    print(f"\nМетрики записаны: {jl_path}")

    plot_aggregate(all_metrics, args.output)
    print(f"Сводные графики в каталоге: {args.output}")


if __name__ == "__main__":
    main()
