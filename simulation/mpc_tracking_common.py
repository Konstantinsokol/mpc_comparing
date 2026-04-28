"""Общая логика трекинг-симуляций IPOPT / OSQP / ActiveSet.

Сценарий, извлечение препятствий, прямая reference-линия, скользящее окно
reference и цикл receding horizon — одинаковые; отличаются только класс MPC,
подписи окна/лога и заголовок графика времени. Скрипты в корне `simulation/`
остаются точками входа и местом для `mpc_config`."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpc_tracking_render import render_tracking_frame


def _normalize_gif_output_path(path: str) -> str:
    """Путь без расширения (как `../gifs/foo/bar`) дополняем `.gif`, иначе Pillow не знает формат."""
    path = os.path.expanduser(path)
    root, ext = os.path.splitext(path)
    if not ext:
        return root + ".gif"
    return path


def _save_bgr_frames_as_gif(frames: List[np.ndarray], path: str, fps: float) -> str:
    """Сохраняет кадры в GIF; возвращает фактический путь (с `.gif`, если расширение было пустым)."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Для сохранения GIF установите Pillow: pip install pillow") from e
    if not frames:
        raise ValueError("Нет кадров для GIF")
    path = _normalize_gif_output_path(path)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))
    rgbs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imgs = [Image.fromarray(f) for f in rgbs]
    first, *rest = imgs
    first.save(
        path,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )
    return os.path.abspath(path)

# Корень репозитория (родитель каталога simulation/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def scenario_arg_to_yaml_path(scenario: Optional[str]) -> str:
    """Преобразует аргумент сценария в путь к YAML (как раньше в `resolve_yaml_path`)."""
    if scenario is None or scenario == "":
        return os.path.join(_REPO_ROOT, "scenarios", "free_path", "free_path.yaml")
    if scenario.endswith(".yaml"):
        return os.path.abspath(scenario)
    return os.path.join(_REPO_ROOT, "scenarios", scenario, f"{scenario}.yaml")


def resolve_yaml_path(argv: Optional[List[str]] = None) -> str:
    """Путь к YAML: из argv[1] или сценарий free_path по умолчанию (устаревший способ; см. `parse_tracking_cli`)."""
    argv = argv if argv is not None else sys.argv
    scenario = argv[1] if len(argv) > 1 else None
    return scenario_arg_to_yaml_path(scenario)


@dataclass(frozen=True)
class TrackingCliOptions:
    """Результат разбора CLI для `UnicycleMPC_Tracking*.py`."""

    yaml_path: str
    gif_path: Optional[str] = None
    gif_fps: float = 10.0
    render: bool = True
    max_steps: int = 500


def parse_tracking_cli(argv: Optional[List[str]] = None) -> TrackingCliOptions:
    """Разбор аргументов командной строки (всё после имени скрипта).

    Примеры::

        python UnicycleMPC_Tracking_OSQP.py dead_end
        python UnicycleMPC_Tracking_OSQP.py scenarios/head_on/head_on.yaml --gif out.gif
        python UnicycleMPC_Tracking_OSQP.py --no-render --gif run.gif head_on
    """
    raw = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Симуляция трекинга MPC: YAML ir-sim или имя сценария из каталога scenarios/.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default=None,
        help="Путь к .yaml или имя сценария (например dead_end). По умолчанию: free_path.",
    )
    parser.add_argument(
        "--gif",
        dest="gif_path",
        default=None,
        metavar="FILE",
        help="Сохранить анимацию проезда в GIF (нужен Pillow).",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=10.0,
        help="Частота кадров GIF (по умолчанию 10).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Не открывать окно OpenCV (удобно с --gif на сервере).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        metavar="N",
        help="Ограничение числа шагов симуляции (по умолчанию 500).",
    )
    args = parser.parse_args(raw)
    return TrackingCliOptions(
        yaml_path=scenario_arg_to_yaml_path(args.scenario),
        gif_path=args.gif_path,
        gif_fps=float(args.gif_fps),
        render=not args.no_render,
        max_steps=max(1, int(args.max_steps)),
    )


def get_obstacles_from_env(env: Any) -> List[Dict[str, Any]]:
    """Препятствия в формате, ожидаемом всеми MPC.solve(...)."""
    obstacles: List[Dict[str, Any]] = []
    raw_obs = getattr(env, "obstacle_list", [])

    for obs in raw_obs:
        try:
            pos = np.array(obs.state[:2]).flatten()
        except Exception:
            continue

        r = 0.2
        if hasattr(obs, "shape_inf"):
            s_inf = obs.shape_inf
            if isinstance(s_inf, (list, np.ndarray)):
                r = float(s_inf[0])
            else:
                r = float(s_inf)
        elif hasattr(obs, "radius"):
            r = float(obs.radius)

        vel = np.zeros(2)
        if hasattr(obs, "velocity") and obs.velocity is not None:
            vel = np.array(obs.velocity).flatten()[:2]
        elif len(obs.state) >= 5:
            vel = np.array(obs.state[3:5]).flatten()

        obstacles.append({"position": pos, "velocity": vel, "radius": r})

    return obstacles


def build_straight_path(start: np.ndarray, goal: np.ndarray, step_size: float) -> np.ndarray:
    """Прямой глобальный путь от старта к цели (x, y, theta)."""
    path_start = np.array(start, dtype=float).copy()
    path_goal = np.array(goal[:2], dtype=float).copy()

    dx = path_goal[0] - path_start[0]
    dy = path_goal[1] - path_start[1]
    distance = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx) if distance > 1e-9 else path_start[2]

    n_segments = max(1, int(np.ceil(distance / max(step_size, 1e-6))))
    straight_path = np.zeros((n_segments + 1, 3))

    for idx, alpha in enumerate(np.linspace(0.0, 1.0, n_segments + 1)):
        straight_path[idx, 0] = path_start[0] + dx * alpha
        straight_path[idx, 1] = path_start[1] + dy * alpha
        straight_path[idx, 2] = heading

    straight_path[0, 2] = path_start[2]
    return straight_path


def generate_reference_trajectory(
    path: np.ndarray,
    state: np.ndarray,
    horizon: int,
    last_idx: int = 0,
    obstacles: Any = None,
) -> Tuple[np.ndarray, int]:
    """Локальное окно reference на прямой; `obstacles` зарезервирован (не используется)."""
    _ = obstacles
    ref_traj = np.zeros((horizon + 1, 3))
    ref_traj[0] = state.copy()

    search_window = 30
    end_idx = min(len(path), last_idx + search_window)
    distances = np.linalg.norm(path[last_idx:end_idx, :2] - state[:2], axis=1)
    nearest_local = int(np.argmin(distances))
    nearest_idx = last_idx + nearest_local

    lookahead = 5
    for k in range(1, horizon + 1):
        path_idx = min(nearest_idx + lookahead + k, len(path) - 1)
        ref_traj[k] = path[path_idx]

    eps = 1e-9
    for k in range(horizon):
        dx = ref_traj[k + 1, 0] - ref_traj[k, 0]
        dy = ref_traj[k + 1, 1] - ref_traj[k, 1]
        if np.hypot(dx, dy) > eps:
            ref_traj[k, 2] = np.arctan2(dy, dx)
        else:
            ref_traj[k, 2] = ref_traj[k - 1, 2] if k > 0 else state[2]

    ref_traj[-1, 2] = ref_traj[-2, 2]
    return ref_traj, nearest_idx


@dataclass(frozen=True)
class TrackingRunStyle:
    """Подписи для окна OpenCV и консоли (можно оставить разный UX на солвер)."""

    window_title: str
    log_name: str
    tag_goal_message: bool = True


def run_tracking_simulation(
    env: Any,
    mpc: Any,
    goal: np.ndarray,
    *,
    style: TrackingRunStyle,
    max_steps: int = 500,
    verbose: bool = True,
    render: bool = True,
    gif_path: Optional[str] = None,
    gif_fps: float = 10.0,
) -> Dict[str, Any]:
    """Общий receding-horizon цикл; `mpc.solve(state, ref_traj, obstacles)` как у всех трёх.

    Если задан `gif_path`, на каждом шаге снимается тот же кадр, что и для OpenCV; окно показывается
    только при `render=True`. Можно записать GIF без окна: `render=False`, `gif_path=...`.
    """
    state = env.get_robot_state().flatten()
    trajectory: List[np.ndarray] = [state.copy()]
    solve_times: List[float] = []
    distances: List[float] = []
    controls: List[np.ndarray] = []
    gif_frames: Optional[List[np.ndarray]] = [] if gif_path else None

    tag = style.log_name
    if verbose:
        print(
            f"\n[{tag}] Старт: ({state[0]:.2f}, {state[1]:.2f}, θ={state[2]:.2f}), "
            f"Цель: ({goal[0]:.2f}, {goal[1]:.2f})"
        )

    path_line = build_straight_path(state.copy(), goal, step_size=max(mpc.dt * mpc.v_max, 0.15))
    last_idx = 0

    for step in range(max_steps):
        obstacles = get_obstacles_from_env(env)
        ref_traj, last_idx = generate_reference_trajectory(path_line, state, mpc.N, last_idx, obstacles)
        u, solve_time = mpc.solve(state, ref_traj, obstacles)

        env.step(np.array([[u[0]], [u[1]]]))

        state = env.get_robot_state().flatten()
        trajectory.append(state.copy())
        solve_times.append(solve_time)
        controls.append(np.array(u, copy=True))

        dist = float(np.linalg.norm(state[:2] - goal[:2]))
        distances.append(dist)

        if render or gif_path:
            img = render_tracking_frame(
                env,
                state,
                goal,
                ref_traj,
                path_line,
                trajectory,
                mpc=mpc,
                window_title=style.window_title,
                display=render,
            )
            if gif_frames is not None:
                gif_frames.append(img.copy())

        if verbose:
            print(
                f"Step {step:3d} | Time: {solve_time * 1000:6.2f} ms | Dist: {dist:.3f} | "
                f"v={u[0]:.3f}, w={u[1]:.3f}"
            )

        if dist < 0.2:
            if verbose:
                if style.tag_goal_message:
                    print(f"\n[{tag}] Цель достигнута за {step + 1} шагов")
                else:
                    print(f"\nЦель достигнута за {step + 1} шагов!")
            break

        if env.done():
            break

    if render:
        cv2.destroyAllWindows()

    saved_gif: Optional[str] = None
    if gif_path and gif_frames:
        saved_gif = _save_bgr_frames_as_gif(gif_frames, gif_path, gif_fps)
        if verbose:
            print(f"\n[{tag}] GIF сохранён: {saved_gif} ({len(gif_frames)} кадров, {gif_fps} FPS)")

    return {
        "trajectory": np.array(trajectory),
        "reference_path": path_line,
        "solve_times": solve_times,
        "distances": distances,
        "controls": np.array(controls),
        "steps": len(solve_times),
        "total_time": sum(solve_times),
        "avg_time": float(np.mean(solve_times)) if solve_times else 0.0,
        "final_dist": distances[-1] if distances else float(np.inf),
        "gif_path": saved_gif,
    }


def plot_tracking_results(
    result: Dict[str, Any],
    goal: np.ndarray,
    env: Any = None,
    *,
    save_path: str,
    time_panel_title: str,
) -> None:
    """Единые 2×2 графики; `time_panel_title` — например «Время IPOPT»."""
    traj = result["trajectory"]
    ref_path = result.get("reference_path")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(traj[:, 0], traj[:, 1], "b-o", markersize=3, label="Робот")
    if ref_path is not None:
        ax.plot(ref_path[:, 0], ref_path[:, 1], "--", color="olive", linewidth=2, label="Прямая траектория")
    ax.scatter(traj[0, 0], traj[0, 1], c="green", s=100, marker="o", label="Старт", zorder=5)
    ax.scatter([goal[0]], [goal[1]], c="red", s=200, marker="*", label="Цель", zorder=5)

    if env and hasattr(env, "obstacle_list"):
        for obs in env.obstacle_list:
            try:
                pos = np.array(obs.state[:2]).flatten()
                r = 0.2
                if hasattr(obs, "shape_inf"):
                    s_inf = obs.shape_inf
                    if isinstance(s_inf, (list, np.ndarray)):
                        r = float(s_inf[0])
                    else:
                        r = float(s_inf)
                ax.add_patch(plt.Circle(pos, r, color="gray", alpha=0.5))
            except Exception:
                pass

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Траектория")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    ax = axes[0, 1]
    ax.plot(result["distances"], "b-")
    ax.axhline(y=0.2, color="r", linestyle="--")
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Расстояние")
    ax.set_title("Сходимость")
    ax.grid(True)

    ax = axes[1, 0]
    times_ms = np.array(result["solve_times"]) * 1000
    ax.plot(times_ms, "g-")
    ax.axhline(
        y=result["avg_time"] * 1000,
        color="orange",
        linestyle="--",
        label=f'Ср: {result["avg_time"] * 1000:.2f} мс',
    )
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Время (мс)")
    ax.set_title(time_panel_title)
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    controls = result["controls"]
    ax.plot(controls[:, 0], "b-", label="v")
    ax.plot(controls[:, 1], "r-", label="w")
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Управление")
    ax.set_title("Сигналы")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"График сохранён: {save_path}")
    plt.show()
