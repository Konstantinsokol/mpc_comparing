"""Simulation harness for the OSQP-based unicycle MPC.

This script is intentionally kept solver-facing rather than framework-heavy:
it builds a straight global path, extracts obstacle snapshots from `ir-sim`,
constructs a local lookahead reference, calls the MPC at every step, and
visualizes the resulting closed loop.
"""

import os
import sys

import cv2
import irsim
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MPC.unicycle_mpc_osqp_tracking import UnicycleMPC_OSQP_Tracking


def resolve_yaml_path():
    """Resolve the scenario YAML passed from CLI or fall back to free_path."""
    if len(sys.argv) <= 1:
        return os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "scenarios",
                "free_path",
                "free_path.yaml",
            )
        )

    scenario_arg = sys.argv[1]
    if scenario_arg.endswith(".yaml"):
        return os.path.abspath(scenario_arg)

    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scenarios",
            scenario_arg,
            f"{scenario_arg}.yaml",
        )
    )


def get_obstacles_from_env(env):
    """Extract obstacle geometry in the format expected by the MPC solver.

    Each obstacle is converted to a dictionary with:
    - `position`: current center in the world frame,
    - `velocity`: planar obstacle velocity used for prediction,
    - `radius`: effective collision radius.
    """
    obstacles = []
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
        # print(f"\nvel = {vel[0]:.2f}, pos = ({pos[0]:.2f}, {pos[1]:.2f}), radius={r:.2f})")
        obstacles.append({"position": pos, "velocity": vel, "radius": r})

    return obstacles


def build_straight_path(start, goal, step_size):
    """Build the global nominal path used by the local MPC tracker.

    In this project the global planner is intentionally minimal: we use one
    straight line from the current start state to the goal and let the local
    MPC layer handle obstacle avoidance around that line.
    """
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


def generate_reference_trajectory(path, state, horizon, last_idx=0, obstacles=None):
    """Construct the local horizon reference passed into the MPC.

    The procedure is deliberately simple and deterministic:
    1. find the closest point on the global straight path without jumping back,
    2. apply a fixed lookahead,
    3. fill the next `horizon` points,
    4. recompute heading from the geometric tangent of the local path.

    This gives the controller a smooth tracking target while still letting
    obstacle handling happen inside the MPC layer.
    """

    ref_traj = np.zeros((horizon + 1, 3))
    ref_traj[0] = state.copy()

    # 1. Find the closest path point in a forward-only search window. This
    # prevents the reference index from oscillating backward.
    search_window = 30  # ограничение поиска вперёд
    end_idx = min(len(path), last_idx + search_window)

    distances = np.linalg.norm(path[last_idx:end_idx, :2] - state[:2], axis=1)
    nearest_local = int(np.argmin(distances))
    nearest_idx = last_idx + nearest_local

    # 2. Use a small lookahead so that the local reference stays slightly ahead
    # of the robot instead of collapsing to the nearest point.
    LOOKAHEAD = 5

    for k in range(1, horizon + 1):
        idx = min(nearest_idx + LOOKAHEAD + k, len(path) - 1)
        ref_traj[k] = path[idx]

    # 3. Recompute heading from consecutive reference points. This makes the
    # angular component of the reference consistent with the path geometry.
    for k in range(horizon):
        dx = ref_traj[k + 1, 0] - ref_traj[k, 0]
        dy = ref_traj[k + 1, 1] - ref_traj[k, 1]

        if np.hypot(dx, dy) > 1e-6:
            ref_traj[k, 2] = np.arctan2(dy, dx)
        else:
            ref_traj[k, 2] = ref_traj[k - 1, 2] if k > 0 else state[2]

    ref_traj[-1, 2] = ref_traj[-2, 2]

    return ref_traj, nearest_idx


def render_custom(
    env,
    state,
    goal,
    ref_traj,
    path_line,
    trajectory,
    mpc=None,
    scale=50,
    offset=(50, 50),
):
    """Render the current closed-loop state using OpenCV.

    Visual layers:
    - background grid,
    - global straight path,
    - local MPC reference,
    - optional safe corridor diagnostics,
    - realized robot trajectory,
    - current MPC prediction,
    - robot, goal, and obstacle forecasts.
    """
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255

    for i in range(11):
        x = int(offset[0] + i * scale)
        y = int(offset[1] + i * scale)
        cv2.line(img, (x, offset[1]), (x, offset[1] + 10 * scale), (200, 200, 200), 1)
        cv2.line(img, (offset[0], y), (offset[0] + 10 * scale, y), (200, 200, 200), 1)

    if path_line is not None:
        for i in range(len(path_line) - 1):
            pt1 = (int(offset[0] + path_line[i, 0] * scale), int(offset[1] + path_line[i, 1] * scale))
            pt2 = (int(offset[0] + path_line[i + 1, 0] * scale), int(offset[1] + path_line[i + 1, 1] * scale))
            cv2.line(img, pt1, pt2, (180, 180, 0), 1, cv2.LINE_AA)

    if ref_traj is not None:
        for i in range(len(ref_traj) - 1):
            pt1 = (int(offset[0] + ref_traj[i, 0] * scale), int(offset[1] + ref_traj[i, 1] * scale))
            pt2 = (int(offset[0] + ref_traj[i + 1, 0] * scale), int(offset[1] + ref_traj[i + 1, 1] * scale))
            cv2.line(img, pt1, pt2, (0, 180, 0), 1, cv2.LINE_AA)
            cv2.circle(img, pt1, 2, (0, 150, 0), -1)

    if mpc is not None and getattr(mpc, "last_corridor_polygons", None):
        overlay = img.copy()
        for poly in mpc.last_corridor_polygons:
            pts = np.array(
                [
                    [
                        int(offset[0] + p[0] * scale),
                        int(offset[1] + p[1] * scale),
                    ]
                    for p in poly
                ],
                dtype=np.int32,
            )
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], (210, 235, 255))
                cv2.polylines(img, [pts], True, (140, 170, 220), 1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)

    for i in range(1, len(trajectory)):
        pt1 = (int(offset[0] + trajectory[i - 1][0] * scale), int(offset[1] + trajectory[i - 1][1] * scale))
        pt2 = (int(offset[0] + trajectory[i][0] * scale), int(offset[1] + trajectory[i][1] * scale))
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    if mpc is not None and mpc.warm_start_x is not None:
        pred = mpc.warm_start_x
        for i in range(len(pred) - 1):
            pt1 = (int(offset[0] + pred[i, 0] * scale), int(offset[1] + pred[i, 1] * scale))
            pt2 = (int(offset[0] + pred[i + 1, 0] * scale), int(offset[1] + pred[i + 1, 1] * scale))
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(img, pt2, 3, (0, 200, 0), -1)

    robot_pos = (int(offset[0] + state[0] * scale), int(offset[1] + state[1] * scale))
    cv2.circle(img, robot_pos, int(0.2 * scale), (255, 0, 0), -1)

    arrow_len = 0.5 * scale
    arrow_x = int(robot_pos[0] + arrow_len * np.cos(state[2]))
    arrow_y = int(robot_pos[1] + arrow_len * np.sin(state[2]))
    cv2.arrowedLine(img, robot_pos, (arrow_x, arrow_y), (0, 0, 255), 2)

    goal_pos = (int(offset[0] + goal[0] * scale), int(offset[1] + goal[1] * scale))
    cv2.drawMarker(img, goal_pos, (0, 0, 255), cv2.MARKER_STAR, 20, 2)

    raw_obs = getattr(env, "obstacle_list", [])
    for obs in raw_obs:
        try:
            pos = np.array(obs.state[:2]).flatten()
            vel = np.zeros(2)
            if hasattr(obs, "velocity") and obs.velocity is not None:
                vel = np.array(obs.velocity).flatten()[:2]

            r = 0.2
            if hasattr(obs, "shape_inf"):
                if isinstance(obs.shape_inf, (list, np.ndarray)):
                    r = float(obs.shape_inf[0])
                else:
                    r = float(obs.shape_inf)

            obs_pt = (int(offset[0] + pos[0] * scale), int(offset[1] + pos[1] * scale))
            cv2.circle(img, obs_pt, int(r * scale), (100, 100, 100), -1)
            cv2.circle(img, obs_pt, int(r * scale), (0, 0, 0), 1)

            if np.linalg.norm(vel) > 0.01:
                for t in np.linspace(0.5, 2.0, 4):
                    pred_x = pos[0] + vel[0] * t
                    pred_y = pos[1] + vel[1] * t
                    pred_pt = (int(offset[0] + pred_x * scale), int(offset[1] + pred_y * scale))
                    cv2.circle(img, pred_pt, int(r * scale * 0.7), (150, 150, 150), -1)
                    cv2.circle(img, pred_pt, int(r * scale * 0.7), (0, 0, 0), 1)
        except Exception:
            continue

    cv2.imshow("MPC Tracking OSQP", img)
    cv2.waitKey(1)


def run_simulation(env, mpc, goal, max_steps=500, verbose=True):
    """Execute the receding-horizon loop for the OSQP controller.

    Loop structure:
    1. read the current robot state,
    2. extract obstacle states from the simulator,
    3. build the local reference window,
    4. solve the MPC,
    5. apply only the first control input,
    6. repeat at the next simulator step.
    """
    state = env.get_robot_state().flatten()
    trajectory = [state.copy()]
    solve_times = []
    distances = []
    controls = []

    print(f"\n[OSQP] Старт: ({state[0]:.2f}, {state[1]:.2f}, theta={state[2]:.2f}), Цель: ({goal[0]:.2f}, {goal[1]:.2f})")

    # The global path is fixed for the whole rollout; only the local window on
    # that path is shifted online.
    path_line = build_straight_path(state.copy(), goal, step_size=max(mpc.dt * mpc.v_max, 0.15))
    last_idx = 0
    for step in range(max_steps):
        # Build the exact data passed into the solver at this MPC iteration.
        obstacles = get_obstacles_from_env(env)
        ref_traj, last_idx = generate_reference_trajectory(
            path_line,
            state,
            mpc.N,
            last_idx,
            obstacles
        )
        u, solve_time = mpc.solve(state, ref_traj, obstacles)

        # Receding-horizon principle: apply only u_0, then resolve at the next
        # state measurement.
        env.step(np.array([[u[0]], [u[1]]]))

        state = env.get_robot_state().flatten()
        trajectory.append(state.copy())
        solve_times.append(solve_time)
        controls.append(u.copy())

        dist = np.linalg.norm(state[:2] - goal[:2])
        distances.append(dist)

        render_custom(env, state, goal, ref_traj, path_line, trajectory, mpc=mpc)

        if verbose:
            print(f"Step {step:3d} | Time: {solve_time * 1000:6.2f} ms | Dist: {dist:.3f} | v={u[0]:.3f}, w={u[1]:.3f}")

        if dist < 0.2:
            print(f"\n[OSQP] Цель достигнута за {step + 1} шагов")
            break

        if env.done():
            break

    cv2.destroyAllWindows()

    return {
        "trajectory": np.array(trajectory),
        "reference_path": path_line,
        "solve_times": solve_times,
        "distances": distances,
        "controls": np.array(controls),
        "steps": len(solve_times),
        "total_time": sum(solve_times),
        "avg_time": np.mean(solve_times) if solve_times else 0.0,
        "final_dist": distances[-1] if distances else np.inf,
    }


def plot_results(result, goal, env=None, save_path="mpc_tracking_osqp_results.png"):
    """Plot the main closed-loop diagnostics after the simulation."""
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
                    if isinstance(obs.shape_inf, (list, np.ndarray)):
                        r = float(obs.shape_inf[0])
                    else:
                        r = float(obs.shape_inf)
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
    ax.set_title("Время OSQP")
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


def main():
    """Configure the environment and launch the OSQP tracking experiment."""
    yaml_path = resolve_yaml_path()

    print("=" * 70)
    print("MPC TRACKING С OSQP")
    print("=" * 70)
    print(f"YAML: {yaml_path}")

    if not os.path.exists(yaml_path):
        print("ОШИБКА: Файл не найден!")
        return

    env = irsim.make(yaml_path)
    goal = np.array(env.robot.goal).flatten()

    # The tuning block is kept here, not inside the solver, so experiments can
    # compare OSQP / ActiveSet / IPOPT under explicit and reproducible settings.
    mpc_config = {
        "dt": env.step_time,
        "horizon": 25,
        "q_pos": 5.0,
        "q_theta": 2.0,
        "r": 0.1,
        "r_yaw": 0.1,
        "integration_method": "rk4", #rk4, euler
        "v_max": 1.0,
        "w_max": 1.0,
        "dv_max": 0.25,
        "dw_max": 0.5,
        "safe_distance": 0.4,
        "n_obs": 10,
        "slack_weight": 300.0,
        "max_active_obs": 5,
        "obs_horizon": 6,
        "activation_margin": 3.0,
        "progress_weight": 4.5,
        "ttc_threshold": 4.0,
        "tangent_bias_gain": 0.3,
        "obstacle_slowdown_margin": 1.5,
        "min_speed_factor_near_obstacle": 0.8,
        "dynamic_obstacle_speed_threshold": 0.05,
    }

    mpc = UnicycleMPC_OSQP_Tracking(**mpc_config)

    result = run_simulation(env, mpc, goal, max_steps=500)
    env.end()

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ OSQP")
    print("=" * 70)
    print(f"Шагов: {result['steps']}")
    print(f"Ср. время: {result['avg_time'] * 1000:.2f} мс")
    print(f"Фин. расстояние: {result['final_dist']:.4f}")

    plot_results(result, goal, env, save_path="mpc_tracking_osqp_results.png")


if __name__ == "__main__":
    main()
