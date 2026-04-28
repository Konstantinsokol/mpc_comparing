"""Simulation harness for the OSQP-based unicycle MPC.

It builds a straight global path, extracts obstacle snapshots from `ir-sim`,
constructs a local lookahead reference, calls the MPC at every step, and
visualizes the resulting closed loop.
"""

import os
import sys

import irsim
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_sim_dir = os.path.dirname(os.path.abspath(__file__))
if _sim_dir not in sys.path:
    sys.path.insert(0, _sim_dir)

from mpc_tracking_common import (
    parse_tracking_cli,
    plot_tracking_results,
    run_tracking_simulation,
    TrackingRunStyle,
)
from MPC.unicycle_mpc_osqp_tracking import UnicycleMPC_OSQP_Tracking

_OSQP_STYLE = TrackingRunStyle(
    window_title="MPC Tracking OSQP",
    log_name="OSQP",
)


def run_simulation(
    env,
    mpc,
    goal,
    max_steps=500,
    verbose=True,
    render=True,
    gif_path=None,
    gif_fps=10.0,
):
    """Execute the receding-horizon loop for the OSQP controller."""
    return run_tracking_simulation(
        env,
        mpc,
        goal,
        style=_OSQP_STYLE,
        max_steps=max_steps,
        verbose=verbose,
        render=render,
        gif_path=gif_path,
        gif_fps=gif_fps,
    )


def plot_results(result, goal, env=None, save_path="mpc_tracking_osqp_results.png"):
    """Plot the main closed-loop diagnostics after the simulation."""
    plot_tracking_results(result, goal, env, save_path=save_path, time_panel_title="Время OSQP")


def main():
    """Configure the environment and launch the OSQP tracking experiment."""
    cli = parse_tracking_cli()
    yaml_path = cli.yaml_path

    print("=" * 70)
    print("MPC TRACKING С OSQP")
    print("=" * 70)
    print(f"YAML: {yaml_path}")

    if not os.path.exists(yaml_path):
        print("ОШИБКА: Файл не найден!")
        return

    env = irsim.make(yaml_path)
    goal = np.array(env.robot.goal).flatten()

    mpc_config = {
        "dt": env.step_time,
        "horizon": 25,
        "q_pos": 3.0,
        "q_theta": 2.0,
        "r": 0.1,
        "r_yaw": 0.1,
        "integration_method": "euler",
        "v_max": 1.0,
        "w_max": 1.0,
        "dv_max": 0.25,
        "dw_max": 0.5,
        "safe_distance": 0.35,
        "n_obs": 10,
        "slack_weight": 300.0,
        "max_active_obs": 5,
        "obs_horizon": 8,
        "activation_margin": 3.5,
        "progress_weight": 4.5,
        "ttc_threshold": 4.0,
        "tangent_bias_gain": 0.3,
        "obstacle_slowdown_margin": 1.5,
        "min_speed_factor_near_obstacle": 0.8,
        "dynamic_obstacle_speed_threshold": 0.05,
    }

    mpc = UnicycleMPC_OSQP_Tracking(**mpc_config)

    result = run_simulation(
        env,
        mpc,
        goal,
        max_steps=cli.max_steps,
        render=cli.render,
        gif_path=cli.gif_path,
        gif_fps=cli.gif_fps,
    )
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
