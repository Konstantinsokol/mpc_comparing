"""OpenCV visualization for unicycle MPC tracking simulations.

Shared by IPOPT / OSQP / ActiveSet harnesses so render logic stays in one place.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


def build_tracking_frame_image(
    env: Any,
    state: np.ndarray,
    goal: np.ndarray,
    ref_traj: Optional[np.ndarray],
    path_line: Optional[np.ndarray],
    trajectory: Sequence[np.ndarray],
    mpc: Any = None,
    *,
    scale: int = 50,
    offset: Tuple[int, int] = (50, 50),
    show_corridor: bool = True,
) -> np.ndarray:
    """Return one BGR frame (uint8) — same picture as shown in the OpenCV window."""
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    ox, oy = int(offset[0]), int(offset[1])

    grid_cells = 10
    for i in range(grid_cells + 1):
        x = ox + i * scale
        y = oy + i * scale
        cv2.line(img, (x, oy), (x, oy + grid_cells * scale), (200, 200, 200), 1)
        cv2.line(img, (ox, y), (ox + grid_cells * scale, y), (200, 200, 200), 1)

    def wx(p: Union[np.ndarray, Sequence[float]]) -> int:
        return ox + int(float(p[0]) * scale)

    def wy(p: Union[np.ndarray, Sequence[float]]) -> int:
        return oy + int(float(p[1]) * scale)

    if path_line is not None:
        for i in range(len(path_line) - 1):
            pt1 = (wx(path_line[i]), wy(path_line[i]))
            pt2 = (wx(path_line[i + 1]), wy(path_line[i + 1]))
            cv2.line(img, pt1, pt2, (180, 180, 0), 1, cv2.LINE_AA)

    if ref_traj is not None:
        for i in range(len(ref_traj) - 1):
            pt1 = (wx(ref_traj[i]), wy(ref_traj[i]))
            pt2 = (wx(ref_traj[i + 1]), wy(ref_traj[i + 1]))
            cv2.line(img, pt1, pt2, (0, 180, 0), 1, cv2.LINE_AA)
            cv2.circle(img, pt1, 2, (0, 150, 0), -1)

    if show_corridor and mpc is not None and getattr(mpc, "last_corridor_polygons", None):
        overlay = img.copy()
        for poly in mpc.last_corridor_polygons:
            pts = np.array([[wx(p), wy(p)] for p in poly], dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], (210, 235, 255))
                cv2.polylines(img, [pts], True, (140, 170, 220), 1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)

    for i in range(1, len(trajectory)):
        pt1 = (wx(trajectory[i - 1]), wy(trajectory[i - 1]))
        pt2 = (wx(trajectory[i]), wy(trajectory[i]))
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    if mpc is not None and getattr(mpc, "warm_start_x", None) is not None:
        pred = mpc.warm_start_x
        for i in range(len(pred) - 1):
            pt1 = (wx(pred[i]), wy(pred[i]))
            pt2 = (wx(pred[i + 1]), wy(pred[i + 1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(img, pt2, 3, (0, 200, 0), -1)

    robot_pos = (wx(state), wy(state))
    cv2.circle(img, robot_pos, int(0.2 * scale), (255, 0, 0), -1)

    arrow_len = 0.5 * scale
    arrow_x = int(robot_pos[0] + arrow_len * np.cos(state[2]))
    arrow_y = int(robot_pos[1] + arrow_len * np.sin(state[2]))
    cv2.arrowedLine(img, robot_pos, (arrow_x, arrow_y), (0, 0, 255), 2)

    goal_pos = (wx(goal), wy(goal))
    cv2.drawMarker(img, goal_pos, (0, 0, 255), cv2.MARKER_STAR, 20, 2)

    raw_obs: List[Any] = getattr(env, "obstacle_list", [])
    for obs in raw_obs:
        try:
            pos = np.array(obs.state[:2]).flatten()
            vel = np.zeros(2)
            if hasattr(obs, "velocity") and obs.velocity is not None:
                vel = np.array(obs.velocity).flatten()[:2]

            r = 0.2
            if hasattr(obs, "shape_inf"):
                s_inf = obs.shape_inf
                if isinstance(s_inf, (list, np.ndarray)):
                    r = float(s_inf[0])
                else:
                    r = float(s_inf)

            obs_pt = (wx(pos), wy(pos))
            cv2.circle(img, obs_pt, int(r * scale), (100, 100, 100), -1)
            cv2.circle(img, obs_pt, int(r * scale), (0, 0, 0), 1)

            if np.linalg.norm(vel) > 0.01:
                for t in np.linspace(0.5, 2.0, 4):
                    pred_x = pos[0] + vel[0] * t
                    pred_y = pos[1] + vel[1] * t
                    pred_pt = (wx([pred_x, pred_y]), wy([pred_x, pred_y]))
                    cv2.circle(img, pred_pt, int(r * scale * 0.7), (150, 150, 150), -1)
                    cv2.circle(img, pred_pt, int(r * scale * 0.7), (0, 0, 0), 1)
        except Exception:
            continue

    return img


def render_tracking_frame(
    env: Any,
    state: np.ndarray,
    goal: np.ndarray,
    ref_traj: Optional[np.ndarray],
    path_line: Optional[np.ndarray],
    trajectory: Sequence[np.ndarray],
    mpc: Any = None,
    *,
    window_title: str = "MPC Tracking",
    scale: int = 50,
    offset: Tuple[int, int] = (50, 50),
    show_corridor: bool = True,
    display: bool = True,
) -> np.ndarray:
    """Draw one frame; optionally show in a window. Returns BGR image (for GIF / video pipelines)."""
    img = build_tracking_frame_image(
        env,
        state,
        goal,
        ref_traj,
        path_line,
        trajectory,
        mpc=mpc,
        scale=scale,
        offset=offset,
        show_corridor=show_corridor,
    )
    if display:
        cv2.imshow(window_title, img)
        cv2.waitKey(1)
    return img
