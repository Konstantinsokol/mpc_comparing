import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import irsim
from workspace.mpc.MPC.unicycle_mpc_ipopt import UnicycleMPC


def get_obstacles_from_env(env):
    """Извлекает препятствия с позициями и скоростями."""
    obstacles = []
    raw_obs = getattr(env, 'obstacle_list', [])
    
    for obs in raw_obs:
        try:
            pos = np.array(obs.state[:2]).flatten()
        except:
            continue

        r = 0.4
        if hasattr(obs, 'shape_inf'):
            s_inf = obs.shape_inf
            if isinstance(s_inf, (list, np.ndarray)):
                r = float(s_inf[0])
            else:
                r = float(s_inf)
        elif hasattr(obs, 'radius'):
            r = float(obs.radius)
            
        vel = np.zeros(2)
        if hasattr(obs, 'velocity') and obs.velocity is not None:
            vel = np.array(obs.velocity).flatten()[:2]
        elif len(obs.state) >= 5:
            vel = np.array(obs.state[3:5]).flatten()
        print(f"\nvel = {vel[0]:.2f}, pos = ({pos[0]:.2f}, {pos[1]:.2f}), radius={r:.2f})")
        obstacles.append({
            'position': pos,
            'velocity': vel,
            'radius': r
        })
        
    return obstacles


def render_custom(env, state, goal, obstacles, trajectory, mpc=None, scale=50, offset=(50, 50)):
    """Кастомная визуализация через OpenCV с прогнозами."""
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    
    # Сетка
    for i in range(11):
        x = int(offset[0] + i * scale)
        y = int(offset[1] + i * scale)
        cv2.line(img, (x, offset[1]), (x, offset[1] + 10*scale), (200, 200, 200), 1)
        cv2.line(img, (offset[0], y), (offset[0] + 10*scale, y), (200, 200, 200), 1)
    
    # Траектория робота
    for i in range(1, len(trajectory)):
        pt1 = (int(offset[0] + trajectory[i-1][0] * scale),
               int(offset[1] + trajectory[i-1][1] * scale))
        pt2 = (int(offset[0] + trajectory[i][0] * scale),
               int(offset[1] + trajectory[i][1] * scale))
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    
    # Предсказанная траектория MPC
    if mpc is not None and mpc.warm_start_x is not None:
        pred = mpc.warm_start_x
        for i in range(len(pred)-1):
            pt1 = (int(offset[0] + pred[i, 0] * scale),
                   int(offset[1] + pred[i, 1] * scale))
            pt2 = (int(offset[0] + pred[i+1, 0] * scale),
                   int(offset[1] + pred[i+1, 1] * scale))
            cv2.line(img, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(img, pt2, 3, (0, 200, 0), -1)
    
    # Робот
    robot_pos = (int(offset[0] + state[0] * scale),
                 int(offset[1] + state[1] * scale))
    cv2.circle(img, robot_pos, int(0.2 * scale), (255, 0, 0), -1)
    
    # Направление робота
    arrow_len = 0.5 * scale
    arrow_x = int(robot_pos[0] + arrow_len * np.cos(state[2]))
    arrow_y = int(robot_pos[1] + arrow_len * np.sin(state[2]))
    cv2.arrowedLine(img, robot_pos, (arrow_x, arrow_y), (0, 0, 255), 2)
    
    # Цель
    goal_pos = (int(offset[0] + goal[0] * scale),
                int(offset[1] + goal[1] * scale))
    cv2.drawMarker(img, goal_pos, (0, 0, 255), cv2.MARKER_STAR, 20, 2)
    
    # Препятствия с прогнозом
    raw_obs = getattr(env, 'obstacle_list', [])
    for obs in raw_obs:
        try:
            pos = np.array(obs.state[:2]).flatten()
            vel = np.zeros(2)
            if hasattr(obs, 'velocity') and obs.velocity is not None:
                vel = np.array(obs.velocity).flatten()[:2]
            
            r = 0.4
            if hasattr(obs, 'shape_inf'):
                r = float(obs.shape_inf[0]) if isinstance(obs.shape_inf, (list, np.ndarray)) else float(obs.shape_inf)
            
            obs_pt = (int(offset[0] + pos[0] * scale),
                      int(offset[1] + pos[1] * scale))
            cv2.circle(img, obs_pt, int(r * scale), (100, 100, 100), -1)
            cv2.circle(img, obs_pt, int(r * scale), (0, 0, 0), 1)
            
            # Прогноз траектории препятствия
            if np.linalg.norm(vel) > 0.01:
                for t in np.linspace(0.5, 2.0, 4):
                    pred_x = pos[0] + vel[0] * t
                    pred_y = pos[1] + vel[1] * t
                    pred_pt = (int(offset[0] + pred_x * scale),
                               int(offset[1] + pred_y * scale))
                    cv2.circle(img, pred_pt, int(r * scale * 0.7), (150, 150, 150), -1)
                    cv2.circle(img, pred_pt, int(r * scale * 0.7), (0, 0, 0), 1)
        except:
            continue
    
    cv2.imshow('MPC Simulation', img)
    cv2.waitKey(1)


def run_simulation(env, mpc, goal, max_steps=500, verbose=True):
    state = env.get_robot_state().flatten()
    trajectory = [state.copy()]
    solve_times = []
    distances = []
    controls = []
    
    print(f"\n[СИМУЛЯЦИЯ] Старт: ({state[0]:.2f}, {state[1]:.2f}), Цель: ({goal[0]:.2f}, {goal[1]:.2f})")
    
    for step in range(max_steps):
        obstacles = get_obstacles_from_env(env)
        u, solve_time = mpc.solve(state, goal, obstacles)
        
        action = np.array([[u[0]], [u[1]]])
        env.step(action)
        
        state = env.get_robot_state().flatten()
        trajectory.append(state.copy())
        solve_times.append(solve_time)
        controls.append(u.copy())
        
        dist = np.linalg.norm(state[:2] - goal[:2])
        distances.append(dist)
        
        # 🔥 Кастомная визуализация
        render_custom(env, state, goal, obstacles, trajectory, mpc=mpc)
        
        if verbose:
            print(f"Step {step:3d} | Time: {solve_time*1000:5.2f}ms | Dist: {dist:.3f} | v={u[0]:.3f}, w={u[1]:.3f}")
        
        if dist < 0.2:
            print(f"\nЦель достигнута за {step+1} шагов!")
            break
        
        if env.done():
            break
    
    cv2.destroyAllWindows()
    
    return {
        'trajectory': np.array(trajectory),
        'solve_times': solve_times,
        'distances': distances,
        'controls': np.array(controls),
        'steps': len(solve_times),
        'total_time': sum(solve_times),
        'avg_time': np.mean(solve_times) if solve_times else 0,
        'final_dist': distances[-1] if distances else np.inf
    }


def plot_results(result, goal, env=None, save_path='mpc_results.png'):
    traj = result['trajectory']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Траектория
    ax = axes[0, 0]
    ax.plot(traj[:, 0], traj[:, 1], 'b-o', markersize=3, label='Робот')
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', label='Старт', zorder=5)
    ax.scatter([goal[0]], [goal[1]], c='red', s=200, marker='*', label=f'Цель', zorder=5)
    
    if env and hasattr(env, 'obstacle_list'):
        for obs in env.obstacle_list:
            try:
                pos = np.array(obs.state[:2]).flatten()
                r = 0.4
                if hasattr(obs, 'shape_inf'):
                    r = float(obs.shape_inf[0]) if isinstance(obs.shape_inf, (list, np.ndarray)) else float(obs.shape_inf)
                circle = plt.Circle(pos, r, color='gray', alpha=0.5)
                ax.add_patch(circle)
            except:
                pass
    
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Траектория'); ax.legend(); ax.grid(True); ax.axis('equal')
    
    # Расстояние
    ax = axes[0, 1]
    ax.plot(result['distances'], 'b-')
    ax.axhline(y=0.2, color='r', linestyle='--')
    ax.set_xlabel('Шаг'); ax.set_ylabel('Расстояние')
    ax.set_title('Сходимость'); ax.grid(True)
    
    # Время решения
    ax = axes[1, 0]
    times_ms = np.array(result['solve_times']) * 1000
    ax.plot(times_ms, 'g-')
    ax.axhline(y=result['avg_time']*1000, color='orange', linestyle='--', label=f'Ср: {result["avg_time"]*1000:.2f}мс')
    ax.set_xlabel('Шаг'); ax.set_ylabel('Время (мс)')
    ax.set_title('Время IPOPT'); ax.legend(); ax.grid(True)
    
    # Управления
    ax = axes[1, 1]
    controls = result['controls']
    ax.plot(controls[:, 0], 'b-', label='v')
    ax.plot(controls[:, 1], 'r-', label='w')
    ax.set_xlabel('Шаг'); ax.set_ylabel('Управление')
    ax.set_title('Сигналы'); ax.legend(); ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"График сохранён: {save_path}")
    plt.show()


def main():
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "scenarios", "narrow_gap", "narrow_gap.yaml"
    )
    yaml_path = os.path.abspath(yaml_path)
    
    print("=" * 70)
    print("MPC С ДИНАМИЧЕСКИМИ ПРЕПЯТСТВИЯМИ")
    print("=" * 70)
    print(f"YAML: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        print("ОШИБКА: Файл не найден!")
        return
    
    env = irsim.make(yaml_path)
    goal = np.array(env.robot.goal).flatten()
    
    print(f"Цель: {goal[:2]}")
    
    mpc = UnicycleMPC(
        dt=env.step_time,
        horizon=20,
        q_pos=10.0,
        q_theta=2.0,
        r=0.1,
        v_max=1.0,
        w_max=1.0,
        safe_distance=0.3,
        n_obs=20
    )
    
    result = run_simulation(env, mpc, goal, max_steps=500)
    env.end()
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"Шагов: {result['steps']}")
    print(f"Ср. время: {result['avg_time']*1000:.2f} мс")
    print(f"Фин. расстояние: {result['final_dist']:.4f}")
    
    plot_results(result, goal, env, save_path='mpc_results.png')


if __name__ == "__main__":
    main()