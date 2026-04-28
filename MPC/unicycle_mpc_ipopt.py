import numpy as np
import casadi as ca
import time

class UnicycleMPC:
    """MPC-контроллер для unicycle с учетом динамических препятствий.

    Препятствия прогнозируются по модели постоянной скорости.
    """
    
    def __init__(self, dt=0.1, horizon=10, q_pos=10.0, q_theta=1.0, r=0.1,
                 v_max=1.0, w_max=1.0, safe_distance=0.8, n_obs=20):
        """
        Args:
            dt: шаг времени
            horizon: горизонт предсказания
            q_pos: вес позиции терминальной точки
            q_theta: вес ориентации терминальной точки
            r: вес управления
            v_max: макс. линейная скорость
            w_max: макс. угловая скорость
            safe_distance: безопасное расстояние до препятствий
            n_obs: максимальное число препятствий (для фикс. размерности NLP)
        """
        self.dt = dt
        self.N = horizon
        self.nx = 3  # [x, y, theta]
        self.nu = 2  # [v, w]
        self.v_max = v_max
        self.w_max = w_max
        self.safe_distance = safe_distance
        self.n_obs = n_obs
        
        # Весовые матрицы
        self.Q = np.diag([q_pos, q_pos, q_theta])
        self.R = np.diag([r, r])
        
        # Тёплый старт
        self.warm_start_x = None  # предыдущее состояние для тёплого старта
        self.warm_start_u = None  # предыдущие управления для тёплого старта
        
        self._build_solver()
    
    def _build_solver(self):
        """Строит символическую NLP-задачу и создаёт IPOPT-солвер."""
        N, nx, nu, n_obs = self.N, self.nx, self.nu, self.n_obs
        
        # Переменные оптимизации
        X = ca.SX.sym('X', nx, N + 1)  # состояния
        U = ca.SX.sym('U', nu, N)      # управления
        
        # Параметры задачи
        x0_param = ca.SX.sym('x0', nx)          # начальное состояние
        goal_param = ca.SX.sym('goal', 2)       # цель [x, y]
        
        # Параметры препятствий
        obs_pos_x = ca.SX.sym('obs_x', n_obs)   # позиция препятствия по иксу
        obs_pos_y = ca.SX.sym('obs_y', n_obs)   # позиция препятствия по игреку
        obs_vx = ca.SX.sym('obs_vx', n_obs)     # скорость препятствия по X
        obs_vy = ca.SX.sym('obs_vy', n_obs)     # скорость препятствия по Y
        obs_radius = ca.SX.sym('obs_r', n_obs)  # радиус препятствия
        obs_active = ca.SX.sym('obs_active', n_obs)  # 1 если препятствие присутствует
        
        # ===== ЦЕЛЕВАЯ ФУНКЦИЯ =====
        cost = 0
        goal_state = ca.vertcat(goal_param[0], goal_param[1], 0)  # цель в пространстве [x, y, theta]
        
        for k in range(N):
            state_err = X[:, k] - goal_state
            ctrl = U[:, k]
            cost += ca.mtimes([state_err.T, self.Q, state_err])  # штраф на расхождение с целью
            cost += ca.mtimes([ctrl.T, self.R, ctrl])  # регуляризация управляющих сигналов
        
        # Терминальный штраф (сильнее)
        term_err = X[:, N] - goal_state
        cost += ca.mtimes([term_err.T, self.Q * 5, term_err])  # усиленный терминальный штраф
        
        # ===== ОГРАНИЧЕНИЯ =====
        g = []
        
        # Динамика робота
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            x_next = ca.vertcat(
                xk[0] + uk[0] * ca.cos(xk[2]) * self.dt,
                xk[1] + uk[0] * ca.sin(xk[2]) * self.dt,
                xk[2] + uk[1] * self.dt
            )
            g.append(X[:, k+1] - x_next)  # дискретные уравнения динамики
        
        # Начальное условие
        g.append(X[:, 0] - x0_param)  # фиксируем начальное состояние
        
        # Ограничения на препятствия (динамические)
        for k in range(1, N + 1):  # пропускаем k=0 (оно фиксировано начальным условием)
            robot_x = X[0, k]
            robot_y = X[1, k]
            
            for i in range(n_obs):
                # Прогноз позиции препятствия на шаг k
                obs_x_k = obs_pos_x[i] + obs_vx[i] * (k * self.dt)
                obs_y_k = obs_pos_y[i] + obs_vy[i] * (k * self.dt)
                
                dist_sq = (robot_x - obs_x_k)**2 + (robot_y - obs_y_k)**2
                min_dist = self.safe_distance + obs_radius[i]
                
                # Ограничение: dist >= min_dist  =>  dist_sq - min_dist^2 >= 0
                constraint = dist_sq - min_dist**2
                g.append(obs_active[i] * constraint)  # активное препятствие добавляет ограничение
        
        # ===== ФОРМИРОВАНИЕ NLP =====
        X_vec = ca.reshape(X, -1, 1)
        U_vec = ca.reshape(U, -1, 1)
        vars_sym = ca.vertcat(X_vec, U_vec)
        
        # Границы переменных
        lbx = -ca.inf * ca.DM.ones(vars_sym.shape)
        ubx = ca.inf * ca.DM.ones(vars_sym.shape)
        
        u_start = nx * (N + 1)
        for k in range(N):
            lbx[u_start + k*nu] = 0
            ubx[u_start + k*nu] = self.v_max
            lbx[u_start + k*nu + 1] = -self.w_max
            ubx[u_start + k*nu + 1] = self.w_max
        
        # Границы ограничений
        g_concat = ca.vertcat(*g)
        n_dyn = N * nx + nx  # динамика + начальное условие
        
        lbg = ca.DM.zeros(g_concat.shape)
        ubg = ca.inf * ca.DM.ones(g_concat.shape)
        
        # Динамика и начальное условие — равенства
        lbg[:n_dyn] = 0
        ubg[:n_dyn] = 0
        # Препятствия — неравенства >= 0
        lbg[n_dyn:] = 0
        ubg[n_dyn:] = ca.inf
        
        # Параметры
        p = ca.vertcat(
            x0_param, goal_param,
            obs_pos_x, obs_pos_y, obs_vx, obs_vy, obs_radius, obs_active
        )
        
        nlp = {'x': vars_sym, 'f': cost, 'g': g_concat, 'p': p}
        
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 10,
            'ipopt.warm_start_init_point': 'yes',
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        self.u_start = u_start
        self.n_vars = nx * (N + 1) + nu * N
    
    def solve(self, state, goal, obstacles=None):
        """
        Решает задачу MPC.
        
        Args:
            state: [x, y, theta]
            goal: [x, y]
            obstacles: список словарей с ключами 'position', 'velocity', 'radius'
        
        Returns:
            u_opt: [v, w]
            solve_time: время решения
        """
        N, nx, nu, n_obs = self.N, self.nx, self.nu, self.n_obs
        
        # Подготовка параметров препятствий
        obs_x = np.zeros(n_obs)
        obs_y = np.zeros(n_obs)
        obs_vx = np.zeros(n_obs)
        obs_vy = np.zeros(n_obs)
        obs_r = np.zeros(n_obs)
        obs_active = np.zeros(n_obs)
        
        if obstacles:
            for i, obs in enumerate(obstacles[:n_obs]):
                obs_x[i] = obs['position'][0]
                obs_y[i] = obs['position'][1]
                obs_vx[i] = obs.get('velocity', [0, 0])[0]
                obs_vy[i] = obs.get('velocity', [0, 0])[1]
                obs_r[i] = obs.get('radius', 0.3)
                obs_active[i] = 1.0
        
        # Собираем параметры (состояние, цель и препятствия) в одном векторе
        p = np.concatenate([
            state,
            goal[:2],
            obs_x, obs_y, obs_vx, obs_vy, obs_r, obs_active
        ])
        
        # ===== ТЁПЛЫЙ СТАРТ =====
        if (self.warm_start_x is not None and self.warm_start_u is not None and
            self.warm_start_x.shape == (N + 1, nx) and self.warm_start_u.shape == (N, nu)):
            # Сдвигаем предыдущую траекторию и управления для теплового старта
            x0 = np.zeros(self.n_vars)
            
            # Сдвигаем состояния
            for k in range(N):
                x0[k*nx:(k+1)*nx] = self.warm_start_x[k+1, :]
            
            # Экстраполируем последнее состояние
            last_x = self.warm_start_x[-1, 0]
            last_y = self.warm_start_x[-1, 1]
            last_theta = self.warm_start_x[-1, 2]
            last_v = self.warm_start_u[-1, 0]
            last_w = self.warm_start_u[-1, 1]
            
            x0[N*nx] = last_x + last_v * np.cos(last_theta) * self.dt
            x0[N*nx+1] = last_y + last_v * np.sin(last_theta) * self.dt
            x0[N*nx+2] = last_theta + last_w * self.dt
            
            # Сдвигаем управления
            for k in range(N-1):
                x0[self.u_start + k*nu:self.u_start + (k+1)*nu] = self.warm_start_u[k+1, :]
            # Последнее управление — нулевое
            x0[self.u_start + (N-1)*nu:self.u_start + N*nu] = [0.0, 0.0]
        else:
            # Холодный старт: прямая к цели
            x0 = np.zeros(self.n_vars)
            dx = goal[0] - state[0]
            dy = goal[1] - state[1]
            desired_theta = np.arctan2(dy, dx)
            
            # Интерполируем путь состояний между текущим положением и целью
            for k in range(N + 1):
                alpha = k / max(N, 1)
                x0[k*nx] = state[0] * (1 - alpha) + goal[0] * alpha
                x0[k*nx+1] = state[1] * (1 - alpha) + goal[1] * alpha
                x0[k*nx+2] = desired_theta
            
            # Начальное приближение для управляющих воздействий: небольшая скорость вперед
            for k in range(N):
                x0[self.u_start + k*nu] = min(0.5, self.v_max)
                x0[self.u_start + k*nu+1] = 0.0
        
        # ===== РЕШЕНИЕ =====
        start = time.time()
        try:
            sol = self.solver(
                x0=x0, p=p,
                lbx=self.lbx, ubx=self.ubx,
                lbg=self.lbg, ubg=self.ubg
            )
            
            # Сохраняем для следующего тёплого старта
            self.warm_start_x = sol['x'][:nx*(N+1)].full().reshape(N+1, nx)
            self.warm_start_u = sol['x'][self.u_start:].full().reshape(N, nu)
            
            u_opt = self.warm_start_u[0, :]
            solve_time = time.time() - start
            
        except Exception as e:
            # Если IPOPT не сходится, сбрасываем тёплый старт и выдаём нулевый шаг
            print(f"[MPC] IPOPT failed: {e}, using fallback")
            self.warm_start_x = None
            self.warm_start_u = None
            u_opt = np.array([0, 0])
            solve_time = 0.0
        
        return u_opt, solve_time
