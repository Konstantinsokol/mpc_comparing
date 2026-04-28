"""OSQP-based tracking MPC for a unicycle.

Problem formulation
    At every control step we solve a convex QP obtained from local
    linearization of the nonlinear unicycle dynamics and obstacle constraints.

Decision variables
    z = [x_0, ..., x_N, u_0, ..., u_{N-1}, s]^T
    where x_k are predicted states, u_k are predicted controls, and s are
    nonnegative slack variables for softened obstacle inequalities.

Cost
    0.5 z^T H z + g^T z
    with state-tracking terms, control regularization, terminal cost, and a
    quadratic penalty on obstacle slacks.

Constraints
    - initial-state equality,
    - affine dynamics x_{k+1} = A_k x_k + B_k u_k + c_k,
    - control box bounds,
    - rate limits on v and omega,
    - linearized obstacle half-spaces built around the predicted trajectory.
"""

import time
import casadi as ca
import numpy as np


class UnicycleMPC_OSQP_Tracking:
    """QP-based tracking MPC for a unicycle solved by OSQP.

    Problem formulation
        min_z 0.5 z^T H z + g^T z
        s.t.  lba <= A z <= uba
              lbx <= z <= ubx

    Decision variables
        z = [x_0, ..., x_N, u_0, ..., u_{N-1}, s]^T
        x_k in R^3, u_k in R^2, s >= 0.

    Cost
        - stage tracking (x_k-r_k)^T Q (x_k-r_k),
        - control regularization u_k^T R u_k,
        - terminal tracking with Q_N,
        - obstacle-slack penalty rho_s ||s||^2,
        - small linear forward bias on v_k.

    Constraints
        - x_0 equals the measured state,
        - x_{k+1} = A_k x_k + B_k u_k + c_k,
        - rate limits on v and omega,
        - linearized obstacle inequalities with slacks.
    """

    def __init__(
        self,
        dt=0.1,
        horizon=10,
        q_pos=10.0,
        q_theta=1.0,
        r=0.1,
        r_yaw=0.1,
        integration_method="rk4",
        v_max=1.0,
        w_max=1.0,
        dv_max=0.25,
        dw_max=0.5,
        safe_distance=0.4,
        n_obs=10,
        slack_weight=300.0,
        max_active_obs=5,
        obs_horizon=6,
        activation_margin=3.0,
        progress_weight=4.5,
        ttc_threshold=4.0,
        tangent_bias_gain=0.3,
        obstacle_slowdown_margin=1.5,
        min_speed_factor_near_obstacle=0.8,
        dynamic_obstacle_speed_threshold=0.05,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        self.dt = dt
        self.N = horizon
        self.nx = 3
        self.nu = 2
        self.integration_method = str(integration_method).lower()
        if self.integration_method not in {"euler", "rk4"}:
            raise ValueError("integration_method must be 'euler' or 'rk4'")

        self.v_max = v_max
        self.w_max = w_max
        self.dv_max = dv_max
        self.dw_max = dw_max
        self.safe_distance = safe_distance
        self.n_obs = n_obs
        self.slack_weight = slack_weight
        self.activation_margin = activation_margin
        self.progress_weight = progress_weight
        self.ttc_threshold = ttc_threshold
        self.tangent_bias_gain = tangent_bias_gain
        self.obstacle_slowdown_margin = obstacle_slowdown_margin
        self.min_speed_factor_near_obstacle = min_speed_factor_near_obstacle
        self.dynamic_obstacle_speed_threshold = dynamic_obstacle_speed_threshold

        self.max_active_obs = int(max_active_obs)
        self.obs_horizon = min(int(obs_horizon), self.N)

        self.Q = np.diag([q_pos, q_pos, q_theta])
        self.R = np.diag([r, r_yaw])
        self.Q_term = 5 * self.Q

        # Decision-vector dimensions and block structure:
        #
        #   z =
        #   [ x_0, ..., x_N | u_0, ..., u_{N-1} | s ]^T
        #
        # with
        #   x_k in R^3,
        #   u_k in R^2,
        #   s_(k,j) >= 0 for the obstacle-softening terms.
        self.n_state_vars = self.nx * (self.N + 1)
        self.n_control_vars = self.nu * self.N

        # Slacks are introduced only for the reduced active obstacle set, not
        # for every obstacle present in the environment.
        self.n_slack = self.obs_horizon * self.max_active_obs

        self.n_vars = (
            self.n_state_vars +
            self.n_control_vars +
            self.n_slack
        )

        self.n_constraints = (
            self.nx +
            self.N * self.nx +
            self.N * self.nu +
            self.obs_horizon * self.max_active_obs
        )

        # Explicit start/end indices of each block inside z.
        self.x_start = 0
        self.x_end = self.n_state_vars
        self.u_start = self.n_state_vars
        self.u_end = self.n_state_vars + self.n_control_vars
        self.slack_start = self.u_end
        self.slack_end = self.slack_start + self.n_slack

        self.warm_start_x = None
        self.warm_start_u = None
        self.prev_u_applied = np.zeros(self.nu)

        self._build_dynamics_functions()
        self._build_solver()

    # ---------------- solver ----------------

    def _build_solver(self):
        """Build a fixed-size OSQP template.

        CasADi's conic interface expects only the sparsity pattern at solver
        creation time. Numeric QP matrices are supplied later inside `solve()`.
        This is why we instantiate the solver once and only update `h`, `g`,
        `a`, `lba`, `uba`, `lbx`, and `ubx` online.
        """
        qp = {
            "h": ca.Sparsity.dense(self.n_vars, self.n_vars),
            "a": ca.Sparsity.dense(self.n_constraints, self.n_vars),
        }
        opts = {
            "verbose": False,
            "print_time": False,
            "error_on_fail": False,
            "osqp": {"verbose": False},
        }
        self.solver = ca.conic("osqp_mpc", "osqp", qp, opts)

    # ---------------- indexing ----------------

    def _state_slice(self, k):
        """Indices of x_k inside the stacked QP vector z."""
        return slice(k * self.nx, (k + 1) * self.nx)

    def _control_slice(self, k):
        """Indices of u_k inside the stacked QP vector z."""
        return slice(
            self.u_start + k * self.nu,
            self.u_start + (k + 1) * self.nu
        )

    def _slack_index(self, k, j):
        """Index of slack s_(k,j) for obstacle j at horizon step k."""
        return self.slack_start + k * self.max_active_obs + j

    # ---------------- dynamics ----------------

    def _continuous_dynamics_expr(self, x, u):
        """Continuous-time unicycle model f(x, u)."""
        return ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1],
        )

    def _discrete_step_expr(self, x, u):
        """One-step discretization x_{k+1} = f_d(x_k, u_k).

        We keep this function symbolic because:
        1. IPOPT uses the nonlinear map directly,
        2. OSQP / qpOASES use its Jacobians to build the local affine model
           x_{k+1} ~= A_k x_k + B_k u_k + c_k.
        """
        if self.integration_method == "rk4":
            k1 = self._continuous_dynamics_expr(x, u)
            k2 = self._continuous_dynamics_expr(x + 0.5 * self.dt * k1, u)
            k3 = self._continuous_dynamics_expr(x + 0.5 * self.dt * k2, u)
            k4 = self._continuous_dynamics_expr(x + self.dt * k3, u)
            return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x + self.dt * self._continuous_dynamics_expr(x, u)

    def _build_dynamics_functions(self):
        """Create CasADi functions for the discrete model and its Jacobians.

        A_k = d f_d / d x |_(x_ref, u_ref)
        B_k = d f_d / d u |_(x_ref, u_ref)

        These matrices define the local linear prediction model used by the QP.
        """
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        x_next = self._discrete_step_expr(x, u)
        A = ca.jacobian(x_next, x)
        B = ca.jacobian(x_next, u)
        self._step_fun = ca.Function("unicycle_step", [x, u], [x_next])
        self._A_fun = ca.Function("unicycle_A", [x, u], [A])
        self._B_fun = ca.Function("unicycle_B", [x, u], [B])

    def _linearized_dynamics(self, x_ref, u_ref):
        """Return the affine dynamics triple (A_k, B_k, c_k).

        The affine term is computed as

            c_k = f_d(x_ref, u_ref) - A_k x_ref - B_k u_ref

        so that the approximation is exact at the linearization point.
        """
        x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
        u_ref = np.asarray(u_ref, dtype=float).reshape(-1)
        A = np.array(self._A_fun(x_ref, u_ref), dtype=float)
        B = np.array(self._B_fun(x_ref, u_ref), dtype=float)
        f = np.array(self._step_fun(x_ref, u_ref), dtype=float).reshape(-1)
        c = f - A @ x_ref - B @ u_ref
        return A, B, c

    # ---------------- reference ----------------

    def _reference_inputs(self, ref):
        """Recover nominal controls from a state reference trajectory.

        This function maps a sequence of states into a sequence of nominal
        inputs. In the QP this nominal control is used only as a local
        linearization anchor, not as a hard feedforward command.
        """
        u_ref = np.zeros((self.N, self.nu))

        for k in range(self.N):
            dx = ref[k + 1, 0] - ref[k, 0]
            dy = ref[k + 1, 1] - ref[k, 1]
            dth = ref[k + 1, 2] - ref[k, 2]
            dth = (dth + np.pi) % (2 * np.pi) - np.pi

            u_ref[k, 0] = np.clip(np.hypot(dx, dy) / self.dt, 0.0, self.v_max)
            u_ref[k, 1] = np.clip(dth / self.dt, -self.w_max, self.w_max)

        return u_ref

    def _sanitize_reference(self, state, ref_trajectory):
        """Make the local reference dynamically plausible.

        The incoming path sample may request translational or angular jumps
        that are impossible under the current limits. We therefore clip:
        - Cartesian step length by v_max * dt,
        - heading increment by w_max * dt.

        This prevents the optimizer from being asked to track a trajectory that
        is kinematically infeasible right from the first horizon point.
        """
        ref = np.asarray(ref_trajectory, dtype=float).copy()
        if ref.shape != (self.N + 1, self.nx):
            raise ValueError(
                f"ref_trajectory должна иметь форму ({self.N + 1}, {self.nx}), "
                f"получено {ref.shape}"
            )

        ref[0] = state
        max_step = self.v_max * self.dt
        max_turn = self.w_max * self.dt

        for k in range(1, self.N + 1):
            prev = ref[k - 1].copy()
            cur = ref[k].copy()

            dxy = cur[:2] - prev[:2]
            dist = np.linalg.norm(dxy)
            if dist > max_step and dist > 1e-9:
                cur[:2] = prev[:2] + (max_step / dist) * dxy

            dtheta = cur[2] - prev[2]
            dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
            cur[2] = prev[2] + np.clip(dtheta, -max_turn, max_turn)
            ref[k] = cur

        for k in range(self.N):
            dxy = ref[k + 1, :2] - ref[k, :2]
            if np.linalg.norm(dxy) > 1e-9:
                ref[k, 2] = np.arctan2(dxy[1], dxy[0])
        ref[-1, 2] = ref[-2, 2]
        return ref

    def _build_safe_guess(self, state, ref_trajectory):
        """Build the trajectory used as the linearization anchor.

        If a warm start is available, we linearize around the previously
        predicted trajectory. Otherwise we fall back to the sanitized
        reference. This was one of the key fixes that improved obstacle
        handling, because linearizing around a trajectory already bending away
        from the obstacle is far better than linearizing around a straight line
        crossing the obstacle.
        """
        if (
            self.warm_start_x is not None
            and self.warm_start_x.shape == (self.N + 1, self.nx)
        ):
            guess = self.warm_start_x.copy()
            guess[0] = state
            return guess

        guess = ref_trajectory.copy()
        guess[0] = state
        return guess

    def _initial_guess(self, state, ref_trajectory):
        """Build the primal warm start vector for OSQP.

        The state part is shifted one step ahead, while the control part is
        shifted and zero-padded at the end. This is the classic receding-horizon
        warm-start strategy.
        """
        x0 = np.zeros(self.n_vars)

        if (
            self.warm_start_x is not None
            and self.warm_start_u is not None
            and self.warm_start_x.shape == (self.N + 1, self.nx)
            and self.warm_start_u.shape == (self.N, self.nu)
        ):
            for k in range(self.N):
                x0[self._state_slice(k)] = self.warm_start_x[k + 1]

            last_x, last_y, last_theta = self.warm_start_x[-1]
            last_v, last_w = self.warm_start_u[-1]
            x0[self._state_slice(self.N)] = [
                last_x + last_v * np.cos(last_theta) * self.dt,
                last_y + last_v * np.sin(last_theta) * self.dt,
                last_theta + last_w * self.dt,
            ]

            for k in range(self.N - 1):
                x0[self._control_slice(k)] = self.warm_start_u[k + 1]
            x0[self._control_slice(self.N - 1)] = [0.0, 0.0]
            return x0

        x0[:self.n_state_vars] = ref_trajectory.flatten()
        for k in range(self.N):
            x0[self._control_slice(k)] = [min(0.5, self.v_max), 0.0]
        x0[self._state_slice(0)] = state
        return x0

    # ---------------- obstacle selection ----------------

    def _select_obstacles(self, state, obstacles):
        """Select a bounded active obstacle subset.

        The QP size is fixed, so we only keep the closest obstacles up to
        `max_active_obs`. This is an engineering approximation trading some
        global obstacle awareness for real-time solvability.
        """
        if not obstacles:
            return []

        return sorted(
            obstacles,
            key=lambda o: np.linalg.norm(
                state[:2] - np.array(o["position"])[:2]
            )
        )[: min(self.max_active_obs, self.n_obs)]

    def _speed_limit_at_step(self, point_xy, obs_list, k):
        """Smoothly reduce forward speed near static obstacles.

        This function does not create a separate constraint row in A. Instead,
        it tightens the box constraint on v_k:

            0 <= v_k <= v_cap(k)

        The cap is softened with a smoothstep profile to avoid abrupt
        transitions in the velocity bound.
        """
        if not obs_list:
            return self.v_max

        point_xy = np.asarray(point_xy, dtype=float)[:2]

        nearest_clearance = np.inf
        for obs in obs_list:
            obs_pos = np.asarray(obs["position"], dtype=float)[:2]
            obs_vel = np.asarray(obs.get("velocity", [0.0, 0.0]), dtype=float)[:2]
            if np.linalg.norm(obs_vel) > self.dynamic_obstacle_speed_threshold:
                continue
            obs_radius = float(obs.get("radius", 0.2))
            obs_xy = obs_pos + obs_vel * (k * self.dt)
            safe = self.safe_distance + obs_radius
            clearance = np.linalg.norm(point_xy - obs_xy) - safe
            nearest_clearance = min(nearest_clearance, clearance)

        if not np.isfinite(nearest_clearance):
            return self.v_max

        if nearest_clearance <= 0.0:
            return self.min_speed_factor_near_obstacle * self.v_max

        margin = max(self.obstacle_slowdown_margin, 1e-6)
        if nearest_clearance >= margin:
            return self.v_max

        alpha = np.clip(nearest_clearance / margin, 0.0, 1.0)
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        min_speed = self.min_speed_factor_near_obstacle * self.v_max
        return min_speed + (self.v_max - min_speed) * smooth_alpha

    # ---------------- QP ----------------

    def _build_qp(self, state, ref, obstacles, traj_guess):
        """Assemble the full convex QP for the current MPC step.

        Matrix meanings:
        - H: quadratic cost Hessian,
        - g: linear cost term,
        - A: stacked linear constraints,
        - lba / uba: lower / upper bounds for A z,
        - lbx / ubx: direct bounds on z.
        """

        H = np.zeros((self.n_vars, self.n_vars))
        g = np.zeros(self.n_vars)

        A = np.zeros((self.n_constraints, self.n_vars))
        lba = np.zeros(self.n_constraints)
        uba = np.zeros(self.n_constraints)

        lbx = -np.inf * np.ones(self.n_vars)
        ubx = np.inf * np.ones(self.n_vars)

        u_ref = self._reference_inputs(traj_guess)

        # ---------------- COST ----------------
        # State tracking:
        #   sum_k (x_k - r_k)^T Q (x_k - r_k)
        # Control effort:
        #   sum_k u_k^T R u_k
        # plus a small linear reward on v_k to avoid the trivial "stop
        # forever" solution when a safe forward motion still exists.

        for k in range(self.N):
            xs = self._state_slice(k)
            us = self._control_slice(k)

            H[xs, xs] += 2 * self.Q
            H[us, us] += 2 * self.R
            g[xs] += -2 * self.Q @ ref[k]
            g[us.start] += -self.progress_weight

        xs = self._state_slice(self.N)
        H[xs, xs] += 2 * self.Q_term
        g[xs] += -2 * self.Q_term @ ref[self.N]

        # Slack penalty:
        #   rho_s * ||s||^2
        # This softens the linearized obstacle constraints and helps avoid
        # artificial infeasibility introduced by local convexification.
        for i in range(self.n_slack):
            idx = self.slack_start + i
            H[idx, idx] = 2 * self.slack_weight
            lbx[idx] = 0.0

        # Box bounds on controls. The upper bound on v is optionally reduced
        # near static obstacles by `_speed_limit_at_step`.
        obs_list = self._select_obstacles(state, obstacles)
        for k in range(self.N):
            us = self._control_slice(k)
            v_cap = self._speed_limit_at_step(traj_guess[k, :2], obs_list, k)
            lbx[us] = [0.0, -self.w_max]
            ubx[us] = [max(0.05, v_cap), self.w_max]

        # ---------------- constraints ----------------

        row = 0

        # Rate limits:
        #   |v_k - v_{k-1}| <= dv_max
        #   |w_k - w_{k-1}| <= dw_max
        #
        # The very first control is compared to the previously applied input,
        # which makes the closed-loop behavior smoother.
        for k in range(self.N):
            us = self._control_slice(k)

            if k == 0:
                A[row, us.start] = 1.0
                lba[row] = self.prev_u_applied[0] - self.dv_max
                uba[row] = self.prev_u_applied[0] + self.dv_max
                row += 1

                A[row, us.start + 1] = 1.0
                lba[row] = self.prev_u_applied[1] - self.dw_max
                uba[row] = self.prev_u_applied[1] + self.dw_max
                row += 1
            else:
                us_prev = self._control_slice(k - 1)

                A[row, us.start] = 1.0
                A[row, us_prev.start] = -1.0
                lba[row] = -self.dv_max
                uba[row] = self.dv_max
                row += 1

                A[row, us.start + 1] = 1.0
                A[row, us_prev.start + 1] = -1.0
                lba[row] = -self.dw_max
                uba[row] = self.dw_max
                row += 1

        # initial
        A[row:row+3, self._state_slice(0)] = np.eye(3)
        lba[row:row+3] = state
        uba[row:row+3] = state
        row += 3

        # Linearized dynamics:
        #   x_{k+1} - A_k x_k - B_k u_k = c_k
        for k in range(self.N):
            A_k, B_k, c_k = self._linearized_dynamics(traj_guess[k], u_ref[k])

            A[row:row+3, self._state_slice(k)] = -A_k
            A[row:row+3, self._state_slice(k+1)] = np.eye(3)
            A[row:row+3, self._control_slice(k)] = -B_k

            lba[row:row+3] = c_k
            uba[row:row+3] = c_k
            row += 3

        # ---------------- obstacles ----------------
        # For each active obstacle we linearize the nonconvex constraint
        #
        #   ||x_k - o_k|| >= safe
        #
        # around the guessed position x_hat_k = traj_guess[k].
        # This yields a supporting half-space with optional slack.
        for k in range(1, self.obs_horizon + 1):
            guess_xy = traj_guess[k, :2]
            xs = self._state_slice(k)

            # Оценка скорости робота на шаге k (используем опорное управление)
            if k-1 < len(u_ref):
                v_cmd = u_ref[k-1, 0]
                theta_ref = traj_guess[k-1, 2]
                v_rob = v_cmd * np.array([np.cos(theta_ref), np.sin(theta_ref)])
            else:
                v_rob = np.zeros(2)

            for j in range(self.max_active_obs):
                if j >= len(obs_list):
                    row += 1
                    continue

                obs = obs_list[j]
                pos = np.asarray(obs["position"], dtype=float)[:2]
                vel = np.asarray(obs.get("velocity", [0.0, 0.0]), dtype=float)[:2]
                r = float(obs.get("radius", 0.2))

                # Предсказанное положение препятствия на шаге k
                obs_xy = pos + vel * (k * self.dt)

                # Вектор от препятствия к опорной точке
                diff_guess = guess_xy - obs_xy
                dist_guess = np.linalg.norm(diff_guess)

                # Пороговое безопасное расстояние
                safe = self.safe_distance + r

                # ---------- Активация ограничения ----------
                # Используем TTC (Time-To-Collision) для раннего предупреждения
                rel_pos = state[:2] - obs_xy   # для TTC используем текущее положение
                rel_vel = v_rob - vel
                dist_cur = np.linalg.norm(rel_pos)
                closing_speed = -np.dot(rel_pos, rel_vel) / max(dist_cur, 1e-6)
                ttc = dist_cur / max(closing_speed, 1e-6) if closing_speed > 0 else np.inf

                # Пропускаем, если далеко и нет угрозы столкновения в ближайшее время
                if dist_guess > safe + self.activation_margin and ttc > self.ttc_threshold:
                    row += 1
                    continue

                # ---------- Вычисление нормали ----------
                if dist_guess < 1e-6:
                    # Опорная точка прямо на препятствии — используем направление от текущего положения робота
                    diff_cur = state[:2] - obs_xy
                    dcur = np.linalg.norm(diff_cur)
                    if dcur > 1e-6:
                        n = diff_cur / dcur
                    else:
                        # Экстренный случай: робот уже внутри препятствия
                        heading = state[2]
                        n = np.array([-np.sin(heading), np.cos(heading)])
                else:
                    n = diff_guess / dist_guess

                    # При встречном движении добавляем небольшую касательную компоненту,
                    # чтобы подтолкнуть оптимизацию к выбору стороны объезда.
                    if closing_speed > 0.3:   # относительная скорость > 0.3 м/с навстречу
                        tangent = np.array([-n[1], n[0]])
                        path_dir = ref[min(k + 1, self.N), :2] - ref[max(k - 1, 0), :2]
                        path_norm = np.linalg.norm(path_dir)
                        if path_norm < 1e-6:
                            heading = ref[max(k - 1, 0), 2]
                            path_dir = np.array([np.cos(heading), np.sin(heading)])
                        else:
                            path_dir = path_dir / path_norm

                        # Determine the passing side from the obstacle position
                        # relative to the local reference direction. This is more
                        # stable than using the current robot pose near the boundary.
                        rel_obs = obs_xy - ref[k, :2]
                        side = np.sign(path_dir[0] * rel_obs[1] - path_dir[1] * rel_obs[0])
                        side = 1.0 if abs(side) < 1e-6 else side
                        n = n + self.tangent_bias_gain * side * tangent
                        n /= np.linalg.norm(n)

                # Linearized obstacle condition:
                #
                #   n^T x_k + s_k >= safe + n^T x_hat_k - ||x_hat_k - o_k||
                #
                # where n is the outward normal from obstacle to the guessed
                # state x_hat_k.
                rhs = safe + np.dot(n, guess_xy) - dist_guess

                # Fill the constraint row in the global matrix A.
                A[row, xs.start] = n[0]       # коэффициент при x_k
                A[row, xs.start + 1] = n[1]   # коэффициент при y_k

                slack_idx = self._slack_index(k-1, j)
                A[row, slack_idx] = 1.0

                lba[row] = rhs
                uba[row] = np.inf
                row += 1
        return H, g, A, lba, uba, lbx, ubx

    # ---------------- solve ----------------

    def solve(self, state, ref, obstacles=None):
        """Solve one receding-horizon QP and return the first control input.

        The online loop is:
        1. sanitize reference,
        2. choose a trajectory around which to linearize,
        3. assemble the QP matrices,
        4. warm-start OSQP,
        5. extract the first control and store the rest as the next warm start.
        """
        state = np.asarray(state, dtype=float).flatten()
        ref = self._sanitize_reference(state, ref)

        start = time.time()
        traj_guess = self._build_safe_guess(state, ref)

        H, g, A, lba, uba, lbx, ubx = self._build_qp(
            state, ref, obstacles, traj_guess
        )

        args = dict(
            h=ca.DM(H),
            g=ca.DM(g),
            a=ca.DM(A),
            lba=ca.DM(lba),
            uba=ca.DM(uba),
            lbx=ca.DM(lbx),
            ubx=ca.DM(ubx),
        )

        # -------- warm start --------
        args["x0"] = ca.DM(self._initial_guess(state, ref))

        sol = self.solver(**args)
        stats = self.solver.stats()
        if not stats.get("success", False):
            self.warm_start_x = None
            self.warm_start_u = None
            self.prev_u_applied = np.zeros(self.nu)
            return np.array([0.0, 0.0]), 0.0

        x = np.array(sol["x"]).flatten()

        # update warm start
        self.warm_start_x = x[:self.n_state_vars].reshape(self.N + 1, self.nx)
        self.warm_start_u = x[self.u_start:self.u_start + self.n_control_vars].reshape(self.N, self.nu)

        u0 = self.warm_start_u[0]
        self.prev_u_applied = u0.copy()

        return u0, time.time() - start
