"""qpOASES-based tracking MPC for a unicycle.

Problem formulation
    At every control step we solve the same locally convexified QP structure
    as in the OSQP controller, but with an online active-set backend.

Decision variables
    z = [x_0, ..., x_N, u_0, ..., u_{N-1}, s]^T.

Cost
    Quadratic state-tracking, control regularization, terminal penalty, and
    slack penalization for softened obstacle constraints.

Constraints
    - initial-state equality,
    - affine stage dynamics,
    - input bounds,
    - rate limits,
    - linearized obstacle half-spaces with slacks.
"""

import time

import casadi as ca
import numpy as np


class UnicycleMPC_ActiveSet_Tracking:
    """Tracking MPC for a unicycle solved by qpOASES.

    Problem formulation
        min_z 0.5 z^T H z + g^T z
        s.t.  lba <= A z <= uba
              lbx <= z <= ubx

    Decision variables
        z = [x_0, ..., x_N, u_0, ..., u_{N-1}, s]^T
        with the same state/control/slack partition as in the OSQP solver.

    Cost
        State tracking, control effort, terminal cost, and quadratic slack
        penalty. A small forward bias on v_k is kept to discourage stalling.

    Constraints
        Initial-state equality, affine linearized dynamics, input box bounds,
        input-rate limits, and linearized obstacle inequalities.
    """

    def __init__(
        self,
        dt=0.1,
        horizon=10,
        q_pos=10.0,
        q_theta=1.0,
        r=0.1,
        r_yaw=0.1,
        integration_method="euler",
        v_max=1.0,
        w_max=1.0,
        dv_max=0.2,
        dw_max=0.2,
        safe_distance=0.4,
        n_obs=10,
        slack_weight=300.0,
        max_active_obs=5,
        obs_horizon=10,
        activation_margin=4.0,
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

        self.dt = float(dt)
        self.N = int(horizon)
        self.nx = 3
        self.nu = 2
        self.integration_method = str(integration_method).lower()
        if self.integration_method not in {"euler", "rk4"}:
            raise ValueError("integration_method must be 'euler' or 'rk4'")

        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.dv_max = float(dv_max)
        self.dw_max = float(dw_max)
        self.safe_distance = float(safe_distance)
        self.n_obs = int(n_obs)
        self.slack_weight = float(slack_weight)
        self.max_active_obs = int(max_active_obs)
        self.obs_horizon = min(int(obs_horizon), self.N)
        self.activation_margin = float(activation_margin)
        self.progress_weight = float(progress_weight)
        self.ttc_threshold = float(ttc_threshold)
        self.tangent_bias_gain = float(tangent_bias_gain)
        self.obstacle_slowdown_margin = float(obstacle_slowdown_margin)
        self.min_speed_factor_near_obstacle = float(min_speed_factor_near_obstacle)
        self.dynamic_obstacle_speed_threshold = float(dynamic_obstacle_speed_threshold)
        self.big_bound = 1e8

        self.Q = np.diag([q_pos, q_pos, q_theta]).astype(float)
        self.R = np.diag([r, r_yaw]).astype(float)
        self.Q_term = 5.0 * self.Q

        # Decision-vector layout:
        #
        #   z =
        #   [ x_0, ..., x_N | u_0, ..., u_{N-1} | s ]^T
        #
        # This mirrors the OSQP controller so that differences in behavior are
        # mostly attributable to the numerical backend, not to a different
        # mathematical transcription.
        self.n_state_vars = self.nx * (self.N + 1)
        self.n_control_vars = self.nu * self.N
        self.n_slack = self.obs_horizon * self.max_active_obs
        self.n_vars = self.n_state_vars + self.n_control_vars + self.n_slack
        self.n_constraints = (
            self.nx
            + self.N * self.nx
            + self.N * self.nu
            + self.obs_horizon * self.max_active_obs
        )

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

    def _build_solver(self):
        """Build a fixed-size qpOASES QP template.

        qpOASES is used through CasADi's conic interface. As with OSQP, the
        solver is instantiated once from a sparsity template, while numeric QP
        coefficients are supplied later inside `solve()`.
        """
        qp = {
            "h": ca.Sparsity.dense(self.n_vars, self.n_vars),
            "a": ca.Sparsity.dense(self.n_constraints, self.n_vars),
        }
        opts = {
            "printLevel": "none",
            "print_time": False,
            "error_on_fail": False,
            "enableRegularisation": True,
        }
        self.solver = ca.conic("active_set_tracking", "qpoases", qp, opts)

    def _state_slice(self, k):
        """Indices of x_k inside the stacked decision vector z."""
        start = k * self.nx
        return slice(start, start + self.nx)

    def _control_slice(self, k):
        """Indices of u_k inside the stacked decision vector z."""
        start = self.u_start + k * self.nu
        return slice(start, start + self.nu)

    def _slack_index(self, k, j):
        """Index of slack s_(k,j) in the stacked decision vector z."""
        return self.slack_start + k * self.max_active_obs + j

    def _reference_inputs(self, ref_trajectory):
        """Infer nominal controls from the local state reference."""
        u_ref = np.zeros((self.N, self.nu))
        for k in range(self.N):
            dx = ref_trajectory[k + 1, 0] - ref_trajectory[k, 0]
            dy = ref_trajectory[k + 1, 1] - ref_trajectory[k, 1]
            dtheta = ref_trajectory[k + 1, 2] - ref_trajectory[k, 2]
            dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
            u_ref[k, 0] = np.clip(np.hypot(dx, dy) / self.dt, 0.0, self.v_max)
            u_ref[k, 1] = np.clip(dtheta / self.dt, -self.w_max, self.w_max)
        return u_ref

    def _sanitize_reference(self, state, ref_trajectory):
        """Clip the local reference to what the unicycle can actually do."""
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

    def _continuous_dynamics_expr(self, x, u):
        """Continuous-time unicycle dynamics."""
        return ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1],
        )

    def _discrete_step_expr(self, x, u):
        """Discrete-time step used both for prediction and linearization."""
        if self.integration_method == "rk4":
            k1 = self._continuous_dynamics_expr(x, u)
            k2 = self._continuous_dynamics_expr(x + 0.5 * self.dt * k1, u)
            k3 = self._continuous_dynamics_expr(x + 0.5 * self.dt * k2, u)
            k4 = self._continuous_dynamics_expr(x + self.dt * k3, u)
            return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x + self.dt * self._continuous_dynamics_expr(x, u)

    def _build_dynamics_functions(self):
        """Construct symbolic maps for f_d, A_k, and B_k."""
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        x_next = self._discrete_step_expr(x, u)
        A = ca.jacobian(x_next, x)
        B = ca.jacobian(x_next, u)
        self._step_fun = ca.Function("unicycle_active_step", [x, u], [x_next])
        self._A_fun = ca.Function("unicycle_active_A", [x, u], [A])
        self._B_fun = ca.Function("unicycle_active_B", [x, u], [B])

    def _linearized_dynamics(self, x_ref, u_ref):
        """Return the affine local model x_{k+1} ~= A_k x_k + B_k u_k + c_k."""
        x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
        u_ref = np.asarray(u_ref, dtype=float).reshape(-1)
        A = np.array(self._A_fun(x_ref, u_ref), dtype=float)
        B = np.array(self._B_fun(x_ref, u_ref), dtype=float)
        f_ref = np.array(self._step_fun(x_ref, u_ref), dtype=float).reshape(-1)
        c = f_ref - A @ x_ref - B @ u_ref
        return A, B, c

    def _select_obstacles(self, state, obstacles):
        """Keep only the closest obstacles used by the active-set QP."""
        if not obstacles:
            return []

        ranked = sorted(
            obstacles,
            key=lambda obs: np.linalg.norm(
                state[:2] - np.asarray(obs["position"], dtype=float)[:2]
            ),
        )
        return ranked[: min(self.max_active_obs, self.n_obs)]

    def _build_safe_guess(self, state, ref_trajectory):
        """Choose the trajectory around which obstacle constraints are linearized."""
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

    def _speed_limit_at_step(self, point_xy, obs_list, k):
        """Smoothly reduce forward speed near static obstacles.

        Instead of adding another constraint family, we tighten the upper box
        bound on the translational control v_k. This gives a simple
        obstacle-aware slowing behavior with almost no extra QP structure.
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

            obs_radius = float(obs.get("radius", 0.3))
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

    def _initial_guess(self, state, ref_trajectory):
        """Build the qpOASES hot start vector by shifting the previous solution."""
        if (
            self.warm_start_x is not None
            and self.warm_start_u is not None
            and self.warm_start_x.shape == (self.N + 1, self.nx)
            and self.warm_start_u.shape == (self.N, self.nu)
        ):
            x0 = np.zeros(self.n_vars)
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

        x0 = np.zeros(self.n_vars)
        x0[: self.n_state_vars] = ref_trajectory.flatten()
        for k in range(self.N):
            x0[self._control_slice(k)] = [min(0.5, self.v_max), 0.0]
        x0[self._state_slice(0)] = state
        return x0

    def _build_qp_matrices(self, state, ref_trajectory, obstacles, traj_guess):
        """Assemble H, g, A, lba, uba, lbx, ubx for the active-set QP.

        The meaning of the matrices is the same as in the OSQP formulation.
        We keep the formulation aligned on purpose so that differences in
        behavior mostly come from the numerical solver, not from using two
        fundamentally different MPC problems.
        """
        H = np.zeros((self.n_vars, self.n_vars))
        g = np.zeros(self.n_vars)
        A = np.zeros((self.n_constraints, self.n_vars))
        lba = -self.big_bound * np.ones(self.n_constraints)
        uba = self.big_bound * np.ones(self.n_constraints)
        lbx = -self.big_bound * np.ones(self.n_vars)
        ubx = self.big_bound * np.ones(self.n_vars)

        u_ref = self._reference_inputs(traj_guess)
        obs_list = self._select_obstacles(state, obstacles)

        # Stage cost:
        #   (x_k-r_k)^T Q (x_k-r_k) + u_k^T R u_k
        # plus a small forward reward on v_k.
        for k in range(self.N):
            x_slice = self._state_slice(k)
            u_slice = self._control_slice(k)
            H[x_slice, x_slice] += 2.0 * self.Q
            H[u_slice, u_slice] += 2.0 * self.R
            g[x_slice] += -2.0 * self.Q @ ref_trajectory[k]
            g[u_slice.start] += -self.progress_weight

        x_slice = self._state_slice(self.N)
        H[x_slice, x_slice] += 2.0 * self.Q_term
        g[x_slice] += -2.0 * self.Q_term @ ref_trajectory[self.N]

        # Soft obstacle handling:
        #   rho_s * ||s||^2 with s >= 0
        for k in range(self.n_slack):
            slack_idx = self.slack_start + k
            H[slack_idx, slack_idx] = 2.0 * self.slack_weight
            lbx[slack_idx] = 0.0

        # Box bounds on controls. The translational upper bound is optionally
        # reduced near static obstacles.
        for k in range(self.N):
            u_slice = self._control_slice(k)
            v_cap = self._speed_limit_at_step(traj_guess[k, :2], obs_list, k)
            lbx[u_slice] = [0.0, -self.w_max]
            ubx[u_slice] = [max(0.05, v_cap), self.w_max]

        row = 0

        # Input-rate limits:
        #   |u_k - u_{k-1}| <= [dv_max, dw_max]
        for k in range(self.N):
            u_slice = self._control_slice(k)

            if k == 0:
                A[row, u_slice.start] = 1.0
                lba[row] = self.prev_u_applied[0] - self.dv_max
                uba[row] = self.prev_u_applied[0] + self.dv_max
                row += 1

                A[row, u_slice.start + 1] = 1.0
                lba[row] = self.prev_u_applied[1] - self.dw_max
                uba[row] = self.prev_u_applied[1] + self.dw_max
                row += 1
            else:
                prev_u_slice = self._control_slice(k - 1)

                A[row, u_slice.start] = 1.0
                A[row, prev_u_slice.start] = -1.0
                lba[row] = -self.dv_max
                uba[row] = self.dv_max
                row += 1

                A[row, u_slice.start + 1] = 1.0
                A[row, prev_u_slice.start + 1] = -1.0
                lba[row] = -self.dw_max
                uba[row] = self.dw_max
                row += 1

        A[row : row + self.nx, self._state_slice(0)] = np.eye(self.nx)
        lba[row : row + self.nx] = state
        uba[row : row + self.nx] = state
        row += self.nx

        # Local affine dynamics:
        #   x_{k+1} - A_k x_k - B_k u_k = c_k
        for k in range(self.N):
            xk = self._state_slice(k)
            xkp1 = self._state_slice(k + 1)
            uk = self._control_slice(k)
            A_k, B_k, c_k = self._linearized_dynamics(traj_guess[k], u_ref[k])

            A[row : row + self.nx, xk] = -A_k
            A[row : row + self.nx, xkp1] = np.eye(self.nx)
            A[row : row + self.nx, uk] = -B_k
            lba[row : row + self.nx] = c_k
            uba[row : row + self.nx] = c_k
            row += self.nx

        # Linearized obstacle avoidance:
        #
        # The original nonconvex circle-separation condition is
        #
        #   ||x_k - o_k|| >= d_safe
        #
        # which is reverse-convex in x_k. Around the guessed point x_hat_k we
        # replace it with the supporting half-space
        #
        #   n_k^T x_k + s_k >= d_safe + n_k^T x_hat_k - ||x_hat_k - o_k||
        #
        # where
        #
        #   n_k = (x_hat_k - o_k) / ||x_hat_k - o_k||.
        #
        # This is the local convexification that makes the obstacle block
        # compatible with a real-time QP solver.
        for k in range(1, self.obs_horizon + 1):
            guess_xy = traj_guess[k, :2]
            xs = self._state_slice(k)

            if (k - 1) < len(u_ref):
                v_cmd = u_ref[k - 1, 0]
                theta_ref = traj_guess[k - 1, 2]
                v_rob = v_cmd * np.array([np.cos(theta_ref), np.sin(theta_ref)])
            else:
                v_rob = np.zeros(2)

            for j in range(self.max_active_obs):
                if j < len(obs_list):
                    obs = obs_list[j]
                    obs_pos = np.asarray(obs["position"], dtype=float)[:2]
                    obs_vel = np.asarray(obs.get("velocity", [0.0, 0.0]), dtype=float)[:2]
                    obs_radius = float(obs.get("radius", 0.3))
                    obs_xy = obs_pos + obs_vel * (k * self.dt)
                    safe = self.safe_distance + obs_radius

                    diff_guess = guess_xy - obs_xy
                    dist_guess = np.linalg.norm(diff_guess)

                    rel_pos = state[:2] - obs_xy
                    dist_cur = np.linalg.norm(rel_pos)
                    rel_vel = v_rob - obs_vel
                    closing_speed = -np.dot(rel_pos, rel_vel) / max(dist_cur, 1e-6)
                    ttc = dist_cur / max(closing_speed, 1e-6) if closing_speed > 0 else np.inf

                    if dist_guess <= safe + self.activation_margin or ttc <= self.ttc_threshold:
                        if dist_guess < 1e-6:
                            diff_cur = state[:2] - obs_xy
                            dcur = np.linalg.norm(diff_cur)
                            if dcur > 1e-6:
                                normal = diff_cur / dcur
                            else:
                                heading = state[2]
                                normal = np.array([-np.sin(heading), np.cos(heading)])
                        else:
                            normal = diff_guess / dist_guess
                            if closing_speed > 0.3:
                                tangent = np.array([-normal[1], normal[0]])
                                path_dir = ref_trajectory[min(k + 1, self.N), :2] - ref_trajectory[max(k - 1, 0), :2]
                                path_norm = np.linalg.norm(path_dir)
                                if path_norm < 1e-6:
                                    heading = ref_trajectory[max(k - 1, 0), 2]
                                    path_dir = np.array([np.cos(heading), np.sin(heading)])
                                else:
                                    path_dir = path_dir / path_norm

                                # Use the obstacle position relative to the local path direction
                                # instead of the robot pose; this avoids left/right sign flips when
                                # the robot oscillates near the obstacle boundary.
                                rel_obs = obs_xy - ref_trajectory[k, :2]
                                side = np.sign(path_dir[0] * rel_obs[1] - path_dir[1] * rel_obs[0])
                                side = 1.0 if abs(side) < 1e-6 else side
                                normal = normal + self.tangent_bias_gain * side * tangent
                                normal /= np.linalg.norm(normal)

                        # Linearization RHS evaluated at the guessed state
                        # x_hat_k = traj_guess[k, :2].
                        rhs = safe + np.dot(normal, guess_xy) - dist_guess

                        A[row, xs.start] = normal[0]
                        A[row, xs.start + 1] = normal[1]
                        A[row, self._slack_index(k - 1, j)] = 1.0
                        lba[row] = rhs

                row += 1

        return H, g, A, lba, uba, lbx, ubx

    def solve(self, state, ref_trajectory, obstacles=None):
        """Solve one QP and return the first control action.

        The fallback branch is intentionally kept here because active-set
        methods can become brittle when the working set changes abruptly.
        Returning a neutral command is better than propagating a partially
        valid solution.
        """
        state = np.asarray(state, dtype=float).flatten()
        ref_trajectory = self._sanitize_reference(state, ref_trajectory)

        start = time.time()
        try:
            traj_guess = self._build_safe_guess(state, ref_trajectory)
            x0 = self._initial_guess(state, ref_trajectory)
            H, g, A, lba, uba, lbx, ubx = self._build_qp_matrices(
                state, ref_trajectory, obstacles, traj_guess
            )

            sol = self.solver(
                h=ca.DM(H),
                g=ca.DM(g),
                a=ca.DM(A),
                lba=ca.DM(lba),
                uba=ca.DM(uba),
                lbx=ca.DM(lbx),
                ubx=ca.DM(ubx),
                x0=ca.DM(x0),
            )
            stats = self.solver.stats()
            if not stats.get("success", False):
                raise RuntimeError(stats.get("return_status", "unknown active-set failure"))

            sol_vec = np.array(sol["x"]).reshape(-1)
            traj_guess = sol_vec[: self.n_state_vars].reshape(self.N + 1, self.nx)

            self.warm_start_x = traj_guess
            self.warm_start_u = sol_vec[
                self.u_start : self.u_start + self.n_control_vars
            ].reshape(self.N, self.nu)
            u_opt = self.warm_start_u[0].copy()
            self.prev_u_applied = u_opt.copy()
            solve_time = time.time() - start
        except Exception as exc:
            print(f"[MPC][ActiveSet] solver failed: {exc}, using fallback")
            self.warm_start_x = None
            self.warm_start_u = None
            u_opt = np.array([0.0, 0.0])
            self.prev_u_applied = u_opt.copy()
            solve_time = 0.0

        return u_opt, solve_time
