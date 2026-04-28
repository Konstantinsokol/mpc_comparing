"""IPOPT-based nonlinear multiple-shooting MPC for a unicycle.

Problem formulation
    At every control step we solve the nonlinear finite-horizon optimal
    control problem directly, without convexifying obstacle avoidance.

Decision variables
    z = [X_0, ..., X_N, U_0, ..., U_{N-1}]^T
    where X_k are explicit state variables and U_k are control variables.

Cost
    Nonlinear tracking cost with wrapped heading error, control regularization,
    and a terminal penalty.

Constraints
    - initial-state equality,
    - exact multiple-shooting dynamics X_{k+1} = f_d(X_k, U_k),
    - input bounds,
    - input-rate bounds,
    - exact nonlinear circle-distance obstacle constraints.
"""

import numpy as np
import casadi as ca
import time


class UnicycleMPC_Tracking:
    """Nonlinear multiple-shooting MPC for a unicycle solved by IPOPT.

    In contrast to the OSQP / qpOASES controllers, this class does not
    approximate obstacle avoidance with local half-spaces. Instead, it keeps
    the nonlinear distance constraints directly in the optimization problem.

    Decision variables:
        X = [x_0, ..., x_N]    predicted states
        U = [u_0, ..., u_{N-1}] predicted controls

    The resulting NLP is solved with IPOPT using a primal-dual interior-point
    method with filter line search.
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
        dv_max=0.25,
        dw_max=0.5,
        safe_distance=0.4,
        n_obs=10,
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

        self.Q = np.diag([q_pos, q_pos, q_theta])
        self.R = np.diag([r, r_yaw])
        self.Q_term = 5.0 * self.Q

        # NLP decision-vector dimensions:
        #
        #   z = [vec(X), vec(U)]^T
        #
        # with vec(X) containing all horizon states and vec(U) all horizon
        # controls. Unlike the QP solvers there are no slack variables here,
        # because obstacle constraints are kept in exact nonlinear form.
        self.n_state_vars = self.nx * (self.N + 1)
        self.n_control_vars = self.nu * self.N
        self.n_vars = self.n_state_vars + self.n_control_vars
        self.x_start = 0
        self.x_end = self.n_state_vars
        self.u_start = self.n_state_vars
        self.u_end = self.n_vars

        self.warm_start_x = None
        self.warm_start_u = None
        self.prev_u_applied = np.zeros(self.nu)

        # The symbolic nonlinear program is assembled once and then reused with
        # updated parameters at every MPC step.
        self._build_solver()

    def _continuous_dynamics_expr(self, x, u):
        """Continuous-time unicycle model f(x, u).

        State:
            x = [p_x, p_y, theta]^T
        Control:
            u = [v, omega]^T
        """
        return ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1],
        )

    def _discrete_step_expr(self, x, u):
        """Discrete transition map x_{k+1} = f_d(x_k, u_k).

        This is the step model used inside the multiple-shooting continuity
        constraints. The same symbolic map is reused at every stage k.
        """
        if self.integration_method == "rk4":
            k1 = self._continuous_dynamics_expr(x, u)
            k2 = self._continuous_dynamics_expr(x + 0.5 * self.dt * k1, u)
            k3 = self._continuous_dynamics_expr(x + 0.5 * self.dt * k2, u)
            k4 = self._continuous_dynamics_expr(x + self.dt * k3, u)
            return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x + self.dt * self._continuous_dynamics_expr(x, u)

    def _sanitize_reference(self, state, ref_trajectory):
        """Clip the reference so it respects the robot's motion limits.

        Even though IPOPT solves the nonlinear problem directly, feeding it an
        obviously infeasible reference still hurts convergence. This helper
        therefore makes the local horizon reference kinematically plausible.
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

    def _build_solver(self):
        """Build the full nonlinear program.

        State matrix:
            X in R^(nx x (N+1))
        Control matrix:
            U in R^(nu x N)

        The NLP contains:
        - nonlinear multiple-shooting dynamic equalities,
        - box bounds on controls,
        - control rate constraints,
        - exact nonlinear circle-distance obstacle constraints.
        """
        N, nx, nu, n_obs = self.N, self.nx, self.nu, self.n_obs

        X = ca.SX.sym('X', nx, N + 1)
        U = ca.SX.sym('U', nu, N)

        # ---------- PARAMETERS ----------
        # x0_param      : current measured state
        # u_prev_param  : previously applied control, used for rate limits
        # ref_traj      : local reference over the whole horizon
        # obstacle data : positions, velocities, radii, and activity masks
        x0_param = ca.SX.sym('x0', nx)
        u_prev_param = ca.SX.sym('u_prev', nu)
        ref_traj = ca.SX.sym('ref_traj', nx, N + 1)

        obs_pos_x = ca.SX.sym('obs_x', n_obs)
        obs_pos_y = ca.SX.sym('obs_y', n_obs)
        obs_vx = ca.SX.sym('obs_vx', n_obs)
        obs_vy = ca.SX.sym('obs_vy', n_obs)
        obs_radius = ca.SX.sym('obs_r', n_obs)
        obs_active = ca.SX.sym('obs_active', n_obs)
        
        # ---------- OBJECTIVE ----------
        # We penalize state tracking and control effort:
        #
        #   sum_k (x_k-r_k)^T Q (x_k-r_k) + u_k^T R u_k
        #   + (x_N-r_N)^T Q_N (x_N-r_N)
        #
        # The heading error is wrapped with atan2(sin, cos) so that angles near
        # +/- pi do not create an artificial large error.
        cost = 0
        for k in range(N):
            theta_err = ca.atan2(
                ca.sin(X[2, k] - ref_traj[2, k]),
                ca.cos(X[2, k] - ref_traj[2, k]),
            )
            state_err = ca.vertcat(
                X[0, k] - ref_traj[0, k],
                X[1, k] - ref_traj[1, k],
                theta_err,
            )
            ctrl = U[:, k]
            cost += ca.mtimes([state_err.T, self.Q, state_err])
            cost += ca.mtimes([ctrl.T, self.R, ctrl])

        term_theta_err = ca.atan2(
            ca.sin(X[2, N] - ref_traj[2, N]),
            ca.cos(X[2, N] - ref_traj[2, N]),
        )
        term_err = ca.vertcat(
            X[0, N] - ref_traj[0, N],
            X[1, N] - ref_traj[1, N],
            term_theta_err,
        )
        cost += ca.mtimes([term_err.T, self.Q_term, term_err])

        # ---------- CONSTRAINTS ----------
        g = []

        # Multiple-shooting dynamics:
        #   X_{k+1} - f_d(X_k, U_k) = 0
        #
        # This is the key transcription idea: every state X_k is an explicit
        # decision variable, and continuity is enforced through equality
        # constraints instead of one long forward rollout.
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            x_next = self._discrete_step_expr(xk, uk)
            g.append(X[:, k+1] - x_next)

        # Initial state consistency:
        #   X_0 = x_measured
        g.append(X[:, 0] - x0_param)

        # Control-rate limits:
        #   U_0 - u_prev
        #   U_k - U_{k-1}, k >= 1
        #
        # Later we impose componentwise bounds on these rows:
        #   -dv_max <= Delta v_k <= dv_max
        #   -dw_max <= Delta w_k <= dw_max
        g.append(U[:, 0] - u_prev_param)
        for k in range(1, N):
            g.append(U[:, k] - U[:, k - 1])

        # Exact nonlinear obstacle avoidance:
        #
        #   (x_k - o_{x,k})^2 + (y_k - o_{y,k})^2 >= (d_safe + r_obs)^2
        #
        # Unlike the QP controllers, there is no local half-space
        # approximation here. IPOPT sees the true circle-separation geometry
        # directly, which is mathematically cleaner but also more exposed to
        # local minima in symmetric scenes such as head-on encounters.
        #
        # The activity coefficient obs_active[i] keeps the NLP size fixed while
        # allowing unused obstacle slots to be deactivated.
        for k in range(1, N + 1):
            robot_x = X[0, k]
            robot_y = X[1, k]
            for i in range(n_obs):
                obs_x_k = obs_pos_x[i] + obs_vx[i] * (k * self.dt)
                obs_y_k = obs_pos_y[i] + obs_vy[i] * (k * self.dt)
                dist_sq = (robot_x - obs_x_k)**2 + (robot_y - obs_y_k)**2
                min_dist = self.safe_distance + obs_radius[i]
                constraint = dist_sq - min_dist**2
                g.append(obs_active[i] * constraint)

        # ---------- NLP ASSEMBLY ----------
        X_vec = ca.reshape(X, -1, 1)
        U_vec = ca.reshape(U, -1, 1)
        vars_sym = ca.vertcat(X_vec, U_vec)

        lbx = -ca.inf * ca.DM.ones(vars_sym.shape)
        ubx =  ca.inf * ca.DM.ones(vars_sym.shape)
        u_start = nx * (N + 1)
        # Only controls receive explicit box constraints. States remain free
        # and are shaped indirectly by dynamics, tracking cost, and obstacle
        # constraints.
        for k in range(N):
            lbx[u_start + k*nu] = 0
            ubx[u_start + k*nu] = self.v_max
            lbx[u_start + k*nu + 1] = -self.w_max
            ubx[u_start + k*nu + 1] = self.w_max

        g_concat = ca.vertcat(*g)
        n_dyn = N * nx + nx
        n_rate = N * nu
        lbg = ca.DM.zeros(g_concat.shape)
        ubg = ca.inf * ca.DM.ones(g_concat.shape)
        # Row blocks inside g:
        # 1. dynamics + initial condition equalities -> fixed to zero,
        # 2. rate constraints                        -> lower/upper bounded,
        # 3. obstacle constraints                    -> must stay nonnegative.
        lbg[:n_dyn] = 0
        ubg[:n_dyn] = 0
        lbg[n_dyn:n_dyn + n_rate:2] = -self.dv_max
        ubg[n_dyn:n_dyn + n_rate:2] = self.dv_max
        lbg[n_dyn + 1:n_dyn + n_rate:2] = -self.dw_max
        ubg[n_dyn + 1:n_dyn + n_rate:2] = self.dw_max
        lbg[n_dyn + n_rate:] = 0
        ubg[n_dyn + n_rate:] = ca.inf

        # Parameter vector passed to IPOPT at every control step:
        #
        #   p = [x0, u_prev, vec(ref_traj), obs_x, obs_y, obs_vx, obs_vy,
        #        obs_radius, obs_active]^T
        p = ca.vertcat(
            x0_param,
            u_prev_param,
            ca.reshape(ref_traj, -1, 1),
            obs_pos_x, obs_pos_y, obs_vx, obs_vy, obs_radius, obs_active
        )

        nlp = {'x': vars_sym, 'f': cost, 'g': g_concat, 'p': p}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0,
            'ipopt.max_iter': 1000,
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
    
    def solve(self, state, ref_trajectory, obstacles=None):
        """Solve one nonlinear MPC problem and return the first control.

        Workflow:
        1. sanitize the local reference,
        2. pack the current obstacle snapshot into a fixed-size parameter vector,
        3. build a warm start by shifting the previous optimal trajectory,
        4. solve the NLP,
        5. keep the first control and store the full solution for the next step.
        """
        N, nx, nu, n_obs = self.N, self.nx, self.nu, self.n_obs

        state = np.asarray(state, dtype=float).flatten()
        ref_trajectory = self._sanitize_reference(state, ref_trajectory)

        # Build a fixed-size obstacle parameter block. Only the first n_obs
        # slots are filled; the rest remain inactive through obs_active = 0.
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

        # Parameter vector passed into the symbolic NLP.
        p = np.concatenate([
            state,
            self.prev_u_applied,
            ref_trajectory.flatten(),
            obs_x, obs_y, obs_vx, obs_vy, obs_r, obs_active
        ])

        # ---------- WARM START ----------
        # Shift the previous optimal state/control sequence by one step. This is
        # the nonlinear analogue of the warm-start strategy used in the QP
        # controllers.
        if (self.warm_start_x is not None and self.warm_start_u is not None and
            self.warm_start_x.shape == (N+1, nx) and self.warm_start_u.shape == (N, nu)):
            x0 = np.zeros(self.n_vars)
            # Shift state predictions:
            #   x_1,...,x_N -> new x_0,...,x_{N-1}
            for k in range(N):
                x0[k*nx:(k+1)*nx] = self.warm_start_x[k+1, :]
            # Extrapolate the terminal state using the last predicted control.
            last_x, last_y, last_theta = self.warm_start_x[-1]
            last_v, last_w = self.warm_start_u[-1]
            x0[N*nx]   = last_x + last_v * np.cos(last_theta) * self.dt
            x0[N*nx+1] = last_y + last_v * np.sin(last_theta) * self.dt
            x0[N*nx+2] = last_theta + last_w * self.dt
            # Shift controls:
            #   u_1,...,u_{N-1} -> new u_0,...,u_{N-2}
            for k in range(N-1):
                x0[self.u_start + k*nu:self.u_start + (k+1)*nu] = self.warm_start_u[k+1, :]
            x0[self.u_start + (N-1)*nu:self.u_start + N*nu] = [0.0, 0.0]
        else:
            x0 = np.zeros(self.n_vars)
            # First-iteration guess: follow the reference states and start from
            # a mild forward command with zero turn rate.
            x0[:nx*(N+1)] = ref_trajectory.flatten()
            for k in range(N):
                x0[self.u_start + k*nu]   = min(0.5, self.v_max)
                x0[self.u_start + k*nu+1] = 0.0

        # ---------- SOLVE ----------
        start = time.time()
        try:
            sol = self.solver(
                x0=x0, p=p,
                lbx=self.lbx, ubx=self.ubx,
                lbg=self.lbg, ubg=self.ubg
            )
            stats = self.solver.stats()
            if not stats.get("success", False):
                raise RuntimeError(stats.get("return_status", "unknown IPOPT failure"))

            # Store the full predicted trajectory; the next MPC step will shift
            # it and reuse it as IPOPT's initial point.
            self.warm_start_x = sol['x'][:nx*(N+1)].full().reshape(N+1, nx)
            self.warm_start_u = sol['x'][self.u_start:].full().reshape(N, nu)
            u_opt = self.warm_start_u[0, :]
            self.prev_u_applied = np.asarray(u_opt).copy()
            solve_time = time.time() - start
        except Exception as e:
            # Failed NLP solves should not leak stale warm-start data into the
            # next iteration, so we clear the stored trajectory explicitly.
            print(f"[MPC] IPOPT failed: {e}, using fallback")
            self.warm_start_x = None
            self.warm_start_u = None
            u_opt = np.array([0.0, 0.0])
            self.prev_u_applied = u_opt.copy()
            solve_time = 0.0

        return u_opt, solve_time
