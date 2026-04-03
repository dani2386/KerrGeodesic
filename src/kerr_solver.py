import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.ode_solver import ODESolver


class KerrSolver(ODESolver):
    def __init__(self, dir_name, a):
        self.a = a
        self.r_plus = 1 + np.sqrt(1 - self.a**2)
        self.m = 0

        super().__init__(dir_name, ('tau', 'phi', 'r', 'pr', 'E', 'J'), self._geodesic_eq)

    def _factors(self, r, E, J):
        delta = r**2 + self.a**2 - 2 * r
        A = r**2 + 2 * self.a**2 / r + self.a**2
        B = r**2 + 2 * self.a / r + self.a**2
        Q = delta / (E * A - 2 * self.a * J / r)
        w = (J * Q + 2 * self.a / r) / A

        return delta, A, B, Q, w

    def _geodesic_eq(self, t, state):
        tau, phi, r, pr, E, J = state
        delta, A, B, Q, w = self._factors(r, E, J)

        dr_dt = delta * Q * pr / r ** 2
        dpr_dt = (w**2 * (r**3 - self.a**2) + 2 * self.a * w - 1) / (Q * r**2) + Q * pr**2 * (self.a**2 - r) / r**3
        dE_dt = -32 / 5 * self.m**2 * (1 + self.m) / r**5
        dJ_dt = dE_dt / w

        return np.array([Q, w, dr_dt, dpr_dt, dE_dt, dJ_dt])

    def circ_params(self, tol=1e-10, max_iter=100, **kwargs):
        r, E, J = kwargs.get('r', 10), kwargs.get('E', 1), 3.5
        x = (E, J) if kwargs.get('r') else (r, J)

        for i in range(max_iter):
            if kwargs.get('r'):
                E, J = x
            else:
                r, J = x

            delta, A, B, Q, w = self._factors(r, E, J)

            f = np.array([1 - 2 / r - Q**2 - w**2 * B + 4 * w * self.a / r, w**2 * (r**3 - self.a**2) + 2 * self.a * w - 1])

            if np.linalg.norm(f) < tol: return r, E, J

            df_dJ = np.array([
                -4 * self.a * Q**3 / (r * delta) + (4 * self.a / r - 2 * w * B) * (Q / A + 2 * self.a * J * Q**2 / (r * delta)),
                (2 * w * (r**3 - self.a**2) + 2 * self.a) * (Q / A + 2 * self.a * J * Q**2 / (r * delta))
            ])

            if kwargs.get('r'):
                df_dE = np.array([
                    2 * Q**3 * A / delta + (2 * w * B - 4 * self.a / r) * J * Q**2 / delta,
                    (2 * w * (r**3 - self.a**2) + 2 * self.a) * -J * Q**2 / delta
                ])
                df = np.column_stack((df_dE, df_dJ))
            else:
                dQ_dr = (2 * r - 2 - Q * (E * (2 * r - 2 * self.a**2 / r**2) + 2 * self.a * J / r**2)) / (E * A - 2 * self.a * J / r)
                dw_dr = (J * dQ_dr - 2 * self.a / r**2 - w * (2 * r - 2 * self.a**2 / r**2)) / A
                df_dr = np.array([
                    2 / r**2 - 2 * Q * dQ_dr - w**2 * (2 * r - 2 * self.a / r**2) + dw_dr * (4 * self.a / r - 2 * w * B) - 4 * w * self.a / r**2,
                    dw_dr * (2 * w * (r**3 - self.a**2) + 2 * self.a) + 3 * r**2 * w**2
                ])
                df = np.column_stack((df_dr, df_dJ))

            x = x - np.linalg.solve(df, f)

        return r, E, J

    def solve(self, run_id, depth, params, **kwargs):
        t_max, dt, r, E, J, self.m, r_max = params
        delta, A, B, Q, w = self._factors(r, E, J)
        pr = -(np.sqrt(np.abs(r**2 / (delta * Q**2) * (1 - 2 / r - Q**2 - w**2 * B + 4 * w * self.a / r))))

        stop_cond = lambda t, data: (data[2] <= self.r_plus * 1.05) or (data[2] >= r_max)

        return super().solve(run_id, depth, t_max, dt, np.array([0, 0, r, pr, E, J]), stop_cond, **kwargs)

    def solve_gw(self, run_id, depth, params):
        d, l = params

        data_path = f'{run_id}/data/v1'
        gw_path = f'{run_id}/gw'

        with self._file as file:
            t_max, dt = file.load_metadata(run_id, ('t_max', 'dt'))
            n_max = int(t_max / dt)

            file.create_dataset(gw_path, (2, n_max + 1))

            for n in range(0, n_max + 1, depth):
                buf_len = min(depth, n_max - n + 1)

                phi = file.load(data_path, (slice(n, n + buf_len), 1))
                r = file.load(data_path, (slice(n, n + buf_len), 2))

                h_plus = -2 * self.m / (r * d) * (1 + np.cos(l)**2) * np.cos(2 * phi)
                h_cross = -4 * self.m / (r * d) * np.cos(l) * np.sin(2 * phi)

                file.save(gw_path, (slice(None), slice(n, n + buf_len)), np.stack([h_plus, h_cross]))

    def get_gw(self, run_id, depth):
        gw_path = f'{run_id}/gw'

        with self._file as file:
            t_max, dt = file.load_metadata(run_id, ('t_max', 'dt'))
            n_max = int(t_max / dt)

            for n in range(0, n_max + 1, depth):
                buf_len = min(depth, n_max - n + 1)

                t = np.arange(n, n + buf_len) * dt
                h = file.load(gw_path, (slice(None), slice(n, n + buf_len)))

                yield t, h

    def plot_trajectory(self, run_id, depth, **kwargs):
        fig, ax = plt.subplots()
        ax.grid(True)
        line = None

        ax.add_patch(Circle((0, 0), self.r_plus, color='black', linestyle='--', fill=False))

        for (_, phi), (_, r) in zip(self.get_data(run_id, depth, 'phi'), self.get_data(run_id, depth, 'r')):
            x = np.transpose(r) * np.cos(np.transpose(phi))
            y = np.transpose(r) * np.sin(np.transpose(phi))

            if not line:
                line, = ax.plot(x, y, **kwargs)
            else:
                ax.plot(x, y, color=line.get_color(), linestyle=line.get_linestyle(), label=None)

        return ax

    def plot_gw(self, run_id, depth, **kwargs):
        fig, ax = plt.subplots()
        ax.grid(True)
        line = None

        for t, h in self.get_gw(run_id, depth):
            if line is None:
                line = ax.plot(t, np.transpose(h), **kwargs)
            else:
                for ln, h_p in zip(line, h):
                    ax.plot(t, h_p, color=ln.get_color(), linestyle=ln.get_linestyle(), label=None)

        return ax
