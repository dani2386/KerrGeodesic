import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.ode_solver import ODESolver


class KerrSolver(ODESolver):
    def __init__(self, dir_name, a):
        self.a = a
        self.r_plus = 1 + np.sqrt(1 - self.a**2)
        self.E = 0
        self.J = 0

        super().__init__(dir_name, ('tau', 'phi', 'r', 'pr'), self._geodesic_eq)

    def _factors(self, r):
        delta = r**2 + self.a**2 - 2 * r
        Q = delta / (self.E * (r**2 + 2 * self.a**2 / r + self.a**2) - 2 * self.a * self.J / r)
        w = (self.J * Q + 2 * self.a / r) / (r**2 + 2 * self.a**2 / r + self.a**2)

        return delta, Q, w

    def _geodesic_eq(self, t, state):
        tau, phi, r, pr = state
        delta, Q, w = self._factors(r)

        dr_dt = delta * Q * pr / r**2
        dpr_dt = (w**2 * (r**3 - self.a**2 + 2 * self.a * w - 1)) / (Q * r**2) + Q * pr**2 * (self.a**2 - r) / r**3

        return np.array([Q, w, dr_dt, dpr_dt])

    def solve(self, run_id, depth, params, **kwargs):
        t_max, dt, r, self.E, self.J, r_max = params
        delta, Q, w = self._factors(r)

        stop_cond = lambda t, data: (data[2] <= self.r_plus * 1.01) or (data[2] >= r_max)

        pr = -(np.sqrt(np.abs(r**2 / (delta * Q**2) * (1 - 2 / r - Q**2 - w**2 * (r**2 + 2 * self.a**2 / r + self.a**2) + 4 * w * self.a / r))))

        return super().solve(run_id, depth, t_max, dt, np.array([0, 0, r, pr]), stop_cond, **kwargs)

    def plot_trajectory(self, run_id, depth, ax=None, **kwargs):
        if ax is None: fig, ax = plt.subplots()
        ax.grid(True)

        ax.add_patch(Circle((0, 0), 1 + np.sqrt(1 - self.a**2), color='black', linestyle='--', fill=False))

        data_path = f'{run_id}/data/v1'

        with self._file as file:
            t_max, dt = file.load_metadata(run_id, ('t_max', 'dt'))
            n_max = int(t_max / dt)
            line = None

            for n in range(0, n_max + 1, depth):
                buf_len = min(depth, n_max - n + 1)

                phi = file.load(data_path, (slice(n, n + buf_len), 1))
                r = file.load(data_path, (slice(n, n + buf_len), 2))

                x = r * np.cos(phi)
                y = r * np.sin(phi)

                if not line:
                    line, = ax.plot(x, y, **kwargs)
                else:
                    ax.plot(x, y, color=line.get_color(), linestyle=line.get_linestyle(), label=None)

        return ax
