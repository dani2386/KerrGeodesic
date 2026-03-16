import numpy as np
import matplotlib.pyplot as plt
from src.ode_solver import ODESolver


class KerrSolver(ODESolver):
    def __init__(self, dir_name, a):
        self.a = a
        self.E = 0
        self.J = 0

        super().__init__(dir_name, f=None)

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

    def solve(self, run_id, params, depth):
        t, dt, r, self.E, self.J = params
        delta, Q, w = self._factors(r)

        pr = -(np.sqrt(np.abs(r**2 / (delta * Q**2) * (1 - 2 / r - Q**2 - w**2 * (r**2 + 2 * self.a**2 / r + self.a**2) + 4 * w * self.a / r))))

        self.f = self._geodesic_eq

        return super().solve(run_id, (t, dt, np.array([0, 0, r, pr])), depth)

    def plot(self, run_id, x_axis, y_axis, depth, ax=None, **kwargs):
        opts = {'t': None, 'tau': 0, 'phi': 1, 'r': 2, 'pr': 3}

        super().plot(run_id, opts[x_axis], opts[y_axis], depth, ax, **kwargs)

    def plot_trajectory(self, run_id, depth, ax=None, **kwargs):
        states  = f'{run_id}/states'

        if ax is None: fig, ax = plt.subplots()

        with self._file as file:
            n_max = file.load_metadata(run_id, 'n_max')

            for n in range(0, n_max, depth):
                phi = file.load(states, slice(n, min(n + depth, n_max + 1)))[:, 1]
                r = file.load(states, slice(n, min(n + depth, n_max + 1)))[:, 2]

                x = r * np.cos(phi)
                y = r * np.sin(phi)

                ax.plot(x, y, **kwargs)

        return ax
