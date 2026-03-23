import numpy as np
import matplotlib.pyplot as plt
from src.ode_solver import ODESolver


class KerrSolver(ODESolver):
    def __init__(self, dir_name, a):
        self.a = a
    #    self.E = 0
    #    self.J = 0
        self.m = 1
        self.M = 1

        super().__init__(dir_name, f=None)

    def _factors(self, r,E,J):
        delta = r**2 + self.a**2 - 2 * r
        Q = delta / (E * (r**2 + 2 * self.a**2 / r + self.a**2) - 2 * self.a * J / r)
        w = (J * Q + 2 * self.a / r) / (r**2 + 2 * self.a**2 / r + self.a**2)

        return delta, Q, w

    def _geodesic_eq(self, t, state):
        tau, phi, r, pr, E ,J= state
        delta, Q, w = self._factors(r,E,J)

        dr_dt = delta * Q * pr / r**2
        dpr_dt = (w**2 * (r**3 - self.a**2 + 2 * self.a * w - 1)) / (Q * r**2) + Q * pr**2 * (self.a**2 - r) / r**3
        dE_dt = -(self.m **2)*(32/5)*(w**6)*r**4
        dJ_dt = dE_dt / w
        return np.array([Q, w, dr_dt, dpr_dt, dE_dt, dJ_dt])

    def solve(self, run_id, params, depth):
        t, dt, r, E, J = params
        delta, Q, w = self._factors(r, E, J)

        pr = -(np.sqrt(np.abs(r**2 / (delta * Q**2) * (1 - 2 / r - Q**2 - w**2 * (r**2 + 2 * self.a**2 / r + self.a**2) + 4 * w * self.a / r))))

        self.f = self._geodesic_eq

        return super().solve(run_id, (t, dt, np.array([0, 0, r, pr, E, J])), depth)

    def plot(self, run_id, x_axis, y_axis, depth, ax=None, **kwargs):
        opts = {'t': None, 'tau': 0, 'phi': 1, 'r': 2, 'pr': 3, 'E': 4, 'J': 5}

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

                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                ax.plot(x, y, **kwargs)

        # --- Event horizon ---
        r_plus = self.M + np.sqrt(self.M**2 - self.a**2)

        theta = np.linspace(0, 2*np.pi, 500)
        x_h = r_plus * np.cos(theta)
        y_h = r_plus * np.sin(theta)

        ax.plot(x_h, y_h, 'k--', label='Event Horizon')
        ax.fill(x_h, y_h, color='black', alpha=0.3)


        return ax
