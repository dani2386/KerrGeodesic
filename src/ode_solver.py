import os
import numpy as np
import matplotlib.pyplot as plt
from src.hdf5 import HDF5File


class ODESolver:
    def __init__(self, dir_name, f):
        self.f = f

        os.makedirs(dir_name, exist_ok=True)
        self._file = HDF5File(os.path.join(dir_name, 'data.h5'))

    def solve(self, run_id, params, depth):
        ts = f'{run_id}/ts'
        states  = f'{run_id}/states'

        t_max, dt, init = params
        n_max = int(t_max / dt)

        with self._file as file:
            file.create_group(run_id)
            file.save_metadata(run_id, t_max=t_max, dt=dt, n_max=n_max, init=init)

            file.create_dataset(ts, (n_max + 1,))
            file.create_dataset(states, (n_max + 1, len(init)))

            file.save(ts, 0, 0)
            file.save(states, 0, init)

            for n in range(0, n_max, depth):
                buffer = np.empty([min(depth, n_max - n) + 1, len(init)])
                buffer[0] = file.load(states, n)

                for i in range(min(depth, n_max - n)):
                    k1 = self.f((n + i) * dt, buffer[i])
                    k2 = self.f((n + i + 0.5) * dt, buffer[i] + dt / 2 * k1)
                    k3 = self.f((n + i + 0.5) * dt, buffer[i] + dt / 2 * k2)
                    k4 = self.f((n + i + 1) * dt, buffer[i] + dt * k3)

                    buffer[i + 1] = buffer[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                file.save(ts, slice(n + 1, min(n + depth, n_max) + 1), np.arange(n + 1, min(n + depth, n_max) + 1) * dt)
                file.save(states, slice(n + 1, min(n + depth, n_max) + 1), buffer[1:])

    def plot(self, run_id, idx, idy, depth, ax=None, **kwargs):
        ts = f'{run_id}/ts'
        states  = f'{run_id}/states'

        if ax is None: fig, ax = plt.subplots()

        with self._file as file:
            n_max = file.load_metadata(run_id, 'n_max')

            for n in range(0, n_max, depth):
                x = file.load(ts, slice(n, min(n + depth, n_max + 1))) if idx is None else \
                    file.load(states, slice(n, min(n + depth, n_max + 1)))[:, idx]
                y = file.load(ts, slice(n, min(n + depth, n_max + 1))) if idy is None else \
                    file.load(states, slice(n, min(n + depth, n_max + 1)))[:, idy]

                ax.plot(x, y, **kwargs)

        return ax
