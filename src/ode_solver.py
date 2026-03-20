import os
import numpy as np
import matplotlib.pyplot as plt
from src.hdf5 import HDF5File


class ODESolver:
    def __init__(self, dir_name, labels, f):
        self.labels = labels
        self.f = f

        os.makedirs(dir_name, exist_ok=True)
        self._file = HDF5File(os.path.join(dir_name, 'data.h5'))

    def _solve_rk4(self, dataset, t_max, dt, init, depth):
        n_max = int(t_max / dt)
        dim = len(init)

        with self._file as file:
            file.create_dataset(dataset, (n_max + 1, dim))
            file.save(dataset, 0, init)

            for n in range(0, n_max, depth):
                buf_len = min(depth, n_max - n)
                buffer = np.empty([buf_len + 1, dim])
                buffer[0] = file.load(dataset, n)

                for i in range(buf_len):
                    t = (n + i) * dt

                    k1 = self.f(t, buffer[i])
                    k2 = self.f(t + 0.5 * dt, buffer[i] + dt / 2 * k1)
                    k3 = self.f(t + 0.5 * dt, buffer[i] + dt / 2 * k2)
                    k4 = self.f(t + dt, buffer[i] + dt * k3)

                    buffer[i + 1] = buffer[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                file.save(dataset, slice(n + 1, n + buf_len + 1), buffer[1:])

    def solve(self, run_id, t_max, dt, init, depth, **kwargs):
        n_max = int(t_max / dt)
        dim = len(init)

        conv = kwargs.get('conv')
        conv_res = {'self': ('base', 'med', 'fine'), 'exact': ('base', 'fine'), None: ('base',)}

        with (self._file as file):
            file.create_group(run_id)
            file.create_group(f'{run_id}/states')
            file.save_metadata(run_id, t_max=t_max, dt=dt, init=init)

            for i, res in enumerate(conv_res[conv]):
                self._solve_rk4(f'{run_id}/states/{res}', t_max, dt / 2**i, init, depth)

            if not conv: return

            file.create_group(f'{run_id}/conv')
            file.create_dataset(f'{run_id}/conv/{conv}', (2, n_max + 1, dim))

            for n in range(0, n_max, depth):
                buf_len = min(depth, n_max - n)
                buffer = [file.load(f'{run_id}/states/{res}', slice(n * 2**i, (n + buf_len) * 2**i, 2**i))
                          for i, res in enumerate(conv_res[conv])]

                if conv == 'exact':
                    t = np.arange(n, n + buf_len) * dt
                    exact = kwargs.get('exact_f')(t)

                    err_buffer = np.stack([buffer[0] - exact, buffer[1] - exact])
                else:
                    err_buffer = np.stack([buffer[0] - buffer[1], buffer[1] - buffer[2]])

                file.save(f'{run_id}/conv/{conv}', (slice(None), slice(n, n + buf_len)), err_buffer)

    def plot(self, run_id, label, depth, ax=None, **kwargs):
        if ax is None: fig, ax = plt.subplots()
        line = None

        conv = kwargs.pop('conv', None)
        if conv: ax.set_yscale('log')

        with self._file as file:
            t_max, dt = file.load_metadata(run_id, ('t_max', 'dt'))
            n_max = int(t_max / dt)
            idx = self.labels.index(label)

            for n in range(0, n_max, depth):
                buf_len = min(depth, n_max - n)

                x = np.arange(n, n + buf_len + 1) * dt
                y = [file.load(f'{run_id}/states/base', (slice(n, n + buf_len + 1), idx))] if not conv \
                    else file.load(f'{run_id}/conv/{conv}', (slice(None), slice(n, n + buf_len + 1), idx))

                if line is None:
                    line = ax.plot(x, np.transpose(y), **kwargs)
                else:
                    for ln, y_sub in zip(line, y):
                        ax.plot(x, y_sub, color=ln.get_color(), linestyle=ln.get_linestyle(), label=None)

        return ax
