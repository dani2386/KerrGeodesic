import os
import numpy as np
import matplotlib.pyplot as plt
from src.hdf5 import HDF5File


class ODESolver:
    """
    Generic Runge-Kutta 4th Order (RK4) solver with HDF5 data management

    Supports:
        - Memory-efficient buffering
        - Self-convergence and exact-convergence
        - Automatic plotting of results and errors
    """
    def __init__(self, dir_name, labels, f):
        """
        Initialize the solver

        Args:
            dir_name: Directory to store the HDF5 file
            labels: Tuple of names for the state variables
            f: The derivative function dy/dt = f(t, y)
        """
        self.labels = labels
        self.f = f

        os.makedirs(dir_name, exist_ok=True)
        self._file = HDF5File(os.path.join(dir_name, 'data.h5'))

    def _solve_rk4(self, path, depth, t_max, dt, init, stop_cond):
        """Internal RK4 implementation with HDF5 buffering"""
        n_max = int(t_max / dt)
        stop = False

        with self._file as file:
            file.create_dataset(path, (n_max + 1, len(init)))
            file.save(path, 0, init)

            for n in range(0, n_max, depth):
                buf_len = min(depth, n_max - n)
                buffer = np.empty((buf_len + 1, len(init)))
                buffer[0] = init if n == 0 else prev

                for i in range(buf_len):
                    # Standard RK4 algorithm
                    t = (n + i) * dt

                    k1 = self.f(t, buffer[i])
                    k2 = self.f(t + dt / 2, buffer[i] + dt / 2 * k1)
                    k3 = self.f(t + dt / 2, buffer[i] + dt / 2 * k2)
                    k4 = self.f(t + dt, buffer[i] + dt * k3)

                    buffer[i + 1] = buffer[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                    # Stop condition check
                    if stop_cond and stop_cond(t + dt, buffer[i + 1]):
                        file.save(path, slice(n + 1, n + i + 2), buffer[1:i + 2])
                        file.resize(path, (n + i + 2, len(init)))

                        stop = True
                        break

                if stop: break

                file.save(path, slice(n + 1, n + buf_len + 1), buffer[1:])
                prev = buffer[-1]

            file.save_metadata(path, t_max=t + dt, dt=dt, init=init)

    def solve(self, run_id, depth, t_max, dt, init, stop_cond=None, **kwargs):
        """
        Manages the solving process, with multiple passes for convergence

        Args:
            run_id: Unique string for this simulation run
            depth: Number of steps to compute before writing to disk
            t_max: The maximum simulation time
            dt: The time step for the main simulation
            init: Initial state vector
            stop_cond: A function (t, state) -> bool that triggers early stopping
            **kwargs: Convergence type: 'self' or 'exact'
        """
        conv = kwargs.get('conv')
        levels = range(3 if conv == 'self' else (2 if conv == 'exact' else 1))

        data_path = [f'{run_id}/data/v{2**i}' for i in levels]
        conv_path = f'{run_id}/conv/{conv}'

        with self._file as file:
            file.create_group(run_id)
            file.create_group(f'{run_id}/data')

            # Solves at dt, dt/2, and dt/4 if convergence is requested (self)
            for i in levels: self._solve_rk4(data_path[i], depth, t_max, dt / 2 ** i, init, stop_cond)

            # Sync t_max across all resolution runs
            t_max = min(file.load_metadata(data_path[i], 't_max') for i in levels) // dt * dt
            n_max = int(t_max / dt)

            file.save_metadata(run_id, t_max=t_max, dt=dt, init=init)

            if not conv: return

            # Calculate errors for convergence
            file.create_group(f'{run_id}/conv')
            file.create_dataset(conv_path, (2, n_max + 1, len(init)))

            for n in range(0, n_max + 1, depth):
                buf_len = min(depth, n_max - n + 1)
                buffer = [file.load(data_path[i], slice(n * 2**i, (n + buf_len) * 2**i, 2**i)) for i in levels]

                if conv == 'exact':
                    t = np.arange(n, n + buf_len) * dt
                    exact = kwargs.get('exact_f')(t)

                    err = np.stack([buffer[0] - exact, buffer[1] - exact])
                else:
                    err = np.stack([buffer[0] - buffer[1], buffer[1] - buffer[2]])

                file.save(conv_path, (slice(None), slice(n, n + buf_len)), np.abs(err))

    def get_data(self, run_id, depth, label, conv=None):
        """Generator to yield data buffers from HDF5"""
        data_path = f'{run_id}/data/v1'
        conv_path = f'{run_id}/conv/{conv}'

        with self._file as file:
            t_max, dt = file.load_metadata(run_id, ('t_max', 'dt'))
            n_max = int(t_max / dt)
            idx = self.labels.index(label)

            for n in range(0, n_max + 1, depth):
                buf_len = min(depth, n_max - n + 1)

                t = np.arange(n, n + buf_len) * dt
                # If plotting convergence, scale by a factor of 16
                data = [file.load(data_path, (slice(n, n + buf_len), idx))] if not conv else \
                        file.load(conv_path, (slice(None), slice(n, n + buf_len), idx)) * [[1], [16]]

                yield t, data

    def plot(self, run_id, depth, label, ax=None, **kwargs):
        """Plots the variable indicated by 'label' over time or its convergence errors"""
        if ax is None: fig, ax = plt.subplots()
        ax.grid(True)
        line = None

        conv = kwargs.pop('conv', None)
        if conv: ax.set_yscale('log')

        for t, data in self.get_data(run_id, depth, label, conv):
            if line is None:
                line = ax.plot(t, np.transpose(data), **kwargs)
            else:
                for ln, sub_data in zip(line, data):
                    ax.plot(t, sub_data, color=ln.get_color(), linestyle=ln.get_linestyle(), label=None)

        return ax
