import numpy as np
import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver


a = 0

run_id = 'test_run'
t_max, dt, m, r_max = 10000, 1, 1e-3, 30
d, l = 1000, np.pi / 2

depth = 1000


def main():
    solver = KerrSolver('bin', a)
    r, E, J = solver.circ_params(r=6)

    print('Solving...')

    solver.solve(run_id, depth, (t_max, dt, r, E, J, m, r_max), conv='self')
    solver.solve_gw(run_id, depth, (d, l))

    print('Plotting...')

    solver.plot(run_id, depth, 'phi')
    solver.plot(run_id, depth, 'phi', conv='self')
    solver.plot_trajectory(run_id, depth)
    solver.plot_gw(run_id, depth)

    plt.show()


if __name__ == '__main__':
    main()
