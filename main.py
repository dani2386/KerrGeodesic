import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver


a = 0

run_id = 'test_run'
params = (50, 0.01, 6, 0.1, 2.5, 10)  # (t_max, dt, r, E, J, r_max)

depth = 1000


def main():
    solver = KerrSolver('bin', a)

    print('Solving...')

    solver.solve(run_id, depth, params, conv='self')

    print('Plotting...')

    solver.plot(run_id, depth, 'phi')
    solver.plot(run_id, depth, 'phi', conv='self')
    solver.plot_trajectory(run_id, depth)

    plt.show()


if __name__ == '__main__':
    main()
