import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver


a = 0.1

run_id = 'test_run'
params = (1.5, 0.0001, 6, 0.1, 6)  # (t_max, dt, r, E, J)

depth = 1000


def main():
    solver = KerrSolver('bin', a)

    print('Solving...')

    solver.solve(run_id, params, depth, conv='self')

    print('Plotting...')

    solver.plot(run_id, 'phi', depth)
    solver.plot(run_id, 'phi', depth, conv='self')
    solver.plot_trajectory(run_id, depth)

    plt.show()


if __name__ == '__main__':
    main()
