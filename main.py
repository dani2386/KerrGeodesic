import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver


a = 0.1

run_id = 'test_run'
params = (3, 0.01, 6, 0.1, 4) # (t_max, dt, r, E, J)

depth = 50


def main():
    solver = KerrSolver('bin', a)

    print('Solving...')

    solver.solve(run_id, params, depth, conv='self')

    print('Plotting...')

    solver.plot(run_id, 'phi', depth)
    plt.grid(True)
    solver.plot(run_id, 'phi', depth, conv='self')
    plt.grid(True)
    solver.plot_trajectory(run_id, depth)
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
