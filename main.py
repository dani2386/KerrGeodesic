import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver


a = 0.1

run_id = 'test_run'
params = (185, 0.01, 2, 0.8, 2)  # (t_max, dt, r, E, J)
depth = 50

def main():
    solver = KerrSolver('bin', a)

    print('Solving...')

    solver.solve(run_id, params, depth)

    print('Plotting...')

    solver.plot(run_id, 't', 'r', depth, color='r')
    plt.grid(True)
    solver.plot_trajectory(run_id, depth, color='r')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
