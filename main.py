import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver


a = 0

run_id = 'test_run'
t_max, dt, r_max = 2000, 0.1, 30

depth = 1000


def main():
    solver = KerrSolver('bin', a)
    r, E, J = solver.circ_params(E=0.95)
    # print('r:', r, ' E:', E, ' J:', J) # r: 8.333333333333488  E: 0.95  J: 3.6084391824351596
    print('Solving...')

    solver.solve(run_id, depth, (t_max, dt, 6, E, J + 0.005, r_max), conv='self')

    print('Plotting...')

    solver.plot(run_id, depth, 'phi', conv='self')
    solver.plot_trajectory(run_id, depth)

    plt.show()


if __name__ == '__main__':
    main()
