import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver
import numpy as np


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

    print('Plotting GW polarizations...')
    h_plus = [h[0] for h in solver.h]
    h_cross = [h[1] for h in solver.h]
    t_gw = solver.t_gw  # Time array corresponding to GW data   
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_gw, h_plus, label='h_plus')
    plt.title('Gravitational Wave Polarization h_plus')
    plt.xlabel('Time')
    plt.ylabel('h_plus')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t_gw, h_cross, label='h_cross', color='orange')
    plt.title('Gravitational Wave Polarization h_cross')
    plt.xlabel('Time')
    plt.ylabel('h_cross')
    plt.legend()  

    print('Done.')
    plt.show()


if __name__ == '__main__':
    main()
