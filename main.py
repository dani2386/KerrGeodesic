import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver


a = 0.1
r0 = 9
m,M = 1e-5, 1
E0=(1-2*M/r0+a*M**0.5/r0**1.5)/(1-3*M/r0+2*a*M**0.5/r0**1.5)**0.5
L0 = ((M*r0)**0.5*-2*a*M/r0+M**0.5*a**2/r0**2)/(1-3*M/r0+2*a*M**0.5/r0**1.5)**0.5
params = (180, 0.01, r0, E0, L0)  # (t_max, dt, r, E, J)
run_id = 'test_run'
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
