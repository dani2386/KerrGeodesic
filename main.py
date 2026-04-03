import numpy as np
import matplotlib.pyplot as plt
from src.kerr_solver import KerrSolver


# Global Simulation Constants
a = 0 # Spin parameter (0 = Schwarzschild)

run_id = 'test_run' # Run identifier
t_max, dt, m, r_max = 10000, 1, 1e-3, 30 # Time limit, step, test mass, and max radius
d, l = 1000, np.pi / 2 # GW observer distance and inclination angle

depth = 1000 # Number of steps to keep in RAM


def main():
    """
    This function performs the following steps:
        1. Initializes the KerrSolver physics engine
        2. Calculates stable circular orbit parameters (E, J) for a given radius or (r, J) for a given energy
        3. Integrates the equations of motion with errors
        4. Computes gravitational wave signal based on the resulting trajectory
        5. Generates diagnostic and physical plots
    """
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
