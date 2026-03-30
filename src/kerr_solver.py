import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.ode_solver import ODESolver


class KerrSolver(ODESolver):
    def __init__(self, dir_name, a):
        self.a = a
        self.r_plus = 1 + np.sqrt(1 - self.a**2)
        self.E = 0
        self.J = 0
        self.m = 1e-5

        super().__init__(dir_name, ('tau', 'phi', 'r', 'pr'), self._geodesic_eq)

    def _factors(self, r, E, J):
        delta = r**2 + self.a**2 - 2 * r
        A = r**2 + 2 * self.a**2 / r + self.a**2
        B = r**2 + 2 * self.a / r + self.a**2
        Q = delta / (E * A - 2 * self.a * J / r)
        w = (J * Q + 2 * self.a / r) / A

        return delta, A, B, Q, w

    def _d3I_dt3(self, r, phi, dr, dphi, d2r, d2phi, d3r, d3phi):
        # --- coordinates ---
        c = np.cos(phi)
        s = np.sin(phi)

        x = r * c
        y = r * s

        # --- first derivatives ---
        dx = dr * c - r * s * dphi
        dy = dr * s + r * c * dphi

        # --- second derivatives ---
        d2x = (
                d2r * c
                - 2 * dr * s * dphi
                - r * c * dphi ** 2
                - r * s * d2phi
        )

        d2y = (
                d2r * s
                + 2 * dr * c * dphi
                - r * s * dphi ** 2
                + r * c * d2phi
        )

        # --- third derivatives ---
        d3x = (
                d3r * c
                - 3 * d2r * s * dphi
                - 3 * dr * c * dphi ** 2
                - 3 * dr * s * d2phi
                + r * s * dphi ** 3
                - 3 * r * c * dphi * d2phi
                - r * s * d3phi
        )

        d3y = (
                d3r * s
                + 3 * d2r * c * dphi
                - 3 * dr * s * dphi ** 2
                + 3 * dr * c * d2phi
                - r * c * dphi ** 3
                - 3 * r * s * dphi * d2phi
                + r * c * d3phi
        )

        # --- quadrupole third derivatives ---
        # Ixx = m (x^2 - r^2/3)
        d3Ixx = self.m * (2 * (dx * d2x + x * d3x) - (2 / 3) * (r * d3r + 3 * dr * d2r))

        # Iyy = m (y^2 - r^2/3)
        d3Iyy = self.m * (2 * (dy * d2y + y * d3y) - (2 / 3) * (r * d3r + 3 * dr * d2r))

        # Ixy = m (x y)
        d3Ixy = self.m * (d3x * y + 3 * d2x * dy + 3 * dx * d2y + x * d3y)

        return d3Ixx, d3Iyy, d3Ixy

    def _d2I_dt2(self, r, phi, dr, dphi, d2r, d2phi):
        # --- coordinates ---
        c = np.cos(phi)
        s = np.sin(phi)

        x = r * c
        y = r * s

        # --- first derivatives ---
        dx = dr * c - r * s * dphi
        dy = dr * s + r * c * dphi

        # --- second derivatives ---
        d2x = (
                d2r * c
                - 2 * dr * s * dphi
                - r * c * dphi ** 2
                - r * s * d2phi
        )

        d2y = (
                d2r * s
                + 2 * dr * c * dphi
                - r * s * dphi ** 2
                + r * c * d2phi
        )

        # Ixx = m (x^2 - r^2/3)
        d2Ixx = self.m * (2 * (dx * dx + x * d2x) - (2 / 3) * (dr * dr + r * d2r))

        # Iyy = m (y^2 - r^2/3)
        d2Iyy = self.m * (2 * (dy * dy + y * d2y) - (2 / 3) * (dr * dr + r * d2r))

        # Ixy = m (x y)
        d2Ixy = self.m * (d2x * y + 2 * dx * dy + x * d2y)

        return d2Ixx, d2Iyy, d2Ixy

    def d2r_dt2(self, r, pr, E, J, dr, dpr):
        # d2r_dt2 = ∂f/∂r *dr + ∂f/∂pr * dpr
        return (self.a**2 - 2*r + r**2)*(2*self.a**3*(-(self.a*E) + J)*pr*dr - 2*E*r**5*dpr + E*r**6*dpr + 2*E*r**4*(2*pr*dr + self.a**2*dpr) + 2*r*(-(self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a**3*(self.a*E - J)*dpr) - 2*self.a*r**3*(self.a*E*pr*dr + J*dpr) +  self.a*r**2*(6*(self.a*E - J)*pr*dr + (-4*self.a*E + self.a**3*E + 4*J)*dpr))/(r**2*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)**2)
    def d2phi_dt2(self, r, E, J, dr):
        return (-2*(self.a*(-(self.a*E) + J)**2 + 3*E*(self.a*E - J)*r**2 + E*J*r**3)*dr)/(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)**2
    def d3r_dt3(self, r, pr, E, J, dr, dpr,d2r_dt2):
        return (-2*E*r*(self.a**2 - 2*r + r**2)*(self.a**2 + 3*r**2)*dr*((-2*E*r**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + (E*r**3*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*self.a**3*(-(self.a*E) + J)*pr*dr + 2*E*r**4*((self.a**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*pr*dr)- 2*self.a*r**3*((J*pr**2*(self.a**2 - r)*(self.a**2 + (-2 + r)*r))/(r**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))) + (J*(2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + self.a*E*pr*dr) + 2*r*((self.a**3*(self.a*E - J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) - self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a*r**2*(((-4*self.a*E + self.a**3*E + 4*J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 6*(self.a*E - J)*pr*dr)) + 2*(-1 + r)*r*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*dr*((-2*E*r**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + (E*r**3*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*self.a**3*(-(self.a*E) + J)*pr*dr + 2*E*r**4*((self.a**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*pr*dr)- 2*self.a*r**3*((J*pr**2*(self.a**2 - r)*(self.a**2 + (-2 + r)*r))/(r**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))) + (J*(2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + self.a*E*pr*dr) + 2*r*((self.a**3*(self.a*E - J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) - self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a*r**2*(((-4*self.a*E + self.a**3*E + 4*J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 6*(self.a*E - J)*pr*dr)) - 2*(self.a**2 - 2*r + r**2)*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*dr*((-2*E*r**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + (E*r**3*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*self.a**3*(-(self.a*E) + J)*pr*dr + 2*E*r**4*((self.a**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*pr*dr)- 2*self.a*r**3*((J*pr**2*(self.a**2 - r)*(self.a**2 + (-2 + r)*r))/(r**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))) + (J*(2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + self.a*E*pr*dr) + 2*r*((self.a**3*(self.a*E - J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) - self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a*r**2*(((-4*self.a*E + self.a**3*E + 4*J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 6*(self.a*E - J)*pr*dr)) + 2*r*(self.a**2 - 2*r + r**2)*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*((-5*E*r*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))*dr)/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + (3*E*r**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))*dr)/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 4*E*r**3*dr*((self.a**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*pr*dr)- 3*self.a*r**2*dr*((J*pr**2*(self.a**2 - r)*(self.a**2 + (-2 + r)*r))/(r**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))) + (J*(2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + self.a*E*pr*dr) + dr*((self.a**3*(self.a*E - J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) - self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a*r*dr*(((-4*self.a*E + self.a**3*E + 4*J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 6*(self.a*E - J)*pr*dr) + self.a**3*(-(self.a*E) + J)*dr*dpr + self.a**3*(-(self.a*E) + J)*pr*d2r_dt2 - self.a*(2*self.a*E + self.a**3*E - 2*J)*r*(dr*dpr + pr*d2r_dt2) + 3*self.a*(self.a*E - J)*r**2*(dr*dpr + pr*d2r_dt2) - self.a**2*E*r**3*(dr*dpr + pr*d2r_dt2) + 2*E*r**4*(dr*dpr + pr*d2r_dt2)))/(r**3*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)**3)
    def d3phi_dt3(self, r, E, J, dr,d2r_dt2):
        return (2*(-3*E*r*(2*self.a*E - 2*J + J*r)*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*dr**2 + 2*E*(self.a**2 + 3*r**2)*(self.a*(-(self.a*E) + J)**2 + 3*E*(self.a*E - J)*r**2 + E*J*r**3)*dr**2 - (2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*(self.a*(-(self.a*E) + J)**2 + 3*E*(self.a*E - J)*r**2 + E*J*r**3)*d2r_dt2))/(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)**3

    def _geodesic_eq_gw(self, t, state):
        tau, phi, r, pr, E, J = state
        delta, A, B, Q, w = self._factors(r, E, J)

        dr_dt = delta * Q * pr / r ** 2
        dpr_dt = (w**2 * (r**3 - self.a**2) + 2 * self.a * w - 1) / (Q * r**2) + Q * pr**2 * (self.a**2 - r) / r**3

        d2r_dt2 = self.d2r_dt2(r, pr, E, J, dr_dt, dpr_dt)  # Placeholder for second derivative
        d2phi_dt2 = self.d2phi_dt2(r, E, J, dr_dt)  # Placeholder for second derivative
        d3r_dt3 = self.d3r_dt3(r, pr, E, J, dr_dt, dpr_dt, d2r_dt2)  # Placeholder for third derivative
        d3phi_dt3 = self.d3phi_dt3(r, E, J, dr_dt, d2r_dt2)  # Placeholder for third derivative

        # Inertial quadrupole moment tensor third derivatives
        d3Ixx, d3Iyy, d3Ixy = self._d3I_dt3(r, phi, dr_dt, w, d2r_dt2, d2phi_dt2, d3r_dt3, d3phi_dt3)
        # --- energy flux ---
        flux = (1 / 5) * (d3Ixx ** 2 + d3Iyy ** 2 + 2 * d3Ixy ** 2)
        dE_dt = -flux  # -(self.m **2)*(32/5)*(w**6)*r**4

        # Inertial quadrupole moment tensor second derivatives
        d2Ixx, d2Iyy, d2Ixy = self._d2I_dt2(r, phi, dr_dt, w, d2r_dt2, d2phi_dt2)
        # --- angular momentum flux ---
        dJ_dt = -(2 / 5) * ((d3Ixy * (d2Iyy - d2Ixx) + d2Ixy * (d3Ixx - d3Iyy)))

        return np.array([Q, w, dr_dt, dpr_dt, dE_dt, dJ_dt])

    def _geodesic_eq(self, t, state):
        tau, phi, r, pr = state
        delta, A, B, Q, w = self._factors(r, self.E, self.J)

        dr_dt = delta * Q * pr / r ** 2
        dpr_dt = (w**2 * (r**3 - self.a**2) + 2 * self.a * w - 1) / (Q * r**2) + Q * pr**2 * (self.a**2 - r) / r**3

        return np.array([Q, w, dr_dt, dpr_dt])

    def circ_params(self, tol=1e-10, max_iter=100, **kwargs):
        r, E, J = kwargs.get('r', 10), kwargs.get('E', 1), 3.5
        x = (E, J) if kwargs.get('r') else (r, J)

        for i in range(max_iter):
            if kwargs.get('r'):
                E, J = x
            else:
                r, J = x

            delta, A, B, Q, w = self._factors(r, E, J)

            f = np.array([1 - 2 / r - Q**2 - w**2 * B + 4 * w * self.a / r, w**2 * (r**3 - self.a**2) + 2 * self.a * w - 1])

            if np.linalg.norm(f) < tol: return r, E, J

            df_dJ = np.array([
                -4 * self.a * Q**3 / (r * delta) + (4 * self.a / r - 2 * w * B) * (Q / A + 2 * self.a * J * Q**2 / (r * delta)),
                (2 * w * (r**3 - self.a**2) + 2 * self.a) * (Q / A + 2 * self.a * J * Q**2 / (r * delta))
            ])

            if kwargs.get('r'):
                df_dE = np.array([
                    2 * Q**3 * A / delta + (2 * w * B - 4 * self.a / r) * J * Q**2 / delta,
                    (2 * w * (r**3 - self.a**2) + 2 * self.a) * -J * Q**2 / delta
                ])
                df = np.column_stack((df_dE, df_dJ))
            else:
                dQ_dr = (2 * r - 2 - Q * (E * (2 * r - 2 * self.a**2 / r**2) + 2 * self.a * J / r**2)) / (E * A - 2 * self.a * J / r)
                dw_dr = (J * dQ_dr - 2 * self.a / r**2 - w * (2 * r - 2 * self.a**2 / r**2)) / A
                df_dr = np.array([
                    2 / r**2 - 2 * Q * dQ_dr - w**2 * (2 * r - 2 * self.a / r**2) + dw_dr * (4 * self.a / r - 2 * w * B) - 4 * w * self.a / r**2,
                    dw_dr * (2 * w * (r**3 - self.a**2) + 2 * self.a) + 3 * r**2 * w**2
                ])
                df = np.column_stack((df_dr, df_dJ))

            x = x - np.linalg.solve(df, f)

        return r, E, J

    def solve(self, run_id, depth, params, **kwargs):
        #t_max, dt, r, E, J, r_max = params
        t_max, dt, r, self.E, self.J, r_max = params
        #delta, A, B, Q, w = self._factors(r, E, J)
        delta, A, B, Q, w = self._factors(r, self.E, self.J)
        pr = -(np.sqrt(np.abs(r**2 / (delta * Q**2) * (1 - 2 / r - Q**2 - w**2 * B + 4 * w * self.a / r))))

        stop_cond = lambda t, data: (data[2] <= self.r_plus * 1.01) or (data[2] >= r_max)

        #return super().solve(run_id, depth, t_max, dt, np.array([0, 0, r, pr, E, J]), stop_cond, **kwargs)
        return super().solve(run_id, depth, t_max, dt, np.array([0, 0, r, pr]), stop_cond, **kwargs)

    def plot_trajectory(self, run_id, depth, ax=None, **kwargs):
        if ax is None: fig, ax = plt.subplots()
        ax.grid(True)

        ax.add_patch(Circle((0, 0), 1 + np.sqrt(1 - self.a**2), color='black', linestyle='--', fill=False))

        data_path = f'{run_id}/data/v1'

        with self._file as file:
            t_max, dt = file.load_metadata(run_id, ('t_max', 'dt'))
            n_max = int(t_max / dt)
            line = None

            for n in range(0, n_max + 1, depth):
                buf_len = min(depth, n_max - n + 1)

                phi = file.load(data_path, (slice(n, n + buf_len), 1))
                r = file.load(data_path, (slice(n, n + buf_len), 2))

                x = r * np.cos(phi)
                y = r * np.sin(phi)

                if not line:
                    line, = ax.plot(x, y, **kwargs)
                else:
                    ax.plot(x, y, color=line.get_color(), linestyle=line.get_linestyle(), label=None)

        return ax
