import numpy as np
import matplotlib.pyplot as plt
from src.ode_solver import ODESolver


class KerrSolver(ODESolver):
    def __init__(self, dir_name, a):
        self.a = a
    #    self.E = 0
    #    self.J = 0
        self.m = 1e-5
        self.M = 1

        super().__init__(dir_name, f=None)

    def _flux(self, r, phi, dr, dphi, d2r, d2phi, d3r, d3phi):

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
            - r * c * dphi**2
            - r * s * d2phi
        )

        d2y = (
            d2r * s
            + 2 * dr * c * dphi
            - r * s * dphi**2
            + r * c * d2phi
        )

        # --- third derivatives ---
        d3x = (
            d3r * c
            - 3 * d2r * s * dphi
            - 3 * dr * c * dphi**2
            - 3 * dr * s * d2phi
            + r * s * dphi**3
            - 3 * r * c * dphi * d2phi
            - r * s * d3phi
        )

        d3y = (
            d3r * s
            + 3 * d2r * c * dphi
            - 3 * dr * s * dphi**2
            + 3 * dr * c * d2phi
            - r * c * dphi**3
            - 3 * r * s * dphi * d2phi
            + r * c * d3phi
        )

        # --- quadrupole third derivatives ---
        # Ixx = m (x^2 - r^2/3)
        d3Ixx = self.m * (
            2 * (dx * d2x + x * d3x)
            - (2/3) * (r * d3r + 3 * dr * d2r)
        )

        # Iyy = m (y^2 - r^2/3)
        d3Iyy = self.m * (
            2 * (dy * d2y + y * d3y)
            - (2/3) * (r * d3r + 3 * dr * d2r)
        )

        # Ixy = m (x y)
        d3Ixy = self.m * (
            d3x * y + 3 * d2x * dy + 3 * dx * d2y + x * d3y
        )

        # --- energy flux ---
        flux = (1/5) * (d3Ixx**2 + d3Iyy**2 + 2 * d3Ixy**2)

        return -flux  # negative: energy is lost
    
    def _factors(self, r,E,J):
        delta = r**2 + self.a**2 - 2 * r
        Q = delta / (E * (r**2 + 2 * self.a**2 / r + self.a**2) - 2 * self.a * J / r)
        w = (J * Q + 2 * self.a / r) / (r**2 + 2 * self.a**2 / r + self.a**2)

        return delta, Q, w

    def d2r_dt2(self, r, pr, E, J, dr, dpr):
        # d2r_dt2 = ∂f/∂r *dr + ∂f/∂pr * dpr
        return (self.a**2 - 2*r + r**2)*(2*self.a**3*(-(self.a*E) + J)*pr*dr - 2*E*r**5*dpr + E*r**6*dpr + 2*E*r**4*(2*pr*dr + self.a**2*dpr) + 2*r*(-(self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a**3*(self.a*E - J)*dpr) - 2*self.a*r**3*(self.a*E*pr*dr + J*dpr) +  self.a*r**2*(6*(self.a*E - J)*pr*dr + (-4*self.a*E + self.a**3*E + 4*J)*dpr))/(r**2*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)**2)
    def d2phi_dt2(self, r, E, J, dr):
        return (-2*(self.a*(-(self.a*E) + J)**2 + 3*E*(self.a*E - J)*r**2 + E*J*r**3)*dr)/(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)**2
    def d3r_dt3(self, r, pr, E, J, dr, dpr,d2r_dt2):
        return (-2*E*r*(self.a**2 - 2*r + r**2)*(self.a**2 + 3*r**2)*dr*((-2*E*r**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + (E*r**3*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*self.a**3*(-(self.a*E) + J)*pr*dr + 2*E*r**4*((self.a**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*pr*dr)- 2*self.a*r**3*((J*pr**2*(self.a**2 - r)*(self.a**2 + (-2 + r)*r))/(r**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))) + (J*(2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + self.a*E*pr*dr) + 2*r*((self.a**3*(self.a*E - J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) - self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a*r**2*(((-4*self.a*E + self.a**3*E + 4*J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 6*(self.a*E - J)*pr*dr)) + 2*(-1 + r)*r*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*dr*((-2*E*r**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + (E*r**3*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*self.a**3*(-(self.a*E) + J)*pr*dr + 2*E*r**4*((self.a**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*pr*dr)- 2*self.a*r**3*((J*pr**2*(self.a**2 - r)*(self.a**2 + (-2 + r)*r))/(r**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))) + (J*(2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + self.a*E*pr*dr) + 2*r*((self.a**3*(self.a*E - J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) - self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a*r**2*(((-4*self.a*E + self.a**3*E + 4*J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 6*(self.a*E - J)*pr*dr)) - 2*(self.a**2 - 2*r + r**2)*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*dr*((-2*E*r**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + (E*r**3*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*self.a**3*(-(self.a*E) + J)*pr*dr + 2*E*r**4*((self.a**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*pr*dr)- 2*self.a*r**3*((J*pr**2*(self.a**2 - r)*(self.a**2 + (-2 + r)*r))/(r**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))) + (J*(2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + self.a*E*pr*dr) + 2*r*((self.a**3*(self.a*E - J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) - self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a*r**2*(((-4*self.a*E + self.a**3*E + 4*J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 6*(self.a*E - J)*pr*dr)) + 2*r*(self.a**2 - 2*r + r**2)*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*((-5*E*r*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))*dr)/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + (3*E*r**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))*dr)/((self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 4*E*r**3*dr*((self.a**2*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 2*pr*dr)- 3*self.a*r**2*dr*((J*pr**2*(self.a**2 - r)*(self.a**2 + (-2 + r)*r))/(r**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))) + (J*(2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4)))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + self.a*E*pr*dr) + dr*((self.a**3*(self.a*E - J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) - self.a*(2*self.a*E + self.a**3*E - 2*J)*pr*dr) + self.a*r*dr*(((-4*self.a*E + self.a**3*E + 4*J)*(pr**2*(self.a**2 - r)*r*(self.a**2 + (-2 + r)*r)**2*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r)) + (2*self.a*E + J*(-2 + r))**2*(2*self.a**3*J - self.a**4*E*(2 + r) + E*r**3*(-1 + r**3) - 2*self.a*J*(1 - r + r**3) + self.a**2*E*(2 - r + r**3 + r**4))))/(r**3*(self.a**2 + (-2 + r)*r)*(-2*self.a*J + E*r**3 + self.a**2*E*(2 + r))**2) + 6*(self.a*E - J)*pr*dr) + self.a**3*(-(self.a*E) + J)*dr*dpr + self.a**3*(-(self.a*E) + J)*pr*d2r_dt2 - self.a*(2*self.a*E + self.a**3*E - 2*J)*r*(dr*dpr + pr*d2r_dt2) + 3*self.a*(self.a*E - J)*r**2*(dr*dpr + pr*d2r_dt2) - self.a**2*E*r**3*(dr*dpr + pr*d2r_dt2) + 2*E*r**4*(dr*dpr + pr*d2r_dt2)))/(r**3*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)**3)
    def d3phi_dt3(self, r, E, J, dr,d2r_dt2):
        return (2*(-3*E*r*(2*self.a*E - 2*J + J*r)*(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*dr**2 + 2*E*(self.a**2 + 3*r**2)*(self.a*(-(self.a*E) + J)**2 + 3*E*(self.a*E - J)*r**2 + E*J*r**3)*dr**2 - (2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)*(self.a*(-(self.a*E) + J)**2 + 3*E*(self.a*E - J)*r**2 + E*J*r**3)*d2r_dt2))/(2*self.a*(self.a*E - J) + self.a**2*E*r + E*r**3)**3
    
    def _geodesic_eq(self, t, state):
        tau, phi, r, pr, E ,J= state
        delta, Q, w = self._factors(r,E,J)
        pr = self._pr(r, E, J)


        dr_dt = delta * Q * pr / r**2
        dpr_dt = (w**2 * (r**3 - self.a**2 + 2 * self.a * w - 1)) / (Q * r**2) + Q * pr**2 * (self.a**2 - r) / r**3
        
        d2r_dt2 = self.d2r_dt2(r, pr, E, J, dr_dt, dpr_dt)  # Placeholder for second derivative
        d2phi_dt2 = self.d2phi_dt2(r, E, J, dr_dt)  # Placeholder for second derivative
        d3r_dt3 = self.d3r_dt3(r, pr, E, J, dr_dt, dpr_dt, d2r_dt2)  # Placeholder for third derivative
        d3phi_dt3 = self.d3phi_dt3(r, E, J, dr_dt, d2r_dt2)  # Placeholder for third derivative
        
        dE_dt = self._flux(r, phi, dr_dt, w, d2r_dt2, d2phi_dt2, d3r_dt3, d3phi_dt3)  # -(self.m **2)*(32/5)*(w**6)*r**4
        dJ_dt =  0 # Incorrect
        return np.array([Q, w, dr_dt, dpr_dt, dE_dt, dJ_dt])

    def _pr(self, r, E, J):
        delta, Q, w = self._factors(r, E, J)
        return -(np.sqrt(np.abs(r**2 / (delta * Q**2) * (1 - 2 / r - Q**2 - w**2 * (r**2 + 2 * self.a**2 / r + self.a**2) + 4 * w * self.a / r))))
   
    def solve(self, run_id, params, depth):
        t, dt, r, E, J = params
        delta, Q, w = self._factors(r, E, J)

        pr = self._pr(r, E, J)

        self.f = self._geodesic_eq

        return super().solve(run_id, (t, dt, np.array([0, 0, r, pr, E, J])), depth)

    def plot(self, run_id, x_axis, y_axis, depth, ax=None, **kwargs):
        opts = {'t': None, 'tau': 0, 'phi': 1, 'r': 2, 'pr': 3, 'E': 4, 'J': 5}

        super().plot(run_id, opts[x_axis], opts[y_axis], depth, ax, **kwargs)

    def plot_trajectory(self, run_id, depth, ax=None, **kwargs):
        states  = f'{run_id}/states'

        if ax is None: fig, ax = plt.subplots()

        with self._file as file:
            n_max = file.load_metadata(run_id, 'n_max')

            for n in range(0, n_max, depth):
                phi = file.load(states, slice(n, min(n + depth, n_max + 1)))[:, 1]
                r = file.load(states, slice(n, min(n + depth, n_max + 1)))[:, 2]

                x = r * np.cos(phi)
                y = r * np.sin(phi)

                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                ax.plot(x, y, **kwargs)

        # --- Event horizon ---
        r_plus = self.M + np.sqrt(self.M**2 - self.a**2)

        theta = np.linspace(0, 2*np.pi, 500)
        x_h = r_plus * np.cos(theta)
        y_h = r_plus * np.sin(theta)

        ax.plot(x_h, y_h, 'k--', label='Event Horizon')
        ax.fill(x_h, y_h, color='black', alpha=0.3)


        return ax
