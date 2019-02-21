''' Finite-difference implementation of the thin viscious disk equation,
    with first-order explicit time discretization and second-order central
    difference in space.

    Modified from github.com/python-hydro
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

plt.rc('font', size=20)
plt.rcParams['lines.linewidth'] = 3.0


class Grid1d(object):
    """ Sets up the one-dimensional grid on which the viscous disk equation is
        solved.
    """

    def __init__(self, nr, ng=1, rmin=0.1, rmax=5, gamma=1, constant_nu=False):
        self.nr = nr  # number of grid points
        self.ng = ng  # number of ghost cells at each boundary
        self.constant_nu = constant_nu

        self.rmin = rmin  # disk inner edge
        self.rmax = rmax  # disk outer edge

        self.ilo = ng  # grid point adjacent to the inner boundary
        self.ihi = ng + nr - 1  # grid point adjacent to the outer boundary

        self.dr = (rmax - rmin) / nr  # set up the spatial grid
        self.r = rmin + (np.arange(nr + 2 * ng) - ng + 0.5) * self.dr

        # surface density profile
        self.Sigma = np.zeros((nr + 2 * ng), dtype=np.float64)

        # viscosity profile
        self.gamma = gamma
        if(self.constant_nu):
            self.nu = np.ones_like(self.Sigma)
        else:
            self.nu = pow(self.r, gamma)

        # some helpful factors used in setting the boundary condition
        self.fac1_inner = self.nu[self.ilo] / self.nu[self.ilo - 1]
        self.fac2_inner = pow(self.r[self.ilo] / self.r[self.ilo - 1], 0.5)
        self.fac1_outer = self.nu[self.ihi] / self.nu[self.ihi + 1]
        self.fac2_outer = pow(self.r[self.ihi] / self.r[self.ihi + 1], 0.5)

    def scratch_array(self):
        # Initalize an empty array of the same shape as the grid
        return np.zeros((self.nr + 2 * self.ng), dtype=np.float64)

    def set_boundary_condition(self, sigma_zero, ibc='zero-torque',
                               obc='flat'):
        # set the inner boundary condition
        if(ibc == 'constant'):
            # Sigma = const. at r = rmin
            self.Sigma[0:self.ilo] = sigma_zero
        elif(ibc == 'flat'):
            # dSigma/dr = 0 at r = rmin
            self.Sigma[0:self.ilo] = self.Sigma[self.ilo]
        elif(ibc == 'zero mass flux'):
            # zero mass flux at rmin (vr = 0)
            self.Sigma[0:self.ilo] = (self.Sigma[self.ilo] *
                                      self.fac1_inner * self.fac1_inner)
        elif(ibc == 'zero-torque'):
            # zero torque at rmin
            self.Sigma[0:self.ilo] = 0
        else:
            print('Invalid option for the inner boundary condition')
            sys.exit()

        # set the outer boundary condition
        if(obc == 'flat'):
            # dSigma/dr = 0 at r = rmax
            self.Sigma[self.ihi + 1] = self.Sigma[self.ihi]
        elif(obc == 'zero mass flux'):
            # zero mass flux at rmax (vr = 0)
            self.Sigma[self.ihi + 1] = (self.Sigma[self.ihi] *
                                        self.fac1_outer * self.fac2_outer)
        elif(obc == 'zero-torque'):
            # zero torque at rmax
            self.Sigma[self.ihi + 1] = 0
        else:
            print('Invalid option for the outer boundary condition')
            sys.exit()


class Disk_Evolution(object):
    """ Solves the viscous disk equation explicitly on a 1d grid.
    """

    def __init__(self, grid, save_profiles=False, animation=False,
                 save_png=False, inner_boundary='zero-torque',
                 outer_boundary='flat'):
        self.grid = grid
        self.Sigma_zero = 0
        self.mstar = 0  # stellar mass
        self.mdot = 0
        self.t = 0  # time
        self.j = 0  # index to keep track of the number of time-steps
        self.k = 0  # profile number
        self.tau_visc = 0  # viscous time-scale
        self.save_profiles = save_profiles
        self.save_png = save_png
        self.animation = animation
        self.inner_boundary = inner_boundary
        self.outer_boundary = outer_boundary

        self.j_plot = 10
        self.j_save = 250
        self.j_png = 25

        if(save_profiles and not os.path.isdir('logs')):
            os.mkdir('logs')
        if(save_png and not os.path.isdir('img')):
            os.mkdir('img')

    def set_initial_condition(self, name, *args):
        # set the initial condition
        if(name == 'gaussian'):
            r0, sig = args
            self.grid.Sigma = np.exp(-pow(self.grid.r - r0, 2) / sig**2)
        else:
            # MMSN-like profile
            self.grid.Sigma = pow(self.grid.r, -1.5)
            if(self.inner_boundary == 'zero-torque'):
                self.Sigma_zero = self.grid.Sigma[0]
                self.grid.Sigma[0] = 0
            else:
                self.Sigma_zero = self.grid.Sigma[0]

    def first_spatial(self, f, i, dr):
        # discretized first order spatial derivative
        # Could do central difference instead: (f[i + 1] - f[i - 1]) / 2 dx,
        # but then the outer boundary condition has to be changed
        return ((f[i] - f[i - 1]) / dr)

    def second_spatial(self, f, i, dr):
        # discretized second order spatial derivative
        return ((f[i + 1] - 2 * f[i] + f[i - 1]) / dr**2)

    def do_evolve(self, c, tmax):
        gr = self.grid  # initialize the grid
        # set the time-step
        if(gr.constant_nu):
            dt = c * gr.dr**2 / (gr.nu[0])
            self.tau_visc = gr.r[0]**2 / gr.nu[0]
        else:
            dt = c * gr.dr**2 / max(gr.nu)  # time-step
            self.tau_visc = min(gr.r**2 / gr.nu)
        evol_time = tmax / self.tau_visc
        print('Evolving for {:.2f} viscous time-scales'.format(evol_time))
        Sigma_new = gr.scratch_array()  # initalize an empty array

        # Prepare the animation
        if(self.animation):
            plt.ion()
            fig = plt.figure(figsize=(8, 6))
            fig.set_dpi(128)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(gr.r, gr.Sigma / self.Sigma_zero, ls='--')
            x_arr, = plt.plot(gr.r, gr.Sigma / self.Sigma_zero)
            x_sctr, = plt.plot(gr.r, gr.Sigma / self.Sigma_zero, 'go')
            ax.set_xlabel('r')
            ax.set_ylabel(r'$\Sigma$')
            ax.set_xlim(left=0, right=gr.rmax)
            ann_str = 't = {:.2f}'
            ann = plt.annotate(ann_str.format(self.t), xy=(0.50, 0.75),
                               xycoords='figure fraction')
            fig.canvas.draw()

        # evolve for one time-step
        while self.t < tmax:
            # end at t = tmax
            if(self.t + dt > tmax):
                dt = tmax - self.t
            gr.set_boundary_condition(self.Sigma_zero, self.inner_boundary,
                                      self.outer_boundary)

            # loop over the spatial grid from ilo to ihi
            for i in range(gr.ilo, gr.ihi + 1):
                a1 = gr.nu[i] * self.second_spatial(gr.Sigma, i, gr.dr)
                b1 = gr.nu[i] * self.first_spatial(gr.Sigma, i, gr.dr)
                if(gr.constant_nu):
                    a2 = 0
                    a3 = 0
                    b2 = 0
                else:
                    a2 = (2 * self.first_spatial(gr.nu, i, gr.dr) *
                          self.first_spatial(gr.Sigma, i, gr.dr))
                    a3 = gr.Sigma[i] * self.second_spatial(gr.nu, i, gr.dr)
                    b2 = gr.Sigma[i] * self.first_spatial(gr.nu, i, gr.dr)
                # explicit diffusion step
                a = 3 * (a1 + a2 + a3)
                b = 9 * (b1 + b2) / (2 * gr.r[i])
                Sigma_new[i] = gr.Sigma[i] + dt * (a + b)

            gr.Sigma = Sigma_new  # store the new solution
            if(self.save_profiles):
                if(self.j % self.j_save == 0):
                    fname = 'logs/profile_' + str(self.k) + '.data'
                    np.savetxt(fname, np.c_[gr.r, gr.Sigma],
                               fmt='%.8f')
                    self.k = self.k + 1
            if(self.animation):
                if(self.j % self.j_plot == 0):
                    x_arr.set_ydata(gr.Sigma / self.Sigma_zero)
                    x_sctr.set_ydata(gr.Sigma / self.Sigma_zero)
                    ann.set_text(ann_str.format(self.t / tmax))
                    fig.canvas.draw()
                    plt.pause(0.0001)
                if(self.j % self.j_png == 0 and self.save_png):
                    png_name = 'img/plot_%d' % (1000 + self.j / self.j_png,)
                    plt.savefig(png_name)
            # trapezoidal rule
            delta_m = 0.5 * ((gr.r[1] - gr.r[0]) *
                             (gr.r[0] * gr.Sigma[0] + gr.r[1] * gr.Sigma[1]))
            self.mstar = self.mstar + delta_m
            self.t = self.t + dt
            self.j = self.j + 1


if __name__ == '__main__':
    nr = 256
    c = 0.1  # time-step factor
    tmax = 0.02
    g = Grid1d(nr=nr, ng=1, rmin=0.1, rmax=2, gamma=1, constant_nu=False)
    s = Disk_Evolution(g, save_profiles=False, animation=True, save_png=False,
                       inner_boundary='zero-torque', outer_boundary='flat')
    s.set_initial_condition('mmsn', 1, 0.1)
    s.do_evolve(c, tmax)
    print('Added mass: {:.2f}'.format(s.mstar))
    plt.savefig('surface_density.pdf', format='pdf',
                dpi=128, bbox_inches='tight', pad_inches=0.5)
