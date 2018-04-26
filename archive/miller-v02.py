#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:32:25 2013

@author: dtold
"""
from __future__ import print_function
from __future__ import division

from builtins import range
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as US


def find(val, arr):
    ind = 0
    mindiff = 1000.
    for i,arr_val in enumerate(arr):
        diff = abs(arr_val - val)
        if diff < mindiff:
            ind = i
            mindiff = diff
    return ind


class Miller(object):

    def __init__(self, gfile='pegasus-eq21.geqdsk', plot=True):
        self.gfile = gfile
        self.plot = plot
        self.ntheta = 150
        self.interpol_order = 3

        self.parse_gfile()
        self.calc_psirz_spline()

    def __call__(self, r_ov_a=0.76):
        self.calc_poi(r_ov_a=r_ov_a)

    def parse_gfile(self):
        with open(self.gfile, 'r') as gfile:
            eqdsk = gfile.readlines()

        # parse line 0
        print('Header: %s' % eqdsk[0])
        self.nw = int(eqdsk[0].split()[-2])
        self.nh = int(eqdsk[0].split()[-1])
        print('Resolution: %d x %d' % (self.nw, self.nh))

        # parse line 1
        entrylength = 16
        try:
            self.rdim, self.zdim, _, self.rmin, self.zmid = \
                [float(eqdsk[1][j * entrylength:(j + 1) * entrylength])
                 for j in range(len(eqdsk[1]) // entrylength)]
        except:
            entrylength = 15
            try:
                self.rdim, self.zdim, _, self.rmin, self.zmid = \
                    [float(eqdsk[1][j * entrylength:(j + 1) * entrylength])
                        for j in range(len(eqdsk[1]) // entrylength)]
            except:
                raise(IOError)

        # parse line 2
        self.R0, self.Z0, self.psiax, self.psisep, _ = \
            [float(eqdsk[2][j * entrylength:(j + 1) * entrylength])
                for j in range(len(eqdsk[2]) // entrylength)]

        # parse line 3
        _, psiax2, _, rmag2, _ = \
            [float(eqdsk[3][j * entrylength:(j + 1) * entrylength])
                for j in range(len(eqdsk[3]) // entrylength)]

        # parse line 4
        zmag2, _, psisep2, _, _ = \
            [float(eqdsk[4][j * entrylength:(j + 1) * entrylength])
                for j in range(len(eqdsk[4]) // entrylength)]

        if self.R0 != rmag2:
            sys.exit('Inconsistent self.R0: %7.4g, %7.4g' % (self.R0, rmag2))
        if psiax2 != self.psiax:
            sys.exit('Inconsistent psiax: %7.4g, %7.4g' % (self.psiax, psiax2))
        if self.Z0 != zmag2:
            sys.exit('Inconsistent self.Z0: %7.4g, %7.4g' % (self.Z0, zmag2))
        if psisep2 != self.psisep:
            sys.exit('Inconsistent psisep: %7.4g, %7.4g' %
                     (self.psisep, psisep2))

        print('\n*** Magnetic axis and LCFS ***')
        print('R mag. axis = {:.3g} m'.format(self.R0))
        print('Z mag. axis = {:.3g} m'.format(self.Z0))

        # read flux profiles and 2D flux grid
        # pol current (F=RBt) [T-m] on uniform flux grid
        self.F_ff = np.empty(self.nw, dtype=float)
        # pressure [Pa] on uniform flux grid
        self.p_ff = np.empty(self.nw, dtype=float)
        # FF'=FdF/dpsi on uniform flux grid
        self.ffprime_ff = np.empty(self.nw, dtype=float)
        # dp/dpsi [Pa/(Wb/rad)] on uniform flux grid
        self.pprime_ff = np.empty(self.nw, dtype=float)
        # q safety factor on uniform flux grid
        self.qpsi_ff = np.empty(self.nw, dtype=float)
        # pol. flux [Wb/rad] on rectangular grid
        psirz_1d = np.empty(self.nw * self.nh, dtype=float)
        start_line = 5
        lines = np.arange(self.nw // 5)
        if self.nw % 5 != 0:
            lines = np.arange(self.nw // 5 + 1)
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.F_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.p_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.ffprime_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.pprime_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        lines_twod = np.arange(self.nw * self.nh // 5)
        if self.nw * self.nh % 5 != 0:
            lines_twod = np.arange(self.nw * self.nh // 5 + 1)
        for i in lines_twod:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            psirz_1d[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1
        self.psirz = psirz_1d.reshape(self.nh, self.nw)

        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.qpsi_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        if self.psisep < self.psiax:
            self.psirz = -self.psirz
            self.ffprime_ff = -self.ffprime_ff
            self.pprime_ff = -self.pprime_ff
            self.psiax *= -1
            self.psisep *= -1

        print('psi-axis = {:.3e} Wb/rad'.format(self.psiax))
        print('psi-sep = {:.3e} Wb/rad'.format(self.psisep))

        # plot eqdsk profiles
        self.psinorm_grid = np.linspace(0, 1, self.nw)
        self.psi_grid = np.linspace(self.psiax, self.psisep, self.nw)

        if self.plot:
            # plot profile quantities
            plt.figure()
            plt.gcf().set_size_inches(6,6)
            plt.subplot(3, 2, 1)
            plt.plot(self.psinorm_grid, self.F_ff)
            plt.title(r'$F=R\,B_t$')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$F$ [m-T]', fontsize=14)
            #plt.axvline(self.psin_pos, 0, 1, ls='--', color='k', lw=2)

            plt.subplot(3, 2, 2)
            plt.plot(self.psinorm_grid, self.ffprime_ff)
            plt.title(r"$F*dF/d\Psi$")
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r"$FF'(\Psi)$", fontsize=14)

            plt.subplot(3, 2, 3)
            plt.plot(self.psinorm_grid, self.p_ff/1e3)
            plt.title('Pressure')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$p$ [kPa]', fontsize=14)

            plt.subplot(3, 2, 4)
            plt.plot(self.psinorm_grid, self.pprime_ff/1e3)
            plt.title(r"$dp/d\Psi$")
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r"$p'(\Psi)$", fontsize=14)

            plt.subplot(3, 2, 5)
            plt.plot(self.psinorm_grid, self.qpsi_ff)
            plt.title('Safety Factor')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$q$', fontsize=14)
            plt.tight_layout(pad=0.3)

    def calc_psirz_spline(self):

        # construct rectangular RZ grid
        dw = self.rdim / (self.nw - 1)
        dh = self.zdim / (self.nh - 1)
        rgrid = np.array([self.rmin + i * dw for i in range(self.nw)])
        zgrid = np.array([self.zmid - self.zdim / 2 +
                          i * dh for i in range(self.nh)])
        if self.plot:
            # plot psirz on r/z grid
            plt.figure()
            plt.contour(rgrid, zgrid, self.psirz, 70)
            plt.gca().set_aspect('equal')

        # flux surface R/Z coords. on uniform flux and theta grids
        nr = 100
        self.R_ftgrid = np.empty((self.nw, self.ntheta), dtype=float)
        self.Z_ftgrid = np.empty((self.nw, self.ntheta), dtype=float)
        self.theta_arr = np.linspace(-np.pi, np.pi, self.ntheta)
        t1 = np.arctan2(self.zmid - self.zdim / 2 - self.Z0,
                        self.rmin - self.R0) # angle(mag axis to bot. left)
        t2 = np.arctan2(self.zmid - self.zdim / 2 - self.Z0,
                        self.rmin + self.rdim - self.R0) # angle (mag ax to bot. rt)
        t3 = np.arctan2(self.zmid + self.zdim / 2 - self.Z0,
                        self.rmin + self.rdim - self.R0) # angle (mag ax to top rt)
        t4 = np.arctan2(self.zmid + self.zdim / 2 - self.Z0,
                        self.rmin - self.R0) # angle (mag ax to top left)
        # spline object for psi on RZ grid
        self.psi_spl = RBS(zgrid, rgrid, self.psirz,
                      kx=self.interpol_order,
                      ky=self.interpol_order)
        psilimit = self.psisep + (self.psisep - self.psiax) * 0.05
        for j, theta in enumerate(self.theta_arr):
            if theta < t1 or theta >= t4:
                rad = (self.rmin - self.R0) / np.cos(theta)
            elif theta < t2 and theta >= t1:
                rad = -(self.Z0 - self.zmid + self.zdim / 2) / np.sin(theta)
            elif theta < t3 and theta >= t2:
                rad = (self.rmin + self.rdim - self.R0) / np.cos(theta)
            elif theta < t4 and theta >= t3:
                rad = (self.zmid + self.zdim / 2 - self.Z0) / np.sin(theta)
            else:
                raise(ValueError)
            dr = rad / (nr - 1) * np.cos(theta)
            dz = rad / (nr - 1) * np.sin(theta)
            # RZ coordinates from axis at fixed poloidal angle
            r_pol = np.array([self.R0 + i * dr for i in range(nr)])
            z_pol = np.array([self.Z0 + i * dz for i in range(nr)])
            psi_rad = self.psi_spl.ev(z_pol, r_pol)
            psi_rad[0] = self.psiax
            # must restrict interpolation range because of non-monotonic psi around coils
            end_ind = 0
            for i in range(nr - 1):
                if psi_rad[i] > psilimit:
                    break
                if psi_rad[i + 1] <= psi_rad[i] and i < nr - 2:
                    psi_rad[i + 1] = 0.5 * (psi_rad[i] + psi_rad[i + 2])
                end_ind += 1

            # interp objects for indices
            psi_int = interp1d(psi_rad[:end_ind + 1],
                               np.arange(end_ind + 1),
                               kind=self.interpol_order)
            # near psi-grid index for separatrix
            indsep = int(psi_int(self.psisep)) + 3
            # RZ interp. objects along poloidal line from axis
            R_int = interp1d(psi_rad[:indsep],
                             r_pol[:indsep],
                             kind=self.interpol_order)
            Z_int = interp1d(psi_rad[:indsep],
                             z_pol[:indsep],
                             kind=self.interpol_order)
            # RZ coords of FS grid at fixed theta
            self.R_ftgrid[:, j] = R_int(self.psi_grid)
            self.Z_ftgrid[:, j] = Z_int(self.psi_grid)

        # find average elevation for all flux surfaces
        self.Z_avg_fs = np.empty(self.nw, dtype=float)
        for i in range(self.nw):
            ds = np.empty(self.ntheta, dtype=float)
            ds[1:self.ntheta - 1] = 0.5 * np.sqrt((self.R_ftgrid[i, 2:self.ntheta] \
                - self.R_ftgrid[i, 0:self.ntheta - 2])**2 \
                + (self.Z_ftgrid[i, 2:self.ntheta] \
                - self.Z_ftgrid[i, 0:self.ntheta - 2])**2)
            ds[0] = 0.5 * np.sqrt((self.R_ftgrid[i, 1] - self.R_ftgrid[i, -1]) ** 2 \
                                   + (self.Z_ftgrid[i, 1] - self.Z_ftgrid[i, -1])**2)
            ds[-1] = 0.5 * np.sqrt((self.R_ftgrid[i, 0] - self.R_ftgrid[i, -2]) ** 2 \
                                   + (self.Z_ftgrid[i, 0] - self.Z_ftgrid[i, -2])**2)
            self.Z_avg_fs[i] = np.average(self.Z_ftgrid[i, :], weights=ds)

        # find self.R_major_ff for all flux surfaces
        self.R_major_ff = np.empty(self.nw, dtype=float)
        self.R_major_ff[0] = self.R0
        self.r_minor_ff = np.empty(self.nw, dtype=float)
        self.r_minor_ff[0] = 0.
        itheta = self.ntheta // 4
        # loop over flux grid
        for i in range(1, self.nw):
            # low field side
            R_array = self.R_ftgrid[i, itheta:3*itheta]
            Z_array = self.Z_ftgrid[i, itheta:3*itheta]
            Z_int = interp1d(Z_array, R_array, kind=self.interpol_order)
            R_out = Z_int(self.Z_avg_fs[i])
            # high field side
            R_array = np.roll(self.R_ftgrid[i, :-1], self.ntheta // 2)[itheta:3*itheta]
            Z_array = np.roll(self.Z_ftgrid[i, :-1], self.ntheta // 2)[itheta:3*itheta]
            # have to use negative Z_array here to have increasing order
            Z_int = interp1d(-Z_array, R_array, kind=self.interpol_order)
            R_in = Z_int(-self.Z_avg_fs[i])
            self.R_major_ff[i] = 0.5 * (R_out + R_in)  # R_maj at self.Z_avg_fs
            self.r_minor_ff[i] = 0.5 * (R_out - R_in)  # r_min at self.Z_avg_fs

        self.R0_lcfs = self.R_major_ff[-1]
        self.a_lcfs = self.r_minor_ff[-1]

        print('R0_lcfs = {:.3g} m'.format(self.R0_lcfs))
        print('a_lcfs = {:.3g} m'.format(self.a_lcfs))
        print('eps_lcfs = {:.3g}'.format(self.a_lcfs / self.R0_lcfs))

    def calc_poi(self, r_ov_a=0.76):

        # splines on r_minor space
        psi_grid_spl = US(self.r_minor_ff,
                          self.psi_grid,
                          k=self.interpol_order,
                          s=1e-5)
        q_spl = US(self.r_minor_ff,
                   self.qpsi_ff,
                   k=self.interpol_order,
                   s=1e-5)
        R0_spl = US(self.r_minor_ff,
                    self.R_major_ff,
                    k=self.interpol_order,
                    s=1e-5)
        Z0_spl = US(self.r_minor_ff,
                    self.Z_avg_fs,
                    k=self.interpol_order,
                    s=1e-5)
        F_spl = US(self.r_minor_ff,
                   self.F_ff,
                   k=self.interpol_order,
                   s=1e-5)
        p_spl = US(self.r_minor_ff,
                   self.p_ff,
                   k=self.interpol_order,
                   s=1e-5)
        pprime_spl = US(self.r_minor_ff,
                        self.pprime_ff,
                        k=self.interpol_order,
                        s=1e-4)
        # splines on psi space
        q_spl_psi = US(self.psi_grid,
                       self.qpsi_ff,
                       k=self.interpol_order,
                       s=1e-5)
        r_min_spl = US(self.psi_grid,
                       self.r_minor_ff,
                       k=self.interpol_order,
                       s=1e-5)
        # position values
        r_poi = r_ov_a * self.a_lcfs  # r = r/a * a; FS minor radius
        psi_poi = float(psi_grid_spl(r_poi))  # psi at FS
        self.psin_poi = (psi_poi - self.psiax) / (self.psisep - self.psiax)  # psi-norm at FS
        R0_poi = float(R0_spl(r_poi))  # R_maj of FS
        Z0_poi = float(Z0_spl(r_poi))
        F_poi = float(F_spl(r_poi))  # F of FS
        p_poi = float(p_spl(r_poi))
        pprime_poi = float(pprime_spl(r_poi))
        Bref_poi = F_poi / R0_poi
        q_poi = float(q_spl_psi(psi_poi))
        drdpsi_poi = float(r_min_spl.derivatives(psi_poi)[1])
        omp_poi = -float((self.a_lcfs / p_poi) * (pprime_poi / drdpsi_poi))

        print('\n*** FS at r/a = {:.2f} ***'.format(r_ov_a))
        print('r_min = {:.3f} m'.format(r_poi))
        print('R_maj = {:.3f} m'.format(R0_poi))
        print('eps = {:.3f}'.format(r_poi / R0_poi))
        print('q = {:.3f}'.format(q_poi))
        print('psi = {:.3e} Wb/rad'.format(psi_poi))
        print('psi_N = {:.3f}'.format(self.psin_poi))
        print('p = {:.3g} Pa'.format(p_poi))
        print('dp/dpsi = {:.3g} Pa/(Wb/rad)'.format(pprime_poi))
        print('dr/dpsi = {:.3g} m/(Wb/rad)'.format(drdpsi_poi))
        print('omp = {:.4g} (with Lref=a)'.format(omp_poi))

        # find psi index of interest (for the specified r/a position)
        flux_poi_ind = find(r_ov_a, self.r_minor_ff / self.a_lcfs)

        # psi-width, number of flux surfaces around position of interest
        pw = self.nw // 6
        psi_stencil = np.arange(flux_poi_ind - pw // 2, flux_poi_ind + pw // 2)
        if psi_stencil[0] < 1:
            psi_stencil = [psi_sten + 1 - psi_stencil[0]
                           for i,psi_sten in enumerate(psi_stencil)]
        if psi_stencil[-1] > self.nw - 1:
            psi_stencil = [psi_sten - (psi_stencil[-1] - self.nw + 1)
                for i,psi_sten in enumerate(psi_stencil)]


        # calc kappa and delta
        R_tm = np.empty((pw, self.ntheta), dtype=float)
        Z_tm = np.empty((pw, self.ntheta), dtype=float)
        R_extended = np.empty(2 * self.ntheta - 1, dtype=float)
        Z_extended = np.empty(2 * self.ntheta - 1, dtype=float)
        theta_tmp = np.linspace(-2. * np.pi, 2 * np.pi, 2 * self.ntheta - 1)
        for i in psi_stencil:
            imod = i - psi_stencil[0]
            R_extended[0:(self.ntheta - 1) // 2] = self.R_ftgrid[i, (self.ntheta + 1) // 2:-1]
            R_extended[(self.ntheta - 1) // 2:(3 * self.ntheta - 3) // 2] = self.R_ftgrid[i, :-1]
            R_extended[(3 * self.ntheta - 3) // 2:] = self.R_ftgrid[i, 0:(self.ntheta + 3) // 2]
            Z_extended[0:(self.ntheta - 1) // 2] = self.Z_ftgrid[i, (self.ntheta + 1) // 2:-1]
            Z_extended[(self.ntheta - 1) // 2:(3 * self.ntheta - 3) // 2] = self.Z_ftgrid[i, :-1]
            Z_extended[(3 * self.ntheta - 3) // 2:] = self.Z_ftgrid[i, 0:(self.ntheta + 3) // 2]
            theta_mod_ext = np.arctan2(
                Z_extended - self.Z_avg_fs[i], R_extended - self.R_major_ff[i])
            # introduce 2pi shifts to theta_mod_ext
            for ind in range(self.ntheta):
                if theta_mod_ext[ind + 1] < 0. \
                    and theta_mod_ext[ind] > 0. \
                    and abs(theta_mod_ext[ind + 1] - theta_mod_ext[ind]) > np.pi:
                        lshift_ind = ind
                if theta_mod_ext[-ind - 1] > 0. \
                    and theta_mod_ext[-ind] < 0. \
                    and abs(theta_mod_ext[-ind - 1] - theta_mod_ext[-ind]) > np.pi:
                        rshift_ind = ind
            theta_mod_ext[-rshift_ind:] += 2. * np.pi
            theta_mod_ext[:lshift_ind + 1] -= 2. * np.pi
            theta_int = interp1d(theta_mod_ext, theta_tmp,
                                 kind=self.interpol_order)
            R_int = interp1d(theta_mod_ext, R_extended,
                             kind=self.interpol_order)
            Z_int = interp1d(theta_mod_ext, Z_extended,
                             kind=self.interpol_order)
            R_tm[imod] = R_int(self.theta_arr)
            Z_tm[imod] = Z_int(self.theta_arr)
        R_sym = np.empty((pw, self.ntheta), dtype=float)
        Z_sym = np.empty((pw, self.ntheta), dtype=float)
        for i in psi_stencil:
            imod = i - psi_stencil[0]
            Z_sym[imod, :] = 0.5 * (Z_tm[imod, :] - Z_tm[imod, ::-1]) \
                + self.Z_avg_fs[i]
            R_sym[imod, :] = 0.5 * (R_tm[imod, :] + R_tm[imod, ::-1])
        kappa = np.empty(pw, dtype=float)
        delta_upper = np.empty(pw, dtype=float)
        delta_lower = np.empty(pw, dtype=float)
        for i in psi_stencil:
            imod = i - psi_stencil[0]
            stencil_width = self.ntheta // 10
            for o in range(2):
                if o:
                    ind = np.argmax(Z_sym[imod])
                    section = np.arange(ind + stencil_width // 2,
                                        ind - stencil_width // 2, -1)
                else:
                    ind = np.argmin(Z_sym[imod])
                    section = np.arange(ind - stencil_width // 2,
                                        ind + stencil_width // 2)
                x = R_sym[imod, section]
                y = Z_sym[imod, section]
                y_int = interp1d(x, y, kind=self.interpol_order)
                x_fine = np.linspace(
                    np.amin(x), np.amax(x), stencil_width * 100)
                y_fine = y_int(x_fine)
                if o:
                    x_at_extremum = x_fine[np.argmax(y_fine)]
                    delta_upper[imod] = (
                        self.R_major_ff[i] - x_at_extremum) / self.r_minor_ff[i]
                    Z_max = np.amax(y_fine)
                else:
                    x_at_extremum = x_fine[np.argmin(y_fine)]
                    delta_lower[imod] = (
                        self.R_major_ff[i] - x_at_extremum) / self.r_minor_ff[i]
                    Z_min = np.amin(y_fine)
            kappa[imod] = (Z_max - Z_min) / 2. / self.r_minor_ff[i]
        delta = 0.5 * (delta_upper + delta_lower)


        # calc zeta
        zeta_arr = np.empty((pw, 4), dtype=float)
        zeta = np.empty(pw, dtype=float)
        for i in psi_stencil:
            imod = i - psi_stencil[0]
            x = np.arcsin(delta[imod])
            for o in range(4):
                if o == 0:
                    val = np.pi / 4
                    searchval = np.cos(val + x / np.sqrt(2))
                    searcharr = (R_sym[imod] - self.R_major_ff[i]) / self.r_minor_ff[i]
                elif o == 1:
                    val = 3 * np.pi / 4
                    searchval = np.cos(val + x / np.sqrt(2))
                    searcharr = (R_sym[imod] - self.R_major_ff[i]) / self.r_minor_ff[i]
                elif o == 2:
                    val = -np.pi / 4
                    searchval = np.cos(val - x / np.sqrt(2))
                    searcharr = (R_sym[imod] - self.R_major_ff[i]) / self.r_minor_ff[i]
                elif o == 3:
                    val = -3 * np.pi / 4
                    searchval = np.cos(val - x / np.sqrt(2))
                    searcharr = (R_sym[imod] - self.R_major_ff[i]) / self.r_minor_ff[i]
                if o in [0, 1]:
                    searcharr2 = searcharr[self.ntheta // 2:]
                    ind = find(searchval, searcharr2) + self.ntheta // 2
                else:
                    searcharr2 = searcharr[0:self.ntheta // 2]
                    ind = find(searchval, searcharr2)
                section = np.arange(ind - stencil_width // 2,
                                    ind + stencil_width // 2)
                theta_sec = self.theta_arr[section]
                if o in [0, 1]:
                    theta_int = interp1d(-searcharr[section], theta_sec, kind=self.interpol_order)
                    theta_of_interest = theta_int(-searchval)
                else:
                    theta_int = interp1d(searcharr[section], theta_sec, kind=self.interpol_order)
                    theta_of_interest = theta_int(searchval)
                Z_sec = Z_sym[imod, section]
                Z_sec_int = interp1d(
                    theta_sec, Z_sec, kind=self.interpol_order)
                Z_val = Z_sec_int(theta_of_interest)
                zeta_arr[imod, o] = np.arcsin(
                    (Z_val - self.Z_avg_fs[i]) / kappa[imod] / self.r_minor_ff[i])
            zeta_arr[imod, 1] = np.pi - zeta_arr[imod, 1]
            zeta_arr[imod, 3] = -np.pi - zeta_arr[imod, 3]
            zeta[imod] = 0.25 * (np.pi + zeta_arr[imod, 0] -
                                 zeta_arr[imod, 1] - zeta_arr[imod, 2] + zeta_arr[imod, 3])

        # RZ at point of interest (outer midplane on FS)
        pscan = np.array([-1,0,1])*0.01
        modB = np.zeros(pscan.shape)
        for i,p in enumerate(pscan):
            R_poi = R0_poi + r_poi + p
            dpsidr = self.psi_spl(Z0_poi, R_poi, dx=0, dy=1)
            dpsidz = self.psi_spl(Z0_poi, R_poi, dx=1, dy=0)
            Br = -dpsidz/R_poi
            Bz = dpsidr/R_poi
            Bt = F_poi/R_poi
            modB[i] = np.sqrt(Br**2+Bz**2+Bt**2)
        B_poi = modB[1]
        dBdr = (modB[2]-modB[0]) / (pscan[2]-pscan[0])

        print('*** Outer midplane point-of-interest ***')
        print('R_omp = R_maj+r_min = {:.3f} m'.format(R0_poi+r_poi))
        print('Z_omp = {:.3f} m'.format(Z0_poi))
        print('|B_omp| = {:.3g} T'.format(B_poi))
        print('d|B|/dr = {:.3g} T/m'.format(dBdr))

        # calc dr/dR, amhd, and derivs
        drR = np.empty(pw, dtype=float)
        amhd = np.empty(pw, dtype=float)
        s_kappa = np.empty(pw, dtype=float)
        s_delta = np.empty(pw, dtype=float)
        s_zeta = np.empty(pw, dtype=float)
        bp = np.empty(pw, dtype=float)
        bt = np.empty(pw, dtype=float)
        b = np.empty(pw, dtype=float)
        kappa_spl = US(self.r_minor_ff[psi_stencil],
                       kappa, k=self.interpol_order, s=1e-5)
        delta_spl = US(self.r_minor_ff[psi_stencil],
                       delta, k=self.interpol_order, s=1e-5)
        zeta_spl = US(self.r_minor_ff[psi_stencil],
                      zeta, k=self.interpol_order, s=1e-5)
        for i in psi_stencil:
            imod = i - psi_stencil[0]
            amhd[imod] = -self.qpsi_ff[i]**2 * self.R_major_ff[i] * self.pprime_ff[i] * \
                8 * np.pi * 1e-7 / Bref_poi**2 / \
                r_min_spl.derivatives(self.psi_grid[i])[1]
            drR[imod] = R0_spl.derivatives(self.r_minor_ff[i])[1]
            s_kappa[imod] = kappa_spl.derivatives(self.r_minor_ff[i])[
                1] * self.r_minor_ff[i] / kappa[imod]
            s_delta[imod] = delta_spl.derivatives(self.r_minor_ff[i])[
                1] * self.r_minor_ff[i] / np.sqrt(1 - delta[imod]**2)
            s_zeta[imod] = zeta_spl.derivatives(self.r_minor_ff[i])[
                1] * self.r_minor_ff[i]
            R = self.R_major_ff[i] + self.r_minor_ff[i]
            Z = self.Z_avg_fs[i]
            Br = -self.psi_spl(Z, R, dx=1, dy=0)/R
            Bz = self.psi_spl(Z, R, dx=0, dy=1)/R
            bp[imod] = np.sqrt(Br**2+Bz**2)
            bt[imod] = self.F_ff[i]/R
            b[imod] = np.sqrt(bp[imod]**2+bt[imod]**2)
        amhd_spl = US(self.r_minor_ff[psi_stencil],
                      amhd,
                      k=self.interpol_order,
                      s=1e-3)
        drR_spl = US(self.r_minor_ff[psi_stencil],
                     drR,
                     k=self.interpol_order,
                     s=1e-5)
        b_spl = US(self.psinorm_grid[psi_stencil],
                     b,
                     k=self.interpol_order,
                     s=1e-5)

        if self.plot:
            plt.figure()
            plt.gcf().set_size_inches(6,8)
            plt.subplot(4, 2, 1)
            plt.plot(self.psinorm_grid[psi_stencil], kappa, '-d')
            plt.title('Elongation')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$\kappa$', fontsize=14)
            plt.axvline(self.psin_poi, 0, 1, ls='--', color='k', lw=2)

            plt.subplot(4, 2, 2)
            plt.plot(self.psinorm_grid[psi_stencil], s_kappa, '-d')
            plt.title(r'$r/\kappa*(d\kappa/dr)$')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$s_\kappa$', fontsize=14)
            plt.axvline(self.psin_poi, 0, 1, ls='--', color='k', lw=2)

            plt.subplot(4, 2, 3)
            plt.plot(self.psinorm_grid[psi_stencil], delta, '-d')
            plt.title('Triangularity')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$\delta$', fontsize=14)
            plt.axvline(self.psin_poi, 0, 1, ls='--', color='k', lw=2)

            plt.subplot(4, 2, 4)
            plt.plot(self.psinorm_grid[psi_stencil], s_delta, '-d')
            plt.title(r'$r/\delta*(d\delta/dr)$')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$s_\delta$', fontsize=14)
            plt.axvline(self.psin_poi, 0, 1, ls='--', color='k', lw=2)

            plt.subplot(4, 2, 5)
            plt.plot(self.psinorm_grid[psi_stencil], zeta, '-d')
            plt.title('Squareness')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$\zeta$', fontsize=14)
            plt.axvline(self.psin_poi, 0, 1, ls='--', color='k', lw=2)

            plt.subplot(4, 2, 6)
            plt.plot(self.psinorm_grid[psi_stencil], s_zeta, '-d')
            plt.title(r'$r/\zeta*(d\zeta/dr)$')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$s_\zeta$', fontsize=14)
            plt.axvline(self.psin_poi, 0, 1, ls='--', color='k', lw=2)

            plt.subplot(4, 2, 7)
            plt.plot(self.psinorm_grid[psi_stencil], b, '-d')
            plt.plot(self.psinorm_grid[psi_stencil], bp)
            plt.plot(self.psinorm_grid[psi_stencil], bt)
            plt.title('|B|')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'|B|', fontsize=14)
            plt.axvline(self.psin_poi, 0, 1, ls='--', color='k', lw=2)

            plt.subplot(4, 2, 8)
            plt.plot(self.psinorm_grid[psi_stencil[1:-2]],
                     b_spl(self.psinorm_grid[psi_stencil[1:-2]], nu=1),
                     '-d')
            plt.title('|B| deriv.')
            plt.xlabel(r'$\Psi_N$', fontsize=14)
            plt.ylabel(r'$d|B|/d\Psi_N$', fontsize=14)
            plt.axvline(self.psin_poi, 0, 1, ls='--', color='k', lw=2)
            plt.tight_layout(pad=0.3)


        ind = flux_poi_ind - psi_stencil[0]
        print('\n\nShaping parameters for flux surface r=%9.5g, r/a=%9.5g:' %
              (r_poi, r_ov_a))
        print('Copy the following block into a GENE parameters file:\n')
        print('trpeps  = %9.5g' % (r_poi / R0_poi))
        print('q0      = %9.5g' % q_spl(r_poi))
        print('shat    = %9.5g !(defined as r/q*dq_dr)' % (r_poi / q_spl(r_poi) \
                                 * q_spl.derivatives(r_poi)[1]))
        print('amhd    = %9.5g' % amhd_spl(r_poi))
        print('drR     = %9.5g' % drR_spl(r_poi))
        print('kappa   = %9.5g' % kappa_spl(r_poi))
        print('s_kappa = %9.5g' % (kappa_spl.derivatives(r_poi)[1] \
                                   * r_poi / kappa_spl(r_poi)))
        print('delta   = %9.5g' % delta_spl(r_poi))
        print('s_delta = %9.5g' % (delta_spl.derivatives(r_poi)[1] \
                                   * r_poi / np.sqrt(1 - delta_spl(r_poi)**2)))
        print('zeta    = %9.5g' % zeta_spl(r_poi))
        print('s_zeta  = %9.5g' % (zeta_spl.derivatives(r_poi)[1] * r_poi))
        print('minor_r = %9.5g' % (1.0))
        print('major_R = %9.5g' % (R0_poi / r_poi * r_ov_a))
        print('\nAdditional information:')
        print('Lref        = %9.5g !for Lref=a convention' % self.a_lcfs)
        print('Bref        = %9.5g' % Bref_poi)


if __name__ == '__main__':
    eq = Miller(plot=1)
    eq(0.75)
