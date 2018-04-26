#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:32:25 2013

@author: dtold
"""
from __future__ import print_function
from __future__ import division

from builtins import range
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

    def __init__(self, gfile='pegasus-eq21.geqdsk', plot=True, rova=0.75, psinorm=None):
        self.gfile = gfile
        self.plot = plot
        self.rova = rova
        # psinorm input overrides rova
        if psinorm:
            self.psinorm = psinorm
            self.rova = None

        self.ntheta = 150
        self.nw = None
        self.io = 3  # interpolation order
        self.s = 1.e-5

        self.process_gfile()
        self.plot_gfile()
        self.calc_miller()
        self.print_miller()
        self.plot_miller()

    def __call__(self, rova=0.75, psinorm=None):
        self.rova = rova
        # psinorm input overrides rova
        if psinorm:
            self.psinorm = psinorm
            self.rova = None

        self.calc_miller()
        self.print_miller()
        self.plot_miller()

    def process_gfile(self):

        # read geqdsk file
        with open(self.gfile, 'r') as gfile:
            eqdsk = gfile.readlines()

        # parse line 0
        self.nw = int(eqdsk[0].split()[-2])
        self.nh = int(eqdsk[0].split()[-1])
        print('Header: %s' % eqdsk[0])
        print('Resolution: %d x %d' % (self.nw, self.nh))

        # parse line 1
        try:
            entrylength = 16
            rdim, zdim, _, rmin, zmid = \
                [float(eqdsk[1][j * entrylength:(j + 1) * entrylength])
                 for j in range(len(eqdsk[1]) // entrylength)]
        except:
            try:
                entrylength = 15
                rdim, zdim, _, rmin, zmid = \
                    [float(eqdsk[1][j * entrylength:(j + 1) * entrylength])
                        for j in range(len(eqdsk[1]) // entrylength)]
            except:
                raise IOError('Failed to read G-EQDSK line')

        # parse line 2
        self.Rmaxis, self.Zmaxis, self.psiax, self.psisep, _ = \
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

        if self.Rmaxis != rmag2:
            raise ValueError('Inconsistent self.Rmaxis: %7.4g, %7.4g' % (self.Rmaxis, rmag2))
        if self.psiax != psiax2:
            raise ValueError('Inconsistent psiax: %7.4g, %7.4g' % (self.psiax, psiax2))
        if self.Zmaxis != zmag2:
            raise ValueError('Inconsistent self.Zmaxis: %7.4g, %7.4g' % (self.Zmaxis, zmag2))
        if self.psisep != psisep2:
            raise ValueError('Inconsistent psisep: %7.4g, %7.4g' % (self.psisep, psisep2))

        # read flux profiles and 2D flux grid
        # pol current (F=RBt) [T-m] on uniform flux grid
        self.F_ff = np.empty(self.nw)
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

        # pressure [Pa] on uniform flux grid
        self.p_ff = np.empty(self.nw)
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.p_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        # FF'=FdF/dpsi on uniform flux grid
        self.ffprime_ff = np.empty(self.nw)
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.ffprime_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        # dp/dpsi [Pa/(Wb/rad)] on uniform flux grid
        self.pprime_ff = np.empty(self.nw)
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.pprime_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        # pol. flux [Wb/rad] on rectangular grid
        psirz_1d = np.empty(self.nw * self.nh)
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

        # q safety factor on uniform flux grid
        self.qpsi_ff = np.empty(self.nw)
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.qpsi_ff[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        start_line = i + start_line + 1

        # flip signs if psi-axis > psi-separatrix
        if self.psiax > self.psisep:
            self.psirz = -self.psirz
            self.ffprime_ff = -self.ffprime_ff
            self.pprime_ff = -self.pprime_ff
            self.psiax *= -1
            self.psisep *= -1

        #  R,Z grids
        dw = rdim / (self.nw - 1)
        dh = zdim / (self.nh - 1)
        self.rgrid = np.array([rmin + i * dw for i in range(self.nw)])
        self.zgrid = np.array([zmid - zdim / 2 + i * dh \
                          for i in range(self.nh)])

        # theta grid
        self.theta_grid = np.linspace(-np.pi, np.pi, self.ntheta)

        # flux grids
        self.psinorm_grid = np.linspace(0, 1, self.nw)
        self.psi_grid = np.linspace(self.psiax, self.psisep, self.nw)

        # flux surface R/Z coords. on flux and theta grids
        self.R_ftgrid = np.empty((self.nw, self.ntheta))
        self.Z_ftgrid = np.empty((self.nw, self.ntheta))
        t1 = np.arctan2(zmid - zdim / 2 - self.Zmaxis,
                        rmin - self.Rmaxis) # angle(mag axis to bot. left)
        t2 = np.arctan2(zmid - zdim / 2 - self.Zmaxis,
                        rmin + rdim - self.Rmaxis) # angle (mag ax to bot. rt)
        t3 = np.arctan2(zmid + zdim / 2 - self.Zmaxis,
                        rmin + rdim - self.Rmaxis) # angle (mag ax to top rt)
        t4 = np.arctan2(zmid + zdim / 2 - self.Zmaxis,
                        rmin - self.Rmaxis) # angle (mag ax to top left)
        # spline object for psi on RZ grid
        self.psi_spl = RBS(self.zgrid, self.rgrid, self.psirz,
                           kx=self.io,
                           ky=self.io)
        psilimit = self.psisep + (self.psisep - self.psiax) * 0.05
        for j, theta in enumerate(self.theta_grid):
            if theta < t1 or theta >= t4:
                rad = (rmin - self.Rmaxis) / np.cos(theta)
            elif theta < t2 and theta >= t1:
                rad = -(self.Zmaxis - zmid + zdim / 2) / np.sin(theta)
            elif theta < t3 and theta >= t2:
                rad = (rmin + rdim - self.Rmaxis) / np.cos(theta)
            elif theta < t4 and theta >= t3:
                rad = (zmid + zdim / 2 - self.Zmaxis) / np.sin(theta)
            else:
                raise ValueError('Error with theta angle')
            dr = rad / (self.nw - 1) * np.cos(theta)
            dz = rad / (self.nw - 1) * np.sin(theta)
            # RZ coordinates from axis at fixed poloidal angle
            r_pol = np.array([self.Rmaxis + i * dr for i in range(self.nw)])
            z_pol = np.array([self.Zmaxis + i * dz for i in range(self.nw)])
            psi_rad = self.psi_spl.ev(z_pol, r_pol)
            psi_rad[0] = self.psiax
            # must restrict interpolation range because of non-monotonic psi around coils
            end_ind = 0
            for i in range(self.nw - 1):
                if psi_rad[i] > psilimit:
                    break
                if psi_rad[i + 1] <= psi_rad[i] and i < self.nw - 2:
                    psi_rad[i + 1] = 0.5 * (psi_rad[i] + psi_rad[i + 2])
                end_ind += 1

            # interp objects for indices
            psi_int = interp1d(psi_rad[:end_ind + 1],
                               np.arange(end_ind + 1),
                               kind=self.io)
            # near psi-grid index for separatrix
            indsep = int(psi_int(self.psisep)) + 3
            # RZ interp. objects along poloidal line from axis
            R_int = interp1d(psi_rad[:indsep],
                             r_pol[:indsep],
                             kind=self.io)
            Z_int = interp1d(psi_rad[:indsep],
                             z_pol[:indsep],
                             kind=self.io)
            # RZ coords of FS grid at fixed theta
            self.R_ftgrid[:, j] = R_int(self.psi_grid)
            self.Z_ftgrid[:, j] = Z_int(self.psi_grid)

        # find average elevation for all flux surfaces
        self.Z_avg_fs = np.empty(self.nw)
        for i in range(self.nw):
            ds = np.empty(self.ntheta)
            ds[1:self.ntheta - 1] = 0.5 * np.sqrt((self.R_ftgrid[i, 2:self.ntheta] \
                - self.R_ftgrid[i, 0:self.ntheta - 2])**2 \
                + (self.Z_ftgrid[i, 2:self.ntheta] \
                - self.Z_ftgrid[i, 0:self.ntheta - 2])**2)
            ds[0] = 0.5 * np.sqrt((self.R_ftgrid[i, 1] - self.R_ftgrid[i, -1]) ** 2 \
                                   + (self.Z_ftgrid[i, 1] - self.Z_ftgrid[i, -1])**2)
            ds[-1] = 0.5 * np.sqrt((self.R_ftgrid[i, 0] - self.R_ftgrid[i, -2]) ** 2 \
                                   + (self.Z_ftgrid[i, 0] - self.Z_ftgrid[i, -2])**2)
            self.Z_avg_fs[i] = np.average(self.Z_ftgrid[i, :], weights=ds)

        # R_major and r_minor for all flux surfaces
        self.R_major_ff = np.empty(self.nw)
        self.R_major_ff[0] = self.Rmaxis
        self.r_minor_ff = np.empty(self.nw)
        self.r_minor_ff[0] = 0.
        itheta = self.ntheta // 4
        # loop over flux grid
        for i in range(1, self.nw):
            # low field side
            R_array = self.R_ftgrid[i, itheta:3*itheta]
            Z_array = self.Z_ftgrid[i, itheta:3*itheta]
            Z_int = interp1d(Z_array, R_array, kind=self.io)
            R_out = Z_int(self.Z_avg_fs[i])
            # high field side
            R_array = np.roll(self.R_ftgrid[i, :-1], self.ntheta // 2)[itheta:3*itheta]
            Z_array = np.roll(self.Z_ftgrid[i, :-1], self.ntheta // 2)[itheta:3*itheta]
            # have to use negative Z_array here to have increasing order
            Z_int = interp1d(-Z_array, R_array, kind=self.io)
            R_in = Z_int(-self.Z_avg_fs[i])
            self.R_major_ff[i] = 0.5 * (R_out + R_in)  # R_maj at self.Z_avg_fs
            self.r_minor_ff[i] = 0.5 * (R_out - R_in)  # r_min at self.Z_avg_fs

        self.Rmaxis_lcfs = self.R_major_ff[-1]
        self.a_lcfs = self.r_minor_ff[-1]
        self.eps_lcfs = self.a_lcfs / self.Rmaxis_lcfs

        print('\n*** Magnetic axis and LCFS ***')
        print('R mag. axis = {:.3g} m'.format(self.Rmaxis))
        print('Z mag. axis = {:.3g} m'.format(self.Zmaxis))
        print('psi-axis = {:.3e} Wb/rad'.format(self.psiax))
        print('psi-sep = {:.3e} Wb/rad'.format(self.psisep))
        print('R0_lcfs = {:.3g} m'.format(self.Rmaxis_lcfs))
        print('a_lcfs = {:.3g} m'.format(self.a_lcfs))
        print('eps_lcfs = {:.3g}'.format(self.eps_lcfs))


    def plot_gfile(self):
        if not self.plot:
            return

        # plot profile quantities
        plt.figure(figsize=(6,6))
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

        # plot psirz on r/z grid
        plt.figure(figsize=(4,6))
        plt.contour(self.rgrid, self.zgrid, self.psirz, 70)
        plt.title('$\Psi$ contours')
        plt.xlabel(r'$R$ [m]', fontsize=14)
        plt.ylabel(r'$Z$ [m]', fontsize=14)
        plt.gca().set_aspect('equal')


    def calc_miller(self):

        # splines on r_minor space
        psi_grid_spl = US(self.r_minor_ff,
                          self.psi_grid,
                          k=self.io,
                          s=self.s)
        q_spl = US(self.r_minor_ff,
                   self.qpsi_ff,
                   k=self.io,
                   s=self.s)
        R0_spl = US(self.r_minor_ff,
                    self.R_major_ff,
                    k=self.io,
                    s=self.s)
        F_spl = US(self.r_minor_ff,
                   self.F_ff,
                   k=self.io,
                   s=self.s)
        p_spl = US(self.r_minor_ff,
                   self.p_ff,
                   k=self.io,
                   s=self.s)
        pprime_spl = US(self.r_minor_ff,
                        self.pprime_ff,
                        k=self.io,
                        s=1e-4)
        # splines on psi space
        q_spl_psi = US(self.psi_grid,
                       self.qpsi_ff,
                       k=self.io,
                       s=self.s)
        r_min_spl = US(self.psi_grid,
                       self.r_minor_ff,
                       k=self.io,
                       s=self.s)
        # position values
        if self.rova:
            # calc psi, psinorm, and r from r/a
            r_poi = self.rova * self.a_lcfs  # r = r/a * a; FS minor radius
            psi_poi = float(psi_grid_spl(r_poi))  # psi at FS
            self.psinorm = (psi_poi - self.psiax) / (self.psisep - self.psiax)  # psi-norm at FS
        else:
            # calc psi, r, r/a from psi-norm
            psi_poi = self.psinorm * (self.psisep - self.psiax) + self.psiax
            r_poi = float(r_min_spl(psi_poi))
            self.rova = r_poi / self.a_lcfs
        R0_poi = float(R0_spl(r_poi))  # R_maj of FS
        F_poi = float(F_spl(r_poi))  # F of FS
        p_poi = float(p_spl(r_poi))
        pprime_poi = float(pprime_spl(r_poi))
        Bref_poi = F_poi / R0_poi
        q_poi = float(q_spl_psi(psi_poi))
        drdpsi_poi = float(r_min_spl.derivatives(psi_poi)[1])
        omp_poi = -float((self.a_lcfs / p_poi) * (pprime_poi / drdpsi_poi))

        sgstart = self.nw // 10
        subgrid = np.arange(sgstart, self.nw)
        nsg = subgrid.size


        # calc kappa and delta
        R_tm = np.empty((nsg, self.ntheta))
        Z_tm = np.empty((nsg, self.ntheta))
        theta_tmp = np.linspace(-2. * np.pi, 2 * np.pi, 2 * self.ntheta - 1)
        for isg,i in enumerate(subgrid):
            R_extended = np.empty(2 * self.ntheta - 1)
            Z_extended = np.empty(2 * self.ntheta - 1)
            R_extended[0:(self.ntheta - 1) // 2] = self.R_ftgrid[i, (self.ntheta + 1) // 2:-1]
            R_extended[(self.ntheta - 1) // 2:(3 * self.ntheta - 3) // 2] = self.R_ftgrid[i, :-1]
            R_extended[(3 * self.ntheta - 3) // 2:] = self.R_ftgrid[i, 0:(self.ntheta + 3) // 2]
            Z_extended[0:(self.ntheta - 1) // 2] = self.Z_ftgrid[i, (self.ntheta + 1) // 2:-1]
            Z_extended[(self.ntheta - 1) // 2:(3 * self.ntheta - 3) // 2] = self.Z_ftgrid[i, :-1]
            Z_extended[(3 * self.ntheta - 3) // 2:] = self.Z_ftgrid[i, 0:(self.ntheta + 3) // 2]
            theta_mod_ext = np.arctan2(Z_extended - self.Z_avg_fs[i],
                                       R_extended - self.R_major_ff[i])
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
                                 kind=self.io)
            R_int = interp1d(theta_mod_ext, R_extended,
                             kind=self.io)
            Z_int = interp1d(theta_mod_ext, Z_extended,
                             kind=self.io)
            R_tm[isg,:] = R_int(self.theta_grid)
            Z_tm[isg,:] = Z_int(self.theta_grid)
        R_sym = np.empty((nsg, self.ntheta))
        Z_sym = np.empty((nsg, self.ntheta))
        for isg,i in enumerate(subgrid):
            Z_sym[isg, :] = 0.5 * (Z_tm[isg, :] - Z_tm[isg, ::-1]) \
                + self.Z_avg_fs[i]
            R_sym[isg, :] = 0.5 * (R_tm[isg, :] + R_tm[isg, ::-1])
            if isg<=3: print(isg, Z_sym[isg,50], Z_tm[isg,50])
        kappa = np.empty(nsg)
        delta_upper = np.empty(nsg)
        delta_lower = np.empty(nsg)
        stencil_width = self.ntheta // 10
        for isg,i in enumerate(subgrid):
            for o in range(2):
                if o:
                    ind = np.argmax(Z_sym[isg,:])
                    section = np.arange(ind + stencil_width // 2,
                                        ind - stencil_width // 2, -1)
                else:
                    ind = np.argmin(Z_sym[isg,:])
                    section = np.arange(ind - stencil_width // 2,
                                        ind + stencil_width // 2)
                x = R_sym[isg, section]
                y = Z_sym[isg, section]
                if o==0 and isg<=3: print(isg, y[0], y[-1])
                y_int = interp1d(x, y, kind=self.io)
                x_fine = np.linspace(
                    np.amin(x), np.amax(x), stencil_width * 100)
                y_fine = y_int(x_fine)
                if o:
                    x_at_extremum = x_fine[np.argmax(y_fine)]
                    delta_upper[isg] = (
                        self.R_major_ff[i] - x_at_extremum) / self.r_minor_ff[i]
                    Z_max = np.amax(y_fine)
                else:
                    x_at_extremum = x_fine[np.argmin(y_fine)]
                    delta_lower[isg] = (
                        self.R_major_ff[i] - x_at_extremum) / self.r_minor_ff[i]
                    Z_min = np.amin(y_fine)
            kappa[isg] = (Z_max - Z_min) / 2. / self.r_minor_ff[i]
        delta = 0.5 * (delta_upper + delta_lower)


        # calc zeta
        zeta = np.empty(nsg)
        for isg,i in enumerate(subgrid):
            zeta_arr = np.empty(4)
            x = np.arcsin(delta[isg])
            for o in range(4):
                if o == 0:
                    val = np.pi / 4
                    searchval = np.cos(val + x / np.sqrt(2))
                    searcharr = (R_sym[isg,:] - self.R_major_ff[i]) / self.r_minor_ff[i]
                elif o == 1:
                    val = 3 * np.pi / 4
                    searchval = np.cos(val + x / np.sqrt(2))
                    searcharr = (R_sym[isg,:] - self.R_major_ff[i]) / self.r_minor_ff[i]
                elif o == 2:
                    val = -np.pi / 4
                    searchval = np.cos(val - x / np.sqrt(2))
                    searcharr = (R_sym[isg,:] - self.R_major_ff[i]) / self.r_minor_ff[i]
                elif o == 3:
                    val = -3 * np.pi / 4
                    searchval = np.cos(val - x / np.sqrt(2))
                    searcharr = (R_sym[isg,:] - self.R_major_ff[i]) / self.r_minor_ff[i]
                else:
                    raise ValueError('out of range')
                if o in [0, 1]:
                    searcharr2 = searcharr[self.ntheta // 2:]
                    ind = find(searchval, searcharr2) + self.ntheta // 2
                else:
                    searcharr2 = searcharr[0:self.ntheta // 2]
                    ind = find(searchval, searcharr2)
                section = np.arange(ind - stencil_width // 2,
                                    ind + stencil_width // 2)
                theta_sec = self.theta_grid[section]
                if o in [0, 1]:
                    theta_int = interp1d(-searcharr[section], theta_sec, kind=self.io)
                    theta_of_interest = theta_int(-searchval)
                else:
                    theta_int = interp1d(searcharr[section], theta_sec, kind=self.io)
                    theta_of_interest = theta_int(searchval)
                Z_sec = Z_sym[isg, section]
                Z_sec_int = interp1d(theta_sec, Z_sec, kind=self.io)
                Z_val = Z_sec_int(theta_of_interest)
                zeta_arg = (Z_val - self.Z_avg_fs[i]) / kappa[isg] / self.r_minor_ff[i]
                if o==0 and isg<=3:
                    print(isg, Z_sym[isg,section[0]], Z_sym[isg,section[-1]])
                if abs(zeta_arg)>=1:
                    zeta_arg = 0.999999*np.sign(zeta_arg)
                zeta_arr[o] = np.arcsin(zeta_arg)
            zeta_arr[1] = np.pi - zeta_arr[1]
            zeta_arr[3] = -np.pi - zeta_arr[3]
            zeta[isg] = 0.25 * (np.pi + zeta_arr[0] - zeta_arr[1]
                                - zeta_arr[2] + zeta_arr[3])


        # calc dr/dR, amhd, and derivs
        drR = np.empty(nsg)
        amhd = np.empty(nsg)
        s_kappa = np.empty(nsg)
        s_delta = np.empty(nsg)
        s_zeta = np.empty(nsg)
        bp = np.empty(nsg)
        bt = np.empty(nsg)
        b = np.empty(nsg)
        kappa_spl = US(self.r_minor_ff[sgstart:],
                       kappa, k=self.io, s=self.s)
        delta_spl = US(self.r_minor_ff[sgstart:],
                       delta, k=self.io, s=self.s)
        zeta_spl = US(self.r_minor_ff[sgstart:],
                      zeta, k=self.io, s=self.s)
        for isg,i in enumerate(subgrid):
            amhd[isg] = -self.qpsi_ff[i]**2 * self.R_major_ff[i] * self.pprime_ff[i] * \
                8 * np.pi * 1e-7 / Bref_poi**2 / \
                r_min_spl.derivatives(self.psi_grid[i])[1]
            drR[isg] = R0_spl.derivatives(self.r_minor_ff[i])[1]
            s_kappa[isg] = kappa_spl.derivatives(self.r_minor_ff[i])[1] \
                                 * self.r_minor_ff[i] / kappa[isg]
            s_delta[isg] = delta_spl.derivatives(self.r_minor_ff[i])[1] \
                                    * self.r_minor_ff[i] / np.sqrt(1 - delta[isg]**2)
            s_zeta[isg] = zeta_spl.derivatives(self.r_minor_ff[i])[1] \
                                * self.r_minor_ff[i]
            R = self.R_major_ff[i] + self.r_minor_ff[i]
            Z = self.Z_avg_fs[i]
            Br = -self.psi_spl(Z, R, dx=1, dy=0)/R
            Bz = self.psi_spl(Z, R, dx=0, dy=1)/R
            bp[isg] = np.sqrt(Br**2+Bz**2)
            bt[isg] = self.F_ff[i]/R
            b[isg] = np.sqrt(bp[isg]**2+bt[isg]**2)
        amhd_spl = US(self.r_minor_ff[sgstart:],
                      amhd,
                      k=self.io,
                      s=1e-3)
        drR_spl = US(self.r_minor_ff[sgstart:],
                     drR,
                     k=self.io,
                     s=self.s)
        b_spl = US(self.psinorm_grid[sgstart:],
                     b,
                     k=self.io,
                     s=self.s)


        print('\n*** FS at r/a = {:.2f} ***'.format(self.rova))
        print('r_min = {:.3f} m'.format(r_poi))
        print('R_maj = {:.3f} m'.format(R0_poi))
        print('eps = {:.3f}'.format(r_poi / R0_poi))
        print('q = {:.3f}'.format(q_poi))
        print('psi = {:.3e} Wb/rad'.format(psi_poi))
        print('psi_N = {:.3f}'.format(self.psinorm))
        print('p = {:.3g} Pa'.format(p_poi))
        print('dp/dpsi = {:.3g} Pa/(Wb/rad)'.format(pprime_poi))
        print('dr/dpsi = {:.3g} m/(Wb/rad)'.format(drdpsi_poi))
        print('omp = {:.4g} (with Lref=a)'.format(omp_poi))

        print('\n\nShaping parameters for flux surface r=%9.5g, r/a=%9.5g:' %
              (r_poi, self.rova))
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
        print('major_R = %9.5g' % (R0_poi / r_poi * self.rova))
        print('\nAdditional information:')
        print('Lref        = %9.5g !for Lref=a convention' % self.a_lcfs)
        print('Bref        = %9.5g' % Bref_poi)





        if self.plot:
            plt.figure(figsize=(6,8))
            plt.subplot(4, 2, 1)
            plt.plot(self.psinorm_grid[sgstart:], kappa)
            plt.title('Elongation')
            plt.ylabel(r'$\kappa$', fontsize=14)

            plt.subplot(4, 2, 2)
            plt.plot(self.psinorm_grid[sgstart:], s_kappa)
            plt.title(r'$r/\kappa*(d\kappa/dr)$')
            plt.ylabel(r'$s_\kappa$', fontsize=14)

            plt.subplot(4, 2, 3)
            plt.plot(self.psinorm_grid[sgstart:], delta)
            plt.title('Triangularity')
            plt.ylabel(r'$\delta$', fontsize=14)

            plt.subplot(4, 2, 4)
            plt.plot(self.psinorm_grid[sgstart:], s_delta)
            plt.title(r'$r/\delta*(d\delta/dr)$')
            plt.ylabel(r'$s_\delta$', fontsize=14)

            plt.subplot(4, 2, 5)
            plt.plot(self.psinorm_grid[sgstart:], zeta)
            plt.title('Squareness')
            plt.ylabel(r'$\zeta$', fontsize=14)

            plt.subplot(4, 2, 6)
            plt.plot(self.psinorm_grid[sgstart:], s_zeta)
            plt.title(r'$r/\zeta*(d\zeta/dr)$')
            plt.ylabel(r'$s_\zeta$', fontsize=14)

            plt.subplot(4, 2, 7)
            plt.plot(self.psinorm_grid[sgstart:], b)
            plt.plot(self.psinorm_grid[sgstart:], bp)
            plt.plot(self.psinorm_grid[sgstart:], bt)
            plt.title('|B|')
            plt.ylabel(r'|B|', fontsize=14)

            plt.subplot(4, 2, 8)
            plt.plot(self.psinorm_grid[sgstart:],
                     b_spl(self.psinorm_grid[sgstart:], nu=1))
            plt.title('|B| deriv.')
            plt.ylabel(r'$d|B|/d\Psi_N$', fontsize=14)

            for ax in plt.gcf().axes:
                ax.set_xlabel(r'$\Psi_N$', fontsize=14)
                ax.axvline(self.psinorm, 0, 1, ls='--', color='k', lw=2)
            plt.tight_layout(pad=0.3)


    def print_miller(self):
        pass

    def plot_miller(self):
        pass


if __name__ == '__main__':
    eq = Miller(plot=1)