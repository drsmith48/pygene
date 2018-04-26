#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:32:25 2013

@author: dtold

July 2017 - Refactored (David R. Smith)
    "Futurized" for python 2/3 compatability
    Created Geqdsk class to process geqdsk files and
    extract Miller quantities
"""
from __future__ import print_function
from __future__ import division

import os
from builtins import range
import tkinter.filedialog as fd
import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as US


def get_filename():
    root = tk.Tk()
    root.withdraw()
    initialdir = os.path.join(os.environ['GENETOP'], 'eqdsk')
    filename = fd.askopenfilename(initialdir=initialdir,
                                  title='Select GEQDSK file')
    #filename = 'pegasus-eq21.geqdsk'
    return filename


class Geqdsk(object):
    """Class for geqdsk files.

    Args:
        gfile (str, default None): filename for geqdsk file;
            will present file dialog if evaluates to false
        plot (bool, default True): plot profiles

    Methods:
        load_gfile():  Select new geqdsk file from dialog, then load
        miller(...):   Calculate Miller quantities
        __call__(...): Alias for miller(...)

    Return:
        Geqdsk object
    """

    def __init__(self, gfile='', plot=False, quiet=False, save=False):
        """See class docstring."""
        self.plot = plot
        self.save = save
        self.quiet = quiet
        self.gfile = gfile
        if not self.gfile:
            self.gfile = get_filename()

        self.ntheta = 150 # number of theta grid points
        self.degree = 3  # spline degree (eg, 3 = cubic)
        self.ti = 0.1 # ion temp in keV

        self._process_gfile()

    def __call__(self, *args, **kwargs):
        """Alias for miller()."""
        return self.miller(*args, **kwargs)

    def _find(self, val, arr):
        ind = 0
        mindiff = 1000.
        for i,arr_val in enumerate(arr):
            diff = abs(arr_val - val)
            if diff < mindiff:
                ind = i
                mindiff = diff
        return ind

    def load_gfile(self):
        self.gfile = get_filename()
        self._process_gfile()

    def _process_gfile(self):

        self.folder, file = os.path.split(self.gfile)
        self.basename,_ = os.path.splitext(file)

        # read geqdsk file
        with open(self.gfile, 'r') as gfile:
            eqdsk = gfile.readlines()
        
        # scalar quantities
        self.nw = None
        self.nh = None
        self.Rmaxis = None
        self.Zmaxis = None
        self.psiax = None
        self.psisep = None

        ### parse lines 0-4 for scalar quantities
        self.nw = int(eqdsk[0].split()[-2])
        self.nh = int(eqdsk[0].split()[-1])
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
        self.Rmaxis, self.Zmaxis, self.psiax, self.psisep, _ = \
            [float(eqdsk[2][j * entrylength:(j + 1) * entrylength])
                for j in range(len(eqdsk[2]) // entrylength)]
        _, psiax2, _, rmag2, _ = \
            [float(eqdsk[3][j * entrylength:(j + 1) * entrylength])
                for j in range(len(eqdsk[3]) // entrylength)]
        zmag2, _, psisep2, _, _ = \
            [float(eqdsk[4][j * entrylength:(j + 1) * entrylength])
                for j in range(len(eqdsk[4]) // entrylength)]
            
        # validate scalar quantities
        if self.Rmaxis != rmag2:
            raise ValueError('Inconsistent self.Rmaxis: %7.4g, %7.4g' % (self.Rmaxis, rmag2))
        if self.psiax != psiax2:
            raise ValueError('Inconsistent psiax: %7.4g, %7.4g' % (self.psiax, psiax2))
        if self.Zmaxis != zmag2:
            raise ValueError('Inconsistent self.Zmaxis: %7.4g, %7.4g' % (self.Zmaxis, zmag2))
        if self.psisep != psisep2:
            raise ValueError('Inconsistent psisep: %7.4g, %7.4g' % (self.psisep, psisep2))

        # 1D profiles on uniform flux grid (psiax to psisep or 0 to 1)
        self.F_fs = np.empty(self.nw)
        self.p_fs = np.empty(self.nw)
        self.ffprime_fs = np.empty(self.nw)
        self.pprime_fs = np.empty(self.nw)
        self.qpsi_fs = np.empty(self.nw)
        # Flux function on 2D RZ grid
        self.psirz = np.empty((self.nh, self.nw))
        
        # layout of lines in geqdsk file
        if self.nw % 5 == 0:
            lines = np.arange(self.nw // 5)
        else:
            lines = np.arange(self.nw // 5 + 1)
        if self.nw * self.nh % 5 == 0:
            lines_twod = np.arange(self.nw * self.nh // 5)
        else:
            lines_twod = np.arange(self.nw * self.nh // 5 + 1)
            
        # pol current (F=RBt) [T-m] on uniform flux grid
        start_line = 5
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.F_fs[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]

        # pressure [Pa] on uniform flux grid
        start_line = i + start_line + 1
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.p_fs[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]

        # FF'=FdF/dpsi on uniform flux grid
        start_line = i + start_line + 1
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.ffprime_fs[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]

        # dp/dpsi [Pa/(Wb/rad)] on uniform flux grid
        start_line = i + start_line + 1
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.pprime_fs[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]

        # pol. flux [Wb/rad] on rectangular grid
        start_line = i + start_line + 1
        psirz_1d = np.empty(self.nw * self.nh)
        for i in lines_twod:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            psirz_1d[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]
        self.psirz = psirz_1d.reshape(self.nh, self.nw)

        # q safety factor on uniform flux grid
        start_line = i + start_line + 1
        for i in lines:
            n_entries = len(eqdsk[i + start_line]) // entrylength
            self.qpsi_fs[i * 5:i * 5 + n_entries] = \
                [float(eqdsk[i + start_line][j * entrylength:(j + 1) * entrylength])
                    for j in range(n_entries)]

        # flip signs if psi-axis > psi-separatrix
        if self.psiax > self.psisep:
            self.psirz = -self.psirz
            self.ffprime_fs = -self.ffprime_fs
            self.pprime_fs = -self.pprime_fs
            self.psiax *= -1
            self.psisep *= -1

        #  construct R,Z grids
        dw = rdim / (self.nw - 1)
        dh = zdim / (self.nh - 1)
        self.rgrid = np.array([rmin + i * dw for i in range(self.nw)])
        self.zgrid = np.array([zmid - zdim / 2 + i * dh \
                          for i in range(self.nh)])

        # theta grid
        self.theta_grid = np.linspace(-np.pi, np.pi, self.ntheta)

        # uniform flux grids
        self.psinorm_grid = np.linspace(0, 1, self.nw)
        self.psi_grid = np.linspace(self.psiax, self.psisep, self.nw)

        # calc. R/Z values on flux-theta grid
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
        self.psi_spl = RBS(self.zgrid, self.rgrid, self.psirz,
                           kx=self.degree, ky=self.degree)
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
            psi_int = interp1d(psi_rad[:end_ind + 1],
                               np.arange(end_ind + 1),
                               kind=self.degree)
            isep = int(psi_int(self.psisep)) + 3
            R_int = interp1d(psi_rad[:isep], r_pol[:isep], kind=self.degree)
            Z_int = interp1d(psi_rad[:isep], z_pol[:isep], kind=self.degree)
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

        # R_major and r_minor on uniform flux grids
        self.R_major_fs = np.empty(self.nw)
        self.r_minor_fs = np.empty(self.nw)
        self.R_major_fs[0] = self.Rmaxis
        self.r_minor_fs[0] = 0.
        itheta = self.ntheta // 4
        for i in range(1, self.nw):
            # low field side
            R_array = self.R_ftgrid[i, itheta:3*itheta]
            Z_array = self.Z_ftgrid[i, itheta:3*itheta]
            Z_int = interp1d(Z_array, R_array, kind=self.degree)
            zavg = np.min([np.max([self.Z_avg_fs[i],np.min(Z_array)]),np.max(Z_array)])
            #print(i, zavg, np.min(Z_array), np.max(Z_array))
            R_out = Z_int(zavg)
            # high field side
            R_array = np.roll(self.R_ftgrid[i, :-1], self.ntheta // 2)[itheta:3*itheta]
            Z_array = np.roll(self.Z_ftgrid[i, :-1], self.ntheta // 2)[itheta:3*itheta]
            # have to use negative Z_array here to have increasing order
            Z_int = interp1d(-Z_array, R_array, kind=self.degree)
            zavg = np.min([np.max([-self.Z_avg_fs[i],np.min(Z_array)]),np.max(Z_array)])
            R_in = Z_int(zavg)
            self.R_major_fs[i] = 0.5 * (R_out + R_in)  # R_maj at self.Z_avg_fs
            self.r_minor_fs[i] = 0.5 * (R_out - R_in)  # r_min at self.Z_avg_fs
            if self.R_major_fs[i]<0:
                self.R_major_fs[i] = self.R_major_fs[i-1]
            if self.r_minor_fs[i]<0 or self.r_minor_fs[i]>self.R_major_fs[i]:
                self.r_minor_fs[i] = self.r_minor_fs[i-1]+1e-3

        if not self.quiet:
            print('Header: %s' % eqdsk[0])
            print('Resolution: %d x %d' % (self.nw, self.nh))
            print('gfile: {}'.format(self.gfile))
            print('\n*** Magnetic axis and LCFS ***')
            print('R mag. axis = {:.3g} m'.format(self.Rmaxis))
            print('Z mag. axis = {:.3g} m'.format(self.Zmaxis))
            print('psi-axis = {:.3e} Wb/rad'.format(self.psiax))
            print('psi-sep = {:.3e} Wb/rad'.format(self.psisep))
            print('R0_lcfs = {:.3g} m'.format(self.R_major_fs[-1]))
            print('a_lcfs = {:.3g} m'.format(self.r_minor_fs[-1]))
            
        if self.plot:
            self.plot_gfile()


    def plot_gfile(self):
        # plot profile quantities
        plt.figure(figsize=(6,6))
        plt.subplot(3, 2, 1)
        plt.plot(self.psinorm_grid, self.F_fs)
        plt.title(r'$F=R\,B_t$')
        plt.xlabel(r'$\Psi_N$', fontsize=14)
        plt.ylabel(r'$F$ [m-T]', fontsize=14)

        plt.subplot(3, 2, 2)
        plt.plot(self.psinorm_grid, self.ffprime_fs)
        plt.title(r"$F*dF/d\Psi$")
        plt.xlabel(r'$\Psi_N$', fontsize=14)
        plt.ylabel(r"$FF'(\Psi)$", fontsize=14)

        plt.subplot(3, 2, 3)
        plt.plot(self.psinorm_grid, self.p_fs/1e3)
        plt.title('Pressure')
        plt.xlabel(r'$\Psi_N$', fontsize=14)
        plt.ylabel(r'$p$ [kPa]', fontsize=14)

        plt.subplot(3, 2, 4)
        plt.plot(self.psinorm_grid, self.pprime_fs/1e3)
        plt.title(r"$dp/d\Psi$")
        plt.xlabel(r'$\Psi_N$', fontsize=14)
        plt.ylabel(r"$p'(\Psi)$", fontsize=14)

        plt.subplot(3, 2, 5)
        plt.plot(self.psinorm_grid, self.qpsi_fs)
        plt.title('Safety Factor')
        plt.xlabel(r'$\Psi_N$', fontsize=14)
        plt.ylabel(r'$q$', fontsize=14)
        plt.ylim([0,None])
        plt.tight_layout(pad=0.3)

        if self.save:
            filename = os.path.join(self.folder, self.basename+'_gprofiles.pdf')
            plt.savefig(filename)

        # plot psirz on r/z grid
        psirange = self.psisep-self.psiax
        levels = np.asarray([0.02,0.3,0.6,0.995,1.15,
                             2,3,4,5,6,7,8,9])*psirange+self.psiax
        plt.figure(figsize=(4,6))
        plt.contour(self.rgrid, self.zgrid, self.psirz, levels=levels)
        plt.title('$\Psi$ contours')
        plt.xlabel(r'$R$ [m]', fontsize=14)
        plt.ylabel(r'$Z$ [m]', fontsize=14)
        plt.xlim([0,1])
        plt.ylim([-1,1])
        plt.gca().set_aspect('equal')
        if self.save:
            filename = os.path.join(self.folder, self.basename+'_cross-section.pdf')
            plt.savefig(filename)
            

    def miller(self, psinorm=0.8, rova=None, omt_factor=0.2, ref_miller=None):

        output = {}
        self.a_lcfs = self.r_minor_fs[-1]
        self.smooth = 1.e-5
        self.psinorm = psinorm
        self.rova = rova

        # spline for r_minor on uniform flux grid
        r_min_spl = US(self.psi_grid, self.r_minor_fs, k=self.degree, s=self.smooth)

        if self.psinorm:
            print('using `psinorm`, ignoring `rova`')
            psi = self.psinorm * (self.psisep - self.psiax) + self.psiax
            r = r_min_spl(psi)[()]
            self.rova = r / self.a_lcfs
        else:
            print('using `rova`, ignoring `psinorm`')
            r = self.rova * self.a_lcfs  # r = r/a * a; FS minor radius
            psi_grid_spl =  US(self.r_minor_fs, self.psi_grid, k=self.degree, s=self.smooth)
            psi = psi_grid_spl(r)[()]  # psi at FS
            self.psinorm = (psi - self.psiax) / (self.psisep - self.psiax)  # psi-norm at FS

        output['gfile'] = self.gfile
        output['ref_miller'] = ref_miller
        output['psinorm'] = self.psinorm
        output['rova'] = self.rova

        R0_spl = US(self.r_minor_fs, self.R_major_fs, k=self.degree, s=self.smooth)
        R0 = R0_spl(r)[()]  # R_maj of FS
        q_spl = US(self.r_minor_fs, self.qpsi_fs,    k=self.degree, s=self.smooth)
        #q_spl_psi = US(self.psi_grid,   self.qpsi_fs,    k=self.degree, s=self.smooth)
        q = q_spl(r)[()]
        drdpsi = r_min_spl.derivatives(psi)[1]
        eps = r / R0
        if not self.quiet:
            print('\n*** Flux surfance ***')
            print('r_min/a = {:.3f}'.format(self.rova))
            print('psinorm = {:.3f}'.format(self.psinorm))
            print('r_min = {:.3f} m'.format(r))
            print('R_maj = {:.3f} m'.format(R0))
            print('eps = {:.3f}'.format(eps))
            print('q = {:.3f}'.format(q))
            print('psi = {:.3e} Wb/rad'.format(psi))
            print('dr/dpsi = {:.3g} m/(Wb/rad)'.format(drdpsi))

        F_spl = US(self.r_minor_fs, self.F_fs, k=self.degree, s=self.smooth)
        FFp_spl = US(self.r_minor_fs, self.ffprime_fs, k=self.degree, s=self.smooth)
        F = F_spl(r)[()]  # F of FS
        p_spl = US(self.r_minor_fs, self.p_fs, k=self.degree, s=self.smooth)
        pprime_spl = US(self.r_minor_fs, self.pprime_fs, k=self.degree, s=1e-4)

        FFp = FFp_spl(r)[()]  # FF' of FS
        Bref = F / R0
        if ref_miller:
            print('\n*** applying target pressure ***')
            p_total = ref_miller['p_total']
        else:
            p_total = p_spl(r)[()]
        pprime = pprime_spl(r)[()]
        Tref = self.ti
        nref = (p_total/2) / (Tref*1e3*1.602e-19) / 1e19
        beta = 403.e-5*nref*Tref/(Bref**2)
        lam = 24-np.log(np.sqrt(nref*1e13)/(Tref*1e3))
        coll = 2.3031e-5 * lam * self.a_lcfs * nref / (Tref**2)
        output['Lref'] = self.a_lcfs
        output['Bref'] = Bref
        output['Tref'] = Tref
        output['nref'] = nref
        output['F'] = F
        output['FFp'] = FFp
        output['beta'] = beta
        output['coll'] = coll
        output['p_total'] = p_total
        output['pprime'] = pprime
        if not self.quiet:
            print('\n*** Reference values ***')
            print('Lref = {:.3g} m ! for Lref=a convention'.format(self.a_lcfs))
            print('Bref = {:.3g} T'.format(Bref))
            print('Tref = {:.3g} keV'.format(Tref))
            print('nref = {:.3g} 1e19/m^3'.format(nref))
            print('beta (species) = {:.3g}'.format(beta))
            print('coll = {:.3g}'.format(coll))
            print('pres (total) = {:.3g} Pa'.format(p_total))
            print('pprime = dp/dpsi (total) = {:.3g} Pa/(Wb/rad)'.format(pprime))

        if ref_miller:
            dpdr = ref_miller['dpdr']
        else:
            dpdr = (pprime/2)/drdpsi  # convert total gradient to species gradient
        omp = -(self.a_lcfs / (p_total/2)) * dpdr  # convert total pres to species pres
        dpdx_pm = beta*(2*omp)  # convert species gradient to total gradient
        p_mag = Bref**2/(2*4*np.pi*1.0e-7)
        dpdx_pm_2 = -(2*dpdr)/p_mag  # convert species gradient to total gradient
        output['dpdr'] = dpdr
        output['omp'] = omp
        output['dpdx_pm'] = dpdx_pm
        output['dpdx_pm_2'] = dpdx_pm_2
        if not self.quiet:
            print('\n*** Pressure gradients ***')
            print('dp/dr (species)          = {:.3g} Pa/m'.format(dpdr))
            print('omp = Lref/Lp (species)  = {:.3g} ! with Lref=a'.format(omp))
            print('p_mag = 2mu/Bref**2      = {:.3g} Pa'.format(p_mag))
            print('2*beta*omp (total)       = {:.3g} 1/m'.format(dpdx_pm))
            print('-(2*dp/dr)/p_mag (total) = {:.3g} 1/m'.format(dpdx_pm_2))

        omt = omp * omt_factor
        omn = omp * (1-omt_factor)
        output['omt'] = omt
        output['omt_factor'] = omt_factor
        output['omn'] = omn
        if not self.quiet:
            print('\n*** Temp/dens gradients ***')
            print('omt_factor = {:.3g}'.format(omt_factor))
            print('omt = a/T * dT/dr = {:.3g}'.format(omt))
            print('omn = a/n * dn/dr = {:.3g}'.format(omn))
            


        sgstart = self.nw // 10
        subgrid = np.arange(sgstart, self.nw)
        nsg = subgrid.size

        # calc symmetric R/Z on psi/theta grid
        kappa = np.empty(nsg)
        delta = np.empty(nsg)
        zeta = np.empty(nsg)
        drR = np.empty(nsg)
        amhd = np.empty(nsg)
        bp = np.empty(nsg)
        bt = np.empty(nsg)
        b = np.empty(nsg)
        theta_tmp = np.linspace(-2. * np.pi, 2 * np.pi, 2 * self.ntheta - 1)
        stencil_width = self.ntheta // 10
        for i,isg in enumerate(subgrid):
            R_extended = np.empty(2 * self.ntheta - 1)
            Z_extended = np.empty(2 * self.ntheta - 1)
            R_extended[0:(self.ntheta - 1) // 2] = self.R_ftgrid[isg, (self.ntheta + 1) // 2:-1]
            R_extended[(self.ntheta - 1) // 2:(3 * self.ntheta - 3) // 2] = self.R_ftgrid[isg, :-1]
            R_extended[(3 * self.ntheta - 3) // 2:] = self.R_ftgrid[isg, 0:(self.ntheta + 3) // 2]
            Z_extended[0:(self.ntheta - 1) // 2] = self.Z_ftgrid[isg, (self.ntheta + 1) // 2:-1]
            Z_extended[(self.ntheta - 1) // 2:(3 * self.ntheta - 3) // 2] = self.Z_ftgrid[isg, :-1]
            Z_extended[(3 * self.ntheta - 3) // 2:] = self.Z_ftgrid[isg, 0:(self.ntheta + 3) // 2]
            theta_mod_ext = np.arctan2(Z_extended - self.Z_avg_fs[isg],
                                       R_extended - self.R_major_fs[isg])
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
                                 kind=self.degree)
            R_int = interp1d(theta_mod_ext, R_extended,
                             kind=self.degree)
            Z_int = interp1d(theta_mod_ext, Z_extended,
                             kind=self.degree)
            R_tm = R_int(self.theta_grid)
            Z_tm = Z_int(self.theta_grid)

            Z_sym = 0.5 * (Z_tm[:] - Z_tm[::-1]) + self.Z_avg_fs[isg]
            R_sym = 0.5 * (R_tm[:] + R_tm[::-1])
            delta_ul = np.empty(2)
            for o in range(2):
                if o:
                    ind = np.argmax(Z_sym)
                    section = np.arange(ind + stencil_width // 2,
                                        ind - stencil_width // 2, -1)
                else:
                    ind = np.argmin(Z_sym)
                    section = np.arange(ind - stencil_width // 2,
                                        ind + stencil_width // 2)
                x = R_sym[section]
                y = Z_sym[section]
                y_int = interp1d(x, y, kind=self.degree)
                x_fine = np.linspace(np.amin(x), np.amax(x), stencil_width * 100)
                y_fine = y_int(x_fine)
                if o:
                    x_at_extremum = x_fine[np.argmax(y_fine)]
                    Z_max = np.amax(y_fine)
                else:
                    x_at_extremum = x_fine[np.argmin(y_fine)]
                    Z_min = np.amin(y_fine)
                delta_ul[o] = (self.R_major_fs[isg] - x_at_extremum) \
                    / self.r_minor_fs[isg]
            kappa[i] = (Z_max - Z_min) / 2. / self.r_minor_fs[isg]
            delta[i] = delta_ul.mean()
            # calc zeta
            zeta_arr = np.empty(4)
            for o in range(4):
                if o == 0:
                    val = np.pi / 4
                    searchval = np.cos(val + np.arcsin(delta[i]) / np.sqrt(2))
                    searcharr = (R_sym - self.R_major_fs[isg]) / self.r_minor_fs[isg]
                elif o == 1:
                    val = 3 * np.pi / 4
                    searchval = np.cos(val + np.arcsin(delta[i]) / np.sqrt(2))
                    searcharr = (R_sym - self.R_major_fs[isg]) / self.r_minor_fs[isg]
                elif o == 2:
                    val = -np.pi / 4
                    searchval = np.cos(val - np.arcsin(delta[i]) / np.sqrt(2))
                    searcharr = (R_sym - self.R_major_fs[isg]) / self.r_minor_fs[isg]
                elif o == 3:
                    val = -3 * np.pi / 4
                    searchval = np.cos(val - np.arcsin(delta[i]) / np.sqrt(2))
                    searcharr = (R_sym - self.R_major_fs[isg]) / self.r_minor_fs[isg]
                else:
                    raise ValueError('out of range')
                if o in [0, 1]:
                    searcharr2 = searcharr[self.ntheta // 2:]
                    ind = self._find(searchval, searcharr2) + self.ntheta // 2
                else:
                    searcharr2 = searcharr[0:self.ntheta // 2]
                    ind = self._find(searchval, searcharr2)
                section = np.arange(ind - stencil_width // 2,
                                    ind + stencil_width // 2)
                theta_sec = self.theta_grid[section]
                if o in [0, 1]:
                    theta_int = interp1d(-searcharr[section], theta_sec, kind=self.degree)
                    theta_of_interest = theta_int(-searchval)
                else:
                    theta_int = interp1d(searcharr[section], theta_sec, kind=self.degree)
                    theta_of_interest = theta_int(searchval)
                Z_sec = Z_sym[section]
                Z_sec_int = interp1d(theta_sec, Z_sec, kind=self.degree)
                Z_val = Z_sec_int(theta_of_interest)
                zeta_arg = (Z_val - self.Z_avg_fs[isg]) / kappa[i] / self.r_minor_fs[isg]
                if abs(zeta_arg)>=1:
                    zeta_arg = 0.999999*np.sign(zeta_arg)
                zeta_arr[o] = np.arcsin(zeta_arg)
            zeta_arr[1] = np.pi - zeta_arr[1]
            zeta_arr[3] = -np.pi - zeta_arr[3]
            zeta[i] = 0.25 * (np.pi + zeta_arr[0] - zeta_arr[1]
                                - zeta_arr[2] + zeta_arr[3])
            # calc dr/dR, amhd, and derivs
            amhd[i] = -self.qpsi_fs[isg]**2 * self.R_major_fs[isg] \
                * self.pprime_fs[isg] / r_min_spl.derivatives(self.psi_grid[isg])[1] \
                / p_mag
                
            drR[i] = R0_spl.derivatives(self.r_minor_fs[isg])[1]
            R = self.R_major_fs[isg] + self.r_minor_fs[isg]
            Z = self.Z_avg_fs[isg]
            Br = -self.psi_spl(Z, R, dx=1, dy=0)/R
            Bz = self.psi_spl(Z, R, dx=0, dy=1)/R
            bp[i] = np.sqrt(Br**2+Bz**2)
            bt[i] = self.F_fs[isg]/R
            b[i] = np.sqrt(bp[i]**2+bt[i]**2)

        amhd_spl = US(self.r_minor_fs[sgstart:], amhd, k=self.degree, s=1e-3)
        drR_spl = US(self.r_minor_fs[sgstart:], drR, k=self.degree, s=self.smooth)
        b_spl = US(self.psinorm_grid[sgstart:], b, k=self.degree, s=self.smooth)

        # calc derivatives for kappa, delta, zeta
        s_kappa = np.empty(nsg)
        s_delta = np.empty(nsg)
        s_zeta = np.empty(nsg)
        kappa_spl = US(self.r_minor_fs[sgstart:], kappa, k=self.degree, s=self.smooth)
        delta_spl = US(self.r_minor_fs[sgstart:], delta, k=self.degree, s=self.smooth)
        zeta_spl =  US(self.r_minor_fs[sgstart:], zeta,  k=self.degree, s=self.smooth)
        for i,isg in enumerate(subgrid):
            s_kappa[i] = kappa_spl.derivatives(self.r_minor_fs[isg])[1] \
                                 * self.r_minor_fs[isg] / kappa[i]
            s_delta[i] = delta_spl.derivatives(self.r_minor_fs[isg])[1] \
                                    * self.r_minor_fs[isg] / np.sqrt(1 - delta[i]**2)
            s_zeta[i] = zeta_spl.derivatives(self.r_minor_fs[isg])[1] \
                                * self.r_minor_fs[isg]
        output['dbdr'] = b_spl(self.psinorm, nu=1)[()]
        output['trpeps'] = eps
        output['q0'] = q
        output['shat'] = (r / q) * q_spl.derivatives(r)[1]
        output['amhd'] = amhd_spl(r)[()]
        output['drR'] = drR_spl(r)[()]
        output['kappa'] = kappa_spl(r)[()]
        output['s_kappa'] = kappa_spl.derivatives(r)[1] * r / kappa_spl(r)[()]
        output['delta'] = delta_spl(r)[()]
        output['s_delta'] = delta_spl.derivatives(r)[1] * r \
                                        / np.sqrt(1 - delta_spl(r)[()]**2)
        output['zeta'] = zeta_spl(r)[()]
        output['s_zeta'] = zeta_spl.derivatives(r)[1] * r
        output['minor_r'] = 1.0
        output['major_R'] = R0 / self.a_lcfs

        if not self.quiet:
            print('\n\nShaping parameters for flux surface r=%9.5g, r/a=%9.5g:' %
                  (r, self.rova))
            print('Copy the following block into a GENE parameters file:\n')
            print('trpeps  = %9.5g' % (eps))
            print('q0      = %9.5g' % q)
            print('shat    = %9.5g !(defined as r/q*dq_dr)' % (r / q \
                                     * q_spl.derivatives(r)[1]))
            print('amhd    = %9.5g' % amhd_spl(r))
            print('drR     = %9.5g' % drR_spl(r))
            print('kappa   = %9.5g' % kappa_spl(r))
            print('s_kappa = %9.5g' % (kappa_spl.derivatives(r)[1] \
                                       * r / kappa_spl(r)))
            print('delta   = %9.5g' % delta_spl(r))
            print('s_delta = %9.5g' % (delta_spl.derivatives(r)[1] \
                                       * r / np.sqrt(1 - delta_spl(r)**2)))
            print('zeta    = %9.5g' % zeta_spl(r))
            print('s_zeta  = %9.5g' % (zeta_spl.derivatives(r)[1] * r))
            print('minor_r = %9.5g' % (1.0))
            print('major_R = %9.5g' % (R0 / self.a_lcfs))


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
            plt.ylabel(r'|B|[T]', fontsize=14)

            plt.subplot(4, 2, 8)
            plt.plot(self.psinorm_grid[sgstart:],
                     b_spl(self.psinorm_grid[sgstart:], nu=1))
            plt.title('|B| deriv.')
            plt.ylabel(r'$d|B|/d\Psi_N$', fontsize=14)

            for ax in plt.gcf().axes:
                ax.set_xlabel(r'$\Psi_N$', fontsize=14)
                ax.axvline(self.psinorm, 0, 1, ls='--', color='k', lw=2)
            plt.tight_layout(pad=0.3)
            
            if self.save:
                filename = os.path.join(self.folder, self.basename+'_derived.pdf')
                plt.savefig(filename)


        return output
    
    def close(self):
        plt.close('all')


eq50 = '/p/gene/drsmith/eqdsk/ConstBeta_changingBT_eq50.geqdsk'
eq21 = '/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk'

def adjust_miller(gfile = eq50, ref_gfile = eq21, *args, **kwargs):

    #kwargs = {'plot':True}

    ref_geq = Geqdsk(gfile=ref_gfile, *args, **kwargs)
    ref_miller = ref_geq.miller()

    geq = Geqdsk(gfile=gfile, *args, **kwargs)
    miller = geq.miller(ref_miller=ref_miller)

    for key in iter(miller):
        if key == 'gfile': continue
        print('{:10}: {:10.4g}  {:10.4g}'.format(key, miller[key], ref_miller[key]))
        
    keys = ['F','FFp','p_total','Bref','beta',
            'dpdr','dpdx_pm','dpdx_pm_2','pprime','dbdr','amhd','omp']
    for key in keys:
        print('{:10}: {:10.3g}  {:10.3g}  ratio {:.3g}'.format(
                key, miller[key], ref_miller[key], miller[key]/ref_miller[key]))


if __name__ == '__main__':
    #adjust_miller()
    geq = Geqdsk(plot=True, save=True, gfile=eq21)
    miller = geq.miller(psinorm=0.6)
    
    modeq = Geqdsk(plot=True, save=True, gfile=eq50)
    modeq.miller(psinorm=0.6, ref_miller=miller)
#    psinorm=0.8
#    g1 = Geqdsk(gfile='/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk',
#                plot=True, save=True)
#    m1 = g1.miller(psinorm=psinorm)
#    g2 = Geqdsk(gfile='/p/gene/drsmith/eqdsk/ConstBeta_changingBT_eq50.geqdsk',
#                plot=True, save=True)
#    m2 = g2.miller(psinorm=psinorm, ref_pressure=58.6)
#    for key in iter(m1):
#        if key == 'gfile': continue
#        print('{:10}: {:10.4g}  {:10.4g}'.format(key, m1[key], m2[key]))
#        
#    keys = ['F','FFp','p_total','pprime','Bref','beta',
#            'dpdr','dpdx_pm','omp','dbdr','amhd']
#    for key in keys:
#        print('{:10}: {:10.3g}  {:10.3g}  ratio {:.3g}'.format(
#                key, m1[key], m2[key], m1[key]/m2[key]))
