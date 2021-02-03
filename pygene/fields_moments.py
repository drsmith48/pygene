from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import super
from builtins import range
from future import standard_library
standard_library.install_aliases()

import os
import struct

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from . import main
from . import utils

field_names = ['phi', 'apar', 'bpar']
mom_names = ['dens', 'T_para', 'T_perp', 'Q_para', 'Q_perp', 'u_para']

class _DataABC(object):

    def __init__(self, ivar=None, parent=None):

        if not (hasattr(self, '_nvars') and \
                hasattr(self, 'varname') and \
                hasattr(self, 'path')):
            raise AttributeError('subclass must implement _nvars, varname, and path')
        if not isinstance(parent, (main.GeneLinearScan, main.GeneNonlinear)):
            raise ValueError('invalid parent type')

        self._parent = parent
        self._isnonlinear = self._parent._isnonlinear
        self._islinearscan = self._parent._islinearscan
        self._scannum = None if self._isnonlinear else 1
        self._ivar = ivar
        self._processed_parameters = None
        self._ndatapoints = None
        self._binary_configuration = None
        self.tind = None
        self.time = None
        self._get_parent_parameters(self._scannum)
        self._set_binary_configuration()
        self._set_path(self._scannum)

    def __call__(self, *args, **kwargs):
        self._check_data(*args, **kwargs)

    def _get_parent_parameters(self, scannum=None):
        if self._isnonlinear and scannum is None:
            paramsfile = self._parent.path / 'parameters.dat'
        else:
            paramsfile = self._parent.path / 'parameters_{:04d}'.format(scannum)
        self._processed_parameters = \
            self._parent._get_processed_parameters(paramsfile=paramsfile)
#        print('omega ref = {:.3f} kHz'.
#              format(self._processed_parameters['omega_ref']/(2*np.pi)/1e3))
        # get simulation domain, make grids
        self.nx0 = self._processed_parameters['nx0']
        self.nky0 = self._processed_parameters['nky0']
        self.nz0 = self._processed_parameters['nz0']
        self.kymin = self._processed_parameters['kymin']
        self._ndatapoints = np.array([self.nx0, self.nky0, self.nz0]).prod()
        self.lx = self._processed_parameters['lx']
        self.ly = self._processed_parameters['ly']
        if self.lx==0 or self.ly==0:
            return
        self.xres = self.lx / self.nx0
        self.xgrid = np.linspace(-self.lx/2, self.lx/2, self.nx0)
        delkx = 2*np.pi / self.lx
        kxmax = self.nx0/2 * delkx
        self.kxgrid = np.linspace(-(kxmax-delkx), kxmax, self.nx0)
        self.kygrid = np.arange(self.nky0) * self.kymin
        delz = 2*np.pi / self.nz0
        self.zgrid = np.linspace(-np.pi, np.pi-delz, self.nz0)

    def _set_binary_configuration(self):
        if self._processed_parameters['PRECISION']=='DOUBLE':
            realsize = 8
        else:
            realsize = 4
        complexsize = 2*realsize
        intsize = 4
        entrysize = self._ndatapoints * complexsize
        leapfld = self._nvars * (entrysize+2*intsize)
        if self._processed_parameters['ENDIANNESS']=='BIG':
            nprt=(np.dtype(np.float64)).newbyteorder()
            npct=(np.dtype(np.complex128)).newbyteorder()
            fmt = '>idi'
        else:
            nprt=np.dtype(np.float64)
            npct=np.dtype(np.complex128)
            fmt = '=idi'
        te = struct.Struct(fmt)
        tesize = te.size
        self._binary_configuration = (intsize, entrysize, leapfld,
                                     nprt, npct, te, tesize)

    def _set_path(self, scannum=None):
        # implement in subclass
        pass

    def _check_data(self, scannum=None, ivar=None, tind=None):
        if self._islinearscan:
            if scannum is None:
                scannum = 1
            if scannum != self._scannum:
                self._scannum = scannum
                self._get_parent_parameters(scannum)
                self._set_binary_configuration()
                self._set_path(scannum)
                self._read_time_array()
        if self.time is None:
            self._read_time_array()
        if ivar is not None:
            self._ivar = ivar
        if tind is None:
            tind = -1
        self._adjust_tind(tind)
        self._get_data()

    def _read_time_array(self):
        intsize, entrysize, leapfld, nprt, npct, te, tesize = self._binary_configuration
        self.time = np.empty(0)
        with self.path.open('rb') as f:
            filesize = os.path.getsize(self.path.as_posix())
            for i in range(filesize // (leapfld+tesize)):
                value = float(te.unpack(f.read(tesize))[1])
                self.time = np.append(self.time, value)
                f.seek(leapfld,1)

    def _adjust_tind(self, tind):
        # load data based on tind and _ivar
        if isinstance(tind, (tuple,list,np.ndarray)):
            tind = np.asarray(tind, dtype=np.int)
        else:
            tind = np.asarray([tind], dtype=np.int)
        tind[tind<0] += self.time.size
        self.tind = tind

    def _get_data(self):
        intsize, entrysize, leapfld, nprt, npct, te, tesize = self._binary_configuration
        data = np.empty((self.nx0,
                         self.nky0,
                         self.nz0,
                         self.tind.size), dtype=npct)
        offset = self.tind * (tesize+leapfld) + \
            self._ivar * (entrysize+2*intsize) + \
            intsize + tesize
        with self.path.open('rb') as f:
            for i,off in enumerate(offset):
                f.seek(off)
                flatdata = np.fromfile(f, count=self._ndatapoints, dtype=npct)
                if self.nz0*self.nky0*self.nx0 != flatdata.size:
                    raise ValueError
                data[:,:,:,i] = flatdata.reshape((self.nz0,self.nky0,self.nx0)).transpose()
        data_kxkyz_f = data[:,:,:,-1]
        if self.nky0>1:
            data_xyz = np.real(np.fft.ifft2(data_kxkyz_f, axes=[0,1]))
            self._xyimage = data_xyz[:,:,self.nz0//2]
            self._xzimage = data_xyz[:,self.nky0//2,:]
        else:
            data_kxz_f = data_kxkyz_f[:,0,:]
            self._xyimage = None
            self._xzimage = np.real(np.fft.ifft(data_kxz_f, axis=0))
        # roll in kx dimension to order from kxmin to kxmax
        self.data = np.roll(data, self.nx0//2-1, axis=0)
        self.kxspectrum = np.mean(np.square(np.abs(self.data)), axis=(1,2,3))
        self.kxspectrum[self.nx0//2-1] *= 2
        # rehsape to ballooning representation (final time slice only)
        self.ballooning = np.reshape(self.data[0:-1,:,:,-1], ((self.nx0-1)*self.nz0, self.nky0), order='C')
        self.ballooning = np.squeeze(self.ballooning)
        nconnections = self.nx0-1
        paragridsize = nconnections*self.nz0
        self.ballgrid = np.empty(paragridsize)
        for i in np.arange(nconnections):
            self.ballgrid[i*self.nz0:(i+1)*self.nz0] = \
                2*(i-nconnections//2) + self.zgrid/np.pi
        if self.nky0>1:
            return
        # calculate mode parity
        # even parity: sumsig >> diffsig > 0 and parity ~ +1
        # odd parity: diffsig >> sumsig > 0 and parity ~ -1
        imid = self.ballooning.size//2
        diffsig = np.mean(np.abs(self.ballooning[imid+1:] - \
                                 self.ballooning[imid-1:0:-1]))
        sumsig = np.mean(np.abs(self.ballooning[imid+1:] + \
                                self.ballooning[imid-1:0:-1]))
        self.parity = np.mean(np.divide(sumsig-diffsig,sumsig+diffsig+1e-12))
        nz4 = self.nz0//2
        diffsig2 = np.mean(np.abs(self.ballooning[imid+1:imid+1+nz4] - \
                                  self.ballooning[imid-1:imid-1-nz4:-1]))
        sumsig2 = np.mean(np.abs(self.ballooning[imid+1:imid+1+nz4] + \
                                 self.ballooning[imid-1:imid-1-nz4:-1]))
        self.parity2 = np.mean(np.divide(sumsig2-diffsig2,sumsig2+diffsig2+1e-12))
        taillength = np.int(np.floor(0.025*paragridsize))
        absmode = np.abs(self.ballooning)
        absmode[absmode<1e-16] = 1e-16
        tails = np.concatenate((absmode[0:taillength+1], absmode[-(taillength+1):]))
        tailsize = np.max(tails/np.max(absmode))
        self.tailsize = np.max([tailsize,1e-3])
        realmode = np.real(self.ballooning)
        if realmode.max()<1e-16: realmode[0] = 1e-16
        realmode = realmode / np.sqrt(np.sum(np.abs(realmode)**2))
        wavelet = signal.ricker(5,0.5)
        method = signal.choose_conv_method(realmode,wavelet)
        filtsig = signal.correlate(realmode, wavelet, method=method)
        self.gridosc = np.sum(np.abs(filtsig)**2)

    def plot_mode(self, scannum=None, ivar=None, tind=None, save=False):
        self._check_data(scannum=scannum, ivar=ivar, tind=tind)
        plot_title = self._parent.label
        if self._islinearscan:
            plot_title += ' id {:d}'.format(self._scannum)
            # linear sim with nky0=1
            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=[13,2.85])
        else:
            # nonlinear sim with nky0>1
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=[11,5.5])
        # plot parallel mode structure in axis #1
        iax = 0
        plt.sca(ax.flat[iax])
        plt.plot(self.ballgrid, np.abs(self.ballooning), label='Abs()')
        if self._islinearscan:
            plt.plot(self.ballgrid, np.real(self.ballooning), label='Re()')
            plt.legend()
        plt.title(plot_title)
        plt.xlabel('Ballooning angle (rad/pi)')
        plt.ylabel(self.varname)
        plt.annotate('ky={:.3f}'.format(self.kymin),
                     xycoords='axes fraction',
                     xy=[0.05,0.92])
        xmax = self.ballgrid.max()
        itwopi = 0
        ylim = plt.gca().get_ylim()
        while itwopi<=6 and itwopi<=xmax:
            plt.plot(np.ones(2)*itwopi, ylim, color='tab:gray', linestyle='--')
            if itwopi != 0:
                plt.plot(-np.ones(2)*itwopi, ylim, color='tab:gray', linestyle='--')
            itwopi += 2
        if self._islinearscan:
            # plot parallel mode structure in axis #1
            iax += 1
            plt.sca(ax.flat[iax])
            plt.plot(self.ballgrid, np.abs(self.ballooning), label='Abs()')
            if self._islinearscan:
                plt.plot(self.ballgrid, np.real(self.ballooning), label='Re()')
                plt.legend()
            plt.title(plot_title)
            plt.xlabel('Ballooning angle (rad/pi)')
            plt.ylabel(self.varname)
            plt.annotate('ky={:.3f}'.format(self.kymin),
                         xycoords='axes fraction',
                         xy=[0.05,0.92])
            xmax = self.ballgrid.max()
            plt.xlim([-5,5])
            itwopi = 0
            ylim = plt.gca().get_ylim()
            while itwopi<=5:
                plt.plot(np.ones(2)*itwopi, ylim, color='tab:gray',linestyle='--')
                if itwopi != 0:
                    plt.plot(-np.ones(2)*itwopi, ylim, color='tab:gray', linestyle='--')
                itwopi += 2
        # plot kx spectrum in axis #2
        iax += 1
        plt.sca(ax.flat[iax])
        if self.kxgrid.size >= 64:
            style='-'
        else:
            style='d-'
        plt.plot(self.kxgrid[:-1], utils.log1010(self.kxspectrum[:-1]/self.kxspectrum.max()), style)
        plt.xscale('symlog')
        plt.xlabel('kx rho_s')
        plt.ylabel('PSD('+self.varname+') [dB]')
        plt.ylim(-60,0)
        plt.title(plot_title)
        # plot x,z image in axis #3
        iax += 1
        plt.sca(ax.flat[iax])
        plt.imshow(self._xzimage.transpose(),
                   aspect='auto',
                   extent=[-self.lx/2, self.lx/2,-np.pi,np.pi],
                   origin='lower',
                   cmap=plt.cm.bwr,
                   interpolation='bilinear')
        m = np.max(np.abs(self._xzimage))
        plt.clim(-m,m)
        plt.xlabel('x/rho_s')
        plt.ylabel('z (rad)')
        plt.title(plot_title)
        plt.colorbar()
        if self._isnonlinear:
            data = np.mean(np.abs(self.data),axis=3)
            # plot kx,ky spectrum in axis #4
            iax += 1
            plt.sca(ax.flat[iax])
            kxkydata = np.mean(data, axis=2)
            plt.imshow(utils.log1010(kxkydata/kxkydata.max()).transpose(),
                       aspect='auto',
                       extent=[self.kxgrid[0], self.kxgrid[-1],
                               self.kygrid[0], self.kygrid[-1]],
                       origin='lower',
                       cmap=plt.get_cmap('gnuplot2'),
                       interpolation='bilinear')
            plt.clim(-50,0)
            plt.xlabel('kx rho_s')
            plt.ylabel('ky rho_s')
            plt.title(plot_title)
            plt.colorbar()
            # plot ky spectrum in axis #5
            iax += 1
            plt.sca(ax.flat[iax])
            kydata = np.mean(data, axis=(0,2))
            plt.plot(self.kygrid,
                     utils.log1010(kydata/kydata.max()),
                     '-x')
            plt.xlabel('ky rho_s')
            plt.ylim(-50,0)
            plt.ylabel('PSD('+self.varname+') [dB]')
            plt.title(plot_title)
            # plot x,y image in axis #6
            lx = self._processed_parameters['lx']
            ly = self._processed_parameters['ly']
            iax += 1
            plt.sca(ax.flat[iax])
            plt.imshow(self._xyimage.transpose(),
                       origin='lower',
                       cmap=plt.cm.bwr,
                       extent=[-lx/2,lx/2,-ly/2,ly/2],
                       interpolation='bilinear',
                       aspect='auto',
                       )
            m = np.max(np.abs(self._xyimage))
            plt.clim(-m,m)
            plt.colorbar()
            plt.xlabel('x/rho_s')
            plt.ylabel('y/rho_s')
            plt.title(plot_title)
        plt.tight_layout()
        if save:
            varname = self.varname.replace(' ', '-')
            plt.savefig(f'{self._parent.path.parts[-2]}-{self._parent.path.parts[-1]}-{varname}.pdf',
                        format='pdf', transparent=True)


class Field(_DataABC):

    def __init__(self, field='', parent=None):
        self._nvars = len(parent.fields)
        self.varname = field
        self.path = None
        super().__init__(ivar=field_names.index(field), parent=parent)

    def _set_path(self, scannum=None):
        if self._islinearscan and scannum is not None:
            self.path = self._parent.path / 'field_{:04d}'.format(scannum)
        else:
            self.path = self._parent.path / 'field.dat'

    def plot_mode(self, scannum=None, tind=None, save=False):
        super().plot_mode(scannum=scannum, tind=tind, save=save)


class Moment(_DataABC):

    def __init__(self, species='', parent=None):
        self._nvars = 6
        self.varname = None
        self.path = None
        self._imoment = None
        self.species = species
        self.gamma_es = None
        self.gamma_em = None
        self.q_es = None
        self.q_em = None
        self._set_moment()
        self.fluxes = None
        self.flux_names = None
        super().__init__(ivar=self._imoment, parent=parent)

    def _set_moment(self, moment=0):
        self._imoment = moment
        self.varname = self.species[0:3] + ' ' + mom_names[moment]

    def _set_path(self, scannum=None):
        if self._islinearscan and scannum is not None:
            self.path = self._parent.path / 'mom_{}_{:04d}'.format(self.species, scannum)
        else:
            self.path = self._parent.path / 'mom_{}.dat'.format(self.species)
        if not self.path.exists():
            raise ValueError

    def plot_mode(self, scannum=None, moment=None, tind=None, save=False):
        if moment is not None:
            self._set_moment(moment=moment)
        super().plot_mode(scannum=scannum, ivar=moment, tind=tind, save=save)

    def _calc_fluxes(self, tind=None):
        if tind is None:
            tind =np.arange(-1,-3*4,-4)
        else:
            self._adjust_tind(tind)
            tind = np.copy(self.tind)
        self._check_data(tind=tind)
        self._parent.phi._check_data(tind=tind)
        phi = np.copy(self._parent.phi.data)
        ky_tile = np.broadcast_to(self.kygrid.reshape((1,self.nky0,1,1)),
                                  [self.nx0, self.nky0, self.nz0, tind.size])
        vex = -1j * ky_tile * phi
        self._parent.apar._check_data(tind=tind)
        apar = np.copy(self._parent.apar.data)
        bx = 1j * ky_tile * apar
        moms = []
        for imom in range(6):
            self._check_data(ivar=imom, tind=tind)
            moms.append(np.copy(self.data))
        nref = self._processed_parameters['nref']
        Tref = self._processed_parameters['Tref']
        self.fluxes = np.empty_like(np.broadcast_to(vex[...,np.newaxis],
                                    (self.nx0,self.nky0,self.nz0,self.tind.size,4)))
        self.flux_angles = np.empty_like(self.fluxes.real)
        self.flux_names = ['gamma_es', 'gamma_em', 'q_es', 'q_em']
        # gamma ES  Gamma_es = <n * ve_x>
        self.fluxes[...,0] = np.conj(moms[0])*vex
        self.flux_angles[...,0] = np.angle(np.conj(moms[0])*(-phi))/np.pi
        # gamma EM  Gamma_em = <upar * B_x>
        self.fluxes[...,1] = np.conj(moms[5])*bx
        self.flux_angles[...,1] = np.angle(np.conj(moms[5])*(-apar))/np.pi
        # q ES  Q_es = (1/2 Tpar + Tperp + 3/2 n) ve_x
        tmp1 = 1.5*moms[0]*Tref + 0.5*moms[1]*nref + moms[2]*nref
        self.fluxes[...,2] = np.conj(tmp1) * vex
        self.flux_angles[...,2] = np.angle(np.conj(tmp1)*phi)/np.pi
        # q EM  Q_em = (qpar + qperp + 5/2 upar) B_x
        tmp2 = moms[3] + moms[4]
        self.fluxes[...,3] = np.conj(tmp2) * bx
        self.flux_angles[...,3] = np.angle(np.conj(tmp2)*(-apar))/np.pi

    def plot_fluxes(self, save=False, **kwargs):
        if kwargs or self.fluxes is None:
            self._calc_fluxes(**kwargs)
        # kx spectra
        plt.figure(figsize=(11,6))
        plt.subplot(231)
        xspec = np.sum(np.real(self.fluxes), (1,2)) / self.nz0
        xspec_mn = np.mean(xspec,1)
        xspec_std = np.std(xspec, 1)
        for i in range(4):
            d = xspec_mn[...,i]
            dstd = xspec_std[...,i]
            yerr = ( 10*np.log10(d+dstd) - 10*np.log10(d-dstd) ) / 2
            plt.errorbar(self.kxgrid,
                         10*np.log10(d),
                         yerr=yerr,
                         label=self.flux_names[i])
        plt.xscale('symlog')
        plt.xlabel('kx * rho_s')
        plt.ylabel('gamma/gamma_gb, q/q_gb')
        plt.ylim(-40,10)
        plt.legend()
        plt.title(self._parent.label)
        # ky spectra
        plt.subplot(232)
        yspec = np.sum(np.real(self.fluxes), (0,2)) / self.nz0
        yspec_mn = np.mean(yspec[1:,...], 1)
        yspec_std = np.std(yspec[1:,...], 1)
        yspec_mn_neg = -np.copy(yspec_mn)
        yspec_mn[yspec_mn<1e-4] = np.nan
        yspec_mn_neg[yspec_mn_neg<1e-4] = np.nan
        for i in range(4):
            d = yspec_mn[...,i]
            dstd = yspec_std[...,i]
            if np.any(np.isfinite(d)):
                yerr = ( 10*np.log10(d+dstd) - 10*np.log10(d-dstd) ) / 2
                plt.errorbar(self.kygrid[1:],
                             10*np.log10(d),
                             yerr=yerr,
                             marker='x',
                             linestyle='-',
                             label=self.flux_names[i])
            d = yspec_mn_neg[...,i]
            if np.any(np.isfinite(d)):
                yerr = ( 10*np.log10(d+dstd) - 10*np.log10(d-dstd) ) / 2
                plt.errorbar(self.kygrid[1:],
                             10*np.log10(d),
                             yerr=yerr,
                             marker='+',
                             linestyle='--',
                             label=self.flux_names[i]+' (pinch)')
        plt.xlabel('ky * rho_s')
        plt.ylabel('gamma/gamma_gb, q/q_gb')
        plt.legend()
        plt.title(self._parent.label)
        plt.ylim(-40,10)
        # plot 2D histogram of cross-phases vs ky, weighted by flux
        nbins = 80
        abins = np.linspace(-1,1,nbins+1)
        for i in range(4):
            counts = np.empty((self.nky0-1, nbins))
            for iky in range(self.nky0-1):
                counts[iky,:],_ = np.histogram(
                    self.flux_angles[:,iky+1,:,:,i].flatten(),
                    bins=abins,
                    weights=np.abs(self.fluxes[:,iky+1,:,:,i].flatten()),
                    density=True)
            plt.subplot(2,3,i+3)
            plt.contourf((abins[0:-1]+abins[1:])/2, self.kygrid[1:], counts)
            plt.xlabel('cross-phase (rad/pi)')
            plt.ylabel('ky * rho_s')
            plt.title(self._parent.label+' | '+self.flux_names[i])
            plt.colorbar()
        plt.tight_layout()
        if save:
            plt.savefig(f'{self._parent.path.parts[-2]}-{self._parent.path.parts[-1]}-{self.species[0:3]}-fluxes.pdf',
                        format='pdf', transparent=True)
