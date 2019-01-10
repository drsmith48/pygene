from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()

import os
import struct

import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl


class Vspace(object):

    def __init__(self, tind=-1, run=None, parent=None):
        self.tind = tind
        self.run = run
        self.parent = parent
        if self.run:
            self.path = self.parent.path / 'vsp_{:04d}'.format(self.run)
        else:
            self.path = self.parent.path / 'vsp.dat'
        self.shortpath = '/'.join(self.path.parts[-3:])
        self.plotlabel = self.parent.plotlabel
        self.filelabel = self.parent.filelabel
        self._read_paramsfile()
        self._make_grids()
        self._set_binary_configuration()
        self.get_data()
        self._set_plot_title()

    def _set_binary_configuration(self):
        self.binconfig = {}
        if self.params['ENDIANNESS']=='BIG':
            self.binconfig['nprt']=(np.dtype(np.float64)).newbyteorder()
            self.binconfig['format'] = '>idi'
        else:
            self.binconfig['nprt']=np.dtype(np.float64)
            self.binconfig['format'] = '=idi'
        self.binconfig['time_struct'] = struct.Struct(self.binconfig['format'])
        # double size with pad bytes
        self.binconfig['tesize'] = self.binconfig['time_struct'].size # = 16 = 4B int + 8B double + 4B int
        # 4B int
        self.binconfig['intsize'] = 4
        # num. elements in data array [nz,nv,nw,nspec,5]
        self.binconfig['elements'] = self.vdims.prod() * self.nspecies * 5
        # 8B double precision
        realsize = 4 + 4 * (self.params['PRECISION']=='DOUBLE')
        # bytes for double precision data (no pad bytes)
        entrysize = self.binconfig['elements'] * realsize
        # bytes from end of time value to beginning of next time value
        # int's are pre- and post-data pad bytes
        self.binconfig['leapfld'] = entrysize + 2*self.binconfig['intsize']
        # bytes for time stamp + data array + pad bytes
        # should be integer factor of VSP filesize
        self.binconfig['full_time_segment'] = self.binconfig['leapfld'] + self.binconfig['tesize']

    def _read_time_array(self):
        self.time = np.empty(0)
        with self.path.open('rb') as f:
            filesize = os.path.getsize(self.path.as_posix())
            ntime = filesize // self.binconfig['full_time_segment']
            for i in range(ntime):
                unpacked = self.binconfig['time_struct'].unpack(f.read(self.binconfig['tesize']))
                time_value = unpacked[1]
                self.time = np.append(self.time, time_value)
                f.seek(self.binconfig['leapfld'], 1)

    def _make_grids(self):
        # z grid
        delz = 2.0*np.pi / self.params['nz0']
        self.zgrid = np.linspace(-np.pi, np.pi-delz, self.params['nz0'])
        # v_parallel grid
        maxvpar = self.params['lv']
        self.vpargrid = np.linspace(-maxvpar,maxvpar,self.params['nv0'])
        # mu grid
        roots, weights = np.polynomial.laguerre.laggauss(self.params['nw0'])
        weights *= np.exp(roots)
        scale = self.params['lw'] / np.sum(weights)
        self.mugrid = roots*scale
        self.muweights = weights*scale
        
    def _read_paramsfile(self):
        if self.run:
            paramsfile = self.parent.path / 'parameters_{:04d}'.format(self.run)
            self.params = self.parent._read_parameters(paramsfile)
        else:
            self.params = self.parent.params
        self.vdims = np.array([self.params['nz0'],
                               self.params['nv0'],
                               self.params['nw0']])
        self.nspecies = self.params['n_spec']
        self.species = self.params['species']
        self.dtmax = self.params.get('dt_max', 1e-9)
        self.nprocs = self.params.get('n_procs_sim', 0)
        self.wcperstep = self.params.get('step_time', 0.0)
        self.wcperunittimepercpu = self.wcperstep / self.dtmax / self.nprocs

    def get_data(self, tind=None):
        if tind is not None:
            self.tind = tind
        if isinstance(self.tind, (tuple,list,np.ndarray)):
            self.tind = np.asarray(self.tind, dtype=np.int)
        else:
            self.tind = np.asarray([self.tind], dtype=np.int)
        self._read_time_array()
        self.tind[self.tind<0] += self.time.size
        self.timeslices = self.time[self.tind]
        self.vspdata = {}
        for spname in self.species:
            self.vspdata[spname] = np.empty([self.vdims[0], self.vdims[1], 
                                             self.vdims[2], self.tind.size],
                                            dtype=self.binconfig['nprt'])
        with self.path.open('rb') as f:
            for i,it in enumerate(self.tind):
                rawdata_shape = (self.vdims[0], self.vdims[1], self.vdims[2], 
                                 self.nspecies, 5)
                rawdata = np.empty(rawdata_shape, 
                                   dtype=self.binconfig['nprt'])
                offset = it * self.binconfig['full_time_segment'] + \
                    self.binconfig['tesize'] + self.binconfig['intsize']
                f.seek(offset)
                flatdata = np.fromfile(f, count=self.binconfig['elements'], 
                                       dtype=self.binconfig['nprt'])
                rawdata = flatdata.reshape(rawdata_shape[::-1]).transpose()
                for isp,spname in enumerate(self.species):
                    self.vspdata[spname][:,:,:,i] = rawdata[:,:,:,isp,4]
    
    def _set_plot_title(self):
        title = self.shortpath
#        if self.run:
#            title += ' run {}'.format(self.run)
        if self.tind.size==1:
            title += ' t={:.0f}'.format(self.timeslices[0])
        else:
            title += ' t={:.0f}-{:.0f}'.format(self.timeslices[0],
                                               self.timeslices[-1])
        self.plot_title = title
        
    def plot_vspace(self):
        # make uniformly-spaced grid in log10(mu) space
        self.mugrid_log10 = np.log10(self.mugrid)
        mugrid_new = np.linspace(self.mugrid_log10[0], 
                                 self.mugrid_log10[-1], 
                                 self.mugrid_log10.size)
        fig,ax = plt.subplots(nrows=2,
                              ncols=self.nspecies+1,
                              gridspec_kw={'top':0.9,
                                           'left':0.07,
                                           'right':0.93,
                                           'hspace':0.4,
                                           'wspace':0.35},
                              figsize=[9.5,5.5])
        for isp in range(self.nspecies):
            # sqrt(<f^2>) data for species
            vdata = np.mean(self.vspdata[self.species[isp]], axis=-1)
            # value at midplane
            izmid = self.params['nz0']//2
            vdata_zmid = np.squeeze(vdata[izmid,:,:])
            # spline data onto log10(mu) grid
            vdata_spl = RectBivariateSpline(self.vpargrid,
                                            self.mugrid_log10,
                                            vdata_zmid)
            vdata_zmid_logmu = vdata_spl(self.vpargrid, 
                                         mugrid_new, 
                                         grid=True)
            plt.sca(ax[0,isp])
            plt.imshow(vdata_zmid_logmu.transpose(),
                       aspect='auto',
                       extent=[self.vpargrid[0], self.vpargrid[-1],
                               self.mugrid_log10[0], self.mugrid_log10[-1]],
                       origin='lower',
                       cmap=mpl.cm.gnuplot,
                       interpolation='bilinear')
            plt.xlabel('v_par/v_th')
            plt.ylabel('log10(mu/mu_th)')
            plt.title('sqrt(<f^2>) ' + self.species[isp])
            plt.colorbar()
            # integrated spectra
            plt.sca(ax[0,self.nspecies])
            plt.plot(self.vpargrid,
                     vdata_zmid_logmu.sum(axis=1),
                     label=self.species[isp])
            plt.xlabel('v_par/v_th')
            plt.ylabel('sum over mu')
            plt.legend()
            plt.sca(ax[1,self.nspecies])
            plt.plot(mugrid_new,
                     vdata_zmid_logmu.sum(axis=0),
                     label=self.species[isp])
            plt.xlabel('log10(mu/mu_th)')
            plt.ylabel('sum over v_par')
            plt.legend()
            # FFT of v-space spectrum
            fft = np.fft.fftshift(np.fft.rfft2(vdata_zmid_logmu), axes=0)
            f_vpar = np.fft.fftfreq(self.vpargrid.size,
                                     self.vpargrid[1]-self.vpargrid[0])
            f_vpar = np.fft.fftshift(f_vpar)
            f_logmu = np.fft.rfftfreq(self.mugrid_log10.size,
                                     self.mugrid_log10[1]-self.mugrid_log10[0])
            plt.sca(ax[1,isp])
            plt.imshow(np.abs(fft).transpose(),
                       aspect='auto',
                       extent=[f_vpar[0], f_vpar[-1],
                               f_logmu[0], f_logmu[-1]],
                       origin='lower',
                       cmap=mpl.cm.gnuplot,
                       interpolation='bilinear')
            plt.xlabel('f(v_par/v_th)')
            plt.ylabel('f(log10(mu/mu_th))')
            plt.title('FFT(sqrt(<f^2>)) ' + self.species[isp])
            plt.colorbar()
        fig.suptitle(self.plot_title)
        
