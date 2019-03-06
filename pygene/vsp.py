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


class Vspace(object):

    def __init__(self, parent=None):
        self.tind = None
        self._scannum = None
        self._parent = parent
        self.species = self._parent.species
        self.path = None
        self.time = None
        self._isnonlinear = self._parent._isnonlinear
        self._islinearscan = self._parent._islinearscan
        
    def _check_data(self, scannum=None, tind=None):
        if self._islinearscan:
            if scannum is None:
                scannum = 1
            self._scannum = scannum
        if tind is None:
            tind = -1
        self._get_parent_parameters()
        self._make_grids()
        self._set_binary_configuration()
        self._set_vsp_path()
        self._read_time_array()
        self.tind = self._adjust_tind(tind)
        self._get_data()
    
    def _set_vsp_path(self):
        if self._islinearscan:
            self.path = self._parent.path / 'vsp_{:04d}'.format(self._scannum)
        else:
            self.path = self._parent.path / 'vsp.dat'

    def _get_parent_parameters(self):
        if self._isnonlinear:
            paramsfile = self._parent.path / 'parameters.dat'
        else:
            paramsfile = self._parent.path / 'parameters_{:04d}'.format(self._scannum)
        self._processed_parameters = \
            self._parent._get_processed_parameters(paramsfile=paramsfile)
        self.nz0 = self._processed_parameters['nz0']
        self.nv0 = self._processed_parameters['nv0']
        self.nw0 = self._processed_parameters['nw0']
        self._ndatapoints = np.array([self.nz0, self.nv0, self.nw0]).prod()

    def _make_grids(self):
        # z grid
        delz = 2.0*np.pi / self.nz0
        self.zgrid = np.linspace(-np.pi, np.pi-delz, self.nz0)
        # v_parallel grid
        maxvpar = self._processed_parameters['lv']
        self.vpargrid = np.linspace(-maxvpar, maxvpar, self.nv0)
        # mu grid
        roots, weights = np.polynomial.laguerre.laggauss(self.nw0)
        weights *= np.exp(roots)
        scale = self._processed_parameters['lw'] / np.sum(weights)
        self.mugrid = roots*scale
        self.muweights = weights*scale
        
    def _set_binary_configuration(self):
        self.binconfig = {}
        if self._processed_parameters['ENDIANNESS']=='BIG':
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
        self.binconfig['elements'] = self._ndatapoints * len(self.species) * 5
        # 8B double precision
        realsize = 4 + 4 * (self._processed_parameters['PRECISION']=='DOUBLE')
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

    def _adjust_tind(self, tind):
        # load data based on tind and _ivar
        if isinstance(tind, (tuple,list,np.ndarray)):
            tind = np.asarray(tind, dtype=np.int)
        else:
            tind = np.asarray([tind], dtype=np.int)
        tind[tind<0] += self.time.size
        return tind
        
    def _get_data(self):
        self.timeslices = self.time[self.tind]
        self.vspdata = {}
        for spname in self.species:
            self.vspdata[spname] = np.empty([self.nz0, self.nv0, 
                                             self.nw0, self.tind.size],
                                            dtype=self.binconfig['nprt'])
        with self.path.open('rb') as f:
            for i,it in enumerate(self.tind):
                rawdata_shape = (self.nz0, self.nv0, self.nw0, 
                                 len(self.species), 5)
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
    
    def plot_vspace(self, tind=None, scannum=None):
        self._check_data(tind=tind, scannum=scannum)
        title = self._parent.label
        if self._scannum:
            title += ' index {:d}'.format(self._scannum)
        self._plot_title = title
        # make uniformly-spaced grid in log10(mu) space
        self.mugrid_log10 = np.log10(self.mugrid)
        mugrid_new = np.linspace(self.mugrid_log10[0], 
                                 self.mugrid_log10[-1], 
                                 self.mugrid_log10.size)
        fig,ax = plt.subplots(nrows=2,
                              ncols=len(self.species)+1,
                              gridspec_kw={'top':0.9,
                                           'left':0.07,
                                           'right':0.93,
                                           'hspace':0.4,
                                           'wspace':0.35},
                              figsize=[9.5,5.5])
        for isp, species in enumerate(self.species):
            # sqrt(<f^2>) data for species
            vdata = np.mean(self.vspdata[species], axis=-1)
            # value at midplane
            izmid = self._processed_parameters['nz0']//2
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
                       cmap=plt.get_cmap('gnuplot2'),
                       interpolation='bilinear')
            plt.xlabel('v_par/v_th')
            plt.ylabel('log10(mu/mu_th)')
            plt.title('sqrt(<f^2>) ' + species)
            plt.colorbar()
            # integrated spectra
            plt.sca(ax[0,len(self.species)])
            plt.plot(self.vpargrid,
                     vdata_zmid_logmu.sum(axis=1),
                     label=species)
            plt.xlabel('v_par/v_th')
            plt.ylabel('sum over mu')
            plt.legend()
            plt.sca(ax[1,len(self.species)])
            plt.plot(mugrid_new,
                     vdata_zmid_logmu.sum(axis=0),
                     label=species)
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
                       cmap=plt.get_cmap('gnuplot2'),
                       interpolation='bilinear')
            plt.xlabel('f(v_par/v_th)')
            plt.ylabel('f(log10(mu/mu_th))')
            plt.title('FFT(sqrt(<f^2>)) ' + species)
            plt.colorbar()
        fig.suptitle(self._plot_title)
        
