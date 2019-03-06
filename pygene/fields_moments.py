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
dbmin = -30

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
        if self._isnonlinear:
            self._scannum = None
        else:
            self._scannum = -1
        self._ivar = ivar
        self._processed_parameters = None
        self._ndatapoints = None
        self._binary_configuration = None
        self.tind = None
        self.time = None
        
    def _check_data(self, scannum=None, ivar=None, tind=None):
        if self._islinearscan:
            if scannum is None:
                scannum = 0
            self._scannum = scannum
        if ivar is not None:
            self._ivar = ivar
        if tind is None:
            tind = -1
        self._get_parent_parameters()
        self._set_binary_configuration()
        self._set_subclass_path()
        self._read_time_array()
        self._adjust_tind(tind)
        self._get_data()
            
    
#    def _check_linearscan_data(self, scannum=None, ivar=None, tind=None):
#        if scannum is None:
#            scannum = 0
#        if scannum != self._scannum:
#            self._scannum = scannum
#        self._check_data(ivar=ivar, tind=tind)
#        if ivar is not None:
#            self._ivar = ivar
#        if tind is None:
#            tind = -1
#        self._get_parent_parameters()
#        self._set_binary_configuration(
#            nfields=self._nvars,
#            elements=self._ndatapoints,
#            isdouble=self._processed_parameters['PRECISION']=='DOUBLE',
#            isbig=self._processed_parameters['ENDIANNESS']=='BIG')
#        self._set_subclass_path()
#        self._read_time_array()
#        self.tind = self._adjust_tind(tind)
#        self._get_data()

#    def __call__(self, scannum=None, ivar=None, tind=None):
#        """
#        Set tind, scannum, and _ivar
#        Call _load_scan() or _get_data() if needed
#        """
#        do_reload_data = False
#        # check for new data/scan load
#        if (scannum is not None and scannum != self._scannum) or \
#            (not hasattr(self._parent, '_scannum') and not self._scannum):
#            if scannum is not None:
#                self._scannum = scannum
#            else:
#                self._scannum = -1
#            self._load_scan()
#            if tind is None:
#                tind = -1
#            do_reload_data = True
#        if ivar is None and self._ivar is None:
#            ivar = 0
#        if ivar is not None and ivar != self._ivar:
#            # load new moment data
#            # (n/a for fields)
#            self._ivar = ivar
#            do_reload_data = True
#        if tind is None and self.tind is None:
#            tind = -1
#        if tind is not None:
#            tind = self._adjust_tind(tind)
#            if tind != self.tind:
#                self.tind = tind
#                do_reload_data = True
#        # load data/scan, if needed
#        if do_reload_data:
#            self._get_data()
#        return self
            
#    def _load_scan(self):
#        # load scan item configuration based on scannum
#        self._get_parent_parameters()
#        self._set_binary_configuration(
#            nfields=self._nvars,
#            elements=self._ndatapoints,
#            isdouble=self._processed_parameters['PRECISION']=='DOUBLE',
#            isbig=self._processed_parameters['ENDIANNESS']=='BIG')
#        self._set_subclass_path()
#        self._read_time_array()
        
    def _set_subclass_path(self):
        # implement in subclass
        pass
        
    def _get_parent_parameters(self):
        if self._isnonlinear:
            paramsfile = self._parent.path / 'parameters.dat'
        else:
            paramsfile = self._parent.path / 'parameters_{:04d}'.format(self._scannum)
        self._processed_parameters = \
            self._parent._get_processed_parameters(paramsfile=paramsfile)
#        self._process_parameters = self._parent._get_processed_parameters
#        if hasattr(self._parent, '_scannum'):
#            if not isinstance(self._parent._processed_parameters, dict) or \
#                self._scannum != self._parent._processed_parameters.get('_scannum',0):
#                # current scannum does not match parent's archived scannum,
#                # so update processed parameters
#                self._parent._get_processed_parameters(paramsfile=paramsfile, 
#                                                       scannum=self._scannum)
#        else:
#            if not self._parent._processed_parameters or \
#                not isinstance(self._parent._processed_parameters, dict):
#                self._parent._get_processed_parameters(paramsfile=paramsfile)
#        self._processed_parameters = self._parent._processed_parameters
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
        self.kygrid = np.linspace(self.kymin, self.kymin*self.nky0, self.nky0)
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
                data[:,:,:,i] = flatdata.reshape((self.nz0,self.nky0,self.nx0)).transpose()
        data_kxkyz_f = data[:,:,:,-1]
        if self.nky0>1:
            data_xyz = np.square(np.abs(np.fft.ifft2(data_kxkyz_f, axes=[0,1])))
            self._xyimage = data_xyz[:,:,self.nz0//2]
            self._xzimage = np.sum(data_xyz, axis=1)
        else:
            data_kxz_f = data_kxkyz_f[:,0,:]
            self._xyimage = None
            self._xzimage = np.square(2*np.pi*np.abs(np.fft.ifft(data_kxz_f, axis=0)))
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
        diffsig = np.mean(np.abs(self.ballooning[imid+1:] - self.ballooning[imid-1:0:-1]))
        sumsig = np.mean(np.abs(self.ballooning[imid+1:] + self.ballooning[imid-1:0:-1]))
        self.parity = np.mean(np.divide(sumsig-diffsig,sumsig+diffsig))
        taillength = np.int(np.floor(0.025*paragridsize))
        amode = np.abs(self.ballooning)
        tails = np.concatenate((amode[0:taillength+1], amode[-(taillength+1):]))
        tailsize = np.max(tails/amode.max())
        self.tailsize = np.max([tailsize,1e-3])
        realmode = np.real(self.ballooning)
        realmode = realmode / np.sqrt(np.sum(np.abs(realmode)**2))
        wavelet = signal.ricker(5,0.5)
        method = signal.choose_conv_method(realmode,wavelet)
        filtsig = signal.correlate(realmode, wavelet, method=method)
        self.gridosc = np.sum(np.abs(filtsig)**2)

    def plot_mode(self, scannum=None, ivar=None, tind=None):
        self._check_data(scannum=scannum, ivar=ivar, tind=tind)
        if self.tind.size==1:
            plot_title = 't={:.0f}'.format(self.time[self.tind[0]])
        else:
            plot_title+= 't={:.0f}-{:.0f}'.format(self.time[self.tind[0]],
                                               self.time[self.tind[-1]])
        plot_title += ' {}'.format(self._parent.label)
        if self._islinearscan:
            plot_title += '/{:04d}'.format(self._scannum)
        if self.nky0==1:
            # linear sim with nky0=1
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[11,3.25])
        else:
            # nonlinear sim with nky0>1
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=[11,5.5])
        # plot ballooning mode structure in axis #1
        plt.sca(ax.flat[0])
        plt.plot(self.ballgrid, np.abs(self.ballooning), label='Abs()')
        if self._islinearscan:
            plt.plot(self.ballgrid, np.real(self.ballooning), label='Re()')
            plt.legend()
        plt.title(plot_title)
        plt.xlabel('Ballooning angle (rad/pi)')
        plt.ylabel(self.varname)
        xmax = self.ballgrid.max()
        itwopi = 0
        ylim = plt.gca().get_ylim()
        while itwopi<=6 and itwopi<=xmax:
            plt.plot(np.ones(2)*itwopi, ylim, color='tab:gray', linestyle='--')
            if itwopi != 0:
                plt.plot(-np.ones(2)*itwopi, ylim, color='tab:gray', linestyle='--')
            itwopi += 2
        # plot kx spectrum
        plt.sca(ax.flat[1])
        if self.kxgrid.size >= 64:
            style='-'
        else:
            style='d-'
        plt.plot(self.kxgrid[:-1], utils.log1010(self.kxspectrum[:-1]), style)
        plt.xlabel('kx')
        plt.ylabel('PSD('+self.varname+') [dB]')
        plt.title(plot_title)
        # x,z image
        plt.sca(ax.flat[2])
        plt.imshow(self._xzimage.transpose(),
                   aspect='auto',
                   extent=[-self.lx/2, self.lx/2,
                   self.zgrid[0], self.zgrid[-1]],
                   origin='lower',
                   cmap=plt.get_cmap('gnuplot2'),
                   interpolation='bilinear')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(plot_title)
        plt.colorbar()
        plt.tight_layout()
        

class Field(_DataABC):

    def __init__(self, field='', parent=None):
        self._nvars = len(parent.fields)
        self.varname = field
        self.path = None
        super().__init__(ivar=field_names.index(field), parent=parent)
        
    def _set_subclass_path(self):
        if self._islinearscan:
            self.path = self._parent.path / 'field_{:04d}'.format(self._scannum)
        else:
            self.path = self._parent.path / 'field.dat'
            
    def plot_mode(self, scannum=None, tind=None):
        super().plot_mode(scannum=scannum, tind=tind)
        

class Moment(_DataABC):

    def __init__(self, species='', parent=None):
        self._nvars = 6
        self.varname = None
        self.path = None
        self._imoment = None
        self.species = species
        self._set_moment()
        super().__init__(ivar=self._imoment, parent=parent)
        
    def _set_moment(self, moment=0):
        self._imoment = moment
        self.varname = self.species[0:3] + ' ' + mom_names[moment]
        
    def _set_subclass_path(self):
        if self._islinearscan:
            self.path = self._parent.path / 'mom_{}_{:04d}'.format(self.species, self._scannum)
        else:
            self.path = self._parent.path / 'mom_{}.dat'.format(self.species)

    def plot_mode(self, scannum=None, moment=None, tind=None):
        if moment is not None:
            self._set_moment(moment=moment)
        super().plot_mode(scannum=scannum, ivar=moment, tind=tind)
        
