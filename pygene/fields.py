
import os
import struct
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
try:
    from . import utils
except:
    import utils

field_names = ['phi', 'A_para', 'B_para']
mom_names = ['dens', 'T_para', 'T_perp', 'Q_para', 'Q_perp', 'u_para']
dbmin = -30

class _DataABC(object):

    def __init__(self, tind=-1, ivar=0):
        # subclasses must define these attributes
        # before calling super().__init__()
        if not (hasattr(self, 'species') and hasattr(self, 'run') and \
                hasattr(self, 'nvars') and hasattr(self, 'parent') and \
                hasattr(self, 'path') and hasattr(self, 'varnames')):
            raise AttributeError
        self.tind = tind
        self.ivar = ivar
        self.varname = self.varnames[self.ivar]
        self.shortpath = '/'.join(self.path.parts[-2:])
        self.plotlabel = self.parent.plotlabel
        self.filelabel = self.parent.filelabel
        self._read_paramsfile()
        self._make_grids()
        self._set_binary_configuration(
            nfields=self.nvars,
            elements=self.dims.prod(),
            isdouble=self.params['PRECISION']=='DOUBLE',
            isbig=self.params['ENDIANNESS']=='BIG')
        self.get_data()

    def _set_binary_configuration(self, nfields=None, elements=None, 
                                  isdouble=None, isbig=None):
        realsize = 4 + 4 * isdouble
        complexsize = 2*realsize
        intsize = 4
        entrysize = elements * complexsize
        leapfld = nfields * (entrysize+2*intsize)
#        print('*** Fields ***')
#        print('complex size', complexsize)
#        print('elements', elements)
#        print('entry size', entrysize)
#        print('nfields', nfields)
#        print('leapfld', leapfld)
        if isbig:
            nprt=(np.dtype(np.float64)).newbyteorder()
            npct=(np.dtype(np.complex128)).newbyteorder()
            fmt = '>idi'
        else:
            nprt=np.dtype(np.float64)
            npct=np.dtype(np.complex128)
            fmt = '=idi'
        te = struct.Struct(fmt)
        tesize = te.size
        self.binary_configuration = (intsize, entrysize, leapfld, 
                                     nprt, npct, te, tesize)

    def _read_paramsfile(self):
        if self.run:
            paramsfile = self.parent.path / 'parameters_{:04d}'.format(self.run)
            self.params = self.parent._read_parameters(paramsfile)
        else:
            self.params = self.parent.params
        self.dims = np.array([self.params['nx0'],
                              self.params['nky0'],
                              self.params['nz0']])
        self.dtmax = self.params.get('dt_max', 1e-9)
        self.nprocs = self.params.get('n_procs_sim', 0)
        self.wcperstep = self.params.get('step_time', 0.0)
        self.wcperunittimepercpu = self.wcperstep / self.dtmax / self.nprocs
        
    def _make_grids(self):
        delkx = 2*np.pi / self.params['lx']
        kxmax = self.dims[0]/2 * delkx
        self.kxgrid = np.linspace(-(kxmax-delkx), kxmax, self.dims[0])
        self.kygrid = np.linspace(self.params['kymin'],
                                  self.params['kymin']*self.dims[1],
                                  self.dims[1])
        delz = 2.0*np.pi / self.dims[2]
        self.zgrid = np.linspace(-np.pi, np.pi-delz, self.dims[2])

    def _read_time_array(self):
        intsize, entrysize, leapfld, nprt, npct, te, tesize = self.binary_configuration
        self.time = np.empty(0)
        with self.path.open('rb') as f:
            filesize = os.path.getsize(self.path.as_posix())
            for i in range(filesize // (leapfld+tesize)):
                value = float(te.unpack(f.read(tesize))[1])
                self.time = np.append(self.time, value)
                f.seek(leapfld,1)

    def _set_plot_title(self):
        title = self.varname+' '+self.plotlabel
        if self.species:
            title += ' {}'.format(self.species[0:4])
        if self.run:
            title += ' run {}'.format(self.run)
        if self.tind.size==1:
            timestr = ' t={:.0f}'.format(self.timeslices[0])
        else:
            timestr = ' t={:.0f}-{:.0f}'.format(self.timeslices[0],
                                                self.timeslices[-1])
        title += timestr
        self.plot_title = title

    def get_data(self, tind=None, ivar=None):
        if tind is not None:
            self.tind = tind
        if ivar is not None:
            self.ivar = ivar
        if isinstance(self.tind, (tuple,list,np.ndarray)):
            self.tind = np.asarray(self.tind, dtype=np.int)
        else:
            self.tind = np.asarray([self.tind], dtype=np.int)
        self.varname = self.varnames[self.ivar]
        self._read_time_array()
        self.tind[self.tind<0] += self.time.size
        self.timeslices = self.time[self.tind]
        self._set_plot_title()
        intsize, entrysize, leapfld, nprt, npct, te, tesize = self.binary_configuration
        data = np.empty((self.dims[0],
                         self.dims[1],
                         self.dims[2],
                         self.tind.size), dtype=npct)
        offset = self.tind * (tesize+leapfld) + \
            self.ivar * (entrysize+2*intsize) + \
            intsize + tesize
        with self.path.open('rb') as f:
            for i,off in enumerate(offset):
                f.seek(off)
                flatdata = np.fromfile(f, count=self.dims.prod(), dtype=npct)
                data[:,:,:,i] = flatdata.reshape(tuple(self.dims[::-1])).transpose()
        if self.dims[1]>1:
            nzmid, = np.nonzero(self.zgrid==0)
            dataz0 = np.squeeze(data[:,:,nzmid,-1])
            self.xyimage = np.real(np.fft.ifft2(dataz0))
        else:
            self.xyimage = None
        # roll in kx dimension to order from kxmin to kxmax
        data = np.roll(data, self.dims[0]//2-1, axis=0)
        # output data
        self.data = data
        # rehsape to ballooning representation
        self.ballooning = np.reshape(np.squeeze(self.data[0:-1,:,:,:]), -1, order='C')
        nconnections = self.dims[0]-1
        paragridsize = nconnections*self.dims[2]
        self.ballgrid = np.empty(paragridsize)
        for i in np.arange(nconnections):
            self.ballgrid[i*self.dims[2]:(i+1)*self.dims[2]] = \
                2*(i-nconnections//2) + self.zgrid/np.pi
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

    def plot_mode(self, tind=None, ivar=None, filename='', save=False):
        if tind is not None or ivar is not None:
            self.get_data(tind=tind, ivar=ivar)
        plt.figure(figsize=(4.6,4))
        plt.plot(self.ballgrid, np.abs(self.ballooning), label='Abs()')
        plt.plot(self.ballgrid, np.real(self.ballooning), label='Re()')
        plt.legend()
        plt.title(self.plot_title)
        plt.xlabel('Ballooning angle (rad/pi)')
        plt.ylabel(self.varname)
        plt.tight_layout()
        xmax = self.ballgrid.max()
        itwopi = 0
        ylim = plt.gca().get_ylim()
        while itwopi<=6 and itwopi<=xmax:
            plt.plot(np.ones(2)*itwopi, ylim, color='tab:gray', linestyle='--')
            if itwopi != 0:
                plt.plot(-np.ones(2)*itwopi, ylim, color='tab:gray', linestyle='--')
            itwopi += 2
        filename_auto = self.filelabel+'_'+self.varname+'_mode'
        if self.run:
            filename_auto += '_r{:02d}'.format(self.run)
        if save:
            if not filename:
                filename = filename_auto
            plt.gcf().savefig(filename+'.pdf')
        
    def plot_spectra(self, tind=None, ivar=None, filename='', save=False):
        self.get_data(tind=tind, ivar=ivar)
        # average data over time axis
        data = np.mean(np.abs(self.data),axis=3)
        nky = self.dims[1]
        figsize = [11,5.5]
        if nky>1:
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        # kx spectrum
        plt.sca(ax.flat[0])
        plt.plot(self.kxgrid, utils.log1010(np.mean(data, axis=(1,2))))
        plt.ylim(dbmin,None)
        plt.xlabel('kx')
        plt.title(self.plot_title)
        # kx, z spectrum
        plt.sca(ax.flat[1])
        plt.imshow(utils.log1010(np.mean(data, axis=1)).transpose(),
                       aspect='auto',
                       extent=[self.kxgrid[0], self.kxgrid[-1],
                               self.zgrid[0], self.zgrid[-1]],
                               origin='lower',
                               cmap=mpl.cm.gnuplot,
                               interpolation='bilinear')
        plt.clim(dbmin,0)
        plt.xlabel('kx')
        plt.ylabel('z')
        plt.title(self.plot_title)
        plt.colorbar()
        if nky>1:
            # ky spectrum
            plt.sca(ax.flat[3])
            plt.plot(self.kygrid, utils.log1010(np.mean(data, axis=(0,2))))
            plt.xlabel('ky')
            plt.ylim(dbmin,None)
            plt.title(self.plot_title)
            # kx,ky spectrum
            plt.sca(ax.flat[4])
            plt.imshow(utils.log1010(np.mean(data, axis=2)).transpose(),
                       aspect='auto',
                       extent=[self.kxgrid[0], self.kxgrid[-1],
                               self.kygrid[0], self.kygrid[-1]],
                       origin='lower',
                       cmap=mpl.cm.gnuplot,
                       interpolation='bilinear')
            plt.clim(dbmin,0)
            plt.xlabel('kx')
            plt.ylabel('ky')
            plt.title(self.plot_title)
            plt.colorbar()
            # x,y image
            lx = self.params['lx']
            ly = self.params['ly']
            plt.sca(ax.flat[2])
            plt.imshow(self.xyimage.transpose(),
                       origin='lower',
                       cmap=mpl.cm.seismic,
                       extent=[-lx/2,lx/2,-ly/2,ly/2],
                       interpolation='bilinear',
                       aspect='equal')
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(self.plot_title + ' z=0')
            # ky,z spectrum
            plt.sca(ax.flat[5])
            plt.imshow(utils.log1010(np.mean(data, axis=0)).transpose(),
                       aspect='auto',
                       extent=[self.kygrid[0], self.kygrid[-1],
                               self.zgrid[0], self.zgrid[-1]],
                       origin='lower',
                       cmap=mpl.cm.gnuplot,
                       interpolation='bilinear')
            plt.clim(dbmin,0)
            plt.xlabel('ky')
            plt.ylabel('z')
            plt.title(self.plot_title)
            plt.colorbar()
        plt.tight_layout()
        filename_auto = self.filelabel+'_'+self.varname+'_spectra'
        if self.run:
            filename_auto += '_r{:02d}'.format(self.run)
        if save:
            if not filename:
                filename = filename_auto
            plt.gcf().savefig(filename+'.pdf')


class Moment(_DataABC):

    def __init__(self, run=None, parent=None, species='ions', **kwargs):
        self.run = run
        self.varnames = mom_names
        self.parent = parent
        self.nvars = self.parent.nmoments
        if species.lower().startswith('ion'):
            self.species = 'ions'
        elif species.lower().startswith('ele'):
            self.species = 'electrons'
        if self.run:
            self.path = self.parent.path / 'mom_{}_{:04d}'.format(self.species, self.run)
        else:
            self.path = self.parent.path / 'mom_{}.dat'.format(self.species)
        super().__init__(**kwargs)


class Field(_DataABC):

    def __init__(self, run=None, parent=None, **kwargs):
        self.run = run
        self.varnames = field_names
        self.parent = parent
        self.nvars = self.parent.nfields
        self.species = None
        if self.run:
            self.path = self.parent.path / 'field_{:04d}'.format(self.run)
        else:
            self.path = self.parent.path / 'field.dat'
        super().__init__(**kwargs)
