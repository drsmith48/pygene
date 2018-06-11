
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from . import utils

field_names = ['phi', 'A_para', 'B_para']
mom_names = ['dens', 'T_para', 'T_perp', 'Q_para', 'Q_perp', 'u_para']
dbmin = -35

def log1010(data):
    return 10*np.log10(data)

class _DataABC(object):

    def __init__(self, path=None, label='', nfields=0, tind=-1, ifield=0,
                 nosingle=False, ky=None):
        # subclasses must define these attributes
        # before calling super().__init__()
        if not hasattr(self, 'params') \
            or not hasattr(self, 'species') \
            or not hasattr(self, 'run') \
            or not hasattr(self, 'fieldnames'):
                raise AttributeError
        self.path, self.shortpath, self.plotlabel, \
            self.filelabel = utils.path_label(path, label)
        self.nfields = nfields
        self.ky = ky
        if not self.nfields or not self.path.is_file():
            raise ValueError
        self._set_binary_format()
        self._read_time_array()
        self._make_grids()
        self.get_data(tind=tind, ifield=ifield, nosingle=nosingle)

    def _set_binary_format(self):
        self.binary = utils.get_binary_config(
                self.nfields,
                self.dims.prod(),
                self.params['PRECISION']=='DOUBLE',
                self.params['ENDIANNESS']=='BIG')

    def _read_time_array(self):
        intsize, entrysize, leapfld, nprt, npct, te, tesize = self.binary
        self.time = np.empty(0)
        with self.path.open('rb') as f:
            filesize = os.path.getsize(self.path.as_posix())
            for i in range(filesize // (leapfld+tesize)):
                value = float(te.unpack(f.read(tesize))[1])
                self.time = np.append(self.time, value)
                f.seek(leapfld,1)

    def _make_grids(self):
        delkx = 2*np.pi / self.params['lx']
        kxmax = self.dims[0]/2 * delkx
        self.kxgrid = np.linspace(-(kxmax-delkx), kxmax, self.dims[0])
        self.kygrid = np.linspace(self.params['kymin'],
                                  self.params['kymin']*self.dims[1],
                                  self.dims[1])
        delz = 2.0*np.pi / self.dims[2]
        self.zgrid = np.linspace(-np.pi, np.pi-delz, self.dims[2])

    def _get_params(self, paramsfile):
        self.params = utils.read_parameters(paramsfile)
        self.dims = np.array([self.params['nx0'],
                              self.params['nky0'],
                              self.params['nz0']])
        self.dtmax = self.params.get('dt_max', 1e-9)
        self.nprocs = self.params.get('n_procs_sim', 0)
        self.wcperstep = self.params.get('step_time', 0.0)
        self.wcperunittimepercpu = self.wcperstep / self.dtmax / self.nprocs

    def _plot_title(self):
        title = self.fieldname+' '+self.plotlabel
        if self.species:
            title += ' {}'.format(self.species[0:4])
        if self.ky:
            title += ' ky={:.2f}'.format(self.ky)
#        title += ' '+self.plotlabel
        if self.tind.size==1:
            timestr = ' t={:.1f}'.format(self.timeslices[0])
        else:
            timestr = ' t={:.1f}-{:.1f}'.format(self.timeslices[0],
                                                self.timeslices[-1])
        title += timestr
        return title

    def get_data(self, tind=-1, ifield=0, nosingle=False):
        if tind is None:
            tind = -1
        if ifield is None:
            ifield = 0
        if ifield >= self.nfields:
            raise ValueError('ifields {} must be < nfields {}'.format(
                    ifield, self.nfields))
        self.ifield = ifield
        if isinstance(tind, (list,tuple)):
            self.tind = np.arange(tind[0], tind[1]+1, 1, dtype=np.int)
        else:
            self.tind = np.array(tind, ndmin=1)
        self.fieldname = self.fieldnames[self.ifield]
        self.tind[self.tind<0] += self.time.size
        self.timeslices = self.time[self.tind]
        intsize, entrysize, leapfld, nprt, npct, te, tesize = self.binary
        data = np.empty((self.dims[0],
                         self.dims[1],
                         self.dims[2],
                         self.tind.size), dtype=npct)
        offset = self.tind * (tesize+leapfld) + \
            self.ifield * (entrysize+2*intsize) + \
            intsize + tesize
        with self.path.open('rb') as f:
#            for i in np.arange(self.tind.size):
            for i,off in enumerate(offset):
                f.seek(off)
                flatdata = np.fromfile(f, count=self.dims.prod(), dtype=npct)
                data[:,:,:,i] = flatdata.reshape(tuple(self.dims[::-1])).transpose()
        # make finite
#        data[data==0] = np.NaN
        # roll in kx dimension to order from kxmin to kxmax
        data = np.roll(data, self.dims[0]//2-1, axis=0)
        if nosingle:
            data = np.squeeze(data)
        # output data
        self.data = data
        self.pdata = np.power(np.absolute(data), 2)
        self.ndata = data / np.sum(np.sqrt(self.pdata))
        self.mode = np.reshape(np.squeeze(self.data[0:-1,:,:,:]),
                               -1, order='C')
        nconnections = self.dims[0]-1
        self.paragrid = np.empty(nconnections*self.dims[2])
        for i in np.arange(nconnections):
            self.paragrid[i*self.dims[2]:(i+1)*self.dims[2]] = \
                2*(i-nconnections//2) + self.zgrid/np.pi
        imid = self.mode.size//2
        diffsig = self.mode[imid+1:] - self.mode[imid-1:0:-1]
        sumsig = self.mode[imid+1:] + self.mode[imid-1:0:-1]
        self.parity = np.mean(np.square(np.abs(sumsig))) / np.mean(np.square(np.abs(diffsig)))

    def plot_mode(self, tind=None, ifield=None):
        if tind is not None or ifield is not None:
            self.get_data(tind=tind, ifield=ifield)
        plt.figure(figsize=(4.6,4))
        plt.plot(self.paragrid, np.absolute(self.mode), label='Abs()')
        plt.plot(self.paragrid, np.real(self.mode), label='Re()')
        plt.plot(self.paragrid, np.imag(self.mode), label='Im()')
        plt.legend()
        plt.title(self._plot_title())
        plt.xlabel('Ballooning angle (rad/pi)')
        plt.ylabel(self.fieldname)

    def plot_spectra(self, tind=None, ifield=None):
        if tind is not None or ifield is not None:
            self.get_data(tind=tind, ifield=ifield)
        data = np.mean(np.abs(self.data),axis=3)
#        title = 'log |'+self.fieldname+'|'
        nky = self.dims[1]
        figsize = [11,5.5]
        if nky>1:
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        # kx spectrum
        plt.sca(ax.flat[0])
        plt.plot(self.kxgrid, log1010(np.mean(data, axis=(1,2))))
        plt.ylim(dbmin,None)
        plt.xlabel('kx')
        title = self._plot_title()
        plt.title(title)
        # kx, z spectrum
        plt.sca(ax.flat[1])
        plt.imshow(log1010(np.mean(data, axis=1)).transpose(),
                       aspect='auto',
                       extent=[self.kxgrid[0], self.kxgrid[-1],
                               self.zgrid[0], self.zgrid[-1]],
                               origin='lower',
                               cmap=mpl.cm.gnuplot,
                               interpolation='bilinear')
        plt.clim(dbmin,0)
        plt.xlabel('kx')
        plt.ylabel('z')
        plt.title(title)
        plt.colorbar()
        if nky>1:
            # ky spectrum
            plt.sca(ax.flat[3])
            plt.plot(self.kygrid, log1010(np.mean(data, axis=(0,2))))
            plt.xlabel('ky')
            plt.ylim(dbmin,None)
            plt.title(title)
            # kx,ky spectrum
            plt.sca(ax.flat[4])
            plt.imshow(log1010(np.mean(data, axis=2)).transpose(),
                       aspect='auto',
                       extent=[self.kxgrid[0], self.kxgrid[-1],
                               self.kygrid[0], self.kygrid[-1]],
                       origin='lower',
                       cmap=mpl.cm.gnuplot,
                       interpolation='bilinear')
            plt.clim(dbmin,0)
            plt.xlabel('kx')
            plt.ylabel('ky')
            plt.title(title)
            plt.colorbar()
            # ky,z spectrum
            plt.sca(ax.flat[5])
            plt.imshow(log1010(np.mean(data, axis=0)).transpose(),
                       aspect='auto',
                       extent=[self.kygrid[0], self.kygrid[-1],
                               self.zgrid[0], self.zgrid[-1]],
                       origin='lower',
                       cmap=mpl.cm.gnuplot,
                       interpolation='bilinear')
            plt.clim(dbmin,0)
            plt.xlabel('ky')
            plt.ylabel('z')
            plt.title(title)
            plt.colorbar()
        plt.tight_layout()


class Moment(_DataABC):

    def __init__(self, path=None, run=None, species='ions', label='',
                 tind=-1, ifield=0, nosingle=False, ky=None):
        path = utils.validate_path(path)
        self.run = run
        self.fieldnames = mom_names
        if species.lower().startswith('ion'):
            self.species = 'ions'
        elif species.lower().startswith('ele'):
            self.species = 'electrons'
        if self.run:
            file = path / 'mom_{}_{:04d}'.format(self.species, self.run)
            paramsfile = path / 'parameters_{:04d}'.format(self.run)
        else:
            file = path / 'mom_{}.dat'.format(self.species)
            paramsfile = path / 'parameters.dat'
        # set self.params before super'ing Data3D.__init__()
        self.params = None
        self._get_params(paramsfile)
        super().__init__(path=file, nfields=self.params['n_moms'],
             label=label, ifield=ifield, tind=tind, nosingle=nosingle, ky=ky)


class Field(_DataABC):

    def __init__(self, path=None, run=None, label='',
                 tind=-1, ifield=0, nosingle=False, ky=None):
        path = utils.validate_path(path)
        self.run = run
        self.species = None
        self.fieldnames = field_names
        if self.run:
            file = path / 'field_{:04d}'.format(self.run)
            paramsfile = path / 'parameters_{:04d}'.format(self.run)
        else:
            file = path / 'field.dat'
            paramsfile = path / 'parameters.dat'
        # set self.params before super'ing Data3D.__init__()
        self.params = None
        self._get_params(paramsfile)
        super().__init__(path=file, nfields=self.params['n_fields'],
             label=label, ifield=ifield, tind=tind, nosingle=nosingle, ky=ky)
