#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:29:28 2018

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt
from . import utils
from .fields import Moment, Field


class _GeneABC(object):

    def __init__(self, path=None, label=None):
        self.path, \
            self.shortpath, \
            self.plotlabel, \
            self.filelabel, \
            self.figtitle = utils.path_label(path, label)
        self.paramsfile = None
        self._set_params_file()  # implement in subclass
        self.params = utils.read_parameters(self.paramsfile)
        self.dims = np.array([self.params['nx0'],
                              self.params['nky0'],
                              self.params['nz0']])
        self.domain = np.array([self.params['lx'], self.params['ly']])
        self.resolution = np.array([self.domain[0]/self.dims[0],
                                    self.domain[1]/self.dims[1]])
        self.kresolution = np.array([2*np.pi/self.domain[0],
                                     2*np.pi/self.domain[1]])
        self.kmax = self.kresolution * self.dims[0:2] *np.array([0.5,1])
        self.isscan = self.params['isscan']
        self.isnonlinear = self.params['isnonlinear']
        self.nspecies = self.params['n_spec']
        self.species = self.params['species']
        self.moment = None
        self.field = None
        self.energy = None
        self.nrg = None

    def _read_nrg(self, file=None):
        '''
        Read data from a single nrg file
        '''
        nsp = self.nspecies
        data = np.empty((0,8,nsp))
        time = np.empty((0,))
        self.nrgdecimate=1
        with file.open() as f:
            for i,line in enumerate(f):
                if i >= 3000:
                    self.nrgdecimate=20
                    break
        with file.open() as f:
            for i,line in enumerate(f):
                itime = i//(nsp+1)
                if itime%self.nrgdecimate != 0:
                    continue
                iline = i%(nsp+1)
                if iline == 0:
                    time = np.append(time, float(line))
                    data = np.append(data, np.empty((1,8,nsp)), axis=0)
                elif iline <= nsp:
                    rx = utils.re_nrgline.match(line)
                    values = [float(val) for val in rx.groups()]
                    data[-1,:,iline-1] = np.array(values)
                else:
                    raise ValueError(str(iline))
        self.nrg = {'time':time,
                    'noutputs':itime+1}
        for i in range(nsp):
            self.nrg[self.species[i]] = {'nsq':data[:,0,i],
                                         'uparsq':data[:,1,i],
                                         'tparsq':data[:,2,i],
                                         'tperpsq':data[:,3,i],
                                         'games':data[:,4,i],
                                         'gamem':data[:,5,i],
                                         'qes':data[:,6,i],
                                         'qem':data[:,7,i]}

    def _set_params_file(self):
        pass
    
    def grid(self):
        pass

    def get_moment(self, *args, **kwargs):
        self.moment = Moment(path=self.path, label=self.plotlabel,
                             *args, **kwargs)

    def get_field(self, *args, **kwargs):
        self.field = Field(path=self.path, label=self.plotlabel,
                           *args, **kwargs)




class GeneLinearScan(_GeneABC):

    def __init__(self, path='ref03/scanfiles0015', label=''):
        super().__init__(path=path, label=label)
        self.scanlog = None
        self.nscans = None
        self.omega = None
        self._read_scanlog()
        self._read_omega()
        if not self.isscan or self.isnonlinear:
            raise ValueError('isscan {}  isnonlinear {}'.
                             format(self.isscan, self.isnonlinear))

    def _set_params_file(self):
        self.paramsfile = self.path/'parameters'

    def _read_scanlog(self):
        scanfile = self.path / 'scan.log'
        scanlog = []
        with scanfile.open() as f:
            for line in f:
                if line.startswith('#Run'):
                    continue
                rx = utils.re_scanlog.match(line)
                rdict = rx.groupdict()
                scanlog.append(float(rdict['ky']))
        self.scanlog = scanlog
        self.nscans = len(scanlog)

    def _read_omega(self):
        output = {'ky':np.empty(0),
                  'omi':np.empty(0),
                  'omr':np.empty(0)}
        for file in sorted(self.path.glob('omega*')):
            with file.open() as f:
                s = f.readline()
                rx = utils.re_omegafile.match(s)
                if not rx or len(rx.groups()) != 3:
                    print('bad data file: ', file)
                    return False
                    #raise ValueError('Failed RegEx: {}'.format(s))
                for key,value in rx.groupdict().items():
                    v = eval(value)
                    if v==0.0 or (key=='omi' and v<0):
                        v = np.NaN
                    output[key] = np.append(output[key], v)
        self.omega = output

    def _read_nrg_scan(self):
        nrgscan = []
        for i,file in enumerate(sorted(self.path.glob('nrg*'))):
            run = int(file.name[-4:])
            self.nrg = utils.read_nrg(self.path / 'nrg_{:04d}'.format(run),
                                species=self.species)
            self.nrg.update({'ky':self.scanlog[i]})
            nrgscan.append(self.nrg)
        return nrgscan

    def grid(self):
        print('nx, ny, nz: {:d}, {:d}, {:d}'.format(*self.dims))
        print('lx, ly domain: {:.2f}, {:.2f}'.format(*self.domain))
        print('xres, yres: {:.3f}, {:.2f}'.format(*self.resolution))
        print('kxmax, kymax: {:.3f}, {:.3f}'.format(*self.kmax))
        print('kxres, kyres: {:.3f}, {:.3f}'.format(*self.kresolution))
        print('kymin*shat*lx: {:.3f}'.format(
                self.kresolution[1]*self.params['shat']*self.domain[0]))
        
    def get_moment(self, run=1, species='ions', *args, **kwargs):
        super().get_moment(run=run, species=species, ky=self.scanlog[run-1], 
             *args, **kwargs)

    def get_field(self, run=1, *args, **kwargs):
        super().get_field(run=run, ky=self.scanlog[run-1],
             *args, **kwargs)

    def print_scanlog(self):
        print('Scanlog for {}:'.format(self.shortpath))
        for i,ky in enumerate(self.scanlog):
            print('  Run {:02d}: ky = {:.3f}'.format(i+1, ky))

#    def plot_kxflux(self, *args, **kwargs):
#        bref = float(self.params.get('Bref', 1.0))
#        ky = self.params['kymin']
#        self.get_moment(ifield=0, nosingle=True, *args, **kwargs)
#        dens = self.moment.data
#        species = self.moment.species
#        self.get_moment(ifield=1, nosingle=True, *args, **kwargs)
#        tpar = self.moment.data
#        self.get_moment(ifield=2, nosingle=True, *args, **kwargs)
#        tperp = self.moment.data
#        self.get_moment(ifield=3, nosingle=True, *args, **kwargs)
#        qpar = self.moment.data
#        self.get_moment(ifield=4, nosingle=True, *args, **kwargs)
#        qperp = self.moment.data
#        self.get_moment(ifield=5, nosingle=True, *args, **kwargs)
#        upar = self.moment.data
#        if kwargs:
#            kwargs.pop('species', None)
#        self.get_field(ifield=0, nosingle=True, *args, **kwargs)
#        phi = self.field.data
#        ve_x = -1j * (ky/bref) * phi
#        self.get_field(ifield=1, nosingle=True, *args, **kwargs)
#        apar = self.field.data
#        b_x = 1j * (ky/bref) * apar
#        dims = self.field.dims
#        nkx = dims[0] # size of kxgrid
#        i0 = nkx//2-1 # index of kx=0 in kxgrid
#        nkxflux = nkx - i0 # size of kx flux grid
#        kxgrid_flux = self.field.kxgrid[i0:]
#        fluxes = np.zeros((nkxflux, dims[2], 4))
#        fnames = ['Gamma_es', 'Gamma_em', 'Q_es', 'Q_em']
#        def calc_kxflux(cdata):
#            output = np.zeros((nkxflux, dims[2]))
#            for i in np.arange(nkxflux):
#                i1 = i0+i
#                i2 = i0-i
#                if i==0 or i==nkx//2:
#                    output[i,:] = 2*np.real(cdata[i1,:])
#                else:
#                    output[i,:] = 2*np.real(cdata[i1,:]+cdata[i2,:])
#            return output
#        # gamma-es
#        fluxes[:,:,0] = calc_kxflux(np.conj(dens) * ve_x)
#        # gamma-em
#        fluxes[:,:,1] = calc_kxflux(np.conj(upar) * b_x)
#        # Q_es = (1/2 Tpar + Tperp + 3/2 n) ve_x * n * T * Qref
#        fluxes[:,:,2] = calc_kxflux(np.conj(0.5*tpar + tperp + 1.5*dens) * ve_x)
#        # Q_em = (qpar + qperp + 5/2 upar) B_x
#        fluxes[:,:,3] = calc_kxflux(np.conj(qpar + qperp + 2.5*upar) * b_x)
#        fig, axes = plt.subplots(2,4, figsize=(11,4.5))
#        # fig.suptitle(self.figtitle, fontweight='bold')
#        for i in np.arange(4):
#            plt.sca(axes[0,i])
#            fmod = fluxes[:,:,i].copy().transpose()
#            fmax = np.abs(fmod).max()
#            thres = fmax * 1e-3
#            fmod[fmod>thres] = 10*np.log10(fmod[fmod>thres])
#            fmod[fmod<-thres] = -10*np.log10(-fmod[fmod<-thres])
#            fmax = np.abs(fmod).max()
#            plt.imshow(fmod,
#                       aspect='auto',
#                       extent=[kxgrid_flux[0],
#                               kxgrid_flux[-1],
#                               self.field.zgrid[0],
#                               self.field.zgrid[-1]],
#                       origin='lower',
#                       cmap=mpl.cm.seismic,
#                       interpolation='bilinear',
#                       vmin=-fmax, vmax=fmax)
#            plt.xlabel('kx')
#            plt.ylabel('z')
#            plt.title('{} {}'.format(species, fnames[i]))
#            plt.colorbar()
#            plt.sca(axes[1,i])
#            plt.plot(kxgrid_flux, np.sum(fluxes[:,:,i],1))
#            plt.yscale('symlog')
#            plt.xlabel('kx')
#            plt.title('{} {}'.format(species, fnames[i]))
#        fig.tight_layout()
#        return fig
#
#    def plot_crossphase(self, run=1, imom=0, species='ion', ifield=0, imom2=None):
#        self.get_moment(run=run, species=species, ifield=imom)
#        field1 = self.moment
#        if imom2:
#            self.get_moment(run=run, species=species, ifield=imom2)
#            field2 = self.moment
#        else:
#            self.get_field(run=run, ifield=ifield)
#            field2 = self.field
#        crosspower = np.multiply(np.squeeze(field1.ndata),
#                                 np.conj(np.squeeze(field2.ndata)))
#        crossphase = np.multiply(np.angle(crosspower),
#                                 np.sqrt(np.absolute(crosspower)/
#                                         np.amax(np.absolute(crosspower))))
#        fig = plt.figure()
#        fig.suptitle(self.figtitle, fontweight='bold')
#        plt.imshow(crossphase.transpose(),
#                   aspect='auto',
#                   extent=[field1.kxgrid[0], field1.kxgrid[-1],
#                           field1.zgrid[0], field1.zgrid[-1]],
#                   origin='lower',
#                   cmap=mpl.cm.gnuplot,
#                   interpolation='bilinear')
#        plt.clim(-np.pi, np.pi)
#        plt.xlabel('kx')
#        plt.ylabel('z')
#        title = ' '.join([field1.fieldname, field2.fieldname, 'crossphase'])
#        plt.title(title)
#        plt.colorbar()

    def plot_omega(self, oplot=[], save=False, format='pdf'):
        fig, axes = plt.subplots(nrows=2, figsize=(7.5,4.25))
        data = self.omega
        filename = 'omega_'+self.filelabel
        axes[0].plot(data['ky'], data['omi'], '-x', label=self.plotlabel)
        for i,x,y in zip(range(self.nscans), data['ky'], data['omi']):
            axes[0].annotate(str(i+1), (x,y),
                xytext=(2,2), textcoords='offset points')
        axes[1].plot(data['ky'], data['omr'], '-x', label=self.plotlabel)
        for i,x,y in zip(range(self.nscans), data['ky'], data['omr']):
            axes[1].annotate(str(i+1), (x,y),
                xytext=(2,2), textcoords='offset points')
        if not isinstance(oplot, (list, tuple)):
            oplot = [oplot]
        for sim in oplot:
            data = sim.omega
            filename += '_'+sim.filelabel
            axes[0].plot(data['ky'], data['omi'], '-x', label=sim.plotlabel)
            for i,x,y in zip(range(sim.nscans), data['ky'], data['omi']):
                axes[0].annotate(str(i+1), (x,y),
                    xytext=(2,2), textcoords='offset points')
            axes[1].plot(data['ky'], data['omr'], '-x', label=sim.plotlabel)
            for i,x,y in zip(range(sim.nscans), data['ky'], data['omr']):
                axes[1].annotate(str(i+1), (x,y),
                    xytext=(2,2), textcoords='offset points')
        axes[0].set_ylabel('gamma/(c_s/a)')
        axes[1].set_ylabel('omega/(c_s/a)')
        for ax in axes:
            ax.set_xlabel('k_y rho_i')
            ax.set_xscale('log')
            ax.set_xlim(0.1,2)
            ax.set_title(self.figtitle)
            ax.legend()
        fig.tight_layout()
        if save:
            savename = filename+'.'+format
            fig.savefig(savename)

    def plot_nsq(self, species='ions'):
        nrgdata = self._read_nrg_scan()
        nruns = len(nrgdata)
        nlines = 4
        nax = nruns//nlines + int(bool(nruns%nlines))
        ncol = nax//2 + nax%2
        fig, axes = plt.subplots(nrows=2, ncols=ncol, figsize=(12,5))
        i = 0
        for ax in axes.flat:
            for j in range(nlines):
                if i >= nruns:
                    break
                nrg = nrgdata[i]
                time = nrg['time']
                data = nrg[species]
                ky = nrg['ky']
                ax.plot(time, data['nsq'], label='ky={}'.format(ky))
                i += 1
            ax.legend()
            ax.set_xlabel('time (c_s/a)')
            ax.set_ylabel('|n|^2')
            ax.set_yscale('log')
            ax.set_title(self.shortpath)
        fig.tight_layout()

    def plot_energy(self, run):
        fig, axes = plt.subplots()
        datafile = self.path / 'energy_{:04d}'.format(run)
        data = utils.read_energy(self.path / 'energy_{:04}'.format(run))
        axes.plot(data['time'], -data['colldiss']/data['grossdrive'], label='coll')
        axes.plot(data['time'], -data['hypzdiss']/data['grossdrive'], label='hypz')
        axes.plot(data['time'], -data['hypvdiss']/data['grossdrive'], label='hypv')
        axes.plot(data['time'], np.abs(data['curvmisc'])/data['grossdrive'], label='misc')
        axes.legend()
        axes.set_xlabel('Time (c_s/a)')
        axes.set_ylabel('Diss. / Drive')
        axes.set_title('/'.join(datafile.parts[-3:]))
        axes.set_yscale('log')
        axes.set_ylim([1e-4,1e0])
        fig.tight_layout()


class GeneNonlinearRun(_GeneABC):

    def __init__(self, path='ref07/run01', label=''):
        super().__init__(path=path, label=label)
        if self.isscan or not self.isnonlinear:
            raise ValueError('isscan {}  isnonlinear {}'.
                             format(self.isscan, self.isnonlinear))
        self._read_nrg()
        self._read_energy()

    def _set_params_file(self):
        self.paramsfile = self.path / 'parameters.dat'

    def _read_energy(self):
        self.energy = utils.read_energy(self.path / 'energy.dat')

    def _read_nrg(self):
        super()._read_nrg(file=self.path/'nrg.dat')

    def grid(self):
        print('nx, ny, nz: {:d}, {:d}, {:d}'.format(*self.dims))
        print('lx, ly domain: {:.2f}, {:.2f}'.format(*self.domain))
        print('xres, yres: {:.3f}, {:.2f}'.format(*self.resolution))
        print('kxmax, kymax: {:.3f}, {:.3f}'.format(*self.kmax))
        print('kxres, kyres: {:.3f}, {:.3f}'.format(*self.kresolution))
        print('kymin*shat*lx: {:.3f}'.format(
                self.kresolution[1]*self.params['shat']*self.domain[0]))
        if not self.nrg:
            self._read_nrg()
        nrgtime = self.nrg['time']
        print('nrg min/max time: {:.3f} - {:.3f}'.format(
                nrgtime[0], nrgtime[-1]))
        print('nrg timesteps: {:d}'.format(nrgtime.size*20))
        
    def get_moment(self, species='ions', tind=-1, ifield=0, nosingle=False):
        super().get_moment(species=species, tind=tind, ifield=ifield,
             nosingle=nosingle)

    def get_field(self, tind=-1, ifield=0, nosingle=False):
        super().get_field(tind=tind, ifield=ifield, nosingle=nosingle)

    def plot_nrg(self, species='ions'):
        if not self.nrg:
            self._read_nrg()
        time = self.nrg['time']
        nsp = len(self.species)
        fig,ax = plt.subplots(ncols=nsp, figsize=(9,3))
        for i,sp in enumerate(self.species):
            nrg = self.nrg[sp]
            for key,value in nrg.items():
                value[0]=0
                value[np.absolute(value)==0] = 1e-12
                label = '{} ({:.1e})'.format(key, value[-1])
                ax[i].plot(time, value, label=label)
            ax[i].set_yscale('symlog', linthreshy=1e-1)
            ax[i].set_title(sp)
            ax[i].legend(loc='upper left')
            ax[i].set_xlabel('time')
        plt.tight_layout()
        plt.show()

    def plot_energy(self):
        if not self.energy:
            self._read_energy()
        edata = self.energy
        time = edata['time']
        plt.figure()
        for key in ['drive','heatsrc','colldiss','hypvdiss','hypzdiss',
                   'nonlinear','curvmisc']:
            label = '{} ({:.1e})'.format(key, edata[key][-1])
            plt.plot(time, edata[key], label=label)
#        plt.yscale('symlog', linthreshy=1)
#        plt.ylim(1e-1,None)
        plt.legend()
        plt.ylabel('energy term')
        plt.xlabel('time')



if __name__=='__main__':
    plt.close('all')
#    minb = GeneLinearScan()
#    minb.plot_omega()
#    minb.plot_nsq()
#    minb.plot_energy(run=4)
    nl = GeneNonlinearRun()
    nl.get_field()
    nl.field.plot_kxky_spectrum()
    nl.plot_nrg()
