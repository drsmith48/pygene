#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:29:28 2018

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from . import utils
from .fields import Moment, Field
from .vsp import Vspace

dbmin = -30

class _GeneABC(object):

    def __init__(self, path=None, label=None):
        self._path = path
        self._label = label
        self._set_path_label()
            
        # set parameter file and read parameters
        self._set_params_file()  # implement in subclass
        self.params = utils.read_parameters(self.paramsfile)
        self.species = self.params['species']
        self.nspecies = self.params['n_spec']
        self.nfields = self.params['n_fields']
        self.nmoments = self.params['n_moms']
        self.moment = None
        self.field = None
        self.vsp = None
        self.energy = None
        self.nrg = None

    def _set_path_label(self):
        self.path = utils.validate_path(self._path)
        self.shortpath = '/'.join(self.path.parts[-2:])
        if self._label:
            self.plotlabel = self._label
            rx = utils.re_prefix.match(self._label)
            self.filelabel = rx.group(1)
        else:
            self.plotlabel = ''
            self.filelabel = self.path.parts[-1]

    def _set_params_file(self):
        # implement in subclass
        pass
    
    def _read_nrg(self, file=None):
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

    def get_moment(self, tind=-1, ivar=0, run=None, species='ions'):
        self.moment = Moment(tind=tind, ivar=ivar, run=run, 
                             species=species, parent=self)

    def get_field(self, tind=-1, ivar=0, run=None):
        self.field = Field(tind=tind, ivar=ivar, run=run, parent=self)
        
    def get_vsp(self, tind=-1, run=None):
        self.vsp = Vspace(tind=tind, run=run, parent=self)


class GeneLinearScan(_GeneABC):

    def __init__(self, path='ref03/scanfiles0015', label=''):
        super().__init__(path=path, label=label)
        self.scanlog = None
        self.nscans = None
        self.omega = None
        self._read_scanlog()
        self._read_omega()

    def _set_params_file(self):
        self.paramsfile = self.path / 'parameters_0001'

    def _read_scanlog(self):
        scanfile = self.path / 'scan.log'
        scanlog = {'paramname':'',
                   'paramvalues':np.empty(0)}
        with scanfile.open() as f:
            for i,line in enumerate(f):
                if i==0:
                    rx = utils.re_scanlogheader.match(line)
                    scanlog['paramname'] = rx.groupdict()['param']
                else:
                    rx = utils.re_scanlog.match(line)
                    value = float(rx.groupdict()['value'])
                    scanlog['paramvalues'] = np.append(scanlog['paramvalues'], value)
        self.scanlog = scanlog

    def _read_omega(self):
        scanparam = self.scanlog['paramname']
        scanvalues = self.scanlog['paramvalues']
        output = {scanparam:np.empty(0),
                  'omi':np.empty(0),
                  'omr':np.empty(0),
                  'phiparity':np.empty(0),
                  'tailsize':np.empty(0),
                  'gridosc':np.empty(0)}
        for i,file in enumerate(sorted(self.path.glob('omega*'))):
            with file.open() as f:
                s = f.readline()
                rx = utils.re_omegafile.match(s)
                if not rx or len(rx.groups()) != 3:
                    print('bad omega file: {}'.format(file.as_posix()))
                    continue
                for key,value in rx.groupdict().items():
                    if key=='ky':
                        continue
                    v = eval(value)
                    if v==0.0 or (key=='omi' and v<0):
                        v = np.NaN
                    output[key] = np.append(output[key], v)
                output[scanparam] = np.append(output[scanparam], 
                                              scanvalues[i])
            self.get_field(run=i+1)
            output['phiparity'] = np.append(output['phiparity'], 
                                         self.field.parity)
            output['tailsize'] = np.append(output['tailsize'], 
                                         self.field.tailsize)
            output['gridosc'] = np.append(output['gridosc'], 
                                         self.field.gridosc)
        self.nscans = len(output[scanparam])
        self.omega = output

    def _read_nrg_scan(self):
        nrgscan = []
        scanparam = self.scanlog['paramname']
        scanvalues = self.scanlog['paramvalues']
        for i,file in enumerate(sorted(self.path.glob('nrg*'))):
            run = int(file.name[-4:])
            self.nrg = utils.read_nrg(self.path / 'nrg_{:04d}'.format(run),
                                species=self.species)
            self.nrg.update({scanparam:scanvalues[i]})
            nrgscan.append(self.nrg)
        return nrgscan

    def overview(self):
        self.grid()
        self.plot_omega()
        for i in np.arange(1,self.nscans+1,3):
            self.get_field(run=i)
            self.field.plot_mode()
        parity = np.empty(0)
        wc = np.empty(0)
        dtmax = np.empty(0)
        wcperstep = np.empty(0)
        for i in range(self.nscans):
            self.get_field(run=i+1)
            parity = np.append(parity, self.field.parity)
            wc = np.append(wc, self.field.wcperunittimepercpu)
            dtmax = np.append(dtmax, self.field.dtmax)
            wcperstep = np.append(wcperstep, 
                                  self.field.wcperstep/self.field.nprocs)
        f,ax = plt.subplots(nrows=2,ncols=2)
        ax.flat[0].semilogy(parity, '-x', label=self.plotlabel)
        ax.flat[0].set_ylabel('parity')
        ax.flat[1].semilogy(wc, '-x', label=self.plotlabel)
        ax.flat[1].set_ylabel(r'wc/tau/cpu')
        ax.flat[2].semilogy(dtmax, '-x', label=self.plotlabel)
        ax.flat[2].set_ylabel(r'tau/step')
        ax.flat[3].plot(wcperstep, '-x', label=self.plotlabel)
        ax.flat[3].set_ylabel(r'wc/step/cpu')
        for a in ax.flat:
            a.set_xlabel('run number')
            a.legend()
        
#    def get_moment(self, run=1, **kwargs):
#        super().get_moment(run=run, **kwargs)
#
#    def get_field(self, run=1, **kwargs):
#        super().get_field(run=run, **kwargs)
        
    def plot_omega(self, xscale='linear', gammascale='linear', 
                   filename='', oplot=[], save=False, index=False):
        fig, axes = plt.subplots(nrows=5, figsize=(6,6.75), sharex=True)
        data = self.omega
        scanparam = self.scanlog['paramname']
        filename_auto = 'omega_'+self.filelabel
        if index:
            xdata = np.arange(self.nscans)+1
            axes[0].plot(xdata, data['omi'], '-x', label=self.plotlabel)
            axes[1].plot(xdata, data['omr'], '-x', label=self.plotlabel)
            axes[2].plot(xdata, data['phiparity'], '-x', label=self.plotlabel)
            axes[3].plot(xdata, data['tailsize'], '-x', label=self.plotlabel)
            axes[4].plot(xdata, data['gridosc'], '-x', label=self.plotlabel)
        else:
            axes[0].plot(data[scanparam], data['omi'], '-x', label=self.plotlabel)
            axes[1].plot(data[scanparam], data['omr'], '-x', label=self.plotlabel)
            axes[2].plot(data[scanparam], data['phiparity'], '-x', label=self.plotlabel)
            axes[3].plot(data[scanparam], data['tailsize'], '-x', label=self.plotlabel)
            axes[4].plot(data[scanparam], data['gridosc'], '-x', label=self.plotlabel)
            for iax,key in enumerate(['omi','omr','phiparity','tailsize','gridosc']):
                for i,x,y in zip(range(self.nscans), data[scanparam], data[key]):
                    axes[iax].annotate(str(i+1), (x,y),
                        xytext=(2,2), textcoords='offset points')
        if not isinstance(oplot, (list, tuple)):
            oplot = [oplot]
        for sim in oplot:
            data = sim.omega
            filename_auto += '_'+sim.filelabel
            if index:
                simxdata = np.arange(sim.nscans)+1
                axes[0].plot(simxdata, data['omi'], '-x', label=sim.plotlabel)
                axes[1].plot(simxdata, data['omr'], '-x', label=sim.plotlabel)
                axes[2].plot(simxdata, data['phiparity'], '-x', label=sim.plotlabel)
                axes[3].plot(simxdata, data['tailsize'], '-x', label=sim.plotlabel)
                axes[4].plot(simxdata, data['gridosc'], '-x', label=sim.plotlabel)
            else:
                axes[0].plot(data[scanparam], data['omi'], '-x', label=sim.plotlabel)
                axes[1].plot(data[scanparam], data['omr'], '-x', label=sim.plotlabel)
                axes[2].plot(data[scanparam], data['phiparity'], '-x', label=sim.plotlabel)
                axes[3].plot(data[scanparam], data['tailsize'], '-x', label=sim.plotlabel)
                axes[4].plot(data[scanparam], data['gridosc'], '-x', label=sim.plotlabel)
                for iax,key in enumerate(['omi','omr','phiparity','tailsize','gridosc']):
                    for i,x,y in zip(range(sim.nscans), data[scanparam], data[key]):
                        axes[iax].annotate(str(i+1), (x,y),
                            xytext=(2,2), textcoords='offset points')
        axes[0].set_title(self.shortpath)
        axes[0].set_ylabel('gamma/(c_s/a)')
        axes[0].set_yscale(gammascale)
        if gammascale=='linear':
            axes[0].set_ylim(0,None)
        axes[1].set_ylabel('omega/(c_s/a)')
        axes[1].set_ylim()
        axes[2].set_ylim(-1,1)
        axes[2].set_ylabel('phi parity')
        axes[3].set_yscale('log')
        axes[3].set_ylabel('tail size')
        axes[4].set_yscale('log')
        axes[4].set_ylabel('grid osc.')
        if index:
            axes[-1].set_xlabel('index')
        else:
            axes[-1].set_xlabel(scanparam)
        for ax in axes:
            ax.tick_params('both', reset=True, top=False, right=False)
            ax.set_xscale(xscale)
            if len(ax.get_lines())>=2:
                ax.legend()
        fig.tight_layout()
        if save:
            if not filename:
                filename = filename_auto
            fig.savefig(filename+'.pdf')

    def plot_nsq(self, species='ions', save=False, filename=''):
        nrgdata = self._read_nrg_scan()
        scanparam = self.scanlog['paramname']
        nruns = len(nrgdata)
        nlines = 4
        nax = nruns//nlines + int(bool(nruns%nlines))
        ncol = nax//2 + nax%2
        fig, axes = plt.subplots(nrows=2, ncols=ncol, figsize=(12,5), 
                                 sharex=True, sharey=True)
        i = 0
        for ax in axes.flat:
            for j in range(nlines):
                if i >= nruns:
                    break
                nrg = nrgdata[i]
                time = nrg['time']
                data = nrg[species]
                scanvalue = nrg[scanparam]
                ax.plot(time, data['nsq'], label='{}={}'.format(scanparam,scanvalue))
                i += 1
            if len(ax.get_lines())>0:
                ax.legend()
            ax.tick_params('both', reset=True, top=False, right=False)
            ax.set_xlabel('time (c_s/a)')
            ax.set_ylabel('|n|^2')
            ax.set_yscale('log')
            ax.set_title(self.shortpath)
        fig.tight_layout()
        filename_auto = 'nsq_'+self.filelabel
        if save:
            if not filename:
                filename = filename_auto
            fig.savefig(filename+'.pdf')

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
        
    def plot_modes(self, runs=None):
        if runs is not None and not isinstance(runs,(list,tuple,np.ndarray)):
            runs = [runs]
        if runs is None:
            runs = np.arange(self.nscans)
        for i in runs:
            self.get_field(run=i)
            self.field.plot_mode()


class GeneNonlinearRun(_GeneABC):

    def __init__(self, path='ref07/run01', label=''):
        super().__init__(path=path, label=label)
        self._calc_grid()
        self._read_nrg()
        self._read_energy()
        self.lx = self.params['lx']
        self.ly = self.params['ly']

    def _set_params_file(self):
        self.paramsfile = self.path / 'parameters.dat'

    def _read_energy(self):
        self.energy = utils.read_energy(self.path / 'energy.dat')

    def _read_nrg(self):
        super()._read_nrg(file=self.path/'nrg.dat')
        
    def _calc_grid(self):
        self.dims = np.array([self.params['nx0'],
                              self.params['nky0'],
                              self.params['nz0'],
                              self.params['nv0'],
                              self.params['nw0']])
        self.domain = np.array([self.params['lx'], self.params['ly']])
        self.resolution = np.array([self.domain[0]/self.dims[0],
                                    self.domain[1]/self.dims[1]])
        self.kresolution = np.array([2*np.pi/self.domain[0],
                                     2*np.pi/self.domain[1]])
        self.kmax = self.kresolution * self.dims[0:2] *np.array([0.5,1])

#    def grid(self):
#        super().grid()
#        print('lx, ly domain: {:.2f}, {:.2f}'.format(*self.domain))
#        print('xres, yres: {:.3f}, {:.2f}'.format(*self.resolution))
#        print('kxmax, kymax: {:.3f}, {:.3f}'.format(*self.kmax))
#        print('kxres, kyres: {:.3f}, {:.3f}'.format(*self.kresolution))
#        print('kymin*shat*lx: {:.3f}'.format(
#                self.kresolution[1]*self.params['shat']*self.domain[0]))
#        if not self.nrg:
#            self._read_nrg()
#        nrgtime = self.nrg['time']
#        print('nrg min/max time: {:.3f} - {:.3f}'.format(
#                nrgtime[0], nrgtime[-1]))
#        print('nrg timesteps: {:d}'.format(nrgtime.size*20))
        
    def plot_kxky(self, species='ions'):
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=[9,5])
        tind = [-3,-2,-1]
        for ifield in [0,1]:
            self.get_field(ivar=ifield, tind=tind)
            data = np.mean(np.abs(self.field.data),axis=3)
            plt.sca(ax.flat[0])
            plt.plot(self.field.kygrid, 
                     utils.log1010(np.mean(data, axis=(0,2))),
                     label=self.field.varname)
            plt.xlabel('ky')
            plt.sca(ax.flat[3])
            plt.plot(self.field.kxgrid, 
                     utils.log1010(np.mean(data, axis=(1,2))),
                     label=self.field.varname)
            plt.xlabel('kx')
        for imom in [0,1,2]:
            self.get_moment(ivar=imom, tind=tind, species=species)
            data = np.mean(np.abs(self.moment.data),axis=3)
            plt.sca(ax.flat[1])
            plt.plot(self.moment.kygrid, 
                     utils.log1010(np.mean(data, axis=(0,2))),
                     label='{} {}'.format(self.moment.species[0:4],
                            self.moment.varname))
            plt.xlabel('ky')
            plt.sca(ax.flat[4])
            plt.plot(self.moment.kxgrid, 
                     utils.log1010(np.mean(data, axis=(1,2))),
                     label='{} {}'.format(self.moment.species[0:4],
                            self.moment.varname))
            plt.xlabel('kx')
        for imom in [5,3,4]:
            self.get_moment(ivar=imom, tind=tind, species=species)
            data = np.mean(np.abs(self.moment.data),axis=3)
            plt.sca(ax.flat[2])
            plt.plot(self.moment.kygrid, 
                     utils.log1010(np.mean(data, axis=(0,2))),
                     label='{} {}'.format(self.moment.species[0:4],
                            self.moment.varname))
            plt.xlabel('ky')
            plt.sca(ax.flat[5])
            plt.plot(self.moment.kxgrid, 
                     utils.log1010(np.mean(data, axis=(1,2))),
                     label='{} {}'.format(self.moment.species[0:4],
                            self.moment.varname))
            plt.xlabel('kx')
        for axx in ax.flat:
            axx.set_ylim(dbmin,None)
            axx.set_title(self.shortpath)
            axx.legend()
        plt.tight_layout()
        
    def plot_xyimages(self, species='ions'):
        ncols = np.int(np.ceil((self.nfields+self.nmoments)/2.0))
        fig, ax = plt.subplots(nrows=2, 
                               ncols=ncols,
                               figsize=[12, 6])
        iax=0
        for i in range(self.nfields+self.nmoments):
            plt.sca(ax.flat[iax])
            if i<self.nfields:
                self.get_field(ivar=i)
                var = self.field
                title = var.varname + ' (z=0)'
            else:
                self.get_moment(ivar=i-self.nfields,
                                species=species)
                var = self.moment
                title = var.species[0:3] + ' ' + var.varname + ' (z=0)'
            plt.imshow(var.xyimage.transpose(),
                       origin='lower',
                       cmap=mpl.cm.seismic,
                       extent=[-self.lx/2, self.lx/2,
                               -self.ly/2, self.ly/2],
                       interpolation='bilinear',
                       aspect='equal')
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(title)
            iax += 1
        plt.tight_layout()

    def plot_nrg(self):
        if not self.nrg:
            self._read_nrg()
        time = self.nrg['time']
        t1 = np.searchsorted(time, 1.0)
        fig, ax = plt.subplots(ncols=self.nspecies, nrows=2, figsize=(9,5))
        for i,sp in enumerate(self.species):
            nrg = self.nrg[sp]
            for key,value in nrg.items():
                if key.lower().startswith('q') or key.lower().startswith('gam'):
                    plt.sca(ax.flat[i+2])
                else:
                    plt.sca(ax.flat[i])
                label = '{} ({:.1e})'.format(key, value[-1])
                plt.plot(time[t1:], value[t1:], label=label)
            for iax in [i,i+2]:
                plt.sca(ax.flat[iax])
                plt.title(sp)
                plt.legend(loc='upper left')
                plt.xlabel('time')
        plt.tight_layout()

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
        plt.ylabel('energy term')
        plt.xlabel('time')
        plt.legend()


if __name__=='__main__':
    plt.close('all')
    minb = GeneLinearScan()
#    minb.plot_omega()
#    minb.plot_nsq()
#    minb.plot_energy(run=4)
#    nl = GeneNonlinearRun()
#    nl.get_field()
#    nl.field.plot_kxky_spectrum()
#    nl.plot_nrg()
