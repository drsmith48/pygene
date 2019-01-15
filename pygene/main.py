from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import int
from builtins import zip
from builtins import range
from builtins import str
from future import standard_library
standard_library.install_aliases()

from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .fields_moments import Moment, Field
from .vsp import Vspace
from . import utils

root = tk.Tk()
root.withdraw()

genehome = utils.genehome

class GeneBaseClass(object):

    def __init__(self, path=None, label=None):
        
        if not path:
            path = filedialog.askdirectory(initialdir=genehome.as_posix())
            path = Path(path)

        self._paramsfile = ''
        self._input_parameters = {}
        self.species = []
        self.fields = []

        self._set_path_label(path, label)
        self._set_params_file()

        self._get_species_fields()
#        self.species = self._input_parameters['species']
#        self.nfields = self._input_parameters['n_fields']
#        self.nmoments = self._input_parameters['n_moms']
#        self._calc_refs()

#        self.moment = None
#        self.field = None
#        self.vsp = None
        
#        self._continue_init()
        
    def _set_path_label(self, path, label):
        self.path = utils.validate_path(path)
        self.shortpath = '/'.join(self.path.parts[-2:])
        if label:
            self.plotlabel = label
            rx = utils.re_prefix.match(label)
            self.filelabel = rx.group(1)
        else:
            self.plotlabel = ''
            self.filelabel = self.path.parts[-1]

    def _set_params_file(self):
        # implement in subclass
        pass
    
    def _continue_init(self):
        # implement in subclass
        pass

    def _get_species_fields(self):
        with open(self._paramsfile) as f:
            def get_value(pattern, default=None):
                f.seek(0)
                self._input_parameters[pattern] = default
                for line in f:
                    if re.match("^"+pattern+"\s*=",line):
                        match = utils.re_item.match(line)
                        value = match.group('value')
                        if value.lower() in ['t','.t.','true','.true.']:
                            self._input_parameters[pattern] = True
                            break
                        if value.lower() in ['f','.f.','false','.false.']:
                            self._input_parameters[pattern] = False
                            break
                        self._input_parameters[pattern] = eval(match.group('value'))
                        break
            get_value('n_spec')
            get_value('beta', 0.0)
            get_value('bpar', False)
            get_value('nonlinear')
            self.fields.append('phi')
            if self._input_parameters['beta']:
                self.fields.append('apar')
                if self._input_parameters['bpar']:
                    self.fields.append('bpar')
            for field in self.fields:
                setattr(self, field, None)
            f.seek(0)
            for line in f:
                if re.match("^name\s*=", line):
                    match = utils.re_item.match(line)
                    spec_name = eval(match.group('value'))
                    self.species.append(spec_name)
                    if len(self.species) == self._input_parameters['n_spec']:
                        break
            for species in self.species:
                setattr(self, species, None)

    def _read_parameters(self):
        file_obj = utils.validate_path(self._paramsfile)
        params = {'isscan':False,
                  'species':[],
                  'lx':0,
                  'ly':0,
                  'Bref':1.0,
                  'Tref':1.0,
                  'nref':1.0,
                  'mref':2.0,
                  'Lref':1.0,
                  'minor_r':1.0}
        with file_obj.open('r') as f:
            for line in f:
                if utils.re_amp.match(line) or \
                    utils.re_slash.match(line) or \
                    utils.re_comment.match(line) or \
                    utils.re_whitespace.match(line):
                        continue
                rx = utils.re_item.match(line)
                if not rx:
                    continue
                if utils.re_scan.search(line):
                    params['isscan'] = True
                d = rx.groupdict()
                if d['key'].lower() == 'name':
                    params['species'].append(eval(d['value']))
                if d['value'].lower() in ['true', 't', '.t.']:
                    d['value'] = 'True'
                if d['value'].lower() in ['false', 'f', '.f.']:
                    d['value'] = 'False'
                if d['key'].islower():
                    d['value'] = eval(d['value'])
                params[d['key']] = d['value']
        params['isnonlinear'] = params['nonlinear']==True
        self._input_parameters = params

    def _read_nrgdata(self, file):
        nsp = len(self.species)
        data = np.empty((0,8,nsp))
        time = np.empty((0,))
        decimate=1
        with file.open() as f:
            for i,line in enumerate(f):
                if i >= 3000:
                    decimate=20
                    break
        with file.open() as f:
            for i,line in enumerate(f):
                itime = i//(nsp+1)
                if itime%decimate != 0:
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
        nrgdata = {'time':time}
        for i in range(nsp):
            nrgdata[self.species[i]] = {'nsq':data[:,0,i],
                                        'uparsq':data[:,1,i],
                                        'tparsq':data[:,2,i],
                                        'tperpsq':data[:,3,i],
                                        'games':data[:,4,i],
                                        'gamem':data[:,5,i],
                                        'qes':data[:,6,i],
                                        'qem':data[:,7,i]}
        return nrgdata

    def _read_energy(self, file):
        data = np.empty((0,14))
        with file.open() as f:
            for i,line in enumerate(f):
                if i%20 != 0 or i<=14:
                    continue
                rx = utils.re_energy.match(line)
                linedata = np.empty((1,14))
                for i,s in enumerate(rx.groups()):
                    if s == 'NaN':
                        linedata[0,i] = np.NaN
                    else:
                        linedata[0,i] = eval(s)
                data = np.append(data, linedata, axis=0)
        return {'time':data[:,0],
                'etot':data[:,1],
                'ddtetot':data[:,2],
                'drive':data[:,3],
                'heatsrc':data[:,4],
                'colldiss':data[:,5],
                'hypvdiss':data[:,6],
                'hypzdiss':data[:,7],
                'nonlinear':data[:,9],
                'curvmisc':data[:,11],
                'convergance':data[:,12],
                'deltae':data[:,13],
                'netdrive':np.sum(data[:,3:12],axis=1),
                'grossdrive':np.sum(data[:,3:5],axis=1)}
    
    def _convert(self, value):
        if isinstance(value, str):
            return eval(value)
        else:
            return value

    def _calc_refs(self):
        e = 1.6022e-19 # C
        k = 1.3807e-23
        pro_mass = 1.6726e-27 # kg
        self.ref = {}
        self.ref['b'] = self._convert(self._input_parameters['Bref']) # T
        self.ref['t_ev'] = 1e3 * self._convert(self._input_parameters['Tref']) # keV -> eV
        self.ref['t_joules'] = e * self.ref['t_ev'] # J
        self.ref['n'] = self._convert(self._input_parameters['nref']) # 1e19/m**3
        self.ref['l'] = self._convert(self._input_parameters['Lref']) # m
        self.ref['m'] = pro_mass * self._convert(self._input_parameters['mref']) # ref. mass
        self.ref['c'] = np.sqrt(self.ref['t_joules'] / self.ref['m'])
        self.ref['Omega'] = e*self.ref['b'] / self.ref['m']
        self.ref['rho'] = self.ref['c'] / self.ref['Omega']
        self.ref['rhostar'] = self.ref['rho'] / self.ref['l']
        self.ref['gam_gb'] = self.ref['c'] * self.ref['n'] * self.ref['rhostar']**2
        self.ref['q_gb'] = self.ref['gam_gb'] * self.ref['t_ev'] * e/k

    def get_moment(self, tind=-1, ivar=0, run=None, species='ions'):
        self.moment = Moment(tind=tind, ivar=ivar, run=run, 
                             species=species, parent=self)

    def get_field(self, tind=-1, ivar=0, run=None):
        self.field = Field(tind=tind, ivar=ivar, run=run, parent=self)
        
    def get_vsp(self, tind=-1, run=None):
        self.vsp = Vspace(tind=tind, run=run, parent=self)


class GeneLinearScan(GeneBaseClass):

    def _continue_init(self):
        self._read_scanlog()
        self._read_omega()

    def _set_params_file(self):
        self._paramsfile = self.path / 'parameters_0001'

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
        if scanlog['paramvalues'].size == 0:
            scanlog['paramname'] = 'kymin'
        self.scanlog = scanlog

    def _read_omega(self):
        scanvalues = self.scanlog['paramvalues']
        scanparam = self.scanlog['paramname']
        output = {scanparam:np.empty(0),
                  'ky':np.empty(0),
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
#                    if key=='ky':
#                        continue
                    v = eval(value)
                    if v==0.0 or (key=='omi' and v<0):
                        v = np.NaN
                    output[key] = np.append(output[key], v)
                if i < scanvalues.size:
                    output[scanparam] = np.append(output[scanparam], 
                                                  scanvalues[i])
                else:
                    output[scanparam] = np.append(output[scanparam],
                                                  output['ky'][-1])
            self.get_field(run=i+1)
            output['phiparity'] = np.append(output['phiparity'], 
                                         self.field.parity)
            output['tailsize'] = np.append(output['tailsize'], 
                                         self.field.tailsize)
            output['gridosc'] = np.append(output['gridosc'], 
                                         self.field.gridosc)
        self.nscans = len(output[scanparam])
        self.omega = output
        
    def plot_kx(self, run=1):
        fig, axes = plt.subplots(nrows=self.nspecies, ncols=3, sharex=True)
        

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
#        axes[3].set_yscale('log')
        axes[3].set_ylabel('tail size')
#        axes[4].set_yscale('log')
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
        all_nrg = []
        scanparam = self.scanlog['paramname']
        scanvalues = self.scanlog['paramvalues']
        for i,file in enumerate(sorted(self.path.glob('nrg*'))):
            run = int(file.name[-4:])
            nrgfile = self.path / 'nrg_{:04d}'.format(run)
            single_nrg = self._read_nrgdata(nrgfile)
            single_nrg.update({scanparam:scanvalues[i]})
            all_nrg.append(single_nrg)
        scanparam = self.scanlog['paramname']
        nruns = len(all_nrg)
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
                nrg = all_nrg[i]
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
        datafile = self.path / 'energy_{:04d}'.format(run)
        energydata = self._read_energy(datafile)
        plt.figure()
        plt.plot(energydata['time'], -energydata['colldiss']/energydata['grossdrive'], label='coll')
        plt.plot(energydata['time'], -energydata['hypzdiss']/energydata['grossdrive'], label='hypz')
        plt.plot(energydata['time'], -energydata['hypvdiss']/energydata['grossdrive'], label='hypv')
        plt.plot(energydata['time'], np.abs(energydata['curvmisc'])/energydata['grossdrive'], label='misc')
        plt.legend()
        plt.xlabel('Time (c_s/a)')
        plt.ylabel('Diss. / Drive')
        plt.title('/'.join(datafile.parts[-3:]))
        plt.yscale('log')
        plt.ylim([1e-4,1e0])
        plt.tight_layout()
        

class GeneNonlinear(GeneBaseClass):

    def _continue_init(self):
        self._calc_grid()
        self.nrg = self._read_nrgdata(self.path / 'nrg.dat')
        self.energy = self._read_energy(self.path / 'energy.dat')

    def _set_params_file(self):
        self._paramsfile = self.path / 'parameters.dat'

    def _calc_grid(self):
        self.dims = np.array([self._input_parameters['nx0'],
                              self._input_parameters['nky0'],
                              self._input_parameters['nz0'],
                              self._input_parameters['nv0'],
                              self._input_parameters['nw0']])
        self.domain = np.array([self._input_parameters['lx'], 
                                self._input_parameters['ly']])
        self.resolution = np.array([self.domain[0]/self.dims[0],
                                    self.domain[1]/self.dims[1]])
        self.kresolution = np.array([2*np.pi/self.domain[0],
                                     2*np.pi/self.domain[1]])
        self.kmax = self.kresolution * self.dims[0:2] * np.array([0.5,1])

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
            axx.set_ylim(-30,None)
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
                       extent=[-self.domain[0]/2, self.domain[0]/2,
                               -self.domain[1]/2, self.domain[1]/2],
                       interpolation='bilinear',
                       aspect='equal')
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(title)
            iax += 1
        plt.tight_layout()

    def plot_nrg(self):
        time = self.nrg['time']
        t1 = np.searchsorted(time, 1.0)
        fig, ax = plt.subplots(ncols=len(self.species), nrows=2, figsize=(9,5))
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
        time = self.energy['time']
        plt.figure()
        for key in ['drive','heatsrc','colldiss','hypvdiss','hypzdiss',
                   'nonlinear','curvmisc']:
            label = '{} ({:.1e})'.format(key, self.energy[key][-1])
            plt.plot(time, self.energy[key], label=label)
        plt.ylabel('energy term')
        plt.xlabel('time')
        plt.legend()


def concat_nrg(path='', prefix='', xscale='linear', yscale='linear'):
    path = Path(path)
    rundirs = sorted(path.glob(prefix+'*'))
    nrgs = []
    for rundir in rundirs:
        sim=GeneNonlinear(rundir)
        nrgs.append(sim.nrg)
        species = sim.species
    for i,nrg in enumerate(nrgs):
        if i==0:
            newnrg = nrg
        else:
            newnrg['time'] = np.concatenate((newnrg['time'], 
                                             nrg['time'][1:]))
            for sp in species:
                nrg_sp = nrg[sp]
                for key in nrg_sp:
                    newnrg[sp][key] = np.concatenate((newnrg[sp][key], 
                                                      nrg_sp[key][1:]))
    t1 = np.searchsorted(newnrg['time'], 1.0)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(9,5))
    for key in newnrg[species[0]]:
        if key.lower().startswith('t'):
            plt.sca(ax[0,1])
        elif key.lower().startswith('gam'):
            plt.sca(ax[1,0])
        elif key.lower().startswith('q'):
            plt.sca(ax[1,1])
        else:
            plt.sca(ax[0,0])
        for sp in species:
            label = '{} {}'.format(sp, key)
            plt.plot(newnrg['time'][t1:], newnrg[sp][key][t1:], label=label)
    for axes in ax.flat:
        axes.set_xlabel('time')
        axes.legend()
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        if yscale is 'log':
            axes.set_ylim(1e-1,None)
    plt.tight_layout()