from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import super
from builtins import open
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

from .fields_moments import Moment, Field
from .vsp import Vspace
from . import utils

root = tk.Tk()
root.withdraw()

genework = utils.genework


class _GeneBaseClass(object):

    def __init__(self, path=None, label=None):
        # set path and label
        if not path:
            path = filedialog.askdirectory(initialdir=genework.as_posix())
            path = Path(path)
        self.path = utils.validate_path(path)
        self.label = label if label else '/'.join(self.path.parts[-2:])
        # check for 'parameters' file
        try:
            self._paramsfile = utils.validate_path(Path(self.path/'parameters'))
        except:
            self._paramsfile = utils.validate_path(Path(self.path/'parameters.dat'))
        # attribute declarations
        self.scandims = None
        self.nscans = None
        # get basic simulation parameters
        self.params = {}
        self.species = []
        with open(self._paramsfile) as f:
            def get_value(pattern, default=None):
                f.seek(0)
                self.params[pattern] = default
                for line in f:
                    if re.match("^"+pattern+"\s*=",line):
                        match = utils.re_item.match(line)
                        value = match.group('value').lower()
                        if value in ['t','.t.','true','.true.']:
                            value = 'True'
                        if value in ['f','.f.','false','.false.']:
                            value = 'False'
                        try:
                            self.params[pattern] = eval(value)
                        except:
                            self.params[pattern] = value
                        break
            get_value('n_spec')
            get_value('beta', 0.0)
            get_value('apar', False)
            get_value('bpar', False)
            get_value('nonlinear')
            get_value('nky0')
            get_value('scan_dims')
            f.seek(0)
            for line in f:
                if re.match("^name\s*=", line):
                    match = utils.re_item.match(line)
                    spec_name = eval(match.group('value'))
                    self.species.append(spec_name)
                    if len(self.species) == self.params['n_spec']:
                        break
        # assemble fields
        self.fields = ['phi']
        if self.params['beta']:
            self.fields.append('apar')
            self.params['apar'] = True
            if self.params['bpar']:
                self.fields.append('bpar')
        # check for linear scan or nonlinear nonscan
        self._isnonlinear = self.params['nonlinear'] is True and \
                           self.params['nky0']>1 and \
                           self.params['scan_dims'] is None
        self._islinearscan = self.params['nonlinear'] is False and \
                            self.params['nky0']==1 and \
                            self.params['scan_dims']
        assert self._isnonlinear != self._islinearscan
        # set attributes for fields and species
        for field in self.fields:
            setattr(self, field, Field(field=field, parent=self))
        for species in self.species:
            setattr(self, species, Moment(species=species, parent=self))
        self.vsp = Vspace(parent=self)

    def _get_processed_parameters(self, paramsfile=None):
        """
        Used by fields/moments to get post-processed GENE parameters 
        from parameters.dat or parameters_0003
        """
        # only called by Field or Moment attributes
        pfile = utils.validate_path(paramsfile)
        params = {'species':[],
                  'Bref':1.0,
                  'Tref':1.0,
                  'nref':1.0,
                  'mref':2.0,
                  'Lref':1.0,
                  'minor_r':1.0}
        with pfile.open('r') as f:
            for line in f:
                if utils.re_amp.match(line) or \
                    utils.re_slash.match(line) or \
                    utils.re_comment.match(line) or \
                    utils.re_whitespace.match(line):
                        continue
                rx = utils.re_item.match(line)
                if not rx:
                    continue
                key = rx.group('key')
                value = rx.group('value')
                if key == 'name':
                    params['species'].append(eval(value))
                    continue
                if value.lower() in ['t','.t.','true','.true.']:
                    value = 'True'
                if value.lower() in ['f','.f.','false','.false.']:
                    value = 'False'
                try:
                    params[key] = eval(value)
                except:
                    params[key] = value
        e = 1.6022e-19 # C
        k = 1.3807e-23
        proton_mass = 1.6726e-27 # kg
        params['m_kg'] = proton_mass * params['mref'] # ref. mass
        params['T_joules'] = e * 1e3*params['Tref'] # J
        params['c_s'] = np.sqrt(params['T_joules'] / params['m_kg'])
        params['Omega'] = e*params['Bref'] / params['m_kg']
        params['rho'] = params['c_s'] / params['Omega']
        params['rhostar'] = params['rho'] / params['Lref']
        params['gam_gb'] = params['c_s'] * params['nref'] * params['rhostar']**2
        params['q_gb'] = params['gam_gb'] * 1e3*params['Tref'] * e/k
        return params

    def _read_nrgdata(self, file):
        """
        Read a NRG file
        """
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
            nrgdata[self.species[i]] = \
                {'nsq':data[:,0,i],
                 'uparsq':data[:,1,i],
                 'tparsq':data[:,2,i],
                 'tperpsq':data[:,3,i],
                 'games':data[:,4,i],
                 'gamem':data[:,5,i],
                 'qes':data[:,6,i],
                 'qem':data[:,7,i]}
        return nrgdata


class GeneNonlinear(_GeneBaseClass):
    """
    Nonelinear simulation (no scan)
    """
    
    def __init__(self, path=None, label=None):
        super().__init__(path=path, label=label)
        self.nrg = None
        self.energy = None

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
        self.energy = {'time':data[:,0],
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
    
    def plot_nrg(self):
        if not self.nrg:
            self.nrg = self._read_nrgdata(self.path / 'nrg.dat')
        time = self.nrg['time']
        t1 = np.searchsorted(time, 1.0)
        fig, ax = plt.subplots(ncols=len(self.species), nrows=2, figsize=(9,5))
        ax = ax.reshape((2,1))
        for i,sp in enumerate(self.species):
            nrg = self.nrg[sp]
            title = '{} {}'.format(self.label, sp)
            for key,value in nrg.items():
                if key.lower().startswith('q') or key.lower().startswith('gam'):
                    plt.sca(ax[1,i])
                else:
                    plt.sca(ax[0,i])
                label = '{} ({:.1e})'.format(key, value[-1])
                plt.plot(time[t1:], value[t1:], label=label)
            for iax in [0,1]:
                plt.sca(ax[iax,i])
                plt.title(title)
                plt.legend(loc='upper left')
                plt.xlabel('time')
        plt.tight_layout()

    def plot_energy(self):
        if not self.energy:
            self._read_energy(self.path / 'energy.dat')
        time = self.energy['time']
        plt.figure()
        for key in ['drive','heatsrc','colldiss','hypvdiss','hypzdiss',
                   'nonlinear','curvmisc']:
            label = '{} ({:.1e})'.format(key, self.energy[key][-1])
            plt.plot(time, self.energy[key], label=label)
        plt.ylabel('energy term')
        plt.xlabel('time')
        plt.title(self.label)
        plt.legend()


class GeneLinearScan(_GeneBaseClass):
    """
    'scanscript' run of GENE linear IV simulations
    
    Assumes fields and species are identical for all runs.
    Scan can be in 1 or more parameters.
    """
    
    def __init__(self, path=None, label=None):
        super().__init__(path=path, label=label)

        self.omega = {}
        if isinstance(self.params['scan_dims'], str):
            dim_scans = [eval(s) for s in self.params['scan_dims'].split(' ')]
            self.scandims = len(dim_scans)
            self.nscans = np.prod(np.array(dim_scans))
        elif isinstance(self.params['scan_dims'], int):
            self.scandimes = 1
            self.nscans = self.params['scan_dims']
        else:
            raise ValueError('scan_dims is invalid: {}'.
                             format(self.params['scan_dims']))

        scanfile = self.path / 'scan.log'
        paramname = ''
        paramvalues = np.empty(0)
        with scanfile.open() as f:
            for i,line in enumerate(f):
                if i==0:
                    rx = utils.re_scanlogheader.match(line)
                    paramname = rx.group('param')
                else:
                    rx = utils.re_scanlog.match(line)
                    value = float(rx.group('value'))
                    paramvalues = np.append(paramvalues, value)
        if paramvalues.size != 0:
            self.scanlog = {'paramname':paramname, 
                            'paramvalues':paramvalues}
        else:
            self.scanlog = None

    def _read_omega(self):
        nscans = self.nscans
        output = {'ky':np.empty(nscans)*np.NaN,
                  'omi':np.empty(nscans)*np.NaN,
                  'omr':np.empty(nscans)*np.NaN,
                  'phiparity':np.empty(nscans)*np.NaN,
                  'tailsize':np.empty(nscans)*np.NaN,
                  'gridosc':np.empty(nscans)*np.NaN}
        for i in np.arange(nscans):
            omega_file = self.path / 'omega_{:04d}'.format(i+1)
            if not omega_file.exists():
                print('missing omega file: {}'.format(omega_file.as_posix()))
                continue
            self.phi._check_data(scannum=i+1)
            output['phiparity'][i] = self.phi.parity
            output['tailsize'][i] = self.phi.tailsize
            output['gridosc'][i] = self.phi.gridosc
            with omega_file.open() as f:
                s = f.readline()
                match = utils.re_omegafile.match(s)
                if not match or len(match.groups()) != 3:
                    print('bad omega file: {}'.format(omega_file.as_posix()))
                    continue
                output['ky'][i] = eval(match['ky'])
                omi = eval(match['omi'])
                if omi > 0:
                    output['omi'][i] = omi
                    output['omr'][i] = eval(match['omr'])
        self.omega = output

    def plot_omega(self, xscale='linear', gammascale='linear', oplot=[]):
        if not self.omega:
            self._read_omega()
        fig, axes = plt.subplots(nrows=5, figsize=(6,6.75), sharex=True)
        data = self.omega
        if self.scanlog and self.scandims==1:
            xdata = self.scanlog['paramvalues']
        else:
            xdata = np.arange(self.nscans)+1
        axes[0].plot(xdata, data['omi'], '-x', label=self.label)
        axes[1].plot(xdata, data['omr'], '-x', label=self.label)
        axes[2].plot(xdata, data['phiparity'], '-x', label=self.label)
        axes[3].plot(xdata, data['tailsize'], '-x', label=self.label)
        axes[4].plot(xdata, data['gridosc'], '-x', label=self.label)
        if self.scanlog and self.scandims==1:
            for iax,key in enumerate(['omi','omr','phiparity','tailsize','gridosc']):
                for i,x,y in zip(self.scans, xdata, data[key]):
                    axes[iax].annotate(str(i), (x,y),
                        xytext=(2,2), textcoords='offset points')
        axes[0].set_title(self.label)
        axes[0].set_ylabel('gamma/(c_s/a)')
        axes[0].set_yscale(gammascale)
        if gammascale=='linear':
            ylim = axes[0].get_ylim()
            axes[0].set_ylim(0,ylim[1]*1.2)
        axes[1].set_ylabel('omega/(c_s/a)')
        axes[1].set_ylim()
        axes[2].set_ylim(-1,1)
        axes[2].set_ylabel('phi parity')
        axes[3].set_yscale('log')
        axes[3].set_ylabel('phi tails')
        axes[3].set_ylim(1e-3,1)
        axes[4].set_yscale('log')
        axes[4].set_ylabel('phi osc.')
        axes[4].set_ylim(1e-2,1)
        if self.scanlog and self.scandims==1:
            axes[-1].set_xlabel(self.scanlog['paramname'])
        else:
            axes[-1].set_xlabel('scan index')
        for ax in axes:
            ax.tick_params('both', reset=True, top=False, right=False)
            ax.set_xscale(xscale)
            if len(ax.get_lines())>=2:
                ax.legend()
        fig.tight_layout()

    def plot_nsq(self, species=None, save=False, filename=''):
        if species is None:
            species = self.species[0]
        all_nrg = []
        scanparam = self.scanlog['paramname']
        scanvalues = self.scanlog['paramvalues']
        for i,file in enumerate(sorted(self.path.glob('nrg*'))):
            run = int(file.name[-4:])
            nrgfile = self.path / 'nrg_{:04d}'.format(run)
            single_nrg = self._read_nrgdata(nrgfile)
            single_nrg.update({scanparam:scanvalues[i]})
            all_nrg.append(single_nrg)
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
            ax.set_title(self.label)
        fig.tight_layout()
