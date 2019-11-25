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
        # attribute declarations
        self.scandims = None
        self.nscans = None
        self.params = {}
        self.fields = []
        self.species = []
        self.vsp = None
        self._islinearscan = None
        self._isnonlinear = None
        # get basic scan specs from params file
        try:
            self._paramsfile = utils.validate_path(Path(self.path/'parameters.dat'))
        except:
            self._paramsfile = utils.validate_path(Path(self.path/'parameters'))
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
                        if 'scanlist' in value:
                            value = value.rsplit()[0]
                        try:
                            self.params[pattern] = eval(value)
                        except:
                            self.params[pattern] = value
                        break
            get_value('beta', 0.0)
            get_value('apar', False)
            get_value('bpar', False)
            get_value('nonlinear')
            get_value('nky0')
            get_value('nz0')
            get_value('scan_dims')
            get_value('istep_omega')
            get_value('dt_max')
            f.seek(0)
            for line in f:
                if re.match("^name\s*=", line):
                    match = utils.re_item.match(line)
                    spec_name = eval(match.group('value'))
                    self.species.append(spec_name)
        # assemble fields
        self.fields.append('phi')
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

    def _populate_fields_moments_vsp(self):
        # set attributes for fields and species
        for field in self.fields:
            setattr(self, field, Field(field=field, parent=self))
        for species in self.species:
            try:
                setattr(self, species, Moment(species=species, parent=self))
            except:
                self.species.remove(species)
        self.vsp = Vspace(parent=self)

    def _get_processed_parameters(self, paramsfile=None):
        """
        Used by fields/moments to get post-processed GENE parameters
        from parameters.dat or parameters_0003
        """
        # only called by Field or Moment attributes
        pfile = utils.validate_path(paramsfile)
        params = {'species':[],
                  'Bref':1.0, # T
                  'Tref':1.0, # keV
                  'nref':1.0, # 10**19/m**3
                  'mref':2.0, # amu
                  'Lref':1.0, # m
                  'minor_r':1.0} # in units of Lref
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
        e = 1.6022e-19  # C
#        k_B = 1.3807e-23  # J/K
        proton_mass = 1.6726e-27 # kg
        params['m_kg'] = proton_mass * params['mref']  # ref. mass in kg
        params['T_joules'] = e * 1e3*params['Tref']  # J
        params['c_s'] = np.sqrt(params['T_joules'] / params['m_kg'])  # m/s
        params['omega_ref'] = params['c_s'] / params['Lref']  # rad/s
        params['Omega'] = e*params['Bref'] / params['m_kg']  # rad/s
        params['rho'] = params['c_s'] / params['Omega'] # m
        params['rhostar'] = params['rho'] / params['Lref'] # dimensionless
        # m/s * 10**19/m**3 = 10**19/(m**2 s)
        params['gam_gb'] = params['c_s'] * params['nref'] * params['rhostar']**2
        params['q_gb'] = 1e19*params['gam_gb'] * params['T_joules']  # W/m**2
        return params

    def _calc_curvature(self, file):
        file = utils.validate_path(file)
        miller = np.empty((16,self.params['nz0']))
        with file.open('rt') as f:
            for line in f:
                line = line.rstrip()
                if (line=='') or (line[0]=='&') or (line[0:4]=='magn'):
                    continue
                if line[0]=='/':
                    break
                rx = utils.re_miller1.match(line)
                if rx['name']=='Cxy':
                    cxy = float(rx['value'])
                if rx['name']=='gridpoints':
                    assert(eval(rx['value'])==self.params['nz0'])
            icount = 0
            while True:
                line = f.readline()
                line = line.rstrip()
                if line=='': break
                icount += 1
                rx = utils.re_miller2.match(line)
                miller[:,icount-1] = np.array([float(val) for val in rx.groups()])
            assert(icount==self.params['nz0'])
        gxx = miller[0,:]
        gxy = miller[1,:]
        gxz = miller[2,:]
        gyy = miller[3,:]
        gyz = miller[4,:]
#        gzz = miller[5,:]
        dBdx = miller[7,:]
        dBdy = miller[8,:]
        dBdz = miller[9,:]
        gamma1 = gxx*gyy - gxy**2
        gamma2 = gxx*gyz - gxy*gxz
        gamma3 = gxy*gyz - gyy*gxz
        bcurv = np.empty((2,self.params['nz0']))
        bcurv[0,:] = (-dBdy - gamma2/gamma1 * dBdz) / cxy
        bcurv[1,:] = ( dBdx - gamma3/gamma1 * dBdz) / cxy
#        bcurv[2,:] = ( dBdy + gamma2/gamma1 * dBdx) / cxy
        return bcurv

    def plot_curvature(self, bcurv):
        nz0 = self.params['nz0']
        zgrid = np.linspace(-1, 1-2/nz0, nz0)
        xlabels = ['Kx','Ky']
        plt.figure(figsize=[6.6,2.9])
        for i in range(2):
            plt.subplot(1,2,i+1)
            plt.plot(zgrid, bcurv[i,:])
            plt.axhline(np.mean(bcurv[i,:]), linestyle='--')
            plt.xlabel('z (rad/pi)')
            plt.ylabel(xlabels[i])
            plt.title(self.label)
        plt.tight_layout()

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

    def print_domain(self, scannum=1):
        if self._isnonlinear:
            paramsfile = self.path / 'parameters.dat'
        else:
            paramsfile = self.path / 'parameters_{:04d}'.format(scannum)
        params = self._get_processed_parameters(paramsfile=paramsfile)
        nx0 = params['nx0']
        nky0 = params['nky0']
        kymin = params['kymin']
        lx = params['lx']
        ly = params['ly']
        nexc = params.get('nexc', 0)
        n0_global = params.get('n0_global', 0)
        delkx = 2*np.pi / lx
        kxmax = nx0/2 * delkx
        print('x domain')
        print('  nx0 = {:d}'.format(nx0))
        print('  lx/rho-i = {:.3g}'.format(lx))
        print('  xres/rho-i = {:.3g}'.format(lx/nx0))
        print('  kxres*rho-i = {:.3g}'.format(delkx))
        print('  kxmax*rho-i = {:.3g}'.format(kxmax))
        print('  nexc = {:d}'.format(nexc))
        print('ky domain')
        print('  nky0 = {:d}'.format(nky0))
        print('  ly/rho-i = {:.3g}'.format(ly))
        print('  kymin*rho-i = {:.3g}'.format(kymin))
        if nky0>1: print('  kymax*rho-i = {:.3g}'.format(kymin*nky0))
        print('  n0_global = {:d}'.format(n0_global))
        print('nz0 = {:d}'.format(params['nz0']))
        print('nv0 = {:d}'.format(params['nv0']))
        print('nw0 = {:d}'.format(params['nw0']))


class GeneNonlinear(_GeneBaseClass):
    """
    Nonelinear simulation (no scan)
    """

    def __init__(self, path=None, label=None):
        super().__init__(path=path, label=label)
        self._populate_fields_moments_vsp()
        self.nrg = self._read_nrgdata(self.path / 'nrg.dat')
        self._read_energy(self.path / 'energy.dat')

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
                plt.legend(loc='upper left')
                plt.xlabel('time (a/c_s)')
                plt.title(title)
                if iax==1:
                    plt.ylabel('Gamma_gb, Q_gb')
                    q_gb = self.phi._processed_parameters['q_gb']
                    plt.annotate('Q_gb = {:.1f} kW/m**2'.format(q_gb/1e3),
                                 [0.7,0.7], xycoords='axes fraction')
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

    def plot_curvature(self):
        bcurv = self._calc_curvature(self.path/'miller.dat')
        super().plot_curvature(bcurv)


def concat_nrg(inputs):
    if isinstance(inputs[0], GeneNonlinear):
        sims = inputs
    else:
        sims = [GeneNonlinear(d) for d in inputs]
    sim = sims.pop(0)
    if not isinstance(sim, GeneNonlinear):
        raise ValueError
    species = sim.species
    nrg = sim.nrg.copy()
    for sim in sims:
        nrg['time'] = np.append(nrg['time'], sim.nrg['time'])
        for sp in sim.species:
            for key in sim.nrg[sp].keys():
                nrg[sp][key] = np.concatenate([nrg[sp][key], sim.nrg[sp][key]])
    time = nrg['time']
    t1 = np.searchsorted(time, 1.0)
    fig, ax = plt.subplots(ncols=len(species), nrows=2, figsize=(9,5))
    ax = ax.reshape((2,1))
    for i,sp in enumerate(species):
        for key,value in nrg[sp].items():
            if key.lower().startswith('q') or key.lower().startswith('gam'):
                plt.sca(ax[1,i])
            else:
                plt.sca(ax[0,i])
            label = '{} ({:.1e})'.format(key, value[-1])
            plt.plot(time[t1:], value[t1:], label=label)
        for iax in [0,1]:
            plt.sca(ax[iax,i])
            plt.legend(loc='upper left')
            plt.xlabel('time (a/c_s)')
            if iax==1:
                plt.ylabel('Gamma_gb, Q_gb')
                q_gb = sims[0].phi._processed_parameters['q_gb']
                plt.annotate('Q_gb = {:.1f} kW/m**2'.format(q_gb/1e3),
                             [0.7,0.7], xycoords='axes fraction')
    plt.sca(ax[0,0])
    plt.title(sims[-1].label)
    plt.tight_layout()


class GeneLinearScan(_GeneBaseClass):
    """
    'scanscript' run of GENE linear IV simulations

    Assumes fields and species are identical for all runs.
    Scan can be in 1 or more parameters.
    """

    def __init__(self, path=None, label=None):
        super().__init__(path=path, label=label)

        if not (getattr(self, 'params', None) and getattr(self, 'path', None)):
            raise AttributeError()

        self.omega = None
        self.scanlog = None
        self.scandims = None
        self.nscans = None
        # read scan specifications
        if isinstance(self.params['scan_dims'], str):
            scan_dims_list = self.params['scan_dims'].split(' ')
            scan_dims = np.array([eval(s) for s in scan_dims_list])
            self.scandims = scan_dims.size
            self.nscans = scan_dims.prod()
        elif isinstance(self.params['scan_dims'], int):
            self.scandims = 1
            self.nscans = self.params['scan_dims']
        else: raise ValueError()
        self._populate_fields_moments_vsp()
        self._read_scanlog()
        self._read_omega()

    def _read_scanlog(self):
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

    def _read_omega(self):
        nscans = self.nscans
        output = {'ky':np.empty(nscans)*np.NaN,
                  'omi':np.empty(nscans)*np.NaN,
                  'omr':np.empty(nscans)*np.NaN,
                  'parity':np.empty(nscans)*np.NaN,
                  'tailsize':np.empty(nscans)*np.NaN,
                  'gridosc':np.empty(nscans)*np.NaN,
                  'apar-omi':np.empty(nscans)*np.NaN,
                  'apar-omr':np.empty(nscans)*np.NaN,
                  'apar-parity':np.empty(nscans)*np.NaN,
                  'apar-tailsize':np.empty(nscans)*np.NaN,
                  'apar-gridosc':np.empty(nscans)*np.NaN,
                  'omega-ref':None,
                  'rho-ref':None,
                  }
        for i in np.arange(nscans):
            try:
                self.phi(scannum=i+1)
            except:
                continue
            if i==0:
                output['omega-ref'] = self.phi._processed_parameters['omega_ref']/(2*np.pi)/1e3
                output['rho-ref'] = self.phi._processed_parameters['rho']*1e2
            output['parity'][i] = self.phi.parity2
            output['tailsize'][i] = self.phi.tailsize
            output['gridosc'][i] = self.phi.gridosc
            omega_file = self.path / 'omega_{:04d}'.format(i+1)
            if omega_file.exists():
                with omega_file.open() as f:
                    s = f.readline()
                    match = utils.re_omegafile.match(s)
                    if match and len(match.groups()) == 3:
                        output['ky'][i] = eval(match['ky'])
                        omi = eval(match['omi'])
                        omr = eval(match['omr'])
                    else:
                        print('bad omega file: {}'.format(omega_file.as_posix()))
                        omi = omr = 0
                if omi != 0:
                    if omi>0:
                        output['omi'][i] = omi
                        output['omr'][i] = omr
                else:
                    # calc omi/omr from 'field' output file
                    self.phi(scannum=i+1, tind=[-2,-1])
                    istep = self.phi._processed_parameters['istep_field']
                    if istep<=400:
                        dt = self.phi._processed_parameters['dt_max']
                        data_f2 = self.phi.data[...,1]
                        data_f2[data_f2==0] = 1e-16
                        data_f1 = self.phi.data[...,0]
                        data_f1[data_f1==0] = 1e-16
                        del_field = np.log(data_f2/data_f1) / (dt*istep)
                        weights = np.abs(data_f1)
                        om_av = np.sum(del_field*weights) / weights.sum()
                        if np.real(om_av)>0:
                            output['omi'][i] = np.real(om_av)
                            output['omr'][i] = np.imag(om_av)
            if hasattr(self, 'apar'):
                self.apar(scannum=i+1)
                output['apar-parity'][i] = self.apar.parity2
                output['apar-tailsize'][i] = self.apar.tailsize
                output['apar-gridosc'][i] = self.apar.gridosc
                self.apar(scannum=i+1, tind=[-2,-1])
                istep = self.apar._processed_parameters['istep_field']
                if istep<=400:
                    dt = self.apar._processed_parameters['dt_max']
                    data_f2 = self.apar.data[...,1]
                    data_f2[data_f2==0] = 1e-16
                    data_f1 = self.apar.data[...,0]
                    data_f1[data_f1==0] = 1e-16
                    del_field = np.log(data_f2/data_f1) / (dt*istep)
                    weights = np.abs(data_f1)
                    om_av = np.sum(del_field*weights) / weights.sum()
                    if np.real(om_av)>0:
                        output['apar-omi'][i] = np.real(om_av)
                        output['apar-omr'][i] = np.imag(om_av)
        self.omega = output

    def plot_omega(self, xscale='linear', gammascale='linear', oplot=[],
                   gamma_lim=None, ky_lim=None, oplot_color=[],
                   legend=False, apar=False):
        fig, axes = plt.subplots(nrows=5, figsize=(6,6.75), sharex=True)
        data = self.omega
        if self.scanlog and self.scandims==1:
            if self.scanlog['paramname']=='n0_global':
                xdata = data['ky']
            else:
                xdata = self.scanlog['paramvalues']
        else:
            xdata = np.arange(self.nscans)+1
        axes[0].plot(xdata, data['omi'], '-x', color='C0', label=self.label)
        axes[1].plot(xdata, data['omr'], '-x', color='C0', label=self.label)
        axes[2].plot(xdata, data['parity'], '-x', color='C0', label=self.label)
        axes[3].plot(xdata, data['tailsize'], '-x', color='C0', label=self.label)
        axes[4].plot(xdata, data['gridosc'], '-x', color='C0', label=self.label)
        if apar:
            axes[0].plot(xdata, data['apar-omi'], '--+', color='C0', label=self.label)
            axes[1].plot(xdata, data['apar-omr'], '--+', color='C0', label=self.label)
            axes[2].plot(xdata, data['apar-parity'], '--+', color='C0', label=self.label)
            axes[3].plot(xdata, data['apar-tailsize'], '--+', color='C0', label=self.label)
            axes[4].plot(xdata, data['apar-gridosc'], '--+', color='C0', label=self.label)
        if oplot:
            if not isinstance(oplot, (list,tuple)):
                oplot = [oplot]
            if not isinstance(oplot_color, (list,tuple)):
                oplot_color = [oplot_color]
            if not oplot_color:
                oplot_color = [None for op in oplot]
            for sim,color in zip(oplot,oplot_color):
                if not sim.omega:
                    sim._read_omega()
                data = sim.omega
                if sim.scanlog and sim.scandims==1:
                    if sim.scanlog['paramname']=='n0_global':
                        xdata = data['ky']
                    else:
                        xdata = sim.scanlog['paramvalues']
                else:
                    xdata = np.arange(sim.nscans)+1
                axes[0].plot(xdata, data['omi'], '-x',
                    label=sim.label, color=color)
                axes[1].plot(xdata, data['omr'], '-x',
                    label=sim.label, color=color)
                axes[2].plot(xdata, data['parity'], '-x',
                    label=sim.label, color=color)
                axes[3].plot(xdata, data['tailsize'], '-x',
                    label=sim.label, color=color)
                axes[4].plot(xdata, data['gridosc'], '-x',
                    label=sim.label, color=color)
                if apar:
                    axes[0].plot(xdata, data['apar-omi'], '--+',
                        label=sim.label, color=color)
                    axes[1].plot(xdata, data['apar-omr'], '--+',
                        label=sim.label, color=color)
                    axes[2].plot(xdata, data['apar-parity'], '--+',
                        label=sim.label, color=color)
                    axes[3].plot(xdata, data['apar-tailsize'], '--+',
                        label=sim.label, color=color)
                    axes[4].plot(xdata, data['apar-gridosc'], '--+',
                        label=sim.label, color=color)
        axes[0].set_title('/'.join(self.path.parts[-3:]))
        axes[0].set_ylabel('gamma/(c_s/a)')
        axes[0].set_yscale(gammascale)
        if gammascale=='linear':
            ylim = axes[0].get_ylim()
            axes[0].set_ylim(0,ylim[1]*1.2)
        if gamma_lim:
            axes[0].set_ylim(gamma_lim)
        if ky_lim:
            axes[0].set_xlim(ky_lim)
        axes[1].set_ylabel('omega/(c_s/a)')
        axes[1].set_ylim()
        om_text = 'c_s/a = {:.1f} kHz'.format(self.omega['omega-ref'])
        axes[1].annotate(om_text, [0.7,0.7], xycoords='axes fraction')
        rho_text = 'rho_s = {:.2f} cm'.format(self.omega['rho-ref'])
        axes[1].annotate(rho_text, [0.7,0.5], xycoords='axes fraction')
        axes[2].set_ylim(-1,1)
        axes[2].set_ylabel('parity')
        axes[3].set_yscale('log')
        axes[3].set_ylabel('mode tails')
        axes[3].set_ylim(1e-3,1)
        axes[4].set_yscale('log')
        axes[4].set_ylabel('grid osc.')
        axes[4].set_ylim(1e-2,1)
        if self.scanlog and self.scandims==1:
            axes[-1].set_xlabel(self.scanlog['paramname'])
        else:
            axes[-1].set_xlabel('scan index')
        for ax in axes:
            ax.tick_params('both', reset=True, top=False, right=False)
            ax.set_xscale(xscale)
            if legend: ax.legend()
        fig.tight_layout()

    def plot_nsq(self, species=None, save=False, filename=''):
        if species is None:
            species = self.species[0]
        all_nrg = []
        if not self.scanlog:
            return
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
                ax.plot(time, data['nsq'], label='{}={:.3f},omi={:.3f}'.
                        format(scanparam,scanvalue,self.omega['omi'][i]))
                i += 1
            if len(ax.get_lines())>0:
                ax.legend()
            ax.tick_params('both', reset=True, top=False, right=False)
            ax.set_xlabel('time (c_s/a)')
            ax.set_ylabel('|n|^2')
            ax.set_yscale('log')
            ax.set_title('/'.join(self.path.parts[-3:]))
        fig.tight_layout()

    def plot_curvature(self, scannum=1):
        filename = 'miller_{:04d}'.format(scannum)
        bcurv = self._calc_curvature(self.path/filename)
        super().plot_curvature(bcurv)

    def plot_field_ratios(self):
        nscans = self.nscans
        maxphi = np.empty(nscans)
        maxapar = np.empty(nscans)
        for i in np.arange(nscans):
            self.phi(scannum=i+1)
            maxphi[i] = np.max(np.abs(self.phi.ballooning))
            self.apar(scannum=i+1)
            maxapar[i] = np.max(np.abs(self.apar.ballooning))
        plt.figure()
        plt.plot(self.scanlog['paramvalues'], maxphi/maxapar)
        plt.xlabel(self.scanlog['paramname'])
        plt.ylabel('Ratio Phi/Apar')