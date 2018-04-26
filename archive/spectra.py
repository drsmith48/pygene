#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:01:25 2017

@author: drsmith
"""

import os
import glob
import re
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


pattern = r'^.*\|\s(?P<ky>.*)\s\|\s+(?P<omi>\S*)\s+(?P<omr>\S*).*$'
regex = re.compile(pattern)

def readspectra(file='', plot=False):
    output = {}
    keys = ['ky','omr','omi']
    if not file:
        root = tk.Tk()
        root.withdraw()
        file = fd.askopenfilename(initialdir=os.environ['GENEHOME'],
                                  title='Select scan.log',
                                  filetypes=(('Scan log','scan.log'),))
        root.destroy()
    #if type(file) is not pathlib.Path:
    #    file = pathlib.Path(file)
    with open(file) as f:
        for key in keys:
            output[key] = []
        for line in f:
            if line[0] == '#':
                continue
            match = regex.match(line)
            if match:
                for key in iter(output):
                    value = match.group(key)
                    if value == 'NaN':
                        output[key].append(np.NaN)
                    else:
                        output[key].append(np.float(eval(value)))
    for key in iter(output):
        output[key] = np.asarray(output[key])
    if plot:
        plt.subplot('211')
        plt.plot(output['ky'], output['omr'], '-d')
        plt.ylabel(r'$\omega_r/(c_s/a)$')
        plt.subplot('212')
        plt.plot(output['ky'], output['omi'], '-d')
        plt.ylabel(r'$\omega_i/(c_s/a)$')
        plt.xlabel(r'$k_y\rho$')
        plt.tight_layout()        
    return output



def readpattern(pattern='psinorm60', newfig=False, plot=False,
              skip_pattern='', pattern2='', save=False):
    searchpattern = pattern
    if pattern2:
        searchpattern += '*' + pattern2 + '*'
    probdirs = sorted(glob.glob(os.path.join(os.environ['GENEHOME'], searchpattern)))
    output = {}
    for d in probdirs:
        if skip_pattern and skip_pattern in d:
            continue
        else:
            scanlog = os.path.join(d, 'scanfiles0000/scan.log')
            output[d] = readspectra(file=scanlog)
    if plot:
        #fig, ax = plt.subplots(nrows=2, sharex=True)
        if newfig:
            plt.figure()
        else:
            plt.clf()
        ax = [plt.subplot(211), plt.subplot(212)]
        for key in iter(output):
            spectrum = output[key]
            ax[0].plot(spectrum['ky'], spectrum['omr'], '-d', label=key)
            ax[1].plot(spectrum['ky'], spectrum['omi'], '-d', label=key)
        ax[0].set_xlim(0,0.8)
        ax[0].set_ylim(-20,20)
        ax[0].set_yscale('symlog', linthresy=1.0)
        ax[0].set_ylabel(r'$\omega_r/(c/a)$')
        ax[0].legend(loc='best')
        ax[1].set_xlim(0,0.8)
        ax[1].set_ylim(5e-2,30)
        ax[1].set_yscale('log')
        ax[1].set_ylabel(r'$\omega_i/(c/a)$')
        ax[1].set_xlabel(r'$k_y\rho$')
        plt.tight_layout()
        if pattern2:
            fname = pattern+'_'+pattern2+'.pdf'
        else:
            fname = pattern+'.pdf'
        fullpathname = os.path.join(os.environ['GENETOP'],'results', fname)
        if save:
            plt.savefig(fullpathname)
    return output

def readmultipattern(pattern='', rng=[], **kwargs):
    for r in rng:
        readpattern(pattern=pattern.format(r), 
                    newfig=True, save=True, **kwargs)

def custom():
    patterns = ['psinorm{}0_omtfac?0', 
                'psinorm{}0*gradB', 
                'psinorm{}0*curv']
    colors = ['b','g','r']
    for r in range(3,10):
        plt.figure()
        ax = [plt.subplot(211), plt.subplot(212)]
        for it in zip(patterns,colors):
            pattern = it[0].format(r) + '_eq50'
            output = readpattern(pattern=pattern)
            for key in iter(output):
                spectrum = output[key]
                label = pattern if 'omtfac20' in key else '_'
                ax[0].plot(spectrum['ky'], spectrum['omr'], it[1]+'-d', label=label)
                ax[1].plot(spectrum['ky'], spectrum['omi']/np.abs(spectrum['omr']), it[1]+'-d')
        ax[0].set_xlim(0,0.8)
        ax[0].set_ylim(-20,20)
        ax[0].set_yscale('symlog', linthresy=1.0)
        ax[0].set_ylabel(r'$\omega_r/(c/a)$')
        ax[0].legend(loc='best')
        ax[1].set_xlim(0,0.8)
        ax[1].set_ylim(1e-3,3)
        ax[1].set_yscale('log')
        #ax[1].set_ylabel(r'$\omega_i/(c/a)$')
        ax[1].set_ylabel(r'$\omega_i/|\omega_r|$')
        ax[1].set_xlabel(r'$k_y\rho$')
        plt.tight_layout()
        filename = pattern[0:9]+'-all-dpdx-eq50.pdf'
        filepath = os.path.join(os.environ['GENETOP'],'results', filename)
        plt.savefig(filepath)

if __name__=='__main__':
    #spectrum = readspectra(plot=True)
    #readmultipattern(pattern='psinorm{}0', rng=range(3,10), skip_pattern='dpdx')
    #readmultipattern(pattern='psinorm{}0', rng=range(3,10), pattern2='dpdxgradB')
    #readmultipattern(pattern='psinorm{}0', rng=range(3,10), pattern2='dpdxcurv')
    custom()
