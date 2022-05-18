#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:01:25 2017

@author: drsmith
"""

import os
import pathlib
import re
import tkinter as tk
import tkinter.filedialog as fd
import numpy as np
import matplotlib.pyplot as plt


pattern = r'^.*\|\s(?P<ky>.*)\s\|\s+(?P<omi>\S*)\s+(?P<omr>\S*).*$'
regex = re.compile(pattern)

def readscan(file=None, plot=False):
    output = {}
    keys = ['ky','omr','omi']
    if not file:
        root = tk.Tk()
        root.withdraw()
        file = fd.askopenfilename(initialdir=os.environ['GENEHOME'],
                                  title='Select scan.log',
                                  filetypes=(('Scan log','scan.log'),))
        root.destroy()
    if type(file) is not pathlib.Path:
        file = pathlib.Path(file)
    with file.open() as f:
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



def batchscan(pattern='psinorm60', plot=False, newfig=False, 
              skip_pattern=None, pattern2=''):
    gene = pathlib.Path(os.environ['GENEHOME'])
    if pattern2:
        probdirs = sorted(gene.glob('*'+pattern+'*'+pattern2+'*'))
    else:
        probdirs = sorted(gene.glob('*'+pattern+'*'))
    output = {}
    for d in probdirs:
        if skip_pattern and skip_pattern in d.name:
            continue
        else:
            scanlog = d / 'scanfiles0000/scan.log'
            output[d.name] = readscan(file=scanlog)
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
        ax[0].set_ylim(-20,20)
        ax[0].set_yscale('symlog', linthresy=1.0)
        ax[1].set_ylim(5e-2,30)
        ax[1].set_yscale('log')
        ax[0].set_ylabel(r'$\omega_r/(c/a)$')
        ax[1].set_ylabel(r'$\omega_i/(c/a)$')
        ax[1].set_xlabel(r'$k_y\rho$')
        ax[0].legend(loc='best')
        plt.tight_layout()
        if pattern2:
            fname = pattern+'_'+pattern2+'.pdf'
        else:
            fname = pattern+'.pdf'
        fullpathname = os.path.join(os.environ['GENETOP'],'results', fname)
        plt.savefig(fullpathname)
    return output

def batchbatchscan(pattern=None, rng=None, **kwargs):
    for r in rng:
        batchscan(pattern=pattern.format(r), plot=True, newfig=True, **kwargs)

if __name__=='__main__':
    #spectrum = readscan(plot=True)
    #batchbatchscan(pattern='psinorm{}0', rng=range(3,10), skip_pattern='dpdx')
    #batchbatchscan(pattern='psinorm{}0', rng=range(3,10), pattern2='dpdxgradB')
    batchbatchscan(pattern='psinorm{}0', rng=range(3,10), pattern2='dpdxcurv')
