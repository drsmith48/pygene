#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:19:38 2017

@author: drsmith
"""

import os
import csv
import re
import matplotlib.pyplot as plt

file = "scan.log"
repattern = "\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*"
reop = re.compile(repattern)

evspec = {}
with open(file) as cfile:
    creader = csv.DictReader(cfile, 
                             fieldnames=['run','ky','ev1','ev2',
                                         'ev3','ev4','ev5','ev6'],
                             restkey='rest',
                             delimiter='|')
    for i,row in enumerate(creader):
        if i==0: continue # skip header line
        ky = eval(row['ky'])
        evs = []
        for ev in ['ev1','ev2','ev3','ev4','ev5','ev6']:
            if row[ev] is not None and 'NaN' not in row[ev]:
                m = reop.match(row[ev])
                if m:
                    imag = eval(m.group(1))
                    real = eval(m.group(2))
                    evs.append([imag, real])
                else:
                    raise ValueError('  Failed RE {}'.format(row[ev]))
                if ev is 'ev2': break
        if evs:
            evspec[ky] = evs

#fig,ax = plt.subplots(2,sharex=True)
plt.figure(0)
plt.clf()
ax = [plt.subplot(211),plt.subplot(212)]
c = ['b', 'g', 'r', 'c', 'm', 'y']
for ky,evs in evspec.items():
    for i,ev in enumerate(evs):
        for v in [0,1]:
            ax[v].plot(ky, ev[v], marker='*', color=c[i])
ax[0].set_ylabel(r'Im($\omega$)')
ax[1].set_ylabel(r'Re($\omega$)')
for nax in [0,1]:
    ax[nax].axhline(color='k')
    ax[nax].set_xscale('log')
    ax[nax].set_xlabel(r'$k_y\rho_i$')
plt.tight_layout()

#genehome = os.environ['GENEHOME']
#filename = os.path.join(genehome, 'diagnostics','output','ev-spectrum.pdf')
plt.savefig('ev-spectrum.pdf')

