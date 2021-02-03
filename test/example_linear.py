#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:26:46 2020

@author: drsmith
"""
import matplotlib.pyplot as plt
from pathlib import Path
import pygene

plt.close('all')

topdir = pygene.genework / 'linear-v02/eq21/pn50/kyscan01/scanfiles0000'
sim = pygene.GeneLinearScan(topdir)
sim.plot_nsq()
sim.plot_omega()
sim.plot_omega(all_plots=True)

sim.phi.plot_mode(scannum=1)
