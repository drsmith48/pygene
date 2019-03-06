#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:47:45 2019

@author: drsmith
"""

import pathlib
import matplotlib.pyplot as plt
from pygene import GeneLinearScan, GeneNonlinear

plt.close('all')

top_dir = pathlib.Path('/p/gene/drsmith/genecode')

def test_linear():
    path = top_dir/'eq21-pn65-n1'/'scanfiles0000'
    sim = GeneLinearScan(path=path)
    sim.plot_nsq()
    sim.plot_omega()
    sim.phi.plot_mode(scannum=1)
    sim.electrons.plot_mode(scannum=1, moment=5)
    sim.vsp.plot_vspace(scannum=1)

def test_nonlinear():
    path = top_dir/'eq21-pn60-nonlinear'/'run-19215555'
    nl = GeneNonlinear(path=path)
    nl.plot_nrg()
    nl.plot_energy()
    nl.phi.plot_mode()
    nl.electrons.plot_mode(moment=5)

    
if __name__=='__main__':
    test_linear()
    test_nonlinear()
