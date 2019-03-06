#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:47:45 2019

@author: drsmith
"""

import pathlib
import matplotlib.pyplot as plt
import pytest
from pygene import GeneLinearScan, GeneNonlinear

plt.close('all')

top_dir = pathlib.Path('/p/gene/drsmith/genecode')

@pytest.fixture
def lin():
    return GeneLinearScan(top_dir/'eq21-pn65-n1'/'scanfiles0000')

def test_linear_scan(lin):
    lin.plot_nsq()
    lin.plot_omega()

def test_linear_field(lin):
    lin.phi.plot_mode()
    lin.apar.plot_mode(scannum=2)
    
def test_linear_moment(lin):
    lin.electrons.plot_mode()
    lin.electrons.plot_mode(scannum=3, moment=5)
    
def test_linear_vsp(lin):
    lin.vsp.plot_vspace()
    lin.vsp.plot_vspace(scannum=4)

@pytest.fixture
def nl():
    return GeneNonlinear(top_dir/'eq21-pn60-nonlinear'/'run-19215555')

def test_nonlinear_timehistory(nl):
    nl.plot_nrg()
    nl.plot_energy()

def test_nonlinear_field(nl):
    nl.phi.plot_mode()
    nl.apar.plot_mode()

def test_nonlinear_moment(nl):
    nl.electrons.plot_mode()
    nl.electrons.plot_mode(moment=5)
    
def test_nonlinear_vsp(nl):
    nl.vsp.plot_vspace()

if __name__=='__main__':
    pytest.main()