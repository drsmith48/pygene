#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:47:45 2019

@author: drsmith
"""

import pygene
import matplotlib.pyplot as plt

plt.close('all')

sim_linear = pygene.GeneLinearScan(pygene.genehome/'ref03'/'scanfiles0049')

sim_nonlinear = pygene.GeneNonlinear(pygene.genehome/'nlruns/minb12/run01-15694952')

for sim in [sim_linear, sim_nonlinear]:
    print(sim._input_parameters)
    print(sim.species)
    print(sim.fields)