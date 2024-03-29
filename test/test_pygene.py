#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pytest
import pygene as pg

plt.close('all')


@pytest.fixture
def lin():
    simdir = pg.genework / 'linear-v02/eq21/pn60/kyscan01/scanfiles0000'
    return pg.GeneLinearScan(simdir)


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
    return pg.GeneNonlinear(pg.genework / 'nl08/eq21-pn50/run-77460')


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


if __name__ == '__main__':
    pytest.main()
    plt.close('all')
#    pytest.main(['-k','field or moment'])
