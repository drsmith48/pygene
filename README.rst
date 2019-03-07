.. highlight:: python

PYGENE
=============

Simple tools to analyze GENE output (http://genecode.org/)

Runs on Python 2/3 with numpy, scipy, matplotlib

Usage
-------------

Linear scan analysis:

.. code-block:: python

  from pygene import GeneLinearScan
  scan = GeneLinearScan('problem_003/scanfiles0016')
  scan.plot_nsq()
  scan.plot_omega()
  scan.phi.plot_mode()
  scan.apar.plot_mode(scan=2)
  scan.ions.plot_mode()
  scan.electrons.plot_mode(scannum=4, moment=3)
  scan.vsp.plot_vspace(species='ions', scannum=3)

Nonlinear simulation analysis:

.. code-block:: python

  from pygene import GeneNonlinear
  nl = GeneNonlinear('problem_004')
  nl.plot_nrg()
  nl.plot_energy()
  nl.apar.plot_mode()
  nl.ions.plot_mode()
  nl.vsp.plot_vspace(species='electrons')