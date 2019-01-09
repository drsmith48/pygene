.. highlight:: python

PYGENE

Python tools for GENE (http://genecode.org/)

Example usage

`scanscript` calculation in `problem-003` directory::

  import pygene
  # load a `scanscript` output directory
  scan = pygene.GeneLinearScan(pygene.genehome/'problem-003'/'scanfiles0016')
  # plot |n|^2 vs. time for all runs
  scan.plot_nsq()
  # plot omega (re/im) vs. scan parameter or run #
  scan.plot_omega()
  # plot phi spectra for run #3
  scan.get_field(run=3)
  scan.field.plot_spectra()
  # plot ion density parallel mode structure for run #5
  scan.get_moment(run=5)
  scan.moment.plot_mode()
  # plot velocity-space spectra for run #7
  scan.get_vsp(run=7)
  scan.plot_vspace()

Notes
-----

**PYGENE** is developed for Python3, but sources files are pasteurized (PYPI `future` package) to run with Python2.