import matplotlib.pyplot as plt
from pathlib import Path
import pygene as pg

plt.close('all')

topdir = Path.home() / 'pegasus/nl08/eq21-pn50'
savefigs=False

# simdir = topdir / 'run-77453'
# sim = pg.GeneNonlinear(simdir)
# sim.plot_nrg()
#
# dirs = ['run-77453','run-77454','run-77455','run-77456']
# sims = [pg.GeneNonlinear(topdir/d) for d in dirs]
# nrgs = pg.concat_nrg(sims)
# sims[-1].plot_nrg(use_nrg=nrgs)
#
# nrgs = pg.concat_nrg(topdir)
# sims[-1].plot_nrg(use_nrg=nrgs)

dirs = pg.allsims(topdir)
sims = [pg.GeneNonlinear(d) for d in dirs]

lastdir = pg.lastsim(topdir)
sim = pg.GeneNonlinear(lastdir)
# sim.plot_nrg(save=True)
sim.plot_nrg(use_nrg=pg.concat_nrg(sims), save=savefigs)
sim.phi.plot_mode(save=savefigs)
sim.apar.plot_mode(save=savefigs)
for moment in range(6):
    sim.electrons.plot_mode(moment=moment, save=savefigs)
sim.electrons.plot_fluxes(save=savefigs)
sim.vsp.plot_vspace(save=savefigs)