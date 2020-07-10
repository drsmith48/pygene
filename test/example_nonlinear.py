from pathlib import Path
import pygene as pg

topdir = Path.home() / 'pegasus/nl08/eq21-pn50'

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

dirs = sorted([p for p in topdir.glob('run-*') if p.is_dir()])
sims = [pg.GeneNonlinear(d) for d in dirs]

sim = sims[-1]
# sim.plot_nrg(save=True)
sim.plot_nrg(use_nrg=pg.concat_nrg(sims), save=True)
sim.phi.plot_mode(save=True)
sim.apar.plot_mode(save=True)
for moment in range(6):
    sim.electrons.plot_mode(moment=moment, save=True)
sim.electrons.plot_fluxes(save=True)
sim.vsp.plot_vspace(save=True)