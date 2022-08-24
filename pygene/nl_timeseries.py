from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pygene as pg

plt.close('all')

topdir = Path.home() / 'pegasus/nl08'
subdirs = [
    # 'eq21-pn50',
    'eq21-pn60',
    'eq21-pn70',
    # 'eq50-pn50',
    'eq50-pn60',
    'eq50-pn70',
]

for sd in subdirs:
    simdir = topdir / sd
    print(simdir.as_posix())
    dirs = pg.allsims(simdir)
    sims = [pg.GeneNonlinear(d) for d in dirs]

    sim = sims[-1]
    sim.plot_nrg(use_nrg=pg.concat_nrg(sims), save=False)
    # sim.phi.plot_mode(save=saveplots)
    # sim.apar.plot_mode(save=saveplots)
    # for moment in range(6):
    #     sim.electrons.plot_mode(moment=moment, save=saveplots)
    # sim.electrons.plot_fluxes(save=saveplots)
    # sim.vsp.plot_vspace(save=saveplots)

