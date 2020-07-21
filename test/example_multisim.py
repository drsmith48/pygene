import matplotlib.pyplot as plt
from pathlib import Path
import pygene as pg

plt.close('all')

topdir = Path.home() / 'pegasus/nl08'
subdirs = [
    'eq21-pn50',
    'eq21-pn60',
    'eq21-pn70',
    'eq50-pn50',
    'eq50-pn60',
    'eq50-pn70',
]

dirs = [topdir/subd for subd in subdirs]
pg.plot_multisim(dirs)