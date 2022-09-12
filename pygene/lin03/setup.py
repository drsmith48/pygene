from pathlib import Path

GENE_DIR = Path.home() / 'gene'
assert GENE_DIR.exists() and GENE_DIR.is_dir()

VERSION = 'lin03'

TIGRESS_DIR = GENE_DIR / 'tigress' / VERSION
TIGRESS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS_DIR = Path('parameter_defaults').resolve()
assert DEFAULTS_DIR.exists() and DEFAULTS_DIR.is_dir()

EQUILIBRIA = [
    'eq21',
    'eq50',
]

PSINORM_VALUES = [
    'pn40',
    'pn50',
    'pn60',
    'pn70',
    'pn80',
]

GENE_EXEC = GENE_DIR / 'genecode/bin/gene_stellar'
assert GENE_EXEC.exists(), f"{GENE_EXEC} does not exist"

GENE_SCANSCRIPT = GENE_DIR / 'genecode/tools/perl/scanscript'
assert GENE_SCANSCRIPT.exists(), f"{GENE_SCANSCRIPT} does not exist"
