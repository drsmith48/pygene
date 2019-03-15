from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()

import os
from pathlib import Path
import re

import numpy as np

# 'GENEWORK' directory; if undefined, default to user home directory
genework = Path(os.getenv('GENEWORK', default = os.getenv('HOME')))

re_nrgline = re.compile(r'^'+''.join([r'\s+([0-9E.+-]+)' for i in range(8)]))
re_energy = re.compile(r'^'+''.join([r'\s+([0-9ENan.+-]+)' for i in range(14)]))
re_prefix = re.compile(r'^([a-zA-Z_-]+)')
re_amp = re.compile(r'^&[a-zA-Z]+')
re_whitespace = re.compile(r'^\s*$')
re_slash = re.compile(r'^/')
re_comment = re.compile(r'^\s*#')
re_item = re.compile(r'^\s*(?P<key>[a-zA-Z0-9_]+)\s*=\s*(?P<value>.+)')
re_scan = re.compile(r'!scan')
re_scanlogheader = re.compile(r'^#Run\s+\|\s+(?P<param>[!a-zA-Z0-9_]+)\s')
re_scanlog = re.compile(r'^(?P<run>[0-9]+)\s+\|\s+(?P<value>[0-9.e+-]+)\s+\|')
re_omegafile = re.compile(r'^\s+(?P<ky>[0-9.Na-]+)\s+(?P<omi>[0-9.Na-]+)\s+(?P<omr>[0-9.Na-]+)')

eps = np.finfo(np.float).eps

def log1010(data):
    return 10*np.log10(data + eps)

def validate_path(path):
    pathout = Path(path)
    if not pathout.is_absolute():
        pathout = genework / path
    if not pathout.exists():
        raise ValueError('Invalid path: {}'.format(path))
    return pathout
