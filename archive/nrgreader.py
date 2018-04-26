#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:15:07 2017

@author: drsmith
"""

import os
import h5py
import numpy as np

genehome = os.environ['GENEHOME']

def querygroup(group):
    queryattrs(group)
    for key in group.keys():
        value = group[key]
        if isinstance(value, h5py._hl.dataset.Dataset):
            queryattrs(value)
            if value.shape:
                print('{}: shape {}  dtype {}'.format(value.name, value.shape, value.dtype))
            else:
                print('{}: {}'.format(value.name, value.value.decode('UTF-8')))
        if isinstance(value, h5py._hl.group.Group):
            querygroup(value)
            
def queryattrs(obj):
    for key in obj.attrs.keys():
        print('  {} attribute {}: {}'.format(obj.name, key, obj.attrs[key]))


if __name__ == '__main__':
    filename = os.path.join(genehome, 
                            'psinorm80_omtfac20_nl', 'run01',
                            'nrgsummaryions_act.h5')
    f = h5py.File(filename, 'r')
    querygroup(f)
    f.close()
    