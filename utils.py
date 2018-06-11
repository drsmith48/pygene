
import os
from pathlib import Path
import struct
import re
import numpy as np

GENEWORK = Path(os.environ['GENEWORK'])

re_nrgline = re.compile(r'^'+''.join([r'\s+([0-9E.+-]+)' for i in range(8)]))
re_energy = re.compile(r'^'+''.join([r'\s+([0-9ENan.+-]+)' for i in range(14)]))
re_prefix = re.compile(r'^([a-zA-Z_-]+)')
re_amp = re.compile(r'^&[a-zA-Z]+')
re_whitespace = re.compile(r'^\s*$')
re_slash = re.compile(r'^/')
re_comment = re.compile(r'^\s*#')
re_item = re.compile(r'^\s*(?P<key>[a-zA-Z0-9_]+)\s*=\s*(?P<value>\S+)')
re_scan = re.compile(r'!scan')
re_scanlogheader = re.compile(r'^#Run\s+\|\s+(?P<param>[a-zA-Z0-9_]+)\s')
re_scanlog = re.compile(r'^(?P<run>[0-9]+)\s+\|\s+(?P<value>[0-9.e+-]+)\s+\|')
re_omegafile = re.compile(r'^\s+(?P<ky>[0-9.Na-]+)\s+(?P<omi>[0-9.Na-]+)\s+(?P<omr>[0-9.Na-]+)')


def validate_path(path):
    pathout = Path(path)
    if not pathout.is_absolute():
        pathout = GENEWORK / path
    if not pathout.exists():
        raise ValueError('Invalid path: {}'.format(path))
    return pathout


def get_binary_config(nfields=None, elements=None, isdouble=None, isbig=None):
    realsize = 4 + 4 * isdouble
    complexsize = 2*realsize
    intsize = 4
    entrysize = elements * complexsize
    leapfld = nfields * (entrysize+2*intsize)
    if isbig:
        nprt=(np.dtype(np.float64)).newbyteorder()
        npct=(np.dtype(np.complex128)).newbyteorder()
        fmt = '>idi'
    else:
        nprt=np.dtype(np.float64)
        npct=np.dtype(np.complex128)
        fmt = '=idi'
    te = struct.Struct(fmt)
    tesize = te.size
    return (intsize, entrysize, leapfld, nprt, npct, te, tesize)


def path_label(path=None, label=''):
    path = validate_path(path)
    shortpath = '/'.join(path.parts[-2:])
    if label:
        plotlabel = label
        rx = re_prefix.match(label)
        filelabel = rx.group(1)
    else:
        plotlabel = ''
        filelabel = path.parts[-1]
    return path, shortpath, plotlabel, filelabel


def read_parameters(file=None):
    file = validate_path(file)
    params = {'isscan':False,
              'species':[],
              'lx':0,
              'ly':0}
    with file.open('r') as f:
        for line in f:
            if re_amp.match(line) or \
                re_slash.match(line) or \
                re_comment.match(line) or \
                re_whitespace.match(line):
                    continue
            rx = re_item.match(line)
            if not rx:
                continue
            if re_scan.search(line):
                params['isscan'] = True
            d = rx.groupdict()
            if d['key'].lower() == 'name':
                params['species'].append(eval(d['value']))
            if d['value'].lower() in ['true', 't', '.t.']:
                d['value'] = 'True'
            if d['value'].lower() in ['false', 'f', '.f.']:
                d['value'] = 'False'
            if d['key'].islower():
                d['value'] = eval(d['value'])
            params[d['key']] = d['value']
    params['isnonlinear'] = params['nonlinear']==True
    return params


def read_energy(file=Path()):
    '''
    Read data from a single energy file
    '''
    data = np.empty((0,14))
    with file.open() as f:
        for i,line in enumerate(f):
            if i%20 != 0 or i<=14:
                continue
            rx = re_energy.match(line)
            linedata = np.empty((1,14))
            for i,s in enumerate(rx.groups()):
                if s == 'NaN':
                    linedata[0,i] = np.NaN
                else:
                    linedata[0,i] = eval(s)
            data = np.append(data, linedata, axis=0)
    return {'time':data[:,0],
            'etot':data[:,1],
            'ddtetot':data[:,2],
            'drive':data[:,3],
            'heatsrc':data[:,4],
            'colldiss':data[:,5],
            'hypvdiss':data[:,6],
            'hypzdiss':data[:,7],
            'nonlinear':data[:,9],
            'curvmisc':data[:,11],
            'convergance':data[:,12],
            'deltae':data[:,13],
            'netdrive':np.sum(data[:,3:12],axis=1),
            'grossdrive':np.sum(data[:,3:5],axis=1)}


def read_nrg(file=Path(), species=[]):
    '''
    Read data from asingle nrg file
    '''
    nsp = len(species)
    data = np.empty((0,8,nsp))
    time = np.empty((0,))
    decimate=1
    with file.open() as f:
        for i,line in enumerate(f):
            if i >= 3000:
                decimate=20
                break
    print(decimate)
    with file.open() as f:
        for i,line in enumerate(f):
            itime = i//(nsp+1)
            if itime%decimate != 0:
                continue
            iline = i%(nsp+1)
            if iline == 0:
                time = np.append(time, float(line))
                data = np.append(data, np.empty((1,8,nsp)), axis=0)
            elif iline <= nsp:
                rx = re_nrgline.match(line)
                values = [float(val) for val in rx.groups()]
                data[-1,:,iline-1] = np.array(values)
            else:
                raise ValueError(str(iline))
    output = {'time':time}
    for i in range(nsp):
        output[species[i]] = {'nsq':data[:,0,i],
                              'uparsq':data[:,1,i],
                              'tparsq':data[:,2,i],
                              'tperpsq':data[:,3,i],
                              'games':data[:,4,i],
                              'gamem':data[:,5,i],
                              'qes':data[:,6,i],
                              'qem':data[:,7,i]}
    return output
