#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:31:13 2017

@author: drsmith
"""

import os
#import shutil
import subprocess as sp
from jinja2 import Environment, FileSystemLoader
import numpy as np
from geqdsk import Geqdsk

eq21 = '/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk'
eq50 = '/p/gene/drsmith/eqdsk/ConstBeta_changingBT_eq50.geqdsk'

def genesubmit(gfile=eq21, ref_gfile='', 
               psinorm=0.8, rova=None, omt_factor=0.2, dryrun=False, 
               beta_factor=None, coll_factor=None, 
               amhd_factor=None, dpdx_factor=None, **kwargs):

    # context for template rendering
    context = {}


    # walltime
    walltime_hr = kwargs.get('walltime_hr', 24)
    context['walltime'] = '{:d}:00:00'.format(walltime_hr)
    context['timelim'] = int(walltime_hr*60*60-500)
    
    
    # miller calculations
    args = {'rova':rova, 'omt_factor':omt_factor, 'psinorm':psinorm}
    # ref. miller, if needed
    ref_miller = None
    if ref_gfile:
        ref_geq = Geqdsk(gfile=ref_gfile, quiet=True)
        ref_miller = ref_geq.miller(**args)
    # open g-file
    geq = Geqdsk(gfile=gfile, quiet=True)
    miller = geq.miller(ref_miller=ref_miller, **args)
    # set min abs(s-hat)
    shat = miller['shat']
    if np.abs(shat)<0.3:
        print("Orig shat: {}".format(shat))
        miller['shat-orig'] = shat
        shat = np.sign(shat)*0.3
        print("New shat: {}".format(shat))
        miller['shat'] = shat
    # update context with miller values
    context.update(miller)
    context['ion_omt'] = miller['omt']
    context['ele_omt'] = miller['omt']
    context['ion_omn'] = miller['omn']
    context['ele_omn'] = miller['omn']
    context['beta_calc'] = miller['beta']
    context['amhd_calc'] = miller['amhd']
    context['coll_calc'] = miller['coll']
    context['dpdx_pm_calc'] = miller['dpdx_pm']
    context['dpdx_pm_2_calc'] = miller['dpdx_pm_2']
    

    # set beta
    if beta_factor is None:
        context['beta'] = None
    else:
        context['beta'] = miller['beta']*beta_factor
    # set amhd
    if amhd_factor is None:
        context['amhd'] = None
    else:
        context['amhd'] = miller['amhd']*amhd_factor
    # set dpdx
    if dpdx_factor is None:
        context['dpdx_pm'] = None
    else:
        context['dpdx_pm'] = miller['dpdx_pm']*dpdx_factor
    # set coll
    if coll_factor is None:
        context['coll'] = None
    else:
        context['coll'] = miller['coll']*coll_factor
        if coll_factor == 0:
            # if zero, disable collisions
            context['collision_op'] = "'none'"
            

    # update context and display values
    context.update(kwargs)
    
    
    # calc lx and nx0 based on kymin and shat
    if context.get('nonlinear',None) == '.T.':
        kymin = context['kymin']
        shat = context['shat']
        lx_fund = 1.0/(kymin*np.abs(shat))
        lx = lx_fund
        lx_min = 60.0
        while lx < lx_min:
            lx += lx_fund
        nx0 = np.int(np.float_power(2,np.ceil(np.log2(lx*1.7))))
        context['nx0'] = nx0
        context['lx_min'] = lx_min
        context['lx_fund'] = lx_fund
        context['lx'] = lx
    
    
    print('Context values:')
    for key,value in iter(context.items()):
        print('  {} : {}'.format(key, value))
    if not dryrun:
        reply = input('[Proceed with job submission?]')
    if dryrun or (reply and reply.lower()[0]=='n'):
        return
    
    # cd to GENE home directory
    initdir = os.path.abspath(os.curdir)
    GENEWORK = os.environ['GENEWORK']
    GENETOP = os.environ['GENETOP']
    os.chdir(GENEWORK)
    ret = sp.run(['./newprob'], timeout=10)
    os.chdir(initdir)
    probname = 'prob{:02d}'.format(ret.returncode)
    
    # jobname
    jobname = kwargs.get('jobname', None)
    if jobname:
        if os.path.exists(os.path.join(GENEWORK, jobname)):
            raise ValueError('jobname exists')
        print('Renaming {} to {}'.format(probname, jobname))
        os.rename(os.path.join(GENEWORK, probname),
                  os.path.join(GENEWORK, jobname))
        probname = jobname
    else:
        jobname = 'GENE-'+probname
    context['jobname'] = jobname
    fullprobdir = os.path.join(GENEWORK, probname)

    with open(os.path.join(fullprobdir,'input.txt'), mode='w') as f:
        f.write('context dictionary\n')
        for key in iter(context):
            f.write('  {} :  {}\n'.format(key, context[key]))
        f.write('miller dictionary\n')
        for key in iter(miller):
            f.write('  {} :  {}\n'.format(key, miller[key]))
        f.write('kwargs dictionary\n')
        for key in iter(kwargs):
            f.write('  {} :  {}\n'.format(key, kwargs[key]))
        f.write('beta_factor :  {}\n'.format(beta_factor))
        f.write('coll_factor :  {}\n'.format(coll_factor))
        f.write('amhd_factor :  {}\n'.format(amhd_factor))
        f.write('dpdx_factor :  {}\n'.format(dpdx_factor))
          
    
    # render templates
    env = Environment(loader=FileSystemLoader(os.path.join(GENETOP, 'tools')))
    template = env.get_template('launcher.template.sh')
    launchfile = os.path.join(fullprobdir, 'launcher.cmd')
    os.remove(launchfile)
    with open(launchfile, 'w') as f:
        f.write(template.render(context))
    template = env.get_template('parameters.template.f90')
    paramfile = os.path.join(fullprobdir, 'parameters')
    os.remove(paramfile)
    with open(paramfile, 'w') as f:
        f.write(template.render(context))
        
    # submit job
    print('Submitting job in {}'.format(probname))
    os.chdir(fullprobdir)
    ret = sp.run(['sbatch', 'launcher.cmd'], timeout=10)
    os.chdir(initdir)


def ref01(psinorm=0.6, jobname='ref01', **kwargs):
    kwargs_out = {'scanscript_comment':' ',
                  'mpiexec_comment':'##',
                  'walltime_hr':24*2,
                  'ntasks':32*2,
                  'kymin':'0.03 !scanlist: 0.03,0.1,0.3,1.0,3.0',
                  'nx0':16,
                  'nz0':16,
                  'hyp_z':1.0,
                  'dryrun':False
                  }
    kwargs_out.update(kwargs)
    genesubmit(psinorm=psinorm, 
               omt_factor=0.7,
               jobname=jobname, 
               **kwargs_out)
    
def ref02(**kwargs):
    kwargs_out = {}
    kwargs_out.update(kwargs)
    ref01(psinorm=0.9, jobname='ref02', **kwargs_out)
    
def ref03(**kwargs):
    kwargs_out = {'dryrun':False,
                  'ntasks':32*8,
                  'walltime_hr':24*4,
                  'kymin':'0.3',
                  'nx0':16,
                  'nz0':'4 !scanlist: 4,8,16',
                  'hyp_z':'0.25 !scanlist: 0.25,1.0,4.0'
                  }
    kwargs_out.update(kwargs)
    ref01(psinorm=0.6, jobname='ref03', **kwargs_out)

def ref06(**kwargs):
    kwargs_out = {'dryrun':False,
                  'ntasks':32*8,
                  'partition':'kruskal,dawson',
                  'walltime_hr':24*4,
                  'kymin':'0.15 !scanfunc: 24,0.15*1.15**(xi-1),2',
                  'nx0':32,
                  'nz0':32,
                  'hyp_z':40,
                  'gfile':eq50,
                  'ref_gfile':eq21,
                  'psinorm':0.6,
                  'jobname':'ref06'
                  }
    kwargs_out.update(kwargs)
    ref01(**kwargs_out)

def coarse_iv_scan(**kwargs):
    psinorm_arr = [0.6]
    omt_factor_arr = [0.3, 0.5, 0.7]
    kwargs_out = {'scanscript_comment':' ',
                  'mpiexec_comment':'##',
                  'walltime_hr':24*2,
                  'ntasks':32*2,
                  'kymin':0.01,
                  'kyscan':'!scanlist: 0.01,0.03,0.1,0.3,1.0,3.0',
                  'nx0':32,
                  'nz0':32,
                  'hyp_z':1.0,
                  'dryrun':False
                  }
    kwargs_out.update(kwargs)
    #suffix = kwargs_out.pop('suffix', 'v01')
    for psinorm in psinorm_arr:
        for omt_factor in omt_factor_arr:
            jobname = 'civ_psinorm{:02n}_omtfac{:02n}'.format(
                    psinorm*1e2, omt_factor*1e2)
            #jobname += '_' + suffix
            print('Jobname: {}'.format(jobname))
            genesubmit(psinorm=psinorm, 
                       omt_factor=omt_factor,
                       jobname=jobname, 
                       **kwargs_out)

def coarse_ev_scan(**kwargs):
    psinorm_arr = [0.3, 0.6, 0.9]
    omt_factor_arr = [0.3, 0.5, 0.7]
    kwargs_out = {'scanscript_comment':' ',
                  'mpiexec_comment':'##',
                  'walltime_hr':24*4,
                  'ntasks':32*4,
                  'perf_vec':'0 0 0 0 0 0 0 0 0',
                  'comp_type':'EV',
                  'nonlinear':'.F.',
                  'bpar':'.F.',
                  'nky0':1,
                  'kymin':0.01,
                  'kyscan':'!scanlist: 0.01,0.03,0.1,0.3,1.0,3.0',
                  'n_ev':2,
                  'nx0':8,
                  'dryrun':True
                  }
    kwargs_out.update(kwargs)
    for psinorm in psinorm_arr:
        for omt_factor in omt_factor_arr:
            jobname = 'coarse_ev_psinorm{:02n}_omtfac{:02n}'.format(
                    psinorm*1e2, omt_factor*1e2)
            print('Jobname: {}'.format(jobname))
            genesubmit(psinorm=psinorm, 
                       omt_factor=omt_factor,
                       jobname=jobname, 
                       **kwargs_out)

def v01_nl(**kwargs):
    #psinorm_arr = [0.4, 0.6, 0.8]
    #omt_factor_arr = [0.2, 0.5]
    #psinorm_arr = [0.4, 0.6, 0.8]
    psinorm_arr = [0.6]
    omt_factor_arr = [0.5]
    kwargs_out = {'mpiexec_comment':' ',
                  'nky0':24,
                  'kymin':0.025,
                  'nonlinear':'.T.',
                  'walltime_hr':24*4,
                  'ntasks':32*2,
                  'dryrun':False
                  }
    kwargs_out.update(kwargs)
    suffix = kwargs_out.pop('suffix', 'v01')
    for psinorm in psinorm_arr:
        for omt_factor in omt_factor_arr:
            jobname = 'psinorm{:02n}_omtfac{:02n}'.format(
                    psinorm*1e2, omt_factor*1e2)
            jobname += '_' + suffix
            print('Jobname: {}'.format(jobname))
            genesubmit(psinorm=psinorm, 
                       omt_factor=omt_factor,
                       jobname=jobname, 
                       **kwargs_out)


def v03_nl(**kwargs):
    kwargs_out = {'ref_gfile':'/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk',
                  'gfile':'/p/gene/drsmith/eqdsk/ConstBeta_changingBT_eq50.geqdsk',
                  'suffix':'v03',
                  'partition':'kruskal,dawson'
                  }
    kwargs_out.update(kwargs)
    v01_nl(**kwargs_out)
    
def v04_nl(**kwargs):
    kwargs_out = {'suffix':'v04',
                  'dryrun':False,
                  'walltime_hr':65,
                  'partition':'kruskal'}
    kwargs_out.update(kwargs)
    v01_nl(**kwargs_out)
    
def v05_lin(**kwargs):
    kwargs_out = {'nonlinear':'.F.',
                  'ntasks':64,
                  'partition':'dawson,kruskal',
                  'mpiexec_comment':'##',
                  'scanscript_comment':' ',
                  'dryrun':False,
                  'nky0':1,
                  'kymin':0.05,
                  'kyscan':'!scanrange: 0.05,0.05,0.8',
                  'nx0':128,
                  'suffix':'v05',
                  }
    kwargs_out.update(kwargs)
    v01_nl(**kwargs_out)
                  
def v06_nl():
    kwargs_out = {'ntasks':16*32,
                  'partition':'kruskal',
                  'mpiexec_comment':' ',
                  'kymin':0.02,
                  'nky0':24,
                  'suffix':'v06',
                  'nx0':256,
                  'dryrun':False}
    v01_nl(**kwargs_out)

def v07_nl(**kwargs):
    kwargs_out = {'ntasks':16*32,
                  'partition':'kruskal',
                  'mpiexec_comment':' ',
                  'kymin':0.02,
                  'nky0':32,
                  'walltime_hr':24*4,
                  'suffix':'v07',
                  'dryrun':False}
    kwargs_out.update(kwargs)
    v01_nl(**kwargs_out)
    
def v08_ev(kmin=0.01, **kwargs):
    kyvalues = kmin*np.power(1.2,np.arange(40))
    scan=''
    for ky in kyvalues:
        if ky != kyvalues[-1]:
            kystr = '{:.4f},'.format(ky)
        else:
            kystr = '{:.4f}'.format(ky)
        scan = ''.join([scan,kystr])
    kwargs_out = {'ntasks':4*32,
                  'scanscript_comment':' ',
                  'mpiexec_comment':'##',
                  'partition':'kruskal',
                  'nx0':4,
                  'nky0':1,
                  'kymin':kmin,
                  'kyscan':'!scanlist: {}'.format(scan),
                  'suffix':'v08',
                  'comp_type':'EV',
                  'nonlinear':'.F.',
                  'n_ev':4,
                  'dryrun':False
                  }
    kwargs_out.update(kwargs)
    v01_nl(**kwargs_out)
    
def v09_nl(**kwargs):
    kwargs_out = {'ref_gfile':'/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk',
                  'gfile':'/p/gene/drsmith/eqdsk/ConstBeta_changingBT_eq50.geqdsk',
                  'suffix':'v09',
                  'partition':'kruskal'
                  }
    kwargs_out.update(kwargs)
    v07_nl(**kwargs_out)
    
def v10_nl(**kwargs):
    kwargs_out = {'kymin':0.01,
                  'suffix':'v10',
                  'dryrun':False}
    kwargs_out.update(kwargs)
    v07_nl(**kwargs_out)
    
def v11_ev(**kwargs):
    kwargs_out = {'dryrun':False,
                  'suffix':'v11',
                  'kmin':0.005}
    kwargs_out.update(kwargs)
    v08_ev(**kwargs_out)

def v12_ev(**kwargs):
    kwargs_out = {'suffix':'v12',
                  'ref_gfile':'/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk',
                  'gfile':'/p/gene/drsmith/eqdsk/ConstBeta_changingBT_eq50.geqdsk',
                  'dryrun':False
                  }
    kwargs_out.update(kwargs)
    v11_ev(**kwargs_out)
    
def v13_nl(**kwargs):
    kwargs_out = {'suffix':'v13',
                  'ref_gfile':'/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk',
                  'gfile':'/p/gene/drsmith/eqdsk/ConstBeta_changingBT_eq50.geqdsk',
                  'dryrun':False}
    kwargs_out.update(kwargs)
    v10_nl(**kwargs_out)
    
def v14_lin(**kwargs):
    kwargs_out = {'dryrun':False,
                  'suffix':'v14',
                  'comp_type':'IV',
                  'kmin':0.005,
                  'ntasks':8*32,
                  'nx0':16
                  }
    kwargs_out.update(kwargs)
    v08_ev(**kwargs_out)

def v15_ev(**kwargs):
    kwargs_out = {'dryrun':False,
                  'suffix':'v15',
                  'kmin':0.02,
                  'n_ev':2,
                  'nx0':8,
                  'ntasks':4*32,
                  'walltime_hr':8*24}
    kwargs_out.update(kwargs)
    v08_ev(**kwargs_out)

def v16_nl(**kwargs):
    kwargs_out = {'kymin':0.015,
                  'nky0':64,
                  'suffix':'v16',
                  'dryrun':False,
                  'walltime_hr':24*8}
    kwargs_out.update(kwargs)
    v07_nl(**kwargs_out)

def v17_ev(**kwargs):
    kwargs_out = {'suffix':'v17',
                  'ref_gfile':'/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk',
                  'gfile':'/p/gene/drsmith/eqdsk/ConstBeta_changingBT_eq50.geqdsk',
                  'dryrun':False
                  }
    kwargs_out.update(kwargs)
    v15_ev(**kwargs_out)

def v18_ev(**kwargs):
    kwargs_out = {'dryrun':False,
                  'suffix':'v18',
                  'kmin':0.01,
                  'n_ev':2,
                  'nx0':16,
                  'bpar':'.T.',
                  'ntasks':4*32,
                  'walltime_hr':8*24}
    kwargs_out.update(kwargs)
    v08_ev(**kwargs_out)
