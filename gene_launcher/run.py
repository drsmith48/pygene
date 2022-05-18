#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:31:13 2017

@author: drsmith
"""

import os
import shutil
import subprocess as sp
from jinja2 import Environment, FileSystemLoader
from geqdsk import Geqdsk

def genesubmit(miller, beta_factor=None, coll_factor=None, **kwargs):
    """
    Prepare and submit a single GENE problem.
    
    beta_factor - None for self-consistent beta; numeric for beta scaling
    coll_factor - None for self-consistent coll.; numeric for beta scaling
    """

    # cd to GENE home directory
    initdir = os.path.abspath(os.curdir)
    os.chdir(os.environ['GENEHOME'])
    ret = sp.run(['./newprob'], timeout=10)
    probdir = 'prob{:02d}'.format(ret.returncode)
    context = {}
    
    # jobname
    jobname = kwargs.pop('jobname',None)
    if jobname:
        print('Renaming {} to {}'.format(probdir, jobname))
        os.rename(probdir, jobname)
        probdir = jobname
    else:
        jobname = 'GENE-'+probdir
    context['jobname'] = jobname
    
    # update context with miller values
    context.update(miller)
    context['ion_omt'] = miller['omt']
    context['ele_omt'] = miller['omt']
    context['ion_omn'] = miller['omn']
    context['ele_omn'] = miller['omn']
    
    if beta_factor is None:
        # auto-calc self-consistent value
        pass
    else:
        # scale with beta_factor
        context['beta'] = miller['beta']*beta_factor
    if coll_factor is None:
        # auto-calc self-consistent value
        pass
    else:
        if coll_factor == 0:
            # disable if zero
            context['collision_op'] = "'none'"
        else:
            # scale with coll_factor
            context['coll'] = miller['coll']*coll_factor
            
    
    if kwargs: context.update(kwargs)
    
    # cd to problem directory
    os.chdir(probdir)
    env = Environment(loader=FileSystemLoader(initdir))
    template = env.get_template('launcher.template.sh')
    os.remove('launcher.cmd')
    with open('launcher.cmd', 'w') as f:
        f.write(template.render(context))
    template = env.get_template('parameters.template.f90')
    os.remove('parameters')
    with open('parameters', 'w') as f:
        f.write(template.render(context))
    shutil.copyfile('parameters','parameters_orig')
    print('Submitting job in {}'.format(probdir))
    ret = sp.run(['sbatch', 'launcher.cmd'], timeout=10)
    
    # cd to initial directory
    os.chdir(initdir)


def genebatch(scanparam=None, scanvalues=None, gfile=None, 
              psinorm=None, rova=None, omt_factor=None, **kwargs):
    """
    Prepare and submit multiple GENE problems.

    * Loads a single Geqdsk file using miller.py
    * Prepares scan parameter, and calls eq.miller() as needed
    * Runs newprob in /p/gene/drsmith/genetools
    * Replace launcher.cmd and parameters
    * Submit jobs to cluster
    """
    
    geq = Geqdsk(gfile=gfile)
    
    if scanparam in ['psinorm','rova','omt_factor']:
        # calc miller for each psinorm/rova/omt_factor value
        args = {'rova':rova, 'omt_factor':omt_factor, 'psinorm':psinorm}
        for value in scanvalues:
            args[scanparam] = value
            m = geq.miller(**args)
            genesubmit(miller=m, **kwargs)
    elif scanparam:
        # calc miller, and substitute parameter value
        m = geq.miller(psinorm=psinorm, rova=rova, omt_factor=omt_factor)
        for value in scanvalues:
            kwargs[scanparam] = value
            genesubmit(miller=m, **kwargs)
    elif scanparam is None and scanvalues is None:
        # calc miller with no scan or substitution
        m = geq.miller(psinorm=psinorm, rova=rova, omt_factor=omt_factor)
        genesubmit(miller=m, **kwargs)
    else:
        raise ValueError('Invalid scanparam or scanvalues:\n{}\n{}'.format(
                scanparam, scanvalues))


if __name__ == '__main__':
    #genebatch(scanparam='omt_factor',
    #          scanvalues=[0.2,0.8])
    psinorm_arr = [x/10. for x in range(3,10)]
    omt_factor_arr = [0.2, 0.3, 0.4, 0.5]
    
    for psinorm in psinorm_arr:
        for omt_factor in omt_factor_arr:
            jobname = 'psinorm{:02n}_omtfac{:02n}'.format(
                    psinorm*1e2, omt_factor*1e2)
            print('Jobname: {}'.format(jobname))
            genebatch(psinorm=psinorm, 
                      omt_factor=omt_factor,
                      jobname=jobname)
            
            jobname = 'psinorm{:02n}_omtfac{:02n}_dpdxgradB'.format(
                    psinorm*1e2, omt_factor*1e2)
            print('Jobname: {}'.format(jobname))
            genebatch(psinorm=psinorm, 
                      omt_factor=omt_factor,
                      jobname=jobname, 
                      dpdx_term="'gradB_eq_curv'")
            
            jobname = 'psinorm{:02n}_omtfac{:02n}_dpdxcurv'.format(
                    psinorm*1e2, omt_factor*1e2)
            print('Jobname: {}'.format(jobname))
            genebatch(psinorm=psinorm, 
                      omt_factor=omt_factor,
                      jobname=jobname, 
                      dpdx_term="'curv_eq_gradB'")