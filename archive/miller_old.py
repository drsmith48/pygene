#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:32:25 2013

@author: dtold
"""
from __future__ import print_function

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as US


def find(val,arr):
    ind=0
    mindiff=1000.
    for i in range(len(arr)):
        diff=abs(arr[i]-val)
        if diff<mindiff:
            ind=i
            mindiff=diff
    return ind


def miller(r_a=0.76, gfile='/p/gene/drsmith/eqdsk/pegasus-eq21.geqdsk'):


    efile = open(gfile, 'r')

    # read eqdsk lines
    eqdsk=efile.readlines()

    # parse line 0
    print('Header: %s' %eqdsk[0])
    nw=int(eqdsk[0].split()[-2])
    nh=int(eqdsk[0].split()[-1])
    print('Resolution: %d x %d' %(nw,nh))

    # parse line 1
    entrylength=16
    try:
        rdim,zdim,rctr,rmin,zmid=[float(eqdsk[1][j*entrylength:(j+1)*entrylength])
            for j in range(len(eqdsk[1])/entrylength)]
    except:
        entrylength=15
        try:
            rdim,zdim,rctr,rmin,zmid=[float(eqdsk[1][j*entrylength:(j+1)*entrylength])
                for j in range(len(eqdsk[1])/entrylength)]
        except:
            sys.exit('Error reading EQDSK file, please check format!')

    # parse line 2
    rmag,zmag,psiax,psisep,Bctr=[float(eqdsk[2][j*entrylength:(j+1)*entrylength])
        for j in range(len(eqdsk[2])/entrylength)]

    # parse line 3
    _,psiax2,_,rmag2,_=[float(eqdsk[3][j*entrylength:(j+1)*entrylength])
        for j in range(len(eqdsk[3])/entrylength)]

    # parse line 4
    zmag2,_,psisep2,_,_=[float(eqdsk[4][j*entrylength:(j+1)*entrylength])
        for j in range(len(eqdsk[4])/entrylength)]

    if rmag!=rmag2: sys.exit('Inconsistent rmag: %7.4g, %7.4g' %(rmag,rmag2))
    if psiax2!=psiax: sys.exit('Inconsistent psiax: %7.4g, %7.4g' %(psiax,psiax2))
    if zmag!=zmag2: sys.exit('Inconsistent zmag: %7.4g, %7.4g' %(zmag,zmag2) )
    if psisep2!=psisep: sys.exit('Inconsistent psisep: %7.4g, %7.4g' %(psisep,psisep2))

    print('R_mag_axis = {:.3g} m'.format(rmag))

    F = np.empty(nw,dtype=float)        # pol current (F=RBt) [T-m] on uniform flux grid
    p = np.empty(nw,dtype=float)        # pressure [Pa] on uniform flux grid
    ffprime = np.empty(nw,dtype=float)  # FF'=FdF/dpsi on uniform flux grid
    pprime = np.empty(nw,dtype=float)   # dp/dpsi [Pa/(Wb/rad)] on uniform flux grid
    qpsi = np.empty(nw,dtype=float)     # q safety factor on uniform flux grid
    psirz_1d = np.empty(nw*nh,dtype=float)   # pol. flux [Wb/rad] on rectangular grid
    start_line=5
    lines=range(nw/5)
    if nw%5!=0: lines=range(nw/5+1)
    for i in lines:
        n_entries=len(eqdsk[i+start_line])/entrylength
        F[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
    start_line=i+start_line+1

    for i in lines:
        n_entries=len(eqdsk[i+start_line])/entrylength
        p[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
    start_line=i+start_line+1

    for i in lines:
        n_entries=len(eqdsk[i+start_line])/entrylength
        ffprime[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
    start_line=i+start_line+1

    for i in lines:
        n_entries=len(eqdsk[i+start_line])/entrylength
        pprime[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
    start_line=i+start_line+1

    lines_twod=range(nw*nh/5)
    if nw*nh%5!=0: lines_twod=range(nw*nh/5+1)
    for i in lines_twod:
        n_entries=len(eqdsk[i+start_line])/entrylength
        psirz_1d[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
    start_line=i+start_line+1
    psirz=psirz_1d.reshape(nh,nw)

    for i in lines:
        n_entries=len(eqdsk[i+start_line])/entrylength
        qpsi[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
    start_line=i+start_line+1

    #invert sign of psi if necessary to guarantee increasing values for interpolation
    if psisep<psiax:
        psirz=-psirz
        ffprime=-ffprime
        pprime=-pprime
        psiax*=-1
        psisep*=-1

    print('psi-axis = {:.3e} Wb/rad'.format(psiax))
    print('psi-sep = {:.3e} Wb/rad'.format(psisep))

    # plot eqdsk profiles
    psinorm_uni_grid = np.linspace(0, 1, nw)
    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(psinorm_uni_grid, F)
    plt.subplot(3,2,2)
    plt.plot(psinorm_uni_grid, ffprime)
    plt.subplot(3,2,3)
    plt.plot(psinorm_uni_grid, p)
    plt.subplot(3,2,4)
    plt.plot(psinorm_uni_grid, pprime)
    plt.subplot(3,2,5)
    plt.plot(psinorm_uni_grid, qpsi)


    # construct rectangular RZ grid
    dw=rdim/(nw-1)
    dh=zdim/(nh-1)
    rgrid = np.array([rmin+i*dw for i in range(nw)])
    zgrid = np.array([zmid-zdim/2.+i*dh for i in range(nh)])
    plt.figure()
    plt.contour(rgrid,zgrid,psirz,70)
    plt.gca().set_aspect('equal')


    #create 5th order 2D spline representation of Psi(R,Z)
    interpol_order=3
    # spline object for psi on RZ grid
    psi_spl=RBS(zgrid,rgrid,psirz,kx=interpol_order,ky=interpol_order)

    #linear grid of psi, on which all 1D fields are defined
    psi_uni_grid = np.linspace(psiax,psisep,nw)

    t1 = np.arctan2(zmid-zdim/2.-zmag,rmin-rmag)
    t2 = np.arctan2(zmid-zdim/2.-zmag,rmin+rdim-rmag)
    t3 = np.arctan2(zmid+zdim/2.-zmag,rmin+rdim-rmag)
    t4 = np.arctan2(zmid+zdim/2.-zmag,rmin-rmag)

    ntheta=150
    theta_arr = np.linspace(-np.pi,np.pi,ntheta)

    #print('Finding flux surface shapes...')
    arr = np.empty(ntheta)
    # flux surface R/Z coords. on uniform flux and theta grids
    R = np.empty((nw,ntheta),dtype=float)
    Z = np.empty((nw,ntheta),dtype=float)
    nr=100

    # loop over theta_arr
    for j in range(ntheta):
        theta=theta_arr[j]
        if theta<t1 or theta>=t4:
            rad=(rmin-rmag)/np.cos(theta)
        if theta<t2 and theta>=t1:
            rad=-(zmag-zmid+zdim/2.)/np.sin(theta)
        if theta<t3 and theta>=t2:
            rad=(rmin+rdim-rmag)/np.cos(theta)
        if theta<t4 and theta>=t3:
            rad=(zmid+zdim/2.-zmag)/np.sin(theta)
        dr = rad/(nr-1)*np.cos(theta)
        dz = rad/(nr-1)*np.sin(theta)
        r_pol = np.array([rmag+i*dr for i in range(nr)])
        z_pol = np.array([zmag+i*dz for i in range(nr)])
        psi_rad = psi_spl.ev(z_pol,r_pol)
        psi_rad[0] = psiax
        #must restrict interpolation range because of non-monotonic psi around coils
        end_ind=0
        for i in range(nr-1):
            if psi_rad[i]<psisep+(psisep-psiax)*0.1:
                if psi_rad[i+1]<=psi_rad[i] and i < nr-2:
                        psi_rad[i+1] = 0.5*(psi_rad[i]+psi_rad[i+2])
            else:
                break
            end_ind+=1

        # interp objects for indices
        psi_int=interp1d(psi_rad[:end_ind+1],range(end_ind+1),kind=interpol_order)
        indsep=int(psi_int(psisep))+3  # near psi-grid index for separatrix
        R_int=interp1d(psi_rad[:indsep],r_pol[:indsep],kind=interpol_order)
        R[:,j]=R_int(psi_uni_grid) # R coords of FS grid at fixed theta
        Z_int=interp1d(psi_rad[:indsep],z_pol[:indsep],kind=interpol_order)
        Z[:,j]=Z_int(psi_uni_grid) # Z coords of FS grid at fixed theta
        arr[j]=indsep*rad

    #print('\nFinding flux surface centers...')
    #find average elevation for all flux surfaces
    Z_avg = np.empty(nw,dtype=float)
    for i in range(nw):
        ds = np.empty(ntheta,dtype=float)
        ds[1:ntheta-1]=0.5*np.sqrt((R[i,2:ntheta]-R[i,0:ntheta-2])**2+(Z[i,2:ntheta]-Z[i,0:ntheta-2])**2)
        ds[0]=0.5*np.sqrt((R[i,1]-R[i,-1])**2+(Z[i,1]-Z[i,-1])**2)
        ds[-1]=0.5*np.sqrt((R[i,0]-R[i,-2])**2+(Z[i,0]-Z[i,-2])**2)
        Z_avg[i] = np.average(Z[i,:],weights=ds)

    #find R0 for all flux surfaces
    R0 = np.empty(nw,dtype=float)
    R0[0]=rmag
    r_avg = np.empty(nw,dtype=float)
    r_avg[0]=0.
    r_maxmin = np.empty(nw,dtype=float)
    r_maxmin[0]=0.
    # loop over flux grid
    for i in range(1,nw):
        #low field side
        R_array=R[i,ntheta/4:3*ntheta/4]
        Z_array=Z[i,ntheta/4:3*ntheta/4]
        R_max = np.amax(R_array)
        Z_int = interp1d(Z_array, R_array, kind=interpol_order)
        R_out = Z_int(Z_avg[i])
        #Z_int=interp1d(Z_array,range(ntheta/2),kind=interpol_order)
        #ind_Zavg=Z_int(Z_avg[i])
        #R_int=interp1d(range(ntheta/2),R_array,kind=interpol_order)
        #R_out=R_int(ind_Zavg)

        #high field side
        R_array = np.roll(R[i,:-1],ntheta/2)[ntheta/4:3*ntheta/4]
        Z_array = np.roll(Z[i,:-1],ntheta/2)[ntheta/4:3*ntheta/4]
        R_min = np.amin(R_array)
        #have to use negative Z_array here to have increasing order
        Z_int=interp1d(-Z_array,R_array,kind=interpol_order)
        R_in=Z_int(-Z_avg[i])
        #R_int=interp1d(range(ntheta/2),R_array,kind=interpol_order)
        #R_in=R_int(ind_Zavg)

        R0[i]=0.5*(R_out+R_in) # R_maj at Z_avg
        r_avg[i]=0.5*(R_out-R_in) # r_min at Z_avg
        r_maxmin[i]=0.5*(R_max-R_min) #

    print('R0_lcfs = {:.3g} m'.format(R0[-1]))
    print('a_lcfs = {:.3g} m'.format(r_avg[-1]))
    print('eps_lcfs = {:.3g}'.format(r_avg[-1]/R0[-1]))

    #modified theta grid for each flux surface
    #arrays equidistant on modified theta grid are marked by 'tm' index!!!

    #print('\nExamine %d flux surfaces around position r/a=%7.4g...' %(pw,r_a))
    psi_uni_grid_spl=US(r_avg,psi_uni_grid,k=interpol_order,s=1e-5)
    #rmaxmin_spl=US(psi_uni_grid,r_maxmin,k=interpol_order,s=1e-5)
    q_spl=US(r_avg,qpsi,k=interpol_order,s=1e-5)
    q_spl_psi=US(psi_uni_grid,qpsi,k=interpol_order,s=1e-5)
    R0_spl=US(r_avg,R0,k=interpol_order,s=1e-5)
    F_spl=US(r_avg,F,k=interpol_order,s=1e-5)
    p_spl=US(r_avg,p,k=interpol_order,s=1e-5)
    pprime_spl=US(r_avg,pprime,k=interpol_order,s=1e-4)
    r=r_a*r_avg[-1]  # r/a * a = r; FS minor radius
    psi=float(psi_uni_grid_spl(r))  # psi at FS
    psi_N=(psi-psiax)/(psisep-psiax)  # psi-norm at FS
    R0_pos=float(R0_spl(r))  # R_maj of FS
    F_pos=float(F_spl(r))  # F of FS
    p_pos=float(p_spl(r))
    pprime_pos=float(pprime_spl(r))
    Bref_miller=F_pos/R0_pos
    q_pos = float(q_spl_psi(psi))

    #create rho_tor grid
    nfine = nw*10
    psi_fine_grid = np.linspace(psiax,psisep,nfine)
    phi_fine = np.empty((nfine),dtype=float)
    phi_fine[0]=0.

    for i in range(1,nfine):
        phi_fine[i] = np.trapz(q_spl_psi(psi_fine_grid[:i+1]), psi_fine_grid[:i+1])
    rho_tor_fine = np.sqrt(phi_fine/phi_fine[-1])
    rho_tor_spl=US(psi_fine_grid,rho_tor_fine,k=interpol_order,s=1e-5)
    rho_tor = float(rho_tor_spl(psi))

    ravg_spl=US(psi_uni_grid,r_avg,k=interpol_order,s=1e-5)
    drdpsi = float(ravg_spl.derivatives(psi)[1])
    omp = -float((r_avg[-1]/p_pos)*(pprime_pos/drdpsi))


    print('*** FS at r/a = {:.2f} ***'.format(r_a))
    print('r_min = {:.3f} m'.format(r))
    print('R_maj = {:.3f} m'.format(R0_pos))
    print('R_max = R_maj+r_min = {:.3f} m'.format(r+R0_pos))
    print('eps = {:.3f}'.format(r/R0_pos))
    print('q = {:.3f}'.format(q_pos))
    print('psi = {:.3e} Wb/rad'.format(psi))
    print('psi_N = {:.3f}'.format(psi_N))
    print('rho_tor_N = {:.3g}'.format(rho_tor))
    print('p = {:.3g} Pa'.format(p_pos))
    print('dp/dpsi = {:.3g} Pa/(Wb/rad)'.format(pprime_pos))
    print('dr/dpsi = {:.3g} m/(Wb/rad)'.format(drdpsi))
    print('omp = {:.3g} (with Lref=a)'.format(omp))


    #find psi index of interest (for the specified r/a position)
    poi_ind=find(r_a,r_avg/r_avg[-1])
    pw=(nw/8/2)*2 #psi-width, number of flux surfaces around position of interest

    #print('Coordinates: r=%8.5g, psi=%8.5g, psi_N=%8.5g, r/R0=%8.5g, rho_tor=%8.5g, R_fs=%8.5g, R_fs+r=%8.5g'%\
    #(r,psi,psi_N,r/R0_pos,rho_tor, R0_pos, R0_pos+r))
    psi_stencil=range(poi_ind-pw/2,poi_ind+pw/2)
    if psi_stencil[0]<1: psi_stencil=[psi_stencil[i]+1-psi_stencil[0] for i in range(len(psi_stencil))]
    if psi_stencil[-1]>nw-1: psi_stencil=[psi_stencil[i]-(psi_stencil[-1]-nw+1) for i in range(len(psi_stencil))]
    R_tm = np.empty((pw,ntheta),dtype=float)
    Z_tm = np.empty((pw,ntheta),dtype=float)
    R_extended = np.empty(2*ntheta-1,dtype=float)
    Z_extended = np.empty(2*ntheta-1,dtype=float)
    #theta_mod[0]=theta_arr
    #R_tm[0]=R[0]
    #Z_tm[0]=Z[0]
    theta_tmp = np.linspace(-2.*np.pi,2*np.pi,2*ntheta-1)

    #print('Interpolating to flux-surface dependent (proper) theta grid...')
    for i in psi_stencil:
        #sys.stdout.write('\r Finished %4.1f%%.' %(psi_stencil.index(i)*100./(len(psi_stencil)-1)))
        #sys.stdout.flush()
        imod=i-psi_stencil[0]
        #print 'Finished %4.1f%%.' %(float(i)/(pw-1)*100)
        R_extended[0:(ntheta-1)/2]=R[i,(ntheta+1)/2:-1]
        R_extended[(ntheta-1)/2:(3*ntheta-3)/2]=R[i,:-1]
        R_extended[(3*ntheta-3)/2:]=R[i,0:(ntheta+3)/2]
        Z_extended[0:(ntheta-1)/2]=Z[i,(ntheta+1)/2:-1]
        Z_extended[(ntheta-1)/2:(3*ntheta-3)/2]=Z[i,:-1]
        Z_extended[(3*ntheta-3)/2:]=Z[i,0:(ntheta+3)/2]
        #for j in range(ntheta):
        theta_mod_ext = np.arctan2(Z_extended-Z_avg[i],R_extended-R0[i])
        #introduce 2pi shifts to theta_mod_ext
        for ind in range(ntheta):
            if theta_mod_ext[ind+1]<0. and theta_mod_ext[ind]>0. and abs(theta_mod_ext[ind+1]-theta_mod_ext[ind])>np.pi:
                lshift_ind=ind
            if theta_mod_ext[-ind-1]>0. and theta_mod_ext[-ind]<0. and abs(theta_mod_ext[-ind-1]-theta_mod_ext[-ind])>np.pi:
                rshift_ind=ind
        theta_mod_ext[-rshift_ind:]+=2.*np.pi
        theta_mod_ext[:lshift_ind+1]-=2.*np.pi
        #print theta_mod, theta_arr
    #    plot(theta_mod_ext)
    #    plot(theta_tmp)
    #    show()
        theta_int=interp1d(theta_mod_ext,theta_tmp,kind=interpol_order)
        #theta_orig_tm=theta_int(theta_arr)
        R_int=interp1d(theta_mod_ext,R_extended,kind=interpol_order)
        Z_int=interp1d(theta_mod_ext,Z_extended,kind=interpol_order)
        R_tm[imod]=R_int(theta_arr)
        Z_tm[imod]=Z_int(theta_arr)
    #    plot(R_tm[imod],Z_tm[imod])
    #gca().set_aspect('equal')

    #now we have the flux surfaces on a symmetric grid in theta (with reference to R0(r), Z0(r))
    #symmetrize flux surfaces
    #figure()
    R_sym = np.empty((pw,ntheta),dtype=float)
    Z_sym = np.empty((pw,ntheta),dtype=float)
    for i in psi_stencil:
        imod=i-psi_stencil[0]
        Z_sym[imod,:]=0.5*(Z_tm[imod,:]-Z_tm[imod,::-1])+Z_avg[i]
        R_sym[imod,:]=0.5*(R_tm[imod,:]+R_tm[imod,::-1])
    #    plot(R_sym[imod],Z_sym[imod])
    #gca().set_aspect('equal')
    #show()
    #plt.figure()
    kappa = np.empty(pw,dtype=float)
    delta_upper = np.empty(pw,dtype=float)
    delta_lower = np.empty(pw,dtype=float)
    for i in psi_stencil:
        imod=i-psi_stencil[0]
        #calculate delta
        stencil_width=ntheta/10
        for o in range(2):
            if o:
                ind = np.argmax(Z_sym[imod])
                section=range(ind+stencil_width/2,ind-stencil_width/2,-1)
            else:
                ind = np.argmin(Z_sym[imod])
                section=range(ind-stencil_width/2,ind+stencil_width/2)
            x=R_sym[imod,section]
            y=Z_sym[imod,section]
            y_int=interp1d(x,y,kind=interpol_order)
            x_fine = np.linspace(np.amin(x),np.amax(x),stencil_width*100)
            y_fine=y_int(x_fine)
            if o:
                x_at_extremum=x_fine[np.argmax(y_fine)]
                delta_upper[imod]=(R0[i]-x_at_extremum)/r_avg[i]
                Z_max = np.amax(y_fine)
            else:
                x_at_extremum=x_fine[np.argmin(y_fine)]
                delta_lower[imod]=(R0[i]-x_at_extremum)/r_avg[i]
                Z_min = np.amin(y_fine)
        #calculate kappa
        kappa[imod]=(Z_max-Z_min)/2./r_avg[i]

    #linear extrapolation (in psi) for axis values
    #delta_upper[0]=2*delta_upper[1]-delta_upper[2]
    #delta_lower[0]=2*delta_lower[1]-delta_lower[2]
    #kappa[0]=2*kappa[1]-kappa[2]
    #zeta[0]=2*zeta[1]-zeta[2]
    delta = np.empty(pw,dtype=float)
    delta=0.5*(delta_upper+delta_lower)

    #calculate zeta
    zeta_arr = np.empty((pw,4),dtype=float)
    zeta = np.empty(pw,dtype=float)
    for i in psi_stencil:
        imod=i-psi_stencil[0]
        x = np.arcsin(delta[imod])
        #find the points that correspond to Miller-theta=+-np.pi/4,+-3/4*np.pi and extract zeta from those
        for o in range(4):
            if o==0:
                val = np.pi/4.
                searchval = np.cos(val+x/np.sqrt(2))
                searcharr=(R_sym[imod]-R0[i])/r_avg[i]
            elif o==1:
                val=3.*np.pi/4
                searchval = np.cos(val+x/np.sqrt(2))
                searcharr=(R_sym[imod]-R0[i])/r_avg[i]
            elif o==2:
                val=-np.pi/4.
                searchval = np.cos(val-x/np.sqrt(2))
                searcharr=(R_sym[imod]-R0[i])/r_avg[i]
            elif o==3:
                val=-3.*np.pi/4
                searchval = np.cos(val-x/np.sqrt(2))
                searcharr=(R_sym[imod]-R0[i])/r_avg[i]
            if o in [0,1]:
                searcharr2=searcharr[ntheta/2:]
                ind=find(searchval,searcharr2)+ntheta/2
            else:
                searcharr2=searcharr[0:ntheta/2]
                ind=find(searchval,searcharr2)
    #        print o,ind
            section=range(ind-stencil_width/2,ind+stencil_width/2)
            theta_sec=theta_arr[section]
            if o in [0,1]:
                theta_int=interp1d(-searcharr[section],theta_sec,kind=interpol_order)
                theta_of_interest=theta_int(-searchval)
            else:
                theta_int=interp1d(searcharr[section],theta_sec,kind=interpol_order)
                theta_of_interest=theta_int(searchval)
            Z_sec=Z_sym[imod,section]
            Z_sec_int=interp1d(theta_sec,Z_sec,kind=interpol_order)
    #        print searchval,val, theta_sec
            Z_val=Z_sec_int(theta_of_interest)
            zeta_arr[imod,o] = np.arcsin((Z_val-Z_avg[i])/kappa[imod]/r_avg[i])
        zeta_arr[imod,1] = np.pi-zeta_arr[imod,1]
        zeta_arr[imod,3]=-np.pi-zeta_arr[imod,3]
    #    print zeta_arr[i]
        zeta[imod]=0.25*(np.pi+zeta_arr[imod,0]-zeta_arr[imod,1]-zeta_arr[imod,2]+zeta_arr[imod,3])



    #Bref_efit=abs(F[0]/R0[0])
    #Lref_efit = np.sqrt(2*abs(phi_fine[-1])/Bref_efit)

    dq_dr_avg = np.empty(pw,dtype=float)
    dq_dpsi = np.empty(pw,dtype=float)
    drR = np.empty(pw,dtype=float)
    s_kappa = np.empty(pw,dtype=float)
    s_delta = np.empty(pw,dtype=float)
    s_zeta = np.empty(pw,dtype=float)
    kappa_spl=US(r_avg[psi_stencil],kappa,k=interpol_order,s=1e-5)
    delta_spl=US(r_avg[psi_stencil],delta,k=interpol_order,s=1e-5)
    zeta_spl=US(r_avg[psi_stencil],zeta,k=interpol_order,s=1e-5)
    amhd = np.empty(pw,dtype=float)
    #amhd_Miller = np.empty(pw,dtype=float)
    Vprime = np.empty(pw,dtype=float)
    dV_dr = np.empty(pw,dtype=float)
    #V = np.empty(pw,dtype=float)
    #V_manual = np.empty(pw,dtype=float)
    r_FS = np.empty(pw,dtype=float)
    for i in psi_stencil:
        imod=i-psi_stencil[0]
        Vprime[imod]=abs(sum(qpsi[i]*R_sym[imod]**2/F[i])*4*np.pi**2/ntheta)
        dV_dr[imod]=abs(sum(qpsi[i]*R_sym[imod]**2/F[i])*4*np.pi**2/ntheta)/ravg_spl.derivatives(psi_uni_grid[i])[1]
    #    V[imod]=trapz(Vprime[:imod+1],psi_uni_grid[psi_stencil])
        r_FS[imod] = np.average(np.sqrt((R_sym[imod]-R0[i])**2+(Z_sym[imod]-Z_avg[i])**2),weights=qpsi[i]*R_sym[imod]**2/F[i])
        amhd[imod]=-qpsi[i]**2*R0[i]*pprime[i]*8*np.pi*1e-7/Bref_miller**2/ravg_spl.derivatives(psi_uni_grid[i])[1]
    #    amhd_Miller[imod]=-2*Vprime[imod]/(2*np.pi)**2*(V[imod]/2/np.pi**2/R0[i])**0.5*4e-7*np.pi*pprime[i]
        dq_dr_avg[imod]=q_spl.derivatives(r_avg[i])[1]
        dq_dpsi[imod]=q_spl_psi.derivatives(psi_uni_grid[i])[1]
        drR[imod]=R0_spl.derivatives(r_avg[i])[1]
        s_kappa[imod]=kappa_spl.derivatives(r_avg[i])[1]*r_avg[i]/kappa[imod]
        s_delta[imod]=delta_spl.derivatives(r_avg[i])[1]*r_avg[i]/np.sqrt(1-delta[imod]**2)
        s_zeta[imod]=zeta_spl.derivatives(r_avg[i])[1]*r_avg[i]
    amhd_spl=US(r_avg[psi_stencil],amhd,k=interpol_order,s=1e-5)
    #rFS_spl=US(r_avg[psi_stencil],r_FS,k=interpol_order,s=1e-5)
    drR_spl=US(r_avg[psi_stencil],drR,k=interpol_order,s=1e-5)
    #Zavg_spl=US(r_avg,Z_avg,k=interpol_order,s=1e-5)

#    plt.figure()
#    plt.subplot(3,2,1)
#    plt.plot(r_avg[psi_stencil],kappa)
#    plt.title('Elongation')
#    plt.xlabel(r'$r_{avg}$',fontsize=14)
#    plt.ylabel(r'$\kappa$',fontsize=14)
#    plt.axvline(r,0,1,ls='--',color='k',lw=2)
#
#    plt.subplot(3,2,2)
#    plt.plot(r_avg[psi_stencil],s_kappa)
#    plt.title('Elongation (Derivative)')
#    plt.xlabel(r'$r_{avg}$',fontsize=14)
#    plt.ylabel(r'$s_\kappa$',fontsize=14)
#    plt.axvline(r,0,1,ls='--',color='k',lw=2)
#
#    plt.subplot(3,2,3)
#    plt.plot(r_avg[psi_stencil],delta)
#    plt.title('Triangularity')
#    plt.xlabel(r'$r_{avg}$',fontsize=14)
#    plt.ylabel(r'$\delta$',fontsize=14)
#    plt.axvline(r,0,1,ls='--',color='k',lw=2)
#
#    plt.subplot(3,2,4)
#    plt.plot(r_avg[psi_stencil],s_delta)
#    plt.title('Triangularity (Derivative)')
#    plt.xlabel(r'$r_{avg}$',fontsize=14)
#    plt.ylabel(r'$s_\delta$',fontsize=14)
#    plt.axvline(r,0,1,ls='--',color='k',lw=2)
#
#    plt.subplot(3,2,5)
#    plt.plot(r_avg[psi_stencil],zeta)
#    plt.title('Squareness')
#    plt.xlabel(r'$r_{avg}$',fontsize=14)
#    plt.ylabel(r'$\zeta$',fontsize=14)
#    plt.axvline(r,0,1,ls='--',color='k',lw=2)
#
#    plt.subplot(3,2,6)
#    plt.plot(r_avg[psi_stencil],s_zeta)
#    plt.title('Squareness (Derivative)')
#    plt.xlabel(r'$r_{avg}$',fontsize=14)
#    plt.ylabel(r'$s_\zeta$',fontsize=14)
#    plt.axvline(r,0,1,ls='--',color='k',lw=2)

    #select a given flux surface
    ind=poi_ind-psi_stencil[0]#find(r_a,r_avg/r_avg[-1])
    #Lref=R0_pos
    print('\n\nShaping parameters for flux surface r=%9.5g, r/a=%9.5g:' %(r,r_a))
    #print 'r_FS= %9.5g (flux-surface averaged radius)\n' %rFS_spl(r)
    print('Copy the following block into a GENE parameters file:\n')
    print('trpeps  = %9.5g' %(r/R0_pos))
    print('q0      = %9.5g' %q_spl(r))
    print('shat    = %9.5g !(defined as r/q*dq_dr)' %(r/q_spl(r)*q_spl.derivatives(r)[1]))
    #print 'shat=%9.5g (defined as (psi-psiax)/q*dq_dpsi)' %((psi-psi_uni_grid[0])/q_spl(r)*q_spl_psi.derivatives(psi)[1])
    print('amhd    = %9.5g' %amhd_spl(r))
    #print 'amhd_Miller=%9.5g' %amhd_Miller[ind]
    print('drR     = %9.5g' %drR_spl(r))
    print('kappa   = %9.5g' %kappa_spl(r))
    print('s_kappa = %9.5g' % (kappa_spl.derivatives(r)[1]*r/kappa_spl(r)))
    print('delta   = %9.5g' %delta_spl(r))
    print('s_delta = %9.5g' %(delta_spl.derivatives(r)[1]*r/np.sqrt(1-delta_spl(r)**2)))
    print('zeta    = %9.5g' %zeta_spl(r))
    print('s_zeta  = %9.5g' %(zeta_spl.derivatives(r)[1]*r))
    print('minor_r = %9.5g' %(1.0))
    print('major_R = %9.5g' %(R0_pos/r*r_a))
    print('\nFor normalization to major radius, set instead:')
    print('minor_r = %9.5g' %(r/R0_pos/r_a))
    print('major_R = %9.5g' %(1.0))
    print('The same conversion factor must be considered for input frequencies and gradients.')
    print('\nAdditional information:')
    print('Lref        = %9.5g !for Lref=a convention' %r_avg[-1])
    print('Lref        = %9.5g !for Lref=R0 convention' %R0_pos)
    print('Bref        = %9.5g' %Bref_miller)
    #minor radius at average flux surface elevation (GYRO definition)
#    print('\na (avg elev)= %9.5g' %r_avg[-1])
#    print('R0          = %9.5g' %R0_pos)
#    print('R0+a        = %9.5g' %(R0_pos+r_avg[-1]))
#    print('Z0          = %9.5g' %Zavg_spl(r))
#    print('Lref_efit   = %9.5g' %Lref_efit)
#    print('Bref_efit   = %9.5g' %Bref_efit)
#    print('B_unit(GYRO)= %9.5g' %(q_spl(r)/r/ravg_spl.derivatives(psi)[1]))
#    #print 'Vprime=%9.5g; dV_dr=%9.5g' %(Vprime[ind],dV_dr[ind])
#    #print 'V=%9.5g' %V[ind]
#    #minor radius defined by 0.5(R_max-R_min), where R_max and R_min can have any elevation
#    print('a_maxmin    = %9.5g' %r_maxmin[-1])
#    print('dpsi/dr     = %9.5g' %(1./ravg_spl.derivatives(psi)[1]))
#    #print 'dr_maxmin/dr     = %9.5g' %(rmaxmin_spl.derivatives(psi)[1]/ravg_spl.derivatives(psi)[1])
#    print('drho_tor/dr = %9.5g' %(rho_tor_spl.derivatives(psi)[1]/ravg_spl.derivatives(psi)[1]))
#    print('Gradient conversion omt(rho_tor) -> a/LT; factor = %9.5g' %(r_avg[-1]*(rho_tor_spl.derivatives(psi)[1]/ravg_spl.derivatives(psi)[1])))
#    print('Gradient conversion omt(rho_tor) -> R/LT; factor = %9.5g' %(R0_pos*(rho_tor_spl.derivatives(psi)[1]/ravg_spl.derivatives(psi)[1])))


#    plt.figure()
#    plt.plot(R_tm[ind],Z_tm[ind],'k-',lw=2,label='original')
#    plt.plot(R_sym[ind],Z_sym[ind],'r-',lw=2,label='symmetrized')
#    #for v in range(-10,10,4):
#    #zeta_test=-0.011086#zeta[ind]
#    plt.plot(R0_pos+r*np.cos(theta_arr+np.arcsin(delta_spl(r))*np.sin(theta_arr)),Zavg_spl(r)+kappa_spl(r)*r*np.sin(theta_arr+zeta_spl(r)*np.sin(2*theta_arr)),label='Miller')
#    plt.title('Flux surface shapes')
#    plt.xlabel('$R$/m',fontsize=14)
#    plt.ylabel('$Z$/m',fontsize=14)
#    plt.gca().set_aspect('equal')
#    plt.legend(loc=10,prop={'size':10})


if __name__=='__main__':
    miller()