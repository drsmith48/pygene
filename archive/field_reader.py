#import subprocess
import numpy as np
#import string
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


nx=128 #x is radial
ny=16 #binormal
nz=24 #parallel
data_size=nx*ny*nz

#for proper integration over z (parallel direction)
jacobian=(1.0+0.18*np.cos(np.arange(-np.pi,np.pi,2*np.pi/nz)))*1.4
	
tot_steps=100
phi_data=np.zeros((tot_steps,nz,ny,nx),dtype=np.complex128)
n_data=np.zeros((tot_steps,nz,ny,nx),dtype=np.complex128)
times=np.zeros((tot_steps),dtype=np.float64)
count=0

f1 = open("field.dat","r")
f2 = open("mom_ions.dat","r")

for count in range(100):
	#potential series
	np.fromfile(f1,count=1,dtype=np.uint32)
	times[count]=np.fromfile(f1,count=1,dtype=np.float64)
	np.fromfile(f1,count=2,dtype=np.uint32)
	phi_data[count,:,:,:]=np.fromfile(f1,count=data_size,dtype=np.complex128).reshape(nz,ny,nx)
	np.fromfile(f1,count=1,dtype=np.uint32)
	#plt.imshow(np.fft.irfft2(np.transpose(np.sum(phi_data[count,:,:,:],0))))
	#plt.pause(0.1)

	np.fromfile(f2,count=1,dtype=np.uint32)
	mom_time=np.fromfile(f2,count=1,dtype=np.float64)
	if times[count] != mom_time:
		print(times[count],mom_time)
		break
	np.fromfile(f2,count=2,dtype=np.uint32)
	n_data[count,:,:,:] = np.fromfile(f2,count=data_size,dtype=np.complex128).reshape(nz,ny,nx)
	np.fromfile(f2,count=2,dtype=np.uint32)
	#other stuff in the mom file
	np.fromfile(f2,count=data_size,dtype=np.complex128)
	np.fromfile(f2,count=2,dtype=np.uint32)
	np.fromfile(f2,count=data_size,dtype=np.complex128)
	np.fromfile(f2,count=2,dtype=np.uint32)
	np.fromfile(f2,count=data_size,dtype=np.complex128)
	np.fromfile(f2,count=2,dtype=np.uint32)
	np.fromfile(f2,count=data_size,dtype=np.complex128)
	np.fromfile(f2,count=2,dtype=np.uint32)
	np.fromfile(f2,count=data_size,dtype=np.complex128)
	np.fromfile(f2,count=1,dtype=np.uint32)
	print(mom_time)
	plt.imshow(np.fft.irfft2(np.transpose(np.sum(n_data[count,:,:,:],0))))
	plt.pause(0.05)

		
