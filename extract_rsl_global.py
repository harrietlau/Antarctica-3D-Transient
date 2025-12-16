#! /usr/bin/env python3

import numpy as np
import sys
import util_SK as u

'''
   Extract SL from output and then
   minus present day for those from LIG
   to present to get RSL

   Note - LIG to present index from 0 and have
          3 digits, while modern runs start from 
          1 and have 2 digits.

'''

# INPUTS
past = int(sys.argv[1])
rdir = sys.argv[2]
nt   = int(sys.argv[3])
ngl  = int(sys.argv[4])
odir = sys.argv[5]


# SETTINGS
fdir = '../Runs/'

# BEGIN IMPORTING AND RESHAPING
if (past==1):
    istr  = str(nt-1).zfill(3)
    data  = np.loadtxt(fdir+rdir+'S_'+istr)[:,4]
    clat,elon,slnow = u.get_map(data,ngl)

    rsl   = np.zeros((nt,len(clat),len(elon)))
    rsl[-1,:,:] = slnow

    for i in range(nt-1):
        print('Extracting',rdir,i+1,'of',nt)
        istr = str(i).zfill(3)
        data = np.loadtxt(fdir+rdir+'S_'+istr)[:,4]
        clat,elon,data = u.get_map(data,ngl)
        rsl[i,:,:] = data

    # get RSL:
    for i in range(nt):
        rsl[i,:,:] = rsl[i,:,:] - rsl[-1,:,:]

    # now save:
    for i in range(nt):
        istr = str(i).zfill(3)
        np.savetxt(odir+'RSL_'+istr,rsl[i,:,:],fmt='%f')
    np.savetxt(odir+'clat.dat',clat*180/np.pi,fmt='%f')
    np.savetxt(odir+'elon.dat',elon*180/np.pi,fmt='%f')

else:
    for i in range(nt):
        print('Extracting',rdir,i+1,'of',nt)
        istr = str(i+1).zfill(2)
        data = np.loadtxt(fdir+rdir+'M_'+istr)[:,4]
        clat,elon,data = u.get_map(data,ngl)
        np.savetxt(odir+'RSL_'+istr,data,fmt='%f')
    np.savetxt(odir+'clat.dat',clat*180/np.pi,fmt='%f')
    np.savetxt(odir+'elon.dat',elon*180/np.pi,fmt='%f')


