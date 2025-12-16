#! /usr/bin/env python3

import numpy as np
import sys
import util_SK as u

'''
   Extract SL from output and then
   minus present day for those from LIG
   to present to get RSL.

   Note - LIG to present index from 0 and have
          3 digits, while modern runs start from 
          1 and have 2 digits.
'''

# INPUTS
past = int(sys.argv[1])
rdir = sys.argv[2]
nt   = int(sys.argv[3])
odir = sys.argv[4]


# SETTINGS
gridf = '../Data/aisreg.dat'
fdir  = '../Runs/'
nlat  = 300
nlon  = 720

# BEGIN IMPORTING AND RESHAPING

[x,y,z] = np.loadtxt(gridf,\
                     skiprows=1,unpack=True)
pts0    = u.xyz2latlon(x,y,z)
lats    = np.linspace(-60,-90,nlat)
lons    = np.linspace(-180,180,nlon)

if (past==1):
    rsl   = np.zeros((nt,nlat,nlon))

    for i in range(nt):
        print('Extracting',rdir,i+1,'of',nt)
        istr  = str(i).zfill(3)
        data  = np.loadtxt(fdir+rdir+'Sreg_'+istr)[:,4]
        data  = u.get_ant_map(data,pts0,lats,lons)
        rsl[i,:,:] = data

    # get rsl:
    for i in range(nt):
        rsl[i,:,:] = rsl[i,:,:] - rsl[-1,:,:]
        istr  = str(i).zfill(3)
        np.savetxt(odir+'RSLreg'+istr,rsl[i,:,:],fmt='%f')

    np.savetxt(odir+'lat.dat',lats,fmt='%f')
    np.savetxt(odir+'lon.dat',lons,fmt='%f')

else:
    for i in range(nt):
        print('Extracting',rdir,i+1,'of',nt)
        istr  = str(i+1).zfill(2)
        data  = np.loadtxt(fdir+rdir+'Mreg_'+istr)[:,4]
        data  = u.get_ant_map(data,pts0,lats,lons)
        np.savetxt(odir+'RSLreg_'+istr,data,fmt='%f')        
    np.savetxt(odir+'lat.dat',lats,fmt='%f')
    np.savetxt(odir+'lon.dat',lons,fmt='%f')


