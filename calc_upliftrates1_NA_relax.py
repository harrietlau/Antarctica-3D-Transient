#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

# =============================================== #
# inputs:

# Richmond Gulf (Hudson Bay)
slat  = 56.2494
slon  = -76.2942
slon1 = 360.+slon

frun  = ['EM3Dm/','EM3Tm/']
f1    = 'plotdata_upliftrates_NA_m.dat'
f2    = 'plotdata_RSL_NA_m.dat'
ftime = '../Data/tt_mE5'

col1d = '#1f78b4'
col3d = '#33a02c'

# ===================================================== #
# SL runs: load lat/lon
clat  = np.loadtxt('RESULTS/EM1D_E05/clat.dat')
lat   = 90.-clat
lon   = np.loadtxt('RESULTS/EM1D_E05/elon.dat')
nlat  = len(lat)
nlon  = len(lon)

t     = np.loadtxt(ftime,skiprows=1)
nt    = len(t)
rsl3d = np.zeros((nt,nlat,nlon))
rsl3t = np.zeros((nt,nlat,nlon))

for i in range(nt):
  fin  = 'RESULTS/'+frun[0]+'RSL_'+str(i+1).zfill(2)
  data = np.loadtxt(fin)
  rsl3d[i,:,:] = data

  fin  = 'RESULTS/'+frun[1]+'RSL_'+str(i+1).zfill(2)
  data = np.loadtxt(fin)
  rsl3t[i,:,:] = data


# NOW INTERPOLATE:
rsl3dt = np.zeros(nt)
rsl3tt = np.zeros(nt)

for i in range(nt):
  print(i,'of',nt)
  f = si.interp2d(lon,lat,rsl3d[i,:,:])
  rsl3dt[i] = f(slon1,slat)

  f = si.interp2d(lon,lat,rsl3t[i,:,:])
  rsl3tt[i] = f(slon1,slat)


# Get rates:
drsl3d = np.zeros(nt-1)
drsl3t = np.zeros(nt-1)

for i in range(nt-1):
  dt = t[i+1]-t[i]
  drsl3d[i] = (rsl3dt[i+1]-rsl3dt[i])/dt
  drsl3t[i] = (rsl3tt[i+1]-rsl3tt[i])/dt

t1 = (t[0:-1]+t[1:nt])*0.5

# save output
# drsl
data = np.zeros((nt-1,3))
data[:,0] = t1
data[:,1] = drsl3d
data[:,2] = drsl3t
np.savetxt('plotdata_upliftrates_NA_relax.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')

# rsl
data = np.zeros((nt,3))
data[:,0] = t
data[:,1] = rsl3dt
data[:,2] = rsl3tt
np.savetxt('plotdata_RSL_NA_relax.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')
