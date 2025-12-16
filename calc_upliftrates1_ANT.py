#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

# =============================================== #
# JUST EXTRACTS AND CALCULATES UPLIFT RATES FOR
# ONE OF THE FIGURES.
# THIS IS PAIRED WITH calc_upliftrates1_NA.py
# Just easier since they use different projections
# =============================================== #

# inputs:
slat  = -76.5
slon  = -105.
#slat  = -78.513171
#slon  = -125
fruns = ['E05/','EA5/','NCR/','F32/']
ftime = ['tt_mE5','tt_mE5','tt_mFN','tt_mFN']
nrun  = len(fruns)

col1d = '#1f78b4'
col3d = '#33a02c'

lat   = np.loadtxt('RESULTS/EM1D_E05/lat.dat')
lon   = np.loadtxt('RESULTS/EM1D_E05/lon.dat')
# truncate to remove Nans:
ilat0 = 50  # this is ~-65 lat
ilat1 = 250 # this is ~-85 lat
lat   = lat[ilat0:ilat1]
nlat  = len(lat)
nlon  = len(lon)

f1dm  = 'RESULTS/EM1D_'
f1dt  = 'RESULTS/EM1T_'
f3dm  = 'RESULTS/EM3D_'
f3dt  = 'RESULTS/EM3T_'

rsls1d = []
rsls1t = []
rsls3d = []
rsls3t = []

ts     = []

for i in range(nrun):
  t = np.loadtxt('../Data/'+ftime[i],skiprows=1)
  ts.append(t)
  nt = len(t)
  rsl1d = np.zeros((nt,nlat,nlon))
  rsl1t = np.zeros((nt,nlat,nlon))
  rsl3d = np.zeros((nt,nlat,nlon))
  rsl3t = np.zeros((nt,nlat,nlon))

  for j in range(nt):
    print('Loading',i,'run;',j,'time')
    istr = 'RSLreg_'+str(j+1).zfill(2)
    data = np.loadtxt(f1dm+fruns[i]+istr)
    rsl1d[j,:,:] = data[ilat0:ilat1,:]
    data = np.loadtxt(f1dt+fruns[i]+istr)
    rsl1t[j,:,:] = data[ilat0:ilat1,:]
    data = np.loadtxt(f3dm+fruns[i]+istr)
    rsl3d[j,:,:] = data[ilat0:ilat1,:]
    data = np.loadtxt(f3dt+fruns[i]+istr)
    rsl3t[j,:,:] = data[ilat0:ilat1,:]
    
  rsls1d.append(rsl1d)
  rsls1t.append(rsl1t)
  rsls3d.append(rsl3d)
  rsls3t.append(rsl3t)

# now interpolate onto single site:
rslt1d = []
rslt1t = []
rslt3d = []
rslt3t = []

for i in range(nrun):
  nt = len(ts[i])
  rsl1d = np.zeros(nt)
  rsl1t = np.zeros(nt)
  rsl3d = np.zeros(nt)
  rsl3t = np.zeros(nt)
  
  for j in range(nt):
    f = si.interp2d(lon,lat,rsls1d[i][j,:,:])
    rsl1d[j] = f(slon,slat)
    f = si.interp2d(lon,lat,rsls1t[i][j,:,:])
    rsl1t[j] = f(slon,slat)
    f = si.interp2d(lon,lat,rsls3d[i][j,:,:])
    rsl3d[j] = f(slon,slat)
    f = si.interp2d(lon,lat,rsls3t[i][j,:,:])
    rsl3t[j] = f(slon,slat)

  rslt1d.append(rsl1d)
  rslt1t.append(rsl1t)
  rslt3d.append(rsl3d)
  rslt3t.append(rsl3t)

# first let's smooth the trends for F32 and NCR
# and use these RSLs to get the rates.

# now get rates:

drslt1d = []
drslt1t = []
drslt3d = []
drslt3t = []

newt    = []

for i in range(nrun):
  nt = len(ts[i])
  drsl1d = np.zeros(nt-1)
  drsl1t = np.zeros(nt-1)
  drsl3d = np.zeros(nt-1)
  drsl3t = np.zeros(nt-1)
  t      = np.zeros(nt-1)

  for j in range(nt-1):
    dt = ts[i][j+1]-ts[i][j]
    t[j] = (ts[i][j+1]+ts[i][j])*0.5
    drsl1d[j] = (rslt1d[i][j+1]-rslt1d[i][j])/dt
    drsl1t[j] = (rslt1t[i][j+1]-rslt1t[i][j])/dt
    drsl3d[j] = (rslt3d[i][j+1]-rslt3d[i][j])/dt
    drsl3t[j] = (rslt3t[i][j+1]-rslt3t[i][j])/dt

  drslt1d.append(drsl1d)
  drslt1t.append(drsl1t)
  drslt3d.append(drsl3d)
  drslt3t.append(drsl3t)
  newt.append(t)


# SAVE OUTPUT:
data      = np.zeros((len(newt[0]),5))
data[:,0] = newt[0]
data[:,1] = drslt1d[0]
data[:,2] = drslt1t[0]
data[:,3] = drslt3d[0]
data[:,4] = drslt3t[0]
np.savetxt('plotdata_upliftrates_ANT_E05.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')

data      = np.zeros((len(newt[1]),5))
data[:,0] = newt[1]
data[:,1] = drslt1d[1]
data[:,2] = drslt1t[1]
data[:,3] = drslt3d[1]
data[:,4] = drslt3t[1]
np.savetxt('plotdata_upliftrates_ANT_EA5.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')

data      = np.zeros((len(newt[2]),5))
data[:,0] = newt[2]
data[:,1] = drslt1d[2]
data[:,2] = drslt1t[2]
data[:,3] = drslt3d[2]
data[:,4] = drslt3t[2]
np.savetxt('plotdata_upliftrates_ANT_NCR.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')

data      = np.zeros((len(newt[3]),5))
data[:,0] = newt[3]
data[:,1] = drslt1d[3]
data[:,2] = drslt1t[3]
data[:,3] = drslt3d[3]
data[:,4] = drslt3t[3]
np.savetxt('plotdata_upliftrates_ANT_F32.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')


#RSL

data      = np.zeros((len(ts[0]),5))
data[:,0] = ts[0]
data[:,1] = rslt1d[0]
data[:,2] = rslt1t[0]
data[:,3] = rslt3d[0]
data[:,4] = rslt3t[0]
np.savetxt('plotdata_RSL_ANT_E05.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')

data      = np.zeros((len(ts[1]),5))
data[:,0] = ts[1]
data[:,1] = rslt1d[1]
data[:,2] = rslt1t[1]
data[:,3] = rslt3d[1]
data[:,4] = rslt3t[1]
np.savetxt('plotdata_RSL_ANT_EA5.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')

data      = np.zeros((len(ts[2]),5))
data[:,0] = ts[2]
data[:,1] = rslt1d[2]
data[:,2] = rslt1t[2]
data[:,3] = rslt3d[2]
data[:,4] = rslt3t[2]
np.savetxt('plotdata_RSL_ANT_NCR.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')

data      = np.zeros((len(ts[3]),5))
data[:,0] = ts[3]
data[:,1] = rslt1d[3]
data[:,2] = rslt1t[3]
data[:,3] = rslt3d[3]
data[:,4] = rslt3t[3]
np.savetxt('plotdata_RSL_ANT_F32.dat',data,fmt='%e',\
  header = 'time from 2015 (y), 1D, 1T, 3D, 3T')
