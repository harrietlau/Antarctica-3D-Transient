#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

# =============================================== #
# inputs:

# Different time steps for pairs of ice histories.

# Richmond Gulf (Hudson Bay)
slat  = 56.2494
slon  = -76.2942
slon1 = 360.+slon

# pre-history:
frun0  = ['EM_3D/','EM_3T/']
ftime0 = '../Data/tt_S6G'
# post-history:
frun1  = ['EM3Dm/','EM3Tm/']
ftime1 = '../Data/tt_mE5'

# ice 6G:
fice    = '../Data/icedata_S6G'

col1d = '#1f78b4'
col3d = '#33a02c'

# ===================================================== #
# SL runs: load lat/lon
clat  = np.loadtxt('RESULTS/EM1D_E05/clat.dat')
lat   = 90.-clat
lon   = np.loadtxt('RESULTS/EM1D_E05/elon.dat')
nlat  = len(lat)
nlon  = len(lon)

t0    = np.loadtxt(ftime0,skiprows=1)
t1    = np.loadtxt(ftime1,skiprows=1)
nt0   = len(t0)
nt1   = len(t1)
nt    = nt0+nt1
t     = np.concatenate((t0,t1))
rsl3d = np.zeros((nt,nlat,nlon))
rsl3t = np.zeros((nt,nlat,nlon))

# pre history:
for i in range(nt0):
  fin  = 'RESULTS/'+frun0[0]+'RSL_'+str(i).zfill(3)
  data = np.loadtxt(fin)
  rsl3d[i,:,:] = data

  fin  = 'RESULTS/'+frun0[1]+'RSL_'+str(i).zfill(3)
  data = np.loadtxt(fin)
  rsl3t[i,:,:] = data

# post history
for ii in range(nt1):
  i = i + 1
  fin  = 'RESULTS/'+frun1[0]+'RSL_'+str(ii+1).zfill(2)
  data = np.loadtxt(fin)
  rsl3d[i,:,:] = data

  fin  = 'RESULTS/'+frun1[1]+'RSL_'+str(ii+1).zfill(2)
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

# ICE 6G
f       = open(fice,'r')
jk      = f.readline()
tice    = np.array(f.readline().split())
iceclat = np.array(f.readline().split())
iceelon = np.array(f.readline().split())
tice    = tice.astype(float)
icelat  = 90.-iceclat.astype(float) * 180./np.pi
icelon  = iceelon.astype(float) * 180./np.pi
lines   = f.readlines()
f.close()

ntice   = len(tice)
nlatice = len(icelat)
nlonice = len(icelon)

ice     = np.zeros((ntice,nlatice,nlonice))

for i in range(ntice):
  vals = np.array(lines[i].split())
  vals = vals.astype(float)
  ice[i,:,:] = np.reshape(vals,(nlatice,nlonice))

iceh = np.zeros(ntice)

for i in range(ntice):
  print(i,'of',nt)
  f = si.interp2d(icelon,icelat,ice[i,:,:])
  iceh[i] = f(slon1,slat)

# PLOT ALL
plt.figure(1,figsize=(7,3))

ax1 = plt.gca()
ax1.set_xlabel('time (y)')
ax1.set_ylabel('ice height (m)')
ax1.plot(tice*1000.,iceh,'k-',lw=3)
ax1.set_xlim([-11000,300])
ax1.set_ylim([0,2000])
ax2 = ax1.twinx()

ax2.plot([-100,-100],[-100,-100],'k-',lw=3) # dummy for legend
ax2.plot(t,rsl3dt,'-',color=col3d)
ax2.plot(t,rsl3tt,'--',color=col3d)
ax2.set_ylim([0,500])
ax2.set_xlim([-11000,300])
ax2.set_ylabel('RSL (m)')

plt.legend(['ICE6G','3D RSL','3T RSL'],frameon=False)
plt.tight_layout()

#plt.savefig('RESULTS/PAPER_FIGURES/relaxation.pdf')

plt.show()

i = np.where(t>=0)[0]
plt.plot(t[i],rsl3dt[i])
plt.plot(t[i],rsl3tt[i])
plt.show()




