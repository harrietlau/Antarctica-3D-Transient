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

# post-history:
frun  = ['EM3D_EA5/','EM3T_EA5/']
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

# pre history:
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

# should plot rsl relative to 2300:
rsl3dt0 = rsl3dt-rsl3dt[-1]
rsl3tt0 = rsl3tt-rsl3tt[-1]



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

plt.savefig('RESULTS/PAPER_FIGURES/relaxation.pdf')

plt.show()








# === old:


lat   = np.loadtxt('RESULTS/EM3D_E05/lat.dat')
lon   = np.loadtxt('RESULTS/EM3D_E05/lon.dat')
# truncate to remove Nans:
ilat0 = 50  # this is ~-65 lat
ilat1 = 250 # this is ~-85 lat
lat   = lat[ilat0:ilat1]
nlat  = len(lat)
nlon  = len(lon)

t      = np.loadtxt('../Data/'+ftime,skiprows=1)
nt     = len(t)

# with prestress
rsl3d0 = np.zeros((nt,nlat,nlon))
rsl3t0 = np.zeros((nt,nlat,nlon))
# without prestress
rsl3d1 = np.zeros((nt,nlat,nlon))
rsl3t1 = np.zeros((nt,nlat,nlon))
# different 3D model
rsl3d2 = np.zeros((nt,nlat,nlon))
rsl3t2 = np.zeros((nt,nlat,nlon))


for i in range(nt):
  istr = 'RSLreg_'+str(i+1).zfill(2)
  fnme = 'RESULTS/'+frun0[0]+istr
  data = np.loadtxt(fnme)
  rsl3d0[i,:,:] = data[ilat0:ilat1,:]

  fnme = 'RESULTS/'+frun0[1]+istr
  data = np.loadtxt(fnme)
  rsl3t0[i,:,:] = data[ilat0:ilat1,:]

  fnme = 'RESULTS/'+frun1[0]+istr
  data = np.loadtxt(fnme)
  rsl3d1[i,:,:] = data[ilat0:ilat1,:]
  
  fnme = 'RESULTS/'+frun1[1]+istr
  data = np.loadtxt(fnme)
  rsl3t1[i,:,:] = data[ilat0:ilat1,:]

  fnme = 'RESULTS/'+frun2[0]+istr
  data = np.loadtxt(fnme)
  rsl3d2[i,:,:] = data[ilat0:ilat1,:]
  
  fnme = 'RESULTS/'+frun2[1]+istr
  data = np.loadtxt(fnme)
  rsl3t2[i,:,:] = data[ilat0:ilat1,:]


# time series at single site
rsl3d0t = np.zeros((nt,2))
rsl3t0t = np.zeros((nt,2))
rsl3d1t = np.zeros((nt,2))
rsl3t1t = np.zeros((nt,2))
rsl3d2t = np.zeros((nt,2))
rsl3t2t = np.zeros((nt,2))

for i in range(nt):
  f = si.interp2d(lon,lat,rsl3d0[i,:,:])
  rsl3d0t[i,0] = f(slon[0],slat[0])
  rsl3d0t[i,1] = f(slon[1],slat[1])

  f = si.interp2d(lon,lat,rsl3t0[i,:,:])
  rsl3t0t[i,0] = f(slon[0],slat[0])
  rsl3t0t[i,1] = f(slon[1],slat[1])

  f = si.interp2d(lon,lat,rsl3d1[i,:,:])
  rsl3d1t[i,0] = f(slon[0],slat[0])
  rsl3d1t[i,1] = f(slon[1],slat[1])

  f = si.interp2d(lon,lat,rsl3t1[i,:,:])
  rsl3t1t[i,0] = f(slon[0],slat[0])
  rsl3t1t[i,1] = f(slon[1],slat[1])

  f = si.interp2d(lon,lat,rsl3d2[i,:,:])
  rsl3d2t[i,0] = f(slon[0],slat[0])
  rsl3d2t[i,1] = f(slon[1],slat[1])

  f = si.interp2d(lon,lat,rsl3t2[i,:,:])
  rsl3t2t[i,0] = f(slon[0],slat[0])
  rsl3t2t[i,1] = f(slon[1],slat[1])


drsl3d0 = np.zeros((nt-1,2))
drsl3t0 = np.zeros((nt-1,2))
drsl3d1 = np.zeros((nt-1,2))
drsl3t1 = np.zeros((nt-1,2))
drsl3d2 = np.zeros((nt-1,2))
drsl3t2 = np.zeros((nt-1,2))

t1      = np.zeros(nt-1)

for i in range(nt-1):
  dt = t[i+1]-t[i]
  t1[i] = (t[i]+t[i+1])*0.5
  drsl3d0[i,:] = (rsl3d0t[i+1,:]-rsl3d0t[i,:])/dt
  drsl3t0[i,:] = (rsl3t0t[i+1,:]-rsl3t0t[i,:])/dt

  drsl3d1[i,:] = (rsl3d1t[i+1,:]-rsl3d1t[i,:])/dt
  drsl3t1[i,:] = (rsl3t1t[i+1,:]-rsl3t1t[i,:])/dt

  drsl3d2[i,:] = (rsl3d2t[i+1,:]-rsl3d2t[i,:])/dt
  drsl3t2[i,:] = (rsl3t2t[i+1,:]-rsl3t2t[i,:])/dt


# quick plot, but write out data:

plt.figure(1,figsize=(7,3))

plt.subplot(1,2,1)
plt.plot(t1+2015,-drsl3d0[:,0],'-',color=col3d)
plt.plot(t1+2015,-drsl3d2[:,0],'k-')
plt.legend(['3D model herein','Hay et al. (2015)'],\
  frameon=False)
plt.plot(t1+2015,-drsl3t0[:,0],'--',color=col3d)
plt.plot(t1+2015,-drsl3t2[:,0],'k--')
plt.xlabel('year CE')
plt.ylabel('uplift rate (m/y)')
plt.yticks([0.2,0.4,0.6,0.8])
plt.ylim([0.0,0.8])
plt.xticks([2000,2100,2200,2300])
plt.xlim([2000,2300])



plt.subplot(1,2,2)
plt.plot(t1+2015,drsl3d1[:,0]/drsl3d0[:,0],'-',color=col3d)
plt.plot(t1+2015,drsl3t1[:,0]/drsl3t0[:,0],'--',color=col3d)
plt.legend(['Maxwell','Transient'],frameon=False,loc='lower right')
plt.xlabel('year CE')
plt.ylabel('uplift rate ratio')
plt.xticks([2000,2100,2200,2300])
plt.xlim([2000,2300])
plt.yticks([0.25,0.5,0.75,1.0])
plt.ylim([0,1])

ax = plt.gca()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

plt.tight_layout()

#plt.savefig('RESULTS/PAPER_FIGURES/uplift_sensitivity.pdf')

plt.show()

