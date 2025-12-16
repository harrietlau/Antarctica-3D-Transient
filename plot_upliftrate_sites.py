#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

# =============================================== #
# inputs:

# Different time steps for pairs of ice histories.

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




plt.figure(1,figsize=(6,8))

# F32 - most aggressive irun = 3
# NCR - irun = 2
# E05 - irun = 0
# EA5 - irun = 1

# TOP LEFT NCR
plt.subplot(4,2,1)
irun = 2
plt.plot(newt[irun]+2015,-drslt1d[irun]*1000,'-',color=col1d)
plt.plot(newt[irun]+2015,-drslt1t[irun]*1000,'--',color=col1d)
plt.plot(newt[irun]+2015,-drslt3d[irun]*1000,'-',color=col3d)
plt.plot(newt[irun]+2015,-drslt3t[irun]*1000,'--',color=col3d)
plt.ylim([0,140])
plt.yticks([0,35,70,105,140])
plt.xlim([2010,2100])
plt.xticks([])
plt.ylabel('uplift rate (mm/y)')


# BOTTOM LEFT F32
plt.subplot(4,2,3)
irun = 3
plt.plot(newt[irun]+2015,-drslt1d[irun]*1000,'-',color=col1d)
plt.plot(newt[irun]+2015,-drslt1t[irun]*1000,'--',color=col1d)
plt.plot(newt[irun]+2015,-drslt3d[irun]*1000,'-',color=col3d)
plt.plot(newt[irun]+2015,-drslt3t[irun]*1000,'--',color=col3d)
plt.ylim([0,600])
plt.yticks([150,300,450,600])
plt.xlim([2010,2100])
plt.xticks([2010,2040,2070,2100])
plt.ylabel('uplift rate (mm/y)')
plt.xlabel('year CE')

# TOP RIGHT E05
plt.subplot(4,2,2)
irun = 0
plt.plot(newt[irun]+2015,-drslt1d[irun]*1000.,'-',color=col1d)
plt.plot(newt[irun]+2015,-drslt1t[irun]*1000.,'--',color=col1d)
plt.plot(newt[irun]+2015,-drslt3d[irun]*1000.,'-',color=col3d)
plt.plot(newt[irun]+2015,-drslt3t[irun]*1000.,'--',color=col3d)
plt.legend(['1D','1T','3D','3T'],frameon=False,\
  loc='upper left')
plt.xticks([])
plt.xlim([2000,2300])
plt.ylabel('uplift rate (mm/y)')
plt.ylim([0,400])
plt.yticks([0,100,200,300,400])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

# BOTTOM RIGHT EA5
plt.subplot(4,2,4)
irun = 1
plt.plot(newt[irun]+2015,-drslt1d[irun]*1000.,'-',color=col1d)
plt.plot(newt[irun]+2015,-drslt1t[irun]*1000.,'--',color=col1d)
plt.plot(newt[irun]+2015,-drslt3d[irun]*1000.,'-',color=col3d)
plt.plot(newt[irun]+2015,-drslt3t[irun]*1000.,'--',color=col3d)
plt.xlabel('year CE')
plt.ylabel('uplift rate (mm/y)')
plt.ylim([0,600])
plt.xlim([2000,2300])
plt.xticks([2000,2100,2200,2300])
plt.yticks([150,300,450,600])
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.tight_layout()



# TOP LEFT NCR
plt.subplot(4,2,5)
irun = 2
plt.plot(ts[irun]+2015,rslt1d[irun],'-',color=col1d)
plt.plot(ts[irun]+2015,rslt1t[irun],'--',color=col1d)
plt.plot(ts[irun]+2015,rslt3d[irun],'-',color=col3d)
plt.plot(ts[irun]+2015,rslt3t[irun],'--',color=col3d)
plt.xlim([2010,2100])
plt.xticks([])

plt.ylim([-4,0])
plt.yticks([-4,-3,-2,-1,0])
plt.ylabel('RSL (m)')

# BOTTOM LEFT F32
plt.subplot(4,2,7)
irun = 3
plt.plot(ts[irun]+2015,rslt1d[irun],'-',color=col1d)
plt.plot(ts[irun]+2015,rslt1t[irun],'--',color=col1d)
plt.plot(ts[irun]+2015,rslt3d[irun],'-',color=col3d)
plt.plot(ts[irun]+2015,rslt3t[irun],'--',color=col3d)
plt.xlim([2010,2100])
plt.xticks([2010,2040,2070,2100])
plt.xlabel('year CE')

plt.ylim([-40,0])
plt.yticks([-30,-20,-10,0])
plt.ylabel('RSL (m)')


# TOP RIGHT E05
plt.subplot(4,2,6)
irun = 0
plt.plot(ts[irun]+2015,rslt1d[irun],'-',color=col1d)
plt.plot(ts[irun]+2015,rslt1t[irun],'--',color=col1d)
plt.plot(ts[irun]+2015,rslt3d[irun],'-',color=col3d)
plt.plot(ts[irun]+2015,rslt3t[irun],'--',color=col3d)
plt.xticks([])
plt.xlim([2000,2300])
plt.ylim([-32,0])
plt.yticks([-24,-16,-8,0])

plt.ylabel('RSL (m)')
plt.legend(['1D','1T','3D','3T'],frameon=False,\
  loc='lower left')
ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()


# BOTTOM RIGHT EA5
plt.subplot(4,2,8)
irun = 1

plt.plot(ts[irun]+2015,rslt1d[irun],'-',color=col1d)
plt.plot(ts[irun]+2015,rslt1t[irun],'--',color=col1d)
plt.plot(ts[irun]+2015,rslt3d[irun],'-',color=col3d)
plt.plot(ts[irun]+2015,rslt3t[irun],'--',color=col3d)

plt.xlim([2000,2300])
plt.xticks([2000,2100,2200,2300])
plt.xlabel('year CE')
plt.ylim([-60,0])
plt.yticks([-45,-30,-15,0])

plt.ylabel('RSL (m)')

ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()


plt.tight_layout()
plt.savefig('RESULTS/PAPER_FIGURES/ant_sites_main.pdf')



plt.figure(3,figsize=(1.5,1.5))
irun = 0
plt.plot(ts[irun]+2015,rslt1d[irun],'-',color=col1d)
plt.plot(ts[irun]+2015,rslt1t[irun],'--',color=col1d)
plt.plot(ts[irun]+2015,rslt3d[irun],'-',color=col3d)
plt.plot(ts[irun]+2015,rslt3t[irun],'--',color=col3d)
plt.xticks([])
plt.xlim([2275,2300])
plt.xticks([2275,2300])

plt.ylim([-30,-15])
plt.yticks([-15,-20,-25])

#plt.ylabel('RSL (m)')

ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.tight_layout()

plt.savefig('RESULTS/PAPER_FIGURES/ant_sites_inset1.pdf')

plt.figure(4,figsize=(1.5,1.5))
irun = 1
plt.plot(ts[irun]+2015,rslt1d[irun],'-',color=col1d)
plt.plot(ts[irun]+2015,rslt1t[irun],'--',color=col1d)
plt.plot(ts[irun]+2015,rslt3d[irun],'-',color=col3d)
plt.plot(ts[irun]+2015,rslt3t[irun],'--',color=col3d)
plt.xticks([])
plt.xlim([2275,2300])
plt.xticks([2275,2300])

plt.ylim([-70,-30])
plt.yticks([-60,-45,-30])

#plt.ylabel('RSL (m)')

ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.tight_layout()


plt.savefig('RESULTS/PAPER_FIGURES/ant_sites_inset2.pdf')



plt.show()
