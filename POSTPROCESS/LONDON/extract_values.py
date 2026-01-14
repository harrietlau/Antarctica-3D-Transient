#! /usr/bin/env python3

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import mfunc as m

''' 
Try to understand why London is so model
dependent?

Relative to DC.

'''

# 0 = london, 1 = DC
slat = np.array([51.5072, 38.9072])
clat = 90.-slat
slon = np.array([0.0    ,-77.0369])
slon = 360. + slon 
slon[0] = 0.0 # interoplator doesn't like 360
pts  = np.array([clat,slon])

# viscosity model:
f0 = '../model/global_dvis_r'
clats = np.loadtxt('../model/clats.dat')
elons = np.loadtxt('../model/elons.dat')
rad   = np.loadtxt('../model/radius.dat')
vsc1d = 0.5e21
# upper mantle only
idx   = np.where((rad>=5271000)&(rad<=6368000))[0]
rad   = rad[idx]
nr    = len(rad)

# transient model:
[wgts,taus] = np.loadtxt('gmx_seakon10.dat',\
                         skiprows=1,unpack=True)

data1d  = np.loadtxt('stw105.txt',skiprows=3,\
                     usecols=(0,1,3))
mu1dt   = data1d[:,2]*data1d[:,2]*data1d[:,1]
f       = si.interp1d(data1d[:,0],mu1dt)
mu01d   = f(rad)

vsc3d   = np.zeros((nr,2))

for i in range(nr):
    istr = str(idx[i])+'.dat'
    data = np.loadtxt(f0+istr)
    f    = si.RegularGridInterpolator((clats,elons),data)
    vals = f(pts)
    vsc3d[i,:] = vals

# now we get a profile of the variation
# of M3D, M1D, M1T, M3T with different
# time scales
y2s  = 365.*24.*60.*60.
pers = np.array([100,500,1000]) * y2s
freq = 1./pers
nf   = len(freq)

ldon = np.zeros((2,nr,nf))
wdc  = np.zeros((2,nr,nf))
man  = np.zeros((2,nr,nf))

for i in range(nr):
    # London:
    # Maxwell 3D
    v3d  = vsc1d * 10.**vsc3d[i,0]
    mx3d = m.M_mx_LT(freq,mu01d[i],v3d)
    # Transient 3D
    dwgts = wgts*mu01d[i]
    dtaus = taus*(v3d/mu01d[i])
    jk,tr3d = m.makegxm(freq,dwgts,dtaus)
    ldon[0,i,:] = mx3d*1e-9
    ldon[1,i,:] = tr3d*1e-9
    
    # DC:
    # Maxwell 3D
    v3d  = vsc1d * 10.**vsc3d[i,1]
    mx3d = m.M_mx_LT(freq,mu01d[i],v3d)
    # Transient 3D
    dwgts = wgts*mu01d[i]
    dtaus = taus*(v3d/mu01d[i])
    jk,tr3d = m.makegxm(freq,dwgts,dtaus)
    wdc[0,i,:]  = mx3d*1e-9
    wdc[1,i,:]  = tr3d*1e-9

    # 1D
    # Maxwell 1D
    v3d  = vsc1d 
    mx3d = m.M_mx_LT(freq,mu01d[i],v3d)
    # Transient 1D
    dwgts = wgts*mu01d[i]
    dtaus = taus*(v3d/mu01d[i])
    jk,tr3d = m.makegxm(freq,dwgts,dtaus)
    man[0,i,:]  = mx3d*1e-9
    man[1,i,:]  = tr3d*1e-9


r = rad/1000.


plt.figure(1,figsize=(6.5,5))
plt.subplot(1,3,1)
ifreq = 0
plt.fill_betweenx(r,ldon[0,:,ifreq],ldon[1,:,ifreq],alpha=0.3,\
                  ec='none',fc='tab:blue')
plt.fill_betweenx(r,ldon[0,:,ifreq],ldon[1,:,ifreq],\
                  ec='tab:blue',fc='none',lw=1)
plt.fill_betweenx(r,wdc[0,:,ifreq],wdc[1,:,ifreq],alpha=0.3,\
                  ec='none',fc='orange',lw=2)
plt.fill_betweenx(r,wdc[0,:,ifreq],wdc[1,:,ifreq],\
                  ec='orange',fc='none',lw=1)

plt.ylim([5300,6300])
plt.yticks([5300,6300],['660','LAB'])
plt.ylabel('depth')
plt.xlabel(r'$\mu(\tau=100$ y) GPa')
plt.xlim([0,100])
plt.xticks([0,50,100])

plt.subplot(1,3,2)
ifreq = 1
plt.fill_betweenx(r,ldon[0,:,ifreq],ldon[1,:,ifreq],alpha=0.3,\
                  ec='none',fc='tab:blue')
plt.fill_betweenx(r,ldon[0,:,ifreq],ldon[1,:,ifreq],\
                  ec='tab:blue',fc='none',lw=1)
plt.fill_betweenx(r,wdc[0,:,ifreq],wdc[1,:,ifreq],alpha=0.3,\
                  ec='none',fc='orange')
plt.fill_betweenx(r,wdc[0,:,ifreq],wdc[1,:,ifreq],\
                  ec='orange',fc='none',lw=1)

plt.ylim([5300,6300])
plt.yticks([])
plt.xlim([0,80])
plt.xticks([0,40,80])
plt.xlabel(r'$\mu(\tau=500$ y) GPa')

plt.subplot(1,3,3)
ifreq = 2
# dummy:
plt.fill_betweenx(r*1000,ldon[0,:,ifreq],ldon[1,:,ifreq],alpha=0.3,\
                  ec='none',fc='tab:blue')
plt.fill_betweenx(r*1000,wdc[0,:,ifreq],wdc[1,:,ifreq],alpha=0.3,\
                  ec='none',fc='orange')
plt.legend(['London','DC'],loc='lower right',frameon=False)

plt.fill_betweenx(r,ldon[0,:,ifreq],ldon[1,:,ifreq],alpha=0.3,\
                  ec='none',fc='tab:blue')
plt.fill_betweenx(r,ldon[0,:,ifreq],ldon[1,:,ifreq],\
                  ec='tab:blue',fc='none',lw=1)
plt.fill_betweenx(r,wdc[0,:,ifreq],wdc[1,:,ifreq],alpha=0.3,\
                  ec='none',fc='orange')
plt.fill_betweenx(r,wdc[0,:,ifreq],wdc[1,:,ifreq],\
                  ec='orange',fc='none',lw=1)


plt.ylabel('depth')
plt.ylim([5300,6300])
plt.yticks([5300,6300],['660','LAB'])
plt.xlim([0,80])
plt.xticks([0,40,80])
plt.xlabel(r'$\mu(\tau=1000$ y) GPa')

ax = plt.gca()
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.tight_layout()

plt.savefig('london-dc.pdf')

plt.show()
    
    



  

        
    

    
    
