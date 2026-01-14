#! /usr/bin/env python3 

# import differences in the global figures
# near the start (10 years) of the simulation
# and at the end.

import numpy as np
import matplotlib.pyplot as plt
import cartopy as cp
import cartopy.crs as ccrs
import sys
import matplotlib as mpl
import matplotlib.colors as colors


# =============================================== #
# inputs:
fmods = ['EM1D','EM3D','EM1T','EM3T']
fruns = ['NCR', 'F32', 'E05', 'EA5' ]
idx0  = ['04',  '04',   '01',   '01']
idx1  = ['41',  '41',   '56',   '56']
fdir  = 'RESULTS/'
odir  = 'RESULTS/PAPER_FIGURES/'

clat  = np.loadtxt(fdir+'EM1D_E05/clat.dat')
lon   = np.loadtxt(fdir+'EM1D_E05/elon.dat')
lat   = 90.-clat

nmod  = len(fmods)
nrun  = len(fruns)
nlat  = len(clat)
nlon  = len(lon)

rsls0 = np.zeros((nmod,nrun,nlat,nlon))
rsls1 = np.zeros((nmod,nrun,nlat,nlon))

for imod in range(nmod):
  ffdir = fdir+fmods[imod]+'_'

  for irun in range(nrun):
    fname = fruns[irun] + '/RSL_'+idx0[irun]
    data  = np.loadtxt(ffdir+fname)
    rsls0[imod,irun,:,:] = data

    fname = fruns[irun] + '/RSL_'+idx1[irun]
    data  = np.loadtxt(ffdir+fname)
    rsls1[imod,irun,:,:] = data

# FOCUS ONLY ON EA5

# 3T - 3D
# 3D - 1D
# 3T - 1D

irun   = 3

drsls0 = np.zeros((3,nlat,nlon))
drsls1 = np.zeros((3,nlat,nlon))

drsls0[0,:,:] = rsls0[3,irun,:,:] - rsls0[1,irun,:,:]
drsls0[1,:,:] = rsls0[1,irun,:,:] - rsls0[0,irun,:,:]
drsls0[2,:,:] = rsls0[3,irun,:,:] - rsls0[0,irun,:,:]

drsls1[0,:,:] = rsls1[3,irun,:,:] - rsls1[1,irun,:,:]
drsls1[1,:,:] = rsls1[1,irun,:,:] - rsls1[0,irun,:,:]
drsls1[2,:,:] = rsls1[3,irun,:,:] - rsls1[0,irun,:,:]

drsls0c = np.copy(drsls0)
drsls1c = np.copy(drsls1)

# print high low range:
# for 3T-3D:
print('for 3T-3D', np.max(drsls1c[0,:,:]),np.min(drsls1c[0,:,:]))
print('for 3D-3D', np.max(drsls1c[1,:,:]),np.min(drsls1c[1,:,:]))
print('for 3T-3D', np.max(drsls1c[2,:,:]),np.min(drsls1c[2,:,:]))

cmax0  =  0.01
cmin0  = -0.01

cmax1  = 0.5
cmin1  = -0.8

[i,j,k] = np.where(drsls0>=cmax0)
drsls0c[i,j,k] = cmax0
[i,j,k] = np.where(drsls0<=cmin0)
drsls0c[i,j,k] = cmin0

[i,j,k] = np.where(drsls1>=cmax1)
drsls1c[i,j,k] = cmax1
[i,j,k] = np.where(drsls1<=cmin1)
drsls1c[i,j,k] = cmin1

# PLOT

proj = ccrs.Mollweide()

nlevel = 6

fname1   = odir+'globaldrsl_beg_EA5.pdf'
fname2   = odir+'globaldrsl_end_EA5.pdf'
fname3   = odir+'globaldrsl_cbar.pdf'



# START:
fig1,axs = plt.subplots(ncols=1,nrows=3,figsize=(8,6),\
                        subplot_kw={'projection': proj})

# 3T - 3D
divnorm = colors.TwoSlopeNorm(vmin=cmin0,vcenter=0,vmax=cmax0)
level1 = np.linspace(cmin0,0,nlevel)
level1 = level1[0:-1]
level2 = np.linspace(0,cmax0,nlevel)
levels = np.concatenate((level1,level2))
cf     = axs[0].contourf(lon,lat,drsls0c[0,:,:],\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin0,vmax=cmax0,\
                       levels=levels,norm=divnorm)
# Add details to the plot:
axs[0].coastlines();
axs[0].gridlines();

# 3D - 1D
cf     = axs[1].contourf(lon,lat,drsls0c[1,:,:],\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin0,vmax=cmax0,\
                       levels=levels,norm=divnorm)
# Add details to the plot:
axs[1].coastlines();
axs[1].gridlines();

# 3T - 1D
cf     = axs[2].contourf(lon,lat,drsls0c[2,:,:],\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin0,vmax=cmax0,\
                       levels=levels,norm=divnorm)
# Add details to the plot:
axs[2].coastlines();
axs[2].gridlines();
   
plt.tight_layout()
plt.savefig(fname1)


# END

fig2,axs = plt.subplots(ncols=1,nrows=3,figsize=(8,6),\
                        subplot_kw={'projection': proj})

# 3T - 3D
divnorm = colors.TwoSlopeNorm(vmin=cmin0,vcenter=0,vmax=cmax0)
level1 = np.linspace(cmin1,0,nlevel)
level1 = level1[0:-1]
level2 = np.linspace(0,cmax1,nlevel)
levels = np.concatenate((level1,level2))
cf     = axs[0].contourf(lon,lat,drsls1c[0,:,:],\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin1,vmax=cmax1,\
                       levels=levels,norm=divnorm)
# Add details to the plot:
axs[0].coastlines();
axs[0].gridlines();

# 3D - 1D
cf     = axs[1].contourf(lon,lat,drsls1c[1,:,:],\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin1,vmax=cmax1,\
                       levels=levels,norm=divnorm)
# Add details to the plot:
axs[1].coastlines();
axs[1].gridlines();

# 3T - 1D
cf     = axs[2].contourf(lon,lat,drsls1c[2,:,:],\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin1,vmax=cmax1,\
                       levels=levels,norm=divnorm)
# Add details to the plot:
axs[2].coastlines();
axs[2].gridlines();
   
plt.tight_layout()
plt.savefig(fname2)

plt.figure(3,figsize=(8,4))
plt.subplot(1,2,1)
divnorm = colors.TwoSlopeNorm(vmin=cmin0,vcenter=0,vmax=cmax0)
level1 = np.linspace(cmin0,0,nlevel)
level1 = level1[0:-1]
level2 = np.linspace(0,cmax0,nlevel)
levels = np.concatenate((level1,level2))
plt.contourf(drsls0c[0,:,:],cmap='RdBu',vmin=cmin0,vmax=cmax0,\
                       levels=levels,norm=divnorm)
ax = plt.gca()
cbar = plt.colorbar(ticks=[cmin0,0,cmax0],orientation='horizontal')
cbar.ax.set_xlabel(r'$\Delta$ RSL (m)')

plt.subplot(1,2,2)
divnorm = colors.TwoSlopeNorm(vmin=cmin1,vcenter=0,vmax=cmax1)
level1 = np.linspace(cmin1,0,nlevel)
level1 = level1[0:-1]
level2 = np.linspace(0,cmax1,nlevel)
levels = np.concatenate((level1,level2))
plt.contourf(drsls1c[0,:,:],cmap='RdBu',vmin=cmin1,vmax=cmax1,\
                       levels=levels,norm=divnorm)
ax = plt.gca()
cbar = plt.colorbar(ticks=[cmin1,0,cmax1],orientation='horizontal')
cbar.ax.set_xlabel(r'$\Delta$ RSL (m)')

plt.tight_layout()

plt.savefig(fname3)

plt.show()






'''


    fname   = odir+'globaldrsl_end_'+str(irun)+'.pdf'

    fig2,axs = plt.subplots(ncols=2,nrows=2,figsize=(8,6),\
                           subplot_kw={'projection': proj})
    # 3T-3D
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,0],\
      vcenter=0,vmax=cmax1s[irun,0])
    level1 = np.linspace(cmin1s[irun,0],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,0],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[0,0].contourf(lon,lat,drsls1c[irun,0,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin1s[irun,0],vmax=cmax1s[irun,0],\
                         levels=levels,norm=divnorm)
    # Add details to the plot:
    axs[0,0].coastlines();
    axs[0,0].gridlines();

    # 3D-1D
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,1],\
      vcenter=0,vmax=cmax1s[irun,1])
    level1 = np.linspace(cmin1s[irun,1],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,1],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[0,1].contourf(lon,lat,drsls1c[irun,1,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin1s[irun,1],vmax=cmax1s[irun,1],\
                         levels=levels,norm=divnorm)
    # Add details to the plot:
    axs[0,1].coastlines();
    axs[0,1].gridlines();

    # 3T-1T
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,2],\
      vcenter=0,vmax=cmax1s[irun,2])
    level1 = np.linspace(cmin1s[irun,2],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,2],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[1,0].contourf(lon,lat,drsls1c[irun,2,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin1s[irun,2],vmax=cmax1s[irun,2],\
                         levels=levels,norm=divnorm)
    # Add details to the plot:
    axs[1,0].coastlines();
    axs[1,0].gridlines();

    # 3T-1D
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,3],\
      vcenter=0,vmax=cmax1s[irun,3])
    level1 = np.linspace(cmin1s[irun,3],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,3],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[1,1].contourf(lon,lat,drsls1c[irun,3,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin1s[irun,3],vmax=cmax1s[irun,3],\
                         levels=levels,norm=divnorm)
    # Add details to the plot:
    axs[1,1].coastlines();
    axs[1,1].gridlines();

    
    plt.tight_layout()

    plt.savefig(fname)

    plt.close()
    
    plt.figure()
    plt.subplot(2,2,1)
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,0],\
      vcenter=0,vmax=cmax0s[irun,0])
    level1 = np.linspace(cmin0s[irun,0],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,0],nlevel)
    levels = np.concatenate((level1,level2))
    plt.contourf(drsls0c[irun,0,:,:],cmap='RdBu',\
                         vmin=cmin0s[irun,0],vmax=cmax0s[irun,0],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,0],0,cmax0s[irun,0]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-3D}$ (m)')
    
    plt.subplot(2,2,2)
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,1],\
      vcenter=0,vmax=cmax0s[irun,1])
    level1 = np.linspace(cmin0s[irun,1],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,1],nlevel)
    levels = np.concatenate((level1,level2))
    plt.contourf(drsls0c[irun,1,:,:],cmap='RdBu',\
                         vmin=cmin0s[irun,1],vmax=cmax0s[irun,1],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,1],0,cmax0s[irun,1]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3D-1D}$ (m)')

    plt.subplot(2,2,3)
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,2],\
      vcenter=0,vmax=cmax0s[irun,2])
    level1 = np.linspace(cmin0s[irun,2],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,2],nlevel)
    levels = np.concatenate((level1,level2))

    plt.contourf(drsls0c[irun,2,:,:],cmap='RdBu',\
                         vmin=cmin0s[irun,2],vmax=cmax0s[irun,2],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,2],0,cmax0s[irun,2]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-1T}$ (m)')

    plt.subplot(2,2,4)
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,3],\
      vcenter=0,vmax=cmax0s[irun,3])
    level1 = np.linspace(cmin0s[irun,3],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,3],nlevel)
    levels = np.concatenate((level1,level2))
    plt.contourf(drsls0c[irun,3,:,:],cmap='RdBu',\
                         vmin=cmin0s[irun,3],vmax=cmax0s[irun,3],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,3],0,cmax0s[irun,3]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-1D}$ (m)')
    plt.tight_layout()
    
    fname   = odir+'globaldrsl_st_'+str(irun)+'_cbar.pdf'

    plt.savefig(fname)
    
    plt.figure()
    plt.subplot(2,2,1)
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,0],\
      vcenter=0,vmax=cmax1s[irun,0])
    level1 = np.linspace(cmin1s[irun,0],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,0],nlevel)
    levels = np.concatenate((level1,level2))
    plt.contourf(drsls1c[irun,0,:,:],cmap='RdBu',\
                         vmin=cmin1s[irun,0],vmax=cmax1s[irun,0],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin1s[irun,0],0,cmax1s[irun,0]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-3D}$ (m)')

    plt.subplot(2,2,2)
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,1],\
      vcenter=0,vmax=cmax1s[irun,1])
    level1 = np.linspace(cmin1s[irun,1],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,1],nlevel)
    levels = np.concatenate((level1,level2))

    plt.contourf(drsls1c[irun,1,:,:],cmap='RdBu',\
                         vmin=cmin1s[irun,1],vmax=cmax1s[irun,1],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin1s[irun,1],0,cmax1s[irun,1]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3D-1D}$ (m)')
    
    plt.subplot(2,2,3)
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,2],\
      vcenter=0,vmax=cmax1s[irun,2])
    level1 = np.linspace(cmin1s[irun,2],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,2],nlevel)
    levels = np.concatenate((level1,level2))

    plt.contourf(drsls1c[irun,2,:,:],cmap='RdBu',\
                         vmin=cmin1s[irun,2],vmax=cmax1s[irun,2],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin1s[irun,2],0,cmax1s[irun,2]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-1T}$ (m)')
    
    plt.subplot(2,2,4)
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,3],\
      vcenter=0,vmax=cmax1s[irun,3])
    level1 = np.linspace(cmin1s[irun,3],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,3],nlevel)
    levels = np.concatenate((level1,level2))

    plt.contourf(drsls1c[irun,3,:,:],cmap='RdBu',\
                         vmin=cmin1s[irun,3],vmax=cmax1s[irun,3],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin1s[irun,3],0,cmax1s[irun,3]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-1D}$ (m)')

    plt.tight_layout()

    fname   = odir+'globaldrsl_end_'+str(irun)+'_cbar.pdf'
    plt.savefig(fname)

    plt.close()
'''
