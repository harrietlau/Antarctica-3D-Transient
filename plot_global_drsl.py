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
fruns = ['NCR', 'F32', 'EA5', 'EA5' ]
idx0  = ['04',  '04',   '01',   '01']
idx1  = ['41',  '41',   '56',   '56']
fdir  = 'RESULTS/'
odir  = 'RESULTS/global_plots/'

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

# differences:

# plots: 3T-3D [0]   3D-1D [1]
#        3T-1T [2]   3T-1D [3]


drsls0 = np.zeros((nrun,4,nlat,nlon))
drsls1 = np.zeros((nrun,4,nlat,nlon))

for irun in range(nrun):
  drsls0[irun,0,:,:] = rsls0[3,irun,:,:] - \
    rsls0[1,irun,:,:]
  drsls0[irun,1,:,:] = rsls0[1,irun,:,:] - \
    rsls0[0,irun,:,:]
  drsls0[irun,2,:,:] = rsls0[3,irun,:,:] - \
    rsls0[2,irun,:,:]
  drsls0[irun,3,:,:] = rsls0[3,irun,:,:] - \
    rsls0[0,irun,:,:]

  drsls1[irun,0,:,:] = rsls1[3,irun,:,:] - \
    rsls1[1,irun,:,:]
  drsls1[irun,1,:,:] = rsls1[1,irun,:,:] - \
    rsls1[0,irun,:,:]
  drsls1[irun,2,:,:] = rsls1[3,irun,:,:] - \
    rsls1[2,irun,:,:]
  drsls1[irun,3,:,:] = rsls1[3,irun,:,:] - \
    rsls1[0,irun,:,:]


drsls0c = np.copy(drsls0)
drsls1c = np.copy(drsls1)

cmax0s = np.ones((nrun,nmod)) *  0.01
cmin0s = np.ones((nrun,nmod)) * -0.01

cmax1s = np.ones((nrun,nmod))
cmin1s = np.ones((nrun,nmod))

cmax1s[0,:] = np.array([0.1,0.1,0.1,0.1])
cmax1s[1,:] = np.array([0.1,0.1,0.1,0.1])
cmax1s[2,:] = np.array([0.5,0.5,0.5,0.5])
cmax1s[3,:] = np.array([0.5,0.5,0.5,0.5])

cmin1s[0,:] = np.array([-0.1,-0.1,-0.1,-0.1])
cmin1s[1,:] = np.array([-0.1,-0.1,-0.1,-0.1])
cmin1s[2,:] = np.array([-0.8,-0.8,-0.8,-0.8])
cmin1s[3,:] = np.array([-0.8,-0.8,-0.8,-0.8])


for irun in range(nrun):
    for imod in range(nmod):
      print('START Max/Min values for run',fruns[irun],'and model',fmods[imod])
      print(np.max(drsls0[irun,imod,:,:]),np.min(rsls0[irun,imod,:,:]))
      [i,j] = np.where(drsls0[irun,imod,:,:]>=cmax0s[irun,imod])
      drsls0c[irun,imod,i,j] = cmax0s[irun,imod]
      [i,j] = np.where(drsls0[irun,imod,:,:]<=cmin0s[irun,imod])
      drsls0c[irun,imod,i,j] = cmin0s[irun,imod]

      print('END Max/Min values for run',fruns[irun],'and model',fmods[imod])
      print(np.max(rsls1[irun,imod,:,:]),np.min(rsls1[irun,imod,:,:]))

      [i,j] = np.where(drsls1[irun,imod,:,:]>=cmax1s[irun,imod])
      drsls1c[irun,imod,i,j] = cmax1s[irun,imod]
      [i,j] = np.where(drsls1[irun,imod,:,:]<=cmin1s[irun,imod])
      drsls1c[irun,imod,i,j] = cmin1s[irun,imod]


proj = ccrs.Mollweide()

nlevel = 6

# now plot
for irun in range(nrun):

    fname   = odir+'globaldrsl_st_'+str(irun)+'.pdf'
    
    fig1,axs = plt.subplots(ncols=2,nrows=2,figsize=(8,6),\
                           subplot_kw={'projection': proj})
    # 3T-3D
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,0],\
      vcenter=0,vmax=cmax0s[irun,0])
    level1 = np.linspace(cmin0s[irun,0],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,0],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[0,0].contourf(lon,lat,drsls0c[irun,0,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin0s[irun,0],vmax=cmax0s[irun,0],\
                         levels=levels,norm=divnorm)
    # Add details to the plot:
    axs[0,0].coastlines();
    axs[0,0].gridlines();

    # 3D-1D
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,1],\
      vcenter=0,vmax=cmax0s[irun,1])
    level1 = np.linspace(cmin0s[irun,1],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,1],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[0,1].contourf(lon,lat,drsls0c[irun,1,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin0s[irun,1],vmax=cmax0s[irun,1],\
                         levels=levels,norm=divnorm)
    # Add details to the plot:
    axs[0,1].coastlines();
    axs[0,1].gridlines();

    # 3T-1T
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,2],\
      vcenter=0,vmax=cmax0s[irun,2])
    level1 = np.linspace(cmin0s[irun,2],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,2],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[1,0].contourf(lon,lat,drsls0c[irun,2,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin0s[irun,2],vmax=cmax0s[irun,2],\
                         levels = levels,norm=divnorm)
    # Add details to the plot:
    axs[1,0].coastlines();
    axs[1,0].gridlines();

    # 3T-1D
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,3],\
      vcenter=0,vmax=cmax0s[irun,3])
    level1 = np.linspace(cmin0s[irun,3],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,3],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[1,1].contourf(lon,lat,drsls0c[irun,3,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin0s[irun,3],vmax=cmax0s[irun,3],\
                         levels = levels,norm=divnorm)
    # Add details to the plot:
    axs[1,1].coastlines();
    axs[1,1].gridlines();

    
    plt.tight_layout()

    plt.savefig(fname)

    plt.close()

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
