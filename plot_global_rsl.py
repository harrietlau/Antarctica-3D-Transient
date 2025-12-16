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


proj = ccrs.Mollweide()


cmax0s = np.zeros((nrun,nmod))
cmin0s = np.zeros((nrun,nmod))

cmax0s[0,:] = np.ones(nmod) * 0.03 # done
cmax0s[1,:] = np.ones(nmod) * 0.03
cmax0s[2,:] = np.ones(nmod) * 0.05
cmax0s[3,:] = np.ones(nmod) * 0.05

cmin0s[0,:] = np.ones(nmod) * -0.1 # done
cmin0s[1,:] = np.ones(nmod) * -0.1
cmin0s[2,:] = np.ones(nmod) * -0.1
cmin0s[3,:] = np.ones(nmod) * -0.1

cmax1s = np.zeros((nrun,nmod))
cmin1s = np.zeros((nrun,nmod))

cmax1s[0,:] = np.array([0.5,0.5,6,6])
cmax1s[1,:] = np.array([0.5,0.5,6,6])
cmax1s[2,:] = np.array([0.5,0.5,6,6])
cmax1s[3,:] = np.array([0.5,0.5,6,6])

cmin1s[0,:] = np.array([-0.5,-0.5,-10,-10])
cmin1s[1,:] = np.array([-0.5,-0.5,-10,-10])
cmin1s[2,:] = np.array([-0.5,-0.5,-10,-10])
cmin1s[3,:] = np.array([-0.5,-0.5,-10,-10])

nlevel = 6
# clean and apply saturation points

rsls0c = np.copy(rsls0)
rsls1c = np.copy(rsls1)


for irun in range(nrun):
    for imod in range(nmod):
      print('Max/Min values for run',fruns[irun],'and model',fmods[imod])
      print(np.max(rsls0[irun,imod,:,:]),np.min(rsls0[irun,imod,:,:]))
      [i,j] = np.where(rsls0[irun,imod,:,:]>=cmax0s[irun,imod])
      rsls0c[irun,imod,i,j] = cmax0s[irun,imod]
      [i,j] = np.where(rsls0[irun,imod,:,:]<=cmin0s[irun,imod])
      rsls0c[irun,imod,i,j] = cmin0s[irun,imod]

      [i,j] = np.where(rsls1[irun,imod,:,:]>=cmax1s[irun,imod])
      rsls1c[irun,imod,i,j] = cmax1s[irun,imod]
      [i,j] = np.where(rsls1[irun,imod,:,:]<=cmin1s[irun,imod])
      rsls1c[irun,imod,i,j] = cmin1s[irun,imod]

# now plot
for irun in range(nrun):

    fname   = odir+'globalrsl_st_'+str(irun)+'.pdf'
    
    fig1,axs = plt.subplots(ncols=2,nrows=2,figsize=(8,6),\
                           subplot_kw={'projection': proj})
    # 3T-3D
    divnorm = colors.TwoSlopeNorm(vmin=cmin0s[irun,0],\
      vcenter=0,vmax=cmax0s[irun,0])
    level1 = np.linspace(cmin0s[irun,0],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax0s[irun,0],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[0,0].contourf(lon,lat,rsls0c[irun,0,:,:],\
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
    cf = axs[0,1].contourf(lon,lat,rsls0c[irun,1,:,:],\
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
    cf = axs[1,0].contourf(lon,lat,rsls0c[irun,2,:,:],\
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
    cf = axs[1,1].contourf(lon,lat,rsls0c[irun,3,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu',\
                         vmin=cmin0s[irun,3],vmax=cmax0s[irun,3],\
                         levels = levels,norm=divnorm)
    # Add details to the plot:
    axs[1,1].coastlines();
    axs[1,1].gridlines();

    
    plt.tight_layout()

    plt.savefig(fname)

    plt.close()

    fname   = odir+'globalrsl_end_'+str(irun)+'.pdf'

    fig2,axs = plt.subplots(ncols=2,nrows=2,figsize=(8,6),\
                           subplot_kw={'projection': proj})
    # 3T-3D
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,0],\
      vcenter=0,vmax=cmax1s[irun,0])
    level1 = np.linspace(cmin1s[irun,0],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,0],nlevel)
    levels = np.concatenate((level1,level2))
    cf = axs[0,0].contourf(lon,lat,rsls1c[irun,0,:,:],\
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
    cf = axs[0,1].contourf(lon,lat,rsls1c[irun,1,:,:],\
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
    cf = axs[1,0].contourf(lon,lat,rsls1c[irun,2,:,:],\
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
    cf = axs[1,1].contourf(lon,lat,rsls1c[irun,3,:,:],\
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
    plt.contourf(rsls0c[irun,0,:,:],cmap='RdBu',\
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
    plt.contourf(rsls0c[irun,1,:,:],cmap='RdBu',\
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

    plt.contourf(rsls0c[irun,2,:,:],cmap='RdBu',\
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
    plt.contourf(rsls0c[irun,3,:,:],cmap='RdBu',\
                         vmin=cmin0s[irun,3],vmax=cmax0s[irun,3],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,3],0,cmax0s[irun,3]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-1D}$ (m)')
    plt.tight_layout()
    
    fname   = odir+'globalrsl_st_'+str(irun)+'_cbar.pdf'

    plt.savefig(fname)
    
    plt.figure()
    plt.subplot(2,2,1)
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,0],\
      vcenter=0,vmax=cmax1s[irun,0])
    level1 = np.linspace(cmin1s[irun,0],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,0],nlevel)
    levels = np.concatenate((level1,level2))
    plt.contourf(rsls1c[irun,0,:,:],cmap='RdBu',\
                         vmin=cmin1s[irun,0],vmax=cmax1s[irun,0],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,0],0,cmax0s[irun,0]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-3D}$ (m)')

    plt.subplot(2,2,2)
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,1],\
      vcenter=0,vmax=cmax1s[irun,1])
    level1 = np.linspace(cmin1s[irun,1],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,1],nlevel)
    levels = np.concatenate((level1,level2))

    plt.contourf(rsls1c[irun,1,:,:],cmap='RdBu',\
                         vmin=cmin1s[irun,1],vmax=cmax1s[irun,1],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,1],0,cmax0s[irun,1]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3D-1D}$ (m)')
    
    plt.subplot(2,2,3)
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,2],\
      vcenter=0,vmax=cmax1s[irun,2])
    level1 = np.linspace(cmin1s[irun,2],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,2],nlevel)
    levels = np.concatenate((level1,level2))

    plt.contourf(rsls1c[irun,2,:,:],cmap='RdBu',\
                         vmin=cmin1s[irun,2],vmax=cmax1s[irun,2],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,2],0,cmax0s[irun,2]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-1T}$ (m)')
    
    plt.subplot(2,2,4)
    divnorm = colors.TwoSlopeNorm(vmin=cmin1s[irun,3],\
      vcenter=0,vmax=cmax1s[irun,3])
    level1 = np.linspace(cmin1s[irun,3],0,nlevel)
    level1 = level1[0:-1]
    level2 = np.linspace(0,cmax1s[irun,3],nlevel)
    levels = np.concatenate((level1,level2))

    plt.contourf(rsls1c[irun,3,:,:],cmap='RdBu',\
                         vmin=cmin1s[irun,3],vmax=cmax1s[irun,3],\
                         levels=levels,norm=divnorm)
    ax = plt.gca()
    cbar = plt.colorbar(ticks=[cmin0s[irun,3],0,cmax0s[irun,3]])
    cbar.ax.set_ylabel(r'$\Delta$ RSL$_{\rm 3T-1D}$ (m)')

    plt.tight_layout()

    fname   = odir+'globalrsl_end_'+str(irun)+'_cbar.pdf'
    plt.savefig(fname)

    plt.close()
