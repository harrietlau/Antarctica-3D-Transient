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
fmods  = ['EM1D','EM3D','EM1T','EM3T']
fruns  = ['NCR', 'F32', 'EA5', 'EA5' ]
idx0   = ['04',  '04',   '01',   '01']
idx1   = ['41',  '41',   '56',   '56']
fdir   = 'RESULTS/'
fname1 = 'RESULTS/PAPER_FIGURES/rslmap.pdf'
fname2 = 'RESULTS/PAPER_FIGURES/rslmap_cbar.pdf'


irun  = 3
imod  = 3

clat  = np.loadtxt(fdir+'EM1D_E05/clat.dat')
lon   = np.loadtxt(fdir+'EM1D_E05/elon.dat')
lat   = 90.-clat
nlat  = len(clat)
nlon  = len(lon)

ffdir = fdir+fmods[imod]+'_'
fname = fruns[irun] + '/RSL_'+idx1[irun]
rsl   = np.loadtxt(ffdir+fname)

slat    = np.array([29.9509,\
                    38.8935,\
                    51.5072,\
                    6.6137,\
                    35.6764])
                    
slon    = np.array([360-90.0758,\
                    360-77.0145,\
                    0.1276,\
                    3.3553,\
                    139.6500])


proj = ccrs.Mollweide()


cmax = 6.0
cmin =-60.

nlevel = 10
# clean and apply saturation points

rslc = np.copy(rsl)
print('Max/Min values for run',fruns[irun],'and model',fmods[imod])
print(np.max(rsl),np.min(rsl))
[i,j] = np.where(rsl>=cmax)
rslc[i,j] = cmax
[i,j] = np.where(rsl[:,:]<=cmin)
rslc[i,j] = cmin

fig1,axs = plt.subplots(ncols=1,nrows=1,figsize=(5,3),\
                        subplot_kw={'projection': proj})

divnorm = colors.TwoSlopeNorm(vmin=cmin,vcenter=0,vmax=cmax)
level1 = np.linspace(cmin,0,nlevel)
level1 = level1[0:-1]
level2 = np.linspace(0,cmax,nlevel)
levels = np.concatenate((level1,level2))
cf = axs.contourf(lon,lat,rslc,\
                  transform=ccrs.PlateCarree(),cmap='RdBu',\
                  vmin=cmin,vmax=cmax,\
                  levels=levels,norm=divnorm)
axs.contour(lon,lat,rslc,[3.5],colors='lime',lw=2.,\
                    transform=ccrs.PlateCarree())
                    
axs.plot(slon,slat,'m*',\
                      transform=ccrs.PlateCarree())

# Add details to the plot:
axs.coastlines();
axs.gridlines();

plt.tight_layout()

plt.savefig(fname1)
    
plt.figure(figsize=(3,3))
plt.contourf(rslc,cmap='RdBu',\
             vmin=cmin,vmax=cmax,\
             levels=levels,norm=divnorm)
ax = plt.gca()
cbar = plt.colorbar(ticks=[cmin,0,cmax],orientation='vertical')
cbar.ax.set_ylabel(r'RSL$_{\rm 3T}$ (m)')

plt.tight_layout()

plt.savefig(fname2)

plt.show()
