#! /usr/bin/env python3 

# explore regional differences between
# RA5 (most agressive melt).

import numpy as np
import matplotlib.pyplot as plt
import cartopy as cp
import cartopy.crs as ccrs
import sys
import matplotlib as mpl
import matplotlib.colors as colors

# =============================================== #
# inputs:
fdir  = 'RESULTS/EM1D_EA5/'
f1dm  = 'RESULTS/EM1D_EA5/'
f1dt  = 'RESULTS/EM1T_EA5/'
f3dm  = 'RESULTS/EM3D_EA5/'
f3dt  = 'RESULTS/EM3T_EA5/'
ftime = '../Data/tt_mE5'
istr  = '57'
ofile1 = 'RESULTS/PAPER_FIGURES/ant_diff.pdf'
ofile2 = 'RESULTS/PAPER_FIGURES/ant_diff_cbar.pdf'

slat  = -76.5
slon  = -105.

# load stuff
t     = np.loadtxt(ftime,skiprows=1)
nt    = len(t)
lat   = np.loadtxt(fdir+'lat.dat')
lon   = np.loadtxt(fdir+'lon.dat')
nlat  = len(lat)
nlon  = len(lon)

rsl_1dm = np.zeros((nlat,nlon))
rsl_3dm = np.zeros((nlat,nlon))
rsl_1dt = np.zeros((nlat,nlon))
rsl_3dt = np.zeros((nlat,nlon))

rsl_1dm = np.loadtxt(f1dm+'RSLreg_'+istr)
rsl_1dt = np.loadtxt(f1dt+'RSLreg_'+istr)
rsl_3dm = np.loadtxt(f3dm+'RSLreg_'+istr)
rsl_3dt = np.loadtxt(f3dt+'RSLreg_'+istr)

proj = ccrs.SouthPolarStereo()

d3t_3d = rsl_3dt - rsl_3dm
d3d_1d = rsl_3dm - rsl_1dm
d1t_1d = rsl_1dt - rsl_1dm
d3t_1d = rsl_3dt - rsl_1dm

print('3T-3D max/min:',np.nanmax(d3t_3d),np.nanmin(d3t_3d))
print('3D-1D max/min:',np.nanmax(d3d_1d),np.nanmin(d3d_1d))
print('1T-1D max/min:',np.nanmax(d1t_1d),np.nanmin(d1t_1d))
print('3T-1D max/min:',np.nanmax(d3t_1d),np.nanmin(d3t_1d))

nlevel = 6

cmax = 3
cmin = -9

[i,j] = np.where(d3t_3d>=cmax)
d3t_3d[i,j] = cmax
[i,j] = np.where(d3t_3d<=cmin)
d3t_3d[i,j] = cmin

[i,j] = np.where(d3d_1d>=cmax)
d3d_1d[i,j] = cmax
[i,j] = np.where(d3d_1d<=cmin)
d3d_1d[i,j] = cmin

[i,j] = np.where(d1t_1d>=cmax)
d1t_1d[i,j] = cmax
[i,j] = np.where(d1t_1d<=cmin)
d1t_1d[i,j] = cmin

[i,j] = np.where(d3t_1d>=cmax)
d3t_1d[i,j] = cmax
[i,j] = np.where(d3t_1d<=cmin)
d3t_1d[i,j] = cmin

divnorm = colors.TwoSlopeNorm(vmin=cmin,vcenter=0,vmax=cmax)
level1  = np.linspace(cmin,0,nlevel)
level1  = level1[0:-1]
level2  = np.linspace(0,cmax,nlevel)
levels  = np.concatenate((level1,level2))


fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(6,6),\
                       subplot_kw={'projection': proj})
# 3T-3D
axs[0,0].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
cf = axs[0,0].contourf(lon,lat,d3t_3d,\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin,vmax=cmax,\
                       levels=levels,norm=divnorm)

#cbar = plt.colorbar(cf,ticks=[-10,-5,0,5])
#cbar.ax.set_ylabel(r'log$_{10}[\chi^2]$')
# Add details to the plot:
axs[0,0].coastlines();
axs[0,0].gridlines();

# 3D-1D
axs[0,1].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
cf = axs[0,1].contourf(lon,lat,d3d_1d,\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin,vmax=cmax,\
                       levels=levels,norm=divnorm)
#cbar = plt.colorbar(cf,ticks=[-10,-5,0,5])

# Add details to the plot:
axs[0,1].coastlines();
axs[0,1].gridlines();

# 1T-1D
axs[1,0].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
cf = axs[1,0].contourf(lon,lat,d1t_1d,\
                       transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin,vmax=cmax,\
                       levels=levels,norm=divnorm)
#cbar = plt.colorbar(cf,ticks=[-10,-5,0,5])

# Add details to the plot:
axs[1,0].coastlines();
axs[1,0].gridlines();

# 3T-1D
axs[1,1].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
cf = axs[1,1].contourf(lon,lat,d3t_1d,\
                      transform=ccrs.PlateCarree(),cmap='RdBu',\
                       vmin=cmin,vmax=cmax,\
                       levels=levels,norm=divnorm)
#cbar = plt.colorbar(cf,ticks=[-10,-5,0,5])

# Add details to the plot:
axs[1,1].coastlines();
axs[1,1].gridlines();

#axs[1,1].plot([slon,slon],[slat,slat],'^',ms=15,mfc='g',mec='g',\
#                      transform=ccrs.PlateCarree())

plt.tight_layout()

plt.savefig(ofile1)

plt.figure(2)
# dummy figure for colorbar

plt.contourf(d3t_1d,cmap='RdBu',\
                       vmin=cmin,vmax=cmax,\
                       levels=levels,norm=divnorm)
cbar = plt.colorbar(ticks=[-9,-6,-3,0,1,2,3])
cbar.set_label(r'$\Delta$ RSL (m)')

plt.savefig(ofile2)


plt.show()
