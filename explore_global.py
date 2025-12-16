#! /usr/bin/env python3 

# explore regional differences between
# RA5 (most agressive melt).

import numpy as np
import matplotlib.pyplot as plt
import cartopy as cp
import cartopy.crs as ccrs
import sys


# =============================================== #
# inputs:

fdir  = sys.argv[1]
f1dm  = sys.argv[2]
f1dt  = sys.argv[3]
f3dm  = sys.argv[4]
f3dt  = sys.argv[5]
ftime = sys.argv[6]
nzero = int(sys.argv[7])
idx   = int(sys.argv[8])
odir  = sys.argv[9]

# for modern runs 300 year runs:
'''
fdir  = 'RESULTS/EM1D_EA5/'
f1dm  = 'RESULTS/EM1D_EA5/'
f1dt  = 'RESULTS/EM1T_EA5/'
f3dm  = 'RESULTS/EM3D_EA5/'
f3dt  = 'RESULTS/EM3T_EA5/'
ftime = '../Data/tt_mE5'
nzero = 2
idx   = 1
odir  = 'RESULTS/global_plots/EA5/'
'''

'''
# for modern runs NCR year runs:
fdir  = 'RESULTS/EM1D_F32/'
f1dm  = 'RESULTS/EM1D_F32/'
f1dt  = 'RESULTS/EM1T_F32/'
f3dm  = 'RESULTS/EM3D_F32/'
f3dt  = 'RESULTS/EM3T_F32/'
ftime = '../Data/tt_mFN'
nzero = 2
idx   = 1
odir  = 'RESULTS/global_plots/F32/'
'''

'''
# for LIG-present runs
fdir  = 'RESULTS/EM_1D/'
f1dm  = 'RESULTS/EM_1D/'
f1dt  = 'RESULTS/EM_1T/'
f3dm  = 'RESULTS/EM_3D/'
f3dt  = 'RESULTS/EM_3T/'
ftime = '../Data/tt_S6G'
nzero = 3
idx   = 0
odir  = 'RESULTS/global_plots/LIG/'
'''

# load stuff
t     = np.loadtxt(ftime,skiprows=1)
nt    = len(t)
clat  = np.loadtxt(fdir+'clat.dat')
lon   = np.loadtxt(fdir+'elon.dat')
lat   = 90-clat
nlat  = len(lat)
nlon  = len(lon)

rsl_1dm = np.zeros((nt,nlat,nlon))
rsl_3dm = np.zeros((nt,nlat,nlon))
rsl_1dt = np.zeros((nt,nlat,nlon))
rsl_3dt = np.zeros((nt,nlat,nlon))

for i in range(nt):
    print('loading',i+1,'of',nt)
    istr = str(i+idx).zfill(nzero)
    data = np.loadtxt(f1dm+'RSL_'+istr)
    rsl_1dm[i,:,:] = data
    data = np.loadtxt(f1dt+'RSL_'+istr)
    rsl_1dt[i,:,:] = data
    data = np.loadtxt(f3dm+'RSL_'+istr)
    rsl_3dm[i,:,:] = data
    data = np.loadtxt(f3dt+'RSL_'+istr)
    rsl_3dt[i,:,:] = data


proj = ccrs.Mollweide()


d3t_3d = rsl_3dt - rsl_3dm
d3d_1d = rsl_3dm - rsl_1dm
d3t_1t = rsl_3dt - rsl_1dt
d3t_1d = rsl_3dt - rsl_1dm

for i in range(nt):
    print('plot',str(i+1),'of',nt)
    fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(6,6),\
                           subplot_kw={'projection': proj})

    # 3T-3D
    cf = axs[0,0].contourf(lon,lat,d3t_3d[i,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu')
    cbar = plt.colorbar(cf)#cf,ticks=[-3,-2,-1,0,1])
    #cbar.ax.set_ylabel(r'log$_{10}[\chi^2]$')
    # Add details to the plot:
    axs[0,0].coastlines();
    axs[0,0].gridlines();

    # 3D-1D
    cf = axs[0,1].contourf(lon,lat,d3d_1d[i,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu')
    cbar = plt.colorbar(cf)#cf,ticks=[-3,-2,-1,0,1])
    #cbar.ax.set_ylabel(r'log$_{10}[\chi^2]$')
    # Add details to the plot:
    axs[0,1].coastlines();
    axs[0,1].gridlines();

    # 3T-1T
    cf = axs[1,0].contourf(lon,lat,d3t_1t[i,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu')
    cbar = plt.colorbar(cf)#cf,ticks=[-3,-2,-1,0,1])
    #cbar.ax.set_ylabel(r'log$_{10}[\chi^2]$')
    # Add details to the plot:
    axs[1,0].coastlines();
    axs[1,0].gridlines();

    # 3T-1D
    cf = axs[1,1].contourf(lon,lat,d3t_1d[i,:,:],\
                         transform=ccrs.PlateCarree(),cmap='RdBu')
    cbar = plt.colorbar(cf)#cf,ticks=[-3,-2,-1,0,1])
    #cbar.ax.set_ylabel(r'log$_{10}[\chi^2]$')
    # Add details to the plot:
    axs[1,1].coastlines();
    axs[1,1].gridlines();

    
    plt.tight_layout()

    plt.savefig(odir+'global_'+str(i)+'.pdf')

    plt.close()
