#! /usr/bin/env python3

# plot viscosity model at 1 depth

import numpy as np
import matplotlib.pyplot as plt
import cartopy as cp
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

irad = 115
# this is 200 km depth

fglobal = 'global_200km.pdf'
freg    = 'antarctica_200km.pdf'

fdir = '../../POSTPROCESS/model/'
rad  = np.loadtxt(fdir+'radius.dat')
clat = np.loadtxt(fdir+'clats.dat')
lons = np.loadtxt(fdir+'elons.dat')
lats = 90.-clat

print('PLOTTING RADIUS:',rad[irad]*1.e-3,'KM')
print('PLOTTING DEPTH:',6371.-rad[irad]*1.e-3,'KM')
dvis = np.loadtxt(fdir+'global_dvis_r'+str(irad)+'.dat')

# ====================================== #
# Plot data                              #
# ====================================== #

# W. Antarctica, Richmond Gulf
slat  = [-76.5,56.2494]
slon  = [-105.,-76.2942]

cmin = -1.5
cmax = 1.5

pmap = dvis
[ir,ic] = np.where(pmap>=cmax)
pmap[ir,ic] = cmax
[ir,ic] = np.where(pmap<cmin)
pmap[ir,ic] = cmin
    
fig = plt.figure(1,figsize=(10, 5))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
c = ax.contourf(lons, lats, pmap,
                transform=ccrs.PlateCarree(),
                cmap='RdBu')
ax.plot([slon[1],slon[1]],[slat[1],slat[1]],'^',ms=14,mfc='g',mec='g',\
                      transform=ccrs.PlateCarree())
ax.gridlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(),\
  linewidth=4, color='green', linestyle='-')
gl.yformatter = LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60])
gl.xlocator = mticker.FixedLocator([])
ax.coastlines()
ax.set_global()
plt.tight_layout()

fig.colorbar(c)

plt.savefig(fglobal)

plt.figure(2)
# The new projection:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
# Limit the map to -60 degrees latitude and below:
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
cf = ax.contourf(lons,lats,pmap,transform=ccrs.PlateCarree(),cmap='RdBu')
ax.plot([slon[0],slon[0]],[slat[0],slat[0]],'^',ms=15,mfc='g',mec='g',\
                      transform=ccrs.PlateCarree())

cbar = plt.colorbar(cf)#cf,ticks=[-3,-2,-1,0,1])
#cbar.ax.set_ylabel(r'log$_{10}[\chi^2]$')
# Add details to the plot:
ax.coastlines();
ax.gridlines();
gl = ax.gridlines(crs=ccrs.PlateCarree(),linewidth=4, color='green', linestyle='-')
gl.yformatter = LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60])
gl.xlocator = mticker.FixedLocator([])


plt.tight_layout()

plt.savefig(freg)

plt.figure(3)
plt.contourf(pmap,cmap='RdBu')
plt.colorbar(orientation='horizontal',ticks=[-1.5,0,1.5])
plt.savefig('colorbar.pdf')

plt.show()
    



